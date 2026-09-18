"""LoRA adapters, applied by **dotted path** so the scope is auditable — defect A3.

SST's OC-CCL trainer (`src/sst/oc_ccl.py:248`) uses no LoRA at all: it freezes the image
encoder and trains every remaining parameter. LoRA exists only in a separate ablation
script (`experiments/oc_ccl_ablation.py:41`), off by default (`--lora_rank 0`), and two
things there do not do what the docstring says:

1. **It has no scope filter.** The docstring says "memory_attention and mask_decoder" but
   the code walks all of `named_modules()` and replaces any Linear named
   `q_proj/k_proj/v_proj/out_proj` anywhere in the model. It happens not to touch the
   frozen image encoder only because Hiera uses a fused `self.qkv`
   (`sam2/modeling/backbones/hieradet.py:51`) — an accident, not a constraint.
2. **It never reaches `memory_encoder`.** That module's only projection is `out_proj`, an
   `nn.Conv2d` (`sam2/modeling/memory_encoder.py:151-156`), so the `isinstance(orig,
   nn.Linear)` test excludes it. `registry.FAITHFUL_FINETUNE` lists `memory_encoder` as a
   scope, so a name-restricted Linear search would satisfy the spec vacuously.

So this module scopes by dotted-path prefix and, within scope, adapts **every** `nn.Linear`
rather than a fixed name list. In `memory_encoder` that reaches the `CXBlock` pointwise
convolutions (`pwconv1`/`pwconv2`, Linear despite the name), which are the fuser's actual
learnable transforms; in `sam_mask_decoder` it reaches the transformer projections and
MLPs. `apply_lora` **raises if a requested scope matched nothing**, because a LoRA that
silently adapts zero modules trains nothing and still reports a plausible loss curve.
"""

from __future__ import annotations

import torch
from torch import nn

from ..seeding import derive_seed

#: Seeds the adapter initialisation so a fine-tune is REPRODUCIBLE.
#:
#: `lora_A` is drawn from the global torch RNG, which PyTorch seeds from OS entropy — so
#: two runs of the same recipe produced different adapters. Measured: checksums 64.303 and
#: 63.090 from identical configs. That is not merely untidy. `evaluator.run_fingerprint`
#: keys the resume cache on `||dW||`, so a regenerated adapter orphans every evaluation run
#: ever computed against its predecessor. One interrupted butterfly stage lost ~2.5 hours of
#: banked `occl` work to exactly this.
#:
#: It also put unquantified noise directly on the measurement: S8d's deltas are ~0.1-0.4
#: points, with confidence intervals over support seeds at a FIXED adapter. Adapter-draw
#: variance was never in those intervals.
LORA_SEED_TAG = "lora_init"

#: Spec default, `registry.FAITHFUL_FINETUNE["finetune.lora.scopes"]`.
FAITHFUL_SCOPES = ("memory_encoder", "sam_mask_decoder")

#: S14: the same "all linear projections and MLPs in the memory encoder and mask
#: decoder" reading, per model family. sam3 renamed the memory encoder to
#: `maskmem_backbone` (its CXBlock pwconv1/2 are nn.Linear, like SAM 2's); SeC's
#: grounding encoder IS SAM 2.1, so the sam2 strings apply verbatim when `apply_lora`
#: is rooted there. Module names are load-bearing: checkpoints key by path and the
#: RNG stream derives from it, so scope strings are never aliased across families.
FAITHFUL_SCOPES_BY_FAMILY = {
    "sam2": FAITHFUL_SCOPES,
    "sam3": ("maskmem_backbone", "sam_mask_decoder"),
    "sec": FAITHFUL_SCOPES,
}

#: S14sam3's architecture-adapted arm: the ported scopes PLUS `transformer` —
#: sam3's renamed-and-retrained memory attention, the module that turns support
#: memory into query proposals. No arm in this project, on any family, has ever
#: trained it (the paper's A2.2 reading stops at memory encoder + mask decoder),
#: and S13 located the remaining error in UNPROPOSED regions — which the decoder
#: cannot reach but the proposal-forming module might. A strict SUPERSET of the
#: ported scopes, so any delta vs the ported arm is attributable to the
#: transformer addition alone. `transformer` is a top-level sibling of the frozen
#: `backbone` (sam3_tracker_base.py), so the embedding cache stays valid.
SAM3_ARCH_SCOPES = ("maskmem_backbone", "transformer", "sam_mask_decoder")

#: S17's `arch` arm on SAM 2 — the same addition on the original family: the S11
#: scopes PLUS `memory_attention` (SAM2Base's proposal-forming module; its RoPE
#: attention q/k/v/out projections and MLP linears are nn.Linear). Motivated by the
#: S17 anatomy: the closing frame's SAM head replaces every channel it declares absent
#: by the −1024 sentinel through a hard `torch.where`, so the channels that need
#: teaching carry NO gradient and the live signal reaches only what is in scope —
#: on SAM 2 that never included the module that decides what gets proposed. A strict
#: superset of the faithful scopes; the image encoder stays frozen (cache valid).
SAM2_ARCH_SCOPES = ("memory_encoder", "memory_attention", "sam_mask_decoder")


class LoRALinear(nn.Module):
    """``y = W x + (alpha / r) * B A x``, with ``B`` zero-initialised.

    Zero-init on ``B`` — not on ``A`` — is what makes ``ΔW = 0`` at step 0, so the adapted
    model starts *exactly* equal to the base model. That is worth stating because it is the
    only reason an S8 fine-tune can be compared against the S4c baseline at all: if the
    adapters perturbed the model at init, a delta would be measuring the perturbation.
    """

    def __init__(self, orig: nn.Linear, r: int, alpha: float, seed: int | None = None):
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank must be positive, got {r}")
        self.orig = orig
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.r
        w = orig.weight
        # fp32 master params. The forward runs under bf16 autocast; keeping the adapters
        # fp32 keeps the optimiser update well-conditioned at lr ~1e-5.
        self.lora_A = nn.Parameter(torch.zeros(self.r, orig.in_features,
                                               device=w.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(orig.out_features, self.r,
                                               device=w.device, dtype=torch.float32))
        _init_a(self.lora_A, seed)
        for p in self.orig.parameters():
            p.requires_grad = False

    #: S17b `fwdonly`: a LoRALinear can be switched off for a stretch of the forward pass
    #: (`lora_disabled`), returning the base Linear's output bit for bit. A plain Python
    #: attribute — never a parameter or buffer — so checkpoints and `lora_state_dict` are
    #: untouched, and `apply_lora`/`load_lora_state_dict` never see it.
    enabled: bool = True

    def forward(self, x):
        if not self.enabled:
            return self.orig(x)
        delta = (x @ self.lora_A.T.to(x.dtype)) @ self.lora_B.T.to(x.dtype)
        return self.orig(x) + delta * self.scale

    def delta_weight(self) -> torch.Tensor:
        """``ΔW`` as it would be if merged, for `lora_delta_norm`."""
        return (self.lora_B @ self.lora_A) * self.scale


def _init_a(tensor: torch.Tensor, seed: int | None) -> None:
    """Kaiming-uniform init, from a dedicated generator when a seed is given.

    A private `torch.Generator` rather than `torch.manual_seed`, so seeding the adapters
    does not silently reseed everything else in the process.
    """
    if seed is None:
        nn.init.kaiming_uniform_(tensor, a=5 ** 0.5)
        return
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    bound = (6.0 / ((1 + 5.0) * tensor.shape[1])) ** 0.5      # kaiming_uniform, a=sqrt(5)
    with torch.no_grad():
        tensor.copy_(torch.empty(tensor.shape, dtype=torch.float32).uniform_(
            -bound, bound, generator=g).to(tensor.device))


def _in_scope(name: str, scopes) -> bool:
    """Dotted-path prefix match: ``memory_encoder`` matches ``memory_encoder.fuser...``.

    Prefix rather than substring, so a scope cannot silently pick up an unrelated module
    whose name happens to contain it.
    """
    return any(name == s or name.startswith(s + ".") for s in scopes)


def apply_lora(model: nn.Module, r: int = 32, alpha: float = 64.0,
               scopes=FAITHFUL_SCOPES, seed: int | str | None = "auto") -> list[str]:
    """Replace every `nn.Linear` under ``scopes`` with a `LoRALinear`. Returns their paths.

    Raises if any scope matched nothing — see the module docstring for why that failure
    must be loud.
    """
    scopes = tuple(scopes)
    # Per-module seeds derived from the module PATH, so the adapter set is reproducible and
    # independent of iteration order.
    base_seed = (derive_seed(LORA_SEED_TAG, "apply", r, int(alpha))
                 if seed == "auto" else seed)
    # Collect first: mutating during named_modules() iteration is undefined.
    targets: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for mod_name, module in model.named_modules():
        for attr, child in list(module.named_children()):
            path = f"{mod_name}.{attr}" if mod_name else attr
            if isinstance(child, nn.Linear) and _in_scope(path, scopes):
                targets.append((module, attr, path, child))

    for module, attr, _path, child in targets:
        sub = None if base_seed is None else derive_seed(LORA_SEED_TAG, _path, r, base_seed)
        setattr(module, attr, LoRALinear(child, r=r, alpha=alpha, seed=sub))

    paths = sorted(p for _, _, p, _ in targets)
    empty = [s for s in scopes if not any(_in_scope(p, (s,)) for p in paths)]
    if empty:
        raise RuntimeError(
            f"LoRA scope(s) {empty} matched no nn.Linear. A scope that adapts nothing "
            f"trains nothing while still producing a plausible loss curve. "
            f"Matched {len(paths)} module(s) overall under {list(scopes)}."
        )
    return paths


def reset_lora(model: nn.Module) -> int:
    """Return every adapter to its initial state. Returns how many were reset.

    For grid search: a fold trains several learning rates, and rebuilding the model between
    them would reload 898 MB of checkpoint each time. Resetting in place is equivalent and
    ~instant — but only if it is *exactly* equivalent, so this re-draws `lora_A` from the
    same Kaiming distribution and zeroes `lora_B`, which is what `__init__` does.

    ``||dW|| == 0`` afterwards, exactly, which is the property callers should assert: a
    reset that left residue would make the next arm's training start from the previous
    arm's solution and every point on the grid after the first would be contaminated.

    **`requires_grad` is part of that initial state**, and restoring it is not optional.
    `train_lora` calls `freeze_all` before this, which freezes *every* parameter — on the
    first grid point the adapters do not exist yet and `apply_lora` creates them trainable,
    but on reuse they are already present and get frozen with everything else. Leaving them
    frozen produced a run where all ten epochs of the first learning rate trained fine and
    the second died in `loss.backward()` with "element 0 of tensors does not require grad",
    which names neither LoRA nor freezing.
    """
    mods = lora_modules(model)
    base_seed = derive_seed(LORA_SEED_TAG, "apply", mods[0][1].r if mods else 0,
                            int(mods[0][1].alpha) if mods else 0)
    with torch.no_grad():
        for name, m in mods:
            _init_a(m.lora_A, derive_seed(LORA_SEED_TAG, name, m.r, base_seed))
            m.lora_B.zero_()
    for _, m in mods:
        m.lora_A.requires_grad_(True)
        m.lora_B.requires_grad_(True)
    return len(mods)


def lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def set_lora_enabled(model: nn.Module, enabled: bool) -> int:
    """Switch every adapter on or off for the forward pass. Returns the count."""
    mods = lora_modules(model)
    for _, m in mods:
        m.enabled = bool(enabled)
    return len(mods)


class lora_disabled:
    """``with lora_disabled(model):`` — the base model, bit for bit, inside the block.

    S17b's `fwdonly` arm runs the cycle's reverse init and closing frame under this: the
    closing head then cannot learn to read ``x0`` (S17 measured that absorption — the
    closing gain was largest where the prompt was worst), so the only way left to lower
    the closing loss is a better forward mask. Gradient still flows THROUGH the base
    modules to the adapters upstream (the forward leg's memory); only their own ΔW is
    absent. Re-entrant: restores whatever state the adapters had on entry.
    """

    def __init__(self, model: nn.Module | None):
        self.model = model
        self._prev: list[tuple[LoRALinear, bool]] = []

    def __enter__(self):
        if self.model is not None:
            self._prev = [(m, m.enabled) for _, m in lora_modules(self.model)]
            for m, _ in self._prev:
                m.enabled = False
        return self

    def __exit__(self, *exc):
        for m, was in self._prev:
            m.enabled = was
        self._prev = []
        return False


def remove_lora(model: nn.Module) -> int:
    """Unwrap every `LoRALinear` back to its original `nn.Linear`. Returns the count.

    The exact inverse of `apply_lora` by construction: the wrapped Linear is held
    UNTOUCHED at ``.orig`` (the adapter adds ``B A x`` alongside it, never into it),
    so restoring the attribute restores the base module bit for bit. Exists for
    consecutive checkpoints with DIFFERENT scope sets — S14sam3's TTA fan runs the
    stack and arch arms back to back in one worker, and neither reusing the first
    arm's wrappers (the second's transformer keys go unmatched) nor re-applying on
    top of them (every path shifts by ``.orig``) is loadable.
    """
    wrapped = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, mod in wrapped:
        parent = (model.get_submodule(name.rsplit(".", 1)[0])
                  if "." in name else model)
        orig = mod.orig
        for p in orig.parameters():
            p.requires_grad = True      # apply_lora froze them; restore the default
        setattr(parent, name.rsplit(".", 1)[-1], orig)
    return len(wrapped)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for _, m in lora_modules(model) for p in (m.lora_A, m.lora_B)]


def lora_delta_norm(model: nn.Module) -> float:
    """Total ``||ΔW||_F`` over all adapters — the S8 gate's proof that training moved.

    Exactly 0.0 at init by construction, so a non-zero value after an optimiser step is
    evidence the gradient reached the adapters. It cannot distinguish "learned something
    useful" from "moved"; that is the paired delta's job.
    """
    with torch.no_grad():
        return float(sum(m.delta_weight().norm().item() for _, m in lora_modules(model)))


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors — a few MB against the base model's 2.3 GB."""
    out = {}
    for name, m in lora_modules(model):
        out[f"{name}.lora_A"] = m.lora_A.detach().cpu().clone()
        out[f"{name}.lora_B"] = m.lora_B.detach().cpu().clone()
    return out


def load_lora_state_dict(model: nn.Module, sd: dict) -> int:
    """Load adapters, returning the number of key mismatches. **Zero means strict.**

    Returned rather than raised so the S8 record can carry it as a diagnostic
    (`lora_roundtrip_mismatches`). A silent partial load is the failure this guards: the
    evaluation would then score a model that is part fine-tuned and part base, and every
    number after it would be uninterpretable.
    """
    want = {f"{n}.{k}" for n, _ in lora_modules(model) for k in ("lora_A", "lora_B")}
    got = set(sd)
    mismatches = len(want ^ got)
    by_name = dict(lora_modules(model))
    with torch.no_grad():
        for key, tensor in sd.items():
            name, _, which = key.rpartition(".")
            m = by_name.get(name)
            if m is None:
                continue
            target = getattr(m, which, None)
            if target is None or tuple(target.shape) != tuple(tensor.shape):
                mismatches += 1
                continue
            target.copy_(tensor.to(target.device, target.dtype))
    return mismatches
