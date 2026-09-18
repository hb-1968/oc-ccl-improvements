"""One-epoch OC-CCL fine-tune — defects A4 and A5.

**A4 — epochs and learning rate.** SST defaults to 10 epochs at a single untuned
`lr=1e-5` (`oc_ccl.py:353,355`), while `registry.FAITHFUL_FINETUNE["finetune.epochs"]` is
1. Ten epochs over a 120-image train split with a cycle loss that needs no labels is ample
room to overfit the reconstruction while learning nothing transferable, and one LR is a
guess rather than a result. This module runs one epoch; `scripts/run_finetune.py` sweeps
the LR so the choice is measured.

**A5 — train vs eval mode.** `oc_ccl.py:260,293` calls `model.train()`, which activates
dropout in modules the evaluator always runs under `eval()`. The cycle is then supervised
through noise that never appears at scoring time. Here the model stays in `eval()`
throughout — LoRA adapters have no dropout or normalisation state, so nothing needs train
mode for the gradient to flow.

Freezing is explicit and total: every base parameter gets `requires_grad = False` before
LoRA is applied, so the optimiser can only see adapter tensors. SST instead freezes the
image encoder alone (`oc_ccl.py:237-246`) and trains everything else — which is not LoRA at
all, and is defect A3.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch

from .cycle import (MARGIN_FORM, MARGIN_STATISTIC, gradient_report,
                    margin_gradient_report, margin_step, oc_ccl_step)
from .lora import (apply_lora, lora_delta_norm, lora_modules, lora_parameters,
                   lora_state_dict, reset_lora)
from .pairs import CyclePairs
from .tracker import DifferentiableSAM2Tracker


def checkpoint_hash(path) -> str:
    """Short content hash of a checkpoint file.

    Exists for the resume hazard: `evaluator.run_fingerprint` covers module *source*, not
    weights, so scoring a fine-tuned model under a stage name an earlier run used would
    serve that run's cached per-run JSON and hand back the baseline's numbers wearing the
    treatment's label. The hash belongs in the resume slug before S8c, not after.
    """
    h = hashlib.blake2b(digest_size=8)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def detach_inference_caches(model) -> int:
    """Replace module-cached tensors created under `inference_mode` with normal clones.

    SAM 2's `RoPEAttention` memoises its rotary encoding on the module
    (`sam2/modeling/sam/transformer.py:308`) and only recomputes it when the feature shape
    changes (`:327`). So scoring the model **before** training it leaves `freqs_cis` as an
    *inference tensor*, and the next backward pass dies with

        RuntimeError: Inference tensors cannot be saved for backward.

    which points at `apply_rotary_enc` and looks like a bug in the cycle rather than an
    artefact of what ran ten minutes earlier. Training in a fresh process never hits it,
    which is why the S8b sweep was fine and the grid search was not.

    Cloning outside `inference_mode` yields a normal tensor — the fix the error message
    itself suggests — so the ordering of train and score stops being load-bearing.
    Returns how many caches were converted.
    """
    n = 0
    for m in model.modules():
        t = getattr(m, "freqs_cis", None)
        if torch.is_tensor(t) and torch.is_inference(t):
            m.freqs_cis = t.clone()
            n += 1
    return n


def freeze_except_image_encoder(model) -> dict:
    """SST's actual method: `freeze_image_encoder(model)`, everything else trains.

    `src/sst/oc_ccl.py:237-246`. Not an adapter of any kind — ~11.7M of 224.4M parameters
    are optimised directly. This is what produced the paper's numbers; the LoRA path our
    spec required until now appears in SST only as an ablation axis, defaulting to off.
    See the corrected `registry.FAITHFUL_FINETUNE` for the full provenance.

    Returns the counts SST itself prints, so the two can be compared line for line.
    """
    for p in model.parameters():
        p.requires_grad = True
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total,
            "pct": round(100.0 * trainable / total, 1)}


def trainable_state_dict(model) -> dict:
    """Only the tensors that were optimised — the full-fine-tune analogue of `lora_state_dict`.

    ~47 MB in fp32 against LoRA's few, which is why the adapters were convenient. Convenience
    is not faithfulness, and the size is affordable: it is a fifth of the base checkpoint we
    already move around.
    """
    import torch  # noqa: F401  (kept local; this module must import without torch)

    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state_dict(model, sd: dict) -> int:
    """Load a full-fine-tune state dict. Returns key mismatches; **zero means strict.**

    Same contract as `load_lora_state_dict`: returned rather than raised so the record can
    carry it, because a silent partial load would score a model that is part fine-tuned and
    part base.
    """
    import torch

    by_name = dict(model.named_parameters())
    mismatches = len(set(sd) - set(by_name))
    with torch.no_grad():
        for k, v in sd.items():
            p = by_name.get(k)
            if p is None or tuple(p.shape) != tuple(v.shape):
                mismatches += 1
                continue
            p.copy_(v.to(p.device, p.dtype))
    return mismatches


def full_delta_norm(model, theta0: dict) -> float:
    """``||theta - theta0||_F`` over the trained tensors — the full-fine-tune analogue of
    `lora_delta_norm`, and the same gate evidence: exactly 0.0 means nothing moved."""
    import torch

    with torch.no_grad():
        return float(torch.sqrt(sum(
            ((p.detach() - theta0[n].to(p.device)) ** 2).sum()
            for n, p in model.named_parameters() if n in theta0)))


def load_finetune(model, path, *, lora_r: int = 32, lora_alpha: float = 64.0) -> dict:
    """Load a fine-tune checkpoint of **either** method. One place, deliberately.

    `run_eval.py` and `run_arms.py` both need this, and "two callers that must agree about a
    format" is the exact shape that has already produced the triplicated support-draw
    dispatch and three launchers with different environments. The method is read from the
    blob's own metadata rather than guessed from a filename.

    Returns a report for the record. ``mismatches`` is the strictness check and must be 0:
    a full-fine-tune checkpoint pushed through `load_lora_state_dict` would mismatch on
    every key, and a loader that shrugged at that would score a part-trained model.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob.get("meta") or {}
    method = meta.get("method") or ("full" if "full" in blob else "lora")

    if method == "full":
        sd = blob["full"]
        # Snapshot BEFORE loading so ||dW|| is measured against what this process actually
        # started from, rather than against an assumption about the checkpoint on disk.
        theta0 = {n: p.detach().cpu().clone()
                  for n, p in model.named_parameters() if n in sd}
        mismatches = load_trainable_state_dict(model, sd)
        delta = full_delta_norm(model, theta0)
        return {"method": "full", "mismatches": mismatches, "delta_norm": delta,
                "n_tensors": len(sd), "base_state": theta0, "meta": meta}

    from .lora import (FAITHFUL_SCOPES, apply_lora, load_lora_state_dict,
                       remove_lora)

    m = meta.get("lora") or {}
    r, alpha = m.get("r", lora_r), m.get("alpha", lora_alpha)
    # Scopes from the checkpoint's own metadata (S14): a sam3 checkpoint reloaded
    # under the sam2 scope strings would raise on the empty `memory_encoder` match —
    # loud, but wrong. Pre-S14 checkpoints lack the key and get the sam2 default.
    scopes = tuple(m.get("scopes") or FAITHFUL_SCOPES)
    # ALWAYS rebuild the adapter set from THIS checkpoint's own (scopes, r, alpha).
    # The previous reuse-if-wrapped branch was only correct when consecutive
    # checkpoints shared a scope set; S14sam3's TTA fan runs stack -> arch (a
    # strict scope superset) in one worker, where reuse leaves the transformer
    # keys unmatched and the strict guard below kills the worker mid-fan.
    # apply_lora is module surgery, ~ms — nothing worth reusing.
    remove_lora(model)
    paths = apply_lora(model, r=r, alpha=alpha, scopes=scopes)
    mismatches = load_lora_state_dict(model, blob["lora"])
    return {"method": "lora", "mismatches": mismatches,
            "delta_norm": lora_delta_norm(model), "modules": len(paths),
            "r": r, "alpha": alpha, "base_state": None, "meta": meta}


def restore_base(model, base_state: dict) -> float:
    """Put a full-fine-tuned model back to the weights it was built with.

    The full-fine-tune analogue of `reset_lora`, and needed for the same reason: `run_arms`
    scores group-outer/arm-inner, so the base arm runs again after every fine-tuned arm and
    must actually be the base model, not merely labelled one. `reset_lora` cannot do this —
    there are no adapters to zero, the base weights themselves were overwritten.

    Returns ``||theta - theta0||`` afterwards, which callers must assert is exactly 0.0.
    """
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in base_state:
                p.copy_(base_state[n].to(p.device, p.dtype))
    return full_delta_norm(model, base_state)


def _set_mode(model, train_mode: bool) -> None:
    """A5 as an AXIS. `model.eval()` is our fix; `model.train()` is what SST calls.

    Inert under LoRA — adapters have no dropout or normalisation state — which is why this
    was safe to hardcode while S8 used adapters. Under a FULL fine-tune it is not: train
    mode activates dropout across memory attention and the mask decoder, supervising the
    cycle through noise the evaluator never sees, AND lets normalisation layers update
    running statistics that then ship inside the checkpoint. At 10 epochs that could act as
    a regulariser, so it is worth measuring rather than assuming.
    """
    model.train() if train_mode else model.eval()


def freeze_all(model) -> int:
    for p in model.parameters():
        p.requires_grad = False
    return sum(p.numel() for p in model.parameters())


def _make_tracker(model, family: str, soft_sentinel: float | None = None):
    """Family -> differentiable tracker. `sec`'s grounding encoder IS SAM 2.1, so it
    rides the sam2 tracker; sam3 gets its own (image kwarg, 1152 masks).

    ``soft_sentinel`` (S17d) is the SAM 2 tracker's soft no-object gate; the sam3
    tracker has no −1024 sentinel of this shape and refuses it."""
    if family == "sam3":
        from .tracker_sam3 import DifferentiableSAM3Tracker

        if soft_sentinel is not None:
            raise ValueError("soft_sentinel is a SAM 2 knob (the −1024 gate in "
                             "sam2_base._forward_sam_heads); sam3 has no such gate here")
        return DifferentiableSAM3Tracker(model)
    if family in ("sam2", "sec"):
        return DifferentiableSAM2Tracker(model, soft_sentinel=soft_sentinel)
    raise ValueError(f"unknown model family {family!r}")


def train_lora(model, ds, *, lr: float, epochs: int = 1, group: str = "all",
               method: str = "lora", lora_r: int = 32, lora_alpha: float = 64.0,
               train_mode: bool = False, normalization: str = "imagenet",
               prompt_mode: str = "logit_reparam", memory_reset: bool = True,
               mask_mode: str = "merged", reference_id: str | None = None,
               rotation_aug_deg: float | None = None,
               scnp_width: int = 0, scnp_joint: bool = True,
               excl_mode: str = "off", xovl_weight: float = 0.0,
               xovl_where: str = "query", bshape_weight: float = 0.0,
               margin_weight: float = 0.0, margin_m: float = 1.0,
               margin_only: bool = False,
               found_mask: bool = False, bce_w: float = 1.0,
               lora_legs: str = "all", obj_weight: float = 0.0,
               soft_sentinel: float | None = None,
               close_aug_deg: float | None = None,
               family: str = "sam2", image_size: int = 1024,
               mask_size: int | None = None, lora_scopes: tuple | None = None,
               device: str = "cuda", dtype=None, log_every: int = 10,
               grad_clip: float = 1.0, max_steps: int | None = None,
               checkpoint_dir=None) -> dict:
    """Fine-tune the adapters for ``epochs`` epochs. Returns a report for the S8 record.

    ``close_aug_deg`` (S18 `perturb`): every step closes the cycle on the reference under
    its own seeded rotation (`CyclePairs.closing`); the applied angles are recorded per
    step under ``close_aug`` beside ``aug``.

    This ran exactly one epoch regardless of the argument until it was caught: the CLI
    accepted `--epochs 10`, recorded `epochs: 10`, and trained once. A record that
    misstates its own hyperparameters is worse than one that omits them, because every
    comparison drawn from it is silently wrong.

    Each epoch re-derives its pair order from `CyclePairs(..., epoch=n)`, so the model does
    not see the same (reference, query) pairing ten times — only the same images in
    different roles. The LR schedule spans the whole run rather than restarting per epoch.

    ``checkpoint_dir`` writes the adapters after every epoch, which is what makes epoch
    tuning affordable: one training run yields the whole epochs curve, and because LoRA
    never touches the image encoder every checkpoint can be scored against one shared set
    of embeddings.
    """
    # Before anything else: any tensor a previous inference pass memoised on a module is
    # unusable in autograd, and the failure surfaces deep inside attention. See
    # detach_inference_caches.
    if (n_detached := detach_inference_caches(model)):
        print(f"  detached {n_detached} inference-mode cache(s) left by an earlier "
              "scoring pass")

    theta0 = None
    if method == "full":
        # SST's own method. No adapters: freeze the image encoder, train the rest.
        counts = freeze_except_image_encoder(model)
        print(f"  full fine-tune: {counts['trainable']:,} / {counts['total']:,} "
              f"parameters ({counts['pct']}%)", flush=True)
        _set_mode(model, train_mode)                               # A5
        paths = []
        n_total = counts["total"]        # read by the shared summary line below; assigning
                                         # it only in the LoRA branch made the very first
                                         # full fine-tune die with UnboundLocalError after
                                         # loading 898 MB of checkpoint.
        params = [p for p in model.parameters() if p.requires_grad]
        # Snapshot for ||dW||. ~47 MB, kept on CPU so it does not compete with activations.
        theta0 = {n: p.detach().cpu().clone()
                  for n, p in model.named_parameters() if p.requires_grad}
    else:
        n_total = freeze_all(model)                                # A5/A3: nothing trains...
        existing = lora_modules(model)
        if existing:
            # Grid search reuses one built model across learning rates. Re-applying would
            # wrap the adapters in adapters; resetting restores the exact initial state.
            reset_lora(model)
            paths = sorted(n for n, _ in existing)
        else:
            # Per-family scopes (S14): sam3 renamed the memory encoder; None keeps
            # the sam2 spec strings. apply_lora raises on an empty scope match —
            # exactly the guard that protects a mistyped family here.
            from .lora import FAITHFUL_SCOPES_BY_FAMILY
            _scopes = (lora_scopes if lora_scopes is not None
                       else FAITHFUL_SCOPES_BY_FAMILY[family])
            paths = apply_lora(model, r=lora_r, alpha=lora_alpha,
                               scopes=_scopes)             # ...except the adapters
        _set_mode(model, train_mode)                               # A5
        params = lora_parameters(model)
    # Fail here, naming the cause, rather than 600 seconds later inside `loss.backward()`
    # with "element 0 of tensors does not require grad" — which says nothing about LoRA or
    # about freezing, and cost a full 10-epoch training run to reach.
    trainable = [p for p in params if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            f"{len(params)} {'parameters' if method == 'full' else 'LoRA tensors'} exist "
            "but none require grad — nothing would train. For LoRA: `freeze_all` freezes "
            "every parameter, so on a REUSED model the adapters must be re-enabled "
            "(reset_lora does this)."
        )
    if method != "full" and len(trainable) != len(params):
        raise RuntimeError(
            f"only {len(trainable)} of {len(params)} LoRA tensors require grad — a "
            "partially frozen adapter set trains a different function than it reports")
    n_train = sum(p.numel() for p in params)
    what = (f"FULL fine-tune, image_encoder frozen" if method == "full"
            else f"LoRA r={lora_r} alpha={lora_alpha} on {len(paths)} modules")
    print(f"{what} | trainable {n_train:,} / {n_total:,} "
          f"({100 * n_train / n_total:.3f}%) | {epochs} epoch(s)")

    tracker = _make_tracker(model, family, soft_sentinel=soft_sentinel)
    pair_kw = dict(normalization=normalization, mask_mode=mask_mode,
                   reference_id=reference_id, rotation_aug_deg=rotation_aug_deg,
                   size=image_size, mask_size=mask_size, close_aug_deg=close_aug_deg)
    _probe = CyclePairs(ds, group, epoch=0, **pair_kw)
    n_pairs = len(_probe)
    per_epoch = n_pairs if max_steps is None else min(max_steps, n_pairs)
    total_steps = per_epoch * epochs

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    # T_max over the WHOLE run, not one epoch: a per-epoch cosine would restart the LR at
    # its peak every epoch, which is a different (and unintended) schedule.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps,
                                                       eta_min=lr * 0.01)
    cuda = str(device).startswith("cuda")
    dtype = dtype or (torch.bfloat16 if cuda else torch.float32)

    def delta_norm():
        """||dW|| for whichever method is running. Both are exactly 0.0 before the first
        step — LoRA because B is zero-initialised, full because theta == theta0 — so the
        gate's "training moved the weights" evidence reads the same either way."""
        return full_delta_norm(model, theta0) if method == "full" else lora_delta_norm(model)

    delta_before = delta_norm()                                    # exactly 0.0 by design
    losses, per_epoch_loss, checkpoints = [], [], []
    step_log: list[dict] = []
    margin_log: list[dict] = []
    found_log: list[dict] = []
    obj_log: list[dict] = []
    margin_on = bool(margin_weight or margin_only)
    if bshape_weight:
        # Fresh counters per training run, so the report's hit/miss split describes THIS
        # run's pair regime and not whatever ran before in the same process.
        from .bshape import PHI_CACHE
        PHI_CACHE.reset()
    t0, peak_gb = time.time(), 0.0

    for ep in range(epochs):
        ep_losses = []
        pairs = CyclePairs(ds, group, epoch=ep, **pair_kw)
        for i, (x0, xu, y0, _ids) in enumerate(pairs):
            if i >= per_epoch:
                break
            x0, xu, y0 = x0.to(device), xu.to(device), y0.to(device)
            close_kw = {}
            if close_aug_deg is not None:
                # S18 `perturb`: the closing end under its own rotation; the applied
                # angle joins this step's log entry
                x0c, y0c, crec = pairs.closing(i)
                pairs.step_log[-1].update(crec)
                close_kw = {"x0_close": x0c.to(device), "y0_close": y0c.to(device)}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=dtype,
                                enabled=cuda):
                if margin_only:
                    # S13 `margin` arm — the margin objective ALONE, no reconstruction
                    # and no reverse legs (`cycle.margin_step`).
                    loss, _pred, parts = margin_step(tracker, x0, xu, y0,
                                                     margin_m=margin_m)
                else:
                    loss, _pred, parts = oc_ccl_step(tracker, x0, xu, y0,
                                                     prompt_mode=prompt_mode,
                                                     memory_reset=memory_reset,
                                                     scnp_width=scnp_width,
                                                     scnp_joint=scnp_joint,
                                                     excl_mode=excl_mode,
                                                     xovl_weight=xovl_weight,
                                                     xovl_where=xovl_where,
                                                     bshape_weight=bshape_weight,
                                                     margin_weight=margin_weight,
                                                     margin_m=margin_m,
                                                     found_mask=found_mask, bce_w=bce_w,
                                                     lora_legs=lora_legs,
                                                     obj_weight=obj_weight, **close_kw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            sched.step()
            ep_losses.append(float(loss.detach()))
            if found_mask:
                # what the mask actually did, step by step: how many channels were
                # dropped, and how many steps had nothing to train on
                found_log.append({"n_found": parts["n_found"], "n_objects": parts["n_objects"],
                                  "bce": parts["bce"], "bce_found": parts["bce_found"]})
            if "fwd_obj_score" in parts:
                # the forward frame's objectness per step (S17b): how many prompted
                # channels the head declared present, and the term when trained
                obj_log.append({"n_objects": parts["n_objects"],
                                "n_present": sum(1 for v in parts["fwd_obj_score"] if v > 0),
                                "obj": parts.get("obj")})
            if margin_on:
                margin_log.append({k: parts[k] for k in
                                   ("conf_up", "conf_rot", "gap", "margin",
                                    "margin_violated", "inconsistent_won",
                                    "empty_claim_up", "empty_claim_rot")})
            if cuda:
                peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / 1e9)
            if log_every and i % log_every == 0:
                # margin-only parts carry no bce/dice; the margin arms print the
                # statistic they train instead.
                terms = (f"bce {parts['bce']:.4f} dice {parts['dice']:.4f} "
                         if "bce" in parts else "")
                if margin_on:
                    terms += (f"conf {parts['conf_up']:.2f}/{parts['conf_rot']:.2f} "
                              f"gap {parts['gap']:.2f} ")
                print(f"  ep{ep + 1}/{epochs} [{i}/{per_epoch}] loss={ep_losses[-1]:.4f} "
                      f"({terms.rstrip()}) "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
        losses += ep_losses
        step_log += pairs.step_log[:per_epoch]
        per_epoch_loss.append(sum(ep_losses) / len(ep_losses) if ep_losses else None)
        if checkpoint_dir:
            path = Path(checkpoint_dir) / f"ep{ep + 1}.pt"
            save_lora(model, path, method=method,
                      meta={"lora": {"r": lora_r, "alpha": lora_alpha},
                            "epochs": ep + 1, "lr": lr})
            checkpoints.append({"epoch": ep + 1, "path": str(path),
                                "delta_norm": delta_norm(),
                                "loss_mean": per_epoch_loss[-1]})
        print(f"  epoch {ep + 1}/{epochs}: mean loss {per_epoch_loss[-1]:.4f}, "
              f"||dW||={delta_norm():.4f}", flush=True)

    # What actually ran, step by step — the record must carry the applied augmentation
    # and the per-step channel counts, not just the configuration that requested them.
    aug = None
    if rotation_aug_deg is not None:
        aug = {
            "type": "rotation_x0",
            "max_deg": float(rotation_aug_deg),
            "angles": [round(e["aug_angle"], 3) for e in step_log],
            "empty_channel_fallbacks": sum(bool(e.get("aug_fallback"))
                                           for e in step_log),
        }
    close_aug = None
    if close_aug_deg is not None:
        close_aug = {
            "type": "rotation_x0_close",
            "max_deg": float(close_aug_deg),
            "angles": [round(e["close_angle"], 3) for e in step_log],
            "empty_channel_fallbacks": sum(bool(e.get("close_fallback"))
                                           for e in step_log),
        }
    mask_channels: dict[str, int] = {}
    if mask_mode == "per_trait":
        for e in step_log:
            k = str(len(e["class_ids"] or []))
            mask_channels[k] = mask_channels.get(k, 0) + 1

    bshape_report = None
    if bshape_weight:
        from .bshape import PHI_CACHE
        bshape_report = {
            "weight": bshape_weight, "schedule": "const", "dead_zone": 1,
            "normalize": "chan_max",
            "phi_cache": {"hits": PHI_CACHE.hits, "misses": PHI_CACHE.misses}}
    margin_report = None
    if margin_on:
        margin_report = {
            "m": margin_m,
            "weight": (None if margin_only else margin_weight),
            "only": margin_only,
            "statistic": MARGIN_STATISTIC, "form": MARGIN_FORM,
            "consistent": "as_given",
            "violations": sum(e["margin_violated"] for e in margin_log),
            "inconsistent_won_steps": sum(e["inconsistent_won"] for e in margin_log),
            # Step 0 runs with LoRA still an exact no-op, so the earliest steps are a
            # near-free base-calibration reading (later steps have moved weights).
            "inconsistent_won_first20": sum(e["inconsistent_won"]
                                            for e in margin_log[:20]),
            "empty_claim_up": sum(e["empty_claim_up"] for e in margin_log),
            "empty_claim_rot": sum(e["empty_claim_rot"] for e in margin_log),
            "conf_up": [round(e["conf_up"], 4) for e in margin_log],
            "conf_rot": [round(e["conf_rot"], 4) for e in margin_log],
        }

    found_report = None
    if found_mask:
        found_report = {
            "rule": "any_logit_gt_0",
            "steps": len(found_log),
            "channels_total": sum(e["n_objects"] for e in found_log),
            "channels_masked": sum(e["n_objects"] - e["n_found"] for e in found_log),
            "steps_no_found": sum(1 for e in found_log if e["n_found"] == 0),
            # the PLAIN BCE the same closing masks would have been charged, beside the
            # masked term that trained — the share the mask removed, per epoch
            "bce_plain_mean": (sum(e["bce"] for e in found_log) / len(found_log)
                               if found_log else None),
            "bce_found_mean": (sum(e["bce_found"] for e in found_log) / len(found_log)
                               if found_log else None),
        }

    obj_report = None
    if obj_log:
        obj_report = {
            "weight": (float(obj_weight) if obj_weight else None),
            "steps": len(obj_log),
            "channels_total": sum(e["n_objects"] for e in obj_log),
            "channels_declared_present": sum(e["n_present"] for e in obj_log),
            # the first 20 steps run near-base weights: the base head's present-rate
            "present_rate_first20": (sum(e["n_present"] for e in obj_log[:20])
                                     / max(sum(e["n_objects"] for e in obj_log[:20]), 1)),
            "present_rate_last20": (sum(e["n_present"] for e in obj_log[-20:])
                                    / max(sum(e["n_objects"] for e in obj_log[-20:]), 1)),
            "obj_mean": (sum(e["obj"] for e in obj_log if e["obj"] is not None)
                         / max(sum(1 for e in obj_log if e["obj"] is not None), 1)
                         if obj_weight else None),
        }

    return {
        "lr": lr, "epochs": epochs, "steps": len(losses), "group": group,
        "steps_per_epoch": per_epoch,
        "found_mask": found_report,
        "bce_w": (None if bce_w == 1.0 else float(bce_w)),
        "objness": obj_report if obj_weight else None,
        "objectness": obj_report,
        "lora_legs": (None if lora_legs == "all" else lora_legs),
        "soft_sentinel": (None if soft_sentinel is None else float(soft_sentinel)),
        "family": family, "image_size": image_size,
        "mask_size": (image_size if mask_size is None else mask_size),
        "mask_mode": mask_mode,
        "scnp": ({"width": scnp_width, "joint": scnp_joint} if scnp_width else None),
        "excl": ({"mode": excl_mode} if excl_mode != "off" else None),
        "xovl": ({"weight": xovl_weight, "where": xovl_where} if xovl_weight else None),
        "bshape": bshape_report,
        "margin": margin_report,
        "pairs": {"reference": _probe.reference_mode, "reference_id": reference_id},
        "aug": aug,
        "close_aug": close_aug,
        "mask_channels": mask_channels or None,
        # What fraction of the train split was actually usable. Below 1.0 the fine-tune saw
        # only the cached subset, and the record must say so.
        "train_images": sum(a for a, _ in _probe.coverage.values()),
        "train_images_total": sum(t for _, t in _probe.coverage.values()),
        "train_cache_fraction": _probe.cache_fraction,
        "train_coverage_per_group": {g: list(v) for g, v in _probe.coverage.items()},
        "lora": {"r": lora_r, "alpha": lora_alpha, "modules": len(paths), "paths": paths,
                 "trainable_params": n_train},
        "prompt_mode": prompt_mode, "memory_reset": memory_reset,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "loss_per_epoch": per_epoch_loss,
        "losses": losses,
        "checkpoints": checkpoints,
        "method": method,
        "train_mode": train_mode,
        "normalization": normalization,
        "delta_norm_before": delta_before,
        "delta_norm": delta_norm(),
        "peak_vram_gb": round(peak_gb, 3),
        "elapsed_s": round(time.time() - t0, 1),
    }


def train_one_epoch(model, ds, **kw) -> dict:
    """Back-compat shim for S8b. Prefer `train_lora`."""
    kw.pop("epoch", None)
    return train_lora(model, ds, epochs=kw.pop("epochs", 1), **kw)


def diagnostics(model, ds, *, group: str = "all", device: str = "cuda",
                dtype=None, mask_mode: str = "merged",
                margin_m: float | None = None,
                family: str = "sam2", image_size: int = 1024,
                mask_size: int | None = None,
                normalization: str = "imagenet",
                soft_sentinel: float | None = None) -> dict:
    """The three checks the S8 gate reads, measured on one real pair.

    `forward_leg_rel_grad` is computed under BOTH prompt modes. The fixed value alone is
    not evidence — it is only meaningful against the hard-threshold value it replaces,
    which must come back exactly 0.0.

    Under ``mask_mode="per_trait"`` the pair carries C trait channels and each report
    additionally has ``per_channel_forward_to_reverse`` — the S10 gate reads its minimum,
    because a healthy mean can hide one dead channel.

    ``margin_m`` (S13): when set, `cycle.margin_gradient_report` additionally runs on
    the same pair and lands in ``out["margin_grad"]`` — the margin arms' mechanism
    evidence, kept out of `gradient_report` by design (see its docstring).
    """
    tracker = _make_tracker(model, family, soft_sentinel=soft_sentinel)
    x0, xu, y0, ids = CyclePairs(ds, group, mask_mode=mask_mode, size=image_size,
                                 mask_size=mask_size,
                                 normalization=normalization)[0]
    x0, xu, y0 = x0.to(device), xu.to(device), y0.to(device)
    cuda = str(device).startswith("cuda")
    dtype = dtype or (torch.bfloat16 if cuda else torch.float32)
    out = {"pair": list(ids)}
    for mode in ("logit_reparam", "hard_threshold", "ste_threshold", "ste_memory"):
        with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=dtype, enabled=cuda):
            rep = gradient_report(tracker, x0, xu, y0, mode)
        out[f"grad_{mode}"] = rep
        out[f"rel_grad_{mode}"] = rep["forward_to_reverse"]
    if margin_m is not None:
        with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=dtype, enabled=cuda):
            out["margin_grad"] = margin_gradient_report(tracker, x0, xu, y0,
                                                        margin_m=margin_m)
    # The S8 gate's field name. `forward_to_reverse`, not the norm-relative figure — see
    # cycle.gradient_report for why the latter is uninterpretable.
    out["forward_leg_rel_grad"] = out["rel_grad_logit_reparam"]
    return out


def save_lora(model, path, meta: dict | None = None, method: str = "lora") -> str:
    """Write the trained tensors. Key is `lora` or `full` — the LOADER must not guess.

    Both keys are written into one file format rather than two, and the `method` field says
    which is populated. `run_eval`/`run_arms` branch on that key: a full fine-tune loaded
    through `load_lora_state_dict` would report 100% key mismatches, and one loaded through
    a path that ignored the mismatch count would score a part-trained model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"meta": dict(meta or {}, method=method)}
    if method == "full":
        blob["full"] = trainable_state_dict(model)
    else:
        blob["lora"] = lora_state_dict(model)
    torch.save(blob, path)
    return checkpoint_hash(path)
