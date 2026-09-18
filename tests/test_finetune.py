"""Gate S8a — the properties that make an OC-CCL fine-tune interpretable.

A fine-tune can fail silently in ways a loss curve will not show: adapters that never
receive gradient, a cycle whose forward leg is detached, a checkpoint that half-loads, or
training pixels that differ from scoring pixels. Each of those produces a plausible number.
These tests pin the four, and where possible they demonstrate the defect as well as the
fix — an assertion that the fixed path works proves nothing on its own.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from occl.finetune.cycle import (PROMPT_MODES, SAM2_OUT_BIAS, SAM2_OUT_SCALE, dice_loss,
                                 logits_to_prompt)
from occl.finetune.lora import (apply_lora, load_lora_state_dict, lora_delta_norm,
                                lora_modules, lora_parameters, lora_state_dict)


# ── a stand-in with SAM 2's relevant shape ───────────────────────────────────
#
# Mirrors the module names and layer TYPES that decide what LoRA reaches: memory_encoder's
# only Linear layers are the CXBlock pointwise convs, its out_proj is a Conv2d, and Hiera's
# attention is a single fused `qkv`.


class _CX(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dwconv = nn.Conv2d(d, d, 1)
        self.pwconv1 = nn.Linear(d, 4 * d)
        self.pwconv2 = nn.Linear(4 * d, d)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.v_proj, self.out_proj = nn.Linear(d, d), nn.Linear(d, d)


class FakeSam2(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.qkv = nn.Linear(d, 3 * d)     # fused: must never be adapted
        self.memory_attention = _Attn(d)                 # out of spec scope
        self.memory_encoder = nn.Module()
        self.memory_encoder.pix_feat_proj = nn.Conv2d(d, d, 1)
        self.memory_encoder.fuser = nn.Module()
        self.memory_encoder.fuser.layers = nn.ModuleList([_CX(d) for _ in range(2)])
        self.memory_encoder.out_proj = nn.Conv2d(d, d, 1)   # Conv2d, so LoRA must skip it
        self.sam_mask_decoder = nn.Module()
        self.sam_mask_decoder.transformer = nn.ModuleList([_Attn(d)])


# ── A3: LoRA scope, init and round-trip ──────────────────────────────────────


def test_lora_adapts_the_spec_scopes_and_nothing_else():
    """`registry.FAITHFUL_FINETUNE` names memory_encoder and sam_mask_decoder.

    SST's `apply_lora` (`experiments/oc_ccl_ablation.py:41`) has no scope filter at all and
    matches on layer *name*, which reaches neither of these as documented: memory_encoder's
    only projection is a Conv2d, so an `isinstance(nn.Linear)` search skips the module
    entirely and the scope is satisfied vacuously.
    """
    m = FakeSam2()
    paths = apply_lora(m, r=8, alpha=16.0)
    assert paths, "no modules adapted"
    assert not [p for p in paths if p.startswith("image_encoder")], \
        "the frozen image encoder was adapted"
    assert not [p for p in paths if p.startswith("memory_attention")], \
        "memory_attention is not in the spec's scopes"
    assert [p for p in paths if p.startswith("memory_encoder")], \
        "memory_encoder scope is empty — the defect this scoping exists to prevent"
    assert [p for p in paths if p.startswith("sam_mask_decoder")]
    assert not [p for p in paths if p.endswith("out_proj") and "memory_encoder" in p], \
        "memory_encoder.out_proj is a Conv2d and must not be wrapped as Linear"


def test_lora_starts_as_an_exact_no_op():
    """dW == 0 at init, or an S8 delta would measure the perturbation, not the training."""
    m = FakeSam2()
    apply_lora(m, r=8, alpha=16.0)
    assert lora_delta_norm(m) == 0.0
    x = torch.randn(2, 16)
    wrapped = dict(lora_modules(m))["sam_mask_decoder.transformer.0.q_proj"]
    assert torch.allclose(wrapped(x), wrapped.orig(x), atol=1e-6)


def test_lora_moves_after_one_step_and_only_adapters_move():
    m = FakeSam2()
    apply_lora(m, r=8, alpha=16.0)
    base = dict(lora_modules(m))["sam_mask_decoder.transformer.0.q_proj"].orig.weight.clone()
    opt = torch.optim.AdamW(lora_parameters(m), lr=1e-2)
    m.sam_mask_decoder.transformer[0].q_proj(torch.randn(2, 16)).pow(2).mean().backward()
    opt.step()
    assert lora_delta_norm(m) > 0, "gradient never reached the adapters"
    assert torch.equal(
        dict(lora_modules(m))["sam_mask_decoder.transformer.0.q_proj"].orig.weight, base
    ), "the frozen base weight moved"


def test_lora_roundtrip_is_strict_and_counts_mismatches():
    """A partial load would score a model that is part fine-tuned and part base."""
    a = FakeSam2()
    apply_lora(a, r=8, alpha=16.0)
    with torch.no_grad():
        for _, mod in lora_modules(a):
            mod.lora_B.add_(torch.randn_like(mod.lora_B) * 0.1)

    b = FakeSam2()
    apply_lora(b, r=8, alpha=16.0)
    assert load_lora_state_dict(b, lora_state_dict(a)) == 0
    assert lora_delta_norm(b) == pytest.approx(lora_delta_norm(a), rel=1e-6)

    wrong_rank = FakeSam2()
    apply_lora(wrong_rank, r=4, alpha=16.0)
    assert load_lora_state_dict(wrong_rank, lora_state_dict(a)) > 0, \
        "a rank mismatch loaded silently"


def test_empty_lora_scope_raises_rather_than_training_nothing():
    with pytest.raises(RuntimeError, match="matched no nn.Linear"):
        apply_lora(FakeSam2(), r=8, alpha=16.0, scopes=("does_not_exist",))


# ── A2: the differentiable prompt ────────────────────────────────────────────


def test_logit_reparam_is_the_exact_inverse_of_sam2s_mask_transform():
    """`sam2_base.py:417-419` maps a prompt to logits as `mask * 20 - 10`.

    So `(logits + 10) / 20` round-trips to the identity — the reparameterisation changes
    the gradient path without changing the forward value at all. This is what makes it a
    reparameterisation rather than a soft-threshold approximation with error to bound.
    """
    logits = torch.randn(1, 1, 32, 32) * 8.0
    prompt = logits_to_prompt(logits, "logit_reparam")
    reconstructed = prompt * SAM2_OUT_SCALE + SAM2_OUT_BIAS      # what SAM 2 then does
    assert torch.allclose(reconstructed, logits, atol=1e-5)


def test_hard_threshold_kills_the_gradient_and_logit_reparam_does_not():
    """The defect, demonstrated. SST closes the cycle with `(pred > 0).float()`.

    The comparison `>` is not differentiable, so autograd does not merely return zeros — it
    **severs the graph**: the result carries no `grad_fn` and `backward()` through it
    raises. Nothing downstream of the cycle's closing prompt can reach the forward leg at
    all, so only the reverse leg trains.

    Asserting the fixed path has gradient would prove nothing without this half.
    """
    logits = (torch.randn(1, 1, 16, 16) * 4.0).requires_grad_(True)

    hard = logits_to_prompt(logits, "hard_threshold")
    assert not hard.requires_grad and hard.grad_fn is None, \
        "hard threshold unexpectedly stayed attached to the graph"
    with pytest.raises(RuntimeError, match="does not require grad"):
        hard.sum().backward()
    # Keep the graph alive through an independent term, so backward is legal and the hard
    # branch's contribution can be read directly: it is exactly zero.
    (logits_to_prompt(logits, "hard_threshold").sum() + 0.0 * logits.sum()).backward()
    assert float(logits.grad.abs().sum()) == 0.0, \
        "hard threshold contributed gradient it cannot have"
    logits.grad = None

    soft = logits_to_prompt(logits, "logit_reparam")
    assert soft.requires_grad and soft.grad_fn is not None
    soft.sum().backward()
    assert float(logits.grad.abs().sum()) > 0.0, "logit_reparam did not carry gradient"
    # d/dlogits of (l + 10)/20 is exactly 1/20 everywhere — no vanishing regions.
    assert torch.allclose(logits.grad, torch.full_like(logits.grad, 1.0 / SAM2_OUT_SCALE))


def test_unknown_prompt_mode_is_rejected():
    with pytest.raises(ValueError, match="prompt_mode"):
        logits_to_prompt(torch.zeros(1, 1, 4, 4), "soft")
    assert set(PROMPT_MODES) == {"logit_reparam", "hard_threshold", "ste_threshold", "ste_memory"}


def test_dice_loss_is_perfect_on_an_exact_match():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    confident = (target * 2 - 1) * 30.0            # +-30 logits
    assert float(dice_loss(confident, target)) < 1e-3
    assert float(dice_loss(-confident, target)) > 0.9


# ── A1: training pixels are scoring pixels ───────────────────────────────────


def test_training_tensor_is_bit_identical_to_the_inference_tensor():
    """A1. SST normalises with `/255` alone (`butterfly_dataset.py:139-140`) while every
    inference path subtracts ImageNet mean/std (`sam2/utils/misc.py:164-165`).

    Both sides here go through `adapters.sst_sam2.to_model_tensor`, so this asserts the
    single-source-of-truth actually holds rather than that the constants happen to match.
    """
    from occl.adapters.sst_sam2 import IMAGENET_MEAN, IMAGENET_STD, to_model_tensor
    from occl.finetune.pairs import foreground_mask

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (137, 211, 3), dtype=np.uint8)

    train_side = to_model_tensor(img, 1024)
    infer_side = to_model_tensor(img, 1024)
    assert torch.equal(train_side, infer_side)
    assert train_side.shape == (1, 3, 1024, 1024)

    # And it is genuinely NOT the /255-only tensor SST trains on.
    import cv2
    sst = torch.from_numpy(
        cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    ).permute(2, 0, 1).float().div_(255.0)[None]
    assert float((train_side - sst).abs().max()) > 0.5, \
        "normalisation is indistinguishable from SST's — A1 would be a no-op"

    # The transform must be exactly (x/255 - mean) / std, not merely 'some normalisation'.
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    assert torch.allclose(train_side, (sst - mean) / std, atol=1e-6)

    lm = np.zeros((137, 211), dtype=np.int32)
    lm[10:40, 10:60] = 3
    fg = foreground_mask(lm, 1024)
    assert fg.shape == (1, 1, 1024, 1024)
    assert set(np.unique(fg.numpy())) <= {0.0, 1.0}, "mask must stay binary through resize"


# ── pair sampling: train-only, deranged, seeded ──────────────────────────────


def test_cycle_pairs_never_touch_the_test_split():
    """A leak here would put the evaluation set into the weights and void every S8 number."""
    from occl.datasets.beetle import Beetle
    from occl.finetune.pairs import CyclePairs

    ds = Beetle()
    pairs = CyclePairs(ds, "all").pairs
    used = {i for pair in pairs for i in pair}
    assert not (used & set(ds.test_ids())), "test images reached the training pairs"
    assert all(a != b for a, b in pairs), "a pair is degenerate (reference == query)"
    assert sorted(a for a, _ in pairs) == sorted(ds.train_ids())
    assert sorted(b for _, b in pairs) == sorted(ds.train_ids())


def test_cycle_pairs_are_seeded_like_every_other_draw():
    from occl.datasets.beetle import Beetle
    from occl.finetune.pairs import CyclePairs

    ds = Beetle()
    assert CyclePairs(ds, "all").pairs == CyclePairs(ds, "all").pairs
    assert CyclePairs(ds, "all", epoch=1).pairs != CyclePairs(ds, "all").pairs


def test_filing_an_already_filed_bundle_does_not_destroy_it():
    """`check_gate --bundle results/<stage>.zip` must be safe to re-run.

    `file_bundle` unlinks the destination before moving. When the bundle is already at the
    destination — which is exactly what happens re-running a gate against a filed bundle —
    src and dst are the same file, so the unlink DELETES it and the move then fails with
    the result gone for good. A GPU-session bundle is not always cheap to regenerate, and
    this is the one operation in the repo that can lose one.
    """
    import json
    import shutil
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    with __import__("tempfile").TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        (results / "inbox").mkdir(parents=True)
        bundle = results / "s0.zip"
        with zipfile.ZipFile(bundle, "w") as z:
            z.writestr("record.json", json.dumps({"stage": "s0"}))
            z.writestr("env.json", json.dumps({}))
        before = bundle.read_bytes()

        sys.path.insert(0, str(root / "scripts"))
        sys.path.insert(0, str(root / "src"))
        import check_gate

        rec = {"_bundle_path": str(bundle)}
        orig_results_root = check_gate.paths.results_root
        check_gate.paths.results_root = lambda: results
        try:
            check_gate.file_bundle("s0", rec)
        finally:
            check_gate.paths.results_root = orig_results_root

        assert bundle.exists(), "filing an already-filed bundle deleted it"
        assert bundle.read_bytes() == before, "the bundle was rewritten"


def test_s8b_bundle_carries_the_adapters_it_produced():
    """The adapters are S8b's whole product — S8c cannot run without them.

    They were originally written next to the bundle rather than inside it, which put them
    on the Colab VM's local disk. The bundle came back; the weights did not, and the
    session that produced them was already gone. A stage whose artifact does not survive
    the trip is a stage that has to be re-run to be used.
    """
    import json
    import tempfile
    import zipfile
    from pathlib import Path

    from occl.bundle import export_bundle
    from occl.finetune.lora import apply_lora, load_lora_state_dict, lora_delta_norm
    from occl.finetune.train import checkpoint_hash, save_lora

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        m = FakeSam2()
        apply_lora(m, r=32, alpha=64.0)
        with torch.no_grad():
            for _, mod in lora_modules(m):
                mod.lora_B.add_(torch.randn_like(mod.lora_B) * 0.05)
        trained = lora_delta_norm(m)
        assert trained > 0

        pt = tmp / "lr_lora.pt"
        sha = save_lora(m, pt, meta={"lora": {"r": 32, "alpha": 64.0}})
        rec = {"dataset": "beetle", "finetune": {"lora_sha": sha}}
        zp = export_bundle("s8b", rec, out=tmp / "s8b.zip", extra_files={"lora.pt": pt})

        with zipfile.ZipFile(zp) as z:
            assert "lora.pt" in z.namelist(), "adapters are not in the bundle"
            extracted = tmp / "out.pt"
            extracted.write_bytes(z.read("lora.pt"))
            recorded = json.loads(z.read("record.json"))["finetune"]["lora_sha"]

        assert checkpoint_hash(extracted) == recorded, \
            "the shipped adapters do not match the record describing them"

        fresh = FakeSam2()
        apply_lora(fresh, r=32, alpha=64.0)
        blob = torch.load(extracted, map_location="cpu", weights_only=False)
        assert load_lora_state_dict(fresh, blob["lora"]) == 0
        assert lora_delta_norm(fresh) == pytest.approx(trained, rel=1e-9), \
            "adapters changed in transit"


def test_notebook_build_guard_matches_the_package():
    """The notebook refuses to run against a repo zip older than it was packed with.

    They ship as a pair from `scripts/pack.sh`, but only the notebook is visible in Colab,
    so a stale `occl-repo.zip` on Drive is easy to miss and expensive to discover: the run
    succeeds, the bundle looks normal, the gate passes, and the defect surfaces a session
    later. This pins that `REPO_MIN_BUILD` never exceeds the `occl.BUILD` actually shipped
    — otherwise the guard fires on a correct pairing and the only fix is to ignore it.
    """
    import re
    from pathlib import Path

    import occl

    nb = (Path(__file__).resolve().parent.parent / "notebooks" / "run_stage.py").read_text()
    m = re.search(r"^REPO_MIN_BUILD = (\d+)", nb, re.M)
    assert m, "the notebook lost its build guard"
    assert int(m.group(1)) <= occl.BUILD, (
        f"notebook requires build {m.group(1)} but the package ships {occl.BUILD} — "
        "pack.sh would produce a pair that refuses to run"
    )


def test_reset_lora_makes_grid_points_independent():
    """A grid search reuses one built model across learning rates, so each point must
    start from the same state. If a reset left residue, every point after the first would
    be training from its predecessor's solution and the whole grid would be contaminated
    in an order-dependent way that no output would reveal.
    """
    from occl.finetune.lora import reset_lora

    m = FakeSam2()
    apply_lora(m, r=32, alpha=64.0)

    def train_a_bit(seed: int) -> float:
        reset_lora(m)
        assert lora_delta_norm(m) == 0.0, "reset left residue"
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(lora_parameters(m), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            m.sam_mask_decoder.transformer[0].q_proj(torch.ones(4, 16)).pow(2).mean().backward()
            opt.step()
        return lora_delta_norm(m)

    torch.manual_seed(7)
    first = train_a_bit(0)
    _other = train_a_bit(99)            # a different "grid point" runs in between
    torch.manual_seed(7)
    again = train_a_bit(0)
    assert first == again, "a grid point's result depended on what ran before it"
    assert first > 0


def test_reset_lora_restores_an_exact_no_op():
    from occl.finetune.lora import LoRALinear, reset_lora

    m = FakeSam2()
    apply_lora(m, r=16, alpha=32.0)
    with torch.no_grad():
        for _, mod in lora_modules(m):
            mod.lora_B.add_(torch.randn_like(mod.lora_B) * 0.2)
    assert lora_delta_norm(m) > 0
    assert reset_lora(m) == len(lora_modules(m))
    assert lora_delta_norm(m) == 0.0
    w = dict(lora_modules(m))["sam_mask_decoder.transformer.0.q_proj"]
    assert isinstance(w, LoRALinear)
    x = torch.randn(3, 16)
    assert torch.allclose(w(x), w.orig(x), atol=1e-6)


def test_train_lora_reuses_existing_adapters_instead_of_nesting_them():
    """Calling `apply_lora` twice would wrap adapters in adapters — silently doubling the
    parameter count and changing the function being optimised."""
    from occl.finetune.lora import LoRALinear

    m = FakeSam2()
    apply_lora(m, r=8, alpha=16.0)
    n_before = len(lora_modules(m))
    apply_lora_again = [mod for _, mod in lora_modules(m)]
    assert all(isinstance(mod.orig, torch.nn.Linear) for mod in apply_lora_again), \
        "an adapter is already wrapping another adapter"
    # A second apply_lora would nest; reset_lora is what train_lora uses instead.
    from occl.finetune.lora import reset_lora
    reset_lora(m)
    assert len(lora_modules(m)) == n_before
    assert all(not isinstance(mod.orig, LoRALinear) for _, mod in lora_modules(m))


def test_inference_mode_caches_are_detached_before_training():
    """SAM 2 memoises its rotary encoding on the attention modules
    (`sam2/modeling/sam/transformer.py:308`) and only recomputes it when the feature shape
    changes. So scoring a model and then training it in the SAME process dies with

        RuntimeError: Inference tensors cannot be saved for backward.

    raised deep inside `apply_rotary_enc`, where it reads as a bug in the cycle rather than
    a consequence of what ran earlier. Training in a fresh process never hits it, which is
    why the S8b sweep was fine and the grid search — which scores the base arm first —
    failed on all five folds at once.
    """
    from occl.finetune.train import detach_inference_caches

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.freqs_cis = None

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = Attn(), Attn()

    m = Model()
    with torch.inference_mode():                       # what a scoring pass does
        m.a.freqs_cis = torch.randn(4, 4)
        m.b.freqs_cis = torch.randn(4, 4)
    assert torch.is_inference(m.a.freqs_cis)

    # The failure, demonstrated: autograd cannot use these tensors.
    x = torch.randn(4, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match="Inference tensors"):
        (x * m.a.freqs_cis).sum().backward()

    assert detach_inference_caches(m) == 2
    assert not torch.is_inference(m.a.freqs_cis)
    (x * m.a.freqs_cis).sum().backward()               # now legal
    assert x.grad is not None

    # Idempotent, and a no-op on a model that never ran inference.
    assert detach_inference_caches(m) == 0
    assert detach_inference_caches(Model()) == 0


def test_a_reused_model_still_has_trainable_adapters():
    """The exact sequence `train_lora` runs on the SECOND grid point.

    `freeze_all` freezes every parameter. On the first point the adapters do not exist yet,
    so `apply_lora` creates them trainable; on reuse they are already there and get frozen
    with everything else. The earlier reset test created a fresh optimiser without ever
    calling `freeze_all`, so it passed while the real path was broken — a test that
    exercised the function but not the sequence.

    Symptom when this regresses: ten epochs of the first learning rate train correctly and
    the second dies in `loss.backward()` with "element 0 of tensors does not require grad".
    """
    from occl.finetune.lora import reset_lora
    from occl.finetune.train import freeze_all

    m = FakeSam2()

    # first grid point: freeze, then apply
    freeze_all(m)
    apply_lora(m, r=8, alpha=16.0)
    first = lora_parameters(m)
    assert first and all(p.requires_grad for p in first), "fresh adapters are not trainable"

    # second grid point: freeze again (now hitting the adapters), then reset
    freeze_all(m)
    assert not any(p.requires_grad for p in lora_parameters(m)), \
        "freeze_all did not freeze the existing adapters — the premise changed"
    reset_lora(m)
    second = lora_parameters(m)
    assert all(p.requires_grad for p in second), \
        "adapters stayed frozen after reset — the second grid point would train nothing"

    # and a backward pass actually produces gradients
    opt = torch.optim.AdamW(second, lr=1e-2)
    m.sam_mask_decoder.transformer[0].q_proj(torch.randn(4, 16)).pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in second)
    opt.step()
    assert lora_delta_norm(m) > 0

    # base weights must STILL be frozen — the fix must not unfreeze the model
    base = [p for n, p in m.named_parameters() if "lora_" not in n]
    assert not any(p.requires_grad for p in base), "the fix unfroze the base model"


def test_train_lora_refuses_to_run_with_frozen_adapters(monkeypatch):
    """The backstop, exercised by simulating the regression it guards against.

    `reset_lora` now restores `requires_grad`, so this state is unreachable through the
    normal path — which is why the guard needs a deliberately broken reset to test at all.
    It earns its place because the failure it replaces (`element 0 of tensors does not
    require grad`, raised inside autograd) costs a full training run to reach and names
    neither LoRA nor freezing.
    """
    from occl.finetune import train as train_mod
    from occl.finetune.train import freeze_all, train_lora

    m = FakeSam2()
    freeze_all(m)
    apply_lora(m, r=4, alpha=8.0)

    monkeypatch.setattr(train_mod, "reset_lora", lambda _model: 0)   # the regression
    freeze_all(m)
    with pytest.raises(RuntimeError, match="none require grad"):
        train_lora(m, ds=None, lr=1e-4, epochs=1, device="cpu")


def test_cycle_pairs_use_only_images_that_are_cached():
    """The image cache is built for the EVALUATION protocol, not the whole train split.

    Beetle's 180 images are fully covered. Butterfly Major has 4,227 train images of which
    ~1,100 are cached — test queries plus the supports each selection arm draws — so an
    unfiltered fine-tune dies on the first uncached reference. That is exactly how the
    butterfly fine-tune failed twice.

    Filtering is a deviation, so the fraction has to reach the record rather than be
    absorbed. This pins both halves: only cached images are used, and the coverage is
    reported.
    """
    from occl.datasets.butterfly_major import ButterflyMajor
    from occl.finetune.pairs import CyclePairs

    class HalfCache:
        """Only `has` is exercised; a real ImageCache would create directories."""

        def has(self, dataset, image_id):
            return (int(image_id) % 2) == 0 if image_id.isdigit() else True

    ds = ButterflyMajor(cache=HalfCache())
    pairs = CyclePairs(ds, None)
    assert all(ds.cache.has(ds.name, i) for pr in pairs.pairs for i in pr), \
        "an uncached image reached a training pair"
    assert 0 < pairs.cache_fraction < 1, pairs.cache_fraction
    assert set(pairs.coverage) == set(ds.groups()), "coverage is not reported per group"
    for g, (avail, total) in pairs.coverage.items():
        assert 0 < avail <= total, (g, avail, total)


def test_a_group_with_too_few_cached_images_fails_loudly():
    from occl.datasets.butterfly_major import ButterflyMajor
    from occl.finetune.pairs import CyclePairs

    class EmptyCache:
        def has(self, dataset, image_id):
            return False

    with pytest.raises(ValueError, match="usable train images"):
        CyclePairs(ButterflyMajor(cache=EmptyCache()), None)


def test_lora_init_is_reproducible_and_reset_matches_apply():
    """Adapters must be identical across processes, or nothing about a fine-tune resumes.

    `lora_A` came from the global torch RNG, which PyTorch seeds from OS entropy — two runs
    of the same recipe gave checksums 64.303 and 63.090. `evaluator.run_fingerprint` keys
    the resume cache on `||dW||`, so a regenerated adapter orphans every evaluation ever run
    against its predecessor; one interrupted butterfly stage lost hours of banked work that
    way. It also left adapter-draw variance outside every reported confidence interval.
    """
    from occl.finetune.lora import reset_lora

    a, b = FakeSam2(), FakeSam2()
    apply_lora(a, r=8, alpha=16.0)
    apply_lora(b, r=8, alpha=16.0)
    for (na, ma), (nb, mb) in zip(lora_modules(a), lora_modules(b)):
        assert na == nb
        assert torch.equal(ma.lora_A, mb.lora_A), f"{na} differs between fresh models"

    # Distinct modules must NOT share an init, or the adapter set is degenerate.
    mods = dict(lora_modules(a))
    p1 = mods["sam_mask_decoder.transformer.0.q_proj"].lora_A
    p2 = mods["sam_mask_decoder.transformer.0.k_proj"].lora_A
    assert not torch.equal(p1, p2), "every module got the same init"

    # reset must reproduce apply exactly, or a grid point differs from a fresh run.
    with torch.no_grad():
        for _, m in lora_modules(a):
            m.lora_A.add_(1.0)
            m.lora_B.add_(0.5)
    reset_lora(a)
    for (na, ma), (_nb, mb) in zip(lora_modules(a), lora_modules(b)):
        assert torch.equal(ma.lora_A, mb.lora_A), f"{na}: reset != apply"
        assert float(ma.lora_B.abs().sum()) == 0.0
    assert lora_delta_norm(a) == 0.0


def test_unseeded_lora_is_still_reachable_for_ablation():
    a, b = FakeSam2(), FakeSam2()
    apply_lora(a, r=8, alpha=16.0, seed=None)
    apply_lora(b, r=8, alpha=16.0, seed=None)
    same = all(torch.equal(ma.lora_A, mb.lora_A)
               for (_, ma), (_, mb) in zip(lora_modules(a), lora_modules(b)))
    assert not same, "seed=None should draw from the global RNG"


# ── S10: the paper's per-trait cycle, fixed reference, and rotation aug ──────
#
# These pin the three axes S10 varies against the release: per-trait supervision
# (paper §3: y ∈ {0,1}^{W×H×C}), the one-fixed-reference protocol (A2.2), and the
# beetle rotation augmentation on the labeled example (A2.2).


class FakePairDS:
    """A dataset stub with the pieces CyclePairs touches — no cache, tiny images."""

    name = "fakebeetle"

    def groups(self):
        return ("all",)

    def train_ids(self, group="all"):
        return ["a", "b", "c"]

    def test_ids(self, group="all"):
        return ["t"]

    def load_image(self, image_id):
        rng = np.random.default_rng(abs(hash(image_id)) % 2**32)
        return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)

    def load_labelmap(self, image_id):
        lm = np.zeros((24, 32), dtype=np.int32)
        lm[8:16, 10:18] = 1          # centred blob — survives any rotation
        lm[10:14, 20:24] = 3         # second trait, also interior
        return lm


def test_trait_mask_stack_partitions_the_foreground():
    from occl.finetune.pairs import foreground_mask, trait_mask_stack

    lm = np.zeros((30, 40), dtype=np.int32)
    lm[2:8, 3:9] = 1
    lm[12:20, 5:15] = 3
    lm[22:28, 30:38] = 7
    stack, ids = trait_mask_stack(lm, 64)
    assert ids == (1, 3, 7), "channels must be ascending trait ids present in the map"
    assert stack.shape == (3, 1, 64, 64)
    assert set(np.unique(stack.numpy())) <= {0.0, 1.0}

    # The channels partition the merged foreground: disjoint, and their union is
    # exactly foreground_mask — same NEAREST resize, same geometry.
    union = stack.sum(dim=0, keepdim=True)
    assert float(union.max()) == 1.0, "trait channels overlap"
    assert torch.equal(union, foreground_mask(lm, 64))

    with pytest.raises(ValueError, match="no traits"):
        trait_mask_stack(np.zeros((10, 10), dtype=np.int32), 64)


def test_per_trait_at_one_trait_is_bitwise_the_merged_mask():
    """C=1 per-trait must equal merged exactly, so the two modes share one geometry."""
    from occl.finetune.pairs import foreground_mask, trait_mask_stack

    lm = np.zeros((20, 20), dtype=np.int32)
    lm[5:15, 5:15] = 4
    stack, ids = trait_mask_stack(lm, 32)
    assert ids == (4,)
    assert torch.equal(stack, foreground_mask(lm, 32))


def test_fixed_reference_pairs_spend_exactly_one_label():
    """A2.2: "We take one fixed labeled image as x0, and iteratively sample x1 from the
    unlabeled training set." One reference, N-1 queries, test split untouched."""
    from occl.datasets.beetle import Beetle
    from occl.finetune.pairs import CyclePairs

    ds = Beetle()
    ref = sorted(ds.train_ids())[0]
    p = CyclePairs(ds, "all", reference_id=ref)
    assert p.reference_mode == "fixed"
    assert len(p.pairs) == len(ds.train_ids()) - 1
    assert all(a == ref for a, _ in p.pairs), "every pair must use the one labeled x0"
    assert sorted(b for _, b in p.pairs) == sorted(set(ds.train_ids()) - {ref})
    assert not ({q for _, q in p.pairs} & set(ds.test_ids()))

    # Seeded like every other draw: same epoch reproduces, different epoch permutes.
    assert CyclePairs(ds, "all", reference_id=ref).pairs == p.pairs
    assert CyclePairs(ds, "all", reference_id=ref, epoch=1).pairs != p.pairs

    with pytest.raises(ValueError, match="not a usable train image"):
        CyclePairs(ds, "all", reference_id="no-such-image")
    # The derangement default is byte-identical to before the axis existed.
    assert CyclePairs(ds, "all").reference_mode == "derangement"


def test_rotation_aug_is_seeded_bounded_and_leaves_the_query_alone():
    from occl.finetune.pairs import CyclePairs

    ds = FakePairDS()
    kw = dict(mask_mode="per_trait", rotation_aug_deg=27.8, size=64)
    p1 = CyclePairs(ds, "all", **kw)
    p2 = CyclePairs(ds, "all", **kw)
    x0a, xua, y0a, ids_a = p1[0]
    x0b, xub, y0b, ids_b = p2[0]

    # Deterministic: the same (pair, epoch, step) draws the same angle and tensors.
    assert ids_a == ids_b
    assert torch.equal(x0a, x0b) and torch.equal(y0a, y0b)
    log = p1.step_log[0]
    assert abs(log["aug_drawn"]) <= 27.8
    assert log["aug_angle"] == (0.0 if log["aug_fallback"] else log["aug_drawn"])
    assert log["class_ids"] == [1, 3]

    # Interior traits survive rotation: no fallback, and the mask genuinely rotated
    # (the drawn angle for this seeded pair is comfortably non-zero).
    plain = CyclePairs(ds, "all", mask_mode="per_trait", size=64)
    x0p, xup, y0p, _ = plain[0]
    assert not log["aug_fallback"]
    assert abs(log["aug_angle"]) > 1.0, "seeded angle unexpectedly ~0; reseed the fake"
    assert not torch.equal(y0a, y0p), "the mask did not rotate"
    assert not torch.equal(x0a, x0p), "the image did not rotate"
    # A2.2 rotates THE LABELED EXAMPLE; the query must be untouched.
    assert torch.equal(xua, xup)
    assert set(np.unique(y0a.numpy())) <= {0.0, 1.0}, "rotation must keep masks binary"

    # A trait clipped out of frame falls back to 0° and says so.
    class CornerDS(FakePairDS):
        def load_labelmap(self, image_id):
            lm = np.zeros((24, 32), dtype=np.int32)
            lm[0, 0] = 1              # extreme corner: any real rotation clips it
            lm[10:14, 14:18] = 2
            return lm

    pc = CyclePairs(CornerDS(), "all", **kw)
    x0c, _, y0c, _ = pc[0]
    logc = pc.step_log[0]
    if logc["aug_fallback"]:
        assert logc["aug_angle"] == 0.0
        plain_c = CyclePairs(CornerDS(), "all", mask_mode="per_trait", size=64)
        assert torch.equal(y0c, plain_c[0][2]), "fallback must mean the unrotated mask"
    else:  # the seeded angle happened to keep the corner pixel — invariant still holds
        assert (y0c[0].sum() > 0), "trait lost without a recorded fallback"


def test_reparam_roundtrip_and_hard_threshold_are_shape_general():
    """The per-channel cycle relies on `logits_to_prompt` being elementwise: the round
    trip must be exact and the hard threshold structurally dead on EVERY channel."""
    from occl.finetune.cycle import logits_to_prompt

    logits = torch.randn(3, 1, 8, 8) * 15.0
    logits.requires_grad_(True)
    prompt = logits_to_prompt(logits, "logit_reparam")
    assert torch.allclose(prompt * SAM2_OUT_SCALE + SAM2_OUT_BIAS, logits, atol=1e-5)
    prompt.sum().backward()
    assert torch.allclose(logits.grad,
                          torch.full_like(logits, 1.0 / SAM2_OUT_SCALE))

    hard = logits_to_prompt(logits.detach().requires_grad_(True), "hard_threshold")
    assert hard.grad_fn is None, "the hard threshold must sever the graph"
    assert hard.shape == logits.shape


class FakeCycleTracker:
    """Object-independent tracker stub: each channel is transformed on its own, which is
    exactly SAM 2's property (objects share features but never mix in the memory). Lets
    the C=2 cycle be checked against two C=1 cycles without a GPU or a checkpoint."""

    def encode_image(self, img):
        return {"backbone_fpn": [img.mean(dim=1, keepdim=True)],
                "vision_pos_enc": [torch.zeros_like(img[:, :1])]}

    def track_with_mask(self, backbone, mask, is_init_frame=True, output_dict=None,
                        n_objects=1):
        output_dict = output_dict or {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        if is_init_frame:
            assert mask is not None and mask.shape[0] == n_objects
            out = {"pred_masks_high_res": mask * 20.0 - 10.0}      # _use_mask_as_output
            output_dict["cond_frame_outputs"][0] = out["pred_masks_high_res"]
            return out, output_dict
        prev = output_dict["cond_frame_outputs"][0]
        # A fixed, channel-wise "propagation": shift logits toward zero, per object.
        return {"pred_masks_high_res": prev * 0.5}, output_dict


def test_c2_cycle_loss_is_the_mean_of_the_two_c1_cycle_losses():
    """The multi-object cycle must reduce to independent per-channel cycles: BCE is a
    global mean over equal-sized channels and Dice is per-batch-element then mean, so a
    C=2 loss equals the mean of the two C=1 losses when the tracker is object-independent
    — which SAM 2's object batch is. Wiring, not model behaviour, is what this pins.

    Since S13 this identity holds only with every cross-channel knob OFF — `excl`
    breaks it BY DESIGN (`test_excl_breaks_the_c2_mean_identity_by_design`) — so this
    test doubles as the knobs-off regression pin."""
    from occl.finetune.cycle import oc_ccl_step

    torch.manual_seed(0)
    x0 = torch.rand(1, 3, 16, 16)
    xu = torch.rand(1, 3, 16, 16)
    m1 = (torch.rand(1, 1, 16, 16) > 0.6).float()
    m2 = (torch.rand(1, 1, 16, 16) > 0.4).float()

    tr = FakeCycleTracker()
    loss_a, _, parts_a = oc_ccl_step(tr, x0, xu, m1)
    loss_b, _, _ = oc_ccl_step(tr, x0, xu, m2)
    loss_ab, _, parts_ab = oc_ccl_step(tr, x0, xu, torch.cat([m1, m2]))

    assert parts_ab["n_objects"] == 2
    assert len(parts_ab["dice_per_channel"]) == 2
    assert float(loss_ab) == pytest.approx((float(loss_a) + float(loss_b)) / 2, rel=1e-5)


def test_expand_backbone_returns_views_and_never_mutates_the_input():
    from occl.finetune.tracker import _expand_backbone

    fpn = [torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)]
    pos = [torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)]
    backbone = {"backbone_fpn": fpn, "vision_pos_enc": pos, "other": "kept"}
    out = _expand_backbone(backbone, 5)
    assert all(t.shape[0] == 5 for t in out["backbone_fpn"] + out["vision_pos_enc"])
    assert backbone["backbone_fpn"][0].shape[0] == 1, "the caller's dict was mutated"
    assert out["other"] == "kept"
    # Views, not copies — expansion must stay free at every cycle leg.
    assert out["backbone_fpn"][0].data_ptr() == fpn[0].data_ptr()


def test_scnp_composes_per_channel_on_the_per_trait_cycle():
    """S12 lifted the deliberate guard: SCNP x per-trait is now the experiment, and its
    semantics must be per-channel independence — each trait channel transformed against
    its own target, so a C=2 SCNP cycle loss equals the mean of the two C=1 SCNP cycle
    losses under an object-independent tracker. That is what makes "same-class
    neighbour" mean same-TRAIT neighbour."""
    from occl.finetune.cycle import oc_ccl_step

    torch.manual_seed(1)
    x0 = torch.rand(1, 3, 16, 16)
    xu = torch.rand(1, 3, 16, 16)
    m1 = (torch.rand(1, 1, 16, 16) > 0.6).float()
    m2 = (torch.rand(1, 1, 16, 16) > 0.4).float()

    tr = FakeCycleTracker()
    la, _, _ = oc_ccl_step(tr, x0, xu, m1, scnp_width=3)
    lb, _, _ = oc_ccl_step(tr, x0, xu, m2, scnp_width=3)
    lab, _, parts = oc_ccl_step(tr, x0, xu, torch.cat([m1, m2]), scnp_width=3)
    assert parts["n_objects"] == 2
    assert float(lab) == pytest.approx((float(la) + float(lb)) / 2, rel=1e-5)

    # And SCNP genuinely changes the per-trait loss (the guard is gone, not bypassed).
    plain, _, _ = oc_ccl_step(tr, x0, xu, torch.cat([m1, m2]))
    assert float(lab) != pytest.approx(float(plain), rel=1e-6)


def test_compare_pairs_on_a_chosen_per_run_metric():
    """S12's headline is the paired best-of-3 delta, so `registry.compare` must pair on
    a dotted per-run metric path — with the default byte-identical to before."""
    from occl.registry import compare, per_seed_scores

    def rec(vals, best3):
        return {
            "dataset": "beetle",
            "model": {"family": "sam2", "image_size": 1024},
            "metric": {"name": "iou_per_trait_macro", "assignment": "argmax_label",
                       "reduction": "macro_over_gt_traits", "iou_space": "gt_native"},
            "data": {"source_preference": "faithful", "interp": "linear",
                     "cache_format": "png"},
            "protocol": {"shots": 1, "n_runs": len(vals), "groups": ["all"],
                         "support_selection": "canonical_balanced",
                         "support_set_hash": "h"},
            "results": {"miou": sum(vals) / len(vals), "per_run": list(vals)},
            "runs": [{"run_idx": i, "miou": v,
                      "miou_boundary": {"independent_best3": b}}
                     for i, (v, b) in enumerate(zip(vals, best3))],
        }

    a = rec([0.60, 0.62, 0.64], [0.65, 0.66, 0.67])
    b = rec([0.59, 0.62, 0.63], [0.61, 0.63, 0.64])
    plain = compare(a, b, n_boot=200)
    assert plain["metric"] == "miou"
    assert plain["mean_delta"] == pytest.approx((0.01 + 0.0 + 0.01) / 3)

    b3 = compare(a, b, n_boot=200, metric="miou_boundary.independent_best3")
    assert b3["metric"] == "miou_boundary.independent_best3"
    assert b3["mean_delta"] == pytest.approx((0.04 + 0.03 + 0.03) / 3)

    # A non-default metric on a record without per-run data is loudly empty, never a
    # silent fallback to the plain metric.
    del a["runs"]
    assert per_seed_scores(a, "miou_boundary.independent_best3") == {}


def test_s10_aug_magnitude_is_the_a_priori_constant_not_a_tuned_knob():
    """±27.8° = half the beetle train pool's circular std (55.6°, the s4b constant).

    The paper says only "a small degree"; this pin is what stops the angle drifting
    toward whatever scores well. If S10 misses, report the gap — do not move the angle.
    Parsed from the notebook source so the value the Colab session actually uses is the
    value under test.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "notebooks" / "run_stage.py")
    m = re.search(r"^S10_AUG_DEG\s*=\s*([0-9.]+)", src.read_text(), re.M)
    assert m, "S10_AUG_DEG missing from notebooks/run_stage.py"
    assert float(m.group(1)) == pytest.approx(55.6 / 2, abs=0.01)


def test_gradient_report_records_which_traits_the_query_found():
    """The real-model smoke found one trait channel with EXACTLY zero forward gradient —
    the trait the model predicted absent everywhere on the query (its prompt has no value
    > 0, so `fixed_no_obj_ptr` multiplies the differentiable obj_ptr by an exact 0.0).
    That is a property of the pair, not a wiring bug, so the s10ft gate conditions the
    per-channel requirement on `per_channel_query_found`. This pins the recording."""
    from occl.finetune.cycle import gradient_report

    torch.manual_seed(0)
    x0 = torch.rand(1, 3, 16, 16)
    xu = torch.rand(1, 3, 16, 16)
    m_found = (torch.rand(1, 1, 16, 16) > 0.5).float()
    m_absent = torch.zeros(1, 1, 16, 16)      # -> all -10 logits -> never claimed

    rep = gradient_report(FakeCycleTracker(), x0, xu, torch.cat([m_found, m_absent]))
    assert rep["n_objects"] == 2
    assert rep["per_channel_query_found"] == [True, False]
    assert len(rep["per_channel_forward_to_reverse"]) == 2
    assert len(rep["per_channel_forward_norm"]) == 2
    assert len(rep["per_channel_reverse_norm"]) == 2
    # C=1 keeps the scalar shape and still reports foundness.
    rep1 = gradient_report(FakeCycleTracker(), x0, xu, m_found)
    assert rep1["per_channel_forward_to_reverse"] is None
    assert rep1["per_channel_query_found"] == [True]


# ── S13: the five knobs on the cycle ─────────────────────────────────────────
#
# The fake's fixed semantics make every expected value computable in the test itself:
# forward pred = (y0*20-10)*0.5, closing pred = forward*0.5 (reparam round-trips
# exactly), and a 180 flip only permutes pixels, so both margin legs have identical
# claimed confidence and the hinge costs exactly margin_m.


class _CountingTracker(FakeCycleTracker):
    def __init__(self):
        self.encodes = 0
        self.tracks = 0

    def encode_image(self, img):
        self.encodes += 1
        return super().encode_image(img)

    def track_with_mask(self, *a, **k):
        self.tracks += 1
        return super().track_with_mask(*a, **k)


def _pair(seed=0, size=16):
    torch.manual_seed(seed)
    x0 = torch.rand(1, 3, size, size)
    xu = torch.rand(1, 3, size, size)
    m1 = (torch.rand(1, 1, size, size) > 0.6).float()
    m2 = (torch.rand(1, 1, size, size) > 0.4).float()
    return x0, xu, m1, m2


def test_s13_knobs_off_is_byte_identical_and_makes_zero_extra_tracker_calls():
    from occl.finetune.cycle import oc_ccl_step

    x0, xu, m1, m2 = _pair()
    y = torch.cat([m1, m2])
    ta, tb = _CountingTracker(), _CountingTracker()
    la, _, pa = oc_ccl_step(ta, x0, xu, y)
    lb, _, pb = oc_ccl_step(tb, x0, xu, y, excl_mode="off", xovl_weight=0.0,
                            xovl_where="query", bshape_weight=0.0, margin_weight=0.0,
                            margin_m=1.0)
    assert float(la) == float(lb)
    assert pa == pb
    assert (ta.encodes, ta.tracks) == (tb.encodes, tb.tracks) == (2, 4)


def test_excl_breaks_the_c2_mean_identity_by_design_and_is_bce_at_c1():
    """The documented flip of the C=2==mean pin: a cross-channel CE couples channels, so
    the identity MUST fail when excl is on — while at C=1 zerocat==BCE keeps the excl
    cycle bit-identical to the plain cycle."""
    from occl.finetune.cycle import oc_ccl_step

    x0, xu, m1, m2 = _pair()
    tr = FakeCycleTracker()
    la, _, _ = oc_ccl_step(tr, x0, xu, m1, excl_mode="zerocat")
    lb, _, _ = oc_ccl_step(tr, x0, xu, m2, excl_mode="zerocat")
    lab, _, parts = oc_ccl_step(tr, x0, xu, torch.cat([m1, m2]), excl_mode="zerocat")
    assert float(lab) != pytest.approx((float(la) + float(lb)) / 2, rel=1e-5)
    assert parts["excl_mode"] == "zerocat"
    assert {"ce", "bce", "dice", "dice_softmax"} <= set(parts)

    plain1, _, _ = oc_ccl_step(tr, x0, xu, m1)
    assert float(la) == pytest.approx(float(plain1), rel=1e-6)


def test_excl_refuses_scnp_composition():
    from occl.finetune.cycle import oc_ccl_step

    x0, xu, m1, _ = _pair()
    with pytest.raises(ValueError, match="composition"):
        oc_ccl_step(FakeCycleTracker(), x0, xu, m1, scnp_width=3, excl_mode="zerocat")


def test_xovl_lands_on_the_forward_leg_tensor_and_skips_c1():
    """The loss delta equals the penalty evaluated at the FORWARD-leg prediction — the
    tensor no GT can reach — and `both` adds the closing frame. At C=1 the term is
    identically zero, so the step must not even record it."""
    from occl.finetune.cycle import oc_ccl_step, overlap_penalty

    x0, xu, m1, m2 = _pair()
    y = torch.cat([m1, m2])
    tr = FakeCycleTracker()
    off, _, _ = oc_ccl_step(tr, x0, xu, y)
    on, _, parts = oc_ccl_step(tr, x0, xu, y, xovl_weight=0.7)

    fwd = (y * 20.0 - 10.0) * 0.5
    expected_q = float(overlap_penalty(fwd))
    assert float(on) - float(off) == pytest.approx(0.7 * expected_q, rel=1e-5)
    assert parts["xovl"] == pytest.approx(expected_q, rel=1e-5)
    assert parts["xovl_where"] == "query"

    both, _, pboth = oc_ccl_step(tr, x0, xu, y, xovl_weight=0.7, xovl_where="both")
    expected_b = expected_q + float(overlap_penalty(fwd * 0.5))
    assert float(both) - float(off) == pytest.approx(0.7 * expected_b, rel=1e-5)
    assert pboth["xovl"] == pytest.approx(expected_b, rel=1e-5)

    off1, _, _ = oc_ccl_step(tr, x0, xu, m1)
    on1, _, p1 = oc_ccl_step(tr, x0, xu, m1, xovl_weight=0.7)
    assert float(on1) == float(off1) and "xovl" not in p1

    with pytest.raises(ValueError, match="xovl_where"):
        oc_ccl_step(tr, x0, xu, y, xovl_weight=0.7, xovl_where="everywhere")


def test_bshape_lands_on_the_closing_frame_with_the_cached_phi():
    from occl.finetune.bshape import signed_distance_stack
    from occl.finetune.cycle import oc_ccl_step

    x0, xu, m1, m2 = _pair()
    y = torch.cat([m1, m2])
    tr = FakeCycleTracker()
    off, _, _ = oc_ccl_step(tr, x0, xu, y)
    on, _, parts = oc_ccl_step(tr, x0, xu, y, bshape_weight=0.5)

    closing = (y * 20.0 - 10.0) * 0.25
    expected = float((torch.sigmoid(closing) * signed_distance_stack(y)).mean())
    assert float(on) - float(off) == pytest.approx(0.5 * expected, rel=1e-4, abs=1e-7)
    assert parts["bshape"] == pytest.approx(expected, rel=1e-4, abs=1e-7)


def test_margin_step_on_the_flip_symmetric_fake_costs_exactly_m():
    """A 180 flip only permutes pixels, so both legs' claimed confidences are equal and
    the hinge sits at exactly margin_m — which simultaneously proves the two memory
    chains are independent (a shared chain would change the rotated leg's prediction)."""
    from occl.finetune.cycle import margin_step

    x0, xu, m1, m2 = _pair()
    tr = _CountingTracker()
    loss, pred, parts = margin_step(tr, x0, xu, torch.cat([m1, m2]), margin_m=1.0)
    assert float(loss) == pytest.approx(1.0, abs=1e-6)
    assert parts["gap"] == pytest.approx(0.0, abs=1e-6)
    assert parts["margin_violated"] and not parts["inconsistent_won"]
    assert not parts["empty_claim_up"] and not parts["empty_claim_rot"]
    assert parts["n_objects"] == 2 and parts["margin_m"] == 1.0
    assert (tr.encodes, tr.tracks) == (3, 4)  # one extra encode, NO reverse chain
    assert pred.shape == (2, 1, 16, 16)


def test_ptmargin_loss_is_cycle_plus_weight_times_margin():
    from occl.finetune.cycle import margin_step, oc_ccl_step

    x0, xu, m1, m2 = _pair()
    y = torch.cat([m1, m2])
    cycle_only, _, _ = oc_ccl_step(FakeCycleTracker(), x0, xu, y)
    margin_only, _, _ = margin_step(FakeCycleTracker(), x0, xu, y, margin_m=1.0)
    tr = _CountingTracker()
    combo, _, parts = oc_ccl_step(tr, x0, xu, y, margin_weight=0.7, margin_m=1.0)
    assert float(combo) == pytest.approx(
        float(cycle_only) + 0.7 * float(margin_only), rel=1e-5)
    assert parts["margin_weight"] == 0.7 and parts["margin_m"] == 1.0
    assert {"conf_up", "conf_rot", "gap", "margin"} <= set(parts)
    assert (tr.encodes, tr.tracks) == (3, 6)  # 1.5x a plain cycle step


def test_margin_rotates_reference_and_mask_together_and_never_the_query():
    from occl.finetune.cycle import margin_step

    seen = {"encoded": [], "init_masks": [], "fwd_backbones": []}

    class _Spy(FakeCycleTracker):
        def encode_image(self, img):
            seen["encoded"].append(img)
            return super().encode_image(img)

        def track_with_mask(self, backbone, mask, is_init_frame=True,
                            output_dict=None, n_objects=1):
            if is_init_frame:
                seen["init_masks"].append(mask)
            else:
                seen["fwd_backbones"].append(id(backbone))
            return super().track_with_mask(backbone, mask, is_init_frame,
                                           output_dict, n_objects)

    x0, xu, m1, m2 = _pair()
    y = torch.cat([m1, m2])
    margin_step(_Spy(), x0, xu, y, margin_m=1.0)
    assert len(seen["encoded"]) == 3
    assert torch.equal(seen["encoded"][0], x0)
    assert torch.equal(seen["encoded"][1], xu)
    assert torch.equal(seen["encoded"][2], torch.flip(x0, dims=(-2, -1)))
    assert torch.equal(seen["init_masks"][0], y)
    assert torch.equal(seen["init_masks"][1], torch.flip(y, dims=(-2, -1)))
    # both forward legs consume the SAME upright query encode
    assert len(set(seen["fwd_backbones"])) == 1


def test_margin_gradient_report_conditions_on_hinge_activity():
    from occl.finetune.cycle import margin_gradient_report

    x0, xu, m1, m2 = _pair()
    y = torch.cat([m1, m2])
    active = margin_gradient_report(FakeCycleTracker(), x0, xu, y, margin_m=1.0)
    assert active["hinge_active"]
    assert active["grad_up_norm"] > 0 and active["grad_rot_norm"] > 0
    assert active["conf_up"] == pytest.approx(active["conf_rot"], abs=1e-5)

    idle = margin_gradient_report(FakeCycleTracker(), x0, xu, y, margin_m=0.0)
    assert not idle["hinge_active"]
    assert idle["grad_up_norm"] == 0.0 and idle["grad_rot_norm"] == 0.0


def test_gradient_report_signature_admits_no_s13_knobs():
    """The exclusion is load-bearing (see gradient_report's docstring): a future
    "helpful" mirror of xovl/margin into the report would hand the forward probe a
    direct gradient and silently break the s8a/s10ft hard-threshold-exactly-0.0
    predicates. Adding any knob parameter here must be a conscious, test-breaking act."""
    import inspect

    from occl.finetune.cycle import gradient_report

    assert list(inspect.signature(gradient_report).parameters) == [
        "tracker", "x0", "xu", "y0", "prompt_mode"]


def test_s13_margin_m_is_the_a_priori_constant():
    """Mirror of the S10 aug-magnitude pin: m = |SAM2_OUT_BIAS| / 10 = 1.0, its rule
    string recorded beside it — if S13 misses, the report names the gap; the constant
    does not move (the s4b discipline)."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "notebooks"
           / "run_stage.py").read_text()
    m = re.search(r"^S13_MARGIN_M\s*=\s*([0-9.]+)", src, re.M)
    assert m, "S13_MARGIN_M missing from notebooks/run_stage.py"
    assert float(m.group(1)) == abs(SAM2_OUT_BIAS) / 10
    assert re.search(r"^S13_MARGIN_RULE\s*=", src, re.M), "the rule string is gone"


# ── consecutive checkpoints with DIFFERENT scope sets (S14sam3 stack -> arch) ─


class _Sam3ish(nn.Module):
    """Scope-named toy: the three arch scopes plus a frozen backbone sibling."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.maskmem_backbone = nn.Sequential(nn.Linear(4, 4))
        self.transformer = nn.Sequential(nn.Linear(4, 4))
        self.sam_mask_decoder = nn.Sequential(nn.Linear(4, 4))
        self.backbone = nn.Sequential(nn.Linear(4, 4))

    def forward(self, x):
        return self.sam_mask_decoder(self.transformer(self.maskmem_backbone(x)))


def _ckpt(tmp_path, name, scopes, seed):
    from occl.finetune.train import save_lora

    m = _Sam3ish()
    apply_lora(m, r=4, alpha=8.0, scopes=scopes)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, mod in lora_modules(m):
            mod.lora_B.copy_(torch.randn(mod.lora_B.shape, generator=g))
    out = tmp_path / f"{name}.pt"
    save_lora(m, out, meta={"method": "lora", "family": "sam3",
                            "lora": {"r": 4, "alpha": 8.0,
                                     "scopes": list(scopes)}},
              method="lora")
    return out


def test_load_finetune_rebuilds_adapters_when_the_scope_set_changes(tmp_path):
    """The S14sam3 TTA fan runs stack -> arch (a strict scope SUPERSET) in one
    worker. The old reuse-if-wrapped branch left the second checkpoint's
    transformer keys unmatched (strict guard kills the worker mid-fan, after the
    base and stack arms already paid); re-applying on top would nest adapters and
    shift every path by `.orig`. load_finetune must strip and rebuild from the
    checkpoint's OWN scopes — both orders, zero mismatches, and the result is
    bit-identical to loading onto a never-wrapped model."""
    from occl.finetune.lora import remove_lora
    from occl.finetune.train import load_finetune

    stack = ("maskmem_backbone", "sam_mask_decoder")
    arch = ("maskmem_backbone", "transformer", "sam_mask_decoder")
    pt_stack = _ckpt(tmp_path, "stack", stack, seed=1)
    pt_arch = _ckpt(tmp_path, "arch", arch, seed=2)
    x = torch.randn(2, 4, generator=torch.Generator().manual_seed(3))

    seq = _Sam3ish()
    rep1 = load_finetune(seq, pt_stack)
    assert rep1["mismatches"] == 0 and rep1["delta_norm"] > 0
    rep2 = load_finetune(seq, pt_arch)           # superset AFTER subset
    assert rep2["mismatches"] == 0 and rep2["delta_norm"] > 0
    fresh = _Sam3ish()
    assert load_finetune(fresh, pt_arch)["mismatches"] == 0
    assert torch.equal(seq(x), fresh(x)), \
        "stack->arch differs from a clean arch load"

    rep3 = load_finetune(seq, pt_stack)          # and subset AFTER superset
    assert rep3["mismatches"] == 0
    fresh2 = _Sam3ish()
    load_finetune(fresh2, pt_stack)
    assert torch.equal(seq(x), fresh2(x))

    # remove_lora is the exact inverse: back to the pristine base, bit for bit
    n = remove_lora(seq)
    assert n == 2                       # the LAST load was the 2-module stack set
    base = _Sam3ish()
    assert torch.equal(seq(x), base(x)), "remove_lora left residue"
    assert not lora_modules(seq)
