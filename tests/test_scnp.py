"""SCNP — that it implements the paper, and that it does so on OUR tensors.

The claim being tested is not "the pooling is correct" but the paper's actual mechanism:
that isolated false positives and thin breakages collect gradient once per neighbour that
selects them, while well-predicted interiors collect none. A test that only checked the
min/max arithmetic would pass on an implementation that penalises nothing useful.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from occl.finetune.scnp import (  # noqa: E402
    MASK_CONST, neighbor_gradient_multiplicity, scnp_logits, scnp_loss,
)


def _bce(logits, target):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def test_width_one_is_the_identity():
    """The anchor: with no neighbourhood there is nothing to substitute."""
    z = torch.randn(2, 3, 8, 8)
    y = (torch.rand(2, 3, 8, 8) > 0.5).float()
    assert torch.equal(scnp_logits(z, y, width=1), z)


def test_transform_can_only_make_the_loss_harder():
    """Foreground logits fall, background logits rise — SCNP is a pessimisation.

    This is the property that makes it safe to add: it cannot flatter the model. If a
    foreground pixel's substituted logit ever exceeded its own, the transform would be
    *rewarding* a pixel for a well-predicted neighbour, which is the opposite of the paper.
    """
    z = torch.randn(2, 3, 16, 16)
    y = (torch.rand(2, 3, 16, 16) > 0.5).float()
    zt = scnp_logits(z, y, width=3)
    fg, bg = y.bool(), ~y.bool()
    assert torch.all(zt[fg] <= z[fg] + 1e-6), "a foreground logit was raised"
    assert torch.all(zt[bg] >= z[bg] - 1e-6), "a background logit was lowered"
    assert _bce(zt, y) >= _bce(z, y), "the transform reduced the loss"


def test_a_pixel_is_its_own_neighbour_so_no_window_is_empty():
    """A lone foreground pixel must not pick up the masking constant.

    If the window excluded the centre, an isolated foreground pixel would have no
    same-class neighbour, `MinPool` would return +9999, sigmoid would saturate and its loss
    would be ~0 — the transform would silently *exempt* exactly the pixels it exists to
    punish.
    """
    z = torch.full((1, 1, 5, 5), -3.0)
    y = torch.zeros(1, 1, 5, 5)
    y[0, 0, 2, 2] = 1.0                       # one isolated foreground pixel
    zt = scnp_logits(z, y, width=3)
    assert zt[0, 0, 2, 2].item() == pytest.approx(-3.0), "isolated fg took the mask constant"
    assert zt.abs().max().item() < MASK_CONST / 2, "masking constant leaked into the output"


def test_isolated_false_positive_collects_the_most_gradient():
    """The paper's mechanism, on a constructed case.

    A single high-logit pixel inside a background region is the strongest background
    neighbour of every pixel in its 3x3 window, so it is selected 9 times, while a
    well-predicted pixel is selected by nobody. That ratio IS the method; if it collapses,
    SCNP is doing nothing plain BCE does not.

    The background is jittered because `max_pool2d` routes the whole gradient to ONE index
    on an exact tie. With a perfectly flat background every window has nine tied maxima, an
    arbitrary one wins, and the corner reads 4 rather than 0 — an artifact of the constructed
    input, not of the method. Real logits are never exactly tied, so the jitter is the more
    faithful test, but the flat case is worth knowing about before it is met in a diagnostic.
    """
    torch.manual_seed(0)
    y = torch.zeros(1, 1, 9, 9)
    z = -5.0 + 0.01 * torch.randn(1, 1, 9, 9)   # confident, correct background
    z[0, 0, 4, 4] = +5.0                        # one isolated false positive
    mult = neighbor_gradient_multiplicity(z, y, width=3)
    assert mult[0, 0, 4, 4].item() == pytest.approx(9.0), "FP not selected by its 9 neighbours"
    assert mult[0, 0, 4, 4].item() == mult.max().item(), "FP is not the most-penalised pixel"
    assert mult[0, 0, 0, 0].item() == pytest.approx(0.0), "correct interior collected gradient"


def test_gradient_reaches_only_the_selected_neighbour():
    """`d z~_ki / d z_kj` is the indicator that j was the extremum — not a smooth average.

    Consequence worth pinning: in any one step most pixels get no gradient at all. That is
    intended (the paper: "gradients propagate selectively"), but it is also why a loss curve
    under SCNP can look stalled when it is not.

    Counts here are per-position and bounded by the window, NOT by the tensor: on a 3-wide
    row the leftmost position's window spans cols -1..1 and cannot reach col 2. So the
    maximum at col 2 is selected twice, not three times.
    """
    z = torch.tensor([[[[0.0, 1.0, 2.0]]]], requires_grad=True)   # 1x1x1x3
    y = torch.zeros(1, 1, 1, 3)                                   # all background -> MaxPool
    scnp_logits(z, y, width=3).sum().backward()
    assert z.grad.flatten().tolist() == [0.0, 1.0, 2.0]


def test_joint_form_keeps_the_untransformed_term():
    """`joint=True` must be exactly L(Z~) + L(Z), not a reweighting of one of them."""
    z = torch.randn(1, 1, 8, 8)
    y = (torch.rand(1, 1, 8, 8) > 0.5).float()
    plain = scnp_loss(_bce, z, y, width=3, joint=False)
    joint = scnp_loss(_bce, z, y, width=3, joint=True)
    assert joint.item() == pytest.approx((plain + _bce(z, y)).item(), rel=1e-6)


def test_rejects_inputs_that_would_silently_misbehave():
    z, y = torch.randn(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError):
        scnp_logits(z, y, width=2)                       # even window has no centre
    with pytest.raises(ValueError):
        scnp_logits(z, torch.zeros(1, 1, 5, 5), width=3)  # shape mismatch


def test_no_nan_when_a_channel_is_entirely_one_class():
    """`k = inf` in the paper; a finite constant is required for this case.

    An all-background channel makes `Z * Y` all-zero and `k * (1 - Y)` all-k. With k = inf
    the foreground branch is inf everywhere and `inf * 0` in the recombination is NaN. Beetle
    supports routinely lack a trait, so all-background channels are the normal case, not an
    edge case.
    """
    z = torch.randn(1, 2, 8, 8)
    y = torch.zeros(1, 2, 8, 8)
    y[0, 0] = 1.0                                        # ch0 all fg, ch1 all bg
    out = scnp_logits(z, y, width=3)
    assert torch.isfinite(out).all(), "non-finite output on a single-class channel"


def test_cost_is_negligible_at_our_tensor_size():
    """Two pooling ops on the closing mask. If this is ever slow, something is wrong."""
    import time

    z = torch.randn(1, 1, 1024, 1024)
    y = (torch.rand(1, 1, 1024, 1024) > 0.5).float()
    scnp_logits(z, y)                                    # warm
    t0 = time.perf_counter()
    for _ in range(5):
        scnp_logits(z, y)
    per = (time.perf_counter() - t0) / 5
    assert per < 0.5, f"{per * 1000:.0f} ms per call on a 1024^2 channel"


# ── full fine-tune save/load, the S8.5 path ──────────────────────────────────

def _stub_model():
    """A model shaped like SAM 2 where it matters: an `image_encoder` and other modules."""
    from torch import nn

    m = nn.Module()
    m.image_encoder = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    m.memory_encoder = nn.Linear(4, 4)
    m.sam_mask_decoder = nn.Linear(4, 4)
    return m


def test_full_finetune_trains_everything_except_the_image_encoder():
    """SST's actual method — `freeze_image_encoder`, the rest trains."""
    from occl.finetune.train import freeze_except_image_encoder, trainable_state_dict

    m = _stub_model()
    counts = freeze_except_image_encoder(m)
    assert not any(p.requires_grad for p in m.image_encoder.parameters())
    assert all(p.requires_grad for p in m.memory_encoder.parameters())
    sd = trainable_state_dict(m)
    assert not any(k.startswith("image_encoder") for k in sd), "encoder leaked into the save"
    assert counts["trainable"] < counts["total"]


def test_full_finetune_round_trips_and_reports_zero_mismatches():
    from occl.finetune.train import (freeze_except_image_encoder,
                                     load_trainable_state_dict, trainable_state_dict)

    src = _stub_model()
    freeze_except_image_encoder(src)
    with torch.no_grad():
        for p in src.parameters():
            if p.requires_grad:
                p.add_(1.0)
    sd = trainable_state_dict(src)

    dst = _stub_model()
    freeze_except_image_encoder(dst)
    assert load_trainable_state_dict(dst, sd) == 0, "strict load reported mismatches"
    for k, v in sd.items():
        assert torch.equal(dict(dst.named_parameters())[k].detach(), v)


def test_restore_base_is_what_reset_lora_cannot_do():
    """`run_arms` scores group-outer/arm-inner, so the base arm runs after a trained one.

    With LoRA the fix is to zero the adapters. With a full fine-tune the base weights were
    overwritten, so there is nothing to zero — they have to be put back. Getting this wrong
    would score every base arm after the first with the fine-tuned weights and file it under
    `:base`, which is the reference the whole delta is measured against.
    """
    from occl.finetune.train import (freeze_except_image_encoder, full_delta_norm,
                                     restore_base)

    m = _stub_model()
    freeze_except_image_encoder(m)
    theta0 = {n: p.detach().cpu().clone()
              for n, p in m.named_parameters() if p.requires_grad}
    assert full_delta_norm(m, theta0) == 0.0
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.requires_grad:
                p.add_(0.5)
    assert full_delta_norm(m, theta0) > 0.0
    assert restore_base(m, theta0) == 0.0, "base arm would run with fine-tuned weights"


def test_loader_reads_the_method_from_the_checkpoint_not_the_flag_name():
    """A full checkpoint pushed through the LoRA path mismatches on every key."""
    import tempfile
    from pathlib import Path

    from occl.finetune.train import freeze_except_image_encoder, load_finetune, save_lora

    src = _stub_model()
    freeze_except_image_encoder(src)
    with torch.no_grad():
        for p in src.parameters():
            if p.requires_grad:
                p.add_(1.0)
    path = Path(tempfile.mkdtemp()) / "ft.pt"
    save_lora(src, path, meta={"epochs": 10}, method="full")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["meta"]["method"] == "full" and "full" in blob and "lora" not in blob

    dst = _stub_model()
    freeze_except_image_encoder(dst)
    rep = load_finetune(dst, path)
    assert rep["method"] == "full"
    assert rep["mismatches"] == 0
    assert rep["delta_norm"] > 0.0, "loading a trained checkpoint left the weights at base"
    assert rep["base_state"] is not None, "no snapshot to restore the base arm from"


def test_train_lora_binds_every_name_on_both_method_branches():
    """A name assigned in only one branch of `if method == "full"` and read after it.

    `n_total` was set only in the LoRA branch and read by the shared summary line, so the
    very first full fine-tune died with UnboundLocalError **after** loading 898 MB of
    checkpoint onto the GPU. Third instance of this shape in the project — `_t` in `fanout`
    and the `meta` rebinding in `run_eval` were the others — so it is worth a structural
    check rather than another fix.

    Compares names bound in each branch against names read in the statements that follow.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "src" / "occl" / "finetune" / "train.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "train_lora")

    COMP = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def bound(stmts):
        """Names STORED at function scope. Comprehensions have their own scope in Python 3,
        so their loop variables never leak and must not be counted."""
        out = set()
        for st in stmts:
            stack = [st]
            while stack:
                node = stack.pop()
                if isinstance(node, COMP):
                    continue                      # separate scope; nothing escapes
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    out.add(node.id)
                stack.extend(ast.iter_child_nodes(node))
        return out

    for i, st in enumerate(fn.body):
        if not (isinstance(st, ast.If) and "method" in ast.unparse(st.test)):
            continue
        # Assigned on exactly one side AND not already bound before the branch — a
        # pre-binding (`theta0 = None`) makes one-sided assignment perfectly safe.
        only_one = (bound(st.body) ^ bound(st.orelse)) - bound(fn.body[:i]) - {
            a.arg for a in fn.args.args + fn.args.kwonlyargs}
        after = set()
        for later in fn.body[i + 1:]:
            for n in ast.walk(later):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    after.add(n.id)
        leaked = sorted(only_one & after)
        assert not leaked, (
            f"{leaked} bound on only one branch of `if {ast.unparse(st.test)}` but read "
            "afterwards — the other method would raise UnboundLocalError at runtime")
        return
    raise AssertionError("train_lora no longer branches on `method`")
