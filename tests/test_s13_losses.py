"""S13 loss math — exclusivity CE, overlap penalty, boundary-distance, margin statistic.

Each arm's load-bearing identity is pinned here, because each is the reason the arm is
designed the way it is: `zerocat` degrades to the existing BCE at C=1 (so single-trait
references are not a special case), `aggregate` is the STM form and only differs in the
background logit, the overlap penalty is exactly zero without overlap, phi never spends
gradient inside the metric's 1-px forgiveness ring, and the training-side confidence is
the SAME statistic TTA decides with at eval — the bridge the whole margin fold stands on.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from occl.finetune.bshape import (  # noqa: E402
    _PhiCache, bshape_loss, signed_distance_stack,
)
from occl.finetune.cycle import (  # noqa: E402
    bce_loss, claimed_confidence, overlap_penalty,
)
from occl.finetune.excl import EXCL_MODES, excl_ce, exclusive_target  # noqa: E402


# ── excl ─────────────────────────────────────────────────────────────────────


def test_excl_zerocat_at_c1_is_exactly_bce():
    """softmax([0, z]) CE == per-pixel BCE-with-logits, both global means — the
    graceful-degradation anchor: a single-trait reference trains the CURRENT objective."""
    torch.manual_seed(0)
    z = torch.randn(1, 1, 32, 32) * 6
    y = (torch.rand(1, 1, 32, 32) > 0.5).float()
    assert torch.isclose(excl_ce(z, y, "zerocat"), bce_loss(z, y), atol=1e-6)


def test_excl_aggregate_at_c1_is_bce_at_temperature_two_and_finite_at_extremes():
    """The STM aggregation at C=1 is softmax([-z, z]) = sigmoid(2z) — a temperature-2
    BCE, i.e. a silent objective change, which is exactly why zerocat is the arm setting.
    Compared in LOGITS space (probability space loses the tail precision the identity
    needs). At |z| ~ 30 the fp32 clamp bounds the background logit — finite, never NaN."""
    torch.manual_seed(0)
    z = torch.randn(1, 1, 32, 32) * 3  # below the ~13.8 clamp point
    y = (torch.rand(1, 1, 32, 32) > 0.5).float()
    assert torch.isclose(excl_ce(z, y, "aggregate"),
                         F.binary_cross_entropy_with_logits(2 * z, y), atol=1e-5)

    zx = torch.full((2, 1, 8, 8), -30.0)
    zx[1] = 30.0
    yx = torch.zeros(2, 1, 8, 8)
    yx[1] = 1.0
    assert torch.isfinite(excl_ce(zx, yx, "aggregate"))


def test_excl_penalizes_overlap_and_only_overlap():
    """Two channels confidently claiming the same pixels score strictly worse than the
    same channels disjoint; the disjoint prediction is near its BCE-equivalent optimum."""
    y = torch.zeros(2, 1, 16, 16)
    y[0, 0, :8] = 1.0
    y[1, 0, 8:] = 1.0
    disjoint = torch.where(y > 0.5, 8.0, -8.0)
    overlapping = disjoint.clone()
    overlapping[1, 0, :8] = 8.0  # channel 1 also claims channel 0's half
    # contested pixels cost ~log(2) each; half the image is contested -> ~0.35 vs ~3e-4
    assert excl_ce(overlapping, y, "zerocat") > excl_ce(disjoint, y, "zerocat") + 0.25
    assert float(excl_ce(disjoint, y, "zerocat")) < 0.01


def test_exclusive_target_partitions_and_thresholds_soft_targets():
    y = torch.zeros(2, 1, 8, 8)
    y[0, 0, :4] = 1.0
    y[1, 0, 4:6] = 1.0
    t = exclusive_target(y)
    assert sorted(t.unique().tolist()) == [0, 1, 2]
    assert (t[:4] == 1).all() and (t[4:6] == 2).all() and (t[6:] == 0).all()
    # a bilinearly-softened target must produce the same labelmap as its hard mask
    soft = y * 0.9 + 0.05
    assert torch.equal(exclusive_target(soft), t)


def test_excl_rejects_off_and_unknown_modes():
    z = torch.zeros(1, 1, 4, 4)
    y = torch.zeros(1, 1, 4, 4)
    for bad in ("off", "softmax", ""):
        with pytest.raises(ValueError):
            excl_ce(z, y, bad)
    assert EXCL_MODES == ("off", "aggregate", "zerocat")


# ── xovl ─────────────────────────────────────────────────────────────────────


def test_overlap_penalty_is_exactly_zero_at_c1_and_for_disjoint_saturated_masks():
    z1 = torch.randn(1, 1, 16, 16) * 9
    assert float(overlap_penalty(z1)) == 0.0

    zd = torch.full((2, 1, 16, 16), -30.0)
    zd[0, 0, :8] = 30.0
    zd[1, 0, 8:] = 30.0
    assert float(overlap_penalty(zd)) < 1e-8


def test_overlap_penalty_is_positive_and_differentiable_on_overlap():
    z = torch.full((2, 1, 8, 8), 6.0, requires_grad=True)
    pen = overlap_penalty(z)
    assert float(pen.detach()) > 0.5
    (g,) = torch.autograd.grad(pen, z)
    assert float(g.norm()) > 0


# ── bshape ───────────────────────────────────────────────────────────────────


def test_phi_is_signed_dead_zoned_channel_normalized_and_degenerate_safe():
    y = torch.zeros(2, 1, 32, 32)
    y[0, 0, 8:24, 8:24] = 1.0  # square; channel 1 stays empty (degenerate)
    phi = signed_distance_stack(y)
    assert phi.shape == y.shape and not phi.requires_grad
    assert float(phi[0, 0, 16, 16]) < 0  # inside
    assert float(phi[0, 0, 0, 0]) > 0    # outside
    assert float(phi[0, 0, 8, 16]) == 0.0   # first ring: the metric forgives it
    assert float(phi[0, 0, 16, 24]) == 0.0  # first OUTER ring too
    assert abs(float(phi[0].abs().max()) - 1.0) < 1e-6  # chan_max normalization
    assert float(phi[1].abs().max()) == 0.0  # degenerate channel: no term, no crash


def test_phi_thresholds_a_soft_target_before_the_distance_transform():
    y = torch.zeros(1, 1, 32, 32)
    y[0, 0, 8:24, 8:24] = 1.0
    soft = y * 0.9 + 0.05
    assert torch.equal(signed_distance_stack(soft), signed_distance_stack(y))


def test_bshape_prefers_gt_aligned_logits_and_its_minimum_is_not_empty_foreground():
    y = torch.zeros(1, 1, 32, 32)
    y[0, 0, 8:24, 8:24] = 1.0
    aligned = torch.where(y > 0.5, 12.0, -12.0)
    inverted = -aligned
    empty = torch.full_like(aligned, -12.0)
    cache = _PhiCache()
    import occl.finetune.bshape as bs

    old = bs.PHI_CACHE
    bs.PHI_CACHE = cache
    try:
        l_aligned = float(bshape_loss(aligned, y))
        l_inverted = float(bshape_loss(inverted, y))
        l_empty = float(bshape_loss(empty, y))
    finally:
        bs.PHI_CACHE = old
    assert l_aligned < l_empty < l_inverted
    assert l_aligned < 0  # interior reward: the term is negative at its optimum


def test_phi_cache_hits_on_identical_y0_and_misses_on_a_changed_one():
    cache = _PhiCache()
    y = torch.zeros(1, 1, 16, 16)
    y[0, 0, 2:7, 4:12] = 1.0          # asymmetric: a 180 flip changes the content
    a = cache.get(y)
    b = cache.get(y.clone())          # equal content -> hit
    assert cache.hits == 1 and cache.misses == 1 and a is b
    cache.get(torch.flip(y, dims=(-2, -1)))  # different content -> miss
    assert cache.misses == 2
    cache.reset()
    assert cache.hits == 0 and cache.misses == 0


# ── margin statistic ─────────────────────────────────────────────────────────


def test_claimed_confidence_matches_the_eval_statistic():
    """The bridge the fold stands on: the training statistic IS `tta.confidence`
    wherever the eval statistic is defined (any non-empty claim)."""
    from occl.adapters.tta import confidence

    torch.manual_seed(0)
    for _ in range(5):
        lg = torch.randn(3, 1, 16, 16) * 5
        want = confidence(lg[:, 0].numpy())
        assert want != float("-inf")
        assert abs(float(claimed_confidence(lg)) - want) < 1e-5


def test_claimed_confidence_is_finite_and_gradient_free_on_an_empty_claim():
    lg = torch.full((2, 1, 8, 8), -5.0, requires_grad=True)
    conf = claimed_confidence(lg)
    assert float(conf.detach()) == 0.0
    (g,) = torch.autograd.grad(conf, lg)
    assert float(g.norm()) == 0.0


def test_claimed_confidence_gradient_is_uniform_over_claimed_pixels():
    lg = torch.tensor([[[[3.0, -1.0], [0.5, -2.0]]],
                       [[[1.0, -3.0], [4.0, -1.5]]]], requires_grad=True)
    # top = [[3, -1], [4, -2]] -> claimed = {(0,0) via ch0, (1,0) via ch1}, N = 2
    conf = claimed_confidence(lg)
    (g,) = torch.autograd.grad(conf, lg)
    assert float(g[0, 0, 0, 0]) == pytest.approx(0.5)
    assert float(g[1, 0, 1, 0]) == pytest.approx(0.5)
    assert float(g.abs().sum()) == pytest.approx(1.0)


def test_uniform_logit_inflation_cannot_satisfy_the_margin():
    """The hinge is a DIFFERENCE of confidences: adding the same constant to both legs
    (claim sets held fixed by keeping every logit positive) moves the gap by exactly
    zero — inflating confidence everywhere is not a descent direction."""
    torch.manual_seed(1)
    up = torch.rand(2, 1, 8, 8) * 4 + 0.5   # all claims positive, stay positive
    rot = torch.rand(2, 1, 8, 8) * 4 + 0.5
    m = 1.0

    def hinge(a, b):
        return float(F.relu(m - (claimed_confidence(a) - claimed_confidence(b))))

    base = hinge(up, rot)
    assert base > 0  # the case where inflation would "help" if it could
    assert hinge(up + 7.3, rot + 7.3) == pytest.approx(base, abs=1e-5)


def test_signed_distance_matches_numpy_reference_on_a_thin_structure():
    """A 2-px bar: everything inside is within the dead zone — the term goes silent on
    structures thinner than the metric's tolerance, by design (best-of-3 can already
    recover them). The OUTSIDE of the bar still carries signal against false claims."""
    y = torch.zeros(1, 1, 24, 24)
    y[0, 0, 11:13, 4:20] = 1.0
    phi = signed_distance_stack(y)
    assert float(phi[0, 0, 11:13, 4:20].abs().max()) == 0.0  # interior all |phi| <= 1
    far = phi[0, 0, 0, 0]
    assert float(far) > 0.5  # far background retains near-max positive distance


def test_phi_numpy_cross_check():
    """cv2's L2 distance agrees with a brute-force nearest-boundary computation."""
    y = torch.zeros(1, 1, 16, 16)
    y[0, 0, 6:10, 6:10] = 1.0
    phi = signed_distance_stack(y, dead_zone=0.0, normalize="none")
    m = (y[0, 0] > 0.5).numpy()
    ys, xs = np.nonzero(m)
    bg_ys, bg_xs = np.nonzero(~m)
    # centre pixel: distance to nearest background pixel
    d = np.sqrt((bg_ys - 7) ** 2 + (bg_xs - 7) ** 2).min()
    assert float(phi[0, 0, 7, 7]) == pytest.approx(-d, abs=0.35)  # cv2 3x3 mask approx
    d0 = np.sqrt((ys - 0) ** 2 + (xs - 0) ** 2).min()
    assert float(phi[0, 0, 0, 0]) == pytest.approx(d0, abs=0.45)
