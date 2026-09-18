"""Gate S0 — the seven decided edge cases of the paper-faithful metric, plus guards.

A metric bug is the single most likely silent failure in this project: it produces a
plausible number and nothing crashes. These run in ~1s with no GPU and no data.
"""

from __future__ import annotations

import numpy as np
import pytest

from occl.metrics import (
    BACKGROUND,
    aggregate_run,
    binary_iou,
    logits_to_labelmap,
    per_image_miou,
    score_episode,
)

# Two traits (1, 2) plus background. Trait 1 top-left, trait 2 right column-pair.
GT = np.array(
    [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 0, 0],
    ],
    dtype=np.int32,
)


def logits_for(labelmap: np.ndarray, class_ids, hi: float = 5.0, lo: float = -5.0):
    """Confident logits that reproduce ``labelmap`` exactly for the given classes."""
    return np.stack([np.where(labelmap == c, hi, lo) for c in class_ids]).astype(np.float32)


# ── the seven gate cases ─────────────────────────────────────────────────────


def test_1_perfect_prediction_scores_one():
    pred = logits_to_labelmap(logits_for(GT, [1, 2]), [1, 2])
    miou, per = per_image_miou(pred, GT)
    assert miou == pytest.approx(1.0)
    assert per == {1: pytest.approx(1.0), 2: pytest.approx(1.0)}


def test_2_disjoint_prediction_scores_zero():
    # Channel for trait 1 fires on trait 2's pixels and vice versa.
    swapped = np.stack(
        [np.where(GT == 2, 5.0, -5.0), np.where(GT == 1, 5.0, -5.0)]
    ).astype(np.float32)
    miou, per = per_image_miou(logits_to_labelmap(swapped, [1, 2]), GT)
    assert miou == pytest.approx(0.0)
    assert per == {1: pytest.approx(0.0), 2: pytest.approx(0.0)}


def test_3_trait_predicted_but_absent_from_gt_contributes_no_term():
    # Class 3 is prompted and fires on background pixels. GT has only {1, 2}.
    logits = logits_for(np.where(GT == 0, 3, GT), [1, 2, 3])
    miou, per = per_image_miou(logits_to_labelmap(logits, [1, 2, 3]), GT)
    assert set(per) == {1, 2}, "a trait absent from GT must not appear in the average"
    assert miou == pytest.approx(1.0)


def test_4_trait_in_gt_but_never_prompted_scores_zero_and_is_counted():
    # Only trait 1 is in the support, so trait 2 can never be predicted.
    miou, per = per_image_miou(logits_to_labelmap(logits_for(GT, [1]), [1]), GT)
    assert set(per) == {1, 2}
    assert per[1] == pytest.approx(1.0)
    assert per[2] == pytest.approx(0.0)
    assert miou == pytest.approx(0.5), "unpredictable GT traits must drag the mean down"


def test_5_all_background_gt_returns_none():
    empty = np.zeros((4, 4), dtype=np.int32)
    miou, per = per_image_miou(np.zeros((4, 4), dtype=np.int32), empty)
    assert miou is None, "caller must skip the image, not average in a NaN"
    assert per == {}


def test_6_argmax_ties_go_to_the_lowest_channel_index():
    tied = np.full((2, 4, 4), 5.0, dtype=np.float32)
    lab = logits_to_labelmap(tied, [7, 9])
    assert np.all(lab == 7), "tie-break must be deterministic given class_ids order"


def test_7_non_positive_logits_are_background():
    lab = logits_to_labelmap(np.full((2, 4, 4), -1.0, dtype=np.float32), [1, 2])
    assert np.all(lab == BACKGROUND)
    miou, per = per_image_miou(lab, GT)
    assert miou == pytest.approx(0.0)
    assert set(per) == {1, 2}


# ── guards on the surrounding machinery ──────────────────────────────────────


def test_score_episode_resizes_logits_down_not_labels_up():
    # Logits at 8x8, GT at 4x4. gt_native must resize the logits, leaving GT untouched.
    big = np.kron(GT, np.ones((2, 2), dtype=np.int32))
    out = score_episode(logits_for(big, [1, 2]), [1, 2], GT, iou_space="gt_native")
    assert out["miou"] == pytest.approx(1.0)
    assert out["n_traits"] == 2


def test_score_episode_model_space_resizes_gt_up():
    big = np.kron(GT, np.ones((2, 2), dtype=np.int32))
    out = score_episode(logits_for(big, [1, 2]), [1, 2], GT, iou_space="model")
    assert out["miou"] == pytest.approx(1.0)


def test_mismatched_shapes_raise_rather_than_broadcast():
    with pytest.raises(ValueError):
        per_image_miou(np.zeros((8, 8), dtype=np.int32), GT)


def test_class_ids_length_must_match_channel_count():
    with pytest.raises(ValueError):
        logits_to_labelmap(np.zeros((2, 4, 4), dtype=np.float32), [1, 2, 3])


def test_binary_iou_merges_traits_and_is_not_the_paper_metric():
    # Label every foreground pixel trait 2. Binary IoU is a perfect 1.0 because the
    # silhouette is exact, but trait 1 (4px) is entirely lost and trait 2 (6px) is
    # over-claimed to 10px: 0.0 and 6/10, mean 0.3. A 0.70 gap between the two metrics
    # on the same prediction — this is exactly why the repo's number is not the paper's.
    merged = np.where(GT > 0, 2, 0).astype(np.int32)
    assert binary_iou(merged, GT) == pytest.approx(1.0)
    miou, per = per_image_miou(merged, GT)
    assert per[1] == pytest.approx(0.0)
    assert per[2] == pytest.approx(0.6)
    assert miou == pytest.approx(0.3)


def test_aggregate_run_is_macro_over_images_not_pooled():
    # Two images: one scoring 1.0 with 10 traits, one scoring 0.0 with 2 traits.
    # Macro over images = 0.5. A pooled/micro average would be weighted toward the
    # 10-trait image and give ~0.83.
    scores = [
        {"miou": 1.0, "per_trait": {i: 1.0 for i in range(1, 11)}, "iou_binary": 1.0},
        {"miou": 0.0, "per_trait": {1: 0.0, 2: 0.0}, "iou_binary": 0.0},
    ]
    agg = aggregate_run(scores)
    assert agg["miou"] == pytest.approx(0.5)
    assert agg["n_images"] == 2 and agg["n_skipped"] == 0


def test_aggregate_run_skips_none_images():
    scores = [
        {"miou": 0.8, "per_trait": {1: 0.8}, "iou_binary": 0.9},
        {"miou": None, "per_trait": {}, "iou_binary": 0.0},
    ]
    agg = aggregate_run(scores)
    assert agg["miou"] == pytest.approx(0.8)
    assert agg["n_images"] == 1 and agg["n_skipped"] == 1


# ── S5-DIAG: assignment-rule variants ────────────────────────────────────────


def test_variants_agree_when_predictions_are_disjoint_and_confident():
    """With one clean confident channel per trait, every rule must give the same answer.

    If they disagree here the variants are measuring something other than the overlap
    rule, and the S5-DIAG comparison would be meaningless.
    """
    from occl.metrics import ASSIGNMENT_RULES, assignment_variants

    gt = np.zeros((16, 16), dtype=np.int32)
    gt[:8, :] = 1
    gt[8:, :] = 2
    logits = np.full((2, 16, 16), -5.0, dtype=np.float32)
    logits[0, :8, :] = 5.0
    logits[1, 8:, :] = 5.0
    v = assignment_variants(logits, (1, 2), gt)
    assert set(v) == set(ASSIGNMENT_RULES)
    for rule in ASSIGNMENT_RULES:
        assert v[rule] == pytest.approx(1.0), rule


def test_independent_beats_argmax_when_channels_overlap():
    """The whole point of H1: overlapping predictions are penalised by argmax only.

    Both channels claim the same region. argmax awards it to exactly one, so the loser
    scores 0; independent binarisation lets both keep it.
    """
    from occl.metrics import assignment_variants

    gt = np.zeros((16, 16), dtype=np.int32)
    gt[:8, :] = 1
    gt[8:, :] = 2
    logits = np.full((2, 16, 16), 5.0, dtype=np.float32)  # both claim everything
    logits[0] += 0.1                                      # channel 0 wins every pixel
    v = assignment_variants(logits, (1, 2), gt)
    assert v["independent"] > v["argmax"]
    assert v["argmax"] == pytest.approx(0.25)   # trait1 IoU 0.5, trait2 IoU 0.0
    assert v["independent"] == pytest.approx(0.5)


def test_oracle_detects_a_pure_label_permutation():
    """oracle >> argmax is the signature that separates mislabelling from not segmenting.

    Here the regions are segmented perfectly but the two channels are swapped: argmax
    scores 0, while oracle recovers 1.0 because some channel matches each GT trait.
    """
    from occl.metrics import assignment_variants

    gt = np.zeros((16, 16), dtype=np.int32)
    gt[:8, :] = 1
    gt[8:, :] = 2
    logits = np.full((2, 16, 16), -5.0, dtype=np.float32)
    logits[0, 8:, :] = 5.0     # channel for trait 1 covers trait 2's region
    logits[1, :8, :] = 5.0
    v = assignment_variants(logits, (1, 2), gt)
    assert v["argmax"] == pytest.approx(0.0)
    assert v["oracle"] == pytest.approx(1.0)


def test_variants_return_none_for_all_background_gt():
    from occl.metrics import ASSIGNMENT_RULES, assignment_variants

    v = assignment_variants(np.zeros((2, 8, 8), dtype=np.float32), (1, 2),
                            np.zeros((8, 8), dtype=np.int32))
    assert all(v[r] is None for r in ASSIGNMENT_RULES)


def test_trait_absent_from_support_scores_zero_under_every_rule():
    """Matches `per_image_miou` and SST's `per_trait_iou`: unpredictable traits count 0."""
    from occl.metrics import assignment_variants

    gt = np.zeros((16, 16), dtype=np.int32)
    gt[:8, :] = 1
    gt[8:, :] = 2
    logits = np.full((1, 16, 16), -5.0, dtype=np.float32)
    logits[0, :8, :] = 5.0
    v = assignment_variants(logits, (1,), gt)          # trait 2 has no channel
    assert v["argmax"] == pytest.approx(0.5)           # 1.0 and 0.0
    assert v["independent"] == pytest.approx(0.5)


# ── the unpublished best-of-3 boundary tolerance (author correspondence) ─────


def _logits_from_mask(mask, lo=-10.0, hi=10.0):
    import numpy as np
    return np.where(mask, hi, lo)[None].astype(np.float32)


def test_best3_forgives_a_one_pixel_thin_prediction():
    """The detail's whole claim: a prediction one annotation-pixel thin scores below 1
    plain and exactly 1.0 under best-of-3, because dilation reconstructs it."""
    import numpy as np

    from occl.metrics import boundary_variants, score_episode

    gt = np.zeros((20, 20), dtype=np.int32)
    gt[5:15, 5:15] = 7                                   # 10x10 trait
    thin = np.zeros_like(gt, dtype=bool)
    thin[6:14, 6:14] = True                              # eroded by exactly 1
    lg = _logits_from_mask(thin)

    plain = score_episode(lg, [7], gt)["miou"]
    b = boundary_variants(lg, [7], gt)
    assert plain < 1.0
    assert b["argmax_best3"] == pytest.approx(1.0)
    assert b["independent_best3"] == pytest.approx(1.0)


def test_best3_forgives_a_one_pixel_fat_prediction_via_erosion():
    import numpy as np

    from occl.metrics import boundary_variants, score_episode

    gt = np.zeros((20, 20), dtype=np.int32)
    gt[6:14, 6:14] = 3
    fat = np.zeros_like(gt, dtype=bool)
    fat[5:15, 5:15] = True                               # dilated by exactly 1
    lg = _logits_from_mask(fat)

    assert score_episode(lg, [3], gt)["miou"] < 1.0
    assert boundary_variants(lg, [3], gt)["argmax_best3"] == pytest.approx(1.0)


def test_best3_is_never_below_the_plain_score_and_erosion_to_empty_is_safe():
    """As-is is one of the three variants, so best3 >= plain by construction; a 1-px
    structure that erosion wipes out must cost nothing (the max ignores that variant)."""
    import numpy as np

    from occl.metrics import boundary_variants, score_episode

    rng = np.random.default_rng(0)
    gt = np.zeros((24, 24), dtype=np.int32)
    gt[2:9, 2:9] = 1
    gt[12, 2:20] = 4                                     # 1-px-thin line trait
    pred1 = np.zeros_like(gt, dtype=bool)
    pred1[3:10, 2:9] = True                              # shifted, partial overlap
    pred4 = np.zeros_like(gt, dtype=bool)
    pred4[12, 4:18] = True                               # thin line, erodes to empty
    lg = np.stack([_logits_from_mask(pred1)[0], _logits_from_mask(pred4)[0]])

    plain = score_episode(lg, [1, 4], gt)["miou"]
    b = boundary_variants(lg, [1, 4], gt)
    assert b["argmax_best3"] >= plain - 1e-9
    assert b["independent_best3"] >= plain - 1e-9

    # And the joint-per-image rule is what runs: the choice is one variant for ALL
    # traits, so best3 can sit strictly below a per-trait max would.
    assert b["argmax_best3"] <= 1.0


def test_best3_is_none_on_an_all_background_gt():
    import numpy as np

    from occl.metrics import boundary_variants

    lg = _logits_from_mask(np.zeros((8, 8), dtype=bool))
    b = boundary_variants(lg, [1], np.zeros((8, 8), dtype=np.int32))
    assert b == {"argmax_best3": None, "independent_best3": None}
