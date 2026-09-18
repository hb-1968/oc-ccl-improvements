"""The metric. Nothing else in this package computes IoU.

Implements the SST paper's evaluation metric (arXiv 2501.06749 v2): "mean IoU (mIoU)
as the main metric, averaged over traits in images" (v1: "averaged over part or trait
IDs"), with overlapping pixels across classes resolved by assigning each pixel to the
highest-scoring label.

This differs from the SST repo's own harness in two ways that matter:

* `experiments/eval_all_ablations.py::load_mask` does `(mask > 0)`, collapsing every
  trait into one foreground blob. That is a *binary* IoU and is not the paper's number.
  It is still computed here as `iou_binary` for continuity with existing repo results,
  but it is NOT the paper-comparable figure.
* `experiments/bench/butterfly_common.py::per_trait_iou` scores each trait independently
  from a dict of boolean masks, which permits two traits to claim the same pixel. The
  paper's "highest score label" rule forbids that, so we argmax over raw logits instead.

Order of operations is fixed and must not be permuted: **resize logits -> argmax -> IoU**.
Argmax-then-resize would nearest-neighbour a label map and change the answer. `score_episode`
enforces this structurally; prefer it over calling the pieces by hand.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

BACKGROUND = 0

#: What `score_episode` actually computes, as a record block. Single source of truth: the
#: evaluator copies this verbatim and `registry.FAITHFUL_SPEC` asserts against the same
#: strings, so the metric a bundle *claims* cannot drift from the metric that ran.
METRIC_SPEC = {
    "name": "iou_per_trait_macro",
    "assignment": "argmax_label",
    "reduction": "macro_over_gt_traits",
}

#: How IoU is measured when GT resolution and model resolution differ.
#: ``gt_native``  - resize logits down to the GT's own resolution (default).
#: ``model``      - resize the GT label map up to the model's resolution (repo-compatible).
IOU_SPACES = ("gt_native", "model")


def resize_logits(logits: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a (C, H, W) logit stack to ``size`` = (h, w).

    cv2.resize caps out at 4 channels, so this loops. C is the number of traits
    (<= 10 for butterfly, 5 for beetle), so the loop is cheap.
    """
    h, w = size
    if logits.ndim != 3:
        raise ValueError(f"expected (C, H, W) logits, got shape {logits.shape}")
    if logits.shape[-2:] == (h, w):
        return logits.astype(np.float32, copy=False)
    out = np.empty((logits.shape[0], h, w), dtype=np.float32)
    for c in range(logits.shape[0]):
        out[c] = cv2.resize(
            logits[c].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )
    return out


def resize_labelmap(lab: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an integer label map (never interpolate labels)."""
    h, w = size
    if lab.shape == (h, w):
        return lab
    return cv2.resize(lab.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)


def logits_to_labelmap(logits: np.ndarray, class_ids: Sequence[int]) -> np.ndarray:
    """(C, H, W) float logits -> (H, W) integer label map.

    Implements the paper's overlap rule: each pixel takes the highest-scoring label.
    A pixel whose best logit is <= 0 is background -- equivalent to appending a
    constant-zero background channel and taking a plain argmax over C+1, but without
    materialising the extra channel. SAM 2/3 mask logits are calibrated so that
    ``> 0`` means "inside the object", which is what makes that equivalence hold.

    Ties go to the lowest channel index (numpy ``argmax`` semantics), so the result is
    deterministic given a fixed ``class_ids`` order.
    """
    if len(class_ids) != logits.shape[0]:
        raise ValueError(
            f"class_ids has {len(class_ids)} entries but logits has "
            f"{logits.shape[0]} channels"
        )
    if logits.shape[0] == 0:
        return np.full(logits.shape[-2:], BACKGROUND, dtype=np.int32)

    best = logits.argmax(axis=0)
    top = np.take_along_axis(logits, best[None], axis=0)[0]
    lut = np.asarray(class_ids, dtype=np.int32)
    return np.where(top > 0.0, lut[best], BACKGROUND).astype(np.int32)


def per_image_miou(
    pred_lab: np.ndarray, gt_lab: np.ndarray
) -> tuple[float | None, dict[int, float]]:
    """mIoU over the traits present in *this image's* GT.

    Returns ``(mean, {trait_id: iou})``, or ``(None, {})`` when the GT is entirely
    background -- the caller must skip such images rather than averaging in a NaN.

    Edge cases, decided (see plan):

    * Trait predicted but absent from GT -> contributes no term. It still hurts
      indirectly: argmax gave it pixels that GT-present traits wanted, lowering theirs.
    * Trait present in GT but never prompted (absent from the support) -> IoU 0,
      *counted*. This is a real failure of the one-shot support draw and is a main
      driver of the run-to-run std the paper reports. Excluding it would shrink our
      std below theirs.
    * Trait in the support but empty in this GT -> excluded, symmetric with the first.

    Background is never an IoU term; it participates only via the ``top > 0`` rule in
    :func:`logits_to_labelmap`.
    """
    if pred_lab.shape != gt_lab.shape:
        raise ValueError(
            f"pred {pred_lab.shape} and gt {gt_lab.shape} must have the same shape; "
            "resize logits before argmax, never the label map afterwards"
        )

    traits = [int(t) for t in np.unique(gt_lab) if int(t) != BACKGROUND]
    if not traits:
        return None, {}

    per: dict[int, float] = {}
    for t in traits:
        p = pred_lab == t
        g = gt_lab == t
        inter = int(np.count_nonzero(p & g))
        union = int(np.count_nonzero(p | g))
        per[t] = (inter / union) if union else 0.0
    return float(np.mean(list(per.values()))), per


def binary_iou(pred_lab: np.ndarray, gt_lab: np.ndarray) -> float:
    """Foreground IoU with all traits merged.

    This is what `eval_all_ablations.py` computes. Reported for continuity with existing
    repo numbers -- it is NOT the paper's metric.
    """
    p = pred_lab > BACKGROUND
    g = gt_lab > BACKGROUND
    union = int(np.count_nonzero(p | g))
    return (int(np.count_nonzero(p & g)) / union) if union else 0.0


def score_episode(
    logits: np.ndarray,
    class_ids: Sequence[int],
    gt_lab: np.ndarray,
    iou_space: str = "gt_native",
) -> dict:
    """Score one support->query episode. The only entry point callers should use.

    Enforces resize -> argmax -> IoU. ``logits`` are raw (never binarised) so the
    argmax overlap rule is available; adapters must return logits for this reason.

    Returns ``{"miou", "per_trait", "iou_binary", "n_traits", "iou_space"}`` with
    ``miou`` set to ``None`` for an all-background GT.
    """
    if iou_space not in IOU_SPACES:
        raise ValueError(f"iou_space must be one of {IOU_SPACES}, got {iou_space!r}")

    if iou_space == "gt_native":
        lg = resize_logits(logits, gt_lab.shape)
        gt = gt_lab
    else:
        lg = logits
        gt = resize_labelmap(gt_lab, logits.shape[-2:])

    pred = logits_to_labelmap(lg, class_ids)
    miou, per_trait = per_image_miou(pred, gt)
    return {
        "miou": miou,
        "per_trait": per_trait,
        "iou_binary": binary_iou(pred, gt),
        "n_traits": len(per_trait),
        "iou_space": iou_space,
    }


#: Assignment rules that differ ONLY in how a pixel is awarded to a trait. Every one
#: aggregates identically -- macro mean over GT-present traits -- so a difference between
#: them isolates the overlap rule and nothing else.
ASSIGNMENT_RULES = ("argmax", "argmax_no_background", "independent", "oracle")


def assignment_variants(
    logits: np.ndarray,
    class_ids: Sequence[int],
    gt_lab: np.ndarray,
    iou_space: str = "gt_native",
) -> dict[str, float | None]:
    """Score one episode under every plausible reading of the overlap rule.

    The paper defines its metric only as "mean IoU (mIoU) ... averaged over traits in
    images" and **never states how overlapping trait predictions are resolved** (verified
    against v2). `score_episode` uses argmax, which is the reading consistent with SAM 2's
    own non-overlapping constraint; SST's own `per_trait_iou` instead binarises each trait
    independently and permits overlap. Those two can differ by a lot, so which one the
    paper used is a live question that only measurement can settle.

    This function exists to answer it. It deliberately does NOT touch `score_episode` --
    that path produced every recorded number and must stay byte-stable.

    Rules:

    * ``argmax``              -- ours: argmax over channels, ``top > 0`` else background.
    * ``argmax_no_background``-- argmax with no background rule; every pixel gets a trait.
      Isolates how much the ``> 0`` threshold is discarding.
    * ``independent``         -- SST's rule: trait *t* predicted as ``logits[chan(t)] > 0``,
      scored against ``gt == t``. Overlap permitted, so it is never *lower* than argmax on
      a per-trait basis and is the natural suspect if the paper's numbers sit high.
    * ``oracle``              -- for each GT trait, the best IoU achievable from **any**
      channel. An upper bound under an optimal channel->trait permutation. This is the
      load-bearing one: if ``oracle`` is far above ``argmax`` the model located the region
      and merely labelled it wrong (an assignment failure, fixable by support curation);
      if ``oracle`` is also low the region was never segmented at all, and no amount of
      support selection will help.
    """
    if iou_space == "gt_native":
        lg, gt = resize_logits(logits, gt_lab.shape), gt_lab
    else:
        lg, gt = logits, resize_labelmap(gt_lab, logits.shape[-2:])

    traits = [int(t) for t in np.unique(gt) if int(t) != BACKGROUND]
    if not traits:
        return {r: None for r in ASSIGNMENT_RULES}

    ids = list(class_ids)
    chan = {t: i for i, t in enumerate(ids)}
    masks = lg > 0.0                       # (C, H, W) independent binarisation

    def iou(p: np.ndarray, g: np.ndarray) -> float:
        u = int(np.count_nonzero(p | g))
        return (int(np.count_nonzero(p & g)) / u) if u else 0.0

    best = lg.argmax(axis=0)
    top = np.take_along_axis(lg, best[None], 0)[0]
    lut = np.asarray(ids)
    lab_bg = np.where(top > 0.0, lut[best], BACKGROUND)
    lab_nobg = lut[best]

    out: dict[str, float | None] = {}
    out["argmax"] = float(np.mean([iou(lab_bg == t, gt == t) for t in traits]))
    out["argmax_no_background"] = float(np.mean([iou(lab_nobg == t, gt == t) for t in traits]))
    # A trait absent from the support has no channel -> unpredictable -> 0, matching
    # both `per_image_miou` and SST's `per_trait_iou`.
    out["independent"] = float(np.mean([
        iou(masks[chan[t]], gt == t) if t in chan else 0.0 for t in traits
    ]))
    out["oracle"] = float(np.mean([
        max((iou(masks[c], gt == t) for c in range(len(ids))), default=0.0) for t in traits
    ]))
    return out


#: The 8-connected one-pixel structuring element. "One-pixel" is ambiguous between the
#: 3x3 square (8-connected) and the cross (4-connected); the square is the common
#: default and the more generous reading, and it is pinned here so the choice is
#: recorded rather than implicit.
_BOUNDARY_KERNEL = np.ones((3, 3), dtype=np.uint8)


def boundary_variants(
    logits: np.ndarray,
    class_ids: Sequence[int],
    gt_lab: np.ndarray,
    iou_space: str = "gt_native",
) -> dict[str, float | None]:
    """The paper's UNPUBLISHED boundary-tolerant scoring, per author correspondence:

        "after we obtained the predicted mask of a test image, we performed one-pixel
        dilation and one-pixel erosion — creating 3 results per image. Then, for that
        image, we record the highest IoU across these 3 results. We apply the same
        approach to all the compared baselines."

    This appears nowhere in either paper version; it reached us as a recollection
    ("If I recall correctly"), so it is implemented as a RECORDED DIAGNOSTIC beside the
    plain metric, never replacing it. Three readings were fixed a priori and are part of
    the definition here:

    * **Per image, jointly** — one variant ({as-is, dilated, eroded}) is chosen for the
      whole image (the quote says "3 results per image"), maximising that image's mean
      over GT traits. A per-trait max would be strictly more generous and is not what
      the quote describes.
    * **In GT-native space** — the stated rationale is annotator jitter ("they may
      annotate the mask one pixel thinner or one pixel wider"), and the annotation pixel
      is the native-resolution pixel. One pixel at 1024 on a 200-px beetle crop would be
      a fifth of an annotation pixel.
    * **3x3 structuring element** (see `_BOUNDARY_KERNEL`).

    Computed under BOTH assignment rules — ours (`argmax`) and the release's
    (`independent`) — because the two unpublished details compound: the closest reading
    of the paper's full metric is `independent` + best-of-3. Each trait's binary mask is
    dilated/eroded independently within a variant (overlap created by dilation cannot
    matter: scoring is per-trait binary). Best-of-3 is >= the plain score by
    construction, since as-is is one of the three.
    """
    if iou_space == "gt_native":
        lg, gt = resize_logits(logits, gt_lab.shape), gt_lab
    else:
        lg, gt = logits, resize_labelmap(gt_lab, logits.shape[-2:])

    traits = [int(t) for t in np.unique(gt) if int(t) != BACKGROUND]
    if not traits:
        return {"argmax_best3": None, "independent_best3": None}

    ids = list(class_ids)
    chan = {t: i for i, t in enumerate(ids)}
    pred_lab = logits_to_labelmap(lg, ids)
    indep = lg > 0.0

    def iou(p: np.ndarray, g: np.ndarray) -> float:
        u = int(np.count_nonzero(p | g))
        return (int(np.count_nonzero(p & g)) / u) if u else 0.0

    def best3(mask_of) -> float:
        """mask_of(t) -> the plain binary prediction for trait t."""
        best = -1.0
        for op in (None, cv2.dilate, cv2.erode):
            per = []
            for t in traits:
                m = mask_of(t).astype(np.uint8)
                if op is not None:
                    m = op(m, _BOUNDARY_KERNEL, iterations=1)
                per.append(iou(m.astype(bool), gt == t))
            best = max(best, float(np.mean(per)))
        return best

    return {
        "argmax_best3": best3(lambda t: pred_lab == t),
        "independent_best3": best3(
            lambda t: indep[chan[t]] if t in chan else np.zeros_like(gt, dtype=bool)),
    }


def aggregate_run(scores: Sequence[dict]) -> dict:
    """Aggregate per-image scores into one run score.

    Unweighted (macro) mean over non-skipped images -- NOT a pooled/micro IoU. An image
    with 2 traits counts as much as one with 10, which is what "averaged over traits in
    images, then over images" means.
    """
    kept = [s for s in scores if s["miou"] is not None]
    n_skipped = len(scores) - len(kept)
    if not kept:
        return {"miou": None, "iou_binary": None, "n_images": 0, "n_skipped": n_skipped,
                "per_trait_mean": {}}

    per_trait: dict[int, list[float]] = {}
    for s in kept:
        for t, v in s["per_trait"].items():
            per_trait.setdefault(int(t), []).append(v)

    return {
        "miou": float(np.mean([s["miou"] for s in kept])),
        "iou_binary": float(np.mean([s["iou_binary"] for s in kept])),
        "n_images": len(kept),
        "n_skipped": n_skipped,
        "per_trait_mean": {t: float(np.mean(v)) for t, v in sorted(per_trait.items())},
    }
