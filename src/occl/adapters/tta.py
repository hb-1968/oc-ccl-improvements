"""Test-time augmentation over support orientation.

Why this exists: **37% of cyrbia's test specimens are rotated 180 degrees**, and they
score **0.48 mIoU** against 74.95 for the upright ones — the whole of cyrbia's deficit
(0.63*74.95 + 0.37*0.48 = 47.4, its measured score). SST propagates the support's spatial
layout through the memory bank, so an upside-down query receives every trait on its
mirror-image partner. The copy baseline shows it with no model at all: rotated cyrbia
queries score **0.00**.

That defect is **query-side**, which is why nothing support-side has touched it. Placement
curation moved cyrbia +3.56; 5-shot moved it +2.85. Both collapsed the variance and left
the level alone, which is the signature of a defect invariant to support choice.

TTA attacks it directly: predict twice, once with the support as given and once with the
support rotated 180 degrees, and keep whichever the model is more confident about. The
query is never touched, and **no ground truth is consulted** — the choice is made from the
logits alone, so this leaks nothing.

This is a `modified`-track intervention. SST has no orientation handling anywhere, and the
paper's only rotation is beetle-only fine-tuning augmentation plus an OOD stress test that
*injects* rotation. So this measures "SST plus rotation handling", not SST.
"""

from __future__ import annotations

import numpy as np

from ..datasets.base import Episode

#: Suffix for the rotated variant's embedding-cache key. The cache is keyed by the string
#: in `Episode.support_ids`, so a rotated support needs its own key or it would be served
#: the upright image's features — silently, and with no way to notice.
ROT_SUFFIX = "@r180"


def rotate_episode(ep: Episode) -> Episode:
    """The same episode with every SUPPORT rotated 180 degrees, image and mask together.

    Rotating both keeps them pixel-aligned, which is the whole point — a mask rotated
    without its image would be worse than useless. The query is untouched: what matters is
    the support's orientation *relative* to the query, and rotating the support avoids
    having to rotate the prediction back afterwards.
    """
    return Episode(
        query_id=ep.query_id,
        support_ids=tuple(f"{s}{ROT_SUFFIX}" for s in ep.support_ids),
        support_images=tuple(np.rot90(i, 2).copy() for i in ep.support_images),
        support_labelmaps=tuple(np.rot90(m, 2).copy() for m in ep.support_labelmaps),
        query_image=ep.query_image,
        class_ids=ep.class_ids,
    )


def confidence(logits: np.ndarray) -> float:
    """How sure the model is, from the logits alone. No ground truth involved.

    Mean margin over the pixels it actually claims. A prediction propagated from a
    mismatched orientation is scattered and weakly positive; one from a matching
    orientation is compact and strongly positive. Returns -inf when nothing is claimed at
    all, so an empty prediction always loses.
    """
    top = logits.max(axis=0)
    fg = top > 0.0
    return float(top[fg].mean()) if bool(fg.any()) else float("-inf")


class OrientationTTAAdapter:
    """Wraps a tracker adapter and picks the better support orientation per episode.

    Deliberately a wrapper rather than a change to `SSTSam2Adapter`: that adapter produced
    every recorded number and must stay byte-stable. This only composes it.
    """

    def __init__(self, inner):
        self.inner = inner
        self.name = f"{inner.name}+tta_rot180"
        self.n_rotated_chosen = 0
        self.n_total = 0

    def __getattr__(self, item):          # dtype, config_report, clear_embeddings, ...
        if item == "precompute":
            # Served dynamically, NOT as a class method, so `hasattr(wrapper,
            # "precompute")` mirrors the inner adapter exactly. A class-level method
            # made the wrapper always claim precompute, defeating the evaluator's
            # hasattr guard and crashing at call time on a precompute-less inner
            # (S14: models with no separable encoder stage must degrade cleanly).
            inner_pre = getattr(self.inner, "precompute")  # AttributeError -> hasattr False

            def precompute(pairs):
                """Cache the upright AND rotated embedding for every image.

                Without this the rotated pass would miss the cache and re-run the
                encoder — the ~95% of compute the cache exists to remove — making TTA
                roughly 10x slower rather than 2x. Only supports are ever rotated, but
                every image is cached both ways because `precompute` cannot tell which
                is which.
                """
                def both():
                    for key, img in pairs:
                        yield key, img
                        yield f"{key}{ROT_SUFFIX}", np.rot90(img, 2).copy()

                return inner_pre(both())

            return precompute
        return getattr(self.inner, item)

    def build(self, *a, **kw):
        return self.inner.build(*a, **kw)

    def predict(self, ep: Episode) -> np.ndarray:
        up = self.inner.predict(ep)
        ro = self.inner.predict(rotate_episode(ep))
        self.n_total += 1
        if confidence(ro) > confidence(up):
            self.n_rotated_chosen += 1
            return ro
        return up

    def config_report(self) -> dict:
        rep = dict(self.inner.config_report())
        rep |= {"tta": "rot180_by_confidence",
                "tta_rotated_chosen": self.n_rotated_chosen,
                "tta_episodes": self.n_total}
        return rep
