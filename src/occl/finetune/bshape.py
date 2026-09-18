"""Signed-distance gross-shape supervision — S13 arm ``bshape``.

Provenance: Kervadec et al., "Boundary loss for highly unbalanced segmentation"
(arXiv:1812.07032, MIDL 2019). ``L_B = mean(phi_G * s_theta)`` with ``phi_G`` the
signed distance to the GT boundary (negative inside, positive outside) expresses a
contour-distance integral as a regional sum: an error is penalized in proportion to how
far it sits from the true boundary. That is gross-shape supervision — the only shape
signal the project's metric can still pay for, because `metrics.boundary_variants`
(best-of-3, 1-px dilate/erode) forgives the first ring outright.

Metric-aware dead zone (a-priori, recorded): ``phi[|phi| <= 1] = 0``. The best-of-3
metric absorbs exactly 1-px boundary error, so a loss term spending gradient there is
trading in what the metric explicitly forgives.

Normalization (a-priori, recorded): per-channel ``phi / max|phi_c|``, so every
channel's term lives in [-1, 1] regardless of trait size — beetle antennae are a few px
wide, butterfly wings hundreds, and raw pixel distances would weight each channel by
its background extent, not its shape error.

Kervadec's documented failure mode: the boundary term ALONE collapses to empty
foreground (the all-background solution has near-zero boundary gradient). The regional
terms (BCE/CE + Dice) therefore always remain alongside — the s13ft gate pins the
composition. Their per-epoch ramp schedule is degenerate at the spec-pinned epochs=1,
so the weight is constant and travels as the recorded string ``"const"`` (a later ramp
is a value change, not a schema change).

The term is NEGATIVE at its optimum (interior pixels reward sigmoid ~ 1 against
phi < 0), so a total loss with bshape on is not comparable to S8-S12's curves; `parts`
keeps the terms separable.

No scipy — cv2 is already a dependency: ``phi = DT(1-M) - DT(M)`` via
`cv2.distanceTransform` is positive outside and negative inside, and there is no
on-grid zero level set (boundary-adjacent pixels sit at |phi| = 1, which the dead zone
then clears).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def signed_distance_stack(y0: torch.Tensor, dead_zone: float = 1.0,
                          normalize: str = "chan_max") -> torch.Tensor:
    """``(C, 1, H, W)`` GT stack -> same-shape fp32 signed-distance map, no grad.

    Thresholds at 0.5 first, so a soft (bilinearly resized) target produces the same
    phi as its hard mask. A degenerate channel — empty or full after thresholding — has
    no boundary and gets ``phi_c = 0``: no shape supervision rather than a crash.
    """
    m = (y0.detach() > 0.5)[:, 0].to("cpu", torch.uint8).numpy()
    phi = np.zeros(m.shape, np.float32)
    for c in range(m.shape[0]):
        if m[c].min() == m[c].max():
            continue
        phi[c] = (cv2.distanceTransform(1 - m[c], cv2.DIST_L2, 3)
                  - cv2.distanceTransform(m[c], cv2.DIST_L2, 3))
    t = torch.from_numpy(phi)[:, None]
    t[t.abs() <= dead_zone] = 0.0
    if normalize == "chan_max":
        t = t / t.abs().flatten(1).amax(dim=1).clamp(min=1.0)[:, None, None, None]
    return t.to(y0.device)


class _PhiCache:
    """Single-slot phi cache, keyed by exact tensor equality against the last y0.

    Fixed-reference training reuses one y0 all epoch — every step after the first hits.
    Derangement (the S13 carrier) and rotation-aug change y0 per step and miss every
    time, which is fine: the recompute is ~2-4 ms/channel against a multi-second cycle
    step. Hit/miss counters surface in the training report so the record shows which
    regime actually ran.
    """

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self._y0 = None
        self._phi = None

    def reset(self):
        self.hits = 0
        self.misses = 0
        self._y0 = None
        self._phi = None

    def get(self, y0: torch.Tensor) -> torch.Tensor:
        if (self._y0 is not None and self._y0.shape == y0.shape
                and torch.equal(self._y0, y0)):
            self.hits += 1
            return self._phi
        self.misses += 1
        self._y0 = y0.detach()
        self._phi = signed_distance_stack(y0)
        return self._phi


PHI_CACHE = _PhiCache()


def bshape_loss(pred_logits: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    """``mean(sigmoid(z) * phi)`` — Kervadec's regional form of the contour distance.

    phi is a no-grad fp32 constant; the product promotes to fp32 under bf16 autocast,
    which is both correct and cheap.
    """
    return (torch.sigmoid(pred_logits.float()) * PHI_CACHE.get(y0)).mean()
