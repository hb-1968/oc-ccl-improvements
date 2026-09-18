"""The paper's "Copy" baseline — paste the support's mask onto the query, unchanged.

Table A3 reports it as Butterfly Major **20.9 +- 10.9**, Minor 36.1, Fish 10.6,
Beetle **13.2 +- 6.7**. It is the floor any tracking method must clear, and because it
needs no model it doubles as a full end-to-end test of the harness -- protocol
construction, multi-class argmax, per-trait IoU, run aggregation and dispersion -- on CPU,
before a single GPU-hour is spent.

If this reproduces the paper's copy numbers, then a later miss on SAM 2 is a model or
checkpoint problem, not a protocol or metric problem. That is a much cheaper thing to
learn here than after 10,000 episodes on an A100.
"""

from __future__ import annotations

import numpy as np

from ..datasets.base import Episode

CONFIDENCE = 10.0


class CopyBaselineAdapter:
    name = "copy_baseline"
    native_size = 1024

    def __init__(self, native_size: int = 1024):
        self.native_size = native_size
        self.device, self.dtype = "cpu", None

    def build(self, device: str = "cpu", dtype=None, strict: bool = True) -> None:
        self.device = device

    def config_report(self) -> dict:
        return {"image_size": self.native_size, "num_maskmem": 0,
                "checkpoint": None, "dtype": "n/a",
                "load_missing_keys": 0, "load_unexpected_keys": 0}

    def predict(self, ep: Episode) -> np.ndarray:
        """Union the support masks, at the same working resolution the real adapter uses.

        With k > 1 supports a pixel is claimed for class c if *any* support labels it c;
        ties between classes fall to the metric's argmax, which sees equal logits and
        takes the lowest channel index.
        """
        import cv2

        s = self.native_size
        acc = np.full((len(ep.class_ids), s, s), -CONFIDENCE, dtype=np.float32)
        for lm in ep.support_labelmaps:
            lm_r = cv2.resize(lm.astype(np.int32), (s, s), interpolation=cv2.INTER_NEAREST)
            for i, c in enumerate(ep.class_ids):
                acc[i][lm_r == c] = CONFIDENCE
        return acc

    def close(self) -> None:
        pass
