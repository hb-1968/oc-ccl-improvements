"""The seam between the evaluation protocol and whichever tracker is running.

Kept deliberately narrow so that adding SAM 3 in v2 is a new file rather than a
refactor. Two rules do most of the work:

1. **Adapters return logits, never binary masks.** The paper resolves overlapping
   predictions by assigning each pixel to the highest-scoring label, which is impossible
   once the scores have been thresholded away. `experiments/eval_checkpoint.py` in the
   SST checkout binarises inside its predict function (`logits[k] > 0`) and so cannot
   implement the paper's rule at all.
2. **Channel order is pinned by `Episode.class_ids`.** `propagate_in_video` does not
   guarantee its `obj_ids` come back in insertion order, so adapters must reindex
   against the episode's ordering rather than assuming it.

Resolution differences live entirely inside the adapter: SAM 2 runs at 1024, SAM 3's
tracker at 1008 (ViT patch 14 -> 1008 = 14x72). Callers only ever see logits at the
query image's own resolution.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..datasets.base import Episode


@runtime_checkable
class TrackerAdapter(Protocol):
    name: str
    native_size: int

    def build(self, device: str, dtype) -> None:
        """Instantiate the model and load weights. Must fail loudly on a key mismatch."""

    def predict(self, ep: Episode) -> np.ndarray:
        """Run one episode.

        Returns float32 logits of shape ``(len(ep.class_ids), h, w)`` at the adapter's
        own working resolution -- `occl.metrics.score_episode` resizes them to the GT.
        Channel ``i`` corresponds to ``ep.class_ids[i]``.

        Convention: ``logit > 0`` means the pixel belongs to that object, which is what
        makes the metric's ``top > 0`` background rule equivalent to argmax over an
        appended constant-zero background channel.
        """

    def close(self) -> None:
        """Release GPU memory."""


def pick_dtype(device: str):
    """bf16 on Ampere and later, fp16 below.

    Deliberately keyed on compute capability rather than
    ``torch.cuda.is_bf16_supported()``: since torch 2.6 that returns True on sm_75 (T4)
    via *emulation*, which is numerically fine but has no tensor-core path and runs
    glacially. SST hardcodes ``torch.autocast(dtype=torch.bfloat16)`` in three places,
    which is why it degrades so badly on a free-tier Colab GPU.
    """
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
