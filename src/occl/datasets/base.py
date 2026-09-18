"""Dataset-agnostic types shared by the butterfly and beetle adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Specimen:
    """One labeled specimen: an integer trait mask plus a resolvable image source."""

    image_id: str
    group: str
    mask_path: Path
    image_url: str
    split: str  # "train" | "test"


@dataclass(frozen=True)
class Episode:
    """One support -> query prediction task.

    Images are native-resolution RGB uint8; label maps are integer arrays with 0 =
    background. ``class_ids`` is the sorted union of non-zero labels across all supports,
    and is ordered so that adapters can map logit channel -> trait id unambiguously --
    ``propagate_in_video`` does not guarantee its ``obj_ids`` come back in insertion order.
    """

    query_id: str
    support_ids: tuple[str, ...]
    support_images: tuple[np.ndarray, ...]
    support_labelmaps: tuple[np.ndarray, ...]
    query_image: np.ndarray
    class_ids: tuple[int, ...]

    @property
    def shot(self) -> int:
        return len(self.support_ids)


@runtime_checkable
class TraitDataset(Protocol):
    """Minimal surface the protocol driver and evaluator depend on.

    Deliberately splits mask access from image access: masks are committed to the SST
    repo and available offline, while images must be downloaded and cached. Every
    local-only gate (S1) touches masks alone.
    """

    name: str

    def groups(self) -> tuple[str, ...]:
        """Evaluation groups. Butterfly Major = 5 subspecies; Beetle = one universal
        group, because the paper trains a single model across all 12 species."""

    def train_ids(self, group: str) -> list[str]: ...

    def test_ids(self, group: str) -> list[str]: ...

    def load_labelmap(self, image_id: str) -> np.ndarray: ...

    def load_image(self, image_id: str) -> np.ndarray: ...

    def image_url(self, image_id: str) -> str: ...


def read_labelmap(path: Path) -> np.ndarray:
    """Read a committed integer trait mask (pixel value == trait id, 0 = background).

    Read as grayscale, never as a palette image: these are plain mode-'L' PNGs, and a
    palette read would return colour indices. No resizing happens here -- the metric
    resizes *logits* to the GT, never the reverse.
    """
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    return mask.astype(np.int32)


def trait_ids(mask: np.ndarray) -> list[int]:
    """Sorted non-background trait ids present in an integer mask."""
    return [int(v) for v in np.unique(mask) if int(v) != 0]


def union_class_ids(labelmaps: Sequence[np.ndarray]) -> tuple[int, ...]:
    """Sorted union of trait ids across support masks.

    A class missing from support *j* simply gets no prompt on frame *j*; SAM 2 handles
    partially-observed objects natively.
    """
    ids: set[int] = set()
    for lm in labelmaps:
        ids.update(trait_ids(lm))
    return tuple(sorted(ids))
