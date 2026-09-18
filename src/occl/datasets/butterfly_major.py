"""Cambridge Butterfly (Heliconius) "Major" — 5 subspecies, 100 test images each.

Ported from `experiments/bench/butterfly_common.py` in the SST checkout (its loaders are
clean and torch-free); the per-trait IoU there is *not* ported, because it permits
overlapping predictions. See `occl.metrics` for the replacement.

Measured facts about the committed data (verified, not assumed):

* Exactly 100 test entries per subspecies -> **500 total**, matching the paper.
* Train: cyrbia 929, lativitta 1727, malleti 796, (malleti x plesseni) x malleti 404,
  notabilis x lativitta 371 = 4227. The paper says 2831; irrelevant for one-shot, where
  we draw at most 5 per run, but worth knowing.
* Every mask is natively **1024x1024** — the same resolution SAM 2 runs at, so no
  interpolation happens anywhere in the butterfly metric path.
* Trait vocabularies differ per subspecies (6 to 10). This is *why* Major is evaluated
  per-subspecies: a support from one subspecies cannot label another's traits.
* Test image extensions are `.JPG` (365), `.CR2` (132) and `.jpeg` (2). Extension
  handling must be case- and variant-tolerant.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np

from .. import paths
from .base import Specimen, read_labelmap

#: Subspecies -> number of distinct trait ids (labels are 1..N, 0 = background).
#: Asserted by gate S1; a mismatch means the committed masks changed under us.
TRAIT_VOCAB = {
    "cyrbia": 6,
    "lativitta": 8,
    "malleti": 10,
    "(malleti x plesseni) x malleti": 10,
    "notabilis x lativitta": 8,
}

GROUPS = tuple(TRAIT_VOCAB)


class ButterflyMajor:
    name = "butterfly_major"

    #: Orientation is a *no-op* here, which is not the same as unmeasurable. An earlier
    #: note recorded the axis as ill-defined at R 0.11-0.70, but that used a whole-
    #: silhouette principal axis. Measured the way Beetle is -- a directed vector between
    #: two trait centroids -- butterfly is R 0.993-0.9996 (circular std 1.6-6.9 deg):
    #: exceptionally well-defined, and already canonical, so filtering on it would remove
    #: nothing. `occl.curation` skips the criterion when this is None.
    #:
    #: The geometric property that *does* matter here is **placement** -- see
    #: `curation.PLACEMENT_CRITERIA`. These are dissected wings laid loosely on a sheet,
    #: and a specimen can be perfectly upright while sitting in the wrong corner.
    axis_parts = None

    def __init__(self, root: Path | None = None, cache=None):
        self.root = Path(root) if root else paths.butterfly_root()
        self.cache = cache

    def groups(self) -> tuple[str, ...]:
        return GROUPS

    def expected_traits(self, group: str) -> int:
        """Trait count varies by subspecies (6-10), which is why Major is scored per group."""
        return TRAIT_VOCAB[group]

    # ── manifests ────────────────────────────────────────────────────────────

    def _manifest(self, group: str, split: str) -> list[Specimen]:
        path = self.root / "train_test_separate" / group / f"{split}_data.json"
        out = []
        for image_id, url, mask_rel in json.load(open(path)):
            mask_path = self.root / mask_rel
            if not mask_path.exists():
                continue
            out.append(
                Specimen(
                    image_id=str(image_id),
                    group=group,
                    mask_path=mask_path,
                    image_url=url,
                    split=split,
                )
            )
        return out

    @cached_property
    def _index(self) -> dict[str, Specimen]:
        idx: dict[str, Specimen] = {}
        for g in GROUPS:
            for split in ("train", "test"):
                for s in self._manifest(g, split):
                    if s.image_id in idx:
                        raise ValueError(
                            f"duplicate butterfly image_id {s.image_id!r} in "
                            f"{idx[s.image_id].group}/{idx[s.image_id].split} and "
                            f"{s.group}/{s.split} — ids are assumed globally unique"
                        )
                    idx[s.image_id] = s
        return idx

    def specimen(self, image_id: str) -> Specimen:
        return self._index[str(image_id)]

    def train_ids(self, group: str) -> list[str]:
        return [s.image_id for s in self._manifest(group, "train")]

    def test_ids(self, group: str) -> list[str]:
        return [s.image_id for s in self._manifest(group, "test")]

    # ── IO ───────────────────────────────────────────────────────────────────

    def load_labelmap(self, image_id: str) -> np.ndarray:
        return read_labelmap(self.specimen(image_id).mask_path)

    def image_url(self, image_id: str) -> str:
        return self.specimen(image_id).image_url

    def load_image(self, image_id: str) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError(
                "ButterflyMajor was constructed without an image cache. Butterfly images "
                "are not committed to the repo; build the cache first (stage S2)."
            )
        return self.cache.load(self.name, str(image_id))
