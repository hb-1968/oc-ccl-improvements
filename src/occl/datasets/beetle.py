"""2018 NEON Beetles — 12 species, 5 body parts, one universal model.

The paper evaluates Beetle with a **single model across all 12 species**: a support
image drawn from any species must segment beetles of any other. That is what makes 61.9
so much harder than butterfly's 81.0, and why its std is 3.7. Support sampling is
therefore *not* stratified by species — `groups()` returns one group, `"all"`.

**The 120/60 split is ours, not the authors'.** Verified: `data/neon_beetles/metadata.json`
holds 181 entries as three parallel arrays (`url`, `mask_path`, `species`), and *every*
URL points into the `.../Separate_segmented_train_test_splits_80_20/test/...` directory —
that is an unrelated 80/20 split of the whole NEON corpus, not the paper's trait split.
The paper's 10-train/5-test-per-species assignment is unpublished and unrecoverable, so
`scripts/make_beetle_split.py` synthesises a seeded one, commits it, and never
regenerates it. This is the largest single source of expected drift on Beetle: if Beetle
misses its gate while Butterfly Major hits, suspect the split before the code.

Counts are 15 per species except `restrictus` (16), which the split trims to 15 so the
shape matches the paper's 180 = 12 x 15 exactly. Metadata and disk agree 1:1 with zero
missing masks.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np

from .. import paths
from .base import Specimen, read_labelmap

#: Part id -> name. The repo README names the five parts but never states the integer
#: mapping. Confirmed visually from a rendered overlay: on a head-up specimen the labels
#: run 1=head (top), 2=pronotum, 3=elytra (the large wing case), with 4=antenna and
#: 5=legs as the thin structures. Used for reporting only, never for scoring.
PART_NAMES = {1: "head", 2: "pronotum", 3: "elytra", 4: "antenna", 5: "legs"}

N_PARTS = 5
GROUPS = ("all",)
SPLIT_FILE = Path(__file__).parent / "splits" / "beetle_10_5.json"


class Beetle:
    name = "beetle"

    #: (head, tail) part ids defining the body axis, used by `occl.curation` to detect
    #: inverted specimens. Head -> elytra is the longest reliable axis: both parts are
    #: large and present in every mask, unlike antennae or legs.
    axis_parts = (1, 3)

    def __init__(self, root: Path | None = None, cache=None, split_file: Path | None = None):
        self.root = Path(root) if root else paths.beetle_root()
        self.cache = cache
        self.split_file = Path(split_file) if split_file else SPLIT_FILE

    def groups(self) -> tuple[str, ...]:
        return GROUPS

    def expected_traits(self, group: str = "all") -> int:
        """All five parts are labelled on every beetle, so completeness is well defined."""
        return N_PARTS

    # ── manifest ─────────────────────────────────────────────────────────────

    @cached_property
    def _index(self) -> dict[str, Specimen]:
        meta = json.load(open(self.root / "metadata.json"))
        lens = {k: len(v) for k, v in meta.items()}
        if len(set(lens.values())) != 1:
            raise ValueError(f"metadata.json arrays are not parallel: {lens}")

        split = self._split
        idx: dict[str, Specimen] = {}
        for url, mask_rel, species in zip(meta["url"], meta["mask_path"], meta["species"]):
            image_id = f"{species}/{Path(mask_rel).stem}"
            mask_path = self.root / mask_rel
            if not mask_path.exists():
                continue  # guard: README claims 180, metadata holds 181
            if image_id in split["train"]:
                s = "train"
            elif image_id in split["test"]:
                s = "test"
            else:
                continue  # trimmed by the split (the 16th restrictus)
            idx[image_id] = Specimen(image_id, "all", mask_path, url, s)
        return idx

    @cached_property
    def _split(self) -> dict[str, set[str]]:
        if not self.split_file.exists():
            raise FileNotFoundError(
                f"{self.split_file} missing. Generate it once with:\n"
                "  python scripts/make_beetle_split.py"
            )
        raw = json.load(open(self.split_file))
        return {"train": set(raw["train"]), "test": set(raw["test"])}

    def all_specimens(self) -> list[Specimen]:
        return list(self._index.values())

    def specimen(self, image_id: str) -> Specimen:
        return self._index[image_id]

    def train_ids(self, group: str = "all") -> list[str]:
        return sorted(k for k, v in self._index.items() if v.split == "train")

    def test_ids(self, group: str = "all") -> list[str]:
        return sorted(k for k, v in self._index.items() if v.split == "test")

    def species_of(self, image_id: str) -> str:
        return image_id.split("/")[0]

    # ── IO ───────────────────────────────────────────────────────────────────

    def load_labelmap(self, image_id: str) -> np.ndarray:
        return read_labelmap(self.specimen(image_id).mask_path)

    def image_url(self, image_id: str) -> str:
        return self.specimen(image_id).image_url

    def load_image(self, image_id: str) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError(
                "Beetle was constructed without an image cache; build it first (stage S2)."
            )
        return self.cache.load(self.name, image_id)


def read_raw_metadata(root: Path | None = None) -> list[tuple[str, str, str]]:
    """(image_id, mask_rel, url) for all 181 entries, ignoring the split.

    Used by `scripts/make_beetle_split.py`, which must see every candidate before the
    split exists.
    """
    root = Path(root) if root else paths.beetle_root()
    meta = json.load(open(root / "metadata.json"))
    return [
        (f"{sp}/{Path(mp).stem}", mp, url)
        for url, mp, sp in zip(meta["url"], meta["mask_path"], meta["species"])
    ]
