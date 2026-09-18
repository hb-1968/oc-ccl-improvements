"""Cross-validation folds carved out of the TRAIN split — for S8c-tune.

Epochs is a hyperparameter, and the only honest way to choose one is on data the choice is
not then evaluated on. Selecting it by test mIoU would leak the evaluation set into the
model as surely as training on it: every subsequent number would be reported on data used
to make a decision. Beetle has 120 train / 60 test, so there is room to hold out from
train and leave the test split genuinely untouched.

`CVFold` is a **view**, not a copy. It re-partitions `train_ids()` into a fit fold and a
validation fold, and re-points `test_ids()` at the validation fold — so everything
downstream works unchanged: `curation.support_pool` filters the fit fold, `protocol`
samples supports from it, and `evaluator.evaluate` scores the validation fold with the same
metric the real protocol uses. That matters more than it sounds: the selection criterion
is then **per-trait mIoU on held-out specimens**, not the cycle loss, which is a
silhouette-reconstruction proxy that has already been shown to move monotonically with
learning rate without anyone knowing whether it transfers.

The split is **stratified by species and seeded**. Beetle is a universal model scored on
five test images per species; an unstratified fold could hand a species entirely to one
side, and `canonical_balanced`'s round-robin would then draw supports for species the fold
cannot validate.
"""

from __future__ import annotations

import numpy as np

from ..seeding import derive_seed


class CVFold:
    """Dataset view exposing fold ``i`` of ``k`` over the underlying TRAIN split.

    ``train_ids()`` -> the fit fold, ``test_ids()`` -> the validation fold. The real test
    split is unreachable through this object, which is the point: a tuning run physically
    cannot see it.
    """

    def __init__(self, ds, k: int = 5, fold: int = 0, group: str = "all"):
        if not 0 <= fold < k:
            raise ValueError(f"fold must be in [0, {k}), got {fold}")
        # A view exposes ONE fit/validation partition, so it only makes sense for a dataset
        # with a single group. Butterfly Major has five subspecies and no "all" group at
        # all — the same shape that made the first butterfly fine-tune die on a missing
        # split file. Refuse clearly instead of raising FileNotFoundError two frames down.
        if group not in ds.groups():
            raise ValueError(
                f"{ds.name} has groups {list(ds.groups())}, not {group!r}. CVFold holds one "
                "partition; a multi-group dataset needs one fold view per group, which is "
                "not built (only Beetle is tuned).")
        self.ds = ds
        self.k = k
        self.fold = fold
        self._group = group
        self.name = ds.name          # keeps seeding and cache keys identical to the real run
        self._fit, self._val = self._partition()

    # ── the re-partition ─────────────────────────────────────────────────────

    def _partition(self) -> tuple[list[str], list[str]]:
        ids = sorted(self.ds.train_ids(self._group)
                     if _takes_group(self.ds.train_ids) else self.ds.train_ids())
        by_species: dict[str, list[str]] = {}
        for i in ids:
            by_species.setdefault(self._species(i), []).append(i)

        val: list[str] = []
        for species, members in sorted(by_species.items()):
            # Deterministic per (dataset, species, k): the same fold assignment every run,
            # and independent of `fold` so the k folds genuinely partition the split.
            rng = np.random.default_rng(derive_seed(self.ds.name, species, self.k, 0))
            order = [members[int(j)] for j in rng.permutation(len(members))]
            val += order[self.fold::self.k]        # every k-th, offset by the fold index
        val_set = set(val)
        fit = [i for i in ids if i not in val_set]
        if not val or not fit:
            raise ValueError(
                f"fold {self.fold}/{self.k} is degenerate: {len(fit)} fit, {len(val)} val"
            )
        return fit, sorted(val)

    def _species(self, image_id: str) -> str:
        fn = getattr(self.ds, "species_of", None)
        if fn is not None:
            return fn(image_id)
        # Butterfly ids carry no species prefix; fall back to one stratum, which still
        # gives a seeded split, just not a stratified one.
        return image_id.split("/")[0] if "/" in image_id else "all"

    # ── the dataset interface, delegated ─────────────────────────────────────

    def groups(self):
        return self.ds.groups()

    def train_ids(self, group: str = "all") -> list[str]:
        return list(self._fit)

    def test_ids(self, group: str = "all") -> list[str]:
        return list(self._val)

    def __getattr__(self, item):
        # expected_traits, load_image, load_labelmap, image_url, cache, axis_parts, ...
        return getattr(self.ds, item)

    def __repr__(self) -> str:
        return (f"CVFold({self.ds.name}, fold {self.fold}/{self.k}: "
                f"{len(self._fit)} fit, {len(self._val)} val)")


def _takes_group(fn) -> bool:
    import inspect

    try:
        return bool(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False


def fold_report(ds, k: int = 5, group: str = "all") -> dict:
    """Coverage summary for the k folds — cheap, and the thing to look at before trusting one."""
    folds = [CVFold(ds, k, i, group) for i in range(k)]
    all_val = [set(f.test_ids()) for f in folds]
    union = set().union(*all_val)
    train = set(folds[0].train_ids()) | all_val[0]
    return {
        "k": k,
        "train_size": len(train),
        "fit_sizes": [len(f.train_ids()) for f in folds],
        "val_sizes": [len(f.test_ids()) for f in folds],
        # The two properties that make it a partition rather than a resample.
        "folds_are_disjoint": all(
            not (a & b) for i, a in enumerate(all_val) for b in all_val[i + 1:]),
        "folds_cover_train": union == train,
        "species_per_val": [len({f._species(i) for i in f.test_ids()}) for f in folds],
    }
