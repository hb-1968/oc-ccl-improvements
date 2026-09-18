"""Gate S8c-tune — the cross-validation split cannot see the evaluation set.

Choosing a hyperparameter on the test split puts that split into the model as surely as
training on it: every number reported afterwards comes from data used to make a decision.
These tests pin that the folds are carved out of TRAIN only, partition it exactly, and
stay stratified — none of which is visible from a result that merely looks plausible.
"""

from __future__ import annotations

import pytest

from occl.datasets.beetle import Beetle
from occl.datasets.folds import CVFold, fold_report


def test_no_fold_can_reach_the_real_test_split():
    """THE guard. A leak here invalidates every downstream number silently."""
    ds = Beetle()
    real_test = set(ds.test_ids())
    assert real_test, "beetle has no test split — the check would be vacuous"
    for i in range(5):
        f = CVFold(ds, k=5, fold=i)
        reachable = set(f.train_ids()) | set(f.test_ids())
        leaked = reachable & real_test
        assert not leaked, f"fold {i} exposes {len(leaked)} real test images"


def test_folds_partition_the_train_split_exactly():
    rep = fold_report(Beetle(), k=5)
    assert rep["folds_are_disjoint"], "validation folds overlap — some images score twice"
    assert rep["folds_cover_train"], "some train images are in no fold"
    assert sum(rep["val_sizes"]) == rep["train_size"]
    assert all(a + b == rep["train_size"]
               for a, b in zip(rep["fit_sizes"], rep["val_sizes"]))


def test_every_fold_keeps_all_species():
    """Beetle is a universal model over 12 species, and `canonical_balanced` draws by
    round-robin over them. A fold missing a species would validate on supports it has no
    representative for."""
    rep = fold_report(Beetle(), k=5)
    assert rep["species_per_val"] == [12] * 5, rep["species_per_val"]


def test_folds_are_deterministic_and_fold_index_changes_the_split():
    ds = Beetle()
    assert CVFold(ds, 5, 0).test_ids() == CVFold(ds, 5, 0).test_ids()
    assert CVFold(ds, 5, 0).test_ids() != CVFold(ds, 5, 1).test_ids()


def test_curated_support_draw_stays_inside_the_fit_fold():
    """The inner protocol must be the real one, or the selection measures something else."""
    from occl.curation import draw_supports, support_pool

    f = CVFold(Beetle(), k=5, fold=0)
    pool, stats = support_pool(f, "all", "canonical_balanced")
    assert stats["n_strata"] == 12
    fit = set(f.train_ids())
    assert set(pool) <= fit, "the support pool escaped the fit fold"
    for run in range(5):
        drawn = draw_supports(f, "all", pool, "canonical_balanced", 5, run)
        assert set(drawn) <= fit, f"run {run} drew a support from outside the fit fold"


def test_fold_delegates_the_rest_of_the_dataset_interface():
    ds = Beetle()
    f = CVFold(ds, 5, 0)
    assert f.name == ds.name          # keeps seeding and cache keys identical
    assert f.expected_traits("all") == ds.expected_traits("all")
    assert f.groups() == ds.groups()


def test_degenerate_fold_configurations_are_rejected():
    ds = Beetle()
    with pytest.raises(ValueError):
        CVFold(ds, k=5, fold=5)
    with pytest.raises(ValueError):
        CVFold(ds, k=5, fold=-1)


def test_interaction_verdict_thresholds_are_in_fraction_units():
    """Every score in these records is a FRACTION; the thresholds must match.

    The S6c verdict compared a fraction against `0.5`, so a -4.66 **point** interaction —
    strongly redundant — fell through to "additive". The numbers in the record were right
    and only the derived label was wrong, which is the dangerous shape: a reader takes the
    one-word verdict at face value precisely because the arithmetic is there to check it.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "combine_s6c.py").read_text()
    thresholds = [float(x) for x in re.findall(r"interaction [<>] (-?[\d.]+)", src)]
    assert thresholds, "the verdict thresholds vanished"
    assert all(abs(t) < 0.1 for t in thresholds), (
        f"thresholds {thresholds} look like percentage points, but interaction is a "
        "fraction — a 5-point effect would compare as 0.05")


def test_cvfold_refuses_a_dataset_whose_groups_it_cannot_represent():
    """Butterfly Major has five subspecies and no "all" group.

    `CVFold` exposes a single fit/validation partition, so it cannot stand in for a
    multi-group dataset. The failure without this guard is `FileNotFoundError` on a split
    file two frames down — the same shape that killed the first butterfly fine-tune, where
    `--group` defaulted to "all" and `train_ids("all")` went looking for a directory that
    does not exist.
    """
    from occl.datasets.butterfly_major import ButterflyMajor

    with pytest.raises(ValueError, match="has groups"):
        CVFold(ButterflyMajor(), k=5, fold=0)
    # Beetle's single universal group is unaffected.
    assert len(CVFold(Beetle(), k=5, fold=0).test_ids()) == 24


def test_cycle_pairs_cover_every_group_without_crossing_them():
    """The fine-tune is dataset-wide, but a pair must stay inside one subspecies.

    Cross-subspecies pairs would not crash — the cycle supervises a merged foreground, so
    the differing trait vocabularies never surface — but they pose a task the evaluation
    never does, since supports and queries always come from the same subspecies.
    """
    from occl.datasets.butterfly_major import ButterflyMajor
    from occl.finetune.pairs import CyclePairs

    ds = ButterflyMajor()
    pairs = CyclePairs(ds, None).pairs
    group_of = {i: g for g in ds.groups() for i in ds.train_ids(g)}
    assert not [1 for a, b in pairs if group_of[a] != group_of[b]], "a pair crossed groups"
    assert {group_of[a] for a, _ in pairs} == set(ds.groups()), "a subspecies is untrained"
    # interleaved, not blocked by group
    assert len({group_of[a] for a, _ in pairs[:10]}) > 1, "pairs are group-blocked"
    # and no test image anywhere
    test = {i for g in ds.groups() for i in ds.test_ids(g)}
    assert not ({i for pr in pairs for i in pr} & test)


def test_notebook_and_package_builds_are_mutually_satisfiable():
    """`pack.sh` must never emit a pair that refuses to run itself.

    The guard runs both ways: the notebook checks the package is new enough, and the
    package checks the notebook is. Both directions are needed — a stale ZIP was already
    guarded, but a stale NOTEBOOK with a fresh zip was not, and that is the likelier
    mistake because the notebook is the file you have open while "updating the code" means
    re-uploading the zip. Stage dispatch lives in the notebook, so that combination runs
    the previous experiment against new library code and reports it under the current
    stage name.
    """
    import re
    from pathlib import Path

    import occl

    nb = (Path(__file__).resolve().parent.parent
          / "notebooks" / "run_stage.py").read_text()
    repo_min = int(re.search(r"^REPO_MIN_BUILD = (\d+)", nb, re.M).group(1))
    nb_build = int(re.search(r"^NOTEBOOK_BUILD = (\d+)", nb, re.M).group(1))
    assert repo_min <= occl.BUILD, (
        f"notebook needs repo build {repo_min}, package ships {occl.BUILD}")
    assert nb_build >= occl.NOTEBOOK_MIN_BUILD, (
        f"package needs notebook build {occl.NOTEBOOK_MIN_BUILD}, notebook is {nb_build}")
