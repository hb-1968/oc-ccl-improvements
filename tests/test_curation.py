"""Gate S4b — the properties that make canonical support selection defensible.

Curation is exactly where a reproduction can quietly become a fit to the target, so these
tests pin the three things that keep it honest: the threshold is a priori, the estimate is
train-only, and the two arms can never be silently compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from occl.curation import (
    angular_deviation,
    body_axis_angle,
    canonical_pool,
    circular_mean,
    circular_resultant,
    circular_std,
    stratified_supports,
    support_pool,
)
from occl.datasets.beetle import Beetle

KNOWN_BAD = [  # the three worst S4a supports, all badly oriented
    "dubius/beetle_237_994",
    "lachrymosus/beetle_146_1018",
    "dubius/beetle_197_1246",
]


def synthetic(head_at_top: bool = True, size: int = 64) -> np.ndarray:
    """A toy specimen: head (1) at one end, elytra (3) at the other."""
    lm = np.zeros((size, size), dtype=np.int32)
    head, tail = (slice(4, 16), slice(40, 60)) if head_at_top else (slice(40, 60), slice(4, 16))
    lm[head, 24:40] = 1
    lm[tail, 24:40] = 3
    return lm


# ── geometry ─────────────────────────────────────────────────────────────────


def test_axis_flips_by_180_degrees_when_the_specimen_is_inverted():
    """The whole point of the filter: an inverted specimen must be distinguishable."""
    a = body_axis_angle(synthetic(True), 1, 3)
    b = body_axis_angle(synthetic(False), 1, 3)
    assert angular_deviation(a, b) == pytest.approx(180.0, abs=1.0)


def test_axis_is_none_when_a_part_is_missing():
    lm = synthetic()
    lm[lm == 1] = 0
    assert body_axis_angle(lm, 1, 3) is None


def test_circular_mean_wraps_correctly():
    # A plain arithmetic mean would give 0 here, which is 180 degrees wrong.
    assert angular_deviation(circular_mean([179.0, -179.0]), 180.0) < 1.0


def test_circular_std_shrinks_as_concentration_rises():
    tight = circular_std([10.0, 12.0, 8.0, 11.0])
    loose = circular_std([0.0, 90.0, 180.0, 270.0])
    assert tight < loose
    assert circular_resultant([5.0, 5.0, 5.0]) == pytest.approx(1.0)


# ── the three guards ─────────────────────────────────────────────────────────


def test_threshold_is_the_a_priori_training_circular_std():
    """Pinned so a later change cannot silently retune it to close a gap.

    55.6 deg is `sqrt(-2 ln R)` on the Beetle training distribution — a property of how
    the specimens are actually oriented, not a number chosen because it yields 61.9.
    """
    _, stats = canonical_pool(Beetle(), "all")
    assert stats["threshold_deg"] == pytest.approx(55.6, abs=0.5)
    assert stats["train_resultant_R"] == pytest.approx(0.625, abs=0.01)
    assert stats["train_circular_mean_deg"] == pytest.approx(67.1, abs=0.5)


def test_pool_is_estimated_from_train_only():
    """The leakage guard, and the most important test in this file.

    Selecting supports using test-set orientation would inflate a one-shot result with
    information the protocol forbids. Perturbing the test split must change nothing.
    """
    ds = Beetle()
    before, stats_before = canonical_pool(ds, "all")

    original = ds.test_ids
    ds.test_ids = lambda group="all": original(group)[:3]  # brutally truncate test
    after, stats_after = canonical_pool(ds, "all")
    ds.test_ids = original

    assert before == after
    assert stats_before["threshold_deg"] == stats_after["threshold_deg"]


def test_pool_excludes_the_known_bad_supports():
    pool, stats = canonical_pool(Beetle(), "all")
    assert stats["pool_size"] == 79, "pool size is pinned; a change here moves every S4b number"
    assert stats["train_size"] == 120
    for bad in KNOWN_BAD:
        assert bad not in pool, f"{bad} scored 35-44 mIoU in S4a and must be filtered"


def test_criteria_can_be_disabled_independently():
    ds = Beetle()
    orient_only, s1 = canonical_pool(ds, "all", criteria=("orientation",))
    complete_only, s2 = canonical_pool(ds, "all", criteria=("completeness",))
    assert s2["pool_size"] == 118, "only 2 of 120 beetle masks lack a trait"
    assert len(orient_only) < len(complete_only)


def test_uniform_selection_returns_the_whole_train_split():
    pool, stats = support_pool(Beetle(), "all", "uniform")
    assert stats["pool_size"] == 120 and len(pool) == 120


def test_unknown_selection_is_rejected():
    with pytest.raises(ValueError):
        support_pool(Beetle(), "all", "curated")  # near-miss for "canonical"


# ── integration with the rest of the harness ─────────────────────────────────


def test_selection_changes_which_supports_a_seed_draws():
    """Same seed, different pool -> different support. Confirms curation actually bites."""
    from occl.seeding import sample_supports

    ds = Beetle()
    uni, _ = support_pool(ds, "all", "uniform")
    can, _ = support_pool(ds, "all", "canonical")
    drawn_uni = [sample_supports(uni, ds.name, "all", 1, r)[0] for r in range(20)]
    drawn_can = [sample_supports(can, ds.name, "all", 1, r)[0] for r in range(20)]
    assert drawn_uni != drawn_can
    assert all(s in can for s in drawn_can)
    assert not any(b in drawn_can for b in KNOWN_BAD)


# ── S4c: species balance ─────────────────────────────────────────────────────


def test_orientation_filter_unbalances_the_pool():
    """The defect S4c exists to fix, pinned so it cannot silently change."""
    import collections

    pool, stats = canonical_pool(Beetle(), "all")
    per = collections.Counter(i.split("/")[0] for i in pool)
    assert min(per.values()) == 4 and max(per.values()) == 10, (
        "uniform gave every species 10/10; curation leaves a 2.5x skew"
    )


def test_round_robin_covers_every_species_where_random_sampling_does_not():
    """Equal probability is NOT equal representation at n=20.

    Measured, stratified *random* sampling reaches only 9/12 species over 20 runs — a
    coupon-collector problem. Systematic round-robin assignment is what actually balances.
    """
    import collections

    ds = Beetle()
    pool, _ = support_pool(ds, "all", "canonical_balanced")
    drawn = [stratified_supports(pool, ds.name, "all", 1, r)[0] for r in range(20)]
    used = collections.Counter(s.split("/")[0] for s in drawn)
    assert len(used) == 12, f"only {len(used)}/12 species used"
    assert max(used.values()) <= 2 and min(used.values()) >= 1


def test_balanced_draws_stay_inside_the_canonical_pool():
    ds = Beetle()
    pool, _ = support_pool(ds, "all", "canonical_balanced")
    for r in range(20):
        assert stratified_supports(pool, ds.name, "all", 1, r)[0] in pool
        assert not any(b in stratified_supports(pool, ds.name, "all", 5, r) for b in KNOWN_BAD)


def test_balanced_selection_preserves_prefix_nesting():
    """1-shot must remain the first element of 5-shot, or the 1-vs-5 comparison unpairs."""
    ds = Beetle()
    pool, _ = support_pool(ds, "all", "canonical_balanced")
    for r in range(20):
        one = stratified_supports(pool, ds.name, "all", 1, r)
        five = stratified_supports(pool, ds.name, "all", 5, r)
        assert five[:1] == one
        assert len({x.split("/")[0] for x in five}) == 5, "5-shot should span 5 species"


def test_balanced_draw_rejects_more_shots_than_strata():
    ds = Beetle()
    pool, _ = support_pool(ds, "all", "canonical_balanced")
    with pytest.raises(ValueError, match="strata"):
        stratified_supports(pool, ds.name, "all", 13, 0)


def test_registry_refuses_to_pair_the_two_canonical_arms():
    """canonical vs canonical_balanced draw different supports — never pair them."""
    from occl.registry import compare

    from test_registry import make_record

    a = make_record([0.60] * 5, support_selection="canonical")
    b = make_record([0.62] * 5, support_selection="canonical_balanced")
    with pytest.raises(ValueError, match="comparability"):
        compare(b, a)


def test_registry_refuses_to_pair_uniform_against_canonical():
    """They use different supports by construction, so a paired delta is meaningless."""
    from occl.registry import compare

    from test_registry import make_record

    a = make_record([0.60] * 5)
    a["protocol"]["support_selection"] = "uniform"
    b = make_record([0.62] * 5)
    b["protocol"]["support_selection"] = "canonical"
    with pytest.raises(ValueError, match="comparability"):
        compare(b, a)


# ── S5b: placement curation for Butterfly Major ──────────────────────────────


def test_placement_displacement_is_zero_for_an_identical_configuration():
    from occl.curation import placement_displacement

    ref = {1: (10.0, 20.0), 2: (30.0, 40.0)}
    assert placement_displacement(dict(ref), ref) == pytest.approx(0.0)


def test_placement_displacement_measures_what_orientation_cannot():
    """A specimen can be perfectly upright and still be in the wrong place.

    This is the whole reason placement is a separate criterion: translating a
    configuration leaves every angle unchanged, so `body_axis_angle` sees nothing.
    """
    from occl.curation import body_axis_angle, placement_displacement

    lm = synthetic(True)
    shifted = np.zeros_like(lm)
    shifted[:, 10:] = lm[:, :-10]  # translate 10px right, no rotation
    assert body_axis_angle(lm, 1, 3) == pytest.approx(body_axis_angle(shifted, 1, 3), abs=1.0)

    ref = {1: (0.0, 0.0), 2: (0.0, 10.0)}
    moved = {1: (30.0, 0.0), 2: (30.0, 10.0)}
    assert placement_displacement(moved, ref) == pytest.approx(30.0)


def test_placement_displacement_is_none_without_shared_traits():
    from occl.curation import placement_displacement

    assert placement_displacement({1: (0.0, 0.0)}, {2: (0.0, 0.0)}) is None


def test_butterfly_placement_threshold_is_a_priori_and_per_subspecies():
    """Pinned so a later change cannot silently retune it to close a gap.

    The threshold is the train displacement mean + one train std -- a property of how the
    specimens were actually laid out, not a value chosen because it yields 81.0.
    """
    from occl.curation import PLACEMENT_CRITERIA, canonical_pool
    from occl.datasets.butterfly_major import ButterflyMajor

    ds = ButterflyMajor()
    _, cy = canonical_pool(ds, "cyrbia", criteria=PLACEMENT_CRITERIA)
    _, ml = canonical_pool(ds, "malleti", criteria=PLACEMENT_CRITERIA)
    # cyrbia mixes two imaging protocols, so its spread is several times any other's.
    assert cy["train_placement_mean_px"] == pytest.approx(182.5, abs=2.0)
    assert ml["train_placement_mean_px"] == pytest.approx(51.7, abs=2.0)
    assert cy["train_placement_mean_px"] > 3 * ml["train_placement_mean_px"]
    assert "a priori, not tuned" in cy["placement_threshold_rule"]


def test_butterfly_placement_pool_is_estimated_from_train_only():
    """The leakage guard. Selecting supports by test-set placement would inflate a
    one-shot result with information the protocol forbids."""
    from occl.curation import PLACEMENT_CRITERIA, canonical_pool
    from occl.datasets.butterfly_major import ButterflyMajor

    ds = ButterflyMajor()
    before, sb = canonical_pool(ds, "cyrbia", criteria=PLACEMENT_CRITERIA)
    original = ds.test_ids
    ds.test_ids = lambda group="cyrbia": original(group)[:3]
    after, sa = canonical_pool(ds, "cyrbia", criteria=PLACEMENT_CRITERIA)
    ds.test_ids = original

    assert before == after
    assert sb["placement_threshold_px"] == sa["placement_threshold_px"]


def test_placement_filter_drops_the_worst_placed_cyrbia_supports():
    """Retrospective check against S5a: the supports it removes scored 28.28, those it
    keeps scored 49.72. Pinned so the criterion cannot silently invert."""
    from occl.curation import support_pool
    from occl.datasets.butterfly_major import ButterflyMajor

    ds = ButterflyMajor()
    pool, stats = support_pool(ds, "cyrbia", "canonical_placement")
    assert stats["criteria_applied"] == ["completeness", "placement"]
    assert stats["rejected"]["placement"] > 0
    assert len(pool) < stats["train_size"]
    # The three worst-placed S5a supports must not survive.
    for bad in ("42604", "43016", "43590"):
        assert bad not in pool, f"{bad} scored ~28 mIoU in S5a and must be filtered"


def test_orientation_contributes_nothing_on_butterfly():
    """`axis_parts is None` makes it inert -- asserted, so the docstring cannot drift."""
    from occl.curation import canonical_pool
    from occl.datasets.butterfly_major import ButterflyMajor

    _, st = canonical_pool(ButterflyMajor(), "malleti", criteria=("orientation", "completeness"))
    assert "orientation" not in st["criteria_applied"]
    assert st["rejected"]["orientation"] == 0


def test_all_curated_arms_are_faithful_only_uniform_deviates():
    """Every curated arm approximates the same unpublished paper clause."""
    from occl.registry import classify

    from test_registry import make_record

    for sel in ("canonical", "canonical_balanced", "canonical_placement"):
        rec = make_record([0.60] * 5, support_selection=sel)
        dev, _ = classify(rec)
        assert "protocol.support_selection" not in dev, sel
    rec = make_record([0.60] * 5)
    rec["protocol"]["support_selection"] = "uniform"
    dev, track = classify(rec)
    assert "protocol.support_selection" in dev and track == "modified"


# ── S5c / S6b: orientation ───────────────────────────────────────────────────


def test_rotate_episode_keeps_image_and_mask_aligned():
    """A mask rotated without its image would be worse than useless."""
    from occl.adapters.tta import ROT_SUFFIX, rotate_episode
    from occl.datasets.base import Episode

    img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    lm = np.arange(16, dtype=np.int32).reshape(4, 4)
    ep = Episode(query_id="q", support_ids=("s",), support_images=(img,),
                 support_labelmaps=(lm,), query_image=img, class_ids=(1, 2))
    r = rotate_episode(ep)
    assert r.support_ids == ("s" + ROT_SUFFIX,), "rotated support needs its own cache key"
    assert np.array_equal(r.support_images[0], np.rot90(img, 2))
    assert np.array_equal(r.support_labelmaps[0], np.rot90(lm, 2))
    assert np.array_equal(r.query_image, img), "the query must NOT be rotated"
    assert r.class_ids == ep.class_ids and r.query_id == ep.query_id


def test_tta_picks_the_more_confident_orientation_without_ground_truth():
    from occl.adapters.tta import OrientationTTAAdapter, confidence
    from occl.datasets.base import Episode

    weak = np.full((2, 8, 8), 0.1, dtype=np.float32)
    strong = np.full((2, 8, 8), 5.0, dtype=np.float32)
    assert confidence(strong) > confidence(weak)
    assert confidence(np.full((2, 8, 8), -1.0, dtype=np.float32)) == float("-inf")

    class Inner:
        name = "fake"

        def predict(self, ep):
            # the rotated variant is the confident one
            return strong if ep.support_ids[0].endswith("@r180") else weak

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    ep = Episode(query_id="q", support_ids=("s",), support_images=(img,),
                 support_labelmaps=(np.zeros((8, 8), np.int32),),
                 query_image=img, class_ids=(1, 2))
    a = OrientationTTAAdapter(Inner())
    assert np.array_equal(a.predict(ep), strong)
    assert a.n_rotated_chosen == 1 and a.n_total == 1


def test_tta_precompute_caches_both_orientations():
    """Without this the rotated pass misses the cache and re-runs the encoder — the ~95%
    of compute the cache exists to remove."""
    from occl.adapters.tta import ROT_SUFFIX, OrientationTTAAdapter

    seen = []

    class Inner:
        name = "fake"

        def precompute(self, pairs):
            seen.extend(k for k, _ in pairs)
            return {"n_cached": len(seen)}

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    OrientationTTAAdapter(Inner()).precompute([("a", img), ("b", img)])
    assert seen == ["a", "a" + ROT_SUFFIX, "b", "b" + ROT_SUFFIX]


def test_orientation_diverse_spans_both_orientations_for_cyrbia():
    """cyrbia's train pool is 34% rotated, so a 5-shot set can carry both — drawn from
    REAL specimens, not synthesised."""
    from occl.curation import (
        orientation_diverse_supports,
        orientation_strata,
        support_pool,
    )
    from occl.datasets.butterfly_major import ButterflyMajor

    ds = ButterflyMajor()
    pool, stats = support_pool(ds, "cyrbia", "orientation_diverse")
    # The placement filter would reject rotated specimens by construction — it measures
    # displacement from the train MEDIAN — so this arm must not apply it.
    assert stats["criteria_applied"] == ["completeness"]
    state = orientation_strata(ds, "cyrbia", sorted(pool))
    sel = orientation_diverse_supports(ds, "cyrbia", pool, ds.name, 5, 0)
    kinds = {state[i] for i in sel}
    assert kinds == {"upright", "rot180"}, f"5-shot set must span orientations, got {kinds}"
    assert len(sel) == 5 and len(set(sel)) == 5


def test_orientation_diverse_degenerates_where_one_orientation_exists():
    """malleti is 100% upright — the criterion must be a no-op there, not an error."""
    from occl.curation import orientation_diverse_supports, support_pool
    from occl.datasets.butterfly_major import ButterflyMajor

    ds = ButterflyMajor()
    pool, _ = support_pool(ds, "malleti", "orientation_diverse")
    sel = orientation_diverse_supports(ds, "malleti", pool, ds.name, 5, 0)
    assert len(sel) == 5 and len(set(sel)) == 5
    assert len(orientation_diverse_supports(ds, "malleti", pool, ds.name, 1, 0)) == 1


def test_orientation_is_measured_against_the_masks_own_extent():
    """Beetle crops are 172-431 px; butterfly masks are 1024^2.

    Hardcoding 1024 for the mirror was correct for butterfly and silently wrong for
    beetle — the rotated reference landed outside the image, so every specimen scored as
    upright and the criterion became a no-op (118/118 one class). With the mask's own
    extent it separates 106/12.
    """
    from occl.curation import orientation_of

    # A small mask: two traits, and the same configuration flipped about a 300px extent.
    ref = {1: (60.0, 60.0), 2: (240.0, 240.0)}
    flipped = {1: (240.0, 240.0), 2: (60.0, 60.0)}
    assert orientation_of(ref, ref, size=300) == "upright"
    assert orientation_of(flipped, ref, size=300) == "rot180"
    # With the wrong extent the flip is invisible — this is the bug, pinned.
    assert orientation_of(flipped, ref, size=1024) == "upright"


def test_beetle_orientation_strata_separates_both_classes():
    import collections

    from occl.curation import orientation_strata, support_pool
    from occl.datasets.beetle import Beetle

    ds = Beetle()
    pool, _ = support_pool(ds, "all", "orientation_diverse")
    counts = collections.Counter(orientation_strata(ds, "all", sorted(pool)).values())
    assert set(counts) == {"upright", "rot180"}, f"expected both classes, got {dict(counts)}"
    assert counts["rot180"] >= 5, "beetle has genuinely flipped specimens; do not lose them"


def test_orientation_diverse_draw_differs_from_the_uniform_draw():
    """The guard below is only meaningful if the two draws actually disagree.

    If `orientation_diverse` happened to return what `sample_supports` returns, a
    fall-through in `build_cache` would be harmless and the drift guard would be
    vacuous. Beetle has both orientation classes, so they must differ.
    """
    from occl.curation import draw_supports, support_pool
    from occl.datasets.beetle import Beetle
    from occl.seeding import sample_supports

    ds = Beetle()
    pool, _ = support_pool(ds, "all", "orientation_diverse")
    diverse = [draw_supports(ds, "all", pool, "orientation_diverse", 5, r) for r in range(5)]
    uniform = [sample_supports(pool, ds.name, "all", 5, r) for r in range(5)]
    assert diverse != uniform, "orientation-diverse draw is indistinguishable from uniform"


def test_support_draw_is_dispatched_in_exactly_one_place():
    """Everything that resolves a support set must use `curation.draw_supports`.

    Three modules independently branched on `support_selection`, and they drifted on
    `orientation_diverse`: `protocol.build_run` called `orientation_diverse_supports`
    while `build_cache.required_butterfly` and `evaluator.distinct_images` both fell
    through to `sample_supports`. Consequences, in the order they bite:

      * the cache is pre-built from a support set the run never asks for;
      * the preflight reports the WRONG images as missing, so topping up the cache from
        its own error message does not fix it;
      * `precompute` warms embeddings for images no episode uses.

    The first version of this test enumerated the two modules already fixed, so it passed
    while `evaluator.py` — the one that drives both the preflight and precompute — was
    still broken. It now DISCOVERS callers instead: every module under src/ and scripts/
    except the two that legitimately define the draws. A new file cannot opt out of it.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    exempt = {"src/occl/curation.py", "src/occl/seeding.py"}
    banned = ("sample_supports(", "stratified_supports(", "orientation_diverse_supports(")

    files = sorted(list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")))
    checked = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel in exempt or "__pycache__" in rel:
            continue
        body = "\n".join(ln for ln in path.read_text().splitlines()
                          if not ln.lstrip().startswith("#"))
        code = body.split('"""')[0] + '"""'.join(body.split('"""')[2::2])  # drop docstrings
        hits = [n for n in banned if n in code]
        assert not hits, (
            f"{rel} calls {hits} directly — route through curation.draw_supports so the "
            "cache, the preflight and the run cannot disagree"
        )
        if "draw_supports(" in code:
            checked.append(rel)

    # Not vacuous: the three known callers must all be routed through the dispatcher.
    for rel in ("src/occl/protocol.py", "src/occl/evaluator.py", "scripts/build_cache.py"):
        assert rel in checked, f"{rel} does not use the shared dispatcher"


def test_notebook_functions_do_not_shadow_module_imports():
    """A function that assigns a name the module imported breaks the module's use of it.

    `fanout` did exactly this: it bound `_t` for a TTA counter dict while also calling
    `_t.sleep(STAGGER_S)` from the module-level `import time as _t`. Python then treats
    `_t` as local for the WHOLE function body, so the sleep raises UnboundLocalError
    before the assignment is ever reached. It fired only with more than one worker — the
    stagger is skipped for i == 0 — so every serial run passed and the first parallel
    fan-out died after printing its worker plan.

    The notebook cannot be executed here (Colab, GPU, Drive), so this is the only place
    that class of error is catchable before it costs an A100 session.
    """
    import ast
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "notebooks" / "run_stage.py").read_text()

    # Neutralise IPython `!shell` magics, INCLUDING backslash continuations — a wget URL
    # on its own line is not valid Python and would fail the parse before any check runs.
    out, in_magic = [], False
    for ln in src.splitlines():
        # Read the continuation marker from the ORIGINAL line: rewriting it to `pass`
        # first drops the trailing backslash and the continuation is never followed.
        cont = ln.rstrip().endswith("\\")
        if re.match(r"^\s*(!(?!=)|%)", ln):      # !shell and %magic; NOT `!=` continuations
            out.append(re.sub(r"^(\s*).*$", r"\1pass", ln))
        elif in_magic:
            out.append("")
        else:
            out.append(ln)
            cont = False
        in_magic = cont
    tree = ast.parse("\n".join(out))

    imported = set()
    for node in tree.body:                       # module level only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= {a.asname or a.name.split(".")[0] for a in node.names}

    bad = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        assigned = {t.id for n in ast.walk(fn) for t in
                    (n.targets if isinstance(n, ast.Assign) else [])
                    if isinstance(t, ast.Name)}
        assigned |= {n.target.id for n in ast.walk(fn)
                     if isinstance(n, (ast.AugAssign, ast.AnnAssign))
                     and isinstance(n.target, ast.Name)}
        declared = {name for n in ast.walk(fn) if isinstance(n, ast.Global)
                    for name in n.names}
        for name in sorted(assigned & imported - declared):
            bad.append(f"{fn.name}() assigns {name!r}, which the module imports")
    assert not bad, "shadowed module imports:\n  " + "\n  ".join(bad)
