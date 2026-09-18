"""Gate S7 — the registry must refuse bad comparisons and recover a known effect."""

from __future__ import annotations

import numpy as np
import pytest

from occl.registry import (
    FAITHFUL_COMMON,
    classify,
    comparability_key,
    compare,
    per_seed_scores,
)


def make_record(miou_per_run, *, shots=1, support_hash="abc123", finetune=None,
                support_selection="canonical", **over):
    """Build a record that is faithful by default, so each test perturbs exactly one field.

    `support_selection` defaults to "canonical" because that is what the paper does —
    it curates references. A uniform run is the deviation.
    """
    rec = {
        "dataset": "beetle",
        "model": {"family": "sam2", "image_size": 1024, "num_maskmem": 7},
        "metric": {"name": "iou_per_trait_macro", "assignment": "argmax_label",
                   "reduction": "macro_over_gt_traits", "iou_space": "gt_native"},
        "data": {"source_preference": "faithful", "interp": "linear",
                 "cache_format": "png", "normalization": "imagenet"},
        "protocol": {"shots": shots, "n_runs": len(miou_per_run), "groups": ["all"],
                     "support_selection": support_selection,
                     "support_set_hash": support_hash},
        "results": {"miou": float(np.mean(miou_per_run)),
                    # ddof=1 is undefined for a single run; the evaluator reports 0.0 there
                    "std": float(np.std(miou_per_run, ddof=1)) if len(miou_per_run) > 1 else 0.0,
                    "per_run": list(miou_per_run)},
    }
    if finetune is not None:
        rec["finetune"] = finetune
        # Paper-faithful occl block for a beetle record: per-trait cycle (§3's
        # y ∈ {0,1}^{W×H×C}), differentiable prompt (§3.2), one fixed reference (A2.2),
        # rotation aug on x0 (A2.2, beetle-only). Tests perturb one field at a time.
        rec["occl"] = {"prompt_mode": "logit_reparam", "memory_reset": True,
                       "cycle_len": 2, "mask_mode": "per_trait",
                       "reference": "fixed",
                       "augmentation": {"type": "rotation_x0", "max_deg": 27.8,
                                        "rule": "half the train circular std (55.6/2)"}}
    for k, v in over.items():
        rec[k] = v
    return rec


#: The paper's fine-tune block, A2.2 verbatim: LoRA r=32 α=64 on memory encoder + mask
#: decoder, one epoch.
PAPER_FT = {"method": "lora",
            "lora": {"r": 32, "alpha": 64.0,
                     "scopes": ["memory_encoder", "sam_mask_decoder"]},
            "epochs": 1}


# ── track is derived, not asserted ───────────────────────────────────────────


def test_eval_record_with_faithful_settings_is_faithful():
    dev, track = classify(make_record([0.61, 0.62]))
    assert track == "faithful" and dev == []


def test_jpeg_substitution_makes_a_record_modified():
    rec = make_record([0.61, 0.62])
    rec["data"]["source_preference"] = "jpeg"
    dev, track = classify(rec)
    assert track == "modified"
    assert "data.source_preference" in dev


def test_area_interpolation_makes_a_record_modified():
    rec = make_record([0.61])
    rec["data"]["interp"] = "area"
    assert classify(rec)[1] == "modified"


def test_a_typed_track_field_cannot_launder_a_deviation():
    """The whole point: asserting faithfulness must not make it so."""
    rec = make_record([0.61], track="faithful")
    rec["data"]["interp"] = "area"
    dev, track = classify(rec)
    assert track == "modified" and dev == ["data.interp"]


def test_finetune_spec_only_applies_to_finetuned_records():
    assert classify(make_record([0.65]))[1] == "faithful"          # no finetune block
    assert classify(make_record([0.65], finetune=PAPER_FT))[1] == "faithful"
    ft_bad = {**PAPER_FT, "epochs": 10}   # the release's epochs; A2.2 says one
    assert "finetune.epochs" in classify(make_record([0.65], finetune=ft_bad))[0]


def test_paper_lora_config_is_faithful_and_full_ft_deviates():
    """The SECOND flip of this pin, and the history matters, so it is kept here.

    Flip 1: this file originally asserted LoRA r=32/α=64 on memory_encoder+mask_decoder
    was faithful; checking the RELEASED CODE found none of it there (`oc_ccl.py::train`
    full-fine-tunes everything but the image encoder for 10 epochs; LoRA exists only in an
    ablation script, off by default, r 4/8/16, α=1.0), so the pin flipped to `full`.

    Flip 2 (this one): re-reading the PAPER showed the original spec was A2.2 nearly
    verbatim — "we replace all linear projection layers and MLP layers in the memory
    encoder and mask decoder ... with LoRA linear layers, using r=32 and α=64", "train it
    for one epoch" — and that the release (committed ~16 months post-paper, a
    re-implementation) is what deviates. Everything flip 1 said about the release stays
    true OF THE RELEASE; the reproduction target is the paper, so the paper defines
    faithful. Do not flip this again without a source neither flip has already consumed.
    """
    dev, track = classify(make_record([0.65], finetune=PAPER_FT))
    assert track == "faithful" and dev == []

    full = {"method": "full", "frozen": ["image_encoder"], "epochs": 1}
    dev, track = classify(make_record([0.65], finetune=full))
    assert track == "modified"
    assert "finetune.method" in dev


def test_release_config_deviates_on_exactly_the_axes_the_paper_contradicts():
    """The release-vs-paper contradiction table, as a classification.

    SST's shipped OC-CCL: full fine-tune of everything but the image encoder, 10 epochs,
    hard-threshold cycle prompt, merged-foreground mask, every train image a reference, no
    augmentation anywhere. The paper: LoRA r32/α64 on memory encoder + mask decoder, one
    epoch, "this prompt remains differentiable", per-trait y₀, one fixed reference, and
    beetle-only rotation augmentation on the labeled example. Each contradiction is one
    deviating field, and the set is pinned in full so a spec edit that silently absorbs
    one of them fails here.
    """
    rec = make_record([0.65], finetune={"method": "full",
                                        "frozen": ["image_encoder"], "epochs": 10})
    rec["occl"] = {"prompt_mode": "hard_threshold", "memory_reset": True,
                   "cycle_len": 2, "mask_mode": "binary",
                   "reference": "derangement", "augmentation": None}
    dev, track = classify(rec)
    assert track == "modified"
    assert dev == ["finetune.epochs", "finetune.lora.alpha", "finetune.lora.r",
                   "finetune.lora.scopes", "finetune.method", "occl.augmentation.type",
                   "occl.mask_mode", "occl.prompt_mode", "occl.reference"], dev


def test_merged_mask_is_a_deviation_from_the_papers_per_trait_cycle():
    rec = make_record([0.65], finetune=PAPER_FT)
    rec["occl"]["mask_mode"] = "merged"
    assert "occl.mask_mode" in classify(rec)[0]
    rec["occl"]["mask_mode"] = "binary"     # the pre-axis name for the same merge
    assert "occl.mask_mode" in classify(rec)[0]


def test_rotation_aug_is_required_on_beetle_and_forbidden_on_butterfly():
    """A2.2's augmentation clause is dataset-conditional, so the spec must be too."""
    rec = make_record([0.65], finetune=PAPER_FT)
    rec["occl"]["augmentation"] = None
    dev, _ = classify(rec)
    assert "occl.augmentation.type" in dev, dev

    rec = make_record([0.65], finetune=PAPER_FT)
    rec["dataset"] = "butterfly_major"
    dev, _ = classify(rec)      # beetle-style aug on butterfly
    assert "occl.augmentation" in dev, dev
    rec["occl"]["augmentation"] = None
    dev, _ = classify(rec)
    assert "occl.augmentation" not in dev, dev


def test_fixed_and_anchored_references_are_both_faithful_and_derangement_is_not():
    """A2.2 fixes one labeled x0; A2.3 samples a different reference per run; whether the
    OC-CCL rows re-fine-tune per run is unstated. Both readings are admitted, the same way
    `protocol.support_selection` admits several curations of one unpublished clause. The
    derangement (every train image a reference — 120 labels where the paper spends 1) is
    this repo's release-shaped sampler and deviates."""
    for ref in ("fixed", "anchored"):
        rec = make_record([0.65], finetune=PAPER_FT)
        rec["occl"]["reference"] = ref
        assert classify(rec)[1] == "faithful", ref
    rec = make_record([0.65], finetune=PAPER_FT)
    rec["occl"]["reference"] = "derangement"
    assert classify(rec)[0] == ["occl.reference"]


# ── comparability gating ─────────────────────────────────────────────────────


def test_same_settings_share_a_comparability_key():
    assert comparability_key(make_record([0.6])) == comparability_key(make_record([0.7]))


def test_different_support_sets_are_not_comparable():
    a, b = make_record([0.6]), make_record([0.7], support_hash="different")
    assert comparability_key(a) != comparability_key(b)
    with pytest.raises(ValueError, match="comparability"):
        compare(a, b)


def test_different_shot_counts_are_not_comparable():
    with pytest.raises(ValueError):
        compare(make_record([0.6] * 5), make_record([0.7] * 5, shots=5))


def test_metric_variant_gates_comparison():
    a = make_record([0.6] * 5)
    b = make_record([0.7] * 5)
    b["metric"]["iou_space"] = "model"
    with pytest.raises(ValueError):
        compare(a, b)


# ── paired statistics ────────────────────────────────────────────────────────


def test_paired_bootstrap_recovers_a_known_offset():
    rng = np.random.default_rng(0)
    base = rng.normal(0.619, 0.037, 20)          # the paper's Beetle spread
    treat = base + 0.02                          # a clean +2.0 point effect
    c = compare(make_record(treat), make_record(base))
    assert c["mean_delta"] == pytest.approx(0.02, abs=2e-3)
    assert c["ci95"][0] > 0 and c["significant"]
    assert c["n_seeds"] == 20 and c["n_seeds_improved"] == 20


def test_pairing_resolves_an_effect_that_unpaired_spread_would_hide():
    """The Butterfly Major situation: +0.2 against a ±1.0 across-support spread.

    The treatment carries its OWN independent noise on top of the shared support-difficulty
    term, so `paired_std` is genuinely non-zero -- a constant offset would make this test
    pass vacuously by driving the denominator to 0.
    """
    rng = np.random.default_rng(1)
    base = rng.normal(0.810, 0.010, 20)              # support difficulty: the ±1.0 spread
    treat = base + 0.002 + rng.normal(0.0, 0.002, 20)  # +0.2 effect plus run-to-run noise

    c = compare(make_record(treat), make_record(base))
    assert c["paired_std"] > 0, "guard against a vacuous comparison"
    # Support difficulty is the dominant variance term and cancels when differencing.
    # No magic ratio here — the exact factor depends on how correlated the two arms are.
    assert c["paired_std"] < c["unpaired_std_baseline"]
    assert c["significant"], "a consistent +0.2 resolves once support difficulty cancels"

    # The load-bearing claim: the unpaired view genuinely cannot see this effect. The
    # means differ by ~0.2 points against a per-arm spread of ~0.6, i.e. well inside one
    # standard deviation, which is exactly why the paper's 81.0 -> 81.2 is unfalsifiable
    # as reported.
    assert abs(np.mean(treat) - np.mean(base)) < c["unpaired_std_baseline"]


def test_a_null_effect_is_not_called_significant():
    rng = np.random.default_rng(2)
    base = rng.normal(0.619, 0.037, 20)
    treat = base + rng.normal(0.0, 0.005, 20)
    c = compare(make_record(treat), make_record(base))
    assert not c["significant"]


def test_per_seed_scores_averages_groups_within_a_run():
    rec = make_record([0.0])
    rec["runs"] = [
        {"run_idx": 0, "group": "a", "miou": 0.8},
        {"run_idx": 0, "group": "b", "miou": 0.6},
        {"run_idx": 1, "group": "a", "miou": 0.9},
        {"run_idx": 1, "group": "b", "miou": 0.7},
    ]
    assert per_seed_scores(rec) == {0: pytest.approx(0.7), 1: pytest.approx(0.8)}


def test_faithful_spec_covers_the_preprocessing_choices_that_move_numbers():
    for field in ("data.source_preference", "data.interp", "data.normalization",
                  "metric.assignment"):
        assert field in FAITHFUL_COMMON


# ── the evaluator's real record must satisfy the spec it is judged by ────────


def test_evaluator_record_carries_every_field_classify_checks():
    """Regression: the S7 gate passed while EVERY real bundle was `modified`.

    `make_record` above synthesizes `metric`, `data` and `model.family`, so the registry
    tests never noticed that `evaluator._aggregate` emitted none of them. Real records
    therefore deviated on 7 fields that had never varied, and — because `data.interp` is
    in COMPARABILITY_FIELDS — an INTER_LINEAR and an INTER_AREA run hashed to the same
    comparability key and could be pooled. This asserts the two agree.
    """
    from occl.evaluator import _aggregate
    from occl.registry import FAITHFUL_COMMON, classify, get_path

    class FakeCache:
        source_preference = "faithful"
        interp = "linear"

        def policy(self, dataset):
            from occl.imagecache import POLICIES
            return POLICIES[dataset]

    class FakeDS:
        name = "beetle"
        cache = FakeCache()

    class FakeAdapter:
        name = "sst_sam2"

        def config_report(self):
            return {"family": "sam2", "normalization": "imagenet",
                    "image_size": 1024, "num_maskmem": 7}

    runs = [{"group": "all", "run_idx": i, "miou": 0.6, "miou_model_space": 0.6,
             "support_ids": ["a"]} for i in range(2)]
    rec = _aggregate(FakeDS(), FakeAdapter(), 1, 2, ["all"], runs, [], "canonical")
    rec["model"] = FakeAdapter().config_report()

    # Fields whose faithful value IS None (e.g. `model.tta`) are satisfied by absence —
    # a plain run has no TTA, and demanding the key be present would be backwards.
    missing = [k for k, want in FAITHFUL_COMMON.items()
               if want is not None and get_path(rec, k) is None]
    assert not missing, f"evaluator omits fields classify checks: {missing}"

    dev, track = classify(rec)
    assert track == "faithful", f"a faithful run classified {track}: {dev}"


def test_resampling_change_breaks_comparability():
    """Two runs differing only in interpolation must never be pooled."""
    from occl.registry import comparability_key

    a = make_record([0.60] * 5)
    b = make_record([0.60] * 5)
    b["data"]["interp"] = "area"
    assert comparability_key(a) != comparability_key(b)


def test_metric_block_is_not_retyped_from_the_spec():
    """The record's metric block must come from metrics.METRIC_SPEC, not a copy."""
    from occl.metrics import METRIC_SPEC
    from occl.registry import FAITHFUL_COMMON

    for field, want in METRIC_SPEC.items():
        assert FAITHFUL_COMMON[f"metric.{field}"] == want, field


# ── run-splitting: parallel workers must be bit-identical to a sequential run ──


def _fake_run(group, run_idx):
    """Deterministic stand-in for run_one — value depends only on (group, run_idx)."""
    return {"group": group, "run_idx": run_idx,
            "miou": 0.5 + 0.001 * run_idx + 0.01 * len(group),
            "miou_model_space": 0.5 + 0.001 * run_idx,
            "support_ids": [f"s{run_idx}"]}


def test_run_split_aggregates_identically_to_a_full_sweep():
    """The property the whole fan-out rests on: splitting runs changes no number.

    Runs are independent by construction (`derive_seed` is a pure hash of
    (dataset, group, shot, run_idx); every episode does reset_state + del state), so a
    worker computing runs 10-19 must produce exactly what the full sweep produced for
    those indices. `_aggregate` indexes by real run_idx, so partial records line up.
    """
    from occl.evaluator import _aggregate

    class FakeCache:
        source_preference = "faithful"
        interp = "linear"

        def policy(self, dataset):
            from occl.imagecache import POLICIES
            return POLICIES[dataset]

    class FakeDS:
        name = "beetle"
        cache = FakeCache()

    class FakeAdapter:
        name = "sst_sam2"

        def config_report(self):
            return {"family": "sam2", "normalization": "imagenet"}

    groups = ["all"]
    full = [_fake_run(g, i) for g in groups for i in range(20)]
    whole = _aggregate(FakeDS(), FakeAdapter(), 1, 20, groups, full, [], "uniform")

    # Two "workers", disjoint run ranges, results concatenated by the merge pass.
    a = [_fake_run(g, i) for g in groups for i in range(0, 10)]
    b = [_fake_run(g, i) for g in groups for i in range(10, 20)]
    merged = _aggregate(FakeDS(), FakeAdapter(), 1, 20, groups, a + b, [], "uniform")

    assert merged["results"]["miou"] == whole["results"]["miou"]
    assert merged["results"]["std"] == whole["results"]["std"]
    assert merged["results"]["per_run"] == whole["results"]["per_run"]
    assert merged["protocol"]["n_runs"] == 20


def test_a_partial_worker_reports_only_its_own_runs():
    """A lone worker's bundle must not silently claim the full sweep.

    This is the failure mode fan-out introduces: if a worker dies, the merge must look
    short rather than average fewer runs and look fine.
    """
    from occl.evaluator import _aggregate

    class FakeCache:
        source_preference = "faithful"
        interp = "linear"

        def policy(self, dataset):
            from occl.imagecache import POLICIES
            return POLICIES[dataset]

    class FakeDS:
        name = "beetle"
        cache = FakeCache()

    class FakeAdapter:
        name = "sst_sam2"

        def config_report(self):
            return {"family": "sam2", "normalization": "imagenet"}

    partial = [_fake_run("all", i) for i in range(10, 20)]
    rec = _aggregate(FakeDS(), FakeAdapter(), 1, 20, ["all"], partial, [], "uniform")
    assert rec["protocol"]["n_runs"] == 10
    assert len(rec["results"]["per_run"]) == 10


def test_evaluate_rejects_an_invalid_run_range():
    """Overlapping or inverted ranges would double-count or silently drop runs."""
    import pytest

    from occl.evaluator import evaluate

    class FakeDS:
        name = "beetle"

        def groups(self):
            return ["all"]

    for start, end in ((5, 5), (10, 5), (-1, 5), (0, 21)):
        with pytest.raises(ValueError, match="run range"):
            evaluate(FakeDS(), None, shot=1, n_runs=20,
                     run_start=start, run_end=end)


def test_worker_assignment_produces_EXACTLY_the_requested_count():
    """The check the old coverage test was missing, and it cost an OOM.

    The previous `_assign` gave every group `ceil(workers/groups)` splits, so on
    butterfly's 5 groups a request for 6, 7, 8, 9 or 10 all produced **10** — and 7
    workers at 5.64 GB silently became 56 GB on a 39.5 GB card. Coverage was correct
    throughout, which is why the old test passed; only 1, 5 and 10 were ever exercised,
    the three values where the arithmetic is accidentally right.
    """
    def assign(groups, n_runs, n_workers):
        g = list(groups)
        n_workers = max(1, int(n_workers))
        if n_workers <= len(g):
            e = [round(i * len(g) / n_workers) for i in range(n_workers + 1)]
            return [(g[a:b], 0, n_runs) for a, b in zip(e, e[1:]) if b > a]
        base, rem = divmod(n_workers, len(g))
        out = []
        for k, grp in enumerate(g):
            sp = base + (1 if k < rem else 0)
            e = [round(i * n_runs / sp) for i in range(sp + 1)]
            out += [([grp], a, b) for a, b in zip(e, e[1:]) if b > a]
        return out

    for groups in (["all"], ["a", "b", "c", "d", "e"]):
        for w in range(1, 16):            # every width, not just the convenient ones
            work = assign(groups, 20, w)
            assert len(work) == w, f"{groups}: asked {w}, got {len(work)}"
            cov = {}
            for gs, a, b in work:
                for g in gs:
                    cov.setdefault(g, []).extend(range(a, b))
            assert set(cov) == set(groups), f"{groups} at {w}: missing {set(groups)-set(cov)}"
            for g, v in cov.items():
                assert sorted(v) == list(range(20)), f"{g} at {w}: {sorted(v)}"


def test_worker_assignment_partitions_runs_exactly_once():
    """Every (group, run) must be owned by exactly one worker — no gaps, no overlap.

    A gap silently shrinks the average; an overlap makes two workers race to write the
    same resume slug. Mirrors the `_assign` helper in notebooks/run_stage.py.
    """
    def assign(groups, n_runs, n_workers):
        per_group = max(1, -(-n_workers // len(groups)))
        out = []
        for g in groups:
            edges = [round(i * n_runs / per_group) for i in range(per_group + 1)]
            out += [([g], a, b) for a, b in zip(edges, edges[1:]) if b > a]
        return out

    for groups, n_workers in (
        (["all"], 1), (["all"], 4), (["all"], 10),          # beetle: run axis only
        (["a", "b", "c", "d", "e"], 5),                     # butterfly: one per group
        (["a", "b", "c", "d", "e"], 10),                    # both axes
        (["a", "b", "c", "d", "e"], 3),                     # fewer workers than groups
    ):
        work = assign(groups, 20, n_workers)
        for g in groups:
            covered = [i for gs, a, b in work if gs == [g] for i in range(a, b)]
            assert sorted(covered) == list(range(20)), (g, n_workers, sorted(covered))
            assert len(covered) == len(set(covered)), f"overlap for {g} at {n_workers}"


# ── cache format: the pixels the model sees ─────────────────────────────────


def test_lossless_cache_formats_round_trip_bit_exactly():
    """The entire claim of the lossless-cache change.

    SST decodes and resizes straight into the model with no intermediate encode. A cache
    is only a *memoisation* of that if the stored bytes decode back to the same array;
    otherwise it silently alters the input. JPEG q95 did not — measured 2.017/255.
    """
    import cv2
    import numpy as np

    from occl.imagecache import LOSSLESS_FORMATS
    from occl.imagecache.decode import encode

    rng = np.random.default_rng(0)
    # High-frequency noise is the adversarial case for any lossy codec.
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    for fmt in sorted(LOSSLESS_FORMATS):
        buf = encode(img, fmt, quality=101 if fmt == "webp" else 95)
        back = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        back = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)
        assert np.array_equal(back, img), f"{fmt} is not lossless"
    lossy = encode(img, "jpg", quality=95)
    back = cv2.cvtColor(cv2.imdecode(np.frombuffer(lossy, np.uint8), cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB)
    assert not np.array_equal(back, img), "jpg unexpectedly lossless — test is vacuous"


def test_butterfly_policy_is_lossless():
    """Pinned: butterfly was JPEG q95, which inserted an encode SST does not have."""
    from occl.imagecache import POLICIES

    for name, pol in POLICIES.items():
        assert pol.lossless, f"{name} cache format {pol.fmt!r} is lossy"
    assert POLICIES["butterfly_major"].size == 1024   # still mask-aligned


def test_cache_format_change_breaks_comparability():
    """A lossless-cache run must not pool with a lossy-cache one.

    `data.interp` and `data.source_preference` were already comparability fields; format
    was in neither the record nor the field list, so before this change the two would
    have hashed identically and been silently averaged together.
    """
    from occl.registry import comparability_key

    a = make_record([0.70] * 5)
    b = make_record([0.70] * 5)
    b["data"]["cache_format"] = "jpg"
    assert comparability_key(a) != comparability_key(b)


def test_lossy_cache_classifies_as_modified():
    from occl.registry import classify

    rec = make_record([0.70] * 5)
    rec["data"]["cache_format"] = "jpg"
    dev, track = classify(rec)
    assert "data.cache_format" in dev and track == "modified"
    for fmt in ("png", "webp"):
        rec["data"]["cache_format"] = fmt
        assert "data.cache_format" not in classify(rec)[0], fmt


def test_resume_fingerprint_responds_to_the_pixel_pipeline_and_code():
    """Guards the failure mode that has already produced one wrong conclusion here.

    The resume slug was (dataset, group, shot, selection, adapter, run_idx), so any other
    change — a different cache format, dtype, or an edit to the metric — silently served
    stale per-run JSON. With the lossless-cache switch that would have made the re-run a
    no-op returning the JPEG-era numbers, which reads as a confirmation.
    """
    from occl.evaluator import run_fingerprint
    from occl.imagecache import POLICIES, CachePolicy, ImageCache

    class DS:
        name = "butterfly_major"

        def __init__(self, cache):
            self.cache = cache

    class Ad:
        name = "sst_sam2"
        dtype = "torch.bfloat16"

    cache = ImageCache("/tmp/_fp_check")
    base = run_fingerprint(DS(cache), Ad())

    original = POLICIES["butterfly_major"]
    try:
        POLICIES["butterfly_major"] = CachePolicy(size=1024, fmt="jpg", quality=95)
        assert run_fingerprint(DS(cache), Ad()) != base, "cache format must change it"
    finally:
        POLICIES["butterfly_major"] = original

    class Ad16(Ad):
        dtype = "torch.float16"

    assert run_fingerprint(DS(cache), Ad16()) != base, "dtype must change it"
    assert run_fingerprint(DS(cache), Ad()) == base, "and it must be stable otherwise"


def test_every_dispatchable_stage_is_listed_in_the_notebook_menu():
    """The menu is how a stage is discovered — an unlisted one is effectively missing.

    s5c and s6b were fully wired (config entry + dispatch) but absent from the markdown
    table, so they could not be found without reading the source. Keep the two in sync.
    """
    import json
    import re
    from pathlib import Path

    nb = json.loads((Path(__file__).parent.parent / "notebooks/run_stage.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    menu = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")

    dispatch = set(re.findall(r'"(s\w+)"', re.search(r"elif STAGE in \(([^)]*)\):", src).group(1)))
    dispatch |= set(re.findall(r'STAGE == "(s\w+)"', src))
    dispatch |= {"s2"}

    missing = sorted(s for s in dispatch if f"`{s}`" not in menu)
    assert not missing, f"dispatchable but not in the notebook's stage table: {missing}"


def test_fanout_passes_extra_flags_to_workers_and_merge():
    """`--tta` was dropped in the parallel path, so s5c ran as plain s5b — twice.

    The serial branch included `extra`; the worker launch and the merge did not. The
    adapter therefore stayed `sst_sam2`, the resume slug matched s5b's, every run was
    served from cache (0.96 GB peak VRAM, zero inference), and the bundle reported
    numbers bit-identical to the arm it was meant to differ from.
    """
    import json
    import re
    from pathlib import Path

    nb = json.loads((Path(__file__).parent.parent / "notebooks/run_stage.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    body = src[src.index("def fanout("):src.index("BUNDLE = os.path.join")]

    # Check the invariant, not one spelling of it. This asserted `list(extra)` appeared
    # within 400 chars AFTER `_sp.Popen(`, which broke the moment the command was hoisted
    # into a `wcmd` variable so it could be echoed to the log — a cosmetic change that made
    # a correctness test fail while the correctness was untouched.
    launch = body[body.index("for i, (g, a, b) in enumerate(work)"):body.index("wait_all(")]
    assert "list(extra)" in launch, "worker launch must forward `extra`"
    assert "_sp.Popen(" in launch, "the worker launch moved out of this region"
    merge = body[body.index('"--no-precompute"') - 300:]
    assert "list(extra)" in merge[:400], "merge pass must forward `extra`"
    # and the run must be checked against the arm it claimed
    assert "did not reach the workers" in body, "fan-out must verify the arm took effect"


def test_tta_counts_survive_the_merge_pass():
    """The merge does ZERO inference, so the adapter's live counter is 0 there.

    Asserting on `model.tta_episodes` therefore failed a run whose 8 workers had all
    succeeded. The counts have to be recorded per RUN, so they travel in the per-run JSON
    that the merge reads back.
    """
    from occl.evaluator import _aggregate

    class FakeCache:
        source_preference = "faithful"
        interp = "linear"

        def policy(self, dataset):
            from occl.imagecache import POLICIES
            return POLICIES[dataset]

    class FakeDS:
        name = "beetle"
        cache = FakeCache()

    class FakeAdapter:
        name = "sst_sam2+tta_rot180"

        def config_report(self):
            return {"family": "sam2", "normalization": "imagenet"}

    runs = [{"group": "all", "run_idx": i, "miou": 0.6, "miou_model_space": 0.6,
             "support_ids": ["a"], "tta": {"episodes": 60, "rotated_chosen": 22}}
            for i in range(20)]
    rec = _aggregate(FakeDS(), FakeAdapter(), 1, 20, ["all"], runs, [], "uniform")
    assert rec["results"]["tta"] == {"episodes": 1200, "rotated_chosen": 440}

    # and a non-TTA run must not sprout an empty tta block
    plain = [{k: v for k, v in r.items() if k != "tta"} for r in runs]
    assert _aggregate(FakeDS(), FakeAdapter(), 1, 20, ["all"], plain, [],
                      "uniform")["results"]["tta"] is None


def test_test_time_augmentation_is_not_faithful():
    """An intervention must not be able to launder itself as a reproduction.

    S5c lifted Major +4.88 and classified as `faithful` because nothing in the spec
    looked at the adapter. SST has no orientation handling at all.
    """
    rec = make_record([0.75] * 5)
    assert classify(rec)[1] == "faithful"
    rec["model"]["tta"] = "rot180_by_confidence"
    dev, track = classify(rec)
    assert track == "modified" and "model.tta" in dev


def test_tta_and_plain_runs_are_not_comparable():
    a = make_record([0.70] * 5)
    b = make_record([0.75] * 5)
    b["model"]["tta"] = "rot180_by_confidence"
    assert comparability_key(a) != comparability_key(b)
    with pytest.raises(ValueError, match="comparability"):
        compare(b, a)


def test_scnp_is_our_modification_and_deviates():
    """SCNP (S9a) appears nowhere in the paper; a record training it must classify
    modified on exactly that field, while every pre-SCNP record (no `scnp` key at all)
    keeps its classification."""
    rec = make_record([0.65], finetune=PAPER_FT)
    rec["occl"]["scnp"] = {"width": 3, "joint": True}
    assert classify(rec)[0] == ["occl.scnp"]
    rec["occl"]["scnp"] = None
    assert classify(rec)[1] == "faithful"


def test_s13_knobs_are_our_modifications_and_each_deviates_on_its_own_field():
    """The four S13 loss knobs mirror the occl.scnp contract exactly: any non-null value
    classifies the run `modified` on precisely that field, and a record without the key
    (every pre-S13 record) keeps its classification untouched."""
    values = {
        "occl.excl": {"mode": "zerocat"},
        "occl.xovl": {"weight": 1.0, "where": "query"},
        "occl.bshape": {"weight": 1.0, "schedule": "const", "dead_zone": 1,
                        "normalize": "chan_max"},
        "occl.margin": {"m": 1.0, "weight": None, "only": True,
                        "statistic": "hard_claim_mean_top_logit", "form": "hinge",
                        "consistent": "as_given", "rule": "|SAM2_OUT_BIAS|/10"},
    }
    for path, value in values.items():
        key = path.split(".", 1)[1]
        rec = make_record([0.65], finetune=PAPER_FT)
        rec["occl"][key] = value
        assert classify(rec)[0] == [path], path
        rec["occl"][key] = None
        assert classify(rec)[1] == "faithful", path
        clean = make_record([0.65], finetune=PAPER_FT)
        assert key not in clean["occl"]
        assert classify(clean)[1] == "faithful", path


def test_cross_family_compare_is_explicit_recorded_and_hash_pinned():
    """S14: `ignore_fields` is the ONLY way to pair across model families. The draws
    are model-independent (same seeds -> same support_set_hash), which is what makes
    the paired delta valid — so the hash must match, the relaxation is whitelisted to
    the two family fields, and the result carries what was ignored with both values."""
    a = make_record([0.70, 0.71, 0.72])
    b = make_record([0.65, 0.66, 0.67])
    b["model"] = {"family": "sam3", "image_size": 1008, "num_maskmem": 7}

    with pytest.raises(ValueError, match="comparability"):
        compare(a, b)

    cmp = compare(a, b, ignore_fields=("model.family", "model.image_size"))
    assert cmp["n_seeds"] == 3
    assert cmp["mean_delta"] == pytest.approx(0.05)
    assert cmp["ignored_fields"] == {
        "model.family": ["sam2", "sam3"],
        "model.image_size": [1024, 1008],
    }

    # a within-family compare must not carry the key's content
    same = compare(make_record([0.70] * 3), make_record([0.65] * 3))
    assert same["ignored_fields"] is None

    # the whitelist: protocol/metric/data fields change WHAT was measured
    with pytest.raises(ValueError, match="only relax"):
        compare(a, b, ignore_fields=("model.family", "protocol.shots"))

    # and the hash pin: different draws cannot pair, family fields ignored or not
    c = make_record([0.65, 0.66, 0.67], support_hash="zzz999")
    c["model"] = {"family": "sam3", "image_size": 1008, "num_maskmem": 7}
    with pytest.raises(ValueError, match="support"):
        compare(a, c, ignore_fields=("model.family", "model.image_size"))


def test_new_family_records_classify_modified_on_exactly_the_family_fields():
    rec = make_record([0.65] * 3)
    rec["model"] = {"family": "sam3", "image_size": 1008, "num_maskmem": 7}
    rec["data"]["normalization"] = "half"
    dev, track = classify(rec)
    assert track == "modified"
    assert set(dev) == {"model.family", "model.image_size", "data.normalization"}


def test_tta_toggle_pairing_is_whitelisted_and_recorded():
    """S14 pairing 2 ("TTA's value on the new family") compares base_tta vs base —
    same episodes, same support hash, same weights, different inference pipeline.
    model.tta lives in the comparability key, so the toggle must be DECLARED via
    ignore_fields (whitelisted with the family fields — it changes which pipeline
    measured, not what was measured) and travels in the result. The bare compare
    still refuses: the first butterfly combine died on exactly that refusal."""
    a = make_record([0.74, 0.75, 0.76])
    a["model"]["tta"] = "rot180_by_confidence"
    b = make_record([0.70, 0.71, 0.72])

    with pytest.raises(ValueError, match="comparability"):
        compare(a, b)

    cmp = compare(a, b, ignore_fields=("model.tta",))
    assert cmp["n_seeds"] == 3
    assert cmp["mean_delta"] == pytest.approx(0.04)
    assert cmp["ignored_fields"] == {
        "model.tta": ["rot180_by_confidence", None],
    }


def _with_draws(rec, draws):
    """Attach per-run records carrying the support draw, the way the evaluator does."""
    rec["runs"] = [{"group": "all", "run_idx": i, "miou": m, "support_ids": d}
                   for i, (m, d) in enumerate(zip(rec["results"]["per_run"], draws))]
    return rec


def test_nested_shot_pairing_demands_prefix_evidence_and_records_it():
    """S16 pairs 2-shot against 1-shot (+TTA) per seed. That crosses protocol.shots AND
    support_set_hash — both comparability fields, both legitimately different — so the
    bare compare refuses, ignore_fields refuses (protocol fields are never whitelisted),
    and the ONLY route is `nested_shots=True`, which verifies from the per-run draws that
    run i's 1-shot support is the first of its 2-shot pair before masking anything, and
    embeds that proof in the result."""
    b = _with_draws(make_record([0.60, 0.62, 0.64], shots=1, support_hash="h1"),
                    [["a"], ["b"], ["c"]])
    t = _with_draws(make_record([0.65, 0.66, 0.70], shots=2, support_hash="h2"),
                    [["a", "x"], ["b", "y"], ["c", "z"]])

    with pytest.raises(ValueError, match="comparability"):
        compare(t, b)
    with pytest.raises(ValueError, match="only relax"):
        compare(t, b, ignore_fields=("protocol.shots",))

    cmp = compare(t, b, nested_shots=True)
    assert cmp["n_seeds"] == 3
    assert cmp["mean_delta"] == pytest.approx(0.05)
    assert cmp["nested_shots"] == {"treatment_shots": 2, "baseline_shots": 1,
                                   "runs_checked": 3}
    assert cmp["ignored_fields"] == {"protocol.shots": [2, 1],
                                     "protocol.support_set_hash": ["h2", "h1"]}

    # composes with the TTA toggle: 2-shot plain vs 1-shot TTA in one call
    b_tta = _with_draws(make_record([0.63, 0.64, 0.66], shots=1, support_hash="h1"),
                        [["a"], ["b"], ["c"]])
    b_tta["model"]["tta"] = "rot180_by_confidence"
    both = compare(t, b_tta, nested_shots=True, ignore_fields=("model.tta",))
    assert both["mean_delta"] == pytest.approx(0.02 + 0.02 / 3, abs=1e-9)
    assert set(both["ignored_fields"]) == {"model.tta", "protocol.shots",
                                           "protocol.support_set_hash"}

    # a same-protocol compare carries no evidence and no relaxation
    plain = compare(make_record([0.70] * 3), make_record([0.65] * 3))
    assert plain["nested_shots"] is None and plain["ignored_fields"] is None


def test_nested_shot_pairing_refuses_without_the_proof():
    b = _with_draws(make_record([0.60, 0.62, 0.64], shots=1, support_hash="h1"),
                    [["a"], ["b"], ["c"]])

    # not a prefix: the second draw came from a different pool or a different seed
    t_other = _with_draws(make_record([0.65, 0.66, 0.70], shots=2, support_hash="h2"),
                          [["a", "x"], ["q", "b"], ["c", "z"]])
    with pytest.raises(ValueError, match="prefix-nested"):
        compare(t_other, b, nested_shots=True)

    # wrong direction: the treatment must be the wider draw
    t_ok = _with_draws(make_record([0.65, 0.66, 0.70], shots=2, support_hash="h2"),
                       [["a", "x"], ["b", "y"], ["c", "z"]])
    with pytest.raises(ValueError, match="treatment shots > baseline"):
        compare(b, t_ok, nested_shots=True)

    # no per-run draws: a summary-only record cannot prove anything
    with pytest.raises(ValueError, match="support_ids"):
        compare(make_record([0.65, 0.66, 0.70], shots=2, support_hash="h2"), b,
                nested_shots=True)

    # a record whose draw length disagrees with its own protocol.shots is malformed
    t_short = _with_draws(make_record([0.65, 0.66, 0.70], shots=2, support_hash="h2"),
                          [["a"], ["b"], ["c"]])
    with pytest.raises(ValueError, match="prefix-nested"):
        compare(t_short, b, nested_shots=True)


# ── the established SAM 2 fine-tune (S17d) ──────────────────────────────────


def test_sam2_crown_recipe_expands_in_front_and_explicit_flags_win():
    """`--recipe sam2_crown` is the S17d configuration exactly; it is spliced in FRONT so a
    later explicit flag overrides it; unknown names refuse; no recipe = untouched."""
    import argparse
    import ast
    from pathlib import Path

    from occl.registry import RECIPES, SAM2_CROWN_FLAGS, expand_recipe

    assert RECIPES == {"sam2_crown": SAM2_CROWN_FLAGS}
    assert SAM2_CROWN_FLAGS == ("--family", "sam2", "--method", "lora", "--lr", "1e-05",
                                "--epochs", "1", "--mask-mode", "per_trait",
                                "--lora-legs", "forward", "--objness", "--objness-weight", "10",
                                "--excl-mode", "zerocat", "--scope-preset", "sam2_arch")
    argv = ["--dataset", "beetle", "--recipe", "sam2_crown", "--objness-weight", "3"]
    out = expand_recipe(argv)
    assert out[:len(SAM2_CROWN_FLAGS)] == list(SAM2_CROWN_FLAGS) and out[len(SAM2_CROWN_FLAGS):] == argv
    assert expand_recipe(["--dataset", "beetle"]) == ["--dataset", "beetle"]
    assert expand_recipe(["--recipe=sam2_crown"])[:2] == ["--family", "sam2"]
    with pytest.raises(ValueError, match="unknown --recipe"):
        expand_recipe(["--recipe", "nope"])
    # every recipe flag is one run_finetune.py accepts, and the parser keeps the LAST
    # value of a repeated flag (the override contract)
    src = (Path(__file__).resolve().parent.parent / "scripts" / "run_finetune.py").read_text()
    accepted = {n.args[0].value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant)}
    assert {f for f in SAM2_CROWN_FLAGS if f.startswith("--")} | {"--recipe"} <= accepted
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--objness", action="store_true")
    ap.add_argument("--objness-weight", type=float, default=1.0)
    ap.add_argument("--recipe", default=None)
    ns, _ = ap.parse_known_args(out)
    assert ns.objness and ns.objness_weight == 3.0 and ns.recipe == "sam2_crown"


def test_sam2_crown_classifier_reads_the_fields_not_the_label():
    """The S17d candidate record IS the crown; S17c's `comp_arch` (weight 1), `comp_w10`
    (S11 scopes), the soft arm and the carrier each deviate on exactly their axis; a
    record that merely CLAIMS the recipe is not it; and the crown is `modified` on the
    paper's track by construction."""
    import json
    import zipfile
    from pathlib import Path

    from occl.registry import SAM2_CROWN_FINETUNE, is_sam2_crown_finetune

    root = Path(__file__).resolve().parent.parent / "results"
    want = {"s17dft_sam2_beetle_comp_arch_w10.zip": [],
            "s17cft_sam2_beetle_comp_arch.zip": ["occl.objness"],
            "s17cft_sam2_beetle_comp_w10.zip": ["finetune.lora.scopes"],
            "s17dft_sam2_beetle_comp_arch_w10_soft.zip": ["occl.soft_sentinel"],
            "s17ft_sam2_beetle_carrier.zip": ["finetune.lora.scopes", "occl.excl",
                                              "occl.lora_legs", "occl.objness"]}
    if not all((root / n).exists() for n in want):
        pytest.skip("the S17/S17c/S17d fine-tune bundles are not filed here")
    for name, dev in want.items():
        with zipfile.ZipFile(root / name) as z:
            rec = json.loads(z.read("record.json"))
        got, ok = is_sam2_crown_finetune(rec)
        assert got == dev and ok == (not dev), (name, got)
        assert classify(rec)[1] == "modified"
    # a synthetic record that satisfies every field is the crown; flipping one is not
    rec = {}
    for k, v in SAM2_CROWN_FINETUNE.items():
        d = rec
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    assert is_sam2_crown_finetune(rec) == ([], True)
    rec["occl"]["recipe"] = "sam2_crown"
    rec["occl"]["lora_legs"] = None                      # the label lies
    assert is_sam2_crown_finetune(rec) == (["occl.lora_legs"], False)
