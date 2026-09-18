"""The evaluation driver: runs the protocol, scores it, and survives interruption.

Aggregation follows the paper's shape. For Butterfly Major each subspecies is evaluated
independently (their trait vocabularies differ, 6-10 classes, so a support from one
cannot label another's traits), then:

    run_score[group][r] -> mean over that subspecies' test images
    run_score[r]        -> mean over the 5 subspecies, *within* run r
    reported            -> mean +- std(ddof=1) across the 20 run scores

All five subspecies have exactly 100 test images, so equal-weighting subspecies and
pooling all 500 give the same mean; they differ in the std, and the across-run std is the
one the paper reports. `ddof=1` because 20 runs is a sample, and the paper's +- values are
dispersion across runs rather than standard errors.

Beetle has a single universal group: one support, drawn from the pooled 120, must segment
beetles of any of the 12 species. That is what makes 61.9 so much harder than 81.0, and
support sampling is deliberately *not* stratified by species.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .metrics import (ASSIGNMENT_RULES, METRIC_SPEC, aggregate_run,
                      assignment_variants, boundary_variants, score_episode)
from .protocol import build_episodes, build_run
from .seeding import derive_seed, support_set_hash

IOU_SPACES = ("gt_native", "model")


def _score_both_spaces(logits, class_ids, gt) -> dict[str, dict]:
    """Score in both IoU spaces.

    When GT already matches the model resolution -- true for every butterfly mask, which
    is natively 1024x1024 -- the two are identical by construction, so the second is
    copied rather than recomputed. Beetle GT is 172-431 px, where they genuinely differ:
    upsampling a 217-row GT to 1024 with NEAREST manufactures duplicated boundary pixels
    and inflates IoU on thin parts like antennae. `gt_native` is the defensible default;
    `model` is recorded because the SST repo measures that way and the paper's 61.9 came
    from their code.
    """
    native = score_episode(logits, class_ids, gt, iou_space="gt_native")
    if gt.shape == logits.shape[-2:]:
        return {"gt_native": native, "model": dict(native, iou_space="model")}
    return {"gt_native": native,
            "model": score_episode(logits, class_ids, gt, iou_space="model")}


def run_one(ds, adapter, group: str, shot: int, run_idx: int, progress_every: int = 0,
            support_selection: str = "uniform") -> dict:
    """One (group, shot, run): every test image against one support set."""
    support_ids, *_, pool_stats = build_run(ds, group, shot, run_idx, support_selection)
    per_space: dict[str, list[dict]] = {s: [] for s in IOU_SPACES}
    rows: list[dict] = []
    # Every episode is also scored under the other readings of the overlap rule
    # (`metrics.assignment_variants`) — most importantly `independent`, which is what
    # SST's own `per_trait_iou` computes (binarise each trait at logit 0, overlap
    # permitted; the paper never states its rule). Recorded per run as a DIAGNOSTIC:
    # `metric.assignment` stays `argmax_label` and no comparability field changes, so
    # this cannot fork the record space. Until now the `independent` figure existed only
    # via `run_diagnostic.py`, which re-runs inference and is hardcoded to butterfly —
    # meaning no fine-tuned arm ever had it. The logits are already in hand here, so the
    # cost is CPU-only. This edit is the reason the resume fingerprint changed (this file
    # is in `_FINGERPRINTED`); it was landed alone so exactly one re-run pays for it.
    variants: list[dict] = []
    boundary: list[dict] = []

    # TTA counters are cumulative on the adapter; snapshot so each run records its own
    # delta. Without this the statistic lives only in the worker's bundle and vanishes at
    # the merge, which does zero inference and would report 0 episodes.
    _tta0 = (getattr(adapter, "n_total", None), getattr(adapter, "n_rotated_chosen", None))

    t0 = time.time()
    for n, ep in enumerate(
            build_episodes(ds, group, shot, run_idx, support_selection), start=1):
        logits = adapter.predict(ep)
        gt = ds.load_labelmap(ep.query_id)
        scored = _score_both_spaces(logits, ep.class_ids, gt)
        variants.append(assignment_variants(logits, ep.class_ids, gt))
        # The paper's unpublished best-of-3 boundary tolerance (author correspondence,
        # post-S11) — recorded per run beside the plain metric, same diagnostic-only
        # contract as `miou_variants`. Second deliberate fingerprint break; landed
        # alone like the first.
        boundary.append(boundary_variants(logits, ep.class_ids, gt))
        for space in IOU_SPACES:
            per_space[space].append(scored[space])
        for trait, iou in scored["gt_native"]["per_trait"].items():
            rows.append({"group": group, "shot": shot, "run_idx": run_idx,
                         "query_id": ep.query_id, "trait_id": int(trait),
                         "iou": round(float(iou), 6)})
        if progress_every and n % progress_every == 0:
            print(f"    {group} run {run_idx}: {n} images "
                  f"({(time.time() - t0) / n:.2f}s/img)", flush=True)

    elapsed = time.time() - t0
    agg = {s: aggregate_run(per_space[s]) for s in IOU_SPACES}
    miou_variants = {}
    for rule in ASSIGNMENT_RULES:
        vals = [v[rule] for v in variants if v[rule] is not None]
        miou_variants[rule] = float(np.mean(vals)) if vals else None
    miou_boundary = {}
    for rule in ("argmax_best3", "independent_best3"):
        vals = [b[rule] for b in boundary if b[rule] is not None]
        miou_boundary[rule] = float(np.mean(vals)) if vals else None
    tta = None
    if _tta0[0] is not None:
        tta = {"episodes": adapter.n_total - _tta0[0],
               "rotated_chosen": adapter.n_rotated_chosen - _tta0[1]}
    return {
        "group": group, "shot": shot, "run_idx": run_idx,
        "seed": derive_seed(ds.name, group, 0, run_idx),
        "support_ids": list(support_ids),
        "miou": agg["gt_native"]["miou"],
        "miou_model_space": agg["model"]["miou"],
        "iou_binary": agg["gt_native"]["iou_binary"],
        # `argmax` doubles as a self-check against `miou` (same rule, same space);
        # `independent` is the SST-release reading; `oracle` is the assignment ceiling.
        "miou_variants": miou_variants,
        # The unpublished best-of-3 boundary tolerance, under both overlap rules;
        # `independent_best3` is the closest reading of the paper's full metric.
        "miou_boundary": miou_boundary,
        "per_trait_mean": agg["gt_native"]["per_trait_mean"],
        "n_images": agg["gt_native"]["n_images"],
        "n_skipped": agg["gt_native"]["n_skipped"],
        "elapsed_s": round(elapsed, 1),
        "sec_per_image": round(elapsed / max(agg["gt_native"]["n_images"], 1), 3),
        "support_selection": support_selection,
        "pool_size": pool_stats.get("pool_size"),
        "tta": tta,
        "_rows": rows,
    }


def distinct_images(ds, group: str, shot: int, n_runs: int,
                    support_selection: str = "uniform",
                    run_start: int = 0, run_end: int | None = None) -> list[str]:
    """Every image a (group, shot, run-range) sweep will touch, each listed once.

    For Butterfly Major that is 100 test + at most 100 distinct supports per subspecies,
    against 20 x 100 x 2 = 4,000 encoder passes if nothing is cached.

    ``run_start``/``run_end`` narrow it to one parallel worker's slice, so each worker
    encodes only the supports *it* draws. Note the test images are **not** reducible --
    every run scores every test image -- so a 1/10 run slice cuts butterfly's cache from
    1.59 GB to 0.90 GB, not to 0.16 GB. The test set is the floor, and it is duplicated
    across workers; that is the structural cost of process-based fan-out.
    """
    from .curation import draw_supports, support_pool

    keys = set(ds.test_ids(group))
    pool, _ = support_pool(ds, group, support_selection)
    for r in range(run_start, n_runs if run_end is None else run_end):
        # Must be the SAME dispatch protocol.build_run uses. This function drives both
        # the cache preflight and `precompute`, so a mismatch reports the wrong images as
        # missing AND warms the wrong embeddings — it is not a cosmetic duplication.
        keys.update(draw_supports(ds, group, pool, support_selection, shot, r))
    return sorted(keys)


#: Modules whose source determines the numbers a run produces. Any edit to these must
#: invalidate cached per-run JSON. The ACTIVE adapter's own source file is appended
#: dynamically (`_adapter_source`) — see its docstring.
_FINGERPRINTED = ("metrics.py", "protocol.py", "curation.py", "seeding.py",
                  "evaluator.py")


def _adapter_source(adapter):
    """The file defining the (inner) adapter — the sixth fingerprinted module.

    Parameterized per adapter (S14): the old hardcoded ``adapters/sst_sam2.py`` entry
    meant a NEW family's adapter edits would never invalidate its cached per-run JSON
    — the exact stale-resume failure this fingerprint exists to prevent — while
    adding every adapter file to the shared list would orphan all banked SAM 2 resume
    JSON whenever a sibling was edited. The wrapped ``inner`` decides (TTA composes;
    its own source stays un-fingerprinted, as before). For `SSTSam2Adapter` this
    resolves to `adapters/sst_sam2.py` in the same list position as before, so every
    existing SAM 2 fingerprint is byte-identical. Known cost: copy-baseline runs now
    hash `copy_baseline.py` instead — their cached per-run JSON is orphaned; those
    numbers are banked in bundles/CSVs and re-derive in minutes.
    """
    import inspect

    inner = getattr(adapter, "inner", adapter)
    try:
        f = inspect.getsourcefile(type(inner))
    except TypeError:
        return None
    return Path(f) if f else None


def run_fingerprint(ds, adapter) -> str:
    """Short hash of everything that changes a run's numbers but is not already in the slug.

    The resume key used to be (dataset, group, shot, selection, adapter, run_idx). That
    silently served stale results whenever anything *else* changed — it has already
    produced one wrong conclusion here, an "A100 vs L4" comparison that had in fact
    resumed cached L4 results and computed nothing.

    It became acute with the lossless-cache change: the butterfly per-run JSON on Drive
    was computed against a JPEG cache, so re-running under the same stage name would skip
    every run and hand back the old numbers — a no-op that reads as a confirmation.

    Covers the pixel pipeline (cache format, interpolation, source preference), the
    compute dtype, the source of the modules that decide the result, and **the model
    weights**. Hashing source is deliberately conservative: a comment-only edit invalidates
    the cache, which costs a re-run, while the alternative costs a wrong conclusion.

    `adapter.weights_id` is what makes S8c possible at all. Its three arms — base,
    LoRA@1e-5, LoRA@1e-4 — share dataset, protocol, dtype, cache and module source, so
    without it all three hash identically: arms 2 and 3 would resume arm 1's cached per-run
    JSON, report the baseline's numbers three times, and yield a paired delta of exactly
    0.000. A null result that is indistinguishable from a correct one is the worst failure
    this file can produce, and only the weights distinguish those runs.
    """
    cache = getattr(ds, "cache", None)
    parts = [
        getattr(cache, "source_preference", None),
        getattr(cache, "interp", None),
        cache.policy(ds.name).fmt if hasattr(cache, "policy") else None,
        str(getattr(adapter, "dtype", None)),
        getattr(adapter, "weights_id", None),
    ]
    src = Path(__file__).parent
    files = [src / rel for rel in _FINGERPRINTED] + [_adapter_source(adapter)]
    for p in files:
        parts.append(hashlib.sha1(p.read_bytes()).hexdigest()
                     if p is not None and p.exists() else "?")
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8]


def evaluate(ds, adapter, shot: int = 1, n_runs: int = 20, groups=None,
             resume_dir: Path | None = None, progress: bool = True,
             precompute: bool = True, support_selection: str = "uniform",
             run_start: int = 0, run_end: int | None = None,
             keep_embeddings: bool = False) -> dict:
    """Run the full protocol, resuming any runs already on disk.

    Per-run JSON is written the moment a run finishes, so a 12-hour Colab session
    boundary in the middle of Butterfly Major's 10,000 episodes costs one run, not all of
    them. The skip key is (dataset, group, shot, adapter, run_idx).

    ``run_start``/``run_end`` restrict this call to a slice of the runs, which is what
    makes parallel workers possible. Runs are independent by construction -- `derive_seed`
    is a pure hash of ``(dataset, group, shot, run_idx)`` with no global RNG, and every
    episode calls ``reset_state`` then ``del state`` -- so a slice computes exactly what
    the full sweep would have computed for those indices, bit for bit. Workers pointed at
    a shared ``resume_dir`` with disjoint (group, run-range) assignments never collide,
    because the slug carries both.

    ``n_runs`` stays the **total**, not the slice length: aggregation indexes runs by
    their real ``run_idx``, so a worker's partial record still lines up with the others
    when the merge pass re-reads them all.

    ``keep_embeddings`` retains the embedding cache instead of freeing it per group. It
    exists for S8c, where several arms differ **only** in LoRA weights: `apply_lora` scopes
    to `memory_encoder` and `sam_mask_decoder`, so the image encoder is never touched and
    every arm's `backbone_fpn` features are identical. Encoding is roughly 95% of the
    compute, so re-deriving it per arm would triple the stage for nothing. Default False,
    because holding a whole dataset's embeddings is what the per-group free exists to
    prevent -- only a caller that knows the arms share an encoder should ask for this.
    """
    groups = list(groups or ds.groups())
    run_end = n_runs if run_end is None else run_end
    if not 0 <= run_start < run_end <= n_runs:
        raise ValueError(
            f"invalid run range [{run_start}, {run_end}) against n_runs={n_runs}"
        )
    resume_dir = Path(resume_dir) if resume_dir else None
    if resume_dir:
        resume_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    all_rows: list[dict] = []

    # Record the curation parameters per group so the approximation is auditable after
    # the fact — especially `threshold_deg`, which must never be retuned to hit a target.
    from .curation import support_pool

    curation = {g: support_pool(ds, g, support_selection)[1] for g in groups}

    fingerprint = run_fingerprint(ds, adapter)
    if progress:
        print(f"  resume fingerprint {fingerprint} "
              "(pixel pipeline + dtype + module source)", flush=True)

    cache_reports = {}
    for group in groups:
        # Encode this group's distinct images once, then clear before the next group so
        # peak VRAM tracks one group (~1.7 GB) rather than the whole dataset (~8 GB).
        #
        # NOTE this runs even when every run in the slice is already banked, so a resumed
        # pass pays the encode to then skip every run. Tempting to guard — but this file is
        # in `_FINGERPRINTED`, so ANY edit here changes the resume fingerprint and orphans
        # every per-run JSON ever written. The saving is a minute on beetle; the cost is
        # recomputing the entire project. Skip the pass at the CALLER instead (the s8d
        # stage decouples which passes compute from which merge), or use --no-precompute.
        if precompute and hasattr(adapter, "precompute"):
            keys = distinct_images(ds, group, shot, n_runs, support_selection,
                                   run_start, run_end)
            if progress:
                print(f"  precomputing {len(keys)} embeddings for {group}"
                      f"[{run_start}:{run_end}]...", flush=True)
            rep = adapter.precompute((k, ds.load_image(k)) for k in keys)
            cache_reports[group] = rep
            if progress:
                print(f"    {rep['n_cached']} cached, {rep['total_gb']} GB", flush=True)

        for run_idx in range(run_start, run_end):
            slug = f"{ds.name}__{group.replace(' ', '_').replace('/', '_')}" \
                   f"__shot{shot}__{support_selection}__{adapter.name}" \
                   f"__{fingerprint}__run{run_idx}.json"
            cached = (resume_dir / slug) if resume_dir else None

            if cached and cached.exists():
                rec = json.loads(cached.read_text())
                if progress:
                    print(f"  [skip] {group} run {run_idx}: "
                          f"mIoU {100 * rec['miou']:.2f}", flush=True)
            else:
                rec = run_one(ds, adapter, group, shot, run_idx,
                              support_selection=support_selection)
                if cached:  # written with _rows, so a resumed run can still pair
                    cached.write_text(json.dumps(rec, indent=1))
                if progress:
                    print(f"  {group} run {run_idx}: mIoU {100 * rec['miou']:.2f} "
                          f"({rec['n_images']} imgs, {rec['sec_per_image']}s/img)", flush=True)

            all_rows.extend(rec.pop("_rows", []))
            runs.append(rec)

        if precompute and not keep_embeddings and hasattr(adapter, "clear_embeddings"):
            adapter.clear_embeddings()

    out = _aggregate(ds, adapter, shot, n_runs, groups, runs, all_rows,
                     support_selection)
    out["embedding_cache"] = cache_reports
    out["curation"] = curation
    return out


def _aggregate(ds, adapter, shot, n_runs, groups, runs, rows,
               support_selection: str = "uniform") -> dict:
    by_group: dict[str, dict[int, dict]] = {g: {} for g in groups}
    for r in runs:
        by_group[r["group"]][r["run_idx"]] = r

    per_group = {}
    for g in groups:
        vals = [by_group[g][i]["miou"] for i in sorted(by_group[g])]
        # `iou_binary` is the silhouette score — the merged foreground, ignoring which
        # trait each pixel was assigned to. It is already computed per episode and carried
        # per run; aggregating it here makes every arm report its own headroom.
        #
        # That matters for deciding when to STOP stacking techniques. Assignment-side work
        # (curation, orientation handling, a better loss) can only move mIoU toward the
        # silhouette; once an arm sits at it, further technique buys nothing and the
        # remaining gap is mask quality, which is a model-capability question. Reading that
        # off each run beats re-deriving it from a diagnostic months later.
        #
        # NOT a hard bound: mIoU weights every trait equally while this is pixel-weighted
        # over the merged foreground, so a ragged outer boundary with well-segmented small
        # traits scores above it — lativitta does. Strong correlate (r = 0.86), not a cap.
        sil = [by_group[g][i].get("iou_binary") for i in sorted(by_group[g])]
        sil = [v for v in sil if v is not None]
        per_group[g] = {
            "miou": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_runs": len(vals),
            "iou_binary": float(np.mean(sil)) if sil else None,
            "headroom": (float(np.mean(sil)) - float(np.mean(vals))) if sil and vals else None,
        }

    # Combine within each run first, then take the spread across runs -- not the other
    # way round. Averaging the 5 subspecies inside run r correlates them, which is what
    # the paper's single +- per dataset represents.
    combined = [
        float(np.mean([by_group[g][i]["miou"] for g in groups if i in by_group[g]]))
        for i in range(n_runs)
        if all(i in by_group[g] for g in groups)
    ]
    combined_model = [
        float(np.mean([by_group[g][i]["miou_model_space"] for g in groups if i in by_group[g]]))
        for i in range(n_runs)
        if all(i in by_group[g] for g in groups)
    ]
    combined_sil = [
        float(np.mean([by_group[g][i]["iou_binary"] for g in groups if i in by_group[g]]))
        for i in range(n_runs)
        if all(i in by_group[g] and by_group[g][i].get("iou_binary") is not None
               for g in groups)
    ]

    std = float(np.std(combined, ddof=1)) if len(combined) > 1 else 0.0
    supports = {r["run_idx"]: r["support_ids"] for r in runs if r["group"] == groups[0]}

    # `metric` and `data` are what `registry.classify` diffs against FAITHFUL_SPEC, and
    # `data.interp` is in COMPARABILITY_FIELDS. Omitting them does not merely lose
    # provenance: every record lands in the `modified` track on fields that never varied,
    # and two runs differing only in resampling hash to the same comparability key and
    # can be pooled. The metric block is copied from `metrics.METRIC_SPEC` rather than
    # retyped, so what a bundle claims cannot drift from what `score_episode` computes.
    cache = getattr(ds, "cache", None)
    return {
        "dataset": ds.name,
        "adapter": adapter.name,
        "metric": {**METRIC_SPEC, "iou_space": "gt_native"},
        "data": {
            "source_preference": getattr(cache, "source_preference", None),
            "interp": getattr(cache, "interp", None),
            # The storage format changes the pixels the model sees just as surely as the
            # interpolation filter does, so it belongs in the record and in
            # COMPARABILITY_FIELDS. Without it a lossy-cache run and a lossless-cache run
            # hash to the same comparability key and can be pooled silently.
            "cache_format": (cache.policy(ds.name).fmt
                             if hasattr(cache, "policy") else None),
            "normalization": adapter.config_report().get("normalization")
            if hasattr(adapter, "config_report") else None,
        },
        "protocol": {"shots": shot, "n_runs": len(combined), "groups": groups,
                     "support_selection": support_selection,
                     "support_set_hash": support_set_hash(supports)},
        "results": {
            "tta": ({"episodes": sum(r["tta"]["episodes"] for r in runs if r.get("tta")),
                     "rotated_chosen": sum(r["tta"]["rotated_chosen"]
                                           for r in runs if r.get("tta"))}
                    if any(r.get("tta") for r in runs) else None),
            "miou": float(np.mean(combined)) if combined else None,
            "std": std,
            "sem": std / np.sqrt(len(combined)) if combined else None,
            "miou_model_space": float(np.mean(combined_model)) if combined_model else None,
            # Dataset-level silhouette and the gap to it — how much of this arm's remaining
            # distance to the paper is reachable by better ASSIGNMENT versus capped by mask
            # quality. See the per-group note above for why it is a correlate, not a bound.
            "iou_binary": float(np.mean(combined_sil)) if combined_sil else None,
            "headroom": (float(np.mean(combined_sil)) - float(np.mean(combined)))
                        if combined_sil and combined else None,
            "per_group": per_group,
            "per_run": combined,
        },
        "runs": runs,
        "_rows": rows,
    }


def write_per_query_csv(rows: list[dict], path: Path) -> Path:
    """The load-bearing artifact: aggregates can be recomputed, paired statistics cannot.

    Without per-(run, query, trait) scores there is no way to difference two variants on
    the same support seed, and unpaired means cannot resolve an effect the size of the
    paper's +0.2 on Major.
    """
    path = Path(path)
    fields = ["group", "shot", "run_idx", "query_id", "trait_id", "iou"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path
