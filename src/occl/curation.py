"""Support-set curation — approximating the paper's reference selection.

The paper does not sample references uniformly. Verbatim:

    "one labeled image **(with canonical shapes and visually clear traits)** is sampled
    as the reference for the test set."

That parenthetical is load-bearing. Our uniform arm (S4a) measured **55.96 ± 7.90** on
Beetle against the paper's **61.9 ± 3.7** — low mean *and* 2.1x the spread — because
uniform sampling draws upside-down specimens that no human curator would pick. Prompted
with an inverted beetle, SAM 2 confidently assigns `head` to the wrong end: the measured
correlation between support-orientation deviation and score is **-0.710**, while
`iou_binary` stays at 83-90% in every run. The silhouette is always found; only trait
*assignment* fails.

Two criteria, mirroring the paper's two clauses:

* **orientation** ("canonical shapes") -- body axis within one circular standard
  deviation of the training mean.
* **completeness** ("visually clear traits") -- every expected trait present.

Three properties make this defensible rather than a fit to the target:

1. **Train-only estimation.** Selecting supports by test-set orientation would leak test
   information into a one-shot protocol. It is also unnecessary: measured train circular
   mean 67.1 deg vs test 56.8 deg, concentration R 0.625 vs 0.614.
2. **The threshold is a priori.** It is the training distribution's own circular standard
   deviation, `sqrt(-2 ln R)` = 55.6 deg for Beetle -- derived from the data's spread, not
   chosen because it produces 61.9. Retuning it to close a residual gap would convert a
   reproduction into a fit.
3. **It only filters the pool.** `seeding.sample_supports` draws from the result unchanged,
   so the 20-run protocol, the seeds, and the paired-comparison machinery all carry over.

**Orientation is beetle-specific; butterfly needs a different geometric criterion.** The
180-degree flip mode is beetle's -- the paper singles beetles out for "up to 90-degree of
body rotations". An earlier note here recorded butterfly's axis as ill-defined at
R 0.11-0.70; that was the wrong instrument. Measured the way beetle is, with a *directed
trait-pair vector*, butterfly's axis is R 0.993-0.9996 (circular std 1.6-6.9 deg): the
criterion is inert not because it cannot be measured but because pinned specimens are
already canonical. Datasets without an `axis_parts` skip it regardless.

What butterfly actually needs is **placement**. Cambridge Butterfly frames are dissected
wings laid loosely on a sheet, and the position of those pieces varies enormously in one
subspecies: cyrbia's trait centroids scatter with a 219.4 px standard deviation against
36-60 px for the other four. Splitting cyrbia by background shows it is two imaging
protocols filed under one name -- green frames scatter the wings (209.9 px) while neutral
frames are as well-registered as anything else (27.7 px). SAM 2 carries the support's
spatial layout across to the query, so displacement breaks trait assignment directly:
measured corr(run mIoU, support-query displacement) = **-0.988** on cyrbia, against the
-0.710 orientation signal that motivated the entire Beetle arm.

A specimen can be perfectly upright and still be in the wrong corner of the sheet, which is
why orientation cannot stand in for placement.
"""

from __future__ import annotations

import numpy as np

from .datasets.base import trait_ids
from .seeding import derive_seed

SELECTIONS = ("uniform", "canonical", "canonical_balanced", "canonical_placement",
              "orientation_diverse")
DEFAULT_CRITERIA = ("orientation", "completeness")

#: Butterfly's "canonical shape" axis. Orientation is a no-op there (pinned specimens are
#: already canonical, R 0.993-0.9996) but *placement* is not: Cambridge Butterfly images
#: are dissected wings laid loosely on a sheet, and cyrbia mixes two imaging protocols --
#: green-background frames scatter the wings (placement std 209.9 px) while
#: neutral-background frames are as well-registered as every other subspecies (27.7 px).
#: Measured corr(run mIoU, support-query placement displacement) = **-0.988** on cyrbia,
#: stronger than the -0.710 orientation signal that motivated the whole Beetle arm.
PLACEMENT_CRITERIA = ("placement", "completeness")


def stratum_of(image_id: str) -> str:
    """Default stratum key: the species prefix of `<species>/<stem>`."""
    return image_id.split("/")[0]


def body_axis_angle(labelmap: np.ndarray, head_id: int, tail_id: int) -> float | None:
    """Signed angle in degrees of the head->tail axis, or None if either part is absent.

    Directed, not undirected: a specimen and its 180-degree rotation must be
    distinguishable, since that inversion is exactly the failure mode being filtered.
    """
    hy, hx = np.nonzero(labelmap == head_id)
    ty, tx = np.nonzero(labelmap == tail_id)
    if len(hy) == 0 or len(ty) == 0:
        return None
    return float(np.degrees(np.arctan2(ty.mean() - hy.mean(), tx.mean() - hx.mean())))


def circular_mean(angles_deg) -> float:
    """Mean direction. A plain arithmetic mean is wrong on a circle -- it would put the
    average of -179 and +179 at 0 rather than 180."""
    r = np.radians(np.asarray(angles_deg, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def circular_resultant(angles_deg) -> float:
    """Concentration R in [0, 1]: 1 = all identical, 0 = uniformly scattered."""
    r = np.radians(np.asarray(angles_deg, dtype=float))
    return float(np.hypot(np.sin(r).mean(), np.cos(r).mean()))


def circular_std(angles_deg) -> float:
    """Circular standard deviation in degrees, `sqrt(-2 ln R)`.

    This is the a-priori orientation threshold: a support is "canonical" when it lies
    within one standard deviation of how the training specimens are actually oriented.
    """
    R = circular_resultant(angles_deg)
    if R <= 0:
        return 180.0
    return float(np.degrees(np.sqrt(-2.0 * np.log(R))))


def angular_deviation(angle_deg: float, reference_deg: float) -> float:
    """Smallest absolute angle between two directions, wrapped to [0, 180]."""
    return float(abs((angle_deg - reference_deg + 180.0) % 360.0 - 180.0))


def trait_centroids(labelmap: np.ndarray, expected: int) -> dict[int, tuple[float, float]]:
    """Centroid of each present trait, in pixels."""
    out: dict[int, tuple[float, float]] = {}
    for t in range(1, expected + 1):
        ys, xs = np.nonzero(labelmap == t)
        if len(xs):
            out[t] = (float(xs.mean()), float(ys.mean()))
    return out


def placement_displacement(cent: dict[int, tuple[float, float]],
                           reference: dict[int, tuple[float, float]]) -> float | None:
    """Mean distance from a specimen's trait centroids to a reference configuration.

    The butterfly analogue of `body_axis_angle`: it measures whether the specimen is laid
    out where the other specimens are laid out. Orientation cannot capture this -- a
    perfectly upright wing placed in the wrong corner of the sheet has zero angular
    deviation and a large displacement, and it is the displacement that breaks mask
    propagation, because SAM 2 carries the support's spatial layout across to the query.

    Averaged over the traits the two configurations share; None if they share none.
    """
    shared = [t for t in cent if t in reference]
    if not shared:
        return None
    return float(np.mean([
        float(np.hypot(cent[t][0] - reference[t][0], cent[t][1] - reference[t][1]))
        for t in shared
    ]))


ORIENTATIONS = ("upright", "rot180")


def orientation_of(cent: dict[int, tuple[float, float]],
                   reference: dict[int, tuple[float, float]],
                   size: int = 1024) -> str:
    """Is this specimen laid out like the reference, or turned 180 degrees?

    Measured, not assumed: **37% of cyrbia's test specimens are rotated 180 deg** (34% of
    its train pool), against 0-1% for every other subspecies. Those rotated queries score
    **0.48 mIoU** where upright ones score 74.95, which is the whole of cyrbia's deficit
    -- 0.63*74.95 + 0.37*0.48 = 47.4, exactly its measured score.

    SST propagates the support's spatial layout through the memory bank, so an upside-down
    query receives every trait on its mirror-image partner. The copy baseline shows the
    same thing with no model at all: rotated cyrbia queries score **0.00**.

    Compares the configuration against the reference and against the reference rotated
    about the image centre; whichever is closer wins.
    """
    shared = [t for t in cent if t in reference]
    if not shared:
        return "upright"
    up = np.mean([np.hypot(cent[t][0] - reference[t][0], cent[t][1] - reference[t][1])
                  for t in shared])
    ro = np.mean([np.hypot(cent[t][0] - (size - reference[t][0]),
                           cent[t][1] - (size - reference[t][1])) for t in shared])
    return "upright" if up <= ro else "rot180"


def orientation_strata(ds, group: str, ids: list[str]) -> dict[str, str]:
    """``{image_id: "upright"|"rot180"}``, with the reference taken from TRAIN only.

    Train-only for the same reason the placement threshold is: deciding orientation from
    the test split would leak test information into a one-shot protocol.

    Memoised per dataset instance for the same reason `support_pool` is, and it matters
    more here: `orientation_diverse_supports` calls this once per *run*, and the pool it
    passes is identical across all 20 — so the reference centroids over the group's whole
    train split were being recomputed 20 times. Measured on Butterfly Major that is ~15
    minutes of single-threaded mask reading in front of the GPU work, per stage.
    """
    key = (group, tuple(ids))
    cache = getattr(ds, "_orientation_strata_cache", None)
    if cache is None:
        cache = {}
        try:
            ds._orientation_strata_cache = cache
        except AttributeError:  # frozen/slotted dataset — correctness over speed
            cache = None
    if cache is not None and key in cache:
        return dict(cache[key])
    out = _orientation_strata_uncached(ds, group, ids)
    if cache is not None:
        cache[key] = dict(out)
    return out


def _orientation_strata_uncached(ds, group: str, ids: list[str]) -> dict[str, str]:
    expected = ds.expected_traits(group)
    train = sorted(ds.train_ids(group))
    tr_cent = [trait_centroids(ds.load_labelmap(i), expected) for i in train]
    ref = {}
    for t in range(1, expected + 1):
        pts = [c[t] for c in tr_cent if t in c]
        if pts:
            ref[t] = (float(np.median([p[0] for p in pts])),
                      float(np.median([p[1] for p in pts])))
    out = {}
    for i in ids:
        lm = ds.load_labelmap(i)
        # Mirror about the mask's OWN extent. Hardcoding 1024 is correct for butterfly,
        # whose masks are natively 1024x1024, but beetle crops are 172-431 px — there the
        # rotated reference lands outside the image entirely and every specimen scores as
        # upright, silently turning this criterion into a no-op.
        out[i] = orientation_of(trait_centroids(lm, expected), ref,
                                size=max(lm.shape[0], lm.shape[1]))
    return out


def canonical_pool(ds, group: str, criteria=DEFAULT_CRITERIA) -> tuple[list[str], dict]:
    """Filter a group's TRAIN ids down to plausible reference specimens.

    Returns ``(pool_ids, stats)``. ``stats`` goes verbatim into the result record so the
    approximation is auditable after the fact.

    Never reads the test split -- see the module docstring on leakage.
    """
    train = sorted(ds.train_ids(group))
    axis_parts = getattr(ds, "axis_parts", None)
    expected = ds.expected_traits(group) if hasattr(ds, "expected_traits") else None

    labelmaps = {i: ds.load_labelmap(i) for i in train}
    use_orientation = "orientation" in criteria and axis_parts is not None
    use_completeness = "completeness" in criteria and expected is not None
    use_placement = "placement" in criteria and expected is not None

    angles: dict[str, float | None] = {}
    if use_orientation:
        angles = {i: body_axis_angle(lm, *axis_parts) for i, lm in labelmaps.items()}
        valid = [a for a in angles.values() if a is not None]
        mean_deg = circular_mean(valid)
        thresh = circular_std(valid)
        R = circular_resultant(valid)
    else:
        mean_deg = thresh = R = None

    # Placement: same shape of rule as orientation, and the same a-priori threshold
    # discipline -- one standard deviation of the TRAIN displacement distribution. The
    # reference configuration is the train *median* centroid per trait (median, not mean,
    # so a handful of badly-placed specimens cannot drag the reference toward themselves).
    disp: dict[str, float | None] = {}
    if use_placement:
        cents = {i: trait_centroids(lm, expected) for i, lm in labelmaps.items()}
        ref = {}
        for t in range(1, expected + 1):
            pts = [c[t] for c in cents.values() if t in c]
            if pts:
                ref[t] = (float(np.median([p[0] for p in pts])),
                          float(np.median([p[1] for p in pts])))
        disp = {i: placement_displacement(c, ref) for i, c in cents.items()}
        dvalid = [d for d in disp.values() if d is not None]
        disp_mean = float(np.mean(dvalid))
        disp_thresh = disp_mean + float(np.std(dvalid))
    else:
        ref = {}
        disp_mean = disp_thresh = None

    pool, rejected = [], {"orientation": 0, "completeness": 0, "no_axis": 0, "placement": 0}
    for i in train:
        if use_completeness and len(trait_ids(labelmaps[i])) != expected:
            rejected["completeness"] += 1
            continue
        if use_orientation:
            a = angles[i]
            if a is None:
                rejected["no_axis"] += 1
                continue
            if angular_deviation(a, mean_deg) > thresh:
                rejected["orientation"] += 1
                continue
        if use_placement:
            d = disp[i]
            if d is None or d > disp_thresh:
                rejected["placement"] += 1
                continue
        pool.append(i)

    if not pool:
        raise RuntimeError(
            f"curation left no supports for {ds.name}/{group} with criteria {criteria}"
        )

    stats = {
        "criteria": list(criteria),
        # What was *requested* vs what actually ran: `orientation` silently does nothing
        # on a dataset with no `axis_parts`, which is auditable but misleading in a bundle.
        "criteria_applied": [c for c, on in (("orientation", use_orientation),
                                             ("completeness", use_completeness),
                                             ("placement", use_placement)) if on],
        "pool_size": len(pool),
        "train_size": len(train),
        "rejected": rejected,
        "axis_parts": list(axis_parts) if axis_parts else None,
        "expected_traits": expected,
        # Recorded so a later refactor cannot silently move the threshold.
        "threshold_deg": round(thresh, 3) if thresh is not None else None,
        "train_circular_mean_deg": round(mean_deg, 3) if mean_deg is not None else None,
        "train_resultant_R": round(R, 4) if R is not None else None,
        "threshold_rule": "one training circular std, sqrt(-2 ln R); a priori, not tuned",
    }
    if use_placement:
        stats |= {
            "placement_threshold_px": round(disp_thresh, 3),
            "train_placement_mean_px": round(disp_mean, 3),
            "placement_threshold_rule":
                "train mean + 1 train std of centroid displacement; a priori, not tuned",
        }
    return pool, stats


def stratified_supports(pool, dataset: str, group: str, shot: int, run_idx: int,
                        key_fn=stratum_of) -> list[str]:
    """Draw supports with **systematic** stratum coverage, not merely equal probability.

    Orientation filtering leaves the pool species-skewed -- 10/10 specimens survive for
    `melanarius melanarius` but only 4/10 for `dubius`, `grossus` and `restrictus`. Since
    Beetle is a *universal* model scored on exactly 5 test images per species, uniform
    sampling had given every species equal probability, and curation silently broke that.
    This restores it.

    Stratified *random* sampling is not enough, and this is the non-obvious part: 20
    independent draws over 12 species is a coupon-collector problem, and measured, it
    still leaves 3 species unused (9/12 coverage). Equal probability is not equal
    representation at n=20. Systematic round-robin assignment -- run `r` starts at stratum
    `r mod S` -- gives 12/12 coverage with usage 1-2, while the member *within* a stratum
    is still drawn by the run's seed.

    Prefix-nesting survives: run `r`'s 1-shot draw is the first element of its 5-shot
    draw, so the 1-vs-5-shot comparison stays paired. A 5-shot draw additionally spans 5
    *distinct* strata, which suits a model that has to generalise across species.
    """
    strata: dict[str, list[str]] = {}
    for i in sorted(pool):
        strata.setdefault(key_fn(i), []).append(i)
    names = sorted(strata)
    if shot > len(names):
        raise ValueError(
            f"shot={shot} exceeds {len(names)} strata; cannot draw distinct strata"
        )

    rng = np.random.default_rng(derive_seed(dataset, group, 0, run_idx))
    out = []
    for k in range(shot):
        members = strata[names[(run_idx + k) % len(names)]]
        out.append(members[int(rng.integers(len(members)))])
    return out


def orientation_diverse_supports(ds, group: str, pool, dataset: str, shot: int,
                                 run_idx: int) -> list[str]:
    """Draw supports that SPAN orientations, in the train split's own proportion.

    The problem this addresses is **query-side**, which is why nothing support-side has
    touched it: placement curation moved cyrbia +3.56 and 5-shot moved it +2.85, both
    collapsing variance and leaving the level alone. A single upright support cannot
    segment an upside-down query, and no choice of upright support ever will.

    At 5-shot there is room to carry both orientations in the memory bank at once. The
    split follows the **train** prior (cyrbia: 34% rotated -> 2 of 5), never the test
    distribution, which would be leakage. Subspecies with one orientation degenerate to
    the ordinary draw, so this is a no-op everywhere except cyrbia.

    Whether SAM 2 then *selects* the matching orientation rather than blending all five is
    the open question -- blending is the most likely reason plain 5-shot did not help.
    This arm is what answers it.
    """
    from .seeding import sample_supports

    state = orientation_strata(ds, group, sorted(pool))
    by = {}
    for i in sorted(pool):
        by.setdefault(state[i], []).append(i)
    if len(by) < 2 or shot < 2:
        return sample_supports(pool, dataset, group, shot, run_idx)

    # How many of each orientation, proportional to the train pool, at least one each.
    n_rot = min(max(1, round(shot * len(by.get("rot180", [])) / len(pool))), shot - 1)
    want = {"rot180": n_rot, "upright": shot - n_rot}

    rng = np.random.default_rng(derive_seed(dataset, group, 0, run_idx))
    out: list[str] = []
    for name in ORIENTATIONS:                      # deterministic order
        members = by.get(name, [])
        k = min(want.get(name, 0), len(members))
        if k:
            idx = rng.permutation(len(members))[:k]
            out += [members[int(j)] for j in idx]
    # Top up from the whole pool if a class was short, keeping the count exact.
    if len(out) < shot:
        rest = [i for i in sorted(pool) if i not in set(out)]
        idx = rng.permutation(len(rest))[:shot - len(out)]
        out += [rest[int(j)] for j in idx]
    return out[:shot]


def draw_supports(ds, group: str, pool, support_selection: str, shot: int,
                  run_idx: int) -> list[str]:
    """The one place a support selection maps to a draw function.

    This exists because it was duplicated **three** times and the copies drifted.
    `protocol.build_run` dispatched `orientation_diverse` to
    `orientation_diverse_supports`; `build_cache.required_butterfly` and
    `evaluator.distinct_images` both fell through to `sample_supports` for the same arm.
    Consequences, in the order they bite:

    1. the cache is pre-built from a support set the run never asks for;
    2. the preflight reports the *wrong* images as missing — so topping the cache up from
       its own error message does not fix it, and the next run fails the same way;
    3. `precompute` warms embeddings for images no episode uses, and every real support
       misses the cache and re-encodes.

    Same failure shape as `data.interp`, `data.cache_format` and `model.tta` slipping the
    faithfulness spec: several places that must agree, with nothing forcing them to.
    Callers must route through here rather than branching on `support_selection`
    themselves; `tests/test_curation.py` discovers callers rather than listing them,
    because the first version of that guard enumerated the two modules already fixed and
    passed while the third was still broken.
    """
    from .seeding import sample_supports          # local: seeding imports curation

    if support_selection == "canonical_balanced":
        # Systematic stratum coverage; see stratified_supports for why equal
        # probability is not enough at n=20.
        return stratified_supports(pool, ds.name, group, shot, run_idx)
    if support_selection == "orientation_diverse":
        return orientation_diverse_supports(ds, group, pool, ds.name, shot, run_idx)
    return sample_supports(pool, ds.name, group, shot, run_idx)


def support_pool(ds, group: str, support_selection: str = "uniform",
                 criteria=DEFAULT_CRITERIA) -> tuple[list[str], dict]:
    """Resolve the pool `seeding.sample_supports` draws from.

    Memoised **per dataset instance**, because `protocol.build_run` calls this once per
    run and `canonical_pool` re-reads every training mask in the group. On Butterfly Major
    that is 4,227 masks x 20 runs = ~84,500 redundant 1024x1024 reads, which costs more
    wall clock than the GPU work it precedes.

    Deliberately cached *here* and not in `canonical_pool`: the leakage guard in
    `tests/test_curation.py` perturbs `ds.test_ids` and asserts the pool is unchanged, and
    a cache underneath it would make that test pass vacuously. `canonical_pool` therefore
    stays pure and is what those tests call.
    """
    if support_selection not in SELECTIONS:
        raise ValueError(f"support_selection must be one of {SELECTIONS}")
    key = (group, support_selection, tuple(criteria))
    cache = getattr(ds, "_support_pool_cache", None)
    if cache is None:
        cache = {}
        try:
            ds._support_pool_cache = cache
        except AttributeError:  # frozen/slotted dataset — correctness over speed
            cache = None
    if cache is not None and key in cache:
        pool, stats = cache[key]
        return list(pool), dict(stats)
    pool, stats = _support_pool_uncached(ds, group, support_selection, criteria)
    if cache is not None:
        cache[key] = (list(pool), dict(stats))
    return pool, stats


def _support_pool_uncached(ds, group: str, support_selection: str,
                           criteria) -> tuple[list[str], dict]:
    if support_selection == "uniform":
        train = sorted(ds.train_ids(group))
        return train, {"criteria": [], "pool_size": len(train)}
    if support_selection == "canonical_placement":
        # Butterfly's criterion. `orientation` is inert here (no axis_parts) and would
        # silently contribute nothing, so it is not requested at all.
        criteria = PLACEMENT_CRITERIA
    if support_selection == "orientation_diverse":
        # Completeness only. The placement filter measures displacement from the TRAIN
        # MEDIAN, so a 180-degree-rotated specimen is maximally distant and gets rejected
        # by construction — exactly the specimens this arm needs to be able to draw.
        #
        # NOTE this drops the dataset's other curation with it. On Beetle that means
        # losing both the orientation filter AND species balancing, which together are
        # worth +7.2 (S4a 55.96 -> S4c 63.20) — so a Beetle run of this arm would regress
        # for reasons having nothing to do with orientation diversity. The arm targets
        # butterfly's discrete 180-degree flip; Beetle's orientation varies *continuously*
        # (circular std 55.6 deg) and is already handled by `canonical_balanced`.
        criteria = ("completeness",)
    pool, stats = canonical_pool(ds, group, criteria)
    if support_selection == "canonical_balanced":
        strata = {}
        for i in pool:
            strata.setdefault(stratum_of(i), []).append(i)
        stats = stats | {
            "sampling": "round_robin_stratified",
            "n_strata": len(strata),
            "stratum_sizes": {k: len(v) for k, v in sorted(strata.items())},
        }
    else:
        stats = stats | {"sampling": "uniform_over_pool"}
    return pool, stats
