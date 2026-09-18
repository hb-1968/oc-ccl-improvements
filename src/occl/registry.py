"""Dual-track results registry: which numbers may be compared, and how.

Two mechanisms carry the weight.

**`track` is derived, never typed.** A record does not get to *assert* it is a faithful
reproduction; `classify()` diffs it against one canonical `FAITHFUL_SPEC` and returns the
list of fields that deviate. Disagree about what counts as faithful? Edit one dict and
every historical record reclassifies. A hand-written `"track": "faithful"` would rot the
moment someone changed a default.

**`comparability_key()` refuses illegitimate comparisons.** It hashes the fields that must
match for two numbers to belong in the same table -- dataset, split, shots, the *resolved*
support IDs, the metric variant, and the preprocessing that produced the pixels. The
aggregator groups by it and will not emit a cross-group delta.

`support_set_hash` is deliberately a hash of the chosen image IDs rather than of the seed:
two runs can share seeds yet sample different supports if one of them had an image missing
from the manifest, and only the resolved IDs catch that.

**Paired comparison is the point.** The paper's Butterfly Major result is 81.0±1.0 ->
81.2±1.0 with OC-CCL. Differencing those unpaired means is uninformative by construction:
the ±1.0 is across-support-set spread, which dwarfs the +0.2 effect. Because the harness
pins 20 seeded support sets, the right statistic is per-seed `Δ_s = treatment(s) -
baseline(s)` with a bootstrap CI over seeds. Support difficulty is the dominant variance
component and cancels exactly, which is what makes an effect that small resolvable at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_MISSING = object()


def get_path(rec: dict, dotted: str, default=_MISSING):
    cur: Any = rec
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if default is _MISSING:
                return None
            return default
        cur = cur[part]
    return cur


# ── what "faithful" means ────────────────────────────────────────────────────

#: Applies to every record. These are the choices that decide whether a number is a
#: reproduction of the paper or a measurement of something else.
FAITHFUL_COMMON = {
    "model.family": "sam2",
    "model.image_size": 1024,
    "model.num_maskmem": 7,
    "metric.name": "iou_per_trait_macro",
    "metric.assignment": "argmax_label",
    "metric.reduction": "macro_over_gt_traits",
    "data.source_preference": "faithful",  # .CR2 via rawpy, as SST's loaders do
    "data.interp": "linear",               # cv2.resize default, as SST uses
    # SST has no intermediate encode at all — it decodes and resizes straight into the
    # model. A lossless cache reproduces that exactly; a lossy one is a deviation.
    "data.cache_format": frozenset({"png", "webp"}),
    "data.normalization": "imagenet",
    # SST has no orientation handling anywhere, and the paper's only rotation is
    # beetle-only fine-tuning augmentation plus an OOD stress test that *injects* it. So
    # test-time augmentation measures "SST plus rotation handling", not SST. Without this
    # line an arm that lifts Major by +4.88 would classify as a faithful reproduction.
    "model.tta": None,
    # The paper curates references ("canonical shapes and visually clear traits"), so
    # uniform sampling is the deviation, not the baseline. Every curated arm approximates
    # that same unpublished clause with the criterion its dataset actually needs --
    # orientation for Beetle, placement for Butterfly, plus species balance -- so all of
    # them are faithful on this field and only `uniform` deviates. See occl.curation.
    # `orientation_diverse` is deliberately NOT here: deliberately including upside-down
    # references is an intervention the paper never describes, so it belongs in the
    # modified track alongside TTA.
    "protocol.support_selection": frozenset(
        {"canonical", "canonical_balanced", "canonical_placement"}),
}

#: Additionally required when a fine-tune produced the checkpoint being scored.
#:
#: DERIVED FROM THE PAPER (arXiv 2501.06749 v2), re-read in full on 2026-08-12, with the
#: governing sentence quoted on every line. This block has now flipped TWICE, and both
#: flips are kept on the record because each was locally right about its own source:
#:
#:   flip 1 (spec -> release): the original spec said LoRA r=32/alpha=64 on
#:   memory_encoder+mask_decoder with no provenance comment. Checking the RELEASED CODE
#:   found none of it there — `oc_ccl.py::train` full-fine-tunes everything but the image
#:   encoder for 10 epochs, and LoRA exists only in an ablation script, off by default,
#:   sweeping r 4/8/16 at alpha=1.0. So the spec was relabelled to the release's config.
#:
#:   flip 2 (release -> paper, this one): re-reading the paper showed the "unprovenanced"
#:   original spec was A2.2 nearly verbatim, and it is the RELEASE that contradicts the
#:   paper — on the method, the epochs, the mask, the prompt's differentiability, and the
#:   beetle augmentation (absent from the release entirely). The release's OC-CCL code
#:   landed ~16 months after the paper, from a different contributor; it is a
#:   re-implementation, not the code that produced Table 2. The reproduction target is
#:   the paper, so the paper defines `faithful`, and the release's configuration is a
#:   *named deviation set* (pinned as such in tests/test_registry.py).
#:
#: `prompt_mode: logit_reparam` is faithful because §3.2 states "Notably, this prompt
#: remains differentiable" (v1: "This prompt is differentiable") — the released
#: `(pred > 0).float()` severs the graph and is the deviation, not the fix.
FAITHFUL_FINETUNE = {
    # A2.2: "we replace all linear projection layers and MLP layers in the memory encoder
    # and mask decoder of the video segmentation models with LoRA linear layers, using
    # r=32 and alpha=64." §3.2: "We apply LoRA [27] to fine-tune the decoder and memory
    # encoder". Note memory *encoder*, not memory *attention* — the release's ablation
    # docstring says attention, the paper does not.
    "finetune.method": "lora",
    "finetune.lora.r": 32,
    "finetune.lora.alpha": 64.0,
    "finetune.lora.scopes": ["memory_encoder", "sam_mask_decoder"],
    # A2.2: "we follow the default LoRA initialization and learning rate scheduling for
    # LoRA [27], and train it for one epoch." This RESOLVES the note that previously
    # lived here: the release's 10 epochs is the release's, and the release is the
    # deviation. (The paper names no numeric lr, optimizer or batch size — those remain
    # recorded choices, deliberately not spec fields.)
    "finetune.epochs": 1,
    # §3.2: "Notably, this prompt remains differentiable."
    "occl.prompt_mode": "logit_reparam",
    # §3.2, strategy 1: "During the forward pass, we reset the memory bank after
    # processing x1, preventing it from carrying y0 to x0-dagger."
    "occl.memory_reset": True,
    "occl.cycle_len": 2,
    # §3: "y in {0,1}^{W x H x C} ... C distinct traits"; OC-CCL supervises y-hat-0-dagger
    # against that same multi-channel y0, and §3.3 explicitly reserves single-channel
    # handling for retrieval ("y0* — a single channel in the original y0"). The release's
    # merged foreground ((mask > 0), butterfly_dataset.py:121; recorded in older bundles
    # as "binary", newer ones as "merged") is the deviation.
    "occl.mask_mode": "per_trait",
    # A2.2: "We take one fixed labeled image as x0, and iteratively sample x1 from the
    # unlabeled training set." One label — not the release's every-train-image-is-a-
    # reference enumeration, and not this repo's derangement either. `anchored` (one
    # fine-tune per evaluation run, each on that run's own reference) is the A2.2+A2.3
    # joint reading — the paper never says whether the OC-CCL rows re-fine-tune per run,
    # and its OOD section fine-tunes per test image, so both readings are admitted the
    # way `protocol.support_selection` admits several curations of one unpublished clause.
    "occl.reference": frozenset({"fixed", "anchored"}),
    # SCNP (S9a) is ours, not the paper's — any non-null value deviates. Plain None so
    # every pre-SCNP record (which lacks the key entirely; get_path yields None) stays
    # classified exactly as before.
    "occl.scnp": None,
    # The S13 loss knobs are ours, not the paper's — same contract as occl.scnp: any
    # non-null value classifies the run `modified` on exactly that field, and every
    # pre-S13 record (key absent; get_path yields None) stays classified as before.
    # excl: STM/Cutie exclusivity CE in the BCE slot; xovl: SAM-MT overlap penalty on
    # the forward leg; bshape: Kervadec signed-distance term; margin: the
    # orientation-contrastive confidence-margin TTA fold.
    "occl.excl": None,
    "occl.xovl": None,
    "occl.bshape": None,
    "occl.margin": None,
    # S17 knobs on the closing loss (README S17): found_mask drops the channels the
    # forward leg did not find; bce_w != 1 reweights the paper's BCE term (0 = Dice
    # only). Same contract: null when off, any value deviates.
    "occl.found_mask": None,
    "occl.bce_w": None,
    # S17b: objness (a term on the forward frame's objectness score) and lora_legs
    # ("forward" = the closing legs run on base weights). Same contract.
    "occl.objness": None,
    "occl.lora_legs": None,
    # S17d: soft_sentinel (the −1024 no-object gate made differentiable in the training
    # path, K = 20). Same contract.
    "occl.soft_sentinel": None,
    # S18: close_aug (the closing end of the palindrome under its own rotation). Same
    # contract. The other S18 axis is occl.prompt_mode (ste_threshold), pinned above.
    "occl.close_aug": None,
    # A named recipe (S17d: `sam2_crown`) is by definition a deviation set.
    "occl.recipe": None,
}

#: THE ESTABLISHED SAM 2 FINE-TUNE (2026-09-17, after S17/S17b/S17c/S17d closed the
#: family's fine-tuning program — README "The SAM 2 fine-tuning recipe"). Every future
#: run that fine-tunes SAM 2 uses it unless the stage is explicitly about changing it:
#: `run_finetune.py --recipe sam2_crown` expands to exactly `SAM2_CROWN_FLAGS`, records
#: `occl.recipe = "sam2_crown"`, and `is_sam2_crown_finetune` re-derives the
#: classification from the record's fields (the name is a label, the fields are the
#: proof). Measured: beetle 1-shot 65.04 with TTA (+1.18 ✱ vs base, 15/20 seeds; plain
#: 63.71, +0.51 ✱), the same cell as S17c's `comp_arch` (65.03, +0.01 ns on the same
#: seeds — the weight is subsumed by the scope and kept because it flips 16 of 19 fence
#: channels on the training pairs instead of 12); no 5-shot transfer (−0.19 / +0.19 ns).
#:
#:   fwdonly    adapters off the reverse init and the closing frame (`lora_legs=forward`)
#:              — the closing head cannot absorb the objective (S17b: closing +3.7 → +1.8,
#:              the good-prompt degradation gone)
#:   objness    BCE(forward objectness score, present) per prompted channel, weight 10 —
#:              the only handle on SAM 2's −1024 no-object sentinel (S17b/c/d: the
#:              closing loss's own gradient at the fence is 10× weaker than this prior)
#:   excl       exclusivity CE in the BCE slot, zerocat (S13/S17: the eval rule as loss)
#:   arch       LoRA on memory_encoder + memory_attention + sam_mask_decoder (S17c/d:
#:              the scope moves the oracle ceiling, +0.93)
#:   carrier    per-trait channels, derangement pairs, no augmentation, reparam prompt,
#:              memory reset, r=32 α=64, lr 1e-5, ONE epoch (S17c: 3 epochs +0.26 ns,
#:              lr 3e-5 ±0, (3e-5, 3) overfits its own pairs)
#:
#: NOT in it, each measured null or harmful: found_mask (≡ carrier), bce_w=0, the
#: forward_link legs (pure absorption), the soft sentinel (null without the prior,
#: −0.13 ✱ with it), SCNP, xovl, bshape, margin, rotation augmentation, weight 30
#: (= weight 10), 3 epochs.
SAM2_CROWN_FLAGS = ("--family", "sam2", "--method", "lora", "--lr", "1e-05",
                    "--epochs", "1", "--mask-mode", "per_trait",
                    "--lora-legs", "forward", "--objness", "--objness-weight", "10",
                    "--excl-mode", "zerocat", "--scope-preset", "sam2_arch")
SAM2_CROWN_FINETUNE = {
    "finetune.family": "sam2",
    "finetune.method": "lora",
    "finetune.lora.r": 32,
    "finetune.lora.alpha": 64.0,
    "finetune.lora.scopes": ["memory_encoder", "memory_attention", "sam_mask_decoder"],
    "finetune.epochs": 1,
    "finetune.lr": 1e-5,
    "occl.prompt_mode": "logit_reparam",
    "occl.memory_reset": True,
    "occl.mask_mode": "per_trait",
    "occl.reference": "derangement",
    "occl.augmentation": None,
    "occl.lora_legs": "forward",
    "occl.objness": {"weight": 10.0, "target": "present_for_prompted"},
    "occl.excl": {"mode": "zerocat"},
    "occl.found_mask": None,
    "occl.bce_w": None,
    "occl.soft_sentinel": None,
    "occl.close_aug": None,
    "occl.scnp": None,
    "occl.xovl": None,
    "occl.bshape": None,
    "occl.margin": None,
}
RECIPES = {"sam2_crown": SAM2_CROWN_FLAGS}


def expand_recipe(argv: list[str]) -> list[str]:
    """``--recipe <name>`` in a run_finetune.py argv -> the recipe's flags spliced in
    FRONT of everything else, so an explicit flag after it still wins (argparse keeps
    the last occurrence of a valued flag). ``--recipe`` itself stays in the list so the
    parser records the name. Unknown names raise."""
    out = list(argv)
    for i, a in enumerate(argv):
        name = None
        if a == "--recipe" and i + 1 < len(argv):
            name = argv[i + 1]
        elif a.startswith("--recipe="):
            name = a.split("=", 1)[1]
        if name is None:
            continue
        if name not in RECIPES:
            raise ValueError(f"unknown --recipe {name!r}; known: {sorted(RECIPES)}")
        return list(RECIPES[name]) + out
    return out


def is_sam2_crown_finetune(rec: dict) -> tuple[list[str], bool]:
    """``(deviating_fields, is_crown)`` — the record's fields against the established
    SAM 2 recipe, independent of what `occl.recipe` claims."""
    dev = sorted(k for k, want in SAM2_CROWN_FINETUNE.items()
                 if not _satisfies(get_path(rec, k), want))
    return dev, not dev


#: A2.2's augmentation clause is DATASET-CONDITIONAL: "For Beetle [22], due to the
#: inherently high variation in beetle orientations, we introduce a small degree of random
#: 2D rotation augmentation on the labeled example during the fine-tuning stage."
#: Butterfly gets none. A flat spec entry would let a butterfly-with-aug run classify
#: faithful, so `classify` folds this in per dataset. The magnitude is unquantified in the
#: paper and deliberately not a spec field — the run report records the value and the rule
#: it came from.
FINETUNE_AUG_BY_DATASET = {"beetle": "rotation_x0", "butterfly_major": None}


def _satisfies(got, want) -> bool:
    """A spec value may be a *set* of acceptable values rather than a single one.

    Used where several concrete choices are all faithful to the same underspecified
    sentence in the paper — see `protocol.support_selection`. Kept deliberately narrow: a
    set means "the paper does not distinguish these", not "we accept whatever we got".
    """
    if isinstance(want, (set, frozenset)):
        return got in want
    return got == want


def classify(rec: dict) -> tuple[list[str], str]:
    """Return ``(deviating_fields, track)``. ``track`` is "faithful" iff nothing deviates."""
    spec = dict(FAITHFUL_COMMON)
    if get_path(rec, "finetune") is not None:
        spec |= FAITHFUL_FINETUNE
        # Beetle-only rotation augmentation (A2.2) — see FINETUNE_AUG_BY_DATASET.
        want_aug = FINETUNE_AUG_BY_DATASET.get(get_path(rec, "dataset"))
        if want_aug is None:
            spec["occl.augmentation"] = None
        else:
            spec["occl.augmentation.type"] = want_aug
    dev = sorted(k for k, want in spec.items() if not _satisfies(get_path(rec, k), want))
    return dev, ("faithful" if not dev else "modified")


# ── comparability ────────────────────────────────────────────────────────────

COMPARABILITY_FIELDS = [
    "dataset",
    "protocol.shots",
    "protocol.n_runs",
    "protocol.support_selection",
    "protocol.groups",
    "protocol.support_set_hash",
    "metric.name",
    "metric.assignment",
    "metric.reduction",
    "metric.iou_space",
    "data.source_preference",
    "data.interp",
    "data.cache_format",
    "model.family",
    "model.image_size",
    "model.tta",
]


def comparability_key(rec: dict) -> str:
    payload = json.dumps([get_path(rec, f) for f in COMPARABILITY_FIELDS],
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def diff_fields(a: dict, b: dict, fields=COMPARABILITY_FIELDS) -> dict:
    return {f: (get_path(a, f), get_path(b, f))
            for f in fields if get_path(a, f) != get_path(b, f)}


# ── paired comparison ────────────────────────────────────────────────────────


def per_seed_scores(rec: dict, metric: str = "miou") -> dict[int, float]:
    """``{run_idx: score}``. Prefers explicit per-run records over the summary list.

    ``metric`` is a dotted path into each per-run record; the default reproduces the
    original behaviour byte for byte. Diagnostic metrics (``"miou_variants.independent"``,
    ``"miou_boundary.independent_best3"``) exist only in per-run records, so a
    non-default metric on a record without ``runs`` yields ``{}`` — a loud empty rather
    than silently substituting the plain metric.
    """
    runs = rec.get("runs")
    if runs:
        by_run: dict[int, list[float]] = {}
        for r in runs:
            v = get_path(r, metric) if "." in metric else r.get(metric)
            if v is not None:
                by_run.setdefault(int(r["run_idx"]), []).append(float(v))
        # average groups within a run first — the same order the evaluator uses
        return {k: float(np.mean(v)) for k, v in by_run.items()}
    if metric != "miou":
        return {}
    return {i: float(v) for i, v in enumerate(get_path(rec, "results.per_run") or [])}


#: The two comparability fields a shot-count pairing must mask — and may mask ONLY
#: through `compare(nested_shots=True)`, which demands `nested_shot_evidence` first.
NESTED_SHOT_FIELDS = ("protocol.shots", "protocol.support_set_hash")


def nested_shot_evidence(treatment: dict, baseline: dict) -> dict:
    """Prove, run by run, that the baseline's support draw is a PREFIX of the treatment's.

    This is the licence for pairing across shot counts (S16: 2-shot vs 1-shot+TTA).
    `seeding.sample_supports` and `curation.stratified_supports` slice ``[:shot]`` from
    one seeded permutation, so run *i*'s 1-shot support is the first element of its
    2-shot (and 5-shot) draw, and the per-seed difference still cancels the shared
    support's difficulty. But that is a property of the DRAW, not of two records that
    happen to sit side by side: a record built from another pool (a different selection
    arm, a filtered manifest) breaks it silently, and ``protocol.support_set_hash`` — the
    field that normally guards pairing — legitimately DIFFERS across shot counts, so it
    cannot stand guard here. Hence the evidence is read back from each record's per-run
    ``support_ids`` and travels with the comparison.

    Raises ValueError when the pairing is not licensed.
    """
    kt, kb = get_path(treatment, "protocol.shots"), get_path(baseline, "protocol.shots")
    if not (isinstance(kt, int) and isinstance(kb, int)) or kt <= kb:
        raise ValueError(
            "nested-shot pairing needs treatment shots > baseline shots "
            f"(got {kt!r} vs {kb!r})")

    def _draws(rec):
        return {(r["group"], int(r["run_idx"])): [str(x) for x in r["support_ids"]]
                for r in (rec.get("runs") or []) if r.get("support_ids") is not None}

    dt, db = _draws(treatment), _draws(baseline)
    if not dt or not db:
        raise ValueError(
            "nested-shot pairing needs per-run support_ids on both records — a "
            "summary-only record carries no draw to verify")
    shared = sorted(set(dt) & set(db))
    if not shared:
        raise ValueError("no shared (group, run) between the two records")
    bad = [k for k in shared
           if len(dt[k]) != kt or len(db[k]) != kb or dt[k][:kb] != db[k]]
    if bad:
        raise ValueError(
            f"support draws are not prefix-nested on {len(bad)} of {len(shared)} runs "
            f"(e.g. {bad[:3]}) — the records did not come from the same seeded pool")
    return {"treatment_shots": kt, "baseline_shots": kb, "runs_checked": len(shared)}


def compare(treatment: dict, baseline: dict, n_boot: int = 10_000, seed: int = 0,
            metric: str = "miou", ignore_fields: tuple = (),
            nested_shots: bool = False) -> dict:
    """Paired per-seed delta with a bootstrap CI. Refuses incomparable records.

    ``metric`` selects which per-run score is paired (see `per_seed_scores`); the
    default is the plain mIoU and every recorded number before S12 used it.

    ``ignore_fields`` (S14) is the ONLY sanctioned way to pair across model families:
    the named comparability fields are masked out of the key, the relaxation is
    embedded in the result (``ignored_fields`` with both sides' values), and
    ``protocol.support_set_hash`` must still match — the support draws are
    model-independent by construction (seeded from dataset/group/run only), which is
    precisely what makes a cross-family PAIRED delta statistically valid. Masking a
    protocol/metric/data field is refused outright: those change what was measured,
    not what measured it.

    ``nested_shots`` (S16) is the ONLY sanctioned way to pair across SHOT COUNTS. It
    masks `NESTED_SHOT_FIELDS` — but only after `nested_shot_evidence` has verified,
    per (group, run), that the baseline's draw is a prefix of the treatment's; that
    replaces the hash pin, which cannot hold across shot counts. The verified evidence
    travels in the result (``nested_shots``) and the masked fields' values in
    ``ignored_fields``. Naming a protocol field in ``ignore_fields`` is still refused:
    the relaxation must come with its proof, never by assertion. Composes with the
    model.* whitelist, so 2-shot plain vs 1-shot TTA is one call.
    """
    nested = nested_shot_evidence(treatment, baseline) if nested_shots else None
    masked = tuple(ignore_fields) + (NESTED_SHOT_FIELDS if nested_shots else ())
    if masked:
        # The sanctioned class is model.* — fields that change WHICH pipeline
        # measured, never what was measured: family/image_size for the S14
        # cross-family pairings, tta for the TTA-toggle pairing (same episodes,
        # same support hash, base vs base+TTA on the same weights).
        allowed = {"model.family", "model.image_size", "model.tta"}
        bad = sorted(set(ignore_fields) - allowed)
        if bad:
            raise ValueError(
                f"ignore_fields may only relax {sorted(allowed)} — {bad} would change "
                "what was measured, not which model measured it")
        if (not nested_shots
                and get_path(treatment, "protocol.support_set_hash")
                != get_path(baseline, "protocol.support_set_hash")):
            raise ValueError(
                "cross-family pairing requires identical support draws "
                "(protocol.support_set_hash differs)")
        fields = tuple(f for f in COMPARABILITY_FIELDS if f not in masked)

        def _key(rec):
            payload = json.dumps([get_path(rec, f) for f in fields],
                                 sort_keys=True, default=str)
            return hashlib.sha1(payload.encode()).hexdigest()[:12]

        kt, kb = _key(treatment), _key(baseline)
    else:
        kt, kb = comparability_key(treatment), comparability_key(baseline)
    if kt != kb:
        raise ValueError(
            "refusing to compare records with different comparability keys "
            f"({kt} vs {kb}); differing fields: {diff_fields(treatment, baseline)}"
        )

    dt, db = per_seed_scores(treatment, metric), per_seed_scores(baseline, metric)
    shared = sorted(set(dt) & set(db))
    if not shared:
        raise ValueError("no shared support seeds between the two records")

    d = np.array([dt[s] - db[s] for s in shared], dtype=float)
    boot = paired_bootstrap(d, n_boot=n_boot, seed=seed)
    return {
        "mean_delta": boot["mean_delta"],
        "ci95": boot["ci95"],
        "significant": boot["significant"],
        "n_seeds": len(shared),
        "n_seeds_improved": boot["n_improved"],
        "comparability_key": kt,
        "metric": metric,
        # Non-empty ONLY on a cross-family pairing — the relaxation and both sides'
        # values travel with the number so it can never pass as a within-family delta.
        "ignored_fields": {f: [get_path(treatment, f), get_path(baseline, f)]
                           for f in masked} or None,
        # Non-None ONLY on a shot-count pairing: the per-run prefix proof that licensed
        # masking protocol.shots / support_set_hash, so the delta can never pass as a
        # same-protocol one.
        "nested_shots": nested,
        # The whole reason for pairing: the unpaired spread is much larger than the
        # effect, so reporting it alongside makes the gain in resolution explicit.
        "unpaired_std_treatment": float(np.std(list(dt.values()), ddof=1)) if len(dt) > 1 else 0.0,
        "unpaired_std_baseline": float(np.std(list(db.values()), ddof=1)) if len(db) > 1 else 0.0,
        "paired_std": boot["paired_std"],
    }


def paired_bootstrap(d, n_boot: int = 10_000, seed: int = 0) -> dict:
    """The paired delta with its bootstrap CI, over whatever the pairing unit is.

    Lifted out of `compare` (S17) so the same numbers can be computed when the unit is not
    a support seed — S17 pairs per TRAINING PAIR (one cycle pair scored under two arms).
    `compare` calls this, so the rng sequence and every value it reports are byte-identical
    to what it has always produced; only the key names are unit-neutral here (`n`,
    `n_improved`) where `compare` restates them as `n_seeds` / `n_seeds_improved`.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError(f"paired_bootstrap needs a non-empty 1-D vector, got shape {d.shape}")
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0 or hi < 0),
        "n": int(len(d)),
        "n_improved": int((d > 0).sum()),
        "n_degraded": int((d < 0).sum()),
        "paired_std": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
    }


# ── persistence + reporting ──────────────────────────────────────────────────


def append(rec: dict, path: Path) -> dict:
    """Stamp track/deviations/key onto a record and append it to a JSONL log."""
    dev, track = classify(rec)
    rec = rec | {"deviations": dev, "track": track,
                 "comparability_key": comparability_key(rec)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def load(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line
    return out


def aggregate(records: list[dict], baseline_id: str | None = None) -> str:
    """Markdown table, grouped by comparability key so unlike numbers never share one."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("comparability_key") or comparability_key(r), []).append(r)

    out = []
    for key, recs in sorted(groups.items()):
        head = recs[0]
        out.append(f"\n### {head.get('dataset')} · {get_path(head, 'protocol.shots')}-shot "
                   f"· {len(get_path(head, 'protocol.groups') or [])} group(s) · `{key}`\n")
        out.append("| run | track | mIoU | std | Δ paired | CI95 | deviations |")
        out.append("| :-- | :-- | --: | --: | --: | :-- | :-- |")

        base = next((r for r in recs if r.get("track") == "faithful"), recs[0])
        if baseline_id:
            base = next((r for r in recs if r.get("run_id") == baseline_id), base)

        for r in recs:
            m, s = get_path(r, "results.miou"), get_path(r, "results.std")
            delta = ci = "—"
            if r is not base:
                try:
                    c = compare(r, base)
                    delta = f"{100 * c['mean_delta']:+.2f}"
                    ci = (f"[{100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}]"
                          + ("**" if c["significant"] else ""))
                except ValueError as e:
                    delta, ci = "n/a", str(e)[:40]
            dev = ", ".join(r.get("deviations", [])) or "—"
            out.append(f"| {r.get('run_id', r.get('stage', '?'))} | {r.get('track', '?')} "
                       f"| {100 * m:.2f} | {100 * s:.2f} | {delta} | {ci} | {dev} |")
    return "\n".join(out)
