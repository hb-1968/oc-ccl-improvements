# Butterfly Major — query-level diagnosis (all 500 test queries)

Status: **measured offline, 2026-10-06.** Read-only analysis of banked per-query CSVs, the committed SST masks and
the cached images. Nothing is re-scored, no stage or gate changes, and no test label is altered. Levers in §6 are
**proposed, not run**. Companions: [MXP_DIAGNOSIS.md](MXP_DIAGNOSIS.md), [LATIVITTA_DIAGNOSIS.md](LATIVITTA_DIAGNOSIS.md).

Reproduce: `.venv/bin/python scripts/diagnose_queries.py` → `results/diagnostics/queries.json`. It produces:
- every per-query attribute and score;
- the §3 edge-agreement instrument, for the whole label and per trait, with its shifted control;
- the train split's agreement, as calibration.

Re-running reproduces the session's numbers (3827: 0.727, 15682: 0.377, train median 0.99, p1 0.89, Major 78.75).
The session's intermediate files `label_alignment.json` and `trait_tightness.json` hold the same quantities.

## Cells read

| Key | Bundle | Cell |
| :-- | :-- | :-- |
| `sam2_best5` | S21 | `crown_all_mref_lcpool_5_tta` |
| `sam2_best1` | S21 | `crown_all_mref_lcpool_tta` |
| `sam2_base5` | S21 | `base_lcpool_5_tta` |
| `sam3_best5` | S22 | `crown_all_mref_lcpool_5_tta` |
| `sam3_base5` | S22 | `base_lcpool_5_tta` |

All are TTA with 20 seeds under the label-consistent pool.

**Per-query score:** the mean over a query's GT traits of the per-trait IoU, averaged over the 20 runs; run-to-run
spread is recorded alongside. Major is the mean of the group means.

## 1. What a query's score depends on

- **The query, not the model.** Per-query scores correlate at **0.94–0.995** across SAM 2 and SAM 3, base and
  tuned, 1-shot and 5-shot. The median max−min spread across the five cells is 2.4 points. The same queries are
  hard for every model.
- **Not the supports, at 5-shot.** Per-query run-to-run std has median 0.38 points (p90 1.35). At 1-shot it is 1.76
  (p90 10.7). Only 1.2 % of queries swing by more than 20 points across the 20 draws at 5-shot.
- **The deficit is diffuse, not a tail.**
  - The worst 5 % / 10 % / 20 % / 50 % of queries carry 13 % / 22 % / 34 % / 67 % of the total shortfall (1 − mIoU).
  - Only 10 of 500 queries score below 50, and 49 below 70.
  - 121 already score at or above the paper's Major 83.9.

| Subspecies (`sam2_best5`) | mean | p10 | median | p90 | < 50 | < 70 |
| :-- | --: | --: | --: | --: | --: | --: |
| (m×p)×m | 73.5 | 54.8 | 82.4 | 86.0 | 6 | 35 |
| cyrbia | 79.1 | 73.2 | 80.8 | 84.4 | 1 | 5 |
| lativitta | 74.9 | 71.9 | 75.4 | 78.0 | 1 | 4 |
| malleti | 81.7 | 76.3 | 82.3 | 86.7 | 0 | 1 |
| notabilis × lativitta | 84.6 | 74.9 | 81.3 | **96.7** | 2 | 4 |

## 2. Defective test labels (3 unscoreable, 2 partial)

Found among the worst queries, confirmed by overlay and by the registration measure in §3:

| Query | Subspecies | Defect | Score (every cell) |
| :-- | :-- | :-- | --: |
| 5087 | notabilis × lativitta | GT is effectively empty: one trait blob, 0.0 % foreground | 0.0 |
| 3827 | notabilis × lativitta | all 8 traits present but **not on the wings**; edge agreement 0.73 vs its 40-px-shifted chance level 0.62 | 0.0 |
| 15682 | (m×p)×m (dorsal) | **misregistered**: outlines run off the wings (rays drawn outside the hindwings). Edge agreement 0.38 vs chance 0.35. Plausibly annotated on another photo of specimen CAM017555 (the duplicate photos of the MXP diagnosis) | 10 |
| 5280 | (m×p)×m (dorsal) | partly misregistered (agreement 0.58 vs chance 0.47; rays at IoU 10–12) | 53 |
| 5162 | (m×p)×m (dorsal) | partly misregistered (agreement 0.73 vs 0.48) | 60 |

Without the three unscoreable queries, Major reads **79.22 instead of 78.75 (+0.47)**, for `sam2_best5`, `sam3_best5`
and `sam2_best1` alike. That is a **diagnostic reading only**: the headline stays the locked metric over all 500.

## 3. Annotation style: label tightness decides the score in two subspecies

**Instrument (edge agreement).** For each label (or trait), the share of its boundary pixels that lie within 3 px of
an image edge:
- Canny thresholds 30/90 on the 3×3-Gaussian-blurred grey 1024² image.
- The edge map is dilated with a 7×7 kernel.
- Boundary = the mask minus its 3×3 erosion.
- **Control:** the same boundary shifted by (40, 40) px, which gives the agreement a misregistered label would get
  by chance.
- All constants were set a priori and not tuned.

**Calibration:**
- The train split's labels have median 0.99, p1 0.89.
- The control's median is 0.49, p95 0.71.

**Rejected instrument (recorded so it is not retried):** colour distance from the sheet (Lab, median + k·MAD).
Cream and white traits on a light-grey sheet count as background, rulers and colour cards inflate the spread, and
it scored the two known misregistered labels (0.62) like ordinary ones (median 0.75).

| Subspecies | train median (p5) | test median (p5) |
| :-- | :-- | :-- |
| (m×p)×m | 0.989 (0.925) | 0.988 (0.921) |
| cyrbia | 0.993 (0.902) | 0.967 (0.895) |
| **lativitta** | 0.988 (0.950) | **0.819 (0.653)** |
| malleti | 0.986 (0.943) | 0.995 (0.969) |
| notabilis × lativitta | 0.997 (0.969) | 0.977 (0.923) |

- 88 test labels fall below the train p1 (0.89). Of these, 78 are lativitta: its test-only loose ray envelopes,
  re-detected independently.
- Per trait, IoU rises with label tightness: from 43.7 at agreement < 0.6, to 68.0 at 0.6–0.8, to 81.2 at ≥ 0.995
  (n = 4,161 (query, trait) pairs).
- Within subspecies, dorsal, the correlation is strong only where labels are mixed:
  - lativitta ρ +0.56 (74 % of its trait outlines are loose, below 0.95);
  - notabilis ρ +0.66;
  - cyrbia, malleti and (m×p)×m ρ ≈ 0.

**Notabilis × lativitta is two populations by annotation style.**

| Notabilis test queries | n | `sam2_best5` | `sam2_base5` (untuned) | `sam3_best5` | per-trait IoU | edge agreement |
| :-- | --: | --: | --: | --: | :-- | --: |
| "tight" group (score ≥ 90) | 49 | 95.5 | **96.3** | 92.9 | 92–97 | 0.997 |
| "loose" group (60–90) | 49 | 77.2 | 76.3 | 76.2 | rays 6/8 at 58–60 | 0.949 |

- **No other attribute separates them:** not record (both mostly Zenodo 3082688), specimen family (all `CAM`),
  foreground share (9.8 vs 9.9 %), image IDs (interleaved), or run-to-run spread.
- **The labels look different under zoom.** Tight-group rays are traced pixel-tight along the orange edge;
  loose-group rays are hand-drawn envelopes with straight segments across the black (overlays: crops of
  4908/4651 vs 3276/5374).
- **The scores look like model-assisted annotation.** A per-trait IoU of 92–97 sits above what two hand
  annotations of one frame reach (90–92 per trait; MXP_DIAGNOSIS §5). Untuned SAM 2 scores 96.3 on these queries.
  **Interpretation, not proven:** about half of notabilis's test labels were drawn with SAM-family assistance (SAM
  reproduces SAM masks), and the other half by hand. The dataset documents no provenance.
- **The same mechanism likely operates, milder, elsewhere.** A model that traces visible edges scores ~96 on
  tool-traced labels and ~75–82 on hand-drawn ones. Most of Major's diffuse deficit is this hand-drawing jitter, and
  no model lever reaches it.

## 4. Resolution: thin traits are under-resolved on the 256² mask grid

SAM 2 sees the whole photo at 1024² and decodes masks on a 256² grid.

| Trait width (2 × p90 inside distance, in 256-grid cells; dorsal queries) | IoU | n |
| :-- | --: | --: |
| < 2 | 60.6 | 109 |
| 2–3 | 71.8 | 375 |
| 3–5 | 72.6 | 379 |
| 5–8 | 76.5 | 704 |
| 8–12 | 82.7 | 884 |
| 12–20 | 88.3 | 1,164 |
| ≥ 20 | 87.5 | 213 |

- Width is confounded with trait type (rays are thin, bands wide).
- **The resolution signature:** specimen size in the frame (foreground share) predicts thin-trait IoU within
  cyrbia (ρ +0.32) and malleti (+0.44), more weakly in notabilis (+0.13), and wide-trait IoU not at all (≈ 0). That
  pattern is what a resolution limit, rather than trait identity, predicts.
- Per-query score correlates with foreground share at ρ +0.53 to +0.58 within cyrbia, malleti and (m×p)×m. In
  (m×p)×m this is the ventral and shrivelled specimens: the thin-trait split there is ≈ 0.
- **Zoom headroom is limited.** Cropping the whole specimen (bounding box + 10 %) allows only a **1.25–2.0×** zoom
  (median 1.32–1.42 for four subspecies, 1.92 for cyrbia), because the four dissected wings span the sheet. A
  per-wing (quadrant) crop would allow about 2×.

## 5. Other query attributes (within subspecies)

| Attribute | Effect |
| :-- | :-- |
| CR2 vs JPG | 0 to −5 by group (confounded with session; MXP §5) |
| label-layout displacement | ρ −0.23 to −0.54 |
| foreground contrast | +0.12 to +0.57 |
| background saturation (session: e.g. the green-sheet cyrbia/lativitta sessions, Zenodo 5526257) | −0.05 to −0.43 |
| cyrbia rot180 | 77.6 vs upright 80.0 (TTA recovers most) |
| (m×p)×m Zenodo 2548678 (the shrivelled-specimen session) | 57.6, n = 6 |

## 6. What this means, and levers (proposed, NOT run)

- **Most of the remaining Major deficit is label-side or a property of the query, and no model lever reaches it.**
  - Defective labels (+0.47 if excluded).
  - The (m×p)×m underside (about 1.7).
  - Lativitta's loose rays (about 1.35–1.8).
  - Hand-drawn vs tool-traced annotation style: notabilis's loose half alone sits about 19 points below its tight
    half.
- **The paper faced the same test labels.**
- **(a) Zoom / tiling at inference** — the one model-side lever this diagnosis suggests. **CPU-probed 2026-10-06, pre-registered decision GO (quadrant tiles +1.55 ✱, thin traits +6.5 ✱): [ZOOM_PROBE.md](ZOOM_PROBE.md).** The proposal as written before the probe:
  - A wrapper (like `adapters/tta.py`) crops support and query. Supports are cropped to their GT bounding box. The
    query crop is label-free: a fixed quadrant tiling, or a first-pass prediction's bounding box.
  - The crop is predicted at 1024 and the logits are pasted back into the frame.
  - Expected: thin traits gain from more decoder cells. The headroom is ~1.4× per specimen and ~2× per quadrant, so
    the gain is modest, plausibly +0.3 to +1 Major.
  - It needs a CPU probe first, like the MXP probe: full frame vs tiled, per-trait IoU, on cyrbia and malleti
    dorsal queries.
- **(b) Evaluation hygiene — RUN 2026-10-06, see §7:** report two **diagnostic** Major readings beside the locked
  headline:
  - Major without the 3 unscoreable queries (the headline never changes);
  - a label-style-stratified table: notabilis tight vs loose, and lativitta rays.
- **(c) Query supervision** (outside pure OC-CCL; the user's decision) would not help the annotation-style or
  defective-label parts either. It could reach only the (m×p)×m underside.

## 7. Diagnostic readings (run, 2026-10-06)

Status: **measured offline**, read-only. §6(b) carried out. Tables: [DIAGNOSTIC_READS.md](DIAGNOSTIC_READS.md),
generated by `.venv/bin/python scripts/report_diagnostic_reads.py`, which needs `results/inbox/{s18b,s19,s19b,s20,s21,s22,s23}.zip`
and this diagnosis' `results/diagnostics/queries.json`. Code: `src/occl/diagnostic_reads.py`; tests:
`tests/test_diagnostic_reads.py`. **The headline is unchanged: every stage still reports the locked argmax mIoU over
all 500 queries. These readings sit beside it and are never a selection criterion.**

**Method.**
- **Input:** each landed cell's banked `per_query.csv` (per (group, run, query, trait) argmax IoU, 6 decimals) and its
  `record.json`. Twenty cells: the programme's butterfly ladder (SAM 2 S18b/S19/S19b/S21/S23, SAM 3 S20/S22), all
  TTA with 20 seeds.
- **Locked recomputation, checked first:** per query, the mean IoU over its GT traits; per group, the mean over its
  queries; per run, the mean over the five groups; per cell, the mean over runs. It must reproduce the record's
  per-run Major (`registry.per_seed_scores`) to 1e-6. The tolerance is set by the CSV's 6-decimal rounding. The
  observed worst miss over the 20 cells is 1.7e-8.
- **Exclusion:** the three unscoreable queries of §2 (5087, 3827, 15682) are dropped from their groups' means before
  the group → run → cell means. The partial ones (5280, 5162) are kept. The list was fixed from §2, after the test
  labels were inspected, and is disclosed as post hoc. It is the same list for every cell.
- **Strata:** the per-query rule is restricted to a subset of queries, or for a trait stratum to the stratum's traits
  present in each query's GT. It is averaged per run, then over runs, always without the three defective queries.
  - notabilis tight/loose is LABEL-ONLY: whole-label edge agreement (§3) ≥ `TIGHT_EDGE` = 0.985. That threshold was
    set at the gap in notabilis's test distribution (0.984 → 0.988) after inspection.
    - It gives 47 tight and 51 loose.
    - It agrees with §3's score-defined split (≥ 90) on 94 of 98 queries. The 4 disagreements: 4097 is tight but
      scores 69; 19071, 19081 and 19088 are loose-ish (0.95–0.98) but score 91–95.
  - lativitta rays are traits 7/8.
  - (m×p)×m sides come from the filename tag: 67 dorsal after excluding 15682, and 32 ventral.
- **Contrasts:** each stage's headline paired contrast is re-read per support seed with the registry bootstrap
  (10,000 resamples, seed 0, unadjusted), both locked and with the exclusion. It is also split into subspecies
  contributions (group Δ ÷ 5, summing to the Major Δ) and into stratum deltas.
  - The POOL contrasts re-draw malleti and (m×p)×m (`STRIP_GROUPS`), so those components are effectively unpaired.
  - S22 POOL compares two SAM 3 allocations. SAM 3 is not bit-reproducible across them, so the ±0.00 ✱ in its
    unchanged groups are hardware noise, not effects.

**Findings.**
- **The exclusion is a uniform level shift.** It adds +0.45 to +0.47 to every cell. The three queries score
  0 / 0 / about 10 in all 20 cells, so they are model-independent.
  - Best SAM 2 at 5-shot: 78.75 → **79.22**. SAM 3 pooled: 78.35 → 78.82. SAM 2 at 1-shot: 77.31 → 77.78.
  - The gap to the paper's 83.9 shrinks from 5.15 to 4.68.
- **No conclusion changes.** All 10 stage contrasts keep their sign and significance under the exclusion, and every
  delta moves by at most 0.01 (DIAGNOSTIC_READS §2).
- **Fine-tuning trades label styles in notabilis and nets zero there.**
  - The S19/S19b wings lose 0.65–0.69 ✱ on tight (tool-traced-looking) labels and gain 0.67–0.70 ✱ on loose
    (hand-drawn) ones, so their notabilis contribution is +0.01.
  - The tuned gain is in lativitta's rays (+5.0 to +5.4 ✱), the (m×p)×m underside (+4.4 to +4.5 ✱), malleti and
    cyrbia.
  - Interpretation, not proven: the cycle's anchors are mostly hand-drawn train masks, so tuning moves the model's
    boundaries toward hand-drawing and away from edge-exact tracing.
- **SAM 3's butterfly deficit is almost entirely notabilis.** At 5-shot (S22, pooled), notabilis contributes
  −0.36 of the −0.40 ✱. Within it, tight labels lose −2.60 ✱ and loose ones −1.06 ✱. SAM 3 is *better* on
  lativitta's rays (+1.2 ✱) and the (m×p)×m underside (+0.9 ✱).
  - If §3's interpretation holds (tool-traced labels made with a SAM-family model), SAM 2 reproducing them better
    than SAM 3 would fit. This is not proven: the dataset documents no annotation tool.
- **S23 SHAPE does what a boundary term should, on the wrong labels.** It gains +0.52 ✱ on notabilis tight labels,
  where boundaries are edge-exact. It loses on everything hand-drawn: loose −0.10 ✱, lativitta rays −0.93 ✱.
- **The strata gaps remain the size of the deficit.** For the best cell:
  - notabilis tight − loose is +16.8;
  - lativitta rays − other traits is −35.5, with best-of-3 +5.3, constant across every cell;
  - (m×p)×m ventral − dorsal is −27.7, against −33.5 for base. Tuning and POOL closed about 6 points of it.

**Limitations.**
- The exclusion list and `TIGHT_EDGE` were chosen after inspecting the test labels, so they describe this test set
  and nothing beyond it.
- The strata are descriptive: they are not causal, and they are not a corrected metric. A stratum-weighted or
  "clean-label" Major is deliberately not computed, because it would amount to choosing the test set.
