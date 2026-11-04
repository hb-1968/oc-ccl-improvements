# Butterfly Major `(malleti x plesseni) x malleti` — why it sits at ~72

Status: **measured offline, 2026-09-29.** Read-only analysis of the locked, banked cells.
Nothing is re-scored and no stage or gate changes. The one new inference is a local CPU
probe (§4). It is a diagnostic: it enters no record and no run bank.
Levers in §7 are **proposed, not run**.

Reproduce:
`.venv/bin/python scripts/diagnose_mxp.py --annotation-agreement` (bundle and mask reads,
about 5 min) and
`OCCL_SST_ROOT=$PWD/SST .venv/bin/python scripts/probe_mxp_views.py` (CPU SAM 2, about 1 h).
Outputs are `results/diagnostics/mxp_views.json` and `results/diagnostics/mxp_probe.json`.

## 1. Question, and the claim this supersedes

(m×p)×m is the lowest butterfly subspecies in every landed cell: about 72 against 79–85
for the others. This doesn't change with the shot count, the SAM 2 fine-tunes or SAM 3.
Earlier readings that this analysis supersedes:

* **README "Where the residual now lives" (S5b):** (m×p)×m runs "at its silhouette
  limit", a "*silhouette-quality* deficit … that no support curation can touch". The
  observation (mIoU ≈ `iou_binary`) stands. The interpretation (a uniform mask-quality
  limit) is wrong. The subspecies is a mixture of two populations (§2), and the
  silhouette score is low only because one of them is.
* **Memory note "Butterfly Major support analysis":** "Dorsal/ventral mixing … dead end,
  do not re-explore." That check only ruled out a left/right label swap between views.
  It never split the score by view, and that split is the whole effect.

This is the same failure class as cyrbia's rotated 37%. A subspecies mean hid a
bimodal population. The per-subspecies headroom column couldn't show it, because both
populations sit near their own silhouette.

## 2. Observation: two populations, split by the photographed side

View is parsed from the image filename (`_d` / `_v`, regex `_([dv])[_.]`). The
(m×p)×m test split is **68 dorsal / 32 ventral**. In the other four subspecies,
ventral test images number 0–1 of 100 (cyrbia 0, lativitta 1, malleti 1, notabilis 0).
Per-query mIoU is averaged over the query's GT-present traits, then over queries and
runs (the locked metric). Per-query CSVs are from the banked cells.

| Cell (all TTA, 20 seeds) | Dorsal (68) | Ventral (32) | Subspecies |
| :-- | --: | --: | --: |
| SAM 2 base, 1-shot | 79.98 | 45.72 | 69.0 |
| SAM 2 crown_all_mref, 1-shot | 81.42 | 53.23 | 72.4 |
| SAM 2 base, 5-shot | 80.71 | 48.26 | 70.3 |
| SAM 2 crown_all_mref, 5-shot | **81.98** | 52.69 | 72.6 |
| SAM 3 base, 1-shot | 80.35 | 50.71 | 70.9 |
| SAM 3 crown_all_mref, 1-shot | 80.53 | 53.59 | 71.9 |
| SAM 3 base, 5-shot | 80.96 | 52.35 | 71.8 |
| SAM 3 crown_all_mref, 5-shot | 80.99 | 53.66 | 72.2 |

* **Dorsal (m×p)×m is not a weak subspecies.** It scores 80–82, level with malleti's
  81–82 (its backcross parent, same 10-trait vocabulary), in every cell.
* **Ventral is where every intervention landed, and it saturates.** SAM 2 fine-tuning adds
  +7.5 on ventral queries at 1-shot, against +1.4 on dorsal. SAM 3's untuned base adds +5.0
  over SAM 2's base on ventral queries. All tuned cells stop at 52.7–53.7. A ceiling that
  neither the tracker nor the recipe crosses is a signature of the labels, not of model
  capacity.
* **The rest of Major pays the same view penalty.** The lone ventral test specimen scores
  54–67 in malleti (four tuned cells) and 40–43 in lativitta (5-shot cells), against
  80–82 and about 75 for their dorsal specimens. Only (m×p)×m has enough ventral
  specimens to move a mean.

## 3. Why ventral queries fail

### 3a. The same trait ID names a different structure on the underside

The traits are **colour-pattern elements**, not wing regions. The IDs are fixed by
position, but the underside pattern differs. Compare the dorsal and ventral photos of one
specimen, CAM017890 (train 5702 dorsal, test 16815 ventral):

| Trait IDs | Dorsal structure | Ventral structure (test convention) |
| :-- | :-- | :-- |
| 1 / 4 | forewing white-yellow patch | the same patch |
| 2 / 3 | orange forewing band | pale band plus the pale-pink trailing (inner-margin) area |
| 5 / 8 | large cream costal band across the hindwing top | thin dark costal streak |
| 6 / 9 | red basal band | small red basal spots |
| 7 / 10 | orange rays | paler, thinner rays |

Measured GT areas agree (test split, % of the image, ventral/dorsal ratio):

* 5/8: 0.25× and 0.24×.
* 6/9: 0.51× and 0.48×.
* 7/10: 0.61× and 0.55×.
* 1/4: 0.71× and 0.68×.
* 2/3: 1.12× and 1.13×.
* Whole foreground: 6.74 % ventral against 9.81 % dorsal.

Per-trait IoU by query view (SAM 2 crown_all_mref 5-shot; SAM 3 1-shot is within ±8):

| Trait | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Dorsal | 86 | 90 | 89 | 86 | 84 | 78 | 73 | 83 | 78 | 71 |
| Ventral | 81 | 65 | 61 | 82 | **24** | **36** | 60 | **24** | **37** | 62 |

The traits whose meaning changes between sides (2/3, 5/8, 6/9) carry the deficit. The
traits that look the same on both sides (1/4) don't. With 87% of the supports dorsal
(§3b), the model is asked to find a cream band on the ventral side, where the label marks
a thin streak.

### 3b. The train pool annotates the underside three different ways

Ventral masks are classified by one heuristic, chosen **after** looking at the masks (it
is not a measured property):

* Fewer than 8 of 10 traits: **incomplete**.
* Otherwise, traits (5 + 8) as a share of the foreground below 0.18: **strip**. This is
  the convention every complete ventral test mask follows; the test range is 5.3–15.0 %.
* Everything else: **wide**, where the whole hindwing, or most of it, is labelled 5 or 8.

| Split | Dorsal | Ventral strip | Ventral wide | Ventral incomplete | Empty |
| :-- | --: | --: | --: | --: | --: |
| test | 68 | 31 | 0 | 1 | 0 |
| train | 300 | 33 | 43 | 27 | 1 |
| canonical support pool (drawn from) | 287 | 20 | 24 | — | — |
| S18b fit pairs (refs) | 127 | 12 | 10 | 5 | — |

The test split is internally consistent. The train split is not: only 33 of 104 ventral
masks match the test convention. The completeness curation removed the incomplete ones,
but it can't tell strip from wide.

Two consequences follow:

* **1-shot:** a ventral query almost always gets a dorsal support (19 of 20 runs), so the
  semantic mismatch in §3a applies.
* **Adding ventral supports does not help, and slightly hurts.** Ventral queries score
  54.6 with no ventral support among the five, 50.6 with one, and 53.0 with two (SAM 2
  crown_all_mref 5-shot). Over half of the ventral supports in the pool teach the wrong
  convention. The one 1-shot run with a ventral support (run 15, support 16315, a 26 % share,
  so classified wide) gave ventral queries 49.4 against 53.4 with dorsal supports.

The fine-tune's ventral pairs mix all three conventions, so its +7.5 ventral gain was
learned against contradictory targets.

### 3c. Three shrivelled specimens

Three ventral test specimens have a foreground below 3 % of the image. They are
crumpled or very small wings: 17091 (CAM017959, 0.87 %,
mIoU 15.9), 18452 (CAM018510, 0.76 %, 35.7) and 16403 (CAM017786, 1.88 %, 22.4).
Train has the same kind of specimen with one-blob masks carrying one or two trait labels
(for example 18456 and 18496, both CAM0185xx). Without
these three, ventral queries average about 55.6 instead of 52.7. They are 3 of 100
queries, about 0.9 points of the subspecies.

## 4. Causal probe: does a matching ventral support rescue ventral queries?

**Design.**
* Base SAM 2, 1-shot, no TTA, CPU fp32, run with `scripts/probe_mxp_views.py`.
* Five fixed supports × (32 ventral + 8 dorsal test queries).
* Dorsal supports D_run0 (15926, CR2) and D_run1 (5172, JPG): the first supports of the
  locked S18b base runs 0 and 1.
* Ventral supports: V_strip_a (17924, CR2), V_strip_b (16683, JPG) and V_wide (16938, CR2).
  These are the first cached train masks of each convention.
* The prediction was set before the run: if a view- and convention-matched support
  rescued ventral queries, the deficit would be a support-selection problem.
* Wall time: 46 min on 16 CPU threads.
* The banked `results/diagnostics/mxp_probe.json` (sha256 prefix `bd8d8c9f`) was produced
  by a scratchpad twin of the script, written before the script was moved into the repo.
  It has the same supports, queries and scoring calls, plus an extra `gt_area` field. The
  repo script has not been re-run end to end.

| Support | Ventral queries | Dorsal queries | Ventral per-trait IoU (1…10) |
| :-- | --: | --: | :-- |
| D_run0 | 48.96 | 80.46 | 81 39 66 81 **25 30** 58 **22 32** 60 |
| D_run1 | 48.98 | 80.53 | 80 39 53 82 **26 35** 58 **28 34** 60 |
| V_strip_a | 50.33 | 68.60 | 82 66 67 81 **26 21** 56 **25 28** 57 |
| V_strip_b | 50.21 | 68.40 | 82 62 68 82 **26 12** 57 **25 37** 56 |
| V_wide | 50.15 | 67.57 | 82 62 64 80 **14 25** 56 **26 39** 57 |

(D_run0 and D_run1 have near-equal means but differ per query, with a mean absolute
difference of 2.31 points. The per-query mIoUs are not identical.)

**Reading.**
* **Matching the support's side and convention is worth about +1.2 on ventral queries,
  and costs dorsal queries about 12.** It is not the lever.
* It does repair traits 2/3. With a ventral support, trait 2 rises from 39 to 62–66,
  because the underside label includes the pale trailing area (§3a). That part is
  convention.
* Traits 5/8 and 6/9 stay at 12–39 whichever support is used. The model cannot find the
  ventral costal streak and basal spots even when shown them.
* **Width is not the reason.** Measured as twice the 90th-percentile distance transform
  at 1024 px (test median):
  * Ventral 5/8 is 18–22 px wide.
  * The dorsal rays 7/10 are 7–10 px wide, and ventral rays 7 px.
  * The rays score 56–74 regardless.
  * So the mask head's 256² grid, where 20 px is 5 cells, is not the floor.
* **What remains is visual definition.** The ventral streak is dark purple-brown on a brown
  ground and the spots are tiny. The rays are orange on black or brown. Whether annotators
  themselves agree on the ventral streak can't be tested here (§5, annotation agreement).
* This probe is 1-shot base SAM 2 without TTA. It says nothing about a fine-tune trained on
  consistent cross-view pairs (lever 1 in §7), which is the untested route for 5/8 and 6/9.

## 5. Secondary factors, and two checks on our own evidence

* **CR2-sourced images (confounded; not a clean decode effect).**
  * Unpaired, dorsal CR2 queries trail dorsal JPG queries by 3.4–3.7 in (m×p)×m and by
    0.6–2.4 in the other subspecies.
  * Paired within one specimen and side: 20 test specimens appear in the test split as both
    a CR2 and a JPG image. CR2 minus JPG is:
    * −1.89 (SEM 0.58), SAM 2 crown_all_mref 5-shot;
    * −1.26 (0.46), SAM 3 crown_all_mref 5-shot;
    * −1.89 (0.59), SAM 2 base 1-shot.
  * **These pairs are not the same frame.** Their silhouette IoU has a median of 0.85, only
    1 of 20 is at least 0.9, and some are 0.006. The images are re-shoots, often from a
    different Zenodo record or session. The gap therefore mixes the rawpy decode with photo
    session, and the two masks are separate annotations of different frames.
  * An early draft of this note called them "the same photo". That was wrong, and it is
    corrected here.
  * 132 of Major's 500 test images are CR2, so this factor is worth at most about 0.3–0.5 of
    Major. SST decodes the same way.
* **Train/test specimen overlap in SST's committed split.**
  * The duplicate key is the specimen ID: the filename up to the first `_`, e.g.
    `CAM017480` or `19N0527`, plus the view.
  * An earlier pass parsed IDs with a regex that collapsed every `19N…` specimen into one
    key. That inflated lativitta and malleti; the numbers below are the corrected ones.
  * Test specimen and side also present in train:

    | Subspecies | Test specimens in train | Same frame | In the S18b fit set |
    | :-- | --: | --: | --: |
    | cyrbia | 57/100 | 0 | 2 |
    | lativitta | 67/100 | 0 | 7 |
    | malleti | 37/100 | 0 | 9 |
    | (m×p)×m | 83/100 | 0 | **42** |
    | notabilis × lativitta | 10/100 | 4 | 5 |

  * "Same frame" means the silhouette IoU with that train mask is at least 0.9. The S18b fit
    set is the one every SAM 2 and SAM 3 fine-tune here trained on.
  * So the leak is specimen-level (the same wings re-photographed), not image-level. It is
    substantial in the fine-tune only for (m×p)×m.
* **The specimen leak does not inflate the fine-tune gains.**
  * Method: the paired per-query gain (tuned − base, same support draws, averaged over 20
    runs), compared between test queries whose specimen and side is in the fit set and
    those whose isn't.
  * Within (m×p)×m it is smaller for seen specimens, not larger:
    * SAM 2 1-shot: +2.88 vs +3.75, [−2.44, +0.75];
    * SAM 2 5-shot: +2.06 vs +2.44;
    * SAM 3 1-shot: +0.60 vs +1.36, [−1.55, −0.02];
    * SAM 3 5-shot: +0.18 vs +0.63.
  * No subspecies shows a significantly larger gain on seen specimens.
  * Pooled over Major, the in-fit set is 65 queries dominated by (m×p)×m, so the pooled read
    is confounded by subspecies. Within-subspecies is the read that counts.
  * Uncertainty: 2,000-sample query bootstraps, unadjusted. In-fit and unseen queries are not
    randomised, so their baseline difficulty can differ.
  * Direct score effect: a support was the query's own specimen and side in 5 of 2,000
    (m×p)×m 1-shot episodes and 30 of 2,000 at 5-shot. Those episodes scored 80–84 against
    72, which is negligible in the mean.
  * The overlap should still be disclosed next to the memorization probe and any
    generalisation claim.
* **Annotation agreement.** Two masks of the *same frame* (a CR2 and a JPG of one shot,
  silhouette IoU at least 0.9, both within one split) are two annotations of one image:
  * (m×p)×m dorsal: 78 pairs, per-trait agreement 87–97 (1: 90, 2: 94, 3: 95, 4: 90, 5: 97,
    6: 89, 7: 87, 8: 97, 9: 93, 10: 87).
  * malleti dorsal: 97 pairs, 79–95.
  * Dorsal labels are therefore reproducible to about 90–92 mIoU; our dorsal 80–82 sits
    about 10 below.
  * **Only one ventral pair is frame-aligned**, so annotator agreement on the ventral streak
    can't be measured from this data. That pair's per-trait agreement is 74–95.

## 6. What this does to the paper comparison

The paper's supports and per-subspecies scores are unpublished. The test split is the
same, so the paper faced the same 32 ventral queries. Arithmetic on our cells:

| Cell | Major | (m×p)×m | Major if ventral scored like dorsal |
| :-- | --: | --: | --: |
| SAM 2 crown_all_mref 5-shot (best butterfly) | 78.56 | 72.6 | 80.43 (+1.87) |
| SAM 3 crown_all_mref 1-shot | 77.38 | 71.9 | 79.10 (+1.72) |

The ventral half explains about 1.8 of the 5.3-point 5-shot gap to the paper's 83.9. The
CR2-sourced images add at most about 0.3–0.5, and that figure is confounded (§5). Lativitta
(about 75) is the largest remaining piece and has not been diagnosed this way yet. Whether
the paper's (m×p)×m also sat near 72 is unknowable from the paper. If it did, the paper's
other four subspecies must have averaged about 83.5 at 1-shot.

## 7. Levers (proposed, NOT run)

Scoring is locked. Nothing below edits `metrics`, `protocol`, `curation`, `seeding`,
`evaluator` or `adapters/sst_sam2`, and nothing re-runs a landed cell.

1. **View-consistent fine-tune data (training side, allowed).** Drop the wide and
   incomplete ventral masks from the fit and validation pairs, keeping dorsal→ventral
   pairs with strip labels. The model can only learn the mapping in §3a
   (dorsal cream band ↔ ventral thin streak) from consistent cross-view pairs. Only this
   lever addresses traits 5/8 and 6/9, which the probe (§4) shows no support recovers.
   Falsified if ventral queries stay at or below 54 with dorsal supports.
2. **Convention-aware support pool (new selection mode in a wrapper, like
   `adapters/orientation.py`).** Exclude wide and incomplete ventral masks from the pool.
   This is legitimate at any shot count, because it curates labels the way completeness
   curation does. The probe bounds it: even a perfectly matched ventral support is worth
   about +1.2 on ventral queries, so at most about +0.4 on the subspecies.
3. **View-routed supports.** Match the support's side to the query's, inferred from the
   image (ventral ground colour is brown, dorsal is black; no GT needed). This needs a
   labelled support of each side, which is **two labels per subspecies, so it is not a
   1-shot result**. At 5-shot it is a support-composition rule and would be reported as
   such. It has the same probe bound (about +1.2 on ventral queries), and it costs about 12
   on any dorsal query routed to a ventral support by mistake.
4. **CR2 decode with camera white balance, or the existing CR2→JPG substitution in
   `imagecache.fetch`.** This changes the data path (`data.cache_format` is a
   comparability field), so it would be a new, labelled comparison, never pooled with
   landed cells.

## 8. Methods and constants

* **Cells read:**
  * SAM 2: `s18b/base_tta`, `s19/base_5_tta`, `s19b/crown_all_mref_tta` and
    `s19b/crown_all_mref_5_tta`.
  * SAM 3: `s20/{base,crown_all_mref}_{tta,5_tta}`.
  * Unit: per-query rows `(group, run, query, trait, iou)`; 20 runs × 100 queries per
    subspecies.
* **View and format** are parsed from the manifest URL's filename. `?` means no `_d`/`_v`
  token (case-insensitive); it occurs only outside (m×p)×m. `.jpeg` is folded into `jpg`.
  The **specimen ID** is the filename up to the first `_` (for example `CAM017480` or
  `19N0527`). The duplicate key is (specimen ID, view).
  A first pass used the leading `[A-Za-z]*\d+` instead, which truncated `19N…` IDs to
  `19` and merged hundreds of lativitta and malleti specimens. It was found and fixed
  before these numbers were recorded.
* **Area shares** are pixel fractions of the 1024² committed masks.
* **Constants:**
  * The convention classifier's `STRIP_MAX = 0.18` and `MIN_TRAITS = 8` were set after
    viewing the test range (5.3–15.0 %) and the train histogram (0–100 %).
  * `TINY_FG = 0.03`.
  * `ALIGNED = 0.9` is the silhouette IoU above which two masks count as one frame. It was
    set after seeing a bimodal pair distribution (median 0.61, upper mode 0.91–0.94).
  * All four are heuristics, not measured probabilities.
* **Trait width** is twice the 90th-percentile Euclidean distance transform inside each
  trait mask at 1024 px; the median over test specimens is reported.
* **No resampling CIs** are given for the view split. It is a descriptive decomposition
  of banked means. The CR2 pair differences quote a SEM over specimen pairs (n = 20).
  The seen-versus-unseen gain comparison uses 2,000-sample query bootstraps, seed 0,
  unadjusted.
* **Copy baseline by view:** tried and rejected as an instrument here. Pasting a support's
  label map onto a query is dominated by wing placement (a ventral query copies better
  from a dorsal support, 21.0, than from a strip ventral one, 16.0). It matches the
  earlier finding that the copy proxy correlates only r = 0.44 with mIoU.
