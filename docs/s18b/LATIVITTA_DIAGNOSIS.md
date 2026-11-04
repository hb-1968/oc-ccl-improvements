# Butterfly Major `lativitta` — why it sits at ~75

Status: **measured offline, 2026-09-29.** Read-only analysis of the locked, banked cells, the
committed SST masks and the cached images; nothing is re-scored and no stage or gate changes.
One new local CPU probe (§4) is a diagnostic: it enters no record and no run bank. Levers
(§6) are **proposed, not run**. Companion: [MXP_DIAGNOSIS.md](MXP_DIAGNOSIS.md), same method.

Reproduce:
* `.venv/bin/python scripts/diagnose_lativitta.py` (about 10 min) →
  `results/diagnostics/lativitta.json`.
* `OCCL_SST_ROOT=$PWD/SST .venv/bin/python scripts/probe_lativitta_rays.py` (CPU, about
  19 min) → `results/diagnostics/lativitta_probe.json`.

## 1. Question, and the claim this revises

Lativitta is the second-lowest subspecies in every landed cell: 72–76 against malleti's
78–82 and notabilis × lativitta's 82–85. It moved little under fine-tuning and SAM 3, and
not at all with more shots. The README (S5b, "Where the residual now lives") filed it with
(m×p)×m as a "*silhouette-quality* deficit … that no support curation can touch".

The observation stands: its `iou_binary` of 71–74 sits beside its mIoU. The cause is
narrower than silhouette quality, and it is in the labels. **Lativitta's test split outlines
the hindwing rays (traits 7/8) under a different convention from its train split.**

Unlike (m×p)×m, this is **not** a two-population mixture:
* The per-query distribution is unimodal: 5th–90th percentile 70.5–78.0 (SAM 2
  crown_all_mref 5-shot).
* Specimen family (`CAM…` vs `19N…` vs `CS…`), image format and Zenodo record all score
  alike, within about 1 point.
* The one ventral test specimen (17589, crumpled forewing) scores 40, and is the only
  outlier.

## 2. Observation: the deficit is two traits, and it is boundary-shaped

Per-trait IoU (argmax, TTA, 20 seeds):

| Cell | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** | mIoU |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| SAM 2 base, 1-shot | 82 | 85 | 78 | 86 | 83 | 78 | **44** | **41** | 72.15 |
| SAM 2 crown_all_mref, 5-shot | 83 | 86 | 81 | 87 | 85 | 81 | **50** | **46** | 74.86 |
| SAM 3 crown_all_mref, 1-shot | 82 | 86 | 82 | 86 | 84 | 81 | **54** | **47** | 75.30 |
| SAM 3 crown_all_mref, 5-shot | 84 | 85 | 82 | 86 | 85 | 81 | **53** | **46** | 75.30 |

What the traits are, read off overlays:
* 7/8: orange hindwing rays, left and right.
* 1/5: hindwing costal bar.
* 3/6: forewing cream band.
* 2/4: red forewing band.

The same structures in notabilis × lativitta are traits 6/8 (rays), 5/7, 1/4 and 2/3. Its
rays score 71–75 and its other traits 84–91.

**Six of lativitta's eight traits are at notabilis level (81–87). The rays are 41–54.**
Fine-tuning and SAM 3 gained most on the rays (+5 to +10), but nothing closed them.

The scoring variants isolate the kind of error:

| Cell | argmax | independent | oracle | best-of-3 boundary |
| :-- | --: | --: | --: | --: |
| SAM 2 crown_all_mref, 5-shot | 74.86 | 74.87 | 74.87 | **80.20 (+5.34)** |
| SAM 3 crown_all_mref, 5-shot | 75.30 | 75.30 | 75.30 | **80.64 (+5.34)** |

* Independent binarisation and the oracle assignment change nothing. It is not overlap and
  not trait confusion.
* Best-of-3 (per image, the best of the prediction as-is, one-pixel dilation or one-pixel
  erosion, at native 1024 px; `occl.metrics.boundary_variants`) adds **+5.3 in every cell,
  SAM 2 or SAM 3, base or tuned**.
* In every other subspecies it adds +0.0 to +0.5.
* A one-pixel tolerance paying 5 points, with no dependence on the model, says the
  predictions are systematically too thin relative to the ground truth.

## 3. Cause: the test split's rays are loose envelopes; the train split's are tight traces

### 3a. What the labels cover

Measured per test image:
* **Width:** twice the 90th-percentile distance transform at 1024 px.
* **Colour:** the share of label pixels that are orange in HSV. The rule is hue ≤ 22 or
  ≥ 170, saturation ≥ 110, value ≥ 90; it was set by eye and not fitted.

| Test split, dorsal | Ray width (px) | Ray label pixels that are orange | Orange hindwing pixels inside the ray label |
| :-- | --: | --: | --: |
| lativitta | **28.6** | **42 %** | 98 % |
| notabilis × lativitta | 12.3 | 86 % | 67 % |
| malleti | 9.3 | 86 % | 82 % |
| (m×p)×m | 10.1 | 80 % | 84 % |

Zoomed overlays show a hand-drawn envelope around each lativitta ray. It takes in a wide
black margin, and toward the wing base it merges neighbouring rays and the black between
them into one blob. Notabilis's ray labels hug the orange.

A model that segments exactly the visible rays is capped near **IoU ≈ 0.42** on these
traits. The observed 41–54 is at that ceiling, not below it.

### 3b. The convention belongs to the split, not the specimen

**Label colour share by split** (median over images with a cached image):

| Lativitta trait | train `CAM…` | train `19N…` | test `CAM…` | test `19N…` |
| :-- | --: | --: | --: | --: |
| rays | 84 % | 76 % | **45 %** | **40 %** |
| red band | 80 % | 78 % | 70 % | 71 % |
| cream band (pale rule) | 75 % | 78 % | 62 % | 66 % |

**Shape across every dorsal mask** (mask-only, so no cache bias). Test/train ratio of trait
share of the foreground, and of boundary complexity (perimeter / √area, lower = smoother):

* Lativitta rays: share **1.42× and 1.59×**, complexity **0.65×** for both.
* Lativitta's other traits: share 0.82–0.90×, which is renormalisation because the rays
  grew, and complexity 0.97–1.12×.
* Every other subspecies: share 0.87–1.18× and complexity 0.96–1.43×, with no ray-specific
  pattern.

**Loose-ray classifier.** A mask counts as loose when the ray-union complexity is below
22.9, which is 1.2× the test median. The threshold was chosen after seeing the bimodal
split: train median 29.2, test 19.1.

| Lativitta dorsal masks | Loose | Tight |
| :-- | --: | --: |
| test | **96** | 3 |
| train re-shoots of a test specimen (same specimen and side) | 1 | **64** |
| other train | 56 | 1,600 |

Image IDs overlap completely between the splits (train quartiles 19,409 / 21,618 / 39,671;
test 19,684 / 21,950 / 39,796). The convention is not an ID batch. The same butterfly,
re-photographed, is traced tightly in train and outlined loosely in test. SST's dataset
README gives no annotation provenance.

A milder version of the same shift shows in every subspecies. Test ray labels are 83–89 %
orange against 92–98 % in train for notabilis, malleti `CAM…` and (m×p)×m. Only
lativitta's is large enough to cost a trait half its IoU.

### 3c. Why nothing we tried could fix it

* Every support comes from train, so every prompt is a tight ray trace. Propagation follows
  image edges, and the rays' edges are the orange/black boundary.
* The fine-tune trains on train pairs, which are tight on both sides.
* No legitimate training signal anywhere in the pipeline carries the test envelope
  convention. It isn't in the supports, the fit set or the validation set.
* A method could only match it by being told the convention, which is test-label
  information, or by being generous at boundaries. The best-of-3 reading is generous at
  boundaries, which is why it alone moves lativitta.

## 4. Causal probe: are the rays under-segmented, not missed?

**Design.**
* Base SAM 2, 1-shot, no TTA, CPU fp32, run with `scripts/probe_lativitta_rays.py`.
* Supports: the first support of the locked S18b base runs 0 and 1 (39423 and 14406).
* Queries: the first 40 dorsal lativitta test images, 80 episodes in all.
* Per trait: precision (predicted pixels inside the GT), recall, and IoU after a k-pixel
  dilation of the prediction (3×3 kernel, k iterations).
* Written before the run: if the rays are under-covered rather than missed, precision stays
  high, recall is near 0.42, and a few pixels of dilation recover most of the IoU.
* Wall time: about 19 min on 16 CPU threads.

| Trait | Precision | Recall | IoU | +1 px | +2 px | +4 px | +6 px | +8 px |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 costal bar (L) | 95.1 | 85.0 | 81.3 | 82.1 | 81.0 | 76.4 | 71.5 | 66.8 |
| 2 red band (R) | 97.8 | 87.0 | 85.2 | 87.0 | 87.2 | 84.6 | 80.5 | 76.0 |
| 3 cream band (L) | 96.0 | 77.9 | 77.7 | 83.3 | **86.7** | 85.0 | 77.9 | 70.4 |
| 4 red band (L) | 97.5 | 88.7 | 86.7 | 88.2 | 88.0 | 84.5 | 79.7 | 75.0 |
| 5 costal bar (R) | 92.7 | 89.9 | 83.8 | 84.7 | 83.6 | 78.3 | 72.1 | 66.4 |
| 6 cream band (R) | 99.3 | 79.8 | 79.3 | 84.3 | **87.4** | 86.3 | 79.4 | 71.9 |
| **7 rays (L)** | **98.4** | **45.1** | 45.0 | 58.3 | 69.4 | **77.1** | 71.0 | 64.0 |
| **8 rays (R)** | **98.0** | **42.9** | 42.5 | 55.4 | 66.2 | **74.9** | 70.2 | 64.0 |

**Reading.**
* **The rays are found, not missed.** 98 % of predicted ray pixels lie inside the label, but
  they cover only 43–45 % of it. That is the 0.42 ceiling from §3a.
* **Dilation recovers the rays.** Dilating the predicted rays by 4 px lifts their IoU from
  43.8 to 76.0, and it falls again past 4 px. The test envelope is roughly the visible ray
  plus a 4-px margin.
* **The cream band (3/6) is looser in test too, to a lesser degree.** It peaks at +2 px
  (+9). This matches its colour share (62–66 % in test against 75–79 % in train).
* **The probe reproduces the banked tolerance lift.** Image mIoU goes from 72.69 as-is to
  77.91 with one-pixel dilation (+5.2), against the banked cells' best-of-3 +5.3. The
  dilation arm is what wins.
* **Every trait has precision of 93–99 %.** The model's predictions sit inside the
  labels everywhere; recall is the whole deficit.
* The +4 px optimum was measured on **test** labels. It is a diagnostic of the convention,
  not a tuned post-process, and must not be applied (§6).

## 5. What this does to Major and the paper comparison

* **Rays at notabilis level (about 75).** Lativitta goes from 74.86 to about 81.6
  (+6.8), and Major from 78.56 to about 79.9 (+1.35), for SAM 2 crown_all_mref 5-shot.
* **Rays at lativitta's own other-trait level (about 84).** Lativitta goes to about 83.9,
  and Major gains +1.8.
* **With (m×p)×m's ventral half ([MXP_DIAGNOSIS.md](MXP_DIAGNOSIS.md), +1.87).** The two
  label-side findings explain about 3.2–3.7 of the 5.3-point 5-shot gap to the paper's
  83.9.
* **The paper faced the same test masks.** Its metric is unpublished. If it includes the
  best-of-3 tolerance (the author recollection), that alone lifts lativitta by 5.3, and our
  paper-closest reading is 80.2. The scoring detail matters most on exactly this subspecies.

## 6. Levers (proposed, NOT run)

Scoring is locked. Nothing below edits `metrics`, `protocol`, `curation`, `seeding`,
`evaluator` or `adapters/sst_sam2`, and nothing re-runs a landed cell.

1. **Report, don't chase.** The test envelope convention is not learnable from any training
   data. Tuning a dilation or envelope on test would be test-set tuning. The honest outputs
   are:
   * this diagnosis;
   * a per-trait table that shows rays separately;
   * the best-of-3 column (already recorded) as the tolerance-aware secondary.
2. **A train-side prior, only if one existed.** A per-subspecies ray-dilation constant
   chosen on train or validation would choose ≈ 0, because train rays are tight. So it
   cannot help without test knowledge. It is recorded to close the idea.
3. **Disclosure.** Any Major number is partly a function of which annotation pass scored it.
   The shift should accompany a Major claim, as the cyrbia rotation and the (m×p)×m ventral
   half do.

## 7. Methods and constants

* **Cells read:** the same eight as MXP_DIAGNOSIS: SAM 2 and SAM 3, base and
  crown_all_mref, 1-shot and 5-shot, all TTA. Scoring variants come from each run's
  `miou_variants` and `miou_boundary`.
* **Specimen family:** the specimen ID with digits masked (`CAM#`, `#N#`, `CS#`). Specimen
  ID, view and format are parsed as in MXP_DIAGNOSIS §8.
* **Colour rules** (`diagnose_lativitta.ORANGE`, `PALE`) are HSV heuristics set by eye on the
  overlays and not fitted. Colour shares use cached images only, which covers all test
  images and 265 lativitta train images (the ones drawn as supports or fit images). The
  mask-only shape ratios cover every dorsal mask and confirm the same shift without
  that bias.
* **Constants:** `LOOSE_CPX = 22.9` was chosen post hoc (see §3b). The 1024-px width and
  distance-transform choices follow MXP_DIAGNOSIS.
* **No resampling CIs.** These are descriptive decompositions of banked means and mask
  statistics.
