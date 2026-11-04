# S20 evidence read — the SAM 2 picks on SAM 3, with SAM 3's concept wings

Source: results/inbox/s20.zip. Nothing is selected; this report only reads.

Methods and heuristics: [S20_PLAN.md](S20_PLAN.md). Primary metric: argmax mIoU; best-of-3 boundary is the secondary column. All cells TTA, 20 paired support seeds.
CIs: 10,000 paired support-seed resamples, seed 0, unadjusted; ✱ = excludes zero. Wing deltas pair a wing cell with the plain-tracker cell on identical draws (`ignore_fields=model.concept_wing`); shot lifts use the prefix-nested pairing; interactions and additivity are per-seed differences of differences. Cross-family rows pair SAM 3 with SAM 2 on identical support draws (`ignore_fields=model.family, model.image_size`). Probe deltas pair per episode.

## butterfly_major

Screen gate: PASS — 886 fit / 221 held-out; eligible=['crown_all', 'crown_all_mref', 'odd_cf_all_mref']; SAM 3 fits screened; test cells not yet demanded.
Confirmation gate: PASS — 886 fit / 221 held-out; eligible=['crown_all', 'crown_all_mref', 'odd_cf_all_mref']; 32 cells and 8 probe cells complete.

Resumed under an accepted code-hash change: [('50b4fca1ba38', 'd033b682f6a0')].

| SAM 3 fit | ‖ΔW‖ | Fit min | Peak GB | Val mIoU % (base) | Cut-out occupancy % (base) | Translation % (base) | Eligible |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| crown_all | 6.244 | 8.2 | 16.3 | 78.65 (78.05) | 33.58 (20.81) | 92.04 (90.39) | yes |
| crown_all_mref | 5.640 | 17.1 | 22.7 | 78.64 (78.05) | 30.93 (20.81) | 91.93 (90.39) | yes |
| odd_cf_all_mref | 5.910 | 29.1 | 42.2 | 78.45 (78.05) | 9.70 (20.81) | 91.93 (90.39) | yes |

### 1-shot

| Adapter | none | presence | exemplar | stack | Δ vs base (none) | best-of-3 (none) |
| :-- | --: | --: | --: | --: | :-- | --: |
| base | 76.50 | 76.50 | 75.36 | 75.36 | — | 77.90 |
| crown_all | 77.10 | 77.10 | 75.54 | 75.54 | +0.60 [+0.36, +0.85] ✱ | 78.40 |
| crown_all_mref | 77.38 | 77.38 | 76.87 | 76.87 | +0.89 [+0.70, +1.09] ✱ | 78.68 |
| odd_cf_all_mref | 77.25 | 77.25 | 76.66 | 76.66 | +0.76 [+0.58, +0.94] ✱ | 78.51 |

| Wing effect (wing − none) | base | crown_all | crown_all_mref | odd_cf_all_mref |
| :-- | :-- | :-- | :-- | :-- |
| presence | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| exemplar | -1.14 [-2.15, -0.37] ✱ | -1.56 [-2.85, -0.47] ✱ | -0.51 [-1.00, -0.10] ✱ | -0.59 [-1.11, -0.17] ✱ |
| stack | -1.14 [-2.15, -0.37] ✱ | -1.56 [-2.85, -0.47] ✱ | -0.51 [-1.00, -0.10] ✱ | -0.59 [-1.11, -0.17] ✱ |
| additivity (stack − none) − [(presence − none) + (exemplar − none)] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

| Wing × pick interaction ((pick_w − pick) − (base_w − base)) | crown_all | crown_all_mref | odd_cf_all_mref |
| :-- | :-- | :-- | :-- |
| presence | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| exemplar | -0.42 [-1.12, +0.13] | +0.62 [+0.08, +1.37] ✱ | +0.54 [+0.00, +1.33] ✱ |
| stack | -0.42 [-1.12, +0.13] | +0.62 [+0.08, +1.37] ✱ | +0.54 [+0.00, +1.33] ✱ |

crown_all_mref − crown_all: +0.29 [+0.17, +0.41] ✱; odd_cf_all_mref − crown_all_mref: -0.13 [-0.16, -0.10] ✱.

### 5-shot

| Adapter | none | presence | exemplar | stack | Δ vs base (none) | best-of-3 (none) |
| :-- | --: | --: | --: | --: | :-- | --: |
| base | 77.96 | 77.96 | 77.39 | 77.39 | — | 79.39 |
| crown_all | 78.00 | 78.00 | 78.11 | 78.11 | +0.04 [-0.05, +0.13] | 79.32 |
| crown_all_mref | 78.20 | 78.20 | 78.36 | 78.36 | +0.24 [+0.21, +0.26] ✱ | 79.51 |
| odd_cf_all_mref | 78.19 | 78.19 | 78.31 | 78.31 | +0.23 [+0.20, +0.25] ✱ | 79.47 |

| Wing effect (wing − none) | base | crown_all | crown_all_mref | odd_cf_all_mref |
| :-- | :-- | :-- | :-- | :-- |
| presence | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| exemplar | -0.57 [-0.79, -0.37] ✱ | +0.11 [-0.07, +0.27] | +0.16 [+0.03, +0.27] ✱ | +0.13 [-0.00, +0.24] |
| stack | -0.57 [-0.79, -0.37] ✱ | +0.11 [-0.07, +0.27] | +0.16 [+0.03, +0.27] ✱ | +0.13 [-0.00, +0.24] |
| additivity (stack − none) − [(presence − none) + (exemplar − none)] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

| Wing × pick interaction ((pick_w − pick) − (base_w − base)) | crown_all | crown_all_mref | odd_cf_all_mref |
| :-- | :-- | :-- | :-- |
| presence | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| exemplar | +0.68 [+0.45, +0.93] ✱ | +0.73 [+0.51, +0.97] ✱ | +0.70 [+0.49, +0.93] ✱ |
| stack | +0.68 [+0.45, +0.93] ✱ | +0.73 [+0.51, +0.97] ✱ | +0.70 [+0.49, +0.93] ✱ |

crown_all_mref − crown_all: +0.20 [+0.11, +0.28] ✱; odd_cf_all_mref − crown_all_mref: -0.01 [-0.03, +0.02].

| Adapter | Shot lift (none) | Shot interaction ((pick−base)₅ − (pick−base)₁) |
| :-- | :-- | :-- |
| base | +1.46 [+1.13, +1.83] ✱ | — |
| crown_all | +0.90 [+0.60, +1.28] ✱ | -0.56 [-0.83, -0.29] ✱ |
| crown_all_mref | +0.81 [+0.56, +1.13] ✱ | -0.65 [-0.85, -0.46] ✱ |
| odd_cf_all_mref | +0.93 [+0.66, +1.26] ✱ | -0.53 [-0.70, -0.37] ✱ |

| Memorization probe (5-shot) | Cut-out occupancy soft % | hard % | Others mIoU % | Δ vs base, pp | Δ presence − none, pp | Episodes |
| :-- | --: | --: | --: | :-- | :-- | --: |
| base @ none | 23.44 | 23.29 | 77.61 | — | — | 1497 |
| crown_all @ none | 32.36 | 31.59 | 77.24 | +8.93 [+7.98, +9.86] ✱ | — | 1497 |
| crown_all_mref @ none | 32.68 | 31.88 | 77.87 | +9.24 [+8.39, +10.13] ✱ | — | 1497 |
| odd_cf_all_mref @ none | 15.80 | 15.99 | 77.97 | -7.64 [-8.67, -6.62] ✱ | — | 1497 |
| base @ presence | 23.44 | 23.29 | 77.61 | — | +0.00 [+0.00, +0.00] | 1497 |
| crown_all @ presence | 32.36 | 31.59 | 77.24 | +8.93 [+7.98, +9.86] ✱ | +0.00 [+0.00, +0.00] | 1497 |
| crown_all_mref @ presence | 32.68 | 31.88 | 77.87 | +9.24 [+8.39, +10.13] ✱ | +0.00 [+0.00, +0.00] | 1497 |
| odd_cf_all_mref @ presence | 15.80 | 15.99 | 77.97 | -7.64 [-8.67, -6.62] ✱ | +0.00 [+0.00, +0.00] | 1497 |

| Cross-family (SAM 3 − SAM 2, plain tracker, TTA) | 1-shot SAM 2 → SAM 3 | Δ [95% CI] | 5-shot SAM 2 → SAM 3 | Δ [95% CI] |
| :-- | :-- | :-- | :-- | :-- |
| base | 75.18 → 76.50 | +1.32 [+0.67, +2.11] ✱ | 77.23 → 77.96 | +0.73 [+0.65, +0.82] ✱ |
| crown_all | 77.16 → 77.10 | -0.06 [-0.59, +0.66] | 78.53 → 78.00 | -0.53 [-0.65, -0.41] ✱ |
| crown_all_mref | 77.20 → 77.38 | +0.19 [-0.31, +0.86] | 78.56 → 78.20 | -0.37 [-0.43, -0.30] ✱ |
| odd_cf_all_mref | 76.89 → 77.25 | +0.37 [-0.20, +1.15] | 78.43 → 78.19 | -0.24 [-0.31, -0.18] ✱ |

## beetle

Screen gate: PASS — 96 fit / 24 held-out; eligible=['crown_all', 'crown_all_mref', 'odd_cf_all_mref']; SAM 3 fits screened; test cells not yet demanded.
Confirmation gate: PASS — 96 fit / 24 held-out; eligible=['crown_all', 'crown_all_mref', 'odd_cf_all_mref']; 32 cells and 8 probe cells complete.

Resumed under an accepted code-hash change: [('50b4fca1ba38', 'd033b682f6a0')].

| SAM 3 fit | ‖ΔW‖ | Fit min | Peak GB | Val mIoU % (base) | Cut-out occupancy % (base) | Translation % (base) | Eligible |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| crown_all | 2.036 | 0.6 | 9.4 | 55.56 (49.04) | 36.39 (29.91) | 88.78 (88.70) | yes |
| crown_all_mref | 1.667 | 1.2 | 13.0 | 57.16 (49.04) | 36.17 (29.91) | 88.76 (88.70) | yes |
| odd_cf_all_mref | 1.606 | 2.0 | 23.0 | 56.94 (49.04) | 36.01 (29.91) | 88.76 (88.70) | yes |

### 1-shot

| Adapter | none | presence | exemplar | stack | Δ vs base (none) | best-of-3 (none) |
| :-- | --: | --: | --: | --: | :-- | --: |
| base | 66.32 | 66.32 | 54.44 | 54.44 | — | 67.19 |
| crown_all | 69.71 | 69.71 | 58.50 | 58.50 | +3.39 [+2.38, +4.39] ✱ | 70.45 |
| crown_all_mref | 70.19 | 70.19 | 58.53 | 58.53 | +3.87 [+2.97, +4.78] ✱ | 70.92 |
| odd_cf_all_mref | 70.13 | 70.13 | 58.50 | 58.50 | +3.81 [+2.92, +4.73] ✱ | 70.86 |

| Wing effect (wing − none) | base | crown_all | crown_all_mref | odd_cf_all_mref |
| :-- | :-- | :-- | :-- | :-- |
| presence | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| exemplar | -11.88 [-14.13, -9.70] ✱ | -11.21 [-13.55, -8.93] ✱ | -11.66 [-14.07, -9.31] ✱ | -11.63 [-14.00, -9.28] ✱ |
| stack | -11.88 [-14.13, -9.70] ✱ | -11.21 [-13.55, -8.93] ✱ | -11.66 [-14.07, -9.31] ✱ | -11.63 [-14.00, -9.28] ✱ |
| additivity (stack − none) − [(presence − none) + (exemplar − none)] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

| Wing × pick interaction ((pick_w − pick) − (base_w − base)) | crown_all | crown_all_mref | odd_cf_all_mref |
| :-- | :-- | :-- | :-- |
| presence | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| exemplar | +0.67 [-0.45, +1.90] | +0.22 [-0.98, +1.53] | +0.25 [-0.98, +1.59] |
| stack | +0.67 [-0.45, +1.90] | +0.22 [-0.98, +1.53] | +0.25 [-0.98, +1.59] |

crown_all_mref − crown_all: +0.48 [+0.13, +0.80] ✱; odd_cf_all_mref − crown_all_mref: -0.07 [-0.19, +0.07].

### 5-shot

| Adapter | none | presence | exemplar | stack | Δ vs base (none) | best-of-3 (none) |
| :-- | --: | --: | --: | --: | :-- | --: |
| base | 77.96 | 77.96 | 72.52 | 72.52 | — | 78.51 |
| crown_all | 78.51 | 78.51 | 72.43 | 72.43 | +0.55 [+0.13, +0.99] ✱ | 78.98 |
| crown_all_mref | 78.95 | 78.95 | 73.17 | 73.17 | +0.99 [+0.58, +1.38] ✱ | 79.38 |
| odd_cf_all_mref | 78.76 | 78.76 | 73.16 | 73.16 | +0.81 [+0.40, +1.22] ✱ | 79.19 |

| Wing effect (wing − none) | base | crown_all | crown_all_mref | odd_cf_all_mref |
| :-- | :-- | :-- | :-- | :-- |
| presence | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| exemplar | -5.44 [-6.47, -4.46] ✱ | -6.08 [-6.95, -5.20] ✱ | -5.78 [-6.70, -4.86] ✱ | -5.60 [-6.55, -4.64] ✱ |
| stack | -5.44 [-6.47, -4.46] ✱ | -6.08 [-6.95, -5.20] ✱ | -5.78 [-6.70, -4.86] ✱ | -5.60 [-6.55, -4.64] ✱ |
| additivity (stack − none) − [(presence − none) + (exemplar − none)] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

| Wing × pick interaction ((pick_w − pick) − (base_w − base)) | crown_all | crown_all_mref | odd_cf_all_mref |
| :-- | :-- | :-- | :-- |
| presence | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| exemplar | -0.64 [-1.25, +0.08] | -0.34 [-1.01, +0.45] | -0.16 [-0.76, +0.57] |
| stack | -0.64 [-1.25, +0.08] | -0.34 [-1.01, +0.45] | -0.16 [-0.76, +0.57] |

crown_all_mref − crown_all: +0.43 [+0.28, +0.61] ✱; odd_cf_all_mref − crown_all_mref: -0.18 [-0.44, +0.00].

| Adapter | Shot lift (none) | Shot interaction ((pick−base)₅ − (pick−base)₁) |
| :-- | :-- | :-- |
| base | +11.64 [+9.85, +13.43] ✱ | — |
| crown_all | +8.80 [+7.42, +10.26] ✱ | -2.84 [-3.89, -1.82] ✱ |
| crown_all_mref | +8.75 [+7.26, +10.26] ✱ | -2.88 [-3.82, -2.01] ✱ |
| odd_cf_all_mref | +8.64 [+7.21, +10.10] ✱ | -3.00 [-4.04, -2.07] ✱ |

| Memorization probe (5-shot) | Cut-out occupancy soft % | hard % | Others mIoU % | Δ vs base, pp | Δ presence − none, pp | Episodes |
| :-- | --: | --: | --: | :-- | :-- | --: |
| base @ none | 19.96 | 16.20 | 72.35 | — | — | 1200 |
| crown_all @ none | 22.68 | 17.27 | 72.39 | +2.72 [+1.93, +3.55] ✱ | — | 1200 |
| crown_all_mref @ none | 22.39 | 17.45 | 72.60 | +2.42 [+1.72, +3.14] ✱ | — | 1200 |
| odd_cf_all_mref @ none | 22.21 | 17.21 | 72.69 | +2.25 [+1.56, +2.98] ✱ | — | 1200 |
| base @ presence | 19.96 | 16.20 | 72.35 | — | +0.00 [+0.00, +0.00] | 1200 |
| crown_all @ presence | 22.68 | 17.27 | 72.39 | +2.72 [+1.93, +3.55] ✱ | +0.00 [+0.00, +0.00] | 1200 |
| crown_all_mref @ presence | 22.39 | 17.45 | 72.60 | +2.42 [+1.72, +3.14] ✱ | +0.00 [+0.00, +0.00] | 1200 |
| odd_cf_all_mref @ presence | 22.21 | 17.21 | 72.69 | +2.25 [+1.56, +2.98] ✱ | +0.00 [+0.00, +0.00] | 1200 |

| Cross-family (SAM 3 − SAM 2, plain tracker, TTA) | 1-shot SAM 2 → SAM 3 | Δ [95% CI] | 5-shot SAM 2 → SAM 3 | Δ [95% CI] |
| :-- | :-- | :-- | :-- | :-- |
| base | 63.86 → 66.32 | +2.46 [+0.62, +4.54] ✱ | 75.15 → 77.96 | +2.80 [+2.20, +3.49] ✱ |
| crown_all | 64.94 → 69.71 | +4.77 [+3.40, +6.27] ✱ | 75.50 → 78.51 | +3.01 [+2.31, +3.71] ✱ |
| crown_all_mref | 65.31 → 70.19 | +4.88 [+3.37, +6.60] ✱ | 75.81 → 78.95 | +3.14 [+2.55, +3.70] ✱ |
| odd_cf_all_mref | 65.24 → 70.13 | +4.89 [+3.43, +6.60] ✱ | 75.86 → 78.76 | +2.90 [+2.27, +3.48] ✱ |
