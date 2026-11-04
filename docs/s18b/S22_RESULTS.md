# S22 evidence read — POOL (label-consistent supports) on SAM 3, butterfly

Source: results/inbox/s22.zip. Nothing is selected; this report only reads. Methods: [S22_PLAN.md](S22_PLAN.md).

Primary metric: argmax mIoU; best-of-3 boundary secondary. All cells TTA, plain SAM 3 tracker, 20 support seeds. CIs: 10,000 resamples over support seeds, seed 0, unadjusted; ✱ = excludes zero. POOL pairs each pooled cell with its locked S20 curated cell per support SEED (`seed_paired`): identical draws outside malleti and (m×p)×m, re-drawn inside them (effectively unpaired).

Screen gate: PASS — S20 89221b789b9e; parent bound; cells not yet demanded.
Confirmation gate: PASS — S20 89221b789b9e; all 8 pooled cells complete.

## 1-shot

| Adapter | curated (S20) | label-consistent | POOL Δ Major | (m×p)×m | malleti | untouched 3 (band: 0) | best-of-3 Δ |
| :-- | --: | --: | :-- | :-- | :-- | :-- | :-- |
| base | 76.50 | 76.51 | +0.01 [-0.22, +0.23] | +0.09 [-0.86, +0.96] | -0.08 [-0.66, +0.49] | +0.01 [+0.00, +0.02] ✱ | +0.00 [-0.22, +0.22] |
| crown_all | 77.10 | 77.13 | +0.03 [-0.24, +0.30] | -0.10 [-1.35, +1.08] | +0.24 [-0.30, +0.80] | +0.01 [-0.00, +0.03] | +0.04 [-0.23, +0.30] |
| crown_all_mref | 77.38 | 77.42 | +0.04 [-0.19, +0.25] | -0.02 [-1.02, +0.83] | +0.21 [-0.28, +0.72] | -0.00 [-0.00, +0.00] | +0.04 [-0.17, +0.24] |
| odd_cf_all_mref | 77.25 | 77.31 | +0.05 [-0.18, +0.28] | +0.02 [-0.96, +0.85] | +0.26 [-0.25, +0.80] | -0.00 [-0.01, +0.00] | +0.06 [-0.17, +0.28] |

| Cell (per subspecies, mIoU %) | cyrbia | lativitta | malleti | (m×p)×m | notab×lat |
| :-- | --: | --: | --: | --: | --: |
| base_tta | 77.44 | 74.24 | 79.94 | 70.86 | 80.01 |
| base_lcpool_tta | 77.44 | 74.25 | 79.86 | 70.95 | 80.02 |
| crown_all_tta | 77.89 | 75.20 | 79.98 | 70.62 | 81.80 |
| crown_all_lcpool_tta | 77.92 | 75.20 | 80.22 | 70.52 | 81.80 |
| crown_all_mref_tta | 78.00 | 75.30 | 80.06 | 71.91 | 81.65 |
| crown_all_mref_lcpool_tta | 78.00 | 75.30 | 80.27 | 71.89 | 81.65 |
| odd_cf_all_mref_tta | 77.97 | 75.49 | 79.85 | 71.56 | 81.41 |
| odd_cf_all_mref_lcpool_tta | 77.97 | 75.49 | 80.11 | 71.58 | 81.40 |

## 5-shot

| Adapter | curated (S20) | label-consistent | POOL Δ Major | (m×p)×m | malleti | untouched 3 (band: 0) | best-of-3 Δ |
| :-- | --: | --: | :-- | :-- | :-- | :-- | :-- |
| base | 77.96 | 78.07 | +0.11 [+0.03, +0.19] ✱ | +0.71 [+0.35, +1.09] ✱ | -0.15 [-0.36, +0.07] | -0.00 [-0.01, +0.00] | +0.10 [+0.01, +0.18] ✱ |
| crown_all | 78.00 | 78.02 | +0.02 [-0.10, +0.16] | +0.13 [-0.45, +0.77] | -0.12 [-0.36, +0.12] | +0.02 [-0.00, +0.06] | +0.00 [-0.11, +0.14] |
| crown_all_mref | 78.20 | 78.35 | +0.16 [+0.07, +0.25] ✱ | +0.88 [+0.46, +1.30] ✱ | -0.08 [-0.31, +0.14] | -0.00 [-0.01, +0.00] | +0.14 [+0.05, +0.24] ✱ |
| odd_cf_all_mref | 78.19 | 78.34 | +0.15 [+0.07, +0.23] ✱ | +0.83 [+0.46, +1.21] ✱ | -0.07 [-0.31, +0.16] | -0.00 [-0.00, +0.00] | +0.14 [+0.06, +0.22] ✱ |

| Cell (per subspecies, mIoU %) | cyrbia | lativitta | malleti | (m×p)×m | notab×lat |
| :-- | --: | --: | --: | --: | --: |
| base_5_tta | 79.58 | 74.22 | 81.27 | 71.80 | 82.92 |
| base_lcpool_5_tta | 79.59 | 74.21 | 81.12 | 72.52 | 82.91 |
| crown_all_5_tta | 79.39 | 75.26 | 80.98 | 71.63 | 82.74 |
| crown_all_lcpool_5_tta | 79.46 | 75.26 | 80.86 | 71.76 | 82.74 |
| crown_all_mref_5_tta | 79.54 | 75.30 | 81.05 | 72.24 | 82.85 |
| crown_all_mref_lcpool_5_tta | 79.53 | 75.30 | 80.97 | 73.12 | 82.85 |
| odd_cf_all_mref_5_tta | 79.58 | 75.45 | 80.96 | 72.08 | 82.86 |
| odd_cf_all_mref_lcpool_5_tta | 79.58 | 75.45 | 80.88 | 72.91 | 82.86 |

## (m×p)×m by photographed side

| Cell | dorsal (68) | ventral (32) | ventral 5/8 | ventral 6/9 | ventral 2/3 |
| :-- | --: | --: | --: | --: | --: |
| base_5_tta | 80.96 | 52.35 | 22.18 | 42.36 | 59.62 |
| base_lcpool_5_tta | 80.96 | 54.57 | 23.27 | 46.19 | 66.12 |
| base_lcpool_tta | 79.90 | 51.95 | 21.85 | 41.04 | 62.17 |
| base_tta | 80.35 | 50.71 | 21.17 | 39.46 | 57.73 |
| crown_all_5_tta | 80.94 | 51.85 | 22.65 | 41.99 | 60.44 |
| crown_all_lcpool_5_tta | 80.99 | 52.15 | 22.97 | 44.27 | 64.45 |
| crown_all_lcpool_tta | 79.81 | 50.78 | 22.28 | 40.05 | 63.76 |
| crown_all_mref_5_tta | 80.99 | 53.66 | 22.89 | 43.64 | 62.82 |
| crown_all_mref_lcpool_5_tta | 81.01 | 56.35 | 24.07 | 47.76 | 70.71 |
| crown_all_mref_lcpool_tta | 80.01 | 54.64 | 23.84 | 43.51 | 67.99 |
| crown_all_mref_tta | 80.53 | 53.59 | 23.52 | 42.72 | 62.85 |
| crown_all_tta | 80.41 | 49.80 | 22.45 | 39.51 | 58.09 |
| odd_cf_all_mref_5_tta | 81.03 | 53.07 | 22.43 | 43.25 | 61.19 |
| odd_cf_all_mref_lcpool_5_tta | 81.04 | 55.64 | 23.50 | 47.40 | 68.57 |
| odd_cf_all_mref_lcpool_tta | 80.00 | 53.68 | 22.47 | 43.32 | 65.11 |
| odd_cf_all_mref_tta | 80.52 | 52.50 | 22.09 | 42.12 | 59.97 |

## SAM 3 − SAM 2 under the same label-consistent pool (identical draws)

| Adapter | 1-shot SAM 2 → SAM 3 | Δ | 5-shot SAM 2 → SAM 3 | Δ |
| :-- | :-- | :-- | :-- | :-- |
| base | 75.24 → 76.51 | +1.26 [+0.63, +2.03] ✱ | 77.26 → 78.07 | +0.80 [+0.70, +0.91] ✱ |
| crown_all | — | — | — | — |
| crown_all_mref | 77.31 → 77.42 | +0.11 [-0.35, +0.73] | 78.75 → 78.35 | -0.40 [-0.48, -0.31] ✱ |
| odd_cf_all_mref | 77.08 → 77.31 | +0.22 [-0.32, +0.96] | 78.55 → 78.34 | -0.21 [-0.31, -0.11] ✱ |
