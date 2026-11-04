# S21 evidence read — label-consistent OC-CCL on butterfly (SAM 2)

Source: results/inbox/s21.zip. Nothing is selected; this report only reads. Methods: [S21_PLAN.md](S21_PLAN.md).

Primary metric: argmax mIoU; best-of-3 boundary is the secondary column. All cells TTA, 20 support seeds. CIs: 10,000 paired resamples over support seeds, seed 0, unadjusted; ✱ = excludes zero. CLEAN pairs share draws (`registry.compare`). POOL and STACK pair by support SEED (`s21_contract.seed_paired`): draws are identical outside malleti and (m×p)×m and fully re-drawn inside them (the locked protocol permutes the smaller pool), so those two groups are effectively unpaired.

Screen gate: PASS — S19b 0a43787b221c; dropped 20; eligible=['crown_all_mref_lc', 'odd_cf_all_mref_lc']; fits screened; test cells not yet demanded.
Confirmation gate: PASS — S19b 0a43787b221c; dropped 20; eligible=['crown_all_mref_lc', 'odd_cf_all_mref_lc']; all 14 cells and both probes complete.

Resumed under an accepted code-hash change: [('f1d3a738109d', '271ba35d74cc')].

CLEAN manifest `7573573b904a` = S18b `a797e94851eb` with 20 masks spliced out ({'v_incomplete': 2, 'v_wide': 15, 'tiny': 3}); fit 870 pairs, validation 217.

| CLEAN fit | ‖ΔW‖ | Fit min | Val mIoU % (base, CLEAN split) | Eligible |
| :-- | --: | --: | --: | :-- |
| crown_all_mref_lc | 6.595 | 12.5 | 82.21 (80.22) | yes |
| odd_cf_all_mref_lc | 6.894 | 20.6 | 81.79 (80.22) | yes |

## 1-shot

| Adapter | curated pool | label-consistent pool | POOL Δ (all) | POOL Δ (m×p)×m | best-of-3 (curated) |
| :-- | --: | --: | :-- | :-- | --: |
| base | 75.18 | 75.24 | +0.06 [-0.23, +0.37] | +0.22 [-1.01, +1.40] | 76.69 |
| crown_all_mref | 77.20 | 77.31 | +0.11 [-0.11, +0.33] | +0.33 [-0.87, +1.45] | 78.48 |
| odd_cf_all_mref | 76.89 | 77.08 | +0.20 [-0.01, +0.39] | +0.40 [-0.72, +1.40] | 78.15 |
| crown_all_mref_lc | 77.21 | 77.31 | +0.10 [-0.10, +0.30] | +0.30 [-0.81, +1.36] | 78.48 |
| odd_cf_all_mref_lc | 76.78 | 76.99 | +0.21 [-0.02, +0.44] | +0.43 [-0.72, +1.47] | 78.04 |

| CLEAN read | Δ vs parent (all) | (m×p)×m | malleti | best-of-3 | vs base | STACK vs parent | additivity |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| crown_all_mref_lc vs crown_all_mref | +0.01 [-0.03, +0.05] | -0.09 [-0.26, +0.05] | +0.05 [-0.05, +0.17] | +0.01 [-0.03, +0.05] | +2.03 [+1.79, +2.29] ✱ | +0.11 [-0.09, +0.31] | -0.01 [-0.07, +0.04] |
| odd_cf_all_mref_lc vs odd_cf_all_mref | -0.11 [-0.16, -0.06] ✱ | -0.31 [-0.48, -0.16] ✱ | -0.13 [-0.27, -0.03] ✱ | -0.11 [-0.16, -0.06] ✱ | +1.60 [+1.42, +1.82] ✱ | +0.10 [-0.11, +0.31] | +0.02 [-0.05, +0.08] |

| Cell (per subspecies, mIoU %) | cyrbia | lativitta | malleti | (m×p)×m | notab×lat |
| :-- | --: | --: | --: | --: | --: |
| base_tta | 73.19 | 72.15 | 78.10 | 69.02 | 83.45 |
| base_lcpool_tta | 73.19 | 72.15 | 78.19 | 69.23 | 83.45 |
| crown_all_mref_tta | 74.86 | 74.46 | 80.34 | 72.40 | 83.94 |
| crown_all_mref_lcpool_tta | 74.86 | 74.46 | 80.57 | 72.73 | 83.94 |
| odd_cf_all_mref_tta | 74.67 | 74.27 | 80.05 | 71.55 | 83.90 |
| odd_cf_all_mref_lcpool_tta | 74.69 | 74.27 | 80.61 | 71.95 | 83.90 |
| crown_all_mref_lc_tta | 74.92 | 74.45 | 80.39 | 72.30 | 83.97 |
| crown_all_mref_lc_lcpool_tta | 74.92 | 74.45 | 80.60 | 72.60 | 83.97 |
| odd_cf_all_mref_lc_tta | 74.58 | 74.27 | 79.91 | 71.23 | 83.89 |
| odd_cf_all_mref_lc_lcpool_tta | 74.58 | 74.27 | 80.54 | 71.66 | 83.89 |

## 5-shot

| Adapter | curated pool | label-consistent pool | POOL Δ (all) | POOL Δ (m×p)×m | best-of-3 (curated) |
| :-- | --: | --: | :-- | :-- | --: |
| base | 77.23 | 77.26 | +0.04 [-0.05, +0.12] | +0.37 [+0.01, +0.72] ✱ | 78.71 |
| crown_all_mref | 78.56 | 78.75 | +0.19 [+0.08, +0.29] ✱ | +0.89 [+0.43, +1.36] ✱ | 79.83 |
| odd_cf_all_mref | 78.43 | 78.55 | +0.12 [+0.01, +0.23] ✱ | +0.55 [+0.03, +1.05] ✱ | 79.69 |
| crown_all_mref_lc | 78.60 | 78.76 | +0.16 [+0.07, +0.26] ✱ | +0.76 [+0.37, +1.18] ✱ | 79.87 |
| odd_cf_all_mref_lc | 78.45 | 78.51 | +0.07 [-0.03, +0.15] | +0.30 [-0.14, +0.71] | 79.71 |

| CLEAN read | Δ vs parent (all) | (m×p)×m | malleti | best-of-3 | vs base | STACK vs parent | additivity |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| crown_all_mref_lc vs crown_all_mref | +0.04 [+0.01, +0.07] ✱ | +0.11 [-0.01, +0.25] | +0.03 [+0.03, +0.04] ✱ | +0.04 [+0.01, +0.07] ✱ | +1.38 [+1.30, +1.45] ✱ | +0.20 [+0.10, +0.30] ✱ | -0.03 [-0.07, +0.01] |
| odd_cf_all_mref_lc vs odd_cf_all_mref | +0.02 [-0.01, +0.05] | +0.02 [-0.09, +0.13] | +0.03 [+0.02, +0.04] ✱ | +0.02 [-0.01, +0.05] | +1.22 [+1.16, +1.29] ✱ | +0.09 [-0.02, +0.19] | -0.05 [-0.10, -0.01] ✱ |

| Cell (per subspecies, mIoU %) | cyrbia | lativitta | malleti | (m×p)×m | notab×lat |
| :-- | --: | --: | --: | --: | --: |
| base_5_tta | 78.41 | 72.66 | 80.15 | 70.33 | 84.58 |
| base_lcpool_5_tta | 78.41 | 72.66 | 79.98 | 70.69 | 84.58 |
| crown_all_mref_5_tta | 79.10 | 74.86 | 81.63 | 72.61 | 84.61 |
| crown_all_mref_lcpool_5_tta | 79.10 | 74.86 | 81.67 | 73.50 | 84.61 |
| odd_cf_all_mref_5_tta | 79.18 | 74.79 | 81.53 | 72.07 | 84.56 |
| odd_cf_all_mref_lcpool_5_tta | 79.21 | 74.78 | 81.57 | 72.62 | 84.57 |
| crown_all_mref_lc_5_tta | 79.14 | 74.87 | 81.66 | 72.72 | 84.62 |
| crown_all_mref_lc_lcpool_5_tta | 79.14 | 74.87 | 81.69 | 73.47 | 84.62 |
| odd_cf_all_mref_lc_5_tta | 79.27 | 74.77 | 81.56 | 72.09 | 84.55 |
| odd_cf_all_mref_lc_lcpool_5_tta | 79.27 | 74.77 | 81.59 | 72.39 | 84.55 |

## (m×p)×m by photographed side

| Cell | dorsal (68) | ventral (32) | ventral traits 5/8 | ventral traits 6/9 | ventral traits 2/3 |
| :-- | --: | --: | --: | --: | --: |
| base_5_tta | 80.71 | 48.26 | 23.47 | 29.09 | 53.29 |
| base_lcpool_5_tta | 80.78 | 49.26 | 24.29 | 31.55 | 54.91 |
| base_lcpool_tta | 79.74 | 46.90 | 23.19 | 29.01 | 52.18 |
| base_tta | 79.98 | 45.72 | 22.94 | 28.65 | 46.55 |
| crown_all_mref_5_tta | 81.98 | 52.69 | 24.08 | 36.07 | 63.03 |
| crown_all_mref_lc_5_tta | 81.98 | 53.02 | 24.24 | 36.65 | 62.67 |
| crown_all_mref_lc_lcpool_5_tta | 82.02 | 55.31 | 25.61 | 39.79 | 69.16 |
| crown_all_mref_lc_lcpool_tta | 81.06 | 54.63 | 25.28 | 38.37 | 67.62 |
| crown_all_mref_lc_tta | 81.45 | 52.88 | 25.90 | 37.04 | 60.58 |
| crown_all_mref_lcpool_5_tta | 82.01 | 55.41 | 25.71 | 39.27 | 70.96 |
| crown_all_mref_lcpool_tta | 81.03 | 55.10 | 25.60 | 38.07 | 69.53 |
| crown_all_mref_tta | 81.42 | 53.23 | 26.01 | 36.42 | 63.19 |
| odd_cf_all_mref_5_tta | 81.99 | 50.99 | 23.66 | 36.90 | 54.81 |
| odd_cf_all_mref_lc_5_tta | 82.00 | 51.02 | 23.81 | 37.21 | 54.26 |
| odd_cf_all_mref_lc_lcpool_5_tta | 82.04 | 51.88 | 24.34 | 39.73 | 56.99 |
| odd_cf_all_mref_lc_lcpool_tta | 81.07 | 51.68 | 24.16 | 38.05 | 56.77 |
| odd_cf_all_mref_lc_tta | 81.43 | 49.57 | 22.07 | 36.58 | 50.73 |
| odd_cf_all_mref_lcpool_5_tta | 82.03 | 52.61 | 24.70 | 40.21 | 58.71 |
| odd_cf_all_mref_lcpool_tta | 81.07 | 52.58 | 25.08 | 38.73 | 58.42 |
| odd_cf_all_mref_tta | 81.40 | 50.61 | 23.70 | 37.82 | 51.59 |

## lativitta (report only: test-only ray convention, docs/s18b/LATIVITTA_DIAGNOSIS.md)

| Cell | mIoU | rays 7/8 | other traits | best-of-3 − argmax |
| :-- | --: | --: | --: | --: |
| base_5_tta | 72.66 | 43.25 | 82.46 | +5.48 |
| base_lcpool_5_tta | 72.66 | 43.25 | 82.46 | +5.48 |
| base_lcpool_tta | 72.15 | 42.80 | 81.94 | +5.47 |
| base_tta | 72.15 | 42.80 | 81.94 | +5.47 |
| crown_all_mref_5_tta | 74.86 | 48.22 | 83.75 | +5.34 |
| crown_all_mref_lc_5_tta | 74.87 | 48.22 | 83.76 | +5.34 |
| crown_all_mref_lc_lcpool_5_tta | 74.87 | 48.22 | 83.76 | +5.34 |
| crown_all_mref_lc_lcpool_tta | 74.45 | 48.04 | 83.26 | +5.32 |
| crown_all_mref_lc_tta | 74.45 | 48.04 | 83.26 | +5.32 |
| crown_all_mref_lcpool_5_tta | 74.86 | 48.22 | 83.75 | +5.34 |
| crown_all_mref_lcpool_tta | 74.46 | 48.03 | 83.28 | +5.32 |
| crown_all_mref_tta | 74.46 | 48.03 | 83.28 | +5.32 |
| odd_cf_all_mref_5_tta | 74.79 | 48.31 | 83.62 | +5.31 |
| odd_cf_all_mref_lc_5_tta | 74.77 | 48.36 | 83.57 | +5.31 |
| odd_cf_all_mref_lc_lcpool_5_tta | 74.77 | 48.36 | 83.57 | +5.31 |
| odd_cf_all_mref_lc_lcpool_tta | 74.27 | 48.04 | 83.01 | +5.29 |
| odd_cf_all_mref_lc_tta | 74.27 | 48.04 | 83.01 | +5.29 |
| odd_cf_all_mref_lcpool_5_tta | 74.78 | 48.30 | 83.60 | +5.32 |
| odd_cf_all_mref_lcpool_tta | 74.27 | 47.99 | 83.03 | +5.30 |
| odd_cf_all_mref_tta | 74.27 | 47.99 | 83.03 | +5.30 |

| Memorization probe (5-shot) | cut-out occupancy soft % | parent % | Δ vs parent, pp |
| :-- | --: | --: | :-- |
| crown_all_mref_lc | 33.26 | 33.77 | -0.51 [-0.77, -0.27] ✱ |
| odd_cf_all_mref_lc | 18.91 | 19.74 | -0.83 [-1.17, -0.51] ✱ |
