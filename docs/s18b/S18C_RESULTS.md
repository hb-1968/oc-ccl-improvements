# S18c evidence read

Source: results/inbox/s18c.zip. Selection is read, never changed by this report.

Methods and heuristics: [S18C_PLAN.md](S18C_PLAN.md). Parent: the landed S18b root ([PLAN.md](PLAN.md) "Landed result").
Test CIs: 10,000 paired support-seed resamples, seed 0; unadjusted across arms and datasets; one fit per arm.

## butterfly_major

Screen gate: PASS — 886 fit / 221 held-out; eligible=['centre_all_w25', 'centre_all_w10', 'soup_25']; screen complete (not a test win).
Confirmation gate: PASS — 886 fit / 221 held-out; eligible=['centre_all_w25', 'centre_all_w10', 'soup_25']; test complete.

| Arm | Ordinary validation mIoU % | Cut-out soft occupancy % | Translation IoU % | Eligible |
| :-- | --: | --: | --: | :-- |
| base (S18b) | 79.56 | 25.71 | 90.75 | parent |
| crown_all (S18b) | 81.28 | 31.47 | 92.17 | parent |
| odd_cf_all (S18b) | 81.43 | 18.97 | 92.00 | parent |
| soup_25 | 81.41 | 24.07 | 92.08 | yes |
| soup_50 | 81.40 | 27.77 | 92.11 | no: cutout_margin |
| soup_75 | 81.28 | 29.70 | 92.15 | no: cutout_margin |
| centre_all_w05 | 81.47 | 28.26 | 92.10 | no: cutout_margin |
| centre_all_w10 | 81.43 | 25.38 | 92.08 | yes |
| centre_all_w25 | 81.45 | 18.44 | 91.88 | yes |

| Arm | ‖ΔW‖ | Construction |
| :-- | --: | :-- |
| soup_25 | 7.165 | soup: 0.25 crown_all + 0.75 odd_cf_all (rank 64) |
| soup_50 | 7.036 | soup: 0.50 crown_all + 0.50 odd_cf_all (rank 64) |
| soup_75 | 7.108 | soup: 0.75 crown_all + 0.25 odd_cf_all (rank 64) |
| centre_all_w05 | 7.014 | centre_aux, all legs, centre weight 0.05 |
| centre_all_w10 | 7.117 | centre_aux, all legs, centre weight 0.1 |
| centre_all_w25 | 7.434 | centre_aux, all legs, centre weight 0.25 |

Eligible by the pre-registered floors: ['centre_all_w25', 'centre_all_w10', 'soup_25']; rule winner: centre_all_w25. No automatic recipe promotion.

| Test cell | mIoU % | Δ vs crown_all_tta, pp [95% CI] | Δ vs base_tta, pp [95% CI] |
| :-- | --: | :-- | :-- |
| base_tta | 75.18 | -1.98 [-2.21, -1.77] | — |
| centre_all_w05_tta | 77.07 | -0.09 [-0.17, -0.00] | +1.89 [+1.66, +2.14] |
| centre_all_w10_tta | 77.04 | -0.12 [-0.20, -0.03] | +1.86 [+1.64, +2.11] |
| centre_all_w25_tta | 77.00 | -0.15 [-0.22, -0.08] | +1.82 [+1.60, +2.08] |
| crown_all_tta | 77.16 | — | +1.98 [+1.77, +2.21] |
| soup_25_tta | 77.03 | -0.13 [-0.23, -0.02] | +1.85 [+1.61, +2.12] |
| soup_50_tta | 77.11 | -0.05 [-0.12, +0.04] | +1.93 [+1.70, +2.19] |
| soup_75_tta | 77.13 | -0.03 [-0.10, +0.05] | +1.95 [+1.71, +2.22] |

## beetle

Screen gate: PASS — 96 fit / 24 held-out; eligible=[]; screen complete (not a test win).
Confirmation gate: PASS — 96 fit / 24 held-out; eligible=[]; test complete.

| Arm | Ordinary validation mIoU % | Cut-out soft occupancy % | Translation IoU % | Eligible |
| :-- | --: | --: | --: | :-- |
| base (S18b) | 50.09 | 28.16 | 88.06 | parent |
| crown_all (S18b) | 52.28 | 31.00 | 88.17 | parent |
| odd_cf_all (S18b) | 52.23 | 30.79 | 88.16 | parent |
| soup_25 | 52.24 | 30.82 | 88.16 | no: cutout_margin |
| soup_50 | 52.34 | 30.89 | 88.16 | no: cutout_margin |
| soup_75 | 52.17 | 30.93 | 88.17 | no: cutout_margin |
| centre_all_w05 | 52.28 | 30.92 | 88.16 | no: cutout_margin |
| centre_all_w10 | 52.28 | 30.90 | 88.16 | no: cutout_margin |
| centre_all_w25 | 52.10 | 30.84 | 88.15 | no: cutout_margin |

| Arm | ‖ΔW‖ | Construction |
| :-- | --: | :-- |
| soup_25 | 1.641 | soup: 0.25 crown_all + 0.75 odd_cf_all (rank 64) |
| soup_50 | 1.668 | soup: 0.50 crown_all + 0.50 odd_cf_all (rank 64) |
| soup_75 | 1.699 | soup: 0.75 crown_all + 0.25 odd_cf_all (rank 64) |
| centre_all_w05 | 1.696 | centre_aux, all legs, centre weight 0.05 |
| centre_all_w10 | 1.697 | centre_aux, all legs, centre weight 0.1 |
| centre_all_w25 | 1.630 | centre_aux, all legs, centre weight 0.25 |

Eligible by the pre-registered floors: []; rule winner: None. No automatic recipe promotion.

| Test cell | mIoU % | Δ vs crown_all_tta, pp [95% CI] | Δ vs base_tta, pp [95% CI] |
| :-- | --: | :-- | :-- |
| base_tta | 63.86 | -1.08 [-1.73, -0.50] | — |
| centre_all_w05_tta | 65.09 | +0.15 [-0.01, +0.34] | +1.23 [+0.64, +1.87] |
| centre_all_w10_tta | 64.83 | -0.11 [-0.21, -0.02] | +0.97 [+0.41, +1.61] |
| centre_all_w25_tta | 64.88 | -0.06 [-0.29, +0.18] | +1.02 [+0.42, +1.69] |
| crown_all_tta | 64.94 | — | +1.08 [+0.50, +1.73] |
| soup_25_tta | 64.82 | -0.12 [-0.30, +0.02] | +0.95 [+0.45, +1.50] |
| soup_50_tta | 64.89 | -0.05 [-0.20, +0.06] | +1.03 [+0.51, +1.59] |
| soup_75_tta | 64.87 | -0.07 [-0.21, +0.04] | +1.01 [+0.47, +1.59] |
