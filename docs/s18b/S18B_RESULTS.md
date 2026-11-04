# S18b evidence read

Source: results/inbox/s18b.zip. Selection is read, never changed by this report.

Methods and heuristics: [PLAN.md](PLAN.md). Screen completion is not proof of a semantic mechanism.
Reported test CIs use 10,000 paired support-seed resamples, seed 0; unadjusted, conditional on a single fit.

## butterfly_major

Screen gate: PASS — 886 fit / 221 held-out; winner=odd_cf_all; screen complete (not a test win).
Confirmation gate: PASS — 886 fit / 221 held-out; winner=odd_cf_all; test complete.

| Arm | Ordinary validation mIoU % | Cut-out soft occupancy % | Translation IoU % |
| :-- | --: | --: | --: |
| base | 79.56 | 25.71 | 90.75 |
| crown | 78.99 | 34.86 | 90.51 |
| crown_all | 81.28 | 31.47 | 92.17 |
| centre_aux | 79.44 | 17.99 | 90.61 |
| odd_cycle | 78.62 | 34.92 | 90.24 |
| odd_cf | 79.12 | 17.58 | 90.34 |
| odd_cf_all | 81.43 | 18.97 | 92.00 |
| odd_cf_judge | 79.20 | 17.61 | 90.36 |

Odd-return diagnostics (unaugmented, recoverable channels only):

| Arm | Odd closing IoU % | Continuous minus binary return, pp | Audit pairs |
| :-- | --: | --: | --: |
| base | 92.39 | -0.15 | 40 |
| crown | 93.76 | -0.08 | 40 |
| crown_all | 94.18 | -0.08 | 40 |
| centre_aux | 93.69 | -0.49 | 40 |
| odd_cycle | 93.97 | -0.04 | 40 |
| odd_cf | 93.85 | -0.22 | 40 |
| odd_cf_all | 94.21 | -0.17 | 40 |
| odd_cf_judge | 93.81 | -0.44 | 40 |

Validation-selected candidate: odd_cf_all. No automatic recipe promotion.

| Test cell | mIoU % | Delta vs matched control, pp [95% CI] |
| :-- | --: | :-- |
| base_tta | 75.18 | — |
| crown_all_tta | 77.16 | +1.98 [+1.77, +2.21] vs base_tta |
| odd_cf_all_tta | 77.02 | -0.14 [-0.22, -0.04] vs crown_all_tta |

## beetle

Screen gate: PASS — 96 fit / 24 held-out; winner=odd_cf_all; screen complete (not a test win).
Confirmation gate: PASS — 96 fit / 24 held-out; winner=odd_cf_all; test complete.

| Arm | Ordinary validation mIoU % | Cut-out soft occupancy % | Translation IoU % |
| :-- | --: | --: | --: |
| base | 50.09 | 28.16 | 88.06 |
| crown | 52.32 | 30.89 | 88.14 |
| crown_all | 52.28 | 31.00 | 88.17 |
| centre_aux | 52.07 | 30.74 | 88.13 |
| odd_cycle | 52.15 | 30.87 | 88.14 |
| odd_cf | 52.05 | 30.72 | 88.14 |
| odd_cf_all | 52.23 | 30.79 | 88.16 |
| odd_cf_judge | 52.05 | 30.71 | 88.14 |

Odd-return diagnostics (unaugmented, recoverable channels only):

| Arm | Odd closing IoU % | Continuous minus binary return, pp | Audit pairs |
| :-- | --: | --: | --: |
| base | 86.50 | -1.00 | 8 |
| crown | 87.27 | -2.08 | 8 |
| crown_all | 87.82 | -2.12 | 8 |
| centre_aux | 87.26 | -2.07 | 8 |
| odd_cycle | 87.26 | -1.11 | 8 |
| odd_cf | 87.27 | -2.07 | 8 |
| odd_cf_all | 87.84 | -2.19 | 8 |
| odd_cf_judge | 87.27 | -2.07 | 8 |

Validation-selected candidate: odd_cf_all. No automatic recipe promotion.

| Test cell | mIoU % | Delta vs matched control, pp [95% CI] |
| :-- | --: | :-- |
| base_tta | 63.86 | — |
| crown_all_tta | 64.94 | +1.08 [+0.50, +1.73] vs base_tta |
| odd_cf_all_tta | 64.86 | -0.08 [-0.21, +0.05] vs crown_all_tta |
