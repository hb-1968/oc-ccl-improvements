# S19 evidence read — the S18b wings at five-shot inference

Source: results/inbox/s19.zip. Nothing is selected; this report only reads.

Methods and heuristics: [S19_PLAN.md](S19_PLAN.md). Parent: the landed S18b root ([PLAN.md](PLAN.md) "Landed result"); the 1-shot cells below are the parent's own test cells, unchanged.
CIs: 10,000 paired support-seed resamples, seed 0, unadjusted across reads and datasets; ✱ = the interval excludes zero. Shot lifts pair run i's 1-shot support with the 5-shot draw it is the first element of (`compare(nested_shots=True)`); interactions are per-seed differences of differences.

## butterfly_major

Parent binding: PASS — parent e0bf48fa94b3; wings ['crown_all', 'odd_cf_all']; parent bound; 5-shot cells not yet demanded.
Confirmation gate: PASS — parent e0bf48fa94b3; wings ['crown_all', 'odd_cf_all']; 5-shot test complete (paired, prefix-nested on the 1-shot cells).

| Arm | 1-shot TTA mIoU % | 5-shot TTA mIoU % | Shot lift, pp [95% CI] |
| :-- | --: | --: | :-- |
| base | 75.18 | 77.23 | +2.04 [+1.25, +3.09] ✱ |
| crown_all | 77.16 | 78.53 | +1.37 [+0.63, +2.43] ✱ |
| odd_cf_all | 77.02 | 78.46 | +1.45 [+0.75, +2.44] ✱ |

| Pairing | at 1-shot, pp [95% CI] | at 5-shot, pp [95% CI] | Interaction (5-shot gain − 1-shot gain), pp [95% CI] |
| :-- | :-- | :-- | :-- |
| crown_all − base | +1.98 [+1.77, +2.21] ✱ | +1.30 [+1.21, +1.39] ✱ | -0.68 [-0.92, -0.45] ✱ |
| odd_cf_all − base | +1.84 [+1.61, +2.10] ✱ | +1.24 [+1.15, +1.33] ✱ | -0.60 [-0.86, -0.37] ✱ |
| odd_cf_all − crown_all | -0.14 [-0.22, -0.04] ✱ | -0.06 [-0.11, -0.02] ✱ | +0.08 [-0.02, +0.17] |

Prefix nesting verified on 100 (group, run) pairs per arm; the same adapter bytes on both shot counts (delta-norm pin, rel 1e-4).

## beetle

Parent binding: PASS — parent 714d7d17ac14; wings ['crown_all', 'odd_cf_all']; parent bound; 5-shot cells not yet demanded.
Confirmation gate: PASS — parent 714d7d17ac14; wings ['crown_all', 'odd_cf_all']; 5-shot test complete (paired, prefix-nested on the 1-shot cells).

| Arm | 1-shot TTA mIoU % | 5-shot TTA mIoU % | Shot lift, pp [95% CI] |
| :-- | --: | --: | :-- |
| base | 63.86 | 75.15 | +11.29 [+9.02, +13.68] ✱ |
| crown_all | 64.94 | 75.50 | +10.56 [+8.68, +12.53] ✱ |
| odd_cf_all | 64.86 | 75.47 | +10.61 [+8.73, +12.57] ✱ |

| Pairing | at 1-shot, pp [95% CI] | at 5-shot, pp [95% CI] | Interaction (5-shot gain − 1-shot gain), pp [95% CI] |
| :-- | :-- | :-- | :-- |
| crown_all − base | +1.08 [+0.50, +1.73] ✱ | +0.35 [-0.05, +0.74] | -0.73 [-1.47, +0.03] |
| odd_cf_all − base | +1.00 [+0.44, +1.65] ✱ | +0.32 [-0.04, +0.67] | -0.68 [-1.38, +0.00] |
| odd_cf_all − crown_all | -0.08 [-0.21, +0.05] | -0.03 [-0.16, +0.11] | +0.05 [-0.11, +0.19] |

Prefix nesting verified on 20 (group, run) pairs per arm; the same adapter bytes on both shot counts (delta-norm pin, rel 1e-4).
