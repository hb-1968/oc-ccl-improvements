# S19b evidence read — MREF × CANON on the S18b wings at five-shot

Source: results/inbox/s19b.zip. Nothing is selected; this report only reads.

Methods and heuristics: [S19B_PLAN.md](S19B_PLAN.md). Locked cells are the parents' own (S18b 1-shot, S19 5-shot), unchanged. Primary metric: argmax mIoU (every recorded number); best-of-3 boundary (`argmax_best3`) is the secondary column.
CIs: 10,000 paired support-seed resamples, seed 0, unadjusted; ✱ = excludes zero. CANON pairs a canonical cell with its as-drawn cell on identical draws (`ignore_fields=model.support_orientation`); shot lifts use the prefix-nested pairing; interactions and additivity are per-seed differences of differences. Probe deltas pair per episode.

## butterfly_major

Screen gate: PASS — parents e0bf48fa94b3/eebd9a262c46; eligible=['crown_all_mref', 'odd_cf_all_mref']; fits screened; test cells not yet demanded.
Confirmation gate: PASS — parents e0bf48fa94b3/eebd9a262c46; eligible=['crown_all_mref', 'odd_cf_all_mref']; 5-shot cells, 1-shot MREF cells and probes complete.

| MREF fit | ‖ΔW‖ | Val mIoU % (parent) | Cut-out occupancy % (parent) | Translation % (parent) | Incomplete extra refs | Eligible |
| :-- | --: | --: | --: | --: | --: | :-- |
| crown_all_mref | 6.569 | 81.50 (81.28) | 31.22 (31.47) | 91.99 (92.17) | 0 / 3544 | yes |
| odd_cf_all_mref | 6.840 | 81.26 (81.43) | 16.37 (18.97) | 92.10 (92.00) | 0 / 3544 | yes |

| Adapter | 1-shot TTA | 5-shot TTA (as drawn) | 5-shot best-of-3 | 5-shot CANON | CANON − as drawn | Δ vs base (5-shot, as drawn) |
| :-- | --: | --: | --: | --: | :-- | :-- |
| base | 75.18 | 77.23 | 78.71 | 77.22 | -0.01 [-0.01, -0.00] ✱ | — |
| crown_all | 77.16 | 78.53 | 79.79 | 78.53 | +0.00 [+0.00, +0.00] | +1.30 [+1.21, +1.39] ✱ |
| odd_cf_all | 77.02 | 78.46 | 79.72 | 78.46 | +0.00 [+0.00, +0.00] | +1.24 [+1.15, +1.33] ✱ |
| crown_all_mref | 77.20 | 78.56 | 79.83 | 78.56 | +0.00 [+0.00, +0.00] | +1.34 [+1.25, +1.42] ✱ |
| odd_cf_all_mref | 76.89 | 78.43 | 79.69 | 78.43 | +0.00 [+0.00, +0.00] | +1.20 [+1.13, +1.28] ✱ |

| Lever, per wing | crown_all | odd_cf_all |
| :-- | :-- | :-- |
| MREF at 5-shot (mref − parent) | +0.04 [+0.01, +0.07] ✱ | -0.04 [-0.11, +0.03] |
| MREF at 1-shot (mref − parent) | +0.04 [-0.06, +0.15] | -0.13 [-0.30, +0.02] |
| MREF shot interaction ((mref−base)₅ − (mref−base)₁) | -0.68 [-0.92, -0.47] ✱ | -0.50 [-0.75, -0.31] ✱ |
| CANON at 5-shot (canon − as drawn) | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| CANON on the MREF arm | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| STACK (mref_canon − parent) | +0.04 [+0.01, +0.07] ✱ | -0.04 [-0.11, +0.03] |
| STACK vs base | +1.34 [+1.25, +1.42] ✱ | +1.20 [+1.13, +1.28] ✱ |
| Additivity ((stack−p) − [(canon−p)+(mref−p)]) | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| MREF, best-of-3 reading | +0.04 [+0.01, +0.07] ✱ | -0.04 [-0.11, +0.03] |
| CANON, best-of-3 reading | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| STACK, best-of-3 reading | +0.04 [+0.01, +0.07] ✱ | -0.04 [-0.11, +0.03] |

| odd_cf_all − crown_all | Δ, pp [95% CI] | Lever-by-wing interaction |
| :-- | :-- | :-- |
| as drawn, 5-shot (locked) | -0.06 [-0.11, -0.02] ✱ | — |
| under MREF | -0.14 [-0.19, -0.08] ✱ | -0.07 [-0.14, -0.01] ✱ |
| under CANON | -0.06 [-0.11, -0.02] ✱ | +0.00 [+0.00, +0.00] |
| under both (stack) | -0.14 [-0.19, -0.08] ✱ | -0.07 [-0.14, -0.01] ✱ |

| Memorization probe (5-shot, as drawn) | Cut-out occupancy soft % | hard % | Others mIoU % | Copy excess | Δ occupancy vs base, pp | Δ vs crown_all, pp | Episodes |
| :-- | --: | --: | --: | --: | :-- | :-- | --: |
| base | 24.85 | 24.64 | 76.87 | +0.10 | — | -9.94 [-10.84, -9.04] ✱ | 2495 |
| crown_all | 34.80 | 33.92 | 78.16 | +0.47 | +9.94 [+9.04, +10.84] ✱ | — | 2495 |
| odd_cf_all | 25.33 | 25.12 | 78.21 | +0.45 | +0.48 [-0.20, +1.16] | -9.46 [-10.25, -8.72] ✱ | 2495 |
| crown_all_mref | 33.77 | 32.97 | 78.29 | +0.40 | +8.92 [+8.09, +9.76] ✱ | -1.02 [-1.44, -0.63] ✱ | 2495 |
| odd_cf_all_mref | 19.74 | 19.92 | 78.25 | +0.38 | -5.11 [-5.79, -4.45] ✱ | -15.06 [-15.99, -14.16] ✱ | 2495 |

## beetle

Screen gate: PASS — parents 714d7d17ac14/f2364d7c54f0; eligible=['crown_all_mref', 'odd_cf_all_mref']; fits screened; test cells not yet demanded.
Confirmation gate: PASS — parents 714d7d17ac14/f2364d7c54f0; eligible=['crown_all_mref', 'odd_cf_all_mref']; 5-shot cells, 1-shot MREF cells and probes complete.

| MREF fit | ‖ΔW‖ | Val mIoU % (parent) | Cut-out occupancy % (parent) | Translation % (parent) | Incomplete extra refs | Eligible |
| :-- | --: | --: | --: | --: | --: | :-- |
| crown_all_mref | 1.506 | 52.37 (52.28) | 30.83 (31.00) | 88.18 (88.17) | 0 / 384 | yes |
| odd_cf_all_mref | 1.388 | 52.29 (52.23) | 28.09 (30.79) | 88.15 (88.16) | 0 / 384 | yes |

| Adapter | 1-shot TTA | 5-shot TTA (as drawn) | 5-shot best-of-3 | 5-shot CANON | CANON − as drawn | Δ vs base (5-shot, as drawn) |
| :-- | --: | --: | --: | --: | :-- | :-- |
| base | 63.86 | 75.15 | 75.80 | 74.98 | -0.17 [-0.56, +0.06] | — |
| crown_all | 64.94 | 75.50 | 76.10 | 75.42 | -0.08 [-0.22, +0.04] | +0.35 [-0.05, +0.74] |
| odd_cf_all | 64.86 | 75.47 | 76.07 | 75.35 | -0.12 [-0.35, +0.02] | +0.32 [-0.04, +0.67] |
| crown_all_mref | 65.31 | 75.81 | 76.41 | 75.75 | -0.06 [-0.18, +0.00] | +0.66 [+0.26, +1.06] ✱ |
| odd_cf_all_mref | 65.24 | 75.86 | 76.46 | 75.79 | -0.07 [-0.21, +0.00] | +0.71 [+0.24, +1.21] ✱ |

| Lever, per wing | crown_all | odd_cf_all |
| :-- | :-- | :-- |
| MREF at 5-shot (mref − parent) | +0.31 [+0.13, +0.50] ✱ | +0.39 [+0.11, +0.67] ✱ |
| MREF at 1-shot (mref − parent) | +0.37 [-0.10, +0.82] | +0.37 [-0.17, +0.85] |
| MREF shot interaction ((mref−base)₅ − (mref−base)₁) | -0.79 [-1.28, -0.29] ✱ | -0.66 [-1.26, -0.09] ✱ |
| CANON at 5-shot (canon − as drawn) | -0.08 [-0.22, +0.04] | -0.12 [-0.35, +0.02] |
| CANON on the MREF arm | -0.06 [-0.18, +0.00] | -0.07 [-0.21, +0.00] |
| STACK (mref_canon − parent) | +0.25 [+0.05, +0.45] ✱ | +0.32 [+0.01, +0.63] ✱ |
| STACK vs base | +0.60 [+0.12, +1.04] ✱ | +0.64 [+0.08, +1.19] ✱ |
| Additivity ((stack−p) − [(canon−p)+(mref−p)]) | +0.02 [-0.11, +0.16] | +0.05 [-0.03, +0.14] |
| MREF, best-of-3 reading | +0.31 [+0.14, +0.49] ✱ | +0.39 [+0.12, +0.67] ✱ |
| CANON, best-of-3 reading | -0.07 [-0.20, +0.05] | -0.12 [-0.34, +0.02] |
| STACK, best-of-3 reading | +0.25 [+0.06, +0.45] ✱ | +0.32 [+0.01, +0.62] ✱ |

| odd_cf_all − crown_all | Δ, pp [95% CI] | Lever-by-wing interaction |
| :-- | :-- | :-- |
| as drawn, 5-shot (locked) | -0.03 [-0.16, +0.11] | — |
| under MREF | +0.06 [-0.16, +0.27] | +0.08 [-0.13, +0.31] |
| under CANON | -0.07 [-0.24, +0.08] | -0.04 [-0.23, +0.11] |
| under both (stack) | +0.05 [-0.17, +0.26] | +0.07 [-0.14, +0.31] |

| Memorization probe (5-shot, as drawn) | Cut-out occupancy soft % | hard % | Others mIoU % | Copy excess | Δ occupancy vs base, pp | Δ vs crown_all, pp | Episodes |
| :-- | --: | --: | --: | --: | :-- | :-- | --: |
| base | 17.99 | 11.54 | 67.40 | -0.54 | — | -0.83 [-1.25, -0.42] ✱ | 1200 |
| crown_all | 18.82 | 12.09 | 67.19 | -0.56 | +0.83 [+0.42, +1.25] ✱ | — | 1200 |
| odd_cf_all | 18.63 | 11.86 | 67.52 | -0.54 | +0.64 [+0.25, +1.03] ✱ | -0.19 [-0.40, -0.01] ✱ | 1200 |
| crown_all_mref | 18.86 | 12.00 | 67.93 | -0.48 | +0.87 [+0.50, +1.26] ✱ | +0.04 [-0.21, +0.30] | 1200 |
| odd_cf_all_mref | 18.50 | 11.82 | 67.82 | -0.48 | +0.51 [+0.15, +0.88] ✱ | -0.32 [-0.61, -0.04] ✱ | 1200 |
