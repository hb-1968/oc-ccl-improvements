# S18a landed results

Recomputed from the complete local pull; documentation reviewed 2026-09-22.

**Gate: FAIL.** [FAIL]  [butterfly_major] DISQUALIFIER: every adapter lowers its OWN objective on its own pairs under its own prompt value — the bounded (Dice) reading and the epoch total; a plain-STE arm (pointer route only) is recorded, not gated.

The failed merged bundle remains in `results/inbox/`. No gate was relaxed or failed arm removed.

Current S18b implementation: [PLAN.md](PLAN.md). Methods, constants, denominators and limitations: [METHODS.md](METHODS.md). Interpretation: [CALIBRATION.md](CALIBRATION.md). Machine-readable numbers and full source hashes: [s18a_evidence.json](s18a_evidence.json). Gate output: [s18a_gate.txt](s18a_gate.txt).

Reproduce: `.venv/bin/python scripts/report_s18a.py`. This reads input bundles and writes only these documentation artifacts.

Test mIoU is in percent; deltas and CIs are percentage points. CIs are paired percentile bootstrap (10,000 resamples, seed 0), unadjusted for multiplicity. Test pairing unit: 20 support-run indices; centre pairing unit: selected training pair. Neither is a set of independently retrained adapters.

## beetle: test evaluation

| Arm | Plain | TTA |
| :-- | --: | --: |
| `base` | 63.17 | 63.86 |
| `crown` | 63.65 | 64.96 |
| `crown_ste` | 63.43 | 64.69 |
| `crown_stemem` | 63.73 | 65.06 |
| `crown_perturb` | 63.59 | 64.77 |
| `crown_all` | 63.87 | 65.32 |
| `crown_all_perturb` | 63.78 | 65.18 |
| `crown_hot` | 60.92 | 63.91 |
| `crown_stemem_hot` | 64.23 | 65.99 |

| Treatment minus control | Plain delta [95% CI] | TTA delta [95% CI] |
| :-- | :-- | :-- |
| `crown` − `base` | +0.47 [+0.19, +0.77] | +1.10 [+0.55, +1.70] |
| `crown_ste` − `base` | +0.26 [-0.08, +0.59] | +0.83 [+0.24, +1.45] |
| `crown_stemem` − `base` | +0.56 [+0.30, +0.84] | +1.20 [+0.69, +1.75] |
| `crown_perturb` − `base` | +0.41 [+0.15, +0.70] | +0.90 [+0.31, +1.54] |
| `crown_all` − `base` | +0.69 [+0.37, +1.02] | +1.46 [+0.96, +2.04] |
| `crown_all_perturb` − `base` | +0.61 [+0.31, +0.92] | +1.31 [+0.83, +1.89] |
| `crown_hot` − `base` | -2.25 [-3.14, -1.40] | +0.04 [-0.96, +1.02] |
| `crown_stemem_hot` − `base` | +1.06 [+0.24, +1.85] | +2.13 [+1.08, +3.27] |
| `crown_stemem` − `crown` | +0.09 [-0.09, +0.27] | +0.10 [-0.33, +0.57] |
| `crown_all` − `crown` | +0.22 [+0.03, +0.41] | +0.36 [+0.05, +0.68] |
| `crown_perturb` − `crown` | -0.06 [-0.15, +0.03] | -0.19 [-0.48, +0.06] |
| `crown_all_perturb` − `crown_all` | -0.08 [-0.17, -0.01] | -0.14 [-0.31, +0.03] |
| `crown_stemem_hot` − `crown` | +0.59 [-0.07, +1.25] | +1.03 [+0.13, +1.99] |

## beetle: centre calibration

120 distinct references / selected pairs; 11880 rows. Group counts: `{'all': 120}`.

Cut trait counts: `{1: 23, 2: 24, 3: 24, 4: 24, 5: 25}`. Cut-out recall is the removed-channel regional occupancy. Track is undefined for cut-out. Judge values average all channels before counting positive rows; they are not calibrated probabilities.

| Transform | Memory | Arm | Recall % | Track % | IoU(Ty0) % | Recall delta vs base [95% CI], pp | Label LLR/cell | Positive rows % |
| :-- | :-- | :-- | --: | --: | --: | :-- | --: | --: |
| flipH | ref | `base` | 3.44 | 76.46 | 72.55 | — | -1.427 | 1.67 |
| flipH | ref | `crown` | 3.34 | 76.86 | 72.75 | -0.10 [-0.44, +0.17] | -1.454 | 1.67 |
| flipH | ref | `crown_ste` | 3.25 | 76.94 | 72.57 | -0.19 [-0.75, +0.26] | -1.455 | 0.83 |
| flipH | ref | `crown_stemem` | 3.38 | 76.64 | 72.56 | -0.06 [-0.28, +0.08] | -1.442 | 1.67 |
| flipH | ref | `crown_perturb` | 3.35 | 76.60 | 72.45 | -0.08 [-0.43, +0.19] | -1.447 | 1.67 |
| flipH | ref | `crown_all` | 3.16 | 77.19 | 73.10 | -0.27 [-0.81, +0.10] | -1.464 | 0.83 |
| flipH | ref | `crown_all_perturb` | 3.31 | 77.02 | 72.93 | -0.12 [-0.48, +0.14] | -1.460 | 1.67 |
| flipH | ref | `crown_hot` | 3.50 | 77.74 | 73.24 | +0.06 [-0.54, +0.59] | -1.506 | 1.67 |
| flipH | ref | `crown_stemem_hot` | 3.26 | 78.81 | 73.67 | -0.18 [-0.77, +0.29] | -1.498 | 0.83 |
| flipH | chain | `base` | 1.98 | 82.71 | 78.40 | — | -1.593 | 0.00 |
| flipH | chain | `crown` | 2.12 | 82.68 | 78.23 | +0.13 [+0.02, +0.27] | -1.597 | 0.00 |
| flipH | chain | `crown_ste` | 2.06 | 82.68 | 78.02 | +0.08 [-0.01, +0.19] | -1.591 | 0.00 |
| flipH | chain | `crown_stemem` | 2.01 | 82.48 | 77.95 | +0.02 [-0.02, +0.08] | -1.588 | 0.00 |
| flipH | chain | `crown_perturb` | 2.10 | 82.53 | 78.14 | +0.12 [+0.01, +0.25] | -1.594 | 0.00 |
| flipH | chain | `crown_all` | 2.10 | 82.64 | 78.31 | +0.12 [+0.00, +0.26] | -1.593 | 0.00 |
| flipH | chain | `crown_all_perturb` | 2.12 | 82.33 | 78.04 | +0.14 [+0.02, +0.29] | -1.587 | 0.00 |
| flipH | chain | `crown_hot` | 2.21 | 81.87 | 77.36 | +0.23 [-0.02, +0.52] | -1.612 | 0.00 |
| flipH | chain | `crown_stemem_hot` | 2.29 | 83.26 | 78.10 | +0.31 [+0.08, +0.58] | -1.614 | 0.00 |
| rot90 | ref | `base` | 2.27 | 68.50 | 62.40 | — | -1.462 | 0.00 |
| rot90 | ref | `crown` | 2.13 | 68.39 | 62.00 | -0.14 [-0.34, +0.02] | -1.462 | 0.00 |
| rot90 | ref | `crown_ste` | 2.35 | 67.92 | 61.33 | +0.08 [-0.11, +0.38] | -1.457 | 0.00 |
| rot90 | ref | `crown_stemem` | 2.14 | 68.79 | 62.51 | -0.14 [-0.31, +0.00] | -1.474 | 0.00 |
| rot90 | ref | `crown_perturb` | 2.14 | 68.26 | 61.90 | -0.13 [-0.32, +0.03] | -1.460 | 0.00 |
| rot90 | ref | `crown_all` | 2.25 | 68.61 | 62.24 | -0.03 [-0.10, +0.04] | -1.472 | 0.00 |
| rot90 | ref | `crown_all_perturb` | 2.25 | 68.31 | 61.89 | -0.02 [-0.10, +0.05] | -1.463 | 0.00 |
| rot90 | ref | `crown_hot` | 2.65 | 65.70 | 58.36 | +0.38 [-0.13, +0.93] | -1.439 | 0.00 |
| rot90 | ref | `crown_stemem_hot` | 2.05 | 69.94 | 62.24 | -0.22 [-0.77, +0.30] | -1.500 | 0.00 |
| rot90 | chain | `base` | 2.05 | 67.15 | 60.71 | — | -1.436 | 0.00 |
| rot90 | chain | `crown` | 2.03 | 68.47 | 61.57 | -0.02 [-0.14, +0.10] | -1.471 | 0.00 |
| rot90 | chain | `crown_ste` | 2.19 | 68.02 | 60.91 | +0.14 [-0.10, +0.43] | -1.463 | 0.00 |
| rot90 | chain | `crown_stemem` | 2.06 | 68.35 | 61.63 | +0.01 [-0.11, +0.13] | -1.469 | 0.00 |
| rot90 | chain | `crown_perturb` | 2.03 | 68.29 | 61.49 | -0.02 [-0.14, +0.10] | -1.469 | 0.00 |
| rot90 | chain | `crown_all` | 2.00 | 68.45 | 61.84 | -0.05 [-0.18, +0.08] | -1.475 | 0.00 |
| rot90 | chain | `crown_all_perturb` | 2.00 | 68.45 | 61.78 | -0.05 [-0.18, +0.07] | -1.475 | 0.00 |
| rot90 | chain | `crown_hot` | 2.26 | 66.61 | 59.52 | +0.21 [-0.13, +0.61] | -1.447 | 0.00 |
| rot90 | chain | `crown_stemem_hot` | 2.07 | 70.59 | 62.82 | +0.02 [-0.50, +0.40] | -1.516 | 0.00 |
| warp | ref | `base` | 6.43 | 91.27 | 90.02 | — | -1.175 | 0.00 |
| warp | ref | `crown` | 6.54 | 91.51 | 90.08 | +0.11 [+0.06, +0.16] | -1.178 | 0.00 |
| warp | ref | `crown_ste` | 6.62 | 91.60 | 90.06 | +0.19 [+0.14, +0.24] | -1.177 | 0.00 |
| warp | ref | `crown_stemem` | 6.53 | 91.44 | 90.06 | +0.10 [+0.07, +0.14] | -1.176 | 0.00 |
| warp | ref | `crown_perturb` | 6.51 | 91.48 | 90.07 | +0.08 [+0.03, +0.13] | -1.179 | 0.00 |
| warp | ref | `crown_all` | 6.50 | 91.50 | 90.09 | +0.07 [+0.02, +0.13] | -1.179 | 0.00 |
| warp | ref | `crown_all_perturb` | 6.51 | 91.51 | 90.08 | +0.08 [+0.03, +0.14] | -1.180 | 0.00 |
| warp | ref | `crown_hot` | 6.94 | 92.49 | 90.12 | +0.51 [+0.34, +0.71] | -1.199 | 0.00 |
| warp | ref | `crown_stemem_hot` | 7.60 | 92.87 | 89.85 | +1.17 [+0.96, +1.40] | -1.173 | 0.00 |
| warp | chain | `base` | 6.30 | 90.73 | 89.56 | — | -1.169 | 0.00 |
| warp | chain | `crown` | 6.30 | 90.87 | 89.62 | -0.01 [-0.06, +0.05] | -1.175 | 0.00 |
| warp | chain | `crown_ste` | 6.46 | 90.89 | 89.55 | +0.16 [+0.04, +0.33] | -1.171 | 0.00 |
| warp | chain | `crown_stemem` | 6.33 | 90.84 | 89.59 | +0.03 [-0.01, +0.07] | -1.173 | 0.00 |
| warp | chain | `crown_perturb` | 6.28 | 90.83 | 89.61 | -0.02 [-0.08, +0.04] | -1.175 | 0.00 |
| warp | chain | `crown_all` | 6.29 | 90.86 | 89.62 | -0.01 [-0.07, +0.04] | -1.176 | 0.00 |
| warp | chain | `crown_all_perturb` | 6.26 | 90.91 | 89.64 | -0.04 [-0.15, +0.04] | -1.176 | 0.00 |
| warp | chain | `crown_hot` | 6.51 | 91.75 | 89.85 | +0.20 [+0.02, +0.44] | -1.198 | 0.00 |
| warp | chain | `crown_stemem_hot` | 7.30 | 92.11 | 89.43 | +0.99 [+0.75, +1.28] | -1.165 | 0.00 |
| cutout | ref | `base` | 26.80 | undefined | 71.06 | — | -0.143 | 16.67 |
| cutout | ref | `crown` | 27.82 | undefined | 70.06 | +1.01 [+0.07, +2.00] | -0.133 | 15.83 |
| cutout | ref | `crown_ste` | 28.29 | undefined | 70.05 | +1.48 [+0.47, +2.54] | -0.124 | 18.33 |
| cutout | ref | `crown_stemem` | 27.66 | undefined | 70.49 | +0.86 [+0.20, +1.64] | -0.134 | 16.67 |
| cutout | ref | `crown_perturb` | 27.90 | undefined | 70.07 | +1.10 [+0.16, +2.07] | -0.133 | 15.83 |
| cutout | ref | `crown_all` | 27.78 | undefined | 70.24 | +0.97 [+0.02, +1.98] | -0.133 | 15.00 |
| cutout | ref | `crown_all_perturb` | 28.03 | undefined | 70.23 | +1.22 [+0.40, +2.15] | -0.131 | 15.83 |
| cutout | ref | `crown_hot` | 32.66 | undefined | 70.63 | +5.85 [+4.23, +7.54] | -0.094 | 25.00 |
| cutout | ref | `crown_stemem_hot` | 31.08 | undefined | 69.75 | +4.27 [+2.66, +5.98] | -0.097 | 20.83 |
| cutout | chain | `base` | 25.33 | undefined | 70.67 | — | -0.150 | 14.17 |
| cutout | chain | `crown` | 26.00 | undefined | 70.37 | +0.67 [-0.21, +1.66] | -0.147 | 15.00 |
| cutout | chain | `crown_ste` | 26.61 | undefined | 70.19 | +1.28 [+0.14, +2.66] | -0.145 | 15.83 |
| cutout | chain | `crown_stemem` | 25.96 | undefined | 70.46 | +0.63 [-0.02, +1.47] | -0.148 | 15.83 |
| cutout | chain | `crown_perturb` | 25.86 | undefined | 70.36 | +0.53 [-0.37, +1.52] | -0.148 | 14.17 |
| cutout | chain | `crown_all` | 26.08 | undefined | 70.37 | +0.75 [-0.11, +1.76] | -0.149 | 15.00 |
| cutout | chain | `crown_all_perturb` | 25.95 | undefined | 70.45 | +0.62 [-0.33, +1.65] | -0.148 | 15.00 |
| cutout | chain | `crown_hot` | 28.88 | undefined | 70.31 | +3.55 [+1.73, +5.48] | -0.132 | 16.67 |
| cutout | chain | `crown_stemem_hot` | 30.29 | undefined | 69.69 | +4.96 [+2.98, +7.05] | -0.110 | 24.17 |
| translate | ref | `base` | 0.37 | 94.68 | 89.74 | — | -1.974 | 0.00 |
| translate | ref | `crown` | 0.37 | 94.88 | 89.80 | +0.00 [+0.00, +0.01] | -1.979 | 0.00 |
| translate | ref | `crown_ste` | 0.38 | 94.89 | 89.79 | +0.01 [+0.00, +0.02] | -1.978 | 0.00 |
| translate | ref | `crown_stemem` | 0.37 | 94.82 | 89.80 | +0.01 [+0.00, +0.01] | -1.976 | 0.00 |
| translate | ref | `crown_perturb` | 0.37 | 94.85 | 89.80 | +0.00 [-0.01, +0.01] | -1.979 | 0.00 |
| translate | ref | `crown_all` | 0.36 | 94.89 | 89.84 | -0.00 [-0.01, +0.00] | -1.980 | 0.00 |
| translate | ref | `crown_all_perturb` | 0.36 | 94.89 | 89.83 | -0.00 [-0.01, +0.01] | -1.980 | 0.00 |
| translate | ref | `crown_hot` | 0.37 | 95.63 | 90.03 | +0.00 [-0.01, +0.01] | -1.999 | 0.00 |
| translate | ref | `crown_stemem_hot` | 0.41 | 95.63 | 89.87 | +0.05 [+0.04, +0.05] | -1.984 | 0.00 |
| translate | chain | `base` | 0.38 | 94.03 | 89.30 | — | -1.948 | 0.00 |
| translate | chain | `crown` | 0.37 | 94.14 | 89.34 | -0.00 [-0.01, +0.00] | -1.956 | 0.00 |
| translate | chain | `crown_ste` | 0.38 | 94.17 | 89.36 | +0.00 [-0.00, +0.01] | -1.958 | 0.00 |
| translate | chain | `crown_stemem` | 0.38 | 94.09 | 89.31 | +0.01 [-0.00, +0.02] | -1.950 | 0.00 |
| translate | chain | `crown_perturb` | 0.37 | 94.12 | 89.33 | -0.00 [-0.01, +0.00] | -1.958 | 0.00 |
| translate | chain | `crown_all` | 0.37 | 94.15 | 89.34 | -0.00 [-0.01, +0.01] | -1.960 | 0.00 |
| translate | chain | `crown_all_perturb` | 0.37 | 94.17 | 89.36 | -0.00 [-0.01, +0.01] | -1.960 | 0.00 |
| translate | chain | `crown_hot` | 0.37 | 94.88 | 89.74 | -0.01 [-0.03, +0.00] | -1.985 | 0.00 |
| translate | chain | `crown_stemem_hot` | 0.41 | 94.79 | 89.51 | +0.03 [+0.02, +0.04] | -1.956 | 0.00 |

## butterfly_major: test evaluation

| Arm | Plain | TTA |
| :-- | --: | --: |
| `base` | 70.30 | 75.18 |
| `crown` | 71.72 | 75.77 |
| `crown_ste` | 70.89 | 71.79 |
| `crown_stemem` | 70.78 | 74.92 |
| `crown_perturb` | 71.75 | 76.08 |
| `crown_all` | 72.11 | 77.16 |
| `crown_all_perturb` | 72.09 | 77.02 |
| `crown_hot` | 69.95 | 73.80 |
| `crown_stemem_hot` | 4.86 | 4.98 |

| Treatment minus control | Plain delta [95% CI] | TTA delta [95% CI] |
| :-- | :-- | :-- |
| `crown` − `base` | +1.43 [+1.20, +1.67] | +0.59 [+0.24, +0.94] |
| `crown_ste` − `base` | +0.60 [+0.32, +0.93] | -3.39 [-4.54, -2.28] |
| `crown_stemem` − `base` | +0.48 [+0.30, +0.70] | -0.27 [-0.59, +0.05] |
| `crown_perturb` − `base` | +1.45 [+1.26, +1.66] | +0.90 [+0.64, +1.21] |
| `crown_all` − `base` | +1.82 [+1.63, +2.02] | +1.98 [+1.80, +2.20] |
| `crown_all_perturb` − `base` | +1.80 [+1.62, +1.99] | +1.83 [+1.63, +2.08] |
| `crown_hot` − `base` | -0.34 [-0.66, -0.05] | -1.38 [-1.80, -0.96] |
| `crown_stemem_hot` − `base` | -65.43 [-65.89, -64.86] | -70.20 [-70.91, -69.28] |
| `crown_stemem` − `crown` | -0.95 [-1.20, -0.73] | -0.85 [-1.35, -0.35] |
| `crown_all` − `crown` | +0.39 [+0.29, +0.49] | +1.40 [+1.13, +1.70] |
| `crown_perturb` − `crown` | +0.03 [-0.02, +0.07] | +0.31 [+0.10, +0.57] |
| `crown_all_perturb` − `crown_all` | -0.02 [-0.04, -0.00] | -0.15 [-0.22, -0.08] |
| `crown_stemem_hot` − `crown` | -66.86 [-67.28, -66.35] | -70.78 [-71.67, -69.63] |

## butterfly_major: centre calibration

139 distinct references / selected pairs; 13761 rows. Group counts: `{'(malleti x plesseni) x malleti': 22, 'malleti': 36, 'cyrbia': 33, 'lativitta': 27, 'notabilis x lativitta': 21}`.

Cut trait counts: `{1: 73, 9: 13, 7: 16, 5: 19, 3: 18}`. Cut-out recall is the removed-channel regional occupancy. Track is undefined for cut-out. Judge values average all channels before counting positive rows; they are not calibrated probabilities.

| Transform | Memory | Arm | Recall % | Track % | IoU(Ty0) % | Recall delta vs base [95% CI], pp | Label LLR/cell | Positive rows % |
| :-- | :-- | :-- | --: | --: | --: | :-- | --: | --: |
| flipH | ref | `base` | 50.47 | 0.00 | 0.00 | — | +1.036 | 100.00 |
| flipH | ref | `crown` | 52.79 | 0.00 | 0.00 | +2.32 [+2.12, +2.52] | +1.072 | 100.00 |
| flipH | ref | `crown_ste` | 52.73 | 0.00 | 0.00 | +2.26 [+2.07, +2.46] | +1.076 | 100.00 |
| flipH | ref | `crown_stemem` | 51.17 | 0.00 | 0.00 | +0.70 [+0.63, +0.77] | +1.048 | 100.00 |
| flipH | ref | `crown_perturb` | 52.34 | 0.00 | 0.00 | +1.87 [+1.70, +2.04] | +1.063 | 100.00 |
| flipH | ref | `crown_all` | 51.25 | 0.00 | 0.00 | +0.78 [+0.70, +0.86] | +1.049 | 100.00 |
| flipH | ref | `crown_all_perturb` | 51.06 | 0.00 | 0.00 | +0.59 [+0.52, +0.65] | +1.046 | 100.00 |
| flipH | ref | `crown_hot` | 52.62 | 0.01 | 0.00 | +2.15 [+1.91, +2.39] | +1.071 | 100.00 |
| flipH | ref | `crown_stemem_hot` | 37.35 | 1.30 | 0.09 | -13.12 [-15.63, -10.59] | +0.383 | 100.00 |
| flipH | chain | `base` | 50.32 | 0.00 | 0.00 | — | +1.034 | 100.00 |
| flipH | chain | `crown` | 52.59 | 0.00 | 0.00 | +2.27 [+2.06, +2.47] | +1.068 | 100.00 |
| flipH | chain | `crown_ste` | 52.63 | 0.00 | 0.00 | +2.31 [+2.07, +2.56] | +1.073 | 100.00 |
| flipH | chain | `crown_stemem` | 51.07 | 0.00 | 0.00 | +0.75 [+0.62, +0.86] | +1.046 | 100.00 |
| flipH | chain | `crown_perturb` | 52.15 | 0.00 | 0.00 | +1.83 [+1.65, +2.00] | +1.060 | 100.00 |
| flipH | chain | `crown_all` | 51.16 | 0.00 | 0.00 | +0.83 [+0.74, +0.94] | +1.047 | 100.00 |
| flipH | chain | `crown_all_perturb` | 50.96 | 0.00 | 0.00 | +0.64 [+0.55, +0.74] | +1.044 | 100.00 |
| flipH | chain | `crown_hot` | 52.57 | 0.08 | 0.03 | +2.25 [+2.01, +2.50] | +1.065 | 99.28 |
| flipH | chain | `crown_stemem_hot` | 27.20 | 6.80 | 1.39 | -23.12 [-25.33, -20.98] | +0.035 | 64.03 |
| rot90 | ref | `base` | 16.66 | 0.60 | 0.42 | — | +0.319 | 84.89 |
| rot90 | ref | `crown` | 18.70 | 0.51 | 0.30 | +2.04 [+1.46, +2.64] | +0.348 | 80.58 |
| rot90 | ref | `crown_ste` | 19.97 | 1.06 | 0.68 | +3.31 [+2.70, +3.96] | +0.354 | 80.58 |
| rot90 | ref | `crown_stemem` | 17.46 | 0.76 | 0.54 | +0.80 [+0.30, +1.30] | +0.310 | 81.29 |
| rot90 | ref | `crown_perturb` | 18.31 | 0.52 | 0.32 | +1.65 [+1.09, +2.21] | +0.343 | 82.01 |
| rot90 | ref | `crown_all` | 17.70 | 0.58 | 0.39 | +1.04 [+0.48, +1.61] | +0.310 | 80.58 |
| rot90 | ref | `crown_all_perturb` | 17.79 | 0.45 | 0.30 | +1.13 [+0.57, +1.71] | +0.328 | 81.29 |
| rot90 | ref | `crown_hot` | 19.06 | 6.99 | 3.49 | +2.40 [+1.66, +3.15] | +0.199 | 79.14 |
| rot90 | ref | `crown_stemem_hot` | 17.42 | 15.10 | 1.63 | +0.76 [-0.70, +2.23] | +0.082 | 89.21 |
| rot90 | chain | `base` | 14.61 | 3.54 | 3.15 | — | +0.234 | 72.66 |
| rot90 | chain | `crown` | 18.00 | 2.93 | 2.07 | +3.39 [+2.66, +4.17] | +0.315 | 76.98 |
| rot90 | chain | `crown_ste` | 18.14 | 3.14 | 2.05 | +3.53 [+2.75, +4.36] | +0.310 | 76.26 |
| rot90 | chain | `crown_stemem` | 16.38 | 3.05 | 2.27 | +1.76 [+1.15, +2.40] | +0.273 | 76.26 |
| rot90 | chain | `crown_perturb` | 17.43 | 3.52 | 2.73 | +2.82 [+2.10, +3.58] | +0.282 | 76.26 |
| rot90 | chain | `crown_all` | 17.47 | 2.97 | 2.45 | +2.86 [+2.13, +3.65] | +0.308 | 77.70 |
| rot90 | chain | `crown_all_perturb` | 17.41 | 3.12 | 2.54 | +2.80 [+2.08, +3.58] | +0.307 | 77.70 |
| rot90 | chain | `crown_hot` | 18.02 | 7.22 | 3.68 | +3.41 [+2.62, +4.26] | +0.137 | 74.82 |
| rot90 | chain | `crown_stemem_hot` | 8.11 | 17.18 | 4.10 | -6.51 [-7.88, -5.06] | -0.200 | 20.86 |
| warp | ref | `base` | 1.62 | 94.25 | 92.32 | — | -1.134 | 0.00 |
| warp | ref | `crown` | 3.78 | 97.57 | 90.61 | +2.16 [+1.98, +2.35] | -1.166 | 0.00 |
| warp | ref | `crown_ste` | 3.90 | 97.81 | 90.68 | +2.28 [+2.07, +2.49] | -1.164 | 0.00 |
| warp | ref | `crown_stemem` | 2.31 | 95.55 | 92.13 | +0.69 [+0.61, +0.79] | -1.145 | 0.00 |
| warp | ref | `crown_perturb` | 3.19 | 97.25 | 91.32 | +1.57 [+1.42, +1.73] | -1.160 | 0.00 |
| warp | ref | `crown_all` | 2.08 | 96.15 | 92.91 | +0.47 [+0.38, +0.55] | -1.145 | 0.00 |
| warp | ref | `crown_all_perturb` | 1.91 | 95.90 | 92.99 | +0.29 [+0.21, +0.37] | -1.142 | 0.00 |
| warp | ref | `crown_hot` | 4.52 | 94.56 | 87.39 | +2.90 [+2.63, +3.17] | -1.170 | 0.00 |
| warp | ref | `crown_stemem_hot` | 16.93 | 55.25 | 8.52 | +15.31 [+14.47, +16.13] | -0.230 | 1.44 |
| warp | chain | `base` | 1.63 | 93.95 | 91.91 | — | -1.133 | 0.00 |
| warp | chain | `crown` | 3.55 | 97.45 | 90.87 | +1.92 [+1.74, +2.11] | -1.165 | 0.00 |
| warp | chain | `crown_ste` | 3.71 | 97.45 | 90.58 | +2.08 [+1.89, +2.27] | -1.160 | 0.00 |
| warp | chain | `crown_stemem` | 2.27 | 95.16 | 91.84 | +0.64 [+0.55, +0.73] | -1.141 | 0.00 |
| warp | chain | `crown_perturb` | 2.97 | 97.10 | 91.49 | +1.34 [+1.19, +1.48] | -1.159 | 0.00 |
| warp | chain | `crown_all` | 2.03 | 96.02 | 92.82 | +0.39 [+0.30, +0.48] | -1.145 | 0.00 |
| warp | chain | `crown_all_perturb` | 1.85 | 95.71 | 92.89 | +0.22 [+0.13, +0.31] | -1.142 | 0.00 |
| warp | chain | `crown_hot` | 4.50 | 94.96 | 87.67 | +2.86 [+2.59, +3.15] | -1.172 | 0.00 |
| warp | chain | `crown_stemem_hot` | 4.15 | 44.94 | 17.22 | +2.52 [+2.05, +2.98] | -0.651 | 0.72 |
| cutout | ref | `base` | 30.45 | undefined | 86.51 | — | -0.028 | 51.08 |
| cutout | ref | `crown` | 46.09 | undefined | 81.99 | +15.63 [+11.05, +20.51] | +0.050 | 68.35 |
| cutout | ref | `crown_ste` | 49.57 | undefined | 82.07 | +19.12 [+13.66, +24.75] | +0.074 | 76.26 |
| cutout | ref | `crown_stemem` | 44.13 | undefined | 82.88 | +13.67 [+9.13, +18.56] | +0.039 | 65.47 |
| cutout | ref | `crown_perturb` | 44.30 | undefined | 82.59 | +13.85 [+9.25, +18.70] | +0.041 | 66.91 |
| cutout | ref | `crown_all` | 41.25 | undefined | 83.91 | +10.80 [+7.05, +14.84] | +0.027 | 59.71 |
| cutout | ref | `crown_all_perturb` | 40.92 | undefined | 83.69 | +10.47 [+6.83, +14.46] | +0.025 | 59.71 |
| cutout | ref | `crown_hot` | 41.77 | undefined | 79.39 | +11.32 [+7.37, +15.38] | +0.022 | 58.27 |
| cutout | ref | `crown_stemem_hot` | 13.65 | undefined | 10.19 | -16.80 [-21.79, -11.88] | +0.119 | 96.40 |
| cutout | chain | `base` | 25.17 | undefined | 86.62 | — | -0.057 | 43.17 |
| cutout | chain | `crown` | 41.11 | undefined | 82.20 | +15.93 [+11.52, +20.63] | +0.024 | 61.87 |
| cutout | chain | `crown_ste` | 42.14 | undefined | 81.97 | +16.97 [+12.13, +22.11] | +0.036 | 64.75 |
| cutout | chain | `crown_stemem` | 37.70 | undefined | 82.60 | +12.53 [+8.21, +17.12] | +0.009 | 58.27 |
| cutout | chain | `crown_perturb` | 39.75 | undefined | 82.55 | +14.58 [+10.35, +19.15] | +0.018 | 58.99 |
| cutout | chain | `crown_all` | 36.63 | undefined | 84.02 | +11.46 [+7.38, +15.89] | +0.002 | 54.68 |
| cutout | chain | `crown_all_perturb` | 35.65 | undefined | 83.82 | +10.48 [+6.57, +14.73] | -0.003 | 54.68 |
| cutout | chain | `crown_hot` | 38.96 | undefined | 79.60 | +13.79 [+9.75, +17.90] | +0.009 | 51.80 |
| cutout | chain | `crown_stemem_hot` | 12.12 | undefined | 21.68 | -13.05 [-18.82, -7.22] | -0.003 | 47.48 |
| translate | ref | `base` | 0.00 | 94.49 | 91.80 | — | -1.554 | 0.00 |
| translate | ref | `crown` | 0.00 | 97.73 | 89.71 | +0.00 [+0.00, +0.00] | -1.622 | 0.00 |
| translate | ref | `crown_ste` | 0.00 | 97.58 | 90.22 | +0.00 [-0.00, +0.00] | -1.617 | 0.00 |
| translate | ref | `crown_stemem` | 0.00 | 95.73 | 91.60 | -0.00 [-0.00, +0.00] | -1.578 | 0.00 |
| translate | ref | `crown_perturb` | 0.00 | 97.45 | 90.61 | +0.00 [+0.00, +0.00] | -1.608 | 0.00 |
| translate | ref | `crown_all` | 0.00 | 96.43 | 92.41 | +0.00 [-0.00, +0.00] | -1.577 | 0.00 |
| translate | ref | `crown_all_perturb` | 0.00 | 96.20 | 92.51 | +0.00 [-0.00, +0.00] | -1.571 | 0.00 |
| translate | ref | `crown_hot` | 0.01 | 96.23 | 85.58 | +0.01 [+0.00, +0.02] | -1.650 | 0.00 |
| translate | ref | `crown_stemem_hot` | 15.11 | 70.55 | 12.95 | +15.11 [+14.14, +16.09] | -0.610 | 0.00 |
| translate | chain | `base` | 0.00 | 93.67 | 90.81 | — | -1.546 | 0.00 |
| translate | chain | `crown` | 0.00 | 97.40 | 90.31 | +0.00 [+0.00, +0.00] | -1.610 | 0.00 |
| translate | chain | `crown_ste` | 0.00 | 97.22 | 89.87 | +0.00 [-0.00, +0.00] | -1.609 | 0.00 |
| translate | chain | `crown_stemem` | 0.00 | 94.97 | 90.91 | -0.00 [-0.00, +0.00] | -1.568 | 0.00 |
| translate | chain | `crown_perturb` | 0.00 | 96.99 | 90.96 | +0.00 [+0.00, +0.00] | -1.596 | 0.00 |
| translate | chain | `crown_all` | 0.00 | 95.87 | 92.33 | +0.00 [-0.00, +0.00] | -1.568 | 0.00 |
| translate | chain | `crown_all_perturb` | 0.00 | 95.57 | 92.34 | +0.00 [-0.00, +0.00] | -1.562 | 0.00 |
| translate | chain | `crown_hot` | 0.01 | 96.31 | 85.91 | +0.01 [+0.00, +0.01] | -1.645 | 0.00 |
| translate | chain | `crown_stemem_hot` | 1.70 | 57.90 | 31.79 | +1.70 [+1.43, +2.00] | -1.248 | 0.00 |
