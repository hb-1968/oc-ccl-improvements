# S23 evidence read — thin-structure shape terms and ventral oversampling (SAM 2, butterfly)

Source: results/inbox/s23.zip. Nothing is selected; this report only reads. Methods: [S23_PLAN.md](S23_PLAN.md).

Every arm is a refit of `crown_all_mref_lc`'s recipe on S21's CLEAN manifest; every cell uses the label-consistent pool, so each arm pairs with the parent on IDENTICAL draws (`registry.compare`). CIs: 10,000 resamples over support seeds, seed 0, unadjusted; ✱ = excludes zero. "hw" = whether the arm's cell ran on the same GPU type as the parent's.

Screen gate: PASS — S21 8824c5afb995; eligible=['crown_all_mref_lc_shape', 'crown_all_mref_lc_ovs', 'crown_all_mref_lc_shape_ovs']; fits screened; test cells not yet demanded.
Confirmation gate: PASS — S21 8824c5afb995; eligible=['crown_all_mref_lc_shape', 'crown_all_mref_lc_ovs', 'crown_all_mref_lc_shape_ovs']; all 6 cells and 3 probes complete.

| Fit | steps | ‖ΔW‖ | Fit min | Val mIoU % (parent) | Eligible |
| :-- | --: | --: | --: | --: | :-- |
| crown_all_mref_lc_shape | 870 | 6.324 | 13.4 | 82.10 (82.21) | yes |
| crown_all_mref_lc_ovs | 961 | 6.711 | 13.7 | 82.19 (82.21) | yes |
| crown_all_mref_lc_shape_ovs | 961 | 6.648 | 15.0 | 82.04 (82.21) | yes |

## 1-shot (label-consistent pool)

| Arm | mIoU | Δ Major vs parent | (m×p)×m | malleti | lativitta | best-of-3 Δ | vs base | hw |
| :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| crown_all_mref_lc (parent, S21) | 77.31 | | | | | | | NVIDIA A40 |
| crown_all_mref_lc_shape | 77.16 | -0.15 [-0.20, -0.10] ✱ | -0.31 [-0.52, -0.09] ✱ | +0.08 [+0.04, +0.11] ✱ | -0.58 [-0.63, -0.55] ✱ | -0.07 [-0.12, -0.02] ✱ | +1.92 [+1.65, +2.21] ✱ | DIFFERENT/unknown |
| crown_all_mref_lc_ovs | 77.29 | -0.02 [-0.06, +0.02] | +0.04 [-0.12, +0.21] | -0.08 [-0.12, -0.04] ✱ | -0.12 [-0.14, -0.09] ✱ | -0.02 [-0.06, +0.02] | +2.04 [+1.77, +2.35] ✱ | DIFFERENT/unknown |
| crown_all_mref_lc_shape_ovs | 77.18 | -0.13 [-0.18, -0.08] ✱ | -0.04 [-0.22, +0.15] | -0.06 [-0.16, +0.02] | -0.61 [-0.64, -0.57] ✱ | -0.06 [-0.11, -0.01] ✱ | +1.93 [+1.66, +2.23] ✱ | DIFFERENT/unknown |

Stack additivity (stack − parent) − [(shape − parent) + (ovs − parent)]: +0.04 [-0.02, +0.10].

**(m×p)×m by side and trait, 1-shot** (cell mean, and Δ vs parent paired per seed):

| Cell | dorsal | ventral | ventral 5/8 streak | ventral 6/9 spots | ventral 2/3 band | ventral 7/10 rays | ventral 1/4 patch |
| :-- | --: | --: | :-- | :-- | :-- | :-- | :-- |
| crown_all_mref_lc_lcpool_tta | 81.06 | 55.01 | 25.28 | 38.37 | 67.62 | 61.41 | 82.84 |
| crown_all_mref_lc_shape_lcpool_tta | 80.83 | 54.56 | 25.69 (+0.40 [+0.08, +0.74] ✱) | 37.04 (-1.33 [-2.00, -0.68] ✱) | 67.11 (-0.51 [-1.67, +0.63]) | 61.31 (-0.10 [-0.58, +0.42]) | 82.12 (-0.72 [-1.64, +0.15]) |
| crown_all_mref_lc_ovs_lcpool_tta | 81.02 | 55.25 | 26.13 (+0.84 [+0.38, +1.30] ✱) | 37.56 (-0.81 [-1.48, -0.18] ✱) | 69.72 (+2.09 [+1.11, +3.16] ✱) | 61.22 (-0.19 [-0.51, +0.10]) | 82.06 (-0.78 [-1.69, +0.08]) |
| crown_all_mref_lc_shape_ovs_lcpool_tta | 80.78 | 55.51 | 26.37 (+1.08 [+0.67, +1.50] ✱) | 36.93 (-1.44 [-2.29, -0.67] ✱) | 71.30 (+3.68 [+2.40, +4.97] ✱) | 61.20 (-0.20 [-0.61, +0.25]) | 82.12 (-0.72 [-1.62, +0.08]) |

**lativitta rays 7/8 (harm monitor), 1-shot:** crown_all_mref_lc 48.04; crown_all_mref_lc_shape 46.85 (-1.20 [-1.22, -1.17] ✱); crown_all_mref_lc_ovs 47.66 (-0.38 [-0.40, -0.36] ✱); crown_all_mref_lc_shape_ovs 46.68 (-1.36 [-1.40, -1.32] ✱)

| Cell (per subspecies, mIoU %) | cyrbia | lativitta | malleti | (m×p)×m | notab×lat |
| :-- | --: | --: | --: | --: | --: |
| crown_all_mref_lc_lcpool_tta | 74.92 | 74.45 | 80.60 | 72.60 | 83.97 |
| crown_all_mref_lc_shape_lcpool_tta | 74.86 | 73.87 | 80.68 | 72.30 | 84.10 |
| crown_all_mref_lc_ovs_lcpool_tta | 74.99 | 74.33 | 80.52 | 72.65 | 83.95 |
| crown_all_mref_lc_shape_ovs_lcpool_tta | 74.89 | 73.84 | 80.54 | 72.57 | 84.05 |

## 5-shot (label-consistent pool)

| Arm | mIoU | Δ Major vs parent | (m×p)×m | malleti | lativitta | best-of-3 Δ | vs base | hw |
| :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| crown_all_mref_lc (parent, S21) | 78.76 | | | | | | | NVIDIA A40 |
| crown_all_mref_lc_shape | 78.62 | -0.13 [-0.18, -0.09] ✱ | -0.21 [-0.40, -0.01] ✱ | +0.02 [-0.00, +0.05] | -0.56 [-0.58, -0.53] ✱ | -0.06 [-0.10, -0.02] ✱ | +1.36 [+1.28, +1.44] ✱ | DIFFERENT/unknown |
| crown_all_mref_lc_ovs | 78.77 | +0.01 [-0.03, +0.05] | +0.18 [-0.01, +0.38] | -0.04 [-0.07, -0.03] ✱ | -0.14 [-0.15, -0.12] ✱ | +0.01 [-0.03, +0.06] | +1.51 [+1.43, +1.58] ✱ | DIFFERENT/unknown |
| crown_all_mref_lc_shape_ovs | 78.66 | -0.09 [-0.13, -0.05] ✱ | -0.01 [-0.17, +0.15] | -0.05 [-0.08, -0.02] ✱ | -0.59 [-0.62, -0.57] ✱ | -0.03 [-0.07, +0.02] | +1.40 [+1.32, +1.48] ✱ | DIFFERENT/unknown |

Stack additivity (stack − parent) − [(shape − parent) + (ovs − parent)]: +0.03 [-0.02, +0.07].

**(m×p)×m by side and trait, 5-shot** (cell mean, and Δ vs parent paired per seed):

| Cell | dorsal | ventral | ventral 5/8 streak | ventral 6/9 spots | ventral 2/3 band | ventral 7/10 rays | ventral 1/4 patch |
| :-- | --: | --: | :-- | :-- | :-- | :-- | :-- |
| crown_all_mref_lc_lcpool_5_tta | 82.02 | 55.67 | 25.61 | 39.79 | 69.16 | 61.70 | 82.55 |
| crown_all_mref_lc_shape_lcpool_5_tta | 81.80 | 55.50 | 25.92 (+0.32 [+0.01, +0.61] ✱) | 38.80 (-1.00 [-1.56, -0.45] ✱) | 69.42 (+0.26 [-0.64, +1.17]) | 61.64 (-0.06 [-0.56, +0.46]) | 82.15 (-0.40 [-1.52, +0.73]) |
| crown_all_mref_lc_ovs_lcpool_5_tta | 81.98 | 56.33 | 26.43 (+0.82 [+0.33, +1.27] ✱) | 39.53 (-0.27 [-0.83, +0.30]) | 72.28 (+3.12 [+2.24, +4.02] ✱) | 61.42 (-0.28 [-0.77, +0.25]) | 82.36 (-0.19 [-1.37, +0.98]) |
| crown_all_mref_lc_shape_ovs_lcpool_5_tta | 81.79 | 56.16 | 26.46 (+0.86 [+0.38, +1.31] ✱) | 38.48 (-1.32 [-1.90, -0.74] ✱) | 73.34 (+4.18 [+3.26, +5.09] ✱) | 61.26 (-0.44 [-0.89, +0.01]) | 81.62 (-0.93 [-1.94, +0.06]) |

**lativitta rays 7/8 (harm monitor), 5-shot:** crown_all_mref_lc 48.22; crown_all_mref_lc_shape 47.09 (-1.13 [-1.14, -1.12] ✱); crown_all_mref_lc_ovs 47.86 (-0.35 [-0.37, -0.34] ✱); crown_all_mref_lc_shape_ovs 46.93 (-1.29 [-1.30, -1.27] ✱)

| Cell (per subspecies, mIoU %) | cyrbia | lativitta | malleti | (m×p)×m | notab×lat |
| :-- | --: | --: | --: | --: | --: |
| crown_all_mref_lc_lcpool_5_tta | 79.14 | 74.87 | 81.69 | 73.47 | 84.62 |
| crown_all_mref_lc_shape_lcpool_5_tta | 79.02 | 74.31 | 81.72 | 73.27 | 84.81 |
| crown_all_mref_lc_ovs_lcpool_5_tta | 79.21 | 74.73 | 81.65 | 73.66 | 84.61 |
| crown_all_mref_lc_shape_ovs_lcpool_5_tta | 79.14 | 74.27 | 81.64 | 73.47 | 84.80 |

## Lever effects on matched hardware (within S23)

| Effect | shot | Major | (m×p)×m | ventral 5/8 | ventral 6/9 | ventral 2/3 | lativitta rays 7/8 |
| :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- |
| SHAPE (stack − ovs) | 1 | -0.11 [-0.15, -0.07] ✱ | -0.08 [-0.24, +0.08] | +0.24 [-0.07, +0.55] | -0.63 [-1.15, -0.05] ✱ | +1.58 [+0.74, +2.41] ✱ | -0.98 [-1.00, -0.94] ✱ |
| SHAPE (stack − ovs) | 5 | -0.11 [-0.15, -0.07] ✱ | -0.19 [-0.36, -0.00] ✱ | +0.03 [-0.33, +0.43] | -1.05 [-1.59, -0.48] ✱ | +1.06 [+0.30, +1.92] ✱ | -0.94 [-0.95, -0.92] ✱ |
| OVS (stack − shape) | 1 | +0.02 [-0.03, +0.06] | +0.27 [+0.13, +0.41] ✱ | +0.68 [+0.32, +1.04] ✱ | -0.11 [-0.49, +0.25] | +4.18 [+2.84, +5.62] ✱ | -0.16 [-0.19, -0.12] ✱ |
| OVS (stack − shape) | 5 | +0.04 [+0.01, +0.07] ✱ | +0.20 [+0.07, +0.33] ✱ | +0.54 [+0.14, +0.91] ✱ | -0.32 [-0.88, +0.22] | +3.92 [+3.04, +4.87] ✱ | -0.16 [-0.17, -0.15] ✱ |

| Memorization probe (5-shot, curated) | cut-out occupancy soft % | parent % | Δ, pp |
| :-- | --: | --: | :-- |
| crown_all_mref_lc_shape | 33.87 | 33.26 | +0.61 [+0.22, +1.00] ✱ |
| crown_all_mref_lc_ovs | 33.45 | 33.26 | +0.19 [-0.05, +0.45] |
| crown_all_mref_lc_shape_ovs | 34.00 | 33.26 | +0.74 [+0.32, +1.17] ✱ |
