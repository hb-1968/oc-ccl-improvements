# Zoom probe — results

Source: `results/diagnostics/zoom_probe.json`; 80 (support, query) pairs, 40 queries. Pre-registration and method: [ZOOM_PROBE.md](ZOOM_PROBE.md). Per query, each variant is averaged over its supports; CIs bootstrap the queries (registry `paired_bootstrap`, 10,000 resamples, seed 0); ✱ = 95 % CI excludes zero. Per-trait rows bootstrap (query, trait) units.

| Variant | mIoU % (all) | cyrbia | malleti |
| :-- | --: | --: | --: |
| F | 78.08 | 78.47 | 77.69 |
| S | 79.32 | 79.98 | 78.67 |
| S* | 79.92 | 80.04 | 79.80 |
| Q | 79.63 | 79.60 | 79.66 |

| Contrast | pooled | cyrbia | malleti | traits < 5 cells | traits 5–12 | traits ≥ 12 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| S − F | +1.24 [+0.40, +2.18] ✱ | +1.51 [+0.48, +2.87] ✱ | +0.98 [-0.25, +2.23] | +6.10 [+4.64, +7.64] ✱ (n=92) | -0.64 [-2.28, +0.96] (n=162) | -1.23 [-2.97, -0.13] ✱ (n=66) |
| Q − F | +1.55 [+0.55, +2.60] ✱ | +1.13 [-0.08, +2.54] | +1.96 [+0.45, +3.49] ✱ | +6.54 [+4.57, +8.51] ✱ (n=92) | -0.19 [-1.43, +1.08] (n=162) | -0.65 [-2.61, +1.15] (n=66) |
| S* − F | +1.84 [+0.75, +3.08] ✱ | +1.57 [-0.11, +3.67] | +2.11 [+0.86, +3.40] ✱ | +5.84 [+4.02, +7.56] ✱ (n=92) | +0.37 [-1.14, +1.99] (n=162) | +0.21 [-0.65, +1.61] (n=66) |
| S* − S | +0.60 [-0.02, +1.25] | +0.06 [-0.78, +0.99] | +1.14 [+0.26, +2.05] ✱ | -0.26 [-1.24, +0.45] (n=92) | +1.01 [+0.03, +2.12] ✱ (n=162) | +1.44 [+0.02, +3.46] ✱ (n=66) |

Zoom factor (1024 / box side): first-pass S median 1.54 (p10 1.28, p90 1.89); GT box S* median 1.48. Quadrant tiles add a further 1.67× per tile. First-pass fallbacks (nothing predicted): 0.

**Pre-registered decision:** better label-free variant = Q (+1.55 [+0.55, +2.60] ✱). **GO** — ≥ +0.5 ✱ pooled and thin traits gain more than wide ones.
