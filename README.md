# OC-CCL experiments

I started this project to reproduce [Static Segmentation by Tracking](https://arxiv.org/abs/2501.06749)
(SST) and understand when its open–close cycle loss improves segmentation.
SST treats a labeled support image and a query image as two video frames:
the tracker carries trait masks from the support to the query.

The work builds an evaluation harness, then compares support selection,
orientation handling, fine-tuning and tracker architecture. The central question
is whether a better cycle reconstruction also improves the query prediction.
This published version contains S1–S17d and the S18 implementation; S18 results
are not included.

## Findings and limits

Beetle results approached the paper's reported baseline under curated,
species-balanced support selection. Butterfly Major remained below it.
Orientation handling helped, but a stronger tracker did not remove the butterfly
gap. Fine-tuning gains depended on the dataset and configuration.

Selected recorded results, in mIoU (%) over 20 support seeds:

| Configuration | Butterfly Major | Beetle |
| :-- | --: | --: |
| SAM 2, 1-shot, uniform supports (S4a/S5a) | 68.54 ± 3.15 | 55.96 ± 7.90 |
| SAM 2, 1-shot, canonical and species-balanced supports (S4c) | — | 63.20 ± 3.89 |
| SAM 2, 5-shot (S6) | 71.78 ± 0.15 | 73.16 ± 2.31 |
| SAM 3, 5-shot, orientation TTA (S15) | 77.95 ± 0.18 | 77.94 ± 2.19 |
| SAM 3, 5-shot, architecture stack and TTA (S15) | 77.47 ± 0.26 | 78.91 ± 1.99 |

The spreads describe support-seed variation, not repeated training. These rows
use different configurations and should not be read as isolated treatment effects.
The [experiment ledger](EXPERIMENTS.md) retains the paired comparisons, failed
gates and negative results.

Exact reproduction remains limited by unpublished support choices and the
paper's unavailable beetle split. This repository uses its own beetle split and
explicit support-curation rules. Curation also uses more labeled masks than the
one- or five-shot prediction itself; the shot count does not describe the full
label budget. Overlap resolution is inferred rather than specified by the paper.

## Read or run

- [Runbook](RUNBOOK.md): setup, cluster execution, returned bundles and resume rules.
- [Experiment ledger](EXPERIMENTS.md): methods, stage history, results and limitations.
- [Documentation index](docs/README.md): figures and presentation material.

The package is in `src/occl/`; workers and gate checks are in `scripts/`.
Notebook `.py` files are the editable sources for their `.ipynb` counterparts.
Model checkouts, weights and the image cache are external inputs. Follow the
runbook before running tests or GPU stages. A stage is complete only when its
bundle passes the corresponding gate.
