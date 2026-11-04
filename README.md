# OC-CCL experiments

I started this project to reproduce [Static Segmentation by Tracking](https://arxiv.org/abs/2501.06749) (SST) and
understand when its open–close cycle loss improves segmentation. SST treats a
labeled support image and a query image as frames of a video: the tracker carries
trait masks from the support to the query.

The work grew into an evaluation harness and a sequence of controlled changes to
support selection, orientation handling, fine-tuning and tracker architecture.
The main contribution is the comparison: which changes help, on which dataset,
and whether a better cycle reconstruction also improves the query prediction.

The implementation published here covers S1–S17d and the S18 build. The later
findings below are documented from local experiments; their implementation and
full run bundles are not included in this publication.

## What the experiments show

The later experiments separate better cycle behaviour from better segmentation.
Counterfactual centre training reduced mask persistence on edited references,
but slightly lowered butterfly test mIoU against its all-leg control. Training
with multiple references helped beetle; matching butterfly support-label
conventions gave smaller gains. Shape losses improved some structures while
hurting others.

[What changed after S18](docs/LATER_EXPERIMENTS.md) follows these comparisons,
including the negative results and the local tiling probe. The probe gained
1.55 points on 40 selected butterfly queries; it has not yet been confirmed
in S24's full tuned evaluation.

Selected recorded comparisons, in **argmax mIoU (%)**:

| Comparison | Control → treatment | Difference, percentage points | Evidence |
| :-- | :-- | :-- | :-- |
| Beetle, SAM 3, 1-shot: base → `crown_all_mref` | 66.32 → 70.19 | +3.87 [2.97, 4.78] | [S20](docs/s18b/S20_RESULTS.md) |
| Beetle, SAM 3, 5-shot: base → `crown_all_mref` | 77.96 → 78.95 | +0.99 [0.58, 1.38] | [S20](docs/s18b/S20_RESULTS.md) |
| Butterfly Major, SAM 2, 5-shot: curated → label-consistent support pool, `crown_all_mref` | 78.56 → 78.75 | +0.19 [0.08, 0.29] | [S21](docs/s18b/S21_RESULTS.md) |

All rows use orientation test-time augmentation (TTA) and 20 support seeds.
Intervals use 10,000 bootstrap resamples over support seeds, seed 0, and are
unadjusted for multiple comparisons. They are conditional on fitted adapters,
not variation across repeated training. In the pool comparison, two subspecies'
support draws change; pairing by seed does not make those draws identical.
`crown_all_mref` denotes all-leg adaptation with multiple training references.
The butterfly filtering thresholds were informed by test-annotation inspection;
that comparison is diagnostic rather than an untouched-test generalization claim.

These are selected within-project comparisons, not a single leaderboard or proof
of exact reproduction. The beetle split and support curation approximate unpublished
choices; butterfly annotation conventions also affect the remaining gap. Best-of-3
boundary scores are secondary diagnostics. Failed gates and negative results remain
in the evidence record.

S18–S23 have recorded results in this checkout. S18a's merged gate still fails on
the butterfly hot memory-STE collapse. S24 is implemented and preregistered, with a
local zoom probe, but its full evaluation is still marked unrun. A result does not
automatically change the established `sam2_crown` training recipe.

## Read or reproduce

- [Later experiments](docs/LATER_EXPERIMENTS.md): findings, limitations and stage records.
- [Evidence index](docs/README.md): methods, plans and result reports.
- [Runbook](RUNBOOK.md): setup and execution for the published implementation.
- [Experiment ledger](EXPERIMENTS.md): the published reproduction and development history.

Later plans describe the local research implementation at the time of each run.
Their code paths and bundle references are provenance records, not a claim that
those files are included here. S24 remains unrun. The established `sam2_crown`
recipe is unchanged.
