# What changed after S18

The later experiments separate three questions: whether the cycle follows an
edited target, whether adaptation improves query segmentation, and whether the
remaining errors come from support labels or spatial resolution. Those questions
do not have the same answer.

This overview summarizes locally recorded results through S23 and the zoom probe.
The later implementation and full run bundles remain unpublished; linked plans
and reports document the local experiments. S24's full evaluation remains unrun. Stage plans below retain the exact operations,
constants, controls and preregistered decisions; result reports retain all arms.

## A better diagnostic did not make a better segmenter

S18b introduced a known edit to the labeled reference at the centre of the cycle.
The target follows that edit, so predicting the original mask in the same place
is no longer enough. All-leg adaptation (`crown_all`) also leaves LoRA active on
the reverse passes, giving it a matched control for the added objective.

On butterfly validation data, the counterfactual arm reduced soft occupancy in
cut-out regions from **31.47% to 18.97%** relative to `crown_all`. Yet its one-shot
TTA test score fell from **77.16 to 77.02**: −0.14 percentage points
[−0.22, −0.04]. The intervention changed the measured response without improving
the primary segmentation score. Occupancy is not a probability of memorization.

S18c tried adapter mixtures and smaller centre-loss weights. None of its tested
butterfly variants exceeded `crown_all`'s mean test score. This is the useful
negative result: improving the cycle diagnostic was insufficient.
[Method](s18b/PLAN.md), [S18b results](s18b/S18B_RESULTS.md),
[S18c plan](s18b/S18C_PLAN.md) and [results](s18b/S18C_RESULTS.md).

## Training with several references helped beetle

S19 showed that five-shot inference reduced the additional gain from adaptation.
S19b then added four labeled training references to the forward memory, bringing
training closer to the five-reference inference setting. On beetle, this raised
`crown_all`'s five-shot score from **75.50 to 75.81**, a gain of **0.31**
[0.13, 0.50]. The butterfly gain was only **0.04** [0.01, 0.07].

S20 carried the recipe to SAM 3. On beetle, `crown_all_mref` reached **70.19**
at one shot and **78.95** at five shots, versus **66.32** and **77.96** for the
matched SAM 3 base. Those combined-recipe gains are **3.87** [2.97, 4.78] and
**0.99** [0.58, 1.38]; they do not isolate the contribution of extra references.
[Five-shot comparison](s18b/S19_RESULTS.md),
[multiple-reference method](s18b/S19B_PLAN.md) and [results](s18b/S19B_RESULTS.md),
[SAM 3 method](s18b/S20_PLAN.md) and [results](s18b/S20_RESULTS.md).

## Label consistency helped modestly; sharper losses had costs

S21 filtered butterfly support masks for label consistency, separately from
filtering the training anchors. For SAM 2 `crown_all_mref`, changing the support
pool raised five-shot mIoU from **78.56 to 78.75**, or **0.19** [0.08, 0.29].
The corresponding SAM 3 comparison in S22 gained **0.16** [0.07, 0.25].
The filtering thresholds were informed by inspection of test annotations;
these are diagnostic interventions, not an untouched-test generalization claim.
Support draws also change in the two filtered subspecies.
[Filtering method](s18b/S21_PLAN.md), [SAM 2 results](s18b/S21_RESULTS.md),
[SAM 3 method](s18b/S22_PLAN.md) and [results](s18b/S22_RESULTS.md).

S23 tested shape losses and ventral-reference oversampling. Adding shape loss to
oversampling reduced five-shot Major mIoU by **0.11** [−0.15, −0.07] in the
comparison on matched hardware. Some ventral bands improved while lativitta rays
worsened. Comparisons with the older parent also have a GPU-type mismatch, so the
within-S23 contrast is the cleaner read.
[Method](s18b/S23_PLAN.md) and [results](s18b/S23_RESULTS.md).

## Tiling is promising, but still a probe

A local CPU probe tested quadrant crops on base SAM 2: **78.08 → 79.63 mIoU**,
a gain of **1.55** [0.55, 2.60]. It used 40 selected upright dorsal queries from
two butterfly subspecies, two supports per query, one-shot inference and no TTA.
Query crops came from model predictions; query labels were used to select the
probe cohort. Upsampling cached pixels adds no new image detail.

This supported building S24's two tuned, TTA-enabled cells at one and five shots.
It does not establish their outcome. [Probe method](s18b/ZOOM_PROBE.md),
[results](s18b/ZOOM_PROBE_RESULTS.md) and [unrun S24 plan](s18b/S24_PLAN.md).

## How to read the numbers

Segmentation scores use the locked argmax mIoU, in percent; differences are
percentage points. Stage comparisons use TTA and 20 support seeds. Their 95%
intervals use 10,000 support-seed bootstrap resamples, seed 0, conditional on the
fitted adapters and unadjusted for multiple comparisons. They do not measure
variation across repeated training. Pool comparisons pair seeds, not identical
support draws in the filtered groups. The zoom probe instead bootstraps its 40
queries after averaging supports. Cut-out occupancy is a separate diagnostic.

The original reproduction limits still apply: unpublished support choices,
a repository-specific beetle split, and inferred overlap resolution. S18a's
failed gate remains in the record. None of these results automatically changes
the established `sam2_crown` recipe.
