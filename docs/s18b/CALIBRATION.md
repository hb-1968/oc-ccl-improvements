# S18b calibration: evidence and interpretation

Updated 2026-09-22 from the complete S18a pull. Applies to humans and subagents under
the local `AGENTS.md` instructions. Read [METHODS.md](METHODS.md) for exact operations,
constants, denominators and limits; [S18A_RESULTS.md](S18A_RESULTS.md) contains all arms,
transforms, memory conditions, paired CIs and links to artifact hashes.

## What is implemented and what is still proposed

The historical centre probe measures T(x0) under reference-only or chain memory and
stops there. As of 2026-09-22, a separate complete S18b training implementation and CPU
backward smoke exist: [PLAN.md](PLAN.md) is the current contract. It specifies cut-out/warp
counterfactuals, frozen base-derived weights, an original-cycle anchor plus the odd return,
erased-channel protection, prior/leg/perturbation controls and a separate two-judge arm.
The original image-only f(x0,x1) and calibrated posterior interpretation are not claimed.
GPU fitting/generalization remain unmeasured. The older PDF/status deck are historical.
All S18a observations below remain unchanged; new soft-channel metrics must not be
conflated with the historical argmax occupancy.

S18a's full merged gate is **FAIL**, 25/26 checks pass: butterfly `crown_stemem_hot`
diverged. Its TTA test mIoU is 4.98, despite that arm reaching 65.99 on beetle. We retain
that failure and keep `sam2_crown` as the user-established default; deviations remain
explicit experiments. Structural success of training-bundle gates was not sufficient to
establish optimization success.

## New transforms: measured, not inferred from mask separation

These chain-condition figures use 120 beetle and 139 selected butterfly training pairs.
Recall and track are fractions of the respective target-difference regions assigned the
trait, **not prediction-mass percentages**. They have different denominators and need
not sum to one. All deltas below are percentage points vs the same pair under base.

| Dataset / transform | Base recall % | Base track % | Crown recall % | Crown recall delta [95% CI] |
| :-- | --: | --: | --: | :-- |
| beetle / translate | 0.377 | 94.03 | 0.374 | -0.003 [-0.010, +0.003] |
| butterfly / translate | 0.001 | 93.67 | 0.002 | +0.001 [about 0, +0.003] |
| beetle / warp | 6.30 | 90.73 | 6.30 | -0.006 [-0.060, +0.047] |
| butterfly / warp | 1.63 | 93.95 | 3.55 | +1.92 [+1.74, +2.11] |
| beetle / cutout | 25.33 | undefined | 26.00 | +0.67 [-0.21, +1.66] |
| butterfly / cutout | 25.17 | undefined | 41.10 | +15.93 [+11.53, +20.63] |

**Translation:** base can track the shifted object on both datasets. The established
crown shows very little occupancy in the original-only region. This supports a usable
position-control read here, but does not rule out other shortcuts or prove a penalty
would help. Cropping and mean-colour fill remain transform confounds.

**Warp:** both base models track the displaced regions well. Butterfly crown increases
both recall-region occupancy and tracking occupancy (93.95 -> 97.45%), while transformed
IoU falls from 91.91 to 90.87. Thus more old-region occupancy can coexist with more
new-region occupancy; broader masks/boundary errors are alternatives to a recall-specific
mechanism. Beetle hot-memory-STE has recall +0.99 [+0.75,+1.28], while IoU remains 89.43.
The row-mean label judge remains negative on all non-collapsed arms' chain warp rows.

**Cut-out:** this exposes the clearest adapter-associated change. Butterfly crown
predicts the removed trait in 41.10% of its old region vs base 25.17%; `crown_all` is
36.63%, +11.46 [+7.38,+15.90]. The difference persists with reference-only memory
(base 30.45%, crown 46.09%), so adding the intervening query is not required for this
behaviour. Beetle crown's +0.67 CI spans zero, but `crown_hot` is +3.55 [+1.73,+5.48]
and `crown_stemem_hot` +4.96 [+2.98,+7.05]. These are spatial occupancy changes following
inpainting, not established probabilities of hallucination or specimen memorization.

The proposed near-zero base cut-out response was **not observed**. Inpaint artifacts,
contextual/amodal completion, neighbouring-mask damage, or presence bias may contribute.
All these adapters include the objectness prior w10; the experiment does not isolate
that prior causally. A matched prior ablation and image-edit controls are still needed.

### Coverage limits that constrain the conclusion

- Butterfly samples every eighth global pair. Coupled with cut channel `index % C`,
  it removes **only IDs 1,3,5,7,9**, counts 73/18/19/16/13. No even ID is tested.
  These are group-specific classes, not five shared anatomical labels. Future selection
  must balance traits independently of stride; do not generalize this read to all wings.
- Group counts are 22 backcross, 33 cyrbia, 27 lativitta, 36 malleti, 21 notabilis×lativitta.
  Crown cut-out deltas are respectively +19.71, +24.64, +1.04, +4.65, +36.79 pp.
  The 15.93-pp pooled result is pair-weighted, not a group-balanced test score.
- The removed channel is the only nonempty recall-region denominator, so row recall
  already isolates it. By contrast the label judge averages zero-evidence unchanged
  channels with the removed one; a row mean can hide a strong individual trait signal.
- Bootstrap intervals are exploratory, unadjusted across many comparisons, and use a
  fixed trained adapter. Shared query images and training data limit independence.

## Rigid transforms and the earlier interpretation

Beetle CPU subset (40 pairs) gave mostly negative label/base-referenced scores and small
recall differences. The larger read does not justify saying every observation is
'acquitted': even base flipH/ref has 2/120 positive row-mean label scores. Butterfly
flipH/chain has **base recall 50.32%, track 0%, IoU(Ty0) 0%**, positive label score on all
139 rows. Its rot90/chain base track is 3.54%, IoU 3.15%; ref is a different condition.
These base failures disqualify a simple 'positive judge = fine-tuning memorization'
interpretation. Symmetry/side correspondence is a plausible explanation, not established
by these numbers, and the transformed target does not become y0 just because the model
prefers it. Keep label semantics explicit and inspect identity/alignment/appearance.

Earlier statements that a base model is 'honest by construction', that all recall could
only come from a memorizer, or that these scores proved where a hidden code lives were
too strong. The continuous/binary closing gap is a representation-sensitivity diagnostic.
Neither a negative judge nor an insignificant delta proves absence of a learned shortcut.

## Consequence for the S18b penalty experiment

The new cut-out read gives a concrete target for **further calibration**; it does not
validate blanket inversion punishment. A penalty on any resemblance to y0 could penalize
correct shared shape or contextual completion. Translation/warp provide useful controls
where base tracks well; butterfly rigid failures and the collapsed adapter demonstrate
why a two-template score also needs an invalid/weak-tracker outcome.

Before defining f(x0,x1) or turning judges into loss weights, specify valid transforms,
train-only calibration examples, balanced trait sampling, matching weak/empty/known-copy
controls, reference/prior detachment, evidence normalization and outer loss weight.
Evaluate both preserving correct tracking and suppressing the undesired output. The
constants epsilon=.05, stride=16, disagreement=.1 and N_EFF=8 remain heuristics, and EM
on these scores is not a measured memorization prevalence. The full implementation
contract and unresolved choices are in [METHODS.md](METHODS.md).

## Historical CPU subset table (2026-09-19)

Preserved as the earlier 40-pair read, not a replacement for the full GPU report.
'LLR' names the heuristic score. Recall/track use regional denominators as defined above;
confidence intervals reflect the original subset and are not evidence of equivalence.

| arm | T | memory | recall % | track % | LLR/cell (label) | LLR/cell (base) | recall vs base (pp) |
| :-- | :-- | :-- | --: | --: | --: | --: | :-- |
| `base` | flipH | ref | 2.9 | 78.9 | -1.49 | — | — |
| `base` | flipH | chain | 2.8 | 81.1 | -1.51 | — | — |
| `base` | rot90 | ref | 1.5 | 71.4 | -1.54 | — | — |
| `base` | rot90 | chain | 1.3 | 70.3 | -1.51 | — | — |
| `crown` | flipH | ref | 2.9 | 79.1 | -1.49 | -1.65 | +0.01 [-0.04, +0.05] |
| `crown` | flipH | chain | 2.9 | 81.7 | -1.52 | -1.63 | +0.09 [-0.01, +0.22] |
| `crown` | rot90 | ref | 1.5 | 71.4 | -1.56 | -1.77 | +0.07 [-0.08, +0.23] |
| `crown` | rot90 | chain | 1.2 | 69.8 | -1.51 | -1.74 | -0.02 [-0.27, +0.21] |
| `crown_all` | flipH | ref | 2.8 | 79.2 | -1.49 | -1.65 | -0.02 [-0.11, +0.05] |
| `crown_all` | flipH | chain | 2.9 | 81.5 | -1.52 | -1.63 | +0.07 [-0.03, +0.21] |
| `crown_all` | rot90 | ref | 1.5 | 71.4 | -1.55 | -1.77 | +0.02 [-0.11, +0.13] |
| `crown_all` | rot90 | chain | 1.2 | 70.9 | -1.54 | -1.73 | -0.09 [-0.37, +0.16] |
| `crown_hot` | flipH | ref | 3.0 | 80.9 | -1.56 | -1.61 | +0.16 [-0.05, +0.41] |
| `crown_hot` | flipH | chain | 2.8 | 81.2 | -1.55 | -1.59 | -0.03 [-0.45, +0.25] |
| `crown_hot` | rot90 | ref | 2.3 | 67.7 | -1.52 | -1.69 | +0.80 [-0.04, +1.99] |
| `crown_hot` | rot90 | chain | 1.6 | 69.3 | -1.52 | -1.61 | +0.31 [-0.29, +1.14] |
| `crown_stemem` | flipH | ref | 2.9 | 79.0 | -1.49 | -1.65 | +0.03 [-0.01, +0.06] |
| `crown_stemem` | flipH | chain | 2.8 | 81.8 | -1.52 | -1.63 | +0.03 [-0.03, +0.09] |
| `crown_stemem` | rot90 | ref | 1.5 | 71.8 | -1.56 | -1.77 | +0.03 [-0.04, +0.09] |
| `crown_stemem` | rot90 | chain | 1.2 | 70.7 | -1.51 | -1.73 | -0.04 [-0.25, +0.17] |
| `crown_stemem_hot` | flipH | ref | 3.1 | 81.0 | -1.52 | -1.61 | +0.20 [-0.02, +0.41] |
| `crown_stemem_hot` | flipH | chain | 2.9 | 82.8 | -1.55 | -1.60 | +0.07 [-0.29, +0.36] |
| `crown_stemem_hot` | rot90 | ref | 1.5 | 72.6 | -1.57 | -1.75 | -0.02 [-0.14, +0.10] |
| `crown_stemem_hot` | rot90 | chain | 1.4 | 73.5 | -1.59 | -1.61 | +0.16 [-0.18, +0.46] |
| `crown_all_perturb` | flipH | ref | 2.8 | 79.6 | -1.51 | -1.64 | -0.01 [-0.10, +0.07] |
| `crown_all_perturb` | flipH | chain | 2.9 | 81.2 | -1.52 | -1.63 | +0.07 [-0.03, +0.21] |
| `crown_all_perturb` | rot90 | ref | 1.5 | 71.4 | -1.55 | -1.77 | +0.03 [-0.11, +0.15] |
| `crown_all_perturb` | rot90 | chain | 1.1 | 70.8 | -1.54 | -1.74 | -0.15 [-0.39, +0.04] |

The implementation choices resolving these design questions are in [PLAN.md](PLAN.md);
they are prespecified heuristics for an experiment, not conclusions validated by S18a.
