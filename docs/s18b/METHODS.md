# S18 / S18a / S18b methods and heuristics

Reviewed against source and landed artifacts on 2026-09-22. This is the shared technical
reference for people and subagents; follow the local `AGENTS.md` instructions when extending it.
See [calibration](CALIBRATION.md) for interpretation and [S18a results](S18A_RESULTS.md)
for source-derived numbers, hashes and the actual gate failure.

## Scope and implementation status

S18 trains eight SAM 2 adapters per dataset. S18a reuses those weights and completes
training-pair diagnostics, test evaluations and centre-frame calibration. These are
1-shot beetle and Butterfly Major experiments. On 2026-09-22 the full S18b implementation
was added separately from historical cycle.py: the original anchor plus an auxiliary
`x0 -> x1 -> x** -> x1' -> x0'`, fixed train/validation split, frozen teacher admission,
cut-out/warp targets and optional two-judge ablation. Read [PLAN.md](PLAN.md) for the
normative equations, all constants, arm controls, gradient/memory boundaries, sampling,
selection, failure conditions, artifact/command contract and actual validation.

Status: landed 2026-09-23 (PLAN.md "Landed result"): both gates PASS; the counterfactual
centre target halves butterfly cut-out occupancy at a −0.14 pp ✱ TTA test cost versus
crown_all, beetle null, the odd return and judges add nothing — **no recipe change**.
Execution (2026-09-22): the stage runs as one dependency graph with seven planned arms and
a TTA-only sliced confirmation on the shared run bank; the retired arms and modes, the
scheduling rules and the sizing are in [PLAN.md](PLAN.md) "Pipelined execution".
The historical centre probe described below still ends at T(x0). Its old score is not the
new training objective; its argmax occupancy is not S18b's soft independent-channel read.
Inversion means an explicit spatial/content counterfactual, not complementing every mask.
The operational weight uses y0, y**, frozen base predictions and pair context; it is not
an image-only similarity function or a measured probability of honesty.

The usual four role-labelled steps are `x0 -> x1 -> x1' -> x0'`: `x1'` is reverse
initialization on the same query image with its own predicted prompt. Resetting memory
at this transition removes the earlier tracker state, but still passes that prompt and
retains the learned weights. It neither proves nor rules out a learned shortcut.

## Training recipe, arms and gradient paths

The user-established recipe is `scripts/run_finetune.py --recipe sam2_crown`, expanded
by `occl.registry.expand_recipe` before explicit flags (later explicit flags win).
It uses SAM 2, LoRA rank 32 / alpha 64 on memory_encoder, memory_attention and
sam_mask_decoder, learning rate 1e-5, one epoch, per-trait masks, dataset-wide cached-train
pair derangement, ImageNet normalization, reset reverse memory and logit-reparameterized
prompts. Images and masks are 1024 square. The closing objective uses `excl-mode zerocat`;
forward objectness BCE targets presence for every prompted trait, weight 10. LoRA is
active on the forward pass and disabled on both reverse legs; the reverse computation
still carries gradients to the forward prompt. Disabling LoRA is not detaching that path.

| Arm | Change from crown | Consequence to distinguish |
| :-- | :-- | :-- |
| `crown` | none | User default/control, not universally best |
| `crown_ste` | `prompt-mode ste_threshold` | Binary value per channel (`logit > 0`), reparameterization surrogate derivative; not an argmax map |
| `crown_stemem` | `prompt-mode ste_memory` | Same binary forward value, surrogate gradient at the memory-encoder input on reverse initialization |
| `crown_perturb` | closing augmentation up to ±10 degrees | Rotate closing image and target together; opening reference unchanged |
| `crown_all` | `lora-legs all` | LoRA active on reverse initialization and closing too |
| `crown_all_perturb` | both previous changes | Tests perturbation with reverse-leg adaptation |
| `crown_hot` | lr 3e-5, three epochs | Changes both step size and exposure; not an isolated LR test |
| `crown_stemem_hot` | memory STE plus hot settings | Dataset-dependent stability; diverged on butterfly |

Source: `finetune/cycle.py`, `tracker.py`, `train.py`, `pairs.py`, and recipe fields in
training bundles. The continuous prompt is `(logit + 10) / 20`. Threshold STE keeps a
surrogate derivative of 1/20. The downstream sigmoid of a binary ±10 mask input can
suppress the memory-route gradient. Memory STE inserts a sigmoid(logit)-based surrogate
at the memory encoder (see `tracker.memory_ste`):
`masks.detach() + (scale*sigmoid(logits) - (scale*sigmoid(logits)).detach())`,
where `scale` is `model.sigmoid_scale_for_mem_enc`. This preserves the binary-path forward
value. These are deliberately biased surrogate gradients, not differentiation through a
true hard threshold. The objectness prior and hard no-object sentinel are separate.

CyclePairs uses only reference labels during fitting: each cached training image is a
reference once and query once per deranged epoch. The query's GT is accessed afterward
for diagnostics, never to set a fitting weight. Replaying epoch-zero global indices
preserves pair/augmentation seeds. Probe step-zero loss is checked against training under
each arm's own prompt and objective; a matching count alone is not replay proof.

Closing rotations sample uniform(-10, +10) with seed
`derive_seed(dataset, "closeaug|ref|query", epoch, global_index)`. The operation uses
OpenCV rotation about `(S/2-0.5, S/2-0.5)`, linear RGB interpolation, nearest label
interpolation and raw zero border fill. If any previously present trait loses all its
pixels, it falls back to the unrotated pair and records drawn/effective angle and fallback.
This is distinct from the probe's exact rot90 and normalized-tensor translation fill.

## Continuous-versus-binary reconstruction diagnostic

For fixed weights and a matched pair/closing target, replay the closing path using the
forward logits as a continuous prompt and as a per-channel binary prompt. Record
`closing_mIoU_continuous - closing_mIoU_binary`, then its paired change from base.
Historical fields call this the `stego_gap`. It measures reconstruction sensitivity to
prompt representation. It does **not** by itself identify hidden messages, memorization,
or a causal mechanism: confidence calibration and prompt distribution also change.
Similarly, closing gains without forward gains establish a mismatch in objectives, not
uniquely a particular shortcut. A CI spanning zero is not an equivalence result.

## Centre probe: what memory and data it reads

`scripts/probe_centre_frame.py` builds dataset-wide epoch-zero CyclePairs, selects global
indices, and shares frozen image-encoder features across arms. Each arm gets fresh
reference-conditioned state; adapters are active on every propagated frame of this read,
even when that adapter was trained with forward-only LoRA. No optimizer or gradients run.

- `ref`: initialize `(x0,y0)`, then predict on `T(x0)` at frame index 1.
- `chain`: initialize `(x0,y0)`, predict `x1`, store that output in non-conditioning memory
  at index 1, then predict `T(x0)` at index 2 with `num_frames=3`.
- `none/ref`: unchanged reference as query; IoU(y0) is the identity control. The two
  difference regions are empty, so recall and track are undefined, not zero.

The current files contain all 120 beetle pairs and every eighth of 1107 butterfly pairs
(139), nine arms, five transforms × two conditions plus identity = 99 rows per pair.
Neither frames, channels nor repeated arms are independent new specimens.

## Transforms and their exact heuristics

Source: `finetune/centre_transforms.py` and `probe_centre_frame.centre_frame`.

| Transform | Implemented operation | Constants / caveats |
| :-- | :-- | :-- |
| flipH | Reverse last spatial axis of image and mask stack | Exact grid permutation. Correspondence/side-label ambiguity on butterflies must be checked, not explained away as known symmetry |
| rot90 | `torch.rot90(...,1,(-2,-1))` on both | Exact quarter turn; a low base tracking score prevents attributing failure specifically to fine-tuning |
| warp | Uniform random 4×4 displacement field, bicubic upsample, same sampling grid for image and masks | Coarse components uniform ±0.06 of side; normalized coordinates multiply displacement by 2. Seed `derive_seed(ds.name, "warp|"+ref, 0, global_index)`. Image bilinear/reflection padding, masks nearest/zero padding, `align_corners=True`. Bicubic overshoot is possible; 0.06 is a control-point range, not a displacement bound or guaranteed invertible deformation |
| cutout | Inpaint selected trait region; empty that target channel | Select channel `global_index % C`. Dilate mask using ellipse radius 9 (19×19 kernel), Telea radius 5, uint8 resized RGB then normalize. Other target channels unchanged; dilation can alter neighbouring labelled content |
| translate | Shift both tensors down and right | `round(0.15*S)` pixels (154 at 1024), crop beyond edges, zero-fill. On normalized image tensors zero means normalization mean, not raw black. Target is cropped too |

Warp amplitude was chosen using mask separation before this full read. Earlier mask-only
ranges used all 120 beetle references but only 40 butterfly references; those ranges are
not a full butterfly calibration. Transformation strength, padding, interpolation and
sampling are choices to disclose, not guarantees of a natural counterfactual.

**Cut-out sampling limitation:** every-eighth-pair selection interacts with `i % C`.
The landed butterfly read removes only trait IDs 1,3,5,7,9 (73,18,19,16,13 pairs);
even IDs are never removed. IDs also have group-specific meanings. Beetle coverage is
1..5 (23,24,24,24,25). Interpret only this sample; a future balanced read must choose
traits independently of pair stride and retain the selection manifest.

## Metric definitions and aggregation

Let `A_c` be the pixels assigned trait c by argmax with positive winning logit (otherwise
background), `Y_c` the original reference mask, and `Z_c=T(Y_c)` the transformed target.

```
R_c = |A_c intersect (Y_c minus Z_c)| / |Y_c minus Z_c|   # recall-region occupancy
Q_c = |A_c intersect (Z_c minus Y_c)| / |Z_c minus Y_c|   # track-region occupancy
Delta_c = 1 - IoU(Y_c, Z_c)
```

A zero region denominator is undefined and excluded from the row's channel macro mean.
Rows are averaged over selected pairs, not pixels and not equally over butterfly groups.
These are fractions of the respective **target difference regions**, not fractions of
prediction mass and not probabilities that the model memorized. R and Q need not sum to 1.
`found` instead asks whether any raw channel logit is positive, before competing channels
are resolved. Thus found rate alone says little about segmentation quality.

For cut-out only the removed channel has a nonempty recall region; the row's `recall`
already measures that channel's occupancy in its old location. Every track region is
empty, so track is undefined. Delta=1 for the removed trait, not the row's macro Delta
(which includes unchanged channels). IoU(Ty0) is macro over channels with nonempty union;
a false positive in the empty target gets zero, while empty/empty is undefined.
Per-trait strings are rounded to four decimals; use full precision row values for paired
statistics. Inpainting residuals, contextual completion, boundary errors or broader
foreground bias can all yield nonzero cut-out occupancy without memorizing a specimen.

## The two judges: implemented score and unvalidated interpretation

`finetune/judge.py` area-averages sigmoid logits and masks to stride 16. For each channel:

```
m = eps + (1-2*eps) * downsample(Y)
h = eps + (1-2*eps) * downsample(Z)                  # label reference
h = eps + (1-2*eps) * downsample(P_base.detach())    # alternative base reference
L_j = P_j*log(m_j/h_j) + (1-P_j)*log((1-m_j)/(1-h_j))
V = {j: abs(m_j-h_j) > 0.1}
L_cell = sum(L_j for j in V) / max(len(V), 1)
E = 8 * L_cell
q = sigmoid(E + logit(pi))
proposed loss = mean(softplus(E + logit(pi)))
```

The raw sum over all grid cells is also saved. `eps=0.05`, stride=16, disagreement=0.1,
`N_EFF=8`, prior initialization 0.5, EM iterations 20, prior clamp [0.001,0.999] are
heuristics. Eps is not an empirically estimated pixel error rate; 8 is not an estimated
effective sample size. The score is an expected Bernoulli log ratio for soft predictions,
not a calibrated likelihood for independent observed pixel labels. Spatial and batch
correlation remain. Base predictions are a reference, not ground truth or an honest
model by construction; base can share the same spatial bias as the adapter.

The probe **averages channels within each row before** fitting the prior and counting
positive evidence. `frac_convicted` is the fraction of rows whose mean E > 0 at pi=.5;
it is not a fraction of proven memorizing models or traits. Cut-out leaves unchanged
channels with zero label-reference evidence, diluting the row mean by channel count.
Stored per-trait LLRs are raw sums; per-trait per-cell scores are not saved. Do not
silently substitute one for the other.

`fit_prior_em` iterates pi = mean(sigmoid(E+logit(pi))). With these heuristic scores this
is a mixture-style update, not a measured prevalence of memorization. Before training,
reference fields/prior estimation must be detached or otherwise explicitly controlled
so the optimized model cannot move its judge to reduce the penalty. The two-template
model also needs an abstention/invalid or weak-tracker policy. With no voting cells E=0;
at pi=.5 softplus(0)=log(2), not zero. Softplus is unbounded and asymptotically linear on
the positive side. An outer objective weight, normalization and evidence scale remain
choices; this construction does not make penalty-strength selection disappear.

## Pairing, uncertainty and provenance

`registry.compare` checks protocol comparability and uses `per_seed_scores`: butterfly
groups average within each support-run index first, then 20 matched seed differences.
All displayed test CIs use the repo's 10,000-resample percentile bootstrap, RNG seed 0.
We additionally verify actual support IDs, group/run coverage and merged-cell values.
These CIs describe support-selection variation for the fixed trained adapter and fixed
test set; they do not cover training-seed variation or independent dataset sampling.

Centre differences use the same global pair/reference/transform/condition across arms.
There is one reference per selected pair, but derangement query images and training
weights are shared. The bundle's paired bootstrap treats selected pair differences as
exchangeable; its intervals are exploratory, not independence guarantees. Butterfly
centre means weight the sampled pairs (group counts 22/33/27/36/21), unlike test means.
Report group/trait coverage and both `ref` and `chain`; never count pixels as replicates.
No multiplicity correction was applied across arms/transforms/metrics. Stars only mean
that this unadjusted interval excludes zero. Gate PASS is a structural/declared-check
result, not proof of a mechanism; gate FAIL is retained even when other arms are useful.

## Reproduction and provenance limits

Run `.venv/bin/python scripts/report_s18a.py` for the detailed Markdown and JSON evidence.
It hashes 39 input archives, checks all 36 eval cells against the merged record, verifies
actual support draws and full run coverage, and checks centre coverage and adapter hashes.
The unchanged gate runs with `--no-file`, so a failed input is never filed as passing.
The report command succeeding means report generation succeeded, not that the gate passed.

Inputs under `results/inbox/` are ignored by git and must be transferred for another
machine/submodel to rerun this read. The evidence JSON records their exact paths and SHA256.
Bundle metadata does not establish a complete source-tree revision (`occl` is reported as
unknown in the merged gate header). Current source inspection plus matching artifacts
verifies the described implementation/measurements, not a perfect historical source
reconstruction. The report and independent row-mean checks do not rerun GPU inference.

## Historical requirements and their S18b disposition

These requirements were recorded before implementation. [PLAN.md](PLAN.md) now specifies
which are implemented/tested and which remain limitations; GPU benefit is still unmeasured:

1. Keep y0 and T(y0) as explicit templates; verify image/mask alignment, retained area,
   interpolation boundaries, occlusion semantics and trait-balanced sampling.
2. Use matched positive controls that deliberately emit y0 and controls emitting T(y0),
   plus weak/empty/wrong outputs. Test whether the detector distinguishes these at all.
3. Validate base behaviour and transform fidelity on train-only calibration data.
   Larger geometric Delta is insufficient if the base cannot perform the transformed task.
4. Keep existing held-out test results for reporting, not tuning epsilon, N_EFF or strength.
   Record a calibration split or future validation protocol and sensitivity analysis.
5. Isolate competing explanations: an objectness-prior ablation and inpaint/mask controls
   would test cut-out hypotheses. Current arms all carry w10, so they do not isolate it.
6. Specify which legs adapt, memory resets, how the centre label is used, all detached
   quantities, treatment of low evidence/collapse and outer loss weight. Gradient-check
   the whole proposed five-frame path before a GPU experiment.

The measured read supports pursuing cut-out calibration and warp as a contrast/control;
it does not yet license punishing mask similarity in every palindrome.
