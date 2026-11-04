# S18b: counterfactual centre, preserved tracking, and a complete odd palindrome

Design frozen for the first S18b experiment on 2026-09-22, **after** reviewing S18/S18a and
**before** any S18b GPU fit or test result. Status: implemented; synthetic wiring tests and
real SAM 2 CPU forward/backward smokes. This is not evidence of improved generalization.
The user-established SAM 2 recipe remains sam2_crown. No automatic promotion occurs.

This is the canonical implementation contract for humans and subagents. Read
the local `AGENTS.md` instructions, [historical methods](METHODS.md),
[calibration interpretation](CALIBRATION.md), [verified S18a results](S18A_RESULTS.md), and
[RUNBOOK.md](../../RUNBOOK.md). Older PDFs/decks describe earlier proposals.

## What S18 and S18a require us to change

- Memory reset removes tracker state, but the predicted prompt and learned weights still
  cross the cycle. Successful reconstruction alone does not establish semantic tracking.
- Butterfly cut-out gives a useful response to investigate: original-region occupancy
  rises from 25.17% under base to 41.10% under crown. The same effect with reference-only
  memory means it is not uniquely caused by the intervening query. Neither number is a
  memorization probability. Context, inpainting and object-presence bias remain alternatives.
- Butterfly warp improves occupancy on displaced pixels while also increasing occupancy
  on old pixels. We must reward following the new target and preserving correct regions,
  rather than rewarding small/empty masks.
- Translation already tracks well and retains little old-location occupancy. It is a
  preserving control, not the primary punishment. Butterfly flips/90-degree rotations fail
  even under base; their failure cannot serve as a valid fine-tuning memorization label.
- Butterfly all-leg adaptation helped, whereas beetle did not show the same ranking.
  Include a matched all-leg control. Retain S18's mild closing perturbation as a control;
  its gains were conditional, not a universal benefit.
- The hot memory-STE arm collapsed on butterfly. S18a remains **FAIL 25/26**. Do not reuse
  that arm, increase LR/epochs, change thresholds to make it pass, or silently remove it.
- Butterfly stride/channel aliasing omitted even trait IDs in the original cut-out read.
  S18b chooses cut channels independently of global pair stride and records coverage.

## The baseline expression for x**

For a labelled reference (x0,y0) and its unlabelled query x1, draw a predetermined edit A:

    x** = A_image(x0; y0, seed)
    y** = A_target(y0; seed)
    V   = pixels where the edited image and edited annotation can be compared

The annotation is needed to locate a cut-out. This is a **known counterfactual intervention
on the reference**, not a Boolean complement of y0, not a new unlabelled specimen, and not
an adversary optimized online to maximize the model's loss. It challenges persistence of
specific mask content where the intervention supplies evidence that the content should
change. Similarity elsewhere is legitimate.

The competing hypotheses are deliberately narrower than “memorizing” versus “honest”:

    H_copy: keep y0 in its original coordinates/content at x**.
    H_edit: follow the visible edited target y** at x**.

A result favouring H_copy is a response to this intervention; it does not uniquely identify
specimen memorization, hidden coding or the mechanism Chao described. Correct tracking
on an unchanged/common region must not be punished for resembling y0.

### The two training interventions

**Elastic warp.** Draw a 4×4 displacement grid, each coordinate uniform(-0.06,+0.06) image
sides, seed stored per reference. Bicubic upsample with align_corners=True; add twice the
displacement to the normalized [-1,1] sampling grid. This is smooth deformation, not
periodic wrapping. Bicubic interpolation can overshoot the coarse bounds. Sample RGB
bilinearly with reflection padding, masks by nearest neighbour with zero padding. Exclude
out-of-bounds sampling coordinates from V. If any prompted trait retains less than 75%
of its original area within V, use the unchanged reference/target as an identity fallback.
Its separation and counterfactual weight are zero. This protects the closing path from
being required to recover a trait clipped out of the centre image. The fallback is visible
as zero separation in the calibration record; these examples are not evidence of warp
success. The remaining deformation need not be a mathematically invertible diffeomorphism.

**Trait cut-out.** Choose one present trait; require at least two prompted channels. At
1024 resolution, dilate its binary region by a radius-9 ellipse (19×19), use OpenCV Telea
inpainting with radius 5, then normalize RGB again. Set only that trait's y** channel to
zero. The dilation fringe is excluded from V for every channel because it may damage
neighbours; the original trait core remains valid negative supervision. RGB is rounded
through uint8, another edit confound. The unchanged channels' targets stay unchanged.
This is visible-mask supervision: an amodal completion is counted as wrong for this
intervention even if it could be reasonable under a different annotation convention.

Within each group, alternate cut-out and warp in the fixed pair order; a one-channel
reference uses warp. Choose the least-used available cut trait within that group, breaking
ties by SHA256(seed,reference,trait). Seed is 18. Edit RNG uses derive_seed(dataset,
"s18b-edit|reference",0,18). Class IDs remain group-specific. The rule balances availability;
it does not promise equal counts when traits are absent, or guarantee every trait in a
small validation sample. Report the actual counts and abstain from claims about missing IDs.

**Translation audit.** Shift image and masks down/right by 15% of the image side with zero
fill in normalized tensor space; crop shifted-out content and exclude filled target pixels.
This is evaluated on every validation pair, never included in the centre training penalty.
No flip or 60/120/90-degree rotation enters the centre objective.

## The palindrome and information boundary

The five **image visits** are:

    x0 -> x1 -> x** -> x1' -> x0'

There are six tracker calls because the centre is also reinitialized for the reverse
pass. Counting image visits and counting init/propagation calls are different conventions.
The unedited anchor remains the established four-role cycle:

    init(x0,y0) -> predict(x1) -> RESET/init(x1,pred1) -> predict(x0)

The auxiliary odd cycle shares that original forward prefix. Store the x1 output as
non-conditioning memory; predict x** at frame index 2 (num_frames=3), using the original
x0 conditioning memory and the x1 output. Then **discard this state**. Initialize fresh
conditioning state on x** using its own prediction, propagate x1' at index 1, store that
output, and propagate x0' at index 2. The reverse state cannot retrieve the original
labelled memory. The continuous prompt remains (logit+10)/20; no new STE/sentinel rule.
The prefix uses the existing two-frame calls; the extended legs explicitly use three frames.

For forward-only arms, LoRA is enabled on the original forward pass **and on the centre
prediction**, and disabled on reverse init/propagation in both cycles. Disabling adapters
preserves the gradient through the reverse computation to the predicted prompt. The all-leg
arm keeps adapters enabled everywhere. The image encoder is frozen and evaluated without
gradient; its features may enter the differentiable tracker. No query GT argument exists
in the training-step API.

**Irreversible edit safeguard:** after cutting out a trait and resetting at x**, the model
is not required to reconstruct that removed channel at x0'. The odd closing loss uses
only recoverable channels. The original anchor still reconstructs all original channels.
Otherwise correct suppression at the centre and demanded recovery after reset could reward
the very mask-carrying behaviour we intended to discourage. On warp, all channels remain
recoverable; the area guard above rejects severe clipping.

## How strong is the punishment?

Use an operational, frozen function of the pair, reference labels and intervention:

    s_c = 1 - IoU(y0_c, y**_c) on V
    q_c = IoU(hard frozen-base centre prediction_c, y**_c)       [warp]
    q_c = 1 - mean(sigmoid(base centre logits_c) on erased core) [cut channel]
    a_c = 1[q_c >= .50 and s_c >= .05 and retained-area-valid]
    w_c = stop_gradient(a_c * q_c * s_c)

Retained-area validity requires at least 75% for surviving channels. Cut channels are
intentionally empty and exempt. Base predicts the centre from the actual x0→x1→x** chain,
so x1 influences q, not just a global image similarity. This is **not** the originally
suggested image-only f(x0,x1), and q is not a calibrated probability of honesty. It is a
conservative teacher/readability heuristic. Base failure causes abstention, not a larger
punishment. Identical targets have s=0 and receive zero centre/judge penalty. The base
weights, edit, q and w are fixed before any adapter fit. The learner cannot reduce its own
penalty by degrading its own confidence or changing the weighting reference.

For each channel define four disjoint valid regions: removed (y0 minus y**), added
(y** minus y0), common foreground (intersection), and common background (neither). Average
binary cross-entropy with logits inside each nonempty region, then average those region
means. Add an unweighted-by-area object-score BCE whose target is whether y** contains the
trait. Multiply by w_c and average over the **original C channels**, including zero-weight
channels. Empty regions do not vote. This limits domination by the large background and
penalizes both old-region persistence and failure to occupy new regions. The sum includes
one presence BCE per admitted channel, coefficient 1 (not 10).

For cut-out, unchanged channels have separation zero and hence no direct centre loss.
They are protected by the original anchor, the recoverable-channel odd loss, and ordinary
validation. Do not claim the centre term alone supervises all neighbouring channels.

The total for the main odd_cf arm is:

    L = L_anchor + 0.25 L_odd_closing + 0.25 mean_c[w_c L_centre,c]
    L_anchor = CE_zerocat(closing,y0) + Dice(closing,y0)
               + 10 BCE(forward object score, prompted-present)

L_odd_closing uses CE_zerocat + Dice, restricted as above. Auxiliary weights .25/.25 are
pre-experiment scale choices, not optimized values. Continuous prompts can still carry low-amplitude information while satisfying the
centre target; the design does not eliminate every possible encoding shortcut. The original
anchor, original forward presence term and odd reconstruction can still reward undesirable shortcuts: the design
is an experiment to measure that, not a proof the shortcut has become impossible.

### The optional two-judge term

One judge compares the current centre probabilities to softened copy and edited-label
hypotheses. The second compares copy to the frozen base's centre probabilities. They share
evidence and are **not independent Bayesian experts**. Base is not honest by construction.

Average 16×16 pixel cells. Templates are softened as .05 + .90 p. Vote only on fully valid
cells where the two templates differ by more than .1. For each reference h and current
probability p, the mean voting-cell expected Bernoulli log ratio is:

    E_h = mean[p log(m/h) + (1-p) log((1-m)/(1-h))]

Compute separately for edited-label h and frozen-base h. Per channel use their minimum,
so both must favour copy for the combined score to be positive. If either reference has
no voting cells, abstain. A fixed equal prior is implicit; there is no EM prevalence fit.
The optional loss is mean_c[w_c softplus(8 min(E_label,E_base))/8], coefficient .10 in the
total loss. Negative scores still have a small positive softplus penalty; do not describe
this as exactly zero below a decision threshold. N_eff=8 limits the scale, and epsilon,
stride, disagreement and prior are heuristics. Template/base/weight tensors are detached;
current predictions retain gradient. This term is a separate arm so that its incremental
value over the counterfactual objective can be measured.

## The fits per dataset and their controls

All start from the same pretrained SAM 2 weights and deterministically initialized LoRA;
none starts from an already trained S18 adapter. Stable recipe: rank32/alpha64, memory
encoder + memory attention + mask decoder, lr1e-5, one epoch, AdamW weight decay .01,
cosine schedule to .01×initial LR, norm clip1, evaluation model mode, per-trait/zerocat,
ImageNet 1024 images/masks, continuous prompts and reset memory.

| Arm | What it tests |
| :-- | :-- |
| crown | Original cycle, forward-only adapters, forward presence weight10 |
| crown_all | Original cycle with all-leg adaptation; matched all-leg control |
| crown_noobj | Original cycle, forward-only, no forward presence prior |
| crown_perturb | Original cycle; closing image and target rotated together uniformly ±10° |
| centre_aux | Anchor plus weighted centre target; no odd return, isolates extra supervision |
| odd_cycle | Anchor plus complete odd return; no centre target/judge loss |
| odd_cf | Anchor + odd return + calibrated counterfactual centre loss |
| odd_cf_all | Same as odd_cf, adapters active on all legs |
| odd_cf_noobj | Same as odd_cf, remove forward presence prior; keep centre's correct-presence term |
| odd_cf_judge | Same as odd_cf, add the two-judge term |

**Planned versus optional (2026-09-22, "Pipelined execution" below).** The planned screen
is the seven arms `crown, crown_all, centre_aux, odd_cycle, odd_cf, odd_cf_all, odd_cf_judge`
(`s18b_contract.PLANNED_ARMS`; the evidence gate requires exactly these). `crown_perturb`
(S18a: closing rotation null under sam2_crown on both datasets), `crown_noobj` (the presence
prior is part of the established recipe, S17/S17d) and `odd_cf_noobj` (a secondary
interaction, deferred with its control) are OPTIONAL: implemented, selectable with the
driver's `--arms`, outside the gate, each with its retiring result recorded in
`OPTIONAL_ARMS`. Closing perturbation uses the existing CyclePairs.closing ±10° rule,
pair/epoch/index seed and missing-trait fallback; it affects the anchor's closing target
only. No hot, STE, soft-sentinel or 5-shot fan is included. These are explicitly documented deviations from
sam2_crown; metadata records both the established recipe and every S18b deviation.
The registry treats nonempty occl.s18b metadata as modified, regardless of the recipe
label, including the matched crown fit on a restricted training subset.

Primary comparisons: odd_cf vs crown (whole intervention), odd_cf vs centre_aux (odd
return beyond target supervision), odd_cf vs odd_cycle (explicit centre target),
odd_cf_judge vs odd_cf (judge), odd_cf_all vs crown_all and odd_cf_noobj vs crown_noobj
(matched totals). Prior/leg interactions additionally compare each treatment-control delta.
The screen is exploratory and permits a null/no-candidate outcome.

## Splits, coverage and selection

Use only cached official training images. Split each group's sorted IDs by seeded
permutation using derive_seed(dataset,"s18b-split|group",0,18), holding out round(20%) with
minimum 2; require at least 6 cached images per group. Independently derange each partition
with the existing epoch-zero CyclePairs logic. Fit and validation share **no image IDs** in
either reference/query role. One labelled reference per episode; this is 1-shot evaluation,
not a claim the entire fine-tuning dataset contains one annotation.

On the local cache snapshot: beetle 96 fit / 24 validation; butterfly 886 fit / 221
validation (1107 cached of 4227 training images). Counts can differ on the cluster cache;
the manifest binds all IDs, groups, order, cut traits and edit seeds. It hashes actual
cached image bytes and reference label-map shapes/int32 bytes. Recompute before every
worker, refusing changed input coverage or content. Held-out labels may be used for
calibration diagnostics/selection, never optimizer updates. Original test labels are read
only in the separate confirmation phase; original test results do not choose the arm.

On every held-out pair, measure ordinary 1-shot prediction against query GT, edited centre
behaviour, and translation. Ordinary mIoU uses the existing GT-native argmax metric and
macro-averages group means; centre metrics use **independent per-channel sigmoid regional
occupancy** and hard per-channel IoU. These differ from S18a's argmax regional occupancy:
do not compare their percentages as if they were identical. Record per-channel denominators,
separation, retained area, class IDs and per-pair values. Cut-out mean is pair-weighted over
cut-out rows; unchanged channels with no removed pixels do not enter its denominator.

Every validation pair also runs the complete unaugmented odd return under that arm's
forward-only/all-leg setting. Record independent per-channel closing IoU over recoverable
channels, loss components, both centre judge scores and frozen admission weights. Use the
same diagnostic odd read for anchor/centre-only controls even though they were not fitted
with that return objective. These are diagnostic losses, not their training objectives.
The edited-centre prediction is reused from that read; the query GT never enters it.

On the first **8 validation pairs per group in the fixed manifest order**, replay the
centre-to-x1-to-x0 return with a binary (>0) centre prompt, fresh reset, and the same
reverse-leg adapter scope. Record continuous closing IoU minus binary closing IoU, on the
same recoverable channels. At most 8 beetle and 40 butterfly pairs per arm. This is a
bounded, matched representation-sensitivity audit; it does not prove hidden coding.
The first-eight rule is fixed before S18b outcomes, independent of scores, and does not
ensure all traits appear in this small audit. It does not select the candidate. Diagnostic
closing means are pair-weighted; group/sample coverage travels with each row.

Prespecified candidate eligibility relative to its matched crown control:

- ordinary validation mIoU no more than 0.5 percentage point lower;
- each group's ordinary mIoU no more than 2 points lower;
- mean translation IoU no more than 2 points lower;
- cut-out original-region soft occupancy strictly lower.

These are **point-estimate screening floors**, not statistical noninferiority tests. Rank
eligible candidates by ordinary validation group-macro mIoU, then lower cut-out occupancy,
then lexical arm name. Freeze one winner per dataset. If none qualifies, record no candidate
and confirm only base/crown. All-leg/no-prior arms match crown_all/crown_noobj; other
candidates match crown. Do not select from test scores or promote a recipe automatically.

The screen gate requires every planned arm's complete finite evidence and no adapter or
group falling below half its matched dataset/group base validation mIoU (gross-collapse
heuristic). Failed arms remain in operational records and make the full screen gate fail;
selection may still describe surviving candidates, explicitly with incomplete evidence.
There is no requirement that all treatments improve. Never relax S18a's historical gate.

## Bounded confirmation and uncertainty

Confirm the frozen winner, its matched control, and pretrained base: at most 3 arms ×
**one inference mode, orientation TTA** × 20 support seeds per dataset, shot=1. Plain
inference was dropped on 2026-09-22 ("Pipelined execution"): TTA scored above plain on every
landed SAM 2 cell of both datasets (S14–S18a), so the plain read answers nothing new;
`selection.confirm_modes` records the modes and the gate checks exactly those cells.
Use canonical_balanced beetle and canonical_placement butterfly support selection, the
existing test sets and standard predictor. No centre image or judge is injected at test
time. Fitted weights stay from the 80% fitting subset: no unreported refit on validation.
At most 360 group×seed evaluation runs across both datasets (TTA only); if no candidate
qualifies, fewer. Base runs are reused from the stage run bank when S18a banked them under
the same fingerprint (`evaluator.run_fingerprint`: weights, protocol, cache and source);
otherwise they are recomputed inside the same slices. This cap does not imply these
finish within one allocation.

Verify actual (group,seed,support IDs), ordinary protocol and model metadata before pairing.
For test comparisons use registry.compare, 10,000 paired support-seed bootstrap resamples,
seed0, group mean inside each seed. Report ordinary mIoU and CIs; boundary/assignment
variants are secondary. The primary confirmation is each selected candidate versus its
matched control under **TTA** (the only mode run); dataset effects are separate. Intervals are
unadjusted across the two datasets/secondary comparisons, exploratory, conditional on one
trained adapter; they do not measure variability across independent training seeds.
Validation pairs also share reference/query identities across roles, so they are not
independent retrained models. Test results can disconfirm a validation ranking.

Lower cut-out occupancy with worse ordinary tracking, damaged unchanged traits, worse
translation, or empty outputs fails the scientific purpose. Equal performance of centre_aux
and odd_cf would favour extra labelled-reference supervision over a special palindrome
mechanism. Equal odd_cf and judge-arm performance would not support the extra judge. Wins
only without objectness would implicate the presence prior. None alone proves memorization.
Inpaint/sham/background-edit controls, other amplitudes, new annotation conventions,
independent training replicates, second supports and SAM3 remain later work; they are not
silently claimed as validated here.

## Execution, resumability and artifact contract

Entries: finetune/s18b.py (step/edit/weight), s18b_contract.py (selection/gate),
scripts/run_s18b.py (plan/calibrate/train/validate/select), run_s18b_stage.py (driver), and
notebooks/run_s18b_scc.py/.ipynb. BUILD20. Historical cycle.py remains unchanged.

The coordinator reads actual visible GPU name and total memory. Exactly one process per
GPU, no 48GB assumption, MPS or memory oversubscription. Since 2026-09-22 the stage is one
dependency graph rather than barrier phases (see "Pipelined execution"); the default
invocation is `all`. Each invocation has a wall budget (`--minutes`, set from the
allocation by the submit helper; 110 if unset), per-worker log/progress stall limit1200
seconds and a status update every30s.
These are operational heuristics, not estimates of scientific runtime. Stop/cancel terminates
only owned child process groups (TERM, then KILL after10s). There is no blind retry.

Cache staging reuses the existing node-local PNG cache plus optional delta. Worker identity
binds manifest, all occl source files, workers/evaluator, vendored SAM2 source, full checkpoint
SHA256, torch version and fp32CPU/bf16CUDA mode. Different GPU models with the same dtype
may have small numerical differences; identity does not promise hardware-bitwise equality.
Do not edit or push source while workers run. A lock prevents duplicate drivers using one
output root. A stale heartbeat is lost monitoring evidence; its compute host/allocation
must be checked, not a login-node PID.

- Calibration: atomic per-pair weights, frozen 64×64 half-precision base probabilities,
  per-channel statistics and identity, plus completion receipts.
- Fit: atomic optimizer/scheduler/LoRA/history checkpoint every8 steps, including the next
  pair index. Resume exactly that order. At most7 completed updates may need repeating after
  interruption. Model mode is eval; edit/LoRA draws are deterministic. Completed weights
  carry standard strict-load metadata and SHA256. No fit is complete after a capped smoke.
- Stop on nonfinite loss/gradient. From step32, also stop if current loss exceeds max(100,
  20×median of the first16 losses). This is a coarse runaway alarm, not a calibrated collapse
  detector; validation checks remain necessary. Save the last good checkpoint.
- Validation: atomic per-pair JSON, then a completion summary. Confirmation uses the standard
  fingerprinted per-run evaluator cache and hashed completion receipts.
- Bundle: results/s18b.zip contains records, manifests, all completed adapters and selected
  test bundles; unfinished/failed arms cannot satisfy the screen gate. The workspace retains
  larger optimizer/calibration caches for resume. No existing stage is wiped.

## Pipelined execution (2026-09-22)

Direction: S18b spends GPU time only on what the landed stages have not answered.
Fine-tuning versus no fine-tuning (S17–S18a, both datasets), TTA versus plain (TTA above
plain on every landed cell), the closing perturbation (null) and the presence prior
(established) are not re-measured. What remains novel is the odd return, the
counterfactual centre, the judge term, the extra-supervision control and the all-leg
interaction — the seven planned arms — and the validation-selected TTA confirmation.

Heuristics implemented in `scripts/run_s18b_stage.py` (`Runner`, `screen_graph`,
`confirm_graph`) and `scripts/run_s18b.py` (`assemble`, `gate`):

- **One dependency graph, no phase barriers.** Every job carries `deps` (must exit 0),
  optionally `after` (must merely end) and is launched in list order among the eligible
  ones on the first free card; list order is priority, so the notebook passes
  `butterfly_major` first (the critical path). A failed job SKIPs everything downstream
  (rc −999) instead of blocking or retrying. `then` lets a finished job append jobs to the
  running graph.
- **Per dataset:** `plan` → calibration shards (`--shards`, default beetle 1, butterfly 2)
  → `assemble` (verifies every shard receipt and row, writes `calibration.json` with the
  admission statistics). Anchor fits (`crown`, `crown_all`) depend only on the plan and
  start at once; the other fits on `assemble`. Each arm's validation depends on its own
  fit and on `assemble`; `base` validation on `assemble`. `select` waits for `base` and
  `crown` to succeed and for every other validation to END, so failed arms are recorded
  as missing rather than blocking selection. `gate` runs the screen evidence contract for
  that dataset from the work root (adapter hashes, calibration rows, `evidence_check`).
- **Confirmation joins the graph the moment a dataset's gate passes** (`then` on the gate
  in phase `all`), while the other dataset is still screening. It is the S17/S18a fan:
  one `run_arms.py` slice per (group, run range) — 20 runs in `--slices` (5) slices of 4,
  every confirmed arm scored on ONE image encode, per-run JSON banked in the shared
  `--runs-dir` (the stage bank `results/runs`, keyed by the evaluator fingerprint that
  includes the adapter weights) — then one inference-free `run_eval.py --no-precompute`
  merge per cell, receipted by identity and hash. Banked runs are reused on resume; base
  runs banked by S18a are reused when their fingerprint still matches.
- **Sizing (S18/S18a timings on 4 × L40S, estimates):** butterfly calibration ≈ 10 min,
  7 fits ≈ 30–50 min each, 8 validations ≈ 10–20 min each, 25 TTA fan slices ≈ 10–20 min
  each, 3 merges; beetle's whole screen and 5 slices fill the gaps. The whole stage is
  sized for one 8 h allocation (`scc_submit_s18b.sh all 8`; `all 8 2` chains a resume
  allocation as insurance). Unsliced, the old design needed ≈ 40 h of wall.

Batch submission: `scripts/scc_s18b.qsub` + `scripts/scc_submit_s18b.sh <phase> <hours>
[chain]` submit the same headless notebook as an SGE job (4 GPUs, cc ≥ 8.0, ≥ 40 GB,
16 cores, driver budget = h_rt − 25 min, `-hold_jid` chains for multi-allocation resume).
SCC conventions (verified 2026-09-23, RUNBOOK "SCC rules"): `-l gpus` is a count; 12 h is
the default and the shared-GPU ceiling per job; the driver pins each worker to one of the
queue-granted `CUDA_VISIBLE_DEVICES` ids (never a torch ordinal) with `NSLOTS / #GPUs`
threads. A wrong-GPU pin is a reaper kill, not a slowdown.

Use the single-line commands in RUNBOOK. Screen gate: check_gate.py s18b_screen with an
explicit bundle and --no-file. Full gate: check_gate.py s18b. The former is **not** a test
win. S18b success never changes the recorded S18a failure.
The coordinator checks the same evidence contract before reporting screen completion,
before launching confirmation, and before reporting full completion. A failed check keeps
the bundle for diagnosis, marks STATUS incomplete and exits nonzero; no test compute follows
a failed screen, including in the combined `all` phase.

## Landed result (2026-09-23)

One `all 8` batch job (job graph of 77 jobs, 4 × L40S) finished in **1.46 h of wall**, all
jobs complete, both gates PASS ([s18b_gate.txt](s18b_gate.txt)); the evidence read is
[S18B_RESULTS.md](S18B_RESULTS.md) (generated by `report_s18b.py`, selection read, never
changed). Measured job times: butterfly fits 5.8–13.0 min (odd_cf_all longest), beetle
fits ≤ 2 min, validations ≤ 3.9 min, calibration shards ≤ 2.6 min, TTA fan slices 7.5 min
mean / 10.3 max (30 slices), merges ≤ 2 min. The earlier 30–50 min per-fit estimate was
wrong by 4–5×; the stage fits a 3 h allocation with margin.

Provenance checks made before reading the numbers: the S18b butterfly `crown_all` adapter
(‖ΔW‖ 7.378, 886 pairs) differs from S18a's (8.100, 1107 pairs) and 0 of its 100 test runs
equal S18a's, so its 77.16 test mean matching S18a's 77.16 to two decimals is a coincidence
(the CIs differ). The `base_tta` cells are 100 of 100 runs identical to S18a's — the
intended reuse from the run bank.

**Selection.** Both datasets select `odd_cf_all` (matched control `crown_all`).

**Butterfly (validation, 221 held-out pairs).** The counterfactual centre target roughly
halves cut-out soft occupancy of the removed trait: base 25.7 %, crown 34.9 %, crown_all
31.5 % versus centre_aux 18.0 %, odd_cf 17.6 %, odd_cf_all 19.0 %, odd_cf_judge 17.6 %,
with ordinary validation mIoU and translation preserved (odd_cf_all 81.43 / 92.00 vs
crown_all 81.28 / 92.17). `odd_cycle` (odd return, no centre target) sits at 34.9 %, equal
to crown: **the odd return alone does nothing; the explicit centre target does the work.**
`centre_aux` ≈ `odd_cf` ≈ `odd_cf_judge` on every read: the pre-registered reading applies —
this favours extra labelled-reference supervision over a special palindrome mechanism, and
the judge term adds nothing. The continuous-minus-binary return gap is −0.04…−0.49 pp on
every arm (no representation-sensitivity signal, as in S18a).

**Butterfly (test, TTA, 20 paired seeds).** crown_all 77.16 (+1.98 [+1.77, +2.21] vs base,
replicating S18a on the 80 % subset); odd_cf_all 77.02, **−0.14 [−0.22, −0.04] vs crown_all**:
a small, significant ordinary-tracking cost. By the contract, lower cut-out occupancy with
worse ordinary tracking does not meet the scientific purpose; the hallucination reduction
is a real, validated robustness property, not a test-set improvement.

**Beetle.** Validation is flat (52.05–52.32 across every fitted arm; cut-out 30.7–31.0 %
vs base 28.2 %, i.e. every fit raises removed-trait occupancy slightly and no arm lowers
it below base); `odd_cf_all` wins by the rule on a 0.05-point margin. Test: crown_all 64.94
(+1.08 [+0.50, +1.73] vs base); odd_cf_all 64.86, −0.08 [−0.21, +0.05] vs crown_all (ns).
No effect either way.

**Disposition.** No recipe change: `sam2_crown` stands ([sam2-finetune-recipe]); the
all-leg control replicates as the better butterfly recipe (+1.98 ✱), as in S18a. The
counterfactual centre target is a documented, measured regulariser against reproducing an
erased reference trait (butterfly −13 pp occupancy at −0.14 pp test mIoU) that does not
need the odd return or the judges; if that robustness is wanted, `centre_aux` is the
cheapest arm that delivers it. The inversion-penaliser / odd-palindrome mechanism as a
route to a better fine-tune is answered negative on both datasets under this contract.
Caveats as pre-registered: one trained adapter per arm, no independent training
replicates, intervals unadjusted across the two datasets, validation shares image
identities across roles.

## Verification and limits of the current implementation

Synthetic tests cover old-anchor value/gradient parity, fresh reverse memory, actual odd
frame ordering, detached teacher/judge weights, empty-difference abstention, known-copy
versus edited-target loss ordering, erased-channel exclusion, channel balance/split leakage,
selection/gate tampering, optimizer resume and child termination/status.

Real SAM2 CPU smokes run the odd_cf_judge forward and backward on two prompted channels,
using dataset training pair0, outside the experimental fit/validation manifest. Beetle uses
warp; butterfly uses cut-out. The checkpoint is sam2_hiera_large; results, source hash,
finite gradient, loss components, memory and time are in [beetle](smoke_beetle_warp.json)
and [butterfly](smoke_butterfly_cutout.json). Reproduce using scripts/smoke_s18b.py with
--dataset, --transform and --out; CPU threads4, address-space cap36GiB. This proves the
code path/backward works for those examples, **not** full-channel GPU memory requirements,
optimizer convergence, calibrated probabilities, or scientific benefit. Full CUDA fits,
validation and test confirmation landed 2026-09-23 ("Landed result"); independent
retraining remains unrun.

CPU smoke JSONs retain their original source fingerprint. Later changes cover validation
audits, recipe classification and orchestration/gates; the numerical training step is
unchanged. Final worker source fingerprint:
`a488d2caa85af6eb205c3dde3ad8dfbf71144dbe2e402ba7e479be356475b95c`.
Full suite: 373 passed (seven dependency/test warnings); focused checks: 106 passed
(BUILD 20). The 2026-09-22 pipelining changed orchestration, the planned arm set and the
confirmation modes only — not the numerical training step, calibration or validation —
and is covered by `tests/test_s18b.py` (graph ordering, skip propagation, expansion,
fan/merge shapes, contract arms/modes). New-module static analysis and
`git diff --check` pass. The PDF is generated from this Markdown and visually checked.
