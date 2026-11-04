# S19b — two 5-shot levers on the S18b wings, stacked, both datasets (pre-registered 2026-09-24)

Status: **landed 2026-09-25** (gates PASS 2/2 on the resumed root; see "Landed result" at the end, after the run-1 record). Parents: the landed S18b root ([PLAN.md](PLAN.md) "Landed result") and the
landed S19 root ([S19_PLAN.md](S19_PLAN.md) "Landed result"). Both are read-only inputs.

## What is locked (user directive 2026-09-24)

**Scoring.** The primary metric is the argmax mIoU every recorded number uses
(`metrics.score_episode`, `gt_native` space). Nothing in the scoring path changes: the
fingerprinted modules (`metrics`, `protocol`, `curation`, `seeding`, `evaluator`, the
SAM 2 adapter) are untouched, so every banked run stays valid. The paper's unpublished
best-of-3 boundary tolerance (`miou_boundary.argmax_best3`, computed per run since S11)
is **reported as a secondary column**, never as the headline and never as a gate. To the
question "are we using the best-of-3 reading": no — the headline has always been argmax;
best-of-3 sits +1.3 above it on butterfly at 5-shot and is now shown beside it.

**Experiments.** No landed cell is re-run. The S19 5-shot cells (`base_5_tta`,
`crown_all_5_tta`, `odd_cf_all_5_tta`) and the S18b 1-shot cells are copied into the S19b
record from the parents' receipted zips, and the new fans bank under their own keys
(different adapter name for CANON, different shot for the MREF 1-shot cells, the
`cutout/` sub-bank for probes), so the shared run bank is only ever appended to.

**Numbers at the start line** (TTA, 20 paired seeds):

| | butterfly 1-shot → 5-shot | beetle 1-shot → 5-shot |
| :-- | :-- | :-- |
| base | 75.18 → 77.23 | 63.86 → 75.15 |
| crown_all | 77.16 → 78.53 (+1.30 ✱ vs base) | 64.94 → 75.50 (+0.35 ns) |
| odd_cf_all | 77.02 → 78.46 (−0.06 ✱ vs crown_all) | 64.86 → 75.47 (−0.03 ns) |

Paper SAM 2 5-shot: butterfly 83.4 / 83.9 (+OC-CCL), beetle 74.2 / 75.6.

## The two levers

### MREF — multi-reference training (`occl.finetune.s18b.step(extra_refs=…)`)

*Hypothesis.* The adapters sit on memory encoding, memory attention and the mask decoder
and were trained with ONE conditioned frame in memory; at 5-shot the bank holds five. S19's
negative interactions (−0.68 ✱ butterfly, −0.73 beetle) say the adapter and the extra
supports are partial substitutes. Training the forward leg in the 5-frame regime is the
direct response.

*Operation.* For fit pair *i* (reference x₀ with label y₀, query x₁), `REFS − 1 = 4`
extra labelled fit images of the same group are drawn — `extra_reference_ids`: candidates
are the group's fit split minus the pair's own reference and query; those carrying every
trait of the pair (`class_ids`) are drawn first, seeded per pair by
`derive_seed(ds, 's19b-mref|<ref>|<query>', 0, seed=18)`; incomplete candidates only if the
complete ones run out (count recorded per step as `refs_incomplete`). Each extra reference
is encoded (frozen encoder, no grad) and conditioned as an init frame at `frame_idx =
1..4` with its per-trait mask stack in the PAIR's channel order (`reference_masks`: a
trait the extra reference lacks gets an empty channel, which SAM 2 conditions as "object
absent on this frame"). The query is then tracked at `frame_idx = 5, num_frames = 6`, so
memory attention sees five conditioned frames — the inference regime at 5-shot (SAM 2
attends to all conditioning frames; `max_cond_frames_in_attn = −1`). Everything after the
forward output is unchanged: the reverse leg still resets and closes on the pair's own
reference with the same loss; for `odd_cf_all_mref` the centre frame attends to the same
five conditioned frames plus the forward output (`frame_idx = 6, num_frames = 7`), the
frozen-teacher admission weights are the PARENT calibration rows (computed at one
reference; they are admission weights, not memory-regime quantities), and the odd return
is untouched. With no extra references the calls are the historical ones argument for
argument (pinned by test).

*Arms.* `crown_all_mref` (anchor, all legs) and `odd_cf_all_mref` (counterfactual, all
legs), same recipe otherwise (objness w10, excl zerocat, arch scopes, r 32 / α 64, lr 1e-5,
one epoch, S18b fit split and seeds). Cost: the forward leg encodes six images instead of
two (CPU smoke: 37.7 s vs 15.2 s per step on the real model); butterfly fits ≈ 15–30 min.
VRAM: odd_cf_all peaked at 26.4 GB on butterfly at one reference; five references add
four memory encodings and a 5× wider memory cross-attention on one leg — expected to stay
under the 40 GB floor; the training record carries `peak_vram_gb`.

*Validation.* The unchanged S18b held-out read (ordinary 1-shot mIoU, cut-out occupancy,
translation) on the parent calibration rows — deliberately at ONE reference, so the
floors compare like with like against the parent record. Eligibility (annotated, never
enforced — every arm is confirmed): mean ≥ parent − 0.5 pt, every group ≥ −2 pt,
translation ≥ −2 pt, versus the arm's OWN parent (crown_all or odd_cf_all).

### CANON — per-support orientation canonicalisation (`occl.adapters.orientation`)

*Hypothesis.* The orientation TTA flips all supports jointly. Beetle's `canonical_balanced`
draw mixes orientations in proportion to the training pool, so at 5-shot a mixed set stays
mixed under any joint flip; S19 still chose the rotated set in 16–18 % of beetle episodes.

*Operation.* `PerSupportCanonicalAdapter` wraps the SAM 2 adapter INSIDE the TTA wrapper.
Per group it computes `curation.orientation_strata(ds, group, train_ids)` once (train-only
trait-centroid reference; the state the balanced draw stratifies on) and rotates each
support labelled `rot180` by 180°, image and label map together, before prediction. The
TTA joint flip still runs on top. Cache keys: the TTA wrapper suffixes `@r180`; the
canonical wrapper counts that parity, adds its own flip, and hands the inner adapter
parity 0 (`x`) or 1 (`x@r180`) with the image rotated the same number of times, so a
support flipped by TTA and flipped back here is served the upright cached embedding; both
keys are cached by the TTA precompute, so canonicalisation costs no extra encoding. The
support DRAW is untouched (`support_ids` are recorded before any wrapper), so a canonical
cell pairs run by run with its as-drawn cell. Nothing consults a query, a test label or a
score. Butterfly has no axis, but the strata are centroid-based (they found cyrbia's 37 %
rotated specimens), so the lever is defined there too and is a no-op on the four
subspecies with 0–1 % rotated members. Records carry `model.support_orientation =
"canonical"` (absent = as drawn); `registry.compare` refuses to pair the two orientations
unless `ignore_fields=("model.support_orientation",)` names the relaxation, exactly as
`model.tta` is handled. Flags: `run_arms.py` / `run_eval.py --support-orientation
canonical`; the default builds the byte-identical historical stack. Adapter name
`sst_sam2+persupport_canonical+tta_rot180`.

## Cells (per dataset; `s19b_contract.cell_name`)

| Adapter | 1-shot as drawn | 5-shot as drawn | 5-shot CANON |
| :-- | :-- | :-- | :-- |
| base | locked (S18b) | locked (S19) | **new** `base_canon_5_tta` |
| crown_all | locked | locked | **new** `crown_all_canon_5_tta` |
| odd_cf_all | locked | locked | **new** `odd_cf_all_canon_5_tta` |
| crown_all_mref | **new** `crown_all_mref_tta` | **new** `crown_all_mref_5_tta` | **new** `crown_all_mref_canon_5_tta` (stack) |
| odd_cf_all_mref | **new** `odd_cf_all_mref_tta` | **new** `odd_cf_all_mref_5_tta` | **new** `odd_cf_all_mref_canon_5_tta` (stack) |

Nine new cells per dataset, all TTA, 20 paired seeds, the per-dataset canonical selection
(`canonical_placement` butterfly, `canonical_balanced` beetle), the standard predictor.
The 1-shot MREF cells keep S19's shot-interaction read alive for the new adapters. The
wing rows exist for both baselines by construction: every lever is applied to
`crown_all` and to `odd_cf_all`, and the report asks whether a lever helps the palindrome
wing more than the anchor wing.

## Memorization read — the test-time cut-out probe (`occl.probe_cutout`)

On the 5-shot as-drawn cells of all five adapters (beetle all 20 runs; butterfly runs
0–4, 500 episodes per cell), one trait is inpainted out of every query and the model is
asked to predict the edited query under the cell's own inference configuration:

* removed trait = deterministic per query (`derive_seed(ds, 's19b-cutout|<query>', 0, 0)`
  over the traits present in the query and the episode), so every arm cuts the same
  trait from the same query and the per-episode pairing is exact; queries with fewer than
  two usable traits are skipped;
* edit = `centre_transforms.cutout_image` (Telea inpainting over the trait's label region
  dilated by 9 px) at native resolution — the S18b adversary; the edited query gets its
  own embedding key so the cache can never serve the unedited features;
* `occupancy_soft` = mean sigmoid of the removed trait's channel inside its own region at
  model resolution (the S18b `cut_recall` quantity; **lower = less hallucination of a
  trait the supports say should be there**); `occupancy_hard` = the fraction labelled as
  that trait by the argmax-with-background rule; `others_miou` = IoU of the remaining
  traits against the unedited GT (tracking under the edit); `copy_excess` = mean over
  supports of IoU(prediction, support layout) − IoU(GT, support layout) over the
  remaining traits (S18's carry read, on the edited-query prediction).

Read: occupancy per adapter, paired per episode versus base and versus crown_all (10,000
resamples). It is a **consideration** beside the mIoU reads — an arm that gains mIoU by
painting supported traits it cannot see is generalising less than one that does not —
and never a gate. Banked under `<runs>/cutout/` with this module's hash in the slug.

## Pre-registered reads (`s19b_contract.reads`, `probe_reads`)

1. MREF lever, per wing: `X_mref_5 − X_5` and `X_mref_1 − X_1` (plain pairing); the
   shot interaction for the MREF arms (nested pairing, S19's read).
2. CANON lever, per adapter: `X_canon_5 − X_5` (orientation relaxation named); base,
   both wings, both MREF arms.
3. Stack, per wing: `X_mref_canon_5 − X_5`, `− base_5`, and additivity per seed:
   (stack − parent) − [(canon − parent) + (mref − parent)]; zero = the levers add.
4. The wings under each lever: `odd_cf_all − crown_all` as drawn (locked), under MREF,
   under CANON, under both; and the lever-by-wing interaction (lever gain on odd_cf_all −
   lever gain on crown_all) — the direct answer to "does any advantage enhance odd_cf".
5. Every new 5-shot cell versus base; the headline pairings repeated on the best-of-3
   reading (`@best3`).
6. Memorization: occupancy per adapter, paired deltas vs base and vs crown_all.

**Priors (bands, not gates).** CANON on beetle base −0.3 … +1.0 (the mixed sets are
16–18 % of episodes; a per-support fix cannot exceed what the joint flip already
recovers by much); CANON on butterfly ≈ 0 outside cyrbia. MREF versus its parent at
5-shot −0.5 … +1.0 on both datasets; at 1-shot −0.5 … +0.3 (a five-frame regime may cost
the one-frame cell). Stack ≈ sum of the levers. Wing interactions within ±0.2. Occupancy:
odd_cf_all below crown_all on butterfly if the S18b property survives five supports; MREF
arms unknown (the probe is the read). No sign is promised anywhere.

**What would falsify what.** MREF ≈ 0 at 5-shot on both wings → the adapter/support
substitution is not a training-regime artefact and SAM 2 5-shot is closed at S19's
numbers. CANON ≈ 0 on beetle → orientation mixing is not what limits beetle 5-shot. A
stack that beats both single levers with additivity ≈ 0 → two independent gains, the
first SAM 2 5-shot improvement over S19. A wing interaction ✱ in odd_cf_all's favour → the
palindrome wing benefits more and its 1-shot cost is recoverable.

**Gates** (`check_gate.py s19b_screen`, `s19b`): screen = both parents bound (S18b identity,
wing adapters, calibration, validation, 1-shot cells; S19 identity and 5-shot cells;
locked cells paired and prefix-nested), both MREF fits complete with every step's four
extra references inside the pair's group fit split and disjoint from the pair, the frozen
plan and eligibility reproduced; confirm = the nine new cells present with the right
shot, orientation field and selection, draws identical to the locked base cell of that
shot, adapters pinned by delta norm to their training record, the five probe cells with
the required run coverage and draws, every read computable. No gate demands a sign.

## Execution

`scripts/run_s19b_stage.py` (S18b graph runner; `confirm_graph` gained `orientation` and
`label` arguments, default-inert). Per dataset in the driver process: `plan` (binds both
parents, copies the manifest, verifies the cache still reproduces it, writes
`parents.json` and the frozen selection; identity = digest of the usual payload + S18b
identity + S19 identity + stage `s19b` + refs 5 + shot 5). Graph: the two MREF fits
(`run_s18b.py train --stage-name s19b --parent-root <S18b>`) → validations → a `screen`
worker (eligibility, screen gate) → its `then` expands the MREF fans (5-shot as drawn,
5-shot CANON, 1-shot, probes); the locked-adapter fans (5-shot CANON on base, crown_all,
odd_cf_all; their probes) start at once. Sliced `run_arms.py` fans (every arm of a fan
on one encode), `run_cutout_probe.py` fans, one merge per cell (`run_eval.py
--no-precompute`, `run_cutout_probe.py --merge`), receipts, bundle `results/s19b.zip`
(records, both MREF adapters per dataset, the new test and probe zips). Read:
`scripts/report_s19b.py` → [S19B_RESULTS.md](S19B_RESULTS.md).

Batch: the S18b qsub with `OCCL_STAGE=s19b`, root `results/s19b_work_v1`, BUILD ≥ 24:

```bash
ssh hbridges@scc1.bu.edu 'bash /projectnb/bu-mlb/hbridges/sst/occl/oc-ccl-improvements/scripts/scc_submit_s18b.sh all 12 2 s19b'
```

Sizing (estimates from S19's measured 3 min per arm-run on butterfly 5-shot TTA slices):
butterfly CANON fan on five adapters ≈ 25 slices × 60 min ≈ 6.3 h of card time; MREF 5-shot
as-drawn and CANON fans (2 arms each) ≈ 2 × 2.5 h; the 1-shot MREF fan ≈ 0.8 h; probes
(5 arms × 5 runs, one prediction each) ≈ 1.6 h; beetle ≈ 1.5 h in all; fits and validations
in the shadow. ≈ 13–15 h of wall on four cards → two chained 12 h allocations (the second
resumes the same root and exits at once if the first finished). Dropping the two
single-lever CANON cells on butterfly (`crown_all_canon`, `odd_cf_all_canon`) would save
≈ 2.5 h; they are kept because the wing-by-lever read needs them.

## Limitations

One fit per MREF arm per dataset (no training replicates); the extra references are
drawn once per pair (no reference resampling across epochs — one epoch). The MREF
validation read is at one reference by design. Twenty-odd intervals per dataset,
unadjusted. Beetle's 79-image pool makes five-support sets overlap heavily across seeds.
The probe's `copy_excess` is measured on the edited-query prediction, not the ordinary
one. Butterfly probes cover 5 of 20 seeds. CANON cells' resume fingerprint hashes the
wrapper's source in place of the frozen SAM 2 adapter's (one-level unwrap in
`evaluator._adapter_source`); both are pinned in the code layer.

## Run 1 (2026-09-24/25) — incomplete, environment not science

`all 12 2 s19b`: job 7720768 (queue iris-gpu-pub) ran its full 11.6 h driver budget with
142 of 218 jobs exit 0 and none failed — the sizing was short, not the code — then job
7720769 (queue ece-long, node scc-j05) resumed, reused the banked runs, and at +190 min its
node-local `$TMPDIR` cache lost files while the job was still running (the first failing
probe had scored two arms on the same episodes four minutes earlier; the third arm then hit
`FileNotFoundError: butterfly_major/15898 is not cached`). Every later worker died on the
same error at startup: 54 complete, 139 failed, 25 skipped; both gates FAIL on coverage.
Recorded in [s19b_gate.txt](s19b_gate.txt) (run 1). Nothing in the fits or the scoring path
failed. What landed and is banked: both MREF fits and validations on both datasets (all four
eligible; `odd_cf_all_mref` lowered cut-out occupancy further, butterfly 19.0 → 16.4 %,
beetle 30.8 → 28.1 %), the three butterfly CANON cells, and every completed fan slice.

**Partial read (butterfly, 5-shot TTA, CANON − as drawn, paired):** base 77.23 → 77.22
(−0.01 [−0.01, −0.00]), crown_all 78.53 → 78.53 (0.00), odd_cf_all 78.46 → 78.46 (0.00);
cyrbia 78.41 → 78.41. Per-support orientation is a null on butterfly, as pre-registered.
Beetle, where the lever is aimed, has no cells yet.

**Fixes (code layer BUILD 24, 2026-09-25).** (1) `ImageCache.restage`: a missing local
cache file is re-staged from `$OCCL_CACHE_FALLBACK` (the project-space cache zip or folder,
exported by every stage notebook), byte-verified against the manifest's `cache_sha256`,
instead of killing the worker. (2) The job script requeues itself after an incomplete driver
exit (`S18B_AUTO_RESUBMIT` hops, passed by the submit helper's fifth argument, default 2;
an identical allocation resuming the same root; never after a clean finish). (3) The
helper's wall-time cap follows the SCC deck (slide 18): 48 h on academic GPUs, 12 h on
shared GPUs. Resume:

```bash
ssh hbridges@scc1.bu.edu 'ACCEPT_SOURCE_CHANGE=1 bash /projectnb/bu-mlb/hbridges/sst/occl/oc-ccl-improvements/scripts/scc_submit_s18b.sh all 24 1 s19b 2'
```

(one 24 h academic-GPU allocation, two automatic requeues as insurance; it resumes
`s19b_work_v1`, so the fits, the three CANON cells and the banked slices cost nothing.)

**Resume attempt 1 (2026-09-25) stopped at once:** the root's identity binds the code hash, and
the code layer had changed (fixes 1–3 above), so the driver refused the root. (4) The driver
now accepts a code-hash-only provenance change on resume when `ACCEPT_SOURCE_CHANGE=1` is
passed deliberately, keeping the recorded identity and logging `{recorded, accepted}` in
`selection.json` `source_history` (carried in the bundle). Method note: the identity's
purpose is to bind artifacts to the code that produced them; the accepted change (cache
re-stage, job requeue, wall cap) touches no training or scoring path, and the record of
the acceptance is what makes that claim auditable. The command above includes the flag.

## Landed result (2026-09-25)

Resume (`ACCEPT_SOURCE_CHANGE=1 … all 24 1 s19b 2`, first allocation): 190 jobs, all complete,
6.2 h of wall, on the same root; butterfly `source_history` records the accepted code-hash change
(2383424636b8 → b8d6d3b50b52). Beetle's history is empty: it was planned second, under the override
the butterfly acceptance had just set, so its identical recorded hash matched without logging
(fixed since: `plan` compares against the real hash; both roots were planned by the same code
under the same acceptance). Both gates PASS ([s19b_gate.txt](s19b_gate.txt); run 1:
[s19b_gate_run1.txt](s19b_gate_run1.txt)). Read: [S19B_RESULTS.md](S19B_RESULTS.md). Measured:
MREF fits butterfly 19.7 / 34.3 min (peak 16.9 / 32.2 GB), beetle 1.7 / 2.8 min; butterfly
5-shot CANON slices 31.1 min mean / 47.7 max carrying two arms; 1-shot MREF slices 12.5 min.

**Beetle (the target).** MREF works: crown_all_mref **75.81**, odd_cf_all_mref **75.86** at 5-shot
TTA — +0.31 [+0.13, +0.50] ✱ and +0.39 [+0.11, +0.67] ✱ over their parents, and +0.66 ✱ / +0.71 ✱
over base 75.15, where S19's parents were +0.35 / +0.32 ns. This is the first SAM 2 fine-tune
that moves beetle at 5-shot, and 75.86 sits above the paper's SAM 2 + OC-CCL 5-shot 75.6 ± 1.5
(unpaired). It is not a 5-shot-specific effect: at 1-shot MREF is +0.37 on both wings (ns), and
the shot interaction stays negative (−0.79 ✱ / −0.66 ✱), so the five-frame regime adds a general
gain rather than closing the adapter/support substitution. CANON does not help: −0.17 on base
and −0.06 … −0.12 on the adapters (all ns), and the stack is therefore below MREF alone
(+0.25 ✱ / +0.32 ✱ vs parent) with additivity ≈ 0. The wings remain equivalent under every lever
(odd − crown −0.07 … +0.06, all ns).

**Butterfly.** MREF is flat: crown_all_mref 78.56 (+0.04 [+0.01, +0.07] ✱, negligible),
odd_cf_all_mref 78.43 (−0.04 ns); CANON is identical to as-drawn on every adapter (base −0.01;
the placement-curated pool is already upright). crown_all_mref 78.56 is the new SAM 2 butterfly
high by 0.03. Under MREF the odd wing's cost widens (−0.14 ✱, lever-by-wing −0.07 ✱).

**Memorization probe — the sharpest read of the stage.** Butterfly, removed-trait occupancy at
5-shot: base 24.9 %, **crown_all 34.8 % (+9.9 ✱ vs base)**, crown_all_mref 33.8 %, odd_cf_all
25.3 % (+0.5 ns — back to base), **odd_cf_all_mref 19.7 % (−5.1 ✱ below base, −15.1 ✱ vs
crown_all)**. Every adapter tracks the remaining traits equally (others mIoU 78.2–78.3 vs base
76.9). So the anchor fine-tune's butterfly gain comes with painting a removed trait back in
at a much higher rate than the untouched model, the counterfactual wing removes that entirely
at a −0.06 cost, and the multi-reference counterfactual wing hallucinates LESS than the base
model while keeping +1.20 ✱ over it. Beetle occupancy differences are under 1 pp (fine-tunes
+0.5 … +0.9 over base 18.0 %; odd wings 0.2–0.3 below crown).

**Disposition.** No recipe promotion (user rule). For the 5-shot beetle question the answer is
MREF: `odd_cf_all_mref` 75.86 / `crown_all_mref` 75.81, equivalent, both ✱ over base. CANON is
closed on both datasets. On butterfly the best generalising adapter is `odd_cf_all_mref`: within
0.13 of the best mIoU and the only one below base on the memorization probe.

**Correction to the CANON premise (measured after landing, 2026-09-25).** Counting the train-derived
orientation state of every drawn 5-shot support (`orientation_strata` over the locked base draws):
beetle **1 of 100 supports** is `rot180` (run 13 only); butterfly **0 of 500**. The canonical
support pools (`canonical_balanced`, `canonical_placement`) already exclude almost every rotated
specimen, so the pre-registered hypothesis — "the balanced draw mixes orientations, so a joint
flip cannot align a 5-shot set" — was wrong, and CANON was nearly a no-op by construction. S19's
16–18 % TTA rotations on beetle are the joint flip winning on upright sets, not a mixed-set
symptom. The lever's only real effect is that one support in beetle run 13: −3.50 on base,
−1.21 on crown_all_mref (whose canonical and as-drawn cells come from the same allocation and
differ ONLY in that run). CANON is closed as a mis-aimed lever, not as a measured null.

**Cross-allocation noise floor (measured).** Cells computed in different allocations are not
bit-identical on beetle: on the 19 unflipped runs, CANON-vs-locked per-run differences have mean
−0.05 … +0.00 and sd 0.16–0.28 pp (max 0.97) for base / crown_all / odd_cf_all; butterfly wings are
bit-identical and butterfly base differs by sd 0.04. Same-allocation cells are deterministic
(crown_all_mref canon vs as-drawn: identical except run 13). The floor's mean is two orders below
the MREF effects and its variance is inside the paired per-seed CIs, so the MREF reads stand, but
paired deltas under ~0.1 pp across allocations on beetle are not interpretable. Also: the CANON
cells' `support_orientation_flipped` counters read 0 because `config_report` is taken in the
inference-free merge process, not the fans (reporting gap).
