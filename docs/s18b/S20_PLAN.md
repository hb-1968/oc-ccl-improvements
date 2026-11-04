# S20 — the SAM 2 picks on SAM 3, with SAM 3's concept wings (pre-registered 2026-09-27)

Status: **landed 2026-09-29** (gates PASS 2/2 on both datasets; run 1 lost butterfly's fans to the scratch cleaner, the resume completed them — see "Run 1" and "Landed result" at the end). Unit tests on stubs and synthetic
records, and CPU smokes on SAM 3's real architecture with RANDOM weights (no `sam3.pt`
exists locally) — see "Local verification" at the end. Scoring is locked (AGENTS.md):
argmax mIoU primary, best-of-3 boundary a secondary column, no fingerprinted module edited.

## Why

SAM 2 is closed as a platform (S17–S19b): the best SAM 2 cells are butterfly 78.56 at 5-shot
and beetle 75.86 at 5-shot, and every lever now measured on it is priced or null. SAM 3's
tracker already beats SAM 2 on beetle with a plain adapter (S14sam3: arch_tta 69.88 / 70.21
vs SAM 2 crown 64.12 at 1-shot; S15: arch_tta 78.91 at 5-shot) and ties it on butterfly
(76.59 at 1-shot; base_tta 77.95 > arch_tta 77.47 at 5-shot). Two things were never done:
(1) the recipes S17–S19b found on SAM 2 were never carried to SAM 3 — S14's "arch" adapter
is per-trait + excl zerocat + arch scopes on ALL legs, with NO objectness prior (it predates
S17b) and one reference; (2) SAM 3's own addition over SAM 2 — the detector's concept path
— was dropped on purpose by the S14 port (`adapters/sam3.py` grafts only the tracker and
the shared trunk). S20 does both, on both datasets, at 1-shot and 5-shot.

## The picks (from the locked SAM 2 cells)

| Pick | Why | SAM 2 beetle 1 / 5-shot | SAM 2 butterfly 1 / 5-shot | SAM 2 butterfly probe |
| :-- | :-- | :-- | :-- | :-- |
| `crown_all` | the replicated single-reference recipe; control for the other two | 64.94 / 75.50 | 77.16 / 78.53 | 34.8 % |
| `crown_all_mref` | + five-frame forward leg; best SAM 2 butterfly mIoU | 65.31 / 75.81 | 77.20 / 78.56 | 33.8 % |
| `odd_cf_all_mref` | + odd palindrome and counterfactual centre; best SAM 2 beetle 5-shot, only adapter below base on the probe | 65.24 / 75.86 | 76.89 / 78.43 | 19.7 % |

## The SAM 3 fits (training side)

Same recipe as SAM 2 (`s18b.step`, the arm table `s19b_contract.ALL_ARMS`): objness weight
10 on the forward frame's score, excl zerocat + Dice closing loss, per-trait masks,
`logit_reparam` prompt, memory reset, eval-mode training, lr 1e-5 cosine, one epoch, AdamW
wd 0.01, grad clip 1, r 32 / α 64, all legs, bf16 autocast — on SAM 3's tracker with SAM 3's
arch scopes (`SAM3_ARCH_SCOPES` = memory backbone, memory-attention transformer, mask
decoder; S14sam3's arch set). Same S18b fit / validation split (the manifest is
dataset-derived and the driver refuses a manifest whose hash differs from the S18b
root's). `crown_all_mref` / `odd_cf_all_mref` draw the same four extra references per pair
as on SAM 2 (seeded per pair, complete-first). The counterfactual arm's frozen-teacher
admission weights are recomputed with the SAM 3 base model (`calibrate` / `assemble` in
this root; SAM 2's rows would be the wrong teacher).

What had to change for SAM 3, each tested:
* `DifferentiableSAM3Tracker.track_with_mask` gains `frame_idx` / `num_frames` with SAM 2's
  semantics (SAM 3's `_prepare_memory_conditioned_features` keeps them; its memory-selection
  frame filter reads the same dict). The old one-conditioning-entry assertion becomes: an
  init call adds exactly one entry, at `frame_idx`, and `track_step` never stores one itself.
  SAM 3 returns `object_score_logits` natively in eval mode, so the objness term needs no hook.
* The SAM 3 tracker's heads return high-resolution predictions at the 1008-px IMAGE size,
  while mask prompts, targets and every evaluated prediction live on the 1152-px grid
  (`adapters/sam3.py` pins the video resolution to 1152). S14's cycle bridged this by
  nearest-resizing the target down to 1008; the S18b step (centre target, judges,
  calibration pooling, prompts) needs one grid, so `DifferentiableSAM3Tracker(output_size=
  1152)` recomputes each step's `pred_masks_high_res` from its differentiable 288-px
  logits by bilinear upsampling (`align_corners=False`) — exactly SAM 3's video predictor's
  `_get_orig_video_res_output` at inference. The memory is still encoded inside
  `track_step` from SAM 3's own output, as at inference. Found by the CPU smoke on SAM 3's
  real architecture (the closing loss refused a 1008-vs-1152 pair); the default (`None`)
  keeps S14's outputs, so S14/S15 fits are unaffected.
* `s18b.adversary(..., norm=)`: SAM 3 images are half-normalised ((x/255 − 0.5)/0.5), so the
  cut-out's pixel edit undoes that normalisation; SAM 3's masks are 1152 px against 1008-px
  images, so the elastic warp samples the same seeded field at each grid's resolution, the
  cut-out region is the label stack resized NEAREST to the image grid, and the translation
  shifts each grid by round(0.15·side) (151 / 173 px: a 0.43-px disagreement at mask scale).
  With equal grids and ImageNet statistics every branch is the S18b code, bit for bit (test).
* `run_s18b.py --family sam3 --stage-name s20`: SAM 3 adapter and tracker, pair geometry,
  scopes, metadata (`family`, scopes, normalisation, image 1008 / mask 1152, recipe label
  `sam3_port_of_sam2_crown`), identity payload + `family` and `stage` (SAM 2 payloads
  unchanged, test), and SAM 3's validation frees its feature cache after each pair.

Validation: the unchanged S18b held-out read on the SAM 3 calibration rows for base and the
three picks. Eligibility vs SAM 3 base (mean ≥ −0.5 pt, group ≥ −2 pt, translation ≥ −2 pt)
is annotated, never enforced; every pick is confirmed.

## The wings (inference side, `occl.adapters.sam3_concept`)

SAM 3's detector (Promptable Concept Segmentation) takes prompt tokens — text and/or box
exemplars — and returns up to 200 instance masks, a per-instance score and a separate
PRESENCE logit ("is the prompted concept in this image"). Loaded from the `detector.*` half
of `sam3.pt` with a strict key check (SAM 3's own loader only prints missing keys; the
tracker neck `sam2_convs` the detector half also carries is the only tolerated extra).

**Cross-image exemplars (our construction).** SAM 3's exemplar boxes are drawn on the
image being searched. Here, for trait c, every support that carries c contributes its
trait boxes: the connected components of the trait's label region (8-connectivity),
components below 5 % of the trait area dropped, the 3 largest kept, each as its tight box
in normalised cxcywh. Each support's boxes are encoded by SAM 3's geometry encoder against
THAT support's own image features (coordinates, ROI-pooled appearance, three cross-attention
layers over the support). The token sequences of all supports (each with its cls token) are
concatenated behind the "visual" text tokens SAM 3's processor inserts for box-only prompts,
and SAM 3's encoder, decoder and segmentation head ground the QUERY with them. An instance
is kept when σ(instance)·σ(presence) > 0.5 (the processor's default); D_c = the per-pixel
max over kept instances of the mask logit, bilinearly upsampled to the tracker's grid.

| Wing | Rule on the tracker's logits T_c, per prompted trait c | Aimed at |
| :-- | :-- | :-- |
| `none` | unchanged (the plain SAM 3 tracker) | the port read |
| `presence` | if σ(presence_c) < 0.5: T_c := −1e4 (the adapters' "never this label") | the S19b memorization read: fine-tunes paint a supported trait back into a query that lacks it |
| `exemplar` | if an instance is kept: T_c := logit(½σ(T_c) + ½σ(D_c)), exact in log space; else unchanged | detector evidence for traits the tracker under-segments |
| `stack` | exemplar, then presence | additivity |

All constants were set before any run (`PRESENCE_THRESHOLD`, `INSTANCE_THRESHOLD` 0.5,
`BOX_MIN_FRACTION` 0.05, `BOX_MAX_PER_SUPPORT` 3). A trait no support carries is never
grounded. Composition: `OrientationTTAAdapter(Sam3ConceptWing(Sam3TrackerAdapter))`; TTA's
rotated episode rotates the supports, so the exemplars come from the rotated supports and
TTA's confidence choice is made on the post-wing logits. The detector never sees the
tracker's LoRA, so one `Sam3Concept` per fan process caches its results per (query, support
set, trait) for the whole group and serves every (arm, wing) from that cache (fp16 maps,
the same stored values for every consumer). A wing cell's resume fingerprint hashes
`sam3_concept.py` (one-level unwrap); `adapters/sam3.py` is untouched. `registry.compare`
refuses to pair a wing cell with a plain cell unless `model.concept_wing` is named.

## Cells, reads and probe

Per dataset: {base, crown_all, crown_all_mref, odd_cf_all_mref} × {none, presence, exemplar,
stack} × {1-shot, 5-shot}, TTA, 20 paired seeds, canonical selection per dataset — 32 cells
(`s20_contract.cell_name`: `crown_all_presence_5_tta`, …). Memorization probe
(`probe_cutout`, the S19b operation) on the 5-shot cells of all four adapters under `none`
and `presence` — 8 probe cells; beetle all 20 runs, butterfly runs 0–2 (1 500 episodes).

Pre-registered reads (`s20_contract.reads`, `probe_reads`; paired per support seed, 10 000
resamples, seed 0, unadjusted):
1. port — each pick vs base at 1 and 5 shots; crown_all_mref − crown_all; odd_cf_all_mref −
   crown_all_mref; shot lift per adapter (nested pairing); shot interaction per pick;
2. wings — each wing − none per adapter per shot; stack additivity per adapter per shot;
   wing × pick interaction ((pick_w − pick) − (base_w − base));
3. the headline pairings on the best-of-3 reading;
4. probe — occupancy per (adapter, wing), paired per episode vs base and presence − none;
5. cross-family (report only, `--sam2-bundle results/inbox/s19b.zip`) — each SAM 3 cell vs
   its SAM 2 counterpart on identical support draws (`ignore_fields=model.family,
   model.image_size`; the draws are model-independent), for base and the three picks at
   both shots.

**Priors (bands, not gates).** Port: SAM 3 crown_all ≈ S14's arch (it adds the objness prior)
— beetle 1-shot 69–71, 5-shot 78.5–79.5; butterfly 1-shot 76–78, 5-shot 77.2–78.2. MREF on
SAM 3: beetle 5-shot 0 … +0.6 over crown_all, butterfly flat. odd_cf_all_mref: occupancy
below crown_all on butterfly. Wings on ordinary cells: presence −0.3 … +0.1 (test queries
carry their traits, so the gate can only cost where presence misses); exemplar −1 … +1
(cross-image exemplars are untested; the detector was trained on same-image exemplars);
stack ≈ the sum. Probe: presence lowers butterfly occupancy on every adapter if cross-image
presence works. Cross-family: beetle +4 … +6 at 1-shot and +3 … +4 at 5-shot; butterfly
within ±0.5 at 1-shot and −1 … +0.5 at 5-shot. No sign is promised.

**What would falsify what.** Picks ≈ S14 arch on SAM 3 → the SAM 2 levers (objness, MREF,
counterfactual) do not transfer. Presence costs > 0.3 on ordinary cells with no probe gain
→ cross-image presence is not usable as a gate. Exemplar ≤ 0 on both datasets → SAM 3's
exemplars do not transfer across images. A stack above both single wings with additivity
≈ 0 → two independent SAM 3 gains.

**Gates** (`check_gate.py s20_screen`, `s20`): screen = the S18b split, provenance hash,
frozen plan, SAM 3 calibration coverage, three fits (SAM 3 metadata and scopes, pairing,
finite losses, MREF references inside the pair's group fit split), four validations, the
eligibility reproduced; confirm = the 32 cells (shot, SAM 3 family, TTA, wing field, draws
identical to the base cell of that shot, adapters pinned by delta norm, rank, α and scopes,
5-shot prefix-nested on 1-shot), the 8 probe cells (coverage, draws, values), every read
computable. No gate demands a sign.

## Execution

`scripts/run_s20_stage.py` (S18b graph runner). In the driver before any GPU work: `plan`
per dataset (manifest = S18b manifest, frozen selection, `provenance.json`,
`reference.json`). Graph: SAM 3 calibration shards → assemble → `odd_cf_all_mref` fit;
`crown_all` and `crown_all_mref` fits at once; validations; an in-process `screen` worker
whose `then` adds the fans: 5-shot first (every slice scores four adapters × four wings with
one tracker encode and one detector cache), then 1-shot, then the probes; one merge per
cell (`run_eval.py --sam3-wing`, `run_cutout_probe.py --merge`). Bundle `results/s20.zip`.
Read: `scripts/report_s20.py` → [S20_RESULTS.md](S20_RESULTS.md). Resumes follow the S19b
rule (`ACCEPT_SOURCE_CHANGE=1` for a code-hash-only change, logged in `source_history`).

The notebook commissions the SAM 3 detector API of the cluster's sam3 checkout (the
internals the wing calls) before launching, and needs `ckpts/sam3.pt` (S14/S15's).

Batch (BUILD ≥ 25; the job asks 48 GB per GPU for S20 — SAM 2's odd_cf_all_mref peaked at
32.2 GB and SAM 3 runs 1008-px images, 1152-px masks and hosts the detector in every fan):

```bash
ssh hbridges@scc1.bu.edu 'bash /projectnb/bu-mlb/hbridges/sst/occl/oc-ccl-improvements/scripts/scc_submit_s18b.sh all 48 1 s20 2'
```

Sizing (estimates from S14/S15's measured SAM 3 slices: ~6 min per arm-slice at 1-shot TTA,
~11 at 5-shot TTA on butterfly): 16 tracker passes per slice → butterfly 5-shot ≈ 25 slices
× ~3 h, 1-shot ≈ 25 × ~1.6 h, probes ≈ 3 h, beetle ≈ 1/5 of that, fits ≈ 2 h → ≈ 30–35 h of
wall on four cards. One 48-h academic-GPU allocation (the deck's GPU limit), two automatic
requeues as insurance.

## Local verification (2026-09-27)

* Unit tests `tests/test_s20.py`: the SAM 3 tracker's frame positions, memory-dict ownership
  and 1152-grid outputs on a stub; the adversary's SAM 2 bit-identity and SAM 3 grids; the
  worker's family metadata and identity; box rule and exact log-space fusion; every wing
  rule; the registry guard; the contract with planted effects; the driver graph and a
  flag contract of every command it builds against its target script's argparse. A
  synthetic bundle passes `s20_screen` and `s20` and renders through `report_s20.py`.
* CPU smoke, SAM 3's real tracker and detector, random weights, image encoding replaced by
  correctly-shaped noise (SAM 3's ViT and neck run bf16 convolutions that take minutes per
  image on CPU; that code is S14/S15's, unchanged): `crown_all_mref` step 14 s and
  `odd_cf_all_mref` step 29 s, forward and backward, every term finite (anchor, objectness,
  odd return, centre target); the detector builds, encodes support exemplars (2 boxes → 2
  tokens with the cls), grounds the query, caches, and the gate and fusion branches both run
  on the live outputs (kept-instance maps 288 px, fused onto 1152).
* Found and fixed by the smokes: (1) the 1008-vs-1152 prediction grid (above); (2) the
  detector's decoder precomputes a coordinate cache on "cuda" in its constructor — patched,
  like the position encoding's, only when no GPU exists; (3) the detector runs fp32 on CPU
  (bf16 on CPU is ~150× slower; the GPU path stays bf16 as SAM 3 ships). A no-checkpoint
  build leaves two text-encoder parameters uninitialised, which makes SAM 3's OWN grounding
  return NaN presence; the real run loads them under the strict key check.

## Limitations

One fit per pick per dataset. Many intervals (≈ 100 per dataset), unadjusted. The cross-image
exemplar is our construction on a detector trained for same-image exemplars; a null there
is a statement about that construction, not about SAM 3's exemplars. The wings add
detector passes but never touch the tracker, so a wing's effect is conditional on the
tracker's logits. Beetle's small pool overlaps five-support sets heavily. The random-weight
CPU smoke proves the wiring, not the numbers.

## Run 1 (2026-09-27/28) — beetle complete, butterfly fans failed

One allocation, 8.75 h, 201 jobs: 95 complete, 66 failed, 40 skipped. Screen gate PASS on both
datasets; confirm gate PASS beetle, FAIL butterfly (coverage). SAM 3 fits: butterfly 8.8 /
17.7 / 29.6 min (crown_all / crown_all_mref / odd_cf_all_mref), beetle 0.8 / 1.4 / 2.3 min;
calibration 3.9 min per butterfly shard. Beetle fans: 5-shot slices 77 min, 1-shot 53 min
(16 tracker passes each). Butterfly cyrbia slices 0–3 completed (5-shot 167 min, 1-shot
115 min), cyrbia probe runs 0–2 completed; cyrbia runs 16–19 and EVERY slice and probe of the
other four subspecies failed within a minute of starting (rc 1), including probe runs that
started after cyrbia's succeeded. **Cause (from the logs):** `FileNotFoundError: butterfly_major/11892 is
not cached (/scratch/…/occl-cache/…)` — the job's node-local image cache lost files mid-run, as in S19b,
and the re-stage fallback declined silently. The pattern identifies the mechanism: only images no worker
had read recently vanished (cyrbia's, in constant use, survived), and both losses began around
02:30–03:00 — a nightly scratch cleaner deleting files by age. Staging copies the source files'
timestamps (copytree / unzip keep them, weeks old), so every staged image looked stale on arrival.
**Fix (code layer 9f940288):** every stage notebook stamps each staged file's atime/mtime with the
current time after staging; the re-stage fallback now prints why it declines (unset, not found, or
different bytes from the manifest's sha). Resubmitting resumes the root (`ACCEPT_SOURCE_CHANGE=1`: the
code hash changed); beetle's cells and butterfly's fits, validations and cyrbia slices carry receipts.

**Beetle read** ([S20_RESULTS.md](S20_RESULTS.md); TTA, 20 paired seeds):

| | 1-shot | Δ vs SAM 3 base | 5-shot | Δ vs SAM 3 base |
| :-- | --: | :-- | --: | :-- |
| base (SAM 3) | 66.32 | — | 77.96 | — |
| crown_all | 69.71 | +3.39 ✱ | 78.51 | +0.55 ✱ |
| crown_all_mref | **70.19** | +3.87 ✱ | **78.95** | +0.99 ✱ |
| odd_cf_all_mref | 70.13 | +3.81 ✱ | 78.76 | +0.81 ✱ |

* MREF pays on SAM 3 too: crown_all_mref − crown_all +0.48 ✱ at 1-shot, +0.43 ✱ at 5-shot;
  odd_cf_all_mref ≈ crown_all_mref (−0.07 / −0.18, ns). The picks equal S14/S15's SAM 3 arch
  cells (1-shot 69.88 / 70.21, 5-shot 78.91; unpaired) — the SAM 2 levers do not add a
  measurable step on SAM 3 beetle beyond what the arch scopes already gave, MREF aside.
* Cross-family, same recipe, paired on identical draws: SAM 3 − SAM 2 = +4.8 … +4.9 ✱ at
  1-shot and +2.9 … +3.1 ✱ at 5-shot for every pick (base +2.5 ✱ / +2.8 ✱).
* Shot interaction −2.8 … −3.0 ✱: on SAM 3 the fine-tune's beetle gain shrinks from ~+3.8 at
  1-shot to ~+0.8 at 5-shot (SAM 2: +1.1 → +0.35).
* **presence wing: inert.** Identical to the plain tracker on every run of every cell,
  including the memorization probe (Δ 0.00): σ(presence) ≥ 0.5 for every prompted trait of
  every episode — with cross-image exemplars SAM 3's presence head always reports the trait
  present, even when it has been inpainted out of the query. The gate never fired.
* **exemplar wing: harmful.** −11.2 … −11.9 ✱ at 1-shot, −5.4 … −6.1 ✱ at 5-shot on every
  adapter (stack = exemplar, since the gate never fired). Averaging in the detector's masks
  from cross-image exemplars degrades the tracker's segmentation badly.
* Memorization probe: SAM 3 fine-tunes raise removed-trait occupancy by +2.2 … +2.7 pp ✱ over
  SAM 3 base (19.96 %); odd_cf_all_mref lowest of the three, not below base.

## Landed result (2026-09-29)

Resume (job 7766895, `ACCEPT_SOURCE_CHANGE=1`, code hash 50b4fca1 → d033b682 logged on both roots):
141 jobs, all complete, 32 h of wall; butterfly 5-shot slices 153 min mean / 272 max, 1-shot 140 /
187, probes 10 min. Both gates PASS ([s20_gate.txt](s20_gate.txt)); read:
[S20_RESULTS.md](S20_RESULTS.md). SAM 3 butterfly `odd_cf_all_mref` peaked at **42.2 GB** in
training — the 48-GB request was necessary (a 40-GB card would have run out of memory).

**Butterfly, SAM 3, TTA, 20 paired seeds:**

| | 1-shot | Δ vs SAM 3 base | 5-shot | Δ vs SAM 3 base | vs SAM 2 same recipe (5-shot) |
| :-- | --: | :-- | --: | :-- | :-- |
| base | 76.50 | — | 77.96 | — | +0.73 ✱ |
| crown_all | 77.10 | +0.60 ✱ | 78.00 | +0.04 | −0.53 ✱ |
| crown_all_mref | 77.38 | +0.89 ✱ | 78.20 | +0.24 ✱ | −0.37 ✱ |
| odd_cf_all_mref | 77.25 | +0.76 ✱ | 78.19 | +0.23 ✱ | −0.24 ✱ |

* The recipes help SAM 3 on butterfly, but less than they help SAM 2: SAM 3's base is ahead
  (+1.32 ✱ at 1-shot, +0.73 ✱ at 5-shot), the tuned SAM 3 picks tie their SAM 2 counterparts at
  1-shot (−0.06 … +0.37, ns) and trail them at 5-shot (−0.24 … −0.53 ✱). **The programme's best
  butterfly cell stays SAM 2 `crown_all_mref` 78.56 at 5-shot.** MREF pays on SAM 3 butterfly too
  (+0.29 ✱ / +0.20 ✱ over crown_all). Shot interaction −0.53 … −0.65 ✱.
* **Memorization probe — the property transfers and is nearly free on SAM 3.** Removed-trait
  occupancy at 5-shot: base 23.4 %, crown_all 32.4 % (+8.9 ✱), crown_all_mref 32.7 % (+9.2 ✱),
  **odd_cf_all_mref 15.8 % (−7.6 ✱ below base)** at the same mIoU as crown_all_mref (−0.01 ns at
  5-shot; −0.13 ✱ at 1-shot); tracking of the remaining traits unchanged (others mIoU 77.2–78.0).
  Validation cut-out occupancy 9.7 % vs base 20.8 % and crown_all 33.6 %. SAM 2 S19b showed the
  same ordering (19.7 % vs 24.9 %); on SAM 3 the separation is larger and costs nothing measurable.
* **presence wing: inert on both datasets** — identical to the plain tracker on every run of
  every cell and the probe. **exemplar wing:** butterfly −0.5 … −1.6 ✱ at 1-shot; at 5-shot −0.57 ✱
  on base and +0.11 … +0.16 on the picks (crown_all_mref +0.16 ✱, 78.36 — the best SAM 3 butterfly
  cell, still below SAM 2's 78.56); wing × pick +0.7 ✱ at 5-shot (the detector's masks help tuned
  trackers more than the base). Beetle: −11 ✱ / −6 ✱. Stack = exemplar everywhere.

**Disposition.** Beetle belongs to SAM 3 (crown_all_mref 70.19 / 78.95; SAM 3 − SAM 2 +4.9 ✱ /
+3.1 ✱); butterfly stays with SAM 2 at 5-shot (78.56) and ties at 1-shot. Across both families
`odd_cf_all_mref` is the generalisation pick: within 0.2 of the best mIoU everywhere and the only
adapter below base on the memorization probe. SAM 3's concept path, as constructed here
(cross-image exemplars), is closed: the presence gate never fires and the exemplar fusion is
net-negative except a +0.16 on tuned butterfly trackers at 5-shot. No recipe promotion.
