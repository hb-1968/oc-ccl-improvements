# S24 — zoom/tiling at inference on the best SAM 2 butterfly cell (1-shot and 5-shot)

Status: **built and pre-registered 2026-10-06, before any S24 cell exists; not yet run.** Contract
`src/occl/s24_contract.py`; driver `scripts/run_s24_stage.py`; notebook `notebooks/run_s24_scc.py` → `.ipynb`;
gates `s24_screen` / `s24`; report `scripts/report_s24.py`; tests `tests/test_s24.py`; BUILD 29.

## Why

- The query-level diagnosis ([QUERY_DIAGNOSIS.md](QUERY_DIAGNOSIS.md) §4) found thin traits under-resolved on SAM 2's
  256² mask grid.
- The local CPU probe ([ZOOM_PROBE.md](ZOOM_PROBE.md)) was pre-registered and returned **GO**. On base SAM 2,
  1-shot, no TTA, 40 upright dorsal cyrbia/malleti queries:
  - 2×2 quadrant tiles: **+1.55 [+0.55, +2.60] ✱**;
  - thin traits (< 5 grid cells): +6.5 ✱;
  - wide traits: flat.
- **User decision 2026-10-06:** "best model tuned with 1-shot and 5 shot. so just two runs."
  - S24 is therefore the quadrant mode (the probe's better label-free variant) on the best tuned SAM 2 adapter, at
    1-shot and at 5-shot.
  - There is no base arm, no specimen mode and no SAM 3. Those come later, if this pays.

## What

**Two cells.** Both use S19b's `crown_all_mref` adapter, sha-pinned through S21's `parents.json`, under the
label-consistent pool (S21's POOL), with TTA and 20 seeds:

| Cell | Parent (locked S21 cell, never re-run) | Parent score | Parent GPU |
| :-- | :-- | --: | :-- |
| `crown_all_mref_lcpool_zoomq_tta` | `crown_all_mref_lcpool_tta` | 77.31 | A40 |
| `crown_all_mref_lcpool_zoomq_5_tta` | `crown_all_mref_lcpool_5_tta` | **78.75** (best SAM 2 butterfly) | A40 |

**Stack:** `ZoomAdapter(OrientationTTAAdapter(SupportPoolTag(SSTSam2Adapter)), 'quadrants')`.
- The flags are `run_arms.py` / `run_eval.py --support-pool label_consistent --tta --zoom quadrants`.
- The stage driver fans 25 run slices per shot (5 groups × 5 slices of 4 runs), 5-shot first. Each fan scores the one
  arm.
- One inference-free merge per cell follows (`run_s18b_stage.confirm_graph(..., zoom=)`).

### The wrapper (`src/occl/adapters/zoom.py`, full method in its docstring and in ZOOM_PROBE.md)

All constants were fixed a priori on 2026-10-06, before the probe ran: `PAD` 0.10, `TRIM` 0.005, `OVERLAP` 0.10.

**Boxes and tiles:**
- Boxes are square in the 1024² frame: side = (1 + 2·PAD) × the longer side of the foreground's bounding box, shifted
  inside the frame.
- **Support box:** from its own GT. Support labels are the prompt, so this is legal.
- **Query box:** label-free. It is the foreground of the **first pass**, trimmed to the 0.5 %/99.5 % coordinate
  quantiles.
- **Quadrants:** 2×2 tiles of side (0.5 + OVERLAP) × box side. Tile *i* of each support pairs with tile *i* of the
  query.
  - A support whose tile is unlabelled is left out of that tile's sub-episode. An unprompted support frame would enter
    SAM 2's memory as a predicted frame.
  - A tile with no labelled support is skipped.

**Paste-back:** each tile's logits are resized bilinearly into the box.
- Inside the box: the zoomed logits, with a channel-wise max where tiles overlap.
- Outside the box, and in any skipped tile: the first pass.
- If the first pass predicts nothing, it is returned unchanged.

**Over TTA (new in S24):**
- **The first pass IS the parent's TTA prediction.** It is the locked TTA rule's choice between upright and
  180°-rotated supports, by logit confidence.
- **The tiles reuse that choice.** They run on the TTA wrapper's inner stack, with the rotated supports when TTA chose
  them (`OrientationTTAAdapter.last_rotated`, a bookkeeping attribute that changes no output).
- **Cost:** 2 TTA passes + 4 tile passes per episode, instead of a TTA pair per tile.

**Caching:**
- Support crops are fixed per (support, orientation), so they are encoded once through the base adapter's
  `precompute` and held in an LRU of `MAX_SUPPORT_CROPS` = 64.
- 64 is chosen so that one 5-shot TTA run's 40 crops never evict each other; that is about 0.5 GB in bf16.
- Query crops depend on the first pass and are encoded per episode.

**Run bank:**
- The adapter name gains `+zoom_quadrants`, giving its own slug.
- The wrapper's `inner` is itself, so `evaluator._adapter_source` fingerprints **zoom.py** into every zoomed run. An
  edit to the wrapper orphans banked zoomed runs instead of silently resuming them.
- The provenance also binds zoom.py's sha256 (`zoom_source`). Only a code-hash change may resume a root, under
  `ACCEPT_SOURCE_CHANGE=1`; a zoom.py change refuses the root.

### Hardware pin

- **The job script pins A40.** `scc_submit_s18b.sh` passes `-l gpu_type=A40` for s24 and carries it into requeues
  (`S18B_GPU_TYPE`). `GPU_TYPE=` lifts the pin, and `GPU_TYPE=<type>` sets another.
- **Why A40:** both parent cells ran on A40, and A40 allocations reproduce each other exactly. S21's pooled cells
  matched S19b's curated cells to +0.00 [+0.00, +0.00] in every unfiltered group.
- **Why not L40S:** L40S vs A40 bf16 is not bit-reproducible (S22, S23).
- **On matched hardware, the zoom-free first pass is the parent's prediction, so zoom − parent is the zoom alone.**
- **On a mismatch:** the driver warns, every cell records its GPU (`hardware`), and the reads flag
  `hardware_matched`. It is disclosed, not gated.

## Reads

Each zoomed cell is paired with its parent per support seed, on identical draws and weights:
- `registry.compare` with `model.zoom` relaxed. The new registry guard refuses a zoomed/plain pair unless that field
  is named.
- The full per-(group, run) draw identity is asserted first, because the locked evaluator's `support_set_hash` covers
  only the first group.
- CIs: 10,000 resamples over seeds, seed 0, unadjusted; ✱ = excludes zero.

| Read | Where |
| :-- | :-- |
| Major, zoom − parent, per shot; best-of-3 (secondary) | `s24_contract.reads` |
| Per subspecies (`group_compare`) | `s24_contract.reads` |
| GPU type matched to the parent | `s24_contract.reads` / the gate's detail line |
| By GT trait width in 256-grid cells (< 5, 5–12, ≥ 12), per run over (query, trait) units | `report_s24.py` |
| Label-style strata: notabilis tight/loose, lativitta rays 7/8, (m×p)×m ventral (QUERY_DIAGNOSIS §7) | `report_s24.py` |
| Major without the 3 defective test labels (diagnostic only) | `report_s24.py` |

The per-query reads need the S21 bundle (`results/inbox/s21.zip`, for the parents' rows) and
`results/diagnostics/queries.json`.

## Pre-registered bands (never gates)

| Read | Expected band | Falsifier |
| :-- | :-- | :-- |
| **PRIMARY: 5-shot Major, zoom − parent** | −0.3 to +0.8; **success ≥ +0.3 ✱** | ≤ 0, or significantly negative |
| 1-shot Major, zoom − parent | −0.5 to +1.0 | |
| Thin traits (< 5 grid cells), 5-shot | ≥ +2 ✱ (the probe's +6.5 on base, shrunk for the tuned model) | ≤ 0: the mechanism did not transfer |
| Wide traits (≥ 12 cells) | ±0.5 | |
| cyrbia and malleti | ≥ 0 | |
| lativitta rays 7/8 | at risk: their test labels are loose envelopes, and edge-exact predictions lost there under S23's SHAPE | |
| notabilis | tight half ≥ 0; loose half at risk (hand-drawn) | |
| Hardware | matched (A40 pin) | |

**Why the band is wide and centred low.** The probe measured base SAM 2 without TTA on the two subspecies with the
clearest resolution signature. The tuned 5-shot champion has already gained on thin traits: tuning lifted lativitta's
rays by 5 points.

**Local smoke** (below; 4 episodes; not evidence): 3 of 4 tuned-TTA cyrbia episodes lost under zoom. The same
queries lost in the probe too.

**Decision after landing.** Nothing is promoted automatically.
- **If** the 5-shot primary meets success **and** no subspecies loses significantly by more than 1 point, then the
  zoom becomes the recommended butterfly inference. It would be a modified-track pipeline like TTA, on the same locked
  metric over all 500 queries. A SAM 3 port and a base/specimen-mode follow-up become the next questions.
- **If not**, the zoom on cached 1024² pixels is closed for the tuned model.

## Gates

`s24_screen` / `s24` (`s24_contract.evidence_check`) require:
- **S21 parent bound:** the parent cells' receipts are under S21's identity, the adapter sha is pinned, and both
  parent cells are unzoomed, on the label-consistent pool, and carry the adapter's ‖ΔW‖ (rel 1e-4).
- **Provenance binds** the zoom mode and constants, zoom.py's hash, the adapter sha and S21's identity. The selection
  is the frozen plan.
- **`s24` additionally requires both zoomed cells:**
  - 20 runs × the parent's groups, `model.zoom = quadrants` and the frozen constants;
  - draws identical to the parent in **every** group, the same support selection, and the same adapter;
  - a recorded GPU type;
  - every read computable.
- **Never a demand for a win.**

## Execution

The code layer must be BUILD 29: push it first with the RUNBOOK's code-layer one-liner (no wipe). Then submit:

```bash
ssh hbridges@scc1.bu.edu 'bash /projectnb/bu-mlb/hbridges/sst/occl/oc-ccl-improvements/scripts/scc_submit_s18b.sh all 12 1 s24 2'
```

- **Inputs:** the S21 root (read-only), the S19b `crown_all_mref` adapter it points to, `data/occl-cache-delta.zip`,
  `ckpts/sam2_hiera_large.pt`. 40 GB per GPU, **A40 pinned**.
- **Wall-time estimate** (the per-episode multiplier is an estimate, not a measurement):
  - S21 measured 2.5 GPU-h for the 1-shot and 7.9 GPU-h for the 5-shot pooled `crown_all_mref` cell on A40.
  - The zoom adds 4 tile passes to TTA's 2, about 3×. That is ~24 GPU-h at 5-shot and ~7.5 GPU-h at 1-shot, so
    **~30 GPU-h, about 8 h on four cards**.
  - A 12 h allocation fits, with 2 automatic requeues that resume banked runs as insurance. h_rt ≤ 12 h also
    dispatches to shared GPU nodes.
  - The A40 pin may lengthen the queue wait.
- **Pull, gate and read:**

```bash
mkdir -p results/inbox && ssh hbridges@scc1.bu.edu 'cd /projectnb/bu-mlb/hbridges/sst/occl/results && ls -la s24.zip >&2 && tar cf - s24.zip' | tar xf - -C results/inbox/ && .venv/bin/python scripts/check_gate.py s24 --bundle results/inbox/s24.zip --no-file && .venv/bin/python scripts/report_s24.py --bundle results/inbox/s24.zip
```

## Local verification (2026-10-06)

**Tests (`tests/test_s24.py`, `tests/test_zoom.py`) cover:**
- the contract: it passes, and it refuses tampering — zoom mode, constants, draws moved in any group, adapter ‖ΔW‖,
  pool, missing GPU type, a zoomed parent, provenance constants and adapter sha, a missing cell;
- reads recovering a planted +1.5 effect, and the hardware flag, disclosed not gated;
- the registry guard;
- ZoomAdapter over TTA: tiles reuse the rotated supports TTA chose, the first pass stands outside the box, support
  crops are encoded once with a bounded LRU, the run-bank fingerprint hashes zoom.py, and first-pass logits are
  refused over TTA;
- a gate ↔ bundle round trip, PASS complete and FAIL with a cell missing, plus the report rendering;
- job-script and notebook pins.

**An older pin changed:** `tests/test_s20.py` pinned the requeue's `S18B_GPU_MEMORY=$GPUMEM"` string. It now pins
`S18B_GPU_MEMORY=$GPUMEM,S18B_GPU_TYPE=$GPUTYPE"`, because the memory floor still rides every requeue.

**Real-stack CPU smoke** (`ZoomAdapter(OrientationTTAAdapter(SupportPoolTag(SSTSam2Adapter)))`): the S19b
`crown_all_mref` adapter from `results/inbox/s19b.zip` (‖ΔW‖ 6.569), fp32 CPU, on S21's exact run-0 cyrbia 1-shot
episodes. The support draw was checked equal to the banked record's.

| Query | Orientation | Banked parent (A40 bf16) | Plain TTA (CPU) | Zoom (CPU) | TTA rotated | Zoom reused it |
| :-- | :-- | --: | --: | --: | :-- | :-- |
| 43119 | upright | 80.93 | 81.05 | 77.06 | no | no |
| 12768 | upright | 82.78 | 82.76 | 82.33 | no | no |
| 43810 | rot180 | 72.35 | 72.09 | 67.72 | yes | yes |
| 42616 | rot180 | 73.14 | 73.14 | 73.22 | yes | yes |

- **The plain stack reproduces the banked per-query scores** to within 0.26 on fp32 CPU vs bf16 A40.
- **The rotated path works.** The 8 support crops (4 upright + 4 rotated) were encoded once and then hit.
- **Zoom lost on 3 of these 4 episodes** (−4.0, −0.4, −4.4, +0.1). This is 4 episodes, not evidence. 43119 also lost
  under quadrants in the base-model probe (−2.7), whose gain came from other queries. It is recorded here so the
  landed read is not a surprise.

## Limitations

- Cached 1024² pixels: the zoom adds decoder and feature resolution, not image detail.
- **One adapter, one mode.** A null here does not close the specimen mode, base SAM 2 or SAM 3.
- The probe's selection (upright dorsal cyrbia/malleti) is not this stage's population. S24 scores all 500 queries,
  including 37 rotated cyrbia specimens, the (m×p)×m underside, and lativitta's and notabilis's hand-drawn labels.
- The cost multiplier is estimated.
- The A40 pin assumes A40 nodes with four free cards exist in the queue.
