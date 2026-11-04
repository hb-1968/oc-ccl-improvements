> Status update, 2026-11-04: [S21 results](S21_RESULTS.md) record completed
> screen and confirmation gates. The preregistration below retains its original
> pre-run status and decisions. Gate completion does not imply improvement.

# S21 — label-consistent OC-CCL on butterfly (SAM 2, 1-shot and 5-shot)

Status: **pre-registered 2026-10-02; implemented and CPU-smoke-tested; not run.** A SAM 3 follow-up (later stage)
ports only a lever that clears its band. Contract: `src/occl/s21_contract.py`. Driver: `scripts/run_s21_stage.py`.
Gates: `s21_screen`, `s21`. Report: `scripts/report_s21.py` → `docs/s18b/S21_RESULTS.md`.

## Why

The two butterfly diagnoses locate most of the remaining Major gap on the label side:

- **(m×p)×m** ([MXP_DIAGNOSIS.md](MXP_DIAGNOSIS.md)). The 68 dorsal test specimens score 81. The 32 ventral ones
  plateau at ~53 in every SAM 2 and SAM 3 cell. The train split annotates the underside three ways: only 33 of 104
  ventral masks follow the test's thin-streak convention; 43 are "wide" and 27 incomplete. A few shrivelled
  one-blob masks are also present.
- **The fine-tune and the supports both admit those masks.** The S18b fine-tune manifest takes every cached train
  id with no filter, so those masks are cycle anchors (the closing target is the reference's y₀). The curated
  support pool still holds 24 wide ventral masks.
- **Lativitta** ([LATIVITTA_DIAGNOSIS.md](LATIVITTA_DIAGNOSIS.md)). Its rays are a test-only convention that no
  training data carries. **User decision: report only.**
- **User decision: pure OC-CCL, no supervised forward term.** The cycle never reads a query label (`s18b.step`), so
  S21 can change only what the cycle and the evaluator are *shown*.

**Honest prior:** these are small, precise levers. A CPU probe bounds a perfectly matched ventral support at about
+1.2 on ventral queries. The value of S21 is an attributable test of whether label consistency is what holds the
cycle back on butterfly.

## The rule (`occl.conventions`, `RULE = label_consistent_v1`, frozen before any S21 run)

Scope: only malleti and (m×p)×m (`STRIP_GROUPS`, the two 10-trait vocabularies with the hindwing costal bar at
traits 5/8). Every other subspecies is admitted unchanged. Within scope, a TRAIN mask is dropped if it is:

- **empty**;
- **tiny**: foreground below `TINY_REL = 1/3` × the group's median train foreground share (3.1 % of the image for
  (m×p)×m, 2.5 % for malleti);
- **ventral and incomplete**: fewer than `MIN_TRAITS = 8` of the 10 traits;
- **ventral and wide**: traits (5+8) at or above `STRIP_MAX = 0.18` of the foreground.

View is read from the filename (`_d`/`_v`, case-insensitive; no token counts as dorsal). The rule reads train masks
and filenames only.

**Disclosed:** `STRIP_MAX` and `MIN_TRAITS` were set in the diagnosis *after* viewing the test ventral range
(5.3–15.0 %) and the bimodal train histogram, so they are test-informed heuristics. An absolute 3 % tiny threshold was
tried first and rejected *before any run*: it would have dropped 611 of cyrbia's 929 train masks, because cyrbia is
photographed small (median foreground 2.9 %). The relative threshold and the group scope were fixed in response.

Measured effect of the rule:

| Group | Train masks dropped | Of those in the S18b fit/validation manifest | Curated support pool |
| :-- | --: | --: | --: |
| (m×p)×m | 74 (43 wide, 13 incomplete, 17 tiny, 1 empty) | 19 | 299 → 280 |
| malleti | 5 (wide) | 1 | 695 → 692 |
| others | 0 | 0 | unchanged |

## Levers (one wing each, plus the stack, on both S19b baselines)

The baselines are `crown_all_mref` (the best SAM 2 butterfly result, 78.56 at 5-shot) and `odd_cf_all_mref` (the
generalisation pick).

1. **CLEAN** (`crown_all_mref_lc`, `odd_cf_all_mref_lc`): refit with the identical recipe (MREF, refs = 5, all legs,
   objectness 10, lr 1e-5, one epoch) on the CLEAN manifest.
   - **Splicing.** The CLEAN manifest is the S18b manifest with the dropped ids spliced out of each group's pair
     cycle (`s21_contract.splice_manifest`). A dropped image x turns p → x → s into p → s; p's row keeps its
     transform, cut trait and seed.
   - **Every other row is kept verbatim and in order.** So the fit differs from its parent only by the removed masks
     and 14 spliced fit queries (3 in validation): 886 → 870 fit pairs, 221 → 217 validation.
   - **Calibration.** S21 runs its own frozen-teacher calibration: the spliced pairs differ, so the parent's
     calibration rows do not apply. The S18b `make_manifest` is unchanged; the banked S18b manifest still reproduces
     byte for byte.
2. **POOL** (`--support-pool label_consistent`): supports are drawn by the locked protocol over a dataset view whose
   `train_ids` omit the dropped masks (`occl.datasets.filtered.LabelConsistentView`). The adapter stack carries
   `SupportPoolTag` (`+lcpool`), which keeps pooled runs in their own run-bank slug and records `model.support_pool`.
   - Seeds are unchanged, so cyrbia, lativitta and notabilis draw **identically**; this was verified for all 20 runs
     at both shot counts.
   - malleti and (m×p)×m are fully re-drawn, because the locked protocol permutes the smaller pool.
   - Of the 20 curated (m×p)×m draws, 1 at 1-shot and 8 at 5-shot held a dropped mask.
3. **STACK**: CLEAN fits scored with POOL.

## Cells (TTA, 20 seeds, `canonical_placement`)

- **Locked, never re-run:** base (S18b 1-shot, S19 5-shot) and both baselines (S19b), 1-shot and 5-shot, curated pool.
- **New, 14 cells:**
  - POOL × {base, `crown_all_mref`, `odd_cf_all_mref`} × {1, 5};
  - {`crown_all_mref_lc`, `odd_cf_all_mref_lc`} × {curated, label-consistent} × {1, 5}.
- **Probe:** the cut-out memorization probe on the two CLEAN 5-shot curated cells (runs 0–4). These are paired per
  episode with the baselines' S19b probe cells.

## Reads (`s21_contract.reads`; primary argmax mIoU, best-of-3 secondary)

- **CLEAN vs parent:** same draws, `registry.compare`, pooled and per group (`group_compare`) for (m×p)×m and malleti.
- **POOL vs curated:** `seed_paired`, per support seed. `registry.compare` refuses this pair through its
  `model.support_pool` guard. *Corrected 2026-10-04: an earlier version said the support-set hash differs. It doesn't:
  the locked evaluator's `protocol.support_set_hash` hashes only the first group's draws (cyrbia's), so the two pools
  share it.* `seed_paired` proves the rest instead: every other comparability field matches, the weights are
  identical, and draws moved only in `STRIP_GROUPS`. The (m×p)×m and malleti components are effectively unpaired.
- **STACK vs parent** and **additivity**: (stack − parent) − [(CLEAN − parent) + (POOL − parent)].
- **Report tables:**
  - per subspecies;
  - (m×p)×m by photographed side, with ventral traits 5/8, 6/9 and 2/3;
  - lativitta rays vs its other traits and its best-of-3 lift (report only);
  - the probe.
- **CIs:** 10,000 resamples over support seeds, seed 0, unadjusted. Expect multiple comparisons; read the bands, not
  single stars.

## Pre-registered bands (never gates)

| Read | Band | Falsifier |
| :-- | :-- | :-- |
| CLEAN vs parent, (m×p)×m ventral | +1 to +4 | ≤ 0 |
| CLEAN vs parent, (m×p)×m | +0.3 to +1.3 | |
| CLEAN vs parent, other subspecies | ±0.3 | |
| CLEAN vs parent, Major | +0.05 to +0.3 | |
| POOL, (m×p)×m (unpaired) | ≤ +0.4 | |
| POOL, cyrbia / lativitta / notabilis | exactly 0 (identical draws: a correctness check) | any nonzero |
| STACK additivity | reported, no band | |
| Memorization occupancy, CLEAN vs parent | within ±3 pp | |
| Lativitta rays | unchanged ±1 | |
| Lativitta best-of-3 lift | stays about +5.3 | |

## Gates

`s21_screen` (after the fits) and `s21` (everything), both `s21_contract.evidence_check`:

- the manifest re-derives from the S18b manifest and its own drop list;
- the calibration covers the CLEAN pairs;
- training histories match the CLEAN fit pairs, and the extra references lie inside the group's CLEAN fit split;
- validations cover the CLEAN split;
- eligibility reproduces (CLEAN arms vs base on the CLEAN split; recorded, not enforced);
- locked cells are paired and prefix-nested;
- every new cell is paired within its pool, with pooled draws moving only in `STRIP_GROUPS`;
- adapters are pinned by ‖ΔW‖;
- probes are complete;
- every read computes.

A pass is completion, not a win. No selection and no promotion: `sam2_crown` stands.

## Execution

```bash
ssh hbridges@scc1.bu.edu 'bash /projectnb/bu-mlb/hbridges/sst/occl/oc-ccl-improvements/scripts/scc_submit_s18b.sh all 48 1 s21 2'
```

- **Inputs:** the S18b, S19 and S19b roots (read-only) and the BUILD 26 code layer.
- **Wall-time estimate:** calibration (2 shards) plus two CLEAN fits, in the shadow of the POOL fans for the locked
  adapters; then the CLEAN fans and the probe. Roughly 10–14 h on four cards.
- **Pull, gate and read:**

```bash
mkdir -p results/inbox && ssh hbridges@scc1.bu.edu 'cd /projectnb/bu-mlb/hbridges/sst/occl/results && tar cf - s21.zip' | tar xf - -C results/inbox/ && .venv/bin/python scripts/check_gate.py s21 --bundle results/inbox/s21.zip --no-file && .venv/bin/python scripts/report_s21.py --bundle results/inbox/s21.zip
```

## Local verification (2026-10-02)

- `pytest tests/` passed: 417 tests, including `tests/test_s21.py`, which covers:
  - the rule's classes and scope;
  - the splice keeping untouched rows verbatim and leaving a single cycle per group;
  - `pairs_for` installing spliced pairs while keeping the strict check for unspliced manifests;
  - the POOL view's memo isolation and name;
  - contract tampering;
  - planted-effect recovery;
  - the registry refusing a pairing across pools.
- On the real data:
  - the banked S18b manifest reproduces byte for byte;
  - the CLEAN manifest is `7573573b904a` (20 masks dropped);
  - POOL draws match in the three untouched subspecies for all 20 runs at both shot counts.
- **CPU smoke (S21-specific worker plumbing).**
  - `calibrate` ran 2 steps on the CLEAN pairs.
  - `validate --arm base` loaded the CLEAN manifest, computed the S21 identity, installed the spliced pairs, built
    the model and ran a held-out prediction. It then stopped, as expected, at the validation calibration row: the
    smoke had calibrated only 2 fit rows.
- **No local training smoke.** A CLEAN training step ran the 46 GB workstation out of memory. The step is the MREF
  recipe: five conditioned frames plus backward through fp32 SAM 2-large at 1024 px. The training loop is S19b's
  unchanged MREF code; S21 changes only the manifest and the pair list, both unit-tested. **The first real CLEAN
  step therefore happens on the SCC.** The screen gate checks every training history row against the CLEAN pairs.

## Limitations

- **The constants are test-informed** (see above).
- **CLEAN can only reach (m×p)×m through the anchors.** It cannot teach ventral semantics. The cycle has no query
  supervision, so the ventral traits 5/8 and 6/9 (unrecoverable by any support in the probe) may not move.
- **POOL's (m×p)×m and malleti reads are effectively unpaired.**
- **The split overlap disclosed in MXP_DIAGNOSIS still applies.** 42 of the 100 (m×p)×m test specimens are
  re-photographed in the fit set.
- **Lativitta is report-only.**

## Run 1 — 2026-10-03 (screen PASS; confirmation FAIL: the POOL half did not run; the CLEAN half landed)

**What failed and why.** Every POOL fan failed in groups g2 and g3 (malleti and (m×p)×m), and all six POOL cells plus
the four STACK cells were skipped. The image cache was built for the *curated* draws (`build_cache.required_butterfly`
over `SELECTIONS`). The label-consistent pool is smaller, so the locked protocol re-permutes it and draws supports
that were never cached: 56 in malleti and 32 in (m×p)×m, 88 in all. Each fan died on its first cache miss. The other
three subspecies draw identically and were unaffected. **My oversight:** the plan verified that the draws were
identical outside the filtered groups but never checked that the new draws were cached.

**Fixes:**
- `run_s21_stage.pool_cache_shortfall` now refuses to plan if any label-consistent-pool support (5-shot covers 1-shot)
  is missing from the cache.
- `occl-cache-delta.zip` carries the 88 images. They were built with the faithful pipeline (rawpy for CR2,
  INTER_LINEAR, lossless PNG) from archived originals or Zenodo. The notebook already merges this delta.
- The CLEAN cells, fits, calibration and probes are kept. A resume reuses them by receipt and re-runs only the POOL
  and STACK fans. The code-layer change requires `ACCEPT_SOURCE_CHANGE=1`.

### CLEAN result (landed, paired on identical draws; 10,000 resamples; ✱ = CI excludes 0)

| Read | `crown_all_mref_lc` 1-shot | `odd_cf_all_mref_lc` 1-shot | `crown_all_mref_lc` 5-shot | `odd_cf_all_mref_lc` 5-shot |
| :-- | :-- | :-- | :-- | :-- |
| Major vs parent | +0.01 [−0.03, +0.05] | −0.11 [−0.16, −0.06] ✱ | +0.04 [+0.01, +0.07] ✱ | +0.02 [−0.01, +0.05] |
| (m×p)×m vs parent | −0.09 [−0.26, +0.05] | −0.31 [−0.48, −0.16] ✱ | +0.11 [−0.01, +0.25] | +0.02 [−0.09, +0.13] |
| (m×p)×m **ventral** queries vs parent (per-seed mean over the 32 ventral queries) | −0.35 [−0.84, +0.09] | −1.03 [−1.54, −0.55] ✱ | +0.33 [−0.05, +0.75] | +0.03 [−0.29, +0.39] |
| malleti vs parent | +0.05 [−0.05, +0.17] | −0.13 [−0.27, −0.03] ✱ | +0.03 [+0.03, +0.04] ✱ | +0.03 [+0.02, +0.04] ✱ |
| Cut-out occupancy vs parent (5-shot probe), pp | | | −0.51 [−0.77, −0.27] ✱ | −0.83 [−1.17, −0.51] ✱ |

Absolute cells:
- `crown_all_mref_lc`: 77.21 at 1-shot and **78.60** at 5-shot, against its parent's 77.20 and 78.56.
- `odd_cf_all_mref_lc`: 76.78 and 78.45, against 76.89 and 78.43.
- Validation on the CLEAN split: 82.21 and 81.79, against base's 80.22. Both fits are eligible.

**Against the bands, the CLEAN lever is falsified:**
- The ventral band was +1 to +4, with ≤ 0 as the falsifier. Three of the four reads sit at or below zero, one
  significantly negative, and the fourth is +0.33 with a CI crossing zero.
- (m×p)×m pooled (band +0.3 to +1.3) is missed.
- Major (band +0.05 to +0.3) is met only by `crown_all_mref_lc` at 5-shot, at +0.04 ✱.
- The other subspecies stay within ±0.3, as predicted.
- Memorization is within its ±3 pp band.

**Reading.** Removing the 20 contradictory anchors does not move the underside. That fits the diagnosis'
mechanism: the cycle closes on the reference, never on a ventral query's labels, so clean anchors teach nothing about
what a ventral trait looks like. Under pure OC-CCL the ventral deficit cannot be reached from the training side.
POOL (test side) is the remaining lever. The probe bounds it at about +0.4 on the subspecies.

**Delta built 2026-10-03.**
- `occl-cache-delta.zip` (152 MB) holds 88 entries, all fetched from Zenodo (none from the local archive), lossless
  PNG, sha256-verified against their manifest entries.
- CR2 images were decoded by rawpy, exactly as the base cache.
- `pool_cache_shortfall` is now empty locally.
- The resume reuses every receipted cell and re-runs only the 10 POOL/STACK cells.
- Push the delta to `data/occl-cache-delta.zip`, push the code layer, then resubmit with `ACCEPT_SOURCE_CHANGE=1`
  (the code hash changed).

## Run 2 — 2026-10-03 (allocation 7864001: STOP at plan, no job launched)

The delta merged correctly (+88 entries, 0 absent). The driver then stopped with "dataset/cache no longer matches the
S18b manifest".

**Cause:** `make_manifest` defines the S18b split as every cached train id. The delta added 88 train images to the
cache, so rebuilding the manifest no longer reproduced the stored one, although no existing input changed. That was
my second oversight: S21's plan and worker re-derived a *fixed* parent manifest from a cache that S21 itself grows.

**Fix:** S21 now calls `s18b.inputs_unchanged(ds, parent_manifest)`. Every image the manifest names must still be
cached, with its cached-file sha256 and label-map digest equal to the manifest's `input_hashes`.

Verified on the delta-merged local cache:
- the old check fails;
- the new check passes (5 s);
- the CLEAN manifest is unchanged (`7573573b904a`);
- a tampered hash is caught;
- regression test `test_inputs_unchanged_tolerates_added_images_but_not_changed_ones`.

The S18b/S18c/S19b workers keep the strict `make_manifest(ds) == m` check. Those stages are landed, but a **rerun of
any of them would now refuse on a delta-merged cache** and would need the same change. No work was lost: nothing ran
in run 2. Resume as before, with `ACCEPT_SOURCE_CHANGE=1`.

## Landed result — 2026-10-04 (run 3, allocation 7864153; gates `s21_screen` PASS, `s21` PASS)

**Gate fix made after landing (disclosed).** `seed_paired` first compared adapter ‖ΔW‖ for exact equality and
refused the CLEAN arms' own cells: one adapter file, re-loaded in another process, gave 6.5953971687704 vs
6.5953971724957. It now uses the project's adapter-pin tolerance (rel 1e-4, as `_pin_adapter`). No score changed.

### POOL (label-consistent support pool) — the lever that moved

Read per support seed with `seed_paired`. The (m×p)×m and malleti components are re-drawn, so they are effectively
unpaired; ✱ = CI excludes 0.

| Adapter | 1-shot Major | 1-shot (m×p)×m | 5-shot Major | 5-shot (m×p)×m |
| :-- | :-- | :-- | :-- | :-- |
| base | +0.06 [−0.23, +0.37] | +0.22 [−1.01, +1.40] | +0.04 [−0.05, +0.12] | +0.37 [+0.01, +0.72] ✱ |
| crown_all_mref | +0.11 [−0.11, +0.33] | +0.33 [−0.87, +1.45] | **+0.19 [+0.08, +0.29] ✱** | **+0.89 [+0.43, +1.36] ✱** |
| odd_cf_all_mref | +0.20 [−0.01, +0.39] | +0.40 [−0.72, +1.40] | +0.12 [+0.01, +0.23] ✱ | +0.55 [+0.03, +1.05] ✱ |
| crown_all_mref_lc | +0.10 [−0.10, +0.30] | +0.30 [−0.81, +1.36] | +0.16 [+0.07, +0.26] ✱ | +0.76 [+0.37, +1.18] ✱ |
| odd_cf_all_mref_lc | +0.21 [−0.02, +0.44] | +0.43 [−0.72, +1.47] | +0.07 [−0.03, +0.15] | +0.30 [−0.14, +0.71] |

- **`crown_all_mref` + POOL scores 78.75 at 5-shot**, against 78.56 curated. This is the new programme-best SAM 2
  butterfly cell.
- (m×p)×m ventral queries at 5-shot:
  - `crown_all_mref` goes from 52.69 to 55.41.
  - Its ventral traits 2/3 go from 63.0 to 71.0.
  - Its ventral traits 6/9 go from 36.1 to 39.3.
  - Ventral 5/8 stays about 25, the structure no support recovers.
  - Dorsal is unchanged (81.98 vs 82.01).
- **POOL exceeded its pre-registered band** (≤ +0.4 on (m×p)×m) at 5-shot for three of five adapters. The band came
  from the 1-shot, no-TTA, single-support CPU probe. At 5-shot the curated pool mixes wide-convention ventral masks
  into the memory bank, and the diagnosis had already seen ventral queries do worse with one ventral support
  (50.6) than with none (54.6). Removing those masks helps most at 5-shot. That explanation came after the result
  and is a hypothesis, not a measurement.
- 1-shot POOL effects are all non-significant, +0.06 to +0.21.

### STACK and additivity

- STACK (CLEAN + POOL) vs parent:
  - `crown_all_mref_lc` at 5-shot: +0.20 [+0.10, +0.30] ✱.
  - Every other cell is non-significant, +0.09 to +0.11.
- Additivity is about 0: −0.01 and +0.02 at 1-shot, −0.03 and −0.05 ✱ at 5-shot. **STACK ≈ POOL alone; CLEAN adds
  nothing** (see the CLEAN table in Run 1).

### Correctness band: cyrbia / lativitta / notabilis under POOL "exactly 0"

- **Met for 8 of 10 cells**: all 60 runs bit-identical.
- **Missed for both `odd_cf_all_mref` POOL cells.** The draws and the adapter are identical, but every run differs
  (max |Δ| 0.21 at 1-shot and 0.64 at 5-shot; pooled +0.005 and +0.008).
- So the locked S19b `odd_cf_all_mref` cells are not bit-reproducible in a later allocation, plausibly from GPU or
  code-version nondeterminism; the cause is not established.
- The bias is about 20× smaller than the POOL effects above, but it is a measured reproduction gap and is preserved
  here.

### Lativitta (report only)

As the diagnosis predicted, no cell moved its rays:
- rays 7/8 stay at 42.8–48.4;
- the other traits stay at 81.9–83.8;
- the best-of-3 lift stays at +5.29 to +5.48 in every cell.

### Reading

- **The test-side lever moved; the training-side one did not.** Under pure OC-CCL, label consistency matters where
  the model is prompted (the supports), not where the cycle is anchored.
- **The gain is concentrated where predicted:** (m×p)×m ventral traits 2/3, the convention-dependent ones. The ventral
  traits 5/8 stay out of reach.
- **Disclosure:** POOL's thresholds are test-informed (see "The rule"). Removing inconsistent support labels is a
  curation choice of the same kind as completeness curation, but the 78.75 carries that disclosure.
- **Next (proposed):** port POOL, not CLEAN, to SAM 3 (`crown_all_mref` / `odd_cf_all_mref` at 1-shot and 5-shot).
  The SAM 3 adapters already exist from S20, so it needs inference fans only.

**Correction 2026-10-04.** `protocol.support_set_hash` (locked `evaluator.py`) hashes only the first group's
(cyrbia's) support draws. It therefore cannot detect a draw change in any other subspecies, and a pooled cell and its
curated cell share it (e69b1a06ba68 at 5-shot). S21 and S22 never relied on it: their contracts compare the full
per-(group, run) draws (`_draws`). Older stages' pairings carry the same limitation wherever only the hash guarded
them. Recorded, not changed: the evaluator is under the scoring lock.
