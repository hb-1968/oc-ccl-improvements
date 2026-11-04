> Status update, 2026-11-04: [S22 results](S22_RESULTS.md) record completed
> screen and confirmation gates. The preregistration below retains its original
> pre-run status and decisions. Gate completion does not imply improvement.

# S22 — POOL (label-consistent supports) on SAM 3, butterfly, 1-shot and 5-shot

Status: **pre-registered 2026-10-04; implemented and unit-tested; not run.** Inference only.

- Contract: `src/occl/s22_contract.py`.
- Driver: `scripts/run_s22_stage.py`.
- Gates: `s22_screen`, `s22`.
- Report: `scripts/report_s22.py` → `docs/s18b/S22_RESULTS.md`.

## Why

S21 found that drawing butterfly supports from the curated pool minus the label-inconsistent masks moves SAM 2 at
5-shot ([S21_PLAN.md](S21_PLAN.md) "Landed result"):
- `crown_all_mref` 78.56 → 78.75 (+0.19 ✱);
- (m×p)×m +0.89 ✱, carried by its ventral queries (ventral traits 2/3: 63 → 71).

The rule is `occl.conventions`, `label_consistent_v1`. Refitting without those masks (CLEAN) did nothing.

**User decision, 2026-10-04:** port POOL to SAM 3 first. CLEAN is not ported. The thin costal-streak loss idea (ventral
traits 5/8) is deferred.

## Design

- **Adapters:** the four S20 SAM 3 butterfly adapters, read-only and pinned by ‖ΔW‖ (rel 1e-4) to S20's records.
  These are `base` and S20's refits of `crown_all`, `crown_all_mref` and `odd_cf_all_mref`. Plain tracker, no
  concept wing: S20 found presence inert and exemplar net-negative.
- **Cells:**
  - each adapter × {1-shot, 5-shot}, TTA, 20 seeds, `canonical_placement`, `--support-pool label_consistent`;
  - **8 new cells**;
  - partners: S20's locked curated plain-tracker cells (never re-run).
- **Machinery:**
  - The pool is S21's `datasets.filtered.LabelConsistentView` plus `adapters.support_pool.SupportPoolTag` (`+lcpool`
    bank slug, `model.support_pool`).
  - Draws are identical to S20's in cyrbia, lativitta and notabilis × lativitta, and re-drawn in malleti and
    (m×p)×m.
  - The SAM 3 adapter is the S20 one (`--adapter sam3`), at 1008 px with half normalisation. The pool wrapper sits
    inside TTA, as on SAM 2.
- **Preflight (S21 run 1's failure):** `plan` refuses unless every label-consistent-pool support is cached
  (`run_s21_stage.pool_cache_shortfall`). The 88 extra images ship as `occl-cache-delta.zip`, which is already on
  the SCC.
- **No training, no calibration, no memorization probe.** The adapters are unchanged and POOL changes only the
  supports.

## Reads (`s22_contract.reads`; primary argmax mIoU, best-of-3 secondary)

- **POOL per adapter and shot:** pooled cell vs its S20 curated cell, per support seed (`s21_contract.seed_paired`).
  Read for Major, for (m×p)×m and malleti (effectively unpaired, since they are re-drawn), and for the three
  untouched groups (the correctness band).
- **Each pick vs base**, within the pool.
- **Cross-family** (report, with the S21 bundle): each SAM 3 pooled cell − its SAM 2 pooled cell. The pool and seeds
  are the same, so the draws are identical. This is asserted per (group, run) before `registry.compare`.
- **Report tables:** per subspecies; (m×p)×m by photographed side, with ventral traits 5/8, 6/9 and 2/3.
- **CIs:** 10,000 resamples over support seeds, seed 0, unadjusted.

## Pre-registered bands (never gates)

| Read | Band | Basis |
| :-- | :-- | :-- |
| POOL, Major, picks at 5-shot | +0.05 to +0.25 | SAM 2: +0.12 … +0.19 ✱ |
| POOL, (m×p)×m, picks at 5-shot | +0.3 to +1.0 | SAM 2: +0.55 … +0.89 ✱ |
| POOL, (m×p)×m ventral traits 2/3, 5-shot | up | SAM 2: 63 → 71 |
| POOL, ventral traits 5/8 | unchanged ±2 | no support recovers them |
| POOL, 1-shot | non-significant, \|Δ\| ≤ 0.25 | SAM 2: all non-significant |
| POOL, cyrbia / lativitta / notabilis | 0, identical draws | correctness check; SAM 2 missed it on one adapter (reproduction gap ≤ 0.008) |
| SAM 3 pooled − SAM 2 pooled, 5-shot | ≤ 0 for the picks | S20: SAM 3 picks trailed SAM 2 by 0.24–0.53 ✱ at 5-shot on butterfly |

**Falsifiers:**
- POOL on the SAM 3 picks at 5-shot ≤ 0 on (m×p)×m would mean the S21 gain is SAM 2-specific.
- A nonzero untouched-group delta beyond reproduction noise would mean a plumbing error.

## Gates

`s22_screen` checks the parent and the plan:
- the S20 parent is bound;
- the adapters are pinned;
- the locked cells are curated, plain-tracker and mutually paired;
- the frozen plan is in place.

`s22` adds the cells:
- all 8 pooled cells are present, pool-tagged, paired within the pool, moved only in the filtered groups and pinned;
- every read computes.

A pass is completion, not a win. No selection, no promotion.

## Execution

```bash
ssh hbridges@scc1.bu.edu 'bash /projectnb/bu-mlb/hbridges/sst/occl/oc-ccl-improvements/scripts/scc_submit_s18b.sh all 48 1 s22 2'
```

- **Inputs:** the S20 root (read-only), `ckpts/sam3.pt`, the sam3 checkout, `data/occl-cache-delta.zip` and the
  BUILD 27 code layer. The job asks 48 GB per GPU, as for S20.
- **Wall-time estimate:** four adapters on one SAM 3 encode per slice and no wings, which is roughly a quarter of
  S20's wing fans: about 8–10 h on four cards.
- **Pull, gate and read:**

```bash
mkdir -p results/inbox && ssh hbridges@scc1.bu.edu 'cd /projectnb/bu-mlb/hbridges/sst/occl/results && ls -la s22.zip >&2 && tar cf - s22.zip' | tar xf - -C results/inbox/ && .venv/bin/python scripts/check_gate.py s22 --bundle results/inbox/s22.zip --no-file && .venv/bin/python scripts/report_s22.py --bundle results/inbox/s22.zip
```

## Local verification (2026-10-04)

- `pytest tests/` passed (422 tests), including `tests/test_s22.py`:
  - contract tampering;
  - planted POOL effects;
  - the untouched-group band;
  - the cross-family pairing, and its refusal when any group's draws differ.
- `s22_screen` passes on the real S20 record: bound as the parent, 8 locked cells paired, adapters pinned.
- **No local GPU smoke.** SAM 3 inference is S20's tested path, and the pool wrapper is S21's tested path.

## Limitations and side findings

- **The rule's thresholds are test-informed** (see S21).
- **The (m×p)×m and malleti components are effectively unpaired.**
- **`protocol.support_set_hash` covers only the first group's draws** (locked evaluator). Found 2026-10-04 while
  building S22, and recorded in S21_PLAN. S21 and S22 assert full draws themselves.

## Landed result — 2026-10-05 (gates `s22_screen` PASS, `s22` PASS)

POOL per support seed against the locked S20 curated cell; ✱ = CI excludes 0.

| Adapter (SAM 3) | 1-shot Major | 5-shot Major | 5-shot (m×p)×m | 5-shot malleti |
| :-- | :-- | :-- | :-- | :-- |
| base | +0.01 [−0.22, +0.23] | **+0.11 [+0.03, +0.19] ✱** | **+0.71 [+0.35, +1.09] ✱** | −0.15 [−0.36, +0.07] |
| crown_all | +0.03 [−0.24, +0.30] | +0.02 [−0.10, +0.16] | +0.13 [−0.45, +0.77] | −0.12 [−0.36, +0.12] |
| crown_all_mref | +0.04 [−0.19, +0.25] | **+0.16 [+0.07, +0.25] ✱** | **+0.88 [+0.46, +1.30] ✱** | −0.08 [−0.31, +0.14] |
| odd_cf_all_mref | +0.05 [−0.18, +0.28] | **+0.15 [+0.07, +0.23] ✱** | **+0.83 [+0.46, +1.21] ✱** | −0.07 [−0.31, +0.16] |

**Absolute cells:**
- 5-shot: `crown_all_mref` 78.20 → **78.35**; `odd_cf_all_mref` 78.19 → 78.34; base 77.96 → 78.07.
- 1-shot: `crown_all_mref` 77.38 → 77.42 (non-significant).

**(m×p)×m ventral queries at 5-shot** (`crown_all_mref`):
- ventral: 53.66 → 56.35;
- traits 2/3: 62.8 → 70.7;
- traits 6/9: 43.6 → 47.8;
- traits 5/8: 22.9 → 24.1, within the ±2 band;
- dorsal: unchanged (80.99 → 81.01).

**Against the bands:**
- **Picks, 5-shot Major (+0.05 to +0.25):** met for both MREF picks (+0.16, +0.15). Missed for `crown_all` (+0.02,
  non-significant).
- **(m×p)×m (+0.3 to +1.0):** met for both MREF picks (+0.88, +0.83). Missed for `crown_all` (+0.13,
  non-significant).
- **Falsifier (≤ 0 on (m×p)×m):** not triggered.
- **1-shot:** every effect non-significant with |Δ| ≤ 0.05, as predicted.
- **Ventral traits:** 2/3 up and 5/8 flat, as predicted.

**Not predicted:** `crown_all`, the single-reference refit, gains nothing, while both MREF refits and base gain.
The S21 SAM 2 run tested only the MREF baselines and base, so this is new. A possible reading is that adapters trained
on five conditioned frames use a cleaner support set better. That is a hypothesis, not tested.

**Correctness band (untouched subspecies "exactly 0"): not bit-exact on SAM 3.**
- In cyrbia, lativitta and notabilis × lativitta the draws and adapters are identical, yet **all 60 runs differ in
  every cell**.
- Per-run max |Δ| is 0.04–0.16 pp for base and the MREF picks, and 0.66–0.71 pp for `crown_all`.
- The pooled untouched-group delta is ≤ 0.02 in every cell; base at 1-shot is +0.01 [+0.00, +0.02] ✱.
- SAM 3 inference (bf16) is therefore not bit-reproducible across allocations, unlike SAM 2 (bit-exact in 8 of 10
  cells in S21). The noise is about 8× smaller than the 5-shot POOL effects but is a real floor for cross-allocation
  SAM 3 comparisons. S20's within-allocation reads are not affected.

### SAM 3 vs SAM 2 under the same pool (identical draws, asserted per group and run)

| Adapter | 1-shot SAM 2 → SAM 3 | 5-shot SAM 2 → SAM 3 |
| :-- | :-- | :-- |
| base | 75.24 → 76.51, +1.26 [+0.63, +2.03] ✱ | 77.26 → 78.07, +0.80 [+0.70, +0.91] ✱ |
| crown_all_mref | 77.31 → 77.42, +0.11 [−0.35, +0.73] | 78.75 → 78.35, **−0.40 [−0.48, −0.31] ✱** |
| odd_cf_all_mref | 77.08 → 77.31, +0.22 [−0.32, +0.96] | 78.55 → 78.34, −0.21 [−0.31, −0.11] ✱ |

The band (≤ 0 for the picks at 5-shot) is met. POOL lifts both families by a similar amount, so S20's ordering stands:
**butterfly 5-shot still belongs to SAM 2** (`crown_all_mref` + POOL, 78.75). At 1-shot the families tie.

### Butterfly stack after S22 (argmax, TTA)

- **Best 5-shot:** SAM 2 `crown_all_mref` + POOL, **78.75**. Next is SAM 3 `crown_all_mref` + POOL at 78.35.
- **Best 1-shot:** SAM 3 `crown_all_mref` + POOL at 77.42, a non-significant tie with its curated 77.38 and with
  SAM 2 + POOL at 77.31.
- **Paper (OC-CCL rows):** 83.9 at 5-shot and 81.2 at 1-shot.

### Reading

- **POOL transfers to SAM 3 for base and the MREF adapters,** at the same size and in the same place as on SAM 2.
  The (m×p)×m ventral traits 2/3 and 6/9 rise; the ventral 5/8 streak does not.
- **It is a support-set effect, not a model effect.** That is why it adds to both families about equally.
- **The remaining (m×p)×m gap is the 5/8 streak** (the deferred loss idea) plus the label-convention problems that no
  pure-OC-CCL lever reaches.

## Landed result — 2026-10-05 (gates `s22_screen` PASS, `s22` PASS; first run, no fixes needed)

Per support seed vs the locked S20 curated cell; ✱ = CI excludes 0. Full tables: [S22_RESULTS.md](S22_RESULTS.md).

| Adapter (SAM 3) | 5-shot curated → pooled | POOL Δ Major | POOL Δ (m×p)×m | 1-shot POOL Δ Major |
| :-- | :-- | :-- | :-- | :-- |
| base | 77.96 → 78.07 | +0.11 [+0.03, +0.19] ✱ | +0.71 [+0.35, +1.09] ✱ | +0.01 [−0.22, +0.23] |
| crown_all | 78.00 → 78.02 | +0.02 [−0.10, +0.16] | +0.13 [−0.45, +0.77] | +0.03 [−0.24, +0.30] |
| crown_all_mref | 78.20 → **78.35** | **+0.16 [+0.07, +0.25] ✱** | **+0.88 [+0.46, +1.30] ✱** | +0.04 [−0.19, +0.25] |
| odd_cf_all_mref | 78.19 → 78.34 | +0.15 [+0.07, +0.23] ✱ | +0.83 [+0.46, +1.21] ✱ | +0.05 [−0.18, +0.28] |

(m×p)×m ventral queries at 5-shot, curated → pooled:
- `crown_all_mref`: 53.66 → 56.35.
- Ventral traits 2/3: 62.8 → 70.7.
- Ventral traits 6/9: 43.6 → 47.8.
- Ventral traits 5/8: 22.9 → 24.1 (the costal streak, unmoved as predicted).
- Dorsal: 80.99 → 81.01.

**SAM 3 − SAM 2 under the same pool (identical draws, both on A40):**
- base: +1.26 ✱ at 1-shot, +0.80 ✱ at 5-shot.
- `crown_all_mref`: +0.11 at 1-shot, **−0.40 ✱** at 5-shot.
- `odd_cf_all_mref`: +0.22 at 1-shot, −0.21 ✱ at 5-shot.

### Against the pre-registered bands

| Read | Band | Result |
| :-- | :-- | :-- |
| POOL Major, picks, 5-shot | +0.05 to +0.25 | **met** for both MREF picks (+0.16, +0.15 ✱); **missed** for `crown_all` (+0.02, non-significant) |
| POOL (m×p)×m, picks, 5-shot | +0.3 to +1.0 | **met** for both MREF picks (+0.88, +0.83 ✱); **missed** for `crown_all` (+0.13, non-significant) |
| Ventral traits 2/3, 5-shot | up | met: up in all four adapters (+4.0 to +7.9) |
| Ventral traits 5/8 | unchanged ±2 | met (+0.3 to +1.2) |
| 1-shot | non-significant, \|Δ\| ≤ 0.25 | met (+0.01 to +0.05, all non-significant) |
| Untouched groups | 0, identical draws | draws identical; **scores not bit-identical** (see below); mean shift ≤ 0.024 |
| SAM 3 pooled − SAM 2 pooled, picks, 5-shot | ≤ 0 | met (−0.40 ✱, −0.21 ✱) |
| Falsifier: (m×p)×m POOL ≤ 0 on the picks at 5-shot | | not met. The point estimate is positive for all three picks; `crown_all`'s CI crosses 0 |

### Reproduction noise across hardware (measured; disclosed)

- **What differs:** S20's curated cells ran on L40S; S22's pooled cells on A40, with the same torch 2.11 / CUDA 12.8,
  checkpoint and bf16.
- **What the untouched subspecies show:** cyrbia, lativitta and notabilis × lativitta draw identical supports, yet
  every run differs.
- **How large:** per run up to 0.16 (base), 0.71 (`crown_all`) and 0.06 (the MREF picks); mean shift +0.009 / −0.004
  (base), +0.010 / +0.024 (`crown_all`), and ≤ 0.003 for the MREF picks.
- **What it means for the reads:** this is the hardware floor under every S22 POOL read. It is about 1/50 of the MREF
  picks' POOL effect, but comparable to `crown_all`'s (+0.02), which is therefore indistinguishable from noise.
- **Cause:** plausibly bf16 kernel selection on Ada (L40S) vs Ampere (A40). Not established beyond the GPU-type
  difference.

### Reading

- **POOL transfers to SAM 3 almost unchanged.**

  | | SAM 3, S22 | SAM 2, S21 |
  | :-- | :-- | :-- |
  | Major, MREF picks, 5-shot | +0.15 to +0.16 ✱ | +0.12 to +0.19 ✱ |
  | (m×p)×m, MREF picks, 5-shot | +0.83 to +0.88 ✱ | +0.55 to +0.89 ✱ |
  | Ventral traits 2/3, `crown_all_mref` | 62.8 → 70.7 | 63.0 → 71.0 |

  The lever is a property of the supports' labels, not of the tracker.
- **`crown_all` barely moves on SAM 3**, while base and the MREF picks do. One untested reading: the MREF arms were
  trained with five conditioned frames, the 5-shot regime, so they make more use of a cleaner 5-shot support set. S21
  had no `crown_all` POOL cell to compare on SAM 2.
- **The butterfly best stays SAM 2:** `crown_all_mref` + POOL at **78.75**. On SAM 3 the best is `crown_all_mref` +
  POOL at 78.35, and SAM 3 still trails SAM 2 at 5-shot by the same margin S20 found (−0.40 ✱ vs −0.24 … −0.53 ✱).
- **The paper's 83.9 is 5.15 points away.** The parts identified so far:
  - (m×p)×m ventral traits 5/8 and 6/9 (no lever under pure OC-CCL);
  - lativitta's test-only ray convention (report only);
  - the residual (m×p)×m ventral deficit (dorsal 81 vs ventral 56).
