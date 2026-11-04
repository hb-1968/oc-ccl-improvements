# S18c — combining `crown_all` and the counterfactual centre target (pre-registered 2026-09-23)

Status: **landed 2026-09-23** (see "Landed result" at the end). Parent: the landed S18b root ([PLAN.md](PLAN.md) "Landed
result", both gates PASS). Nothing here retrains or rewrites the parent; the S18b root is
read-only input. The established recipe `sam2_crown` stands until a gate passes.

## Why

S18b left `odd_cf_all` 0.14 pp [−0.22, −0.04] below its matched control `crown_all` on the
butterfly test (TTA, 20 paired seeds) while halving removed-trait cut-out occupancy on
validation (31.5 % → 19.0 %), and showed that the centre target alone carries that
property: `odd_cycle` (odd return, no target) equals crown at 34.9 %, `centre_aux` (target,
no odd return) reaches 18.0 %, and the judge term adds nothing. `odd_cf_all` **is**
crown_all + 0.25 × odd + 0.25 × centre on the same data, seeds and adapter, so "combining"
the two means choosing where on that line to sit. S18c measures the line two cheap ways.

## Arms (`occl.s18c_contract.S18C_ARMS`)

| Arm | Construction | Training |
| :-- | :-- | :-- |
| `soup_25` | ΔW = 0.25 ΔW_crown_all + 0.75 ΔW_odd_cf_all | none |
| `soup_50` | 0.50 / 0.50 | none |
| `soup_75` | 0.75 / 0.25 | none |
| `centre_all_w05` | `centre_aux`, all legs, centre weight 0.05 | one epoch, S18b fit split |
| `centre_all_w10` | same, centre weight 0.10 | same |
| `centre_all_w25` | same, centre weight 0.25 (= `centre_aux` on all legs) | same |

**Soups are exact.** Both parents are rank-32 / alpha-64 LoRAs on identical modules
(`SAM2_ARCH_SCOPES`), so ΔW_i = 2 · B_i A_i. The mix is written as one rank-64 adapter with
A = [A₁; A₂], B = [α B₁, (1−α) B₂], r = 64, alpha = 128 (scale 128/64 = 2 unchanged), which
the standard loader (`load_finetune`, reading r/alpha/scopes from the file's own metadata)
reconstructs as α ΔW₁ + (1−α) ΔW₂ module by module; `tests/test_s18b.py` pins this through
the real loader. Provenance: both parent adapter hashes and α travel in `training.json`
and in the adapter metadata (`occl.s18c`), beside the S18b block the registry already
treats as `modified`.

**The sweep** reuses the S18b worker (`run_s18b.py --parent-root`): same pairs, seeds,
optimizer, checkpointing and runaway alarm; `centre_weight` is a per-arm override of the
contract default in `s18b.step` (existing arms carry no override, so their numerical step
is unchanged). Odd return and judges are dropped (S18b: null). The S18c identity hashes
the S18b identity into the usual (manifest, source, checkpoint, torch, dtype) payload;
parent rows and adapters are verified against the parent's recorded identity, never ours.

## Validation, selection, confirmation

Each arm runs the unchanged S18b held-out validation (ordinary 1-shot mIoU, cut-out soft
occupancy, translation, diagnostic odd read) on the parent's calibration rows.

Pre-registered eligibility versus the **parent `crown_all` validation record**
(point-estimate screening floors, as in S18b): mean mIoU ≥ crown_all − 0.5 pt; every group
≥ −2 pt; translation IoU ≥ −2 pt; cut-out occupancy ≤ crown_all − 5 pt. Rank eligible arms
by validation mean, then lower occupancy, then name; record a rule winner or none.

Confirmation runs **every** S18c arm on the test set — TTA, shot 1, 20 paired support
seeds, the standard predictor — together with `base` and the parent `crown_all`, as sliced
`run_arms` fans (all eight arms on one image encode per slice) banked in the stage run
bank, where S18b's base and crown_all runs already sit and cost nothing. The read is each
arm's paired delta versus `crown_all_tta` (10,000 resamples, seed 0). An arm **combines**
the two only if it was eligible on validation AND that CI includes zero. Deltas versus
`base_tta` are reported for scale. Intervals are unadjusted across six arms and two
datasets; one fit per arm; no independent training replicates.

What would falsify the idea: every eligible arm significantly below crown_all on test
(the hallucination property is priced above the noise floor), or no arm eligible (the
property vanishes before the tracking cost does). Either is a valid outcome and closes
the line; S18b's result stands as the record either way.

## Execution

`scripts/run_s18c_stage.py` reuses the S18b graph runner: per dataset `plan` (binds the
parent, copies its manifest) → three `soup` jobs and three `train` jobs, all depending only
on the plan → per-arm `validate` → `select` (after every validation ends) → `gate` (the
S18c evidence contract from the work root) → the confirmation fan joins the graph. Bundle
`results/s18c.zip` (records, six adapters, test cells); gates `check_gate.py s18c_screen`
and `s18c`; read `scripts/report_s18c.py` → `S18C_RESULTS.md`. Batch: the S18b qsub with
`OCCL_STAGE=s18c`, i.e. `scc_submit_s18b.sh all 6 1 s18c` (sizing from S18b's measured
timings: 3 butterfly fits ≈ 13 min, 12 validations ≈ 4 min, 30 fan slices carrying six new
arms ≈ 20–25 min each on four cards ≈ 2.5 h; ≈ 3–4 h in all).

## Landed result (2026-09-23)

Second submission (`all 6 1 s18c`, root `s18c_work_v2`; the first run died on a shared
temp-file race in `atomic_json`, fixed and reproduced in `tests/test_s18b.py`): 76 jobs,
all complete, **3.13 h of wall** on 4 × L40S (fits 0.7 / 8.6 min, validations ≤ 3.9 min,
fan slices 21.5 min mean / 29.1 max carrying eight arms, merges ≤ 2 min). Both gates PASS
([s18c_gate.txt](s18c_gate.txt)); the confirm gate first read FAIL on a gate-side bug
(it demanded an `occl.s18c` block the merge record does not carry; it now pins each cell to
its training record by `occl.s18b.arm`, the manifest, and the loaded LoRA's delta norm
and rank, and `run_eval` passes `s18c` through for future merges). Read:
[S18C_RESULTS.md](S18C_RESULTS.md).

**Butterfly.** The line between crown_all and odd_cf_all is smooth and monotone on both
axes: more centre target → lower removed-trait occupancy → larger test cost.

| Arm | Val cut-out occupancy | Test TTA | Δ vs crown_all [95 % CI] | Eligible |
| :-- | --: | --: | :-- | :-- |
| crown_all (parent) | 31.5 % | 77.16 | — | — |
| soup_75 | 29.7 % | 77.13 | −0.03 [−0.10, +0.05] | no (−1.8 pt) |
| soup_50 | 27.8 % | 77.11 | −0.05 [−0.12, +0.04] | no (−3.7 pt) |
| centre_all_w05 | 28.3 % | 77.07 | −0.09 [−0.17, −0.00] | no (−3.2 pt) |
| centre_all_w10 | 25.4 % | 77.04 | −0.12 [−0.20, −0.03] | yes (−6.1 pt) |
| soup_25 | 24.1 % | 77.03 | −0.13 [−0.23, −0.02] | yes (−7.4 pt) |
| centre_all_w25 | 18.4 % | 77.00 | −0.15 [−0.22, −0.08] | yes (−13.0 pt) |

No arm is both eligible and cost-free: the two soups whose CI includes zero do not reach
the 5-point occupancy margin, and every arm that reaches it pays a significant tracking
cost. No soup beats either endpoint (interpolation, not a soup gain). The property is
priced at roughly 0.01–0.02 pp of test mIoU per point of occupancy removed, and
`centre_all_w25` reproduces `odd_cf_all` (−0.15 vs −0.14) without the odd return.

**Beetle.** No arm eligible: every arm sits at 30.8–30.9 % occupancy against crown_all's
31.0 %, so the centre target does not change beetle's removed-trait behaviour at all.
Test deltas vs crown_all are within noise (−0.12 … +0.15; `centre_all_w10` −0.11
[−0.21, −0.02] is one of twelve unadjusted intervals).

**Disposition.** The pre-registered question — is there a point on the line that keeps
crown_all's tracking and the hallucination reduction — is answered **no** on both
datasets. `sam2_crown` stands; `crown_all` remains the best SAM 2 butterfly recipe
(+1.98 ✱ vs base, now replicated in three stages). If the hallucination property is
wanted, `centre_all_w25` is the cheapest way to buy it, at −0.15 pp. This closes the
counterfactual-centre line; S18b's result stands as the record.
