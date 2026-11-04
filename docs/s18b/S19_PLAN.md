# S19 — the S18b wings at five-shot inference (pre-registered 2026-09-23)

Status: **landed 2026-09-24** (gates PASS 2/2; see "Landed result" at the end). Pre-registration below is unchanged. Parent: the landed S18b root ([PLAN.md](PLAN.md) "Landed result",
both gates PASS; [S18B_RESULTS.md](S18B_RESULTS.md)). Nothing here trains, calibrates,
validates or selects; the S18b root is read-only input. `sam2_crown` stands until a gate
passes and the user amends the rule.

## Why

Every SAM 2 cycle adapter this project has produced was scored at 1-shot. The only 5-shot
read of a cycle adapter is S17c/S17d's transfer of the **forward-only** `comp`/`crown`
adapters on beetle (−0.19 / +0.19 pp, ns: "5-shot transfer null"). The all-leg adapters
never had that read. S18b/S18c left the two all-leg wings near-equivalent at 1-shot:

| 1-shot, TTA, 20 paired seeds | butterfly_major | beetle |
| :-- | :-- | :-- |
| base_tta | 75.18 | 63.86 |
| crown_all_tta | 77.16 (+1.98 [+1.77, +2.21] vs base) | 64.94 (+1.08 [+0.50, +1.73]) |
| odd_cf_all_tta | 77.02 (−0.14 [−0.22, −0.04] vs crown_all) | 64.86 (−0.08 [−0.21, +0.05]) |
| removed-trait cut-out occupancy (validation) | 31.5 % → 19.0 % | 31.0 % → 30.8 % |

The user's question: **how does the fine-tune's effect change when inference moves from
one support to five, and does the near-equivalence of the two wings survive?** The
adapters modify memory encoding, memory attention and the mask decoder; at 5-shot the
memory bank holds five conditioned frames instead of one, so a gain (or a cost) learned
against a single reference can grow, hold or vanish. Three outcomes are distinguishable
with the paired design below.

## Design

**Arms (`occl.s19_contract`).** `WINGS = ('crown_all', 'odd_cf_all')`, `CONFIRM_ARMS =
('base', *WINGS)`, `SHOT = 5`, `CONFIRM_MODES = ('tta',)`. The adapters are the S18b files
themselves — `<parent>/<dataset>/{crown_all,odd_cf_all}/lora.pt`, hash-pinned to the
parent training records — so the 1-shot and 5-shot cells of a wing share the same bytes.
No plain (non-TTA) cells: TTA over plain is settled (S16, S18a/b), and at 5-shot the
butterfly TTA gain on SAM 3 is +5.65 (S15 base 72.30 → base_tta 77.95).

**Protocol.** 5 supports, 20 support seeds, the per-dataset canonical selection the 1-shot
cells used (`canonical_placement` butterfly, `canonical_balanced` beetle — the same
`SUPPORT_SELECTION` table the S18b fans use), the standard predictor, TTA over support
orientation. `seeding.sample_supports` / `curation.stratified_supports` slice `[:shot]`
from one seeded permutation per (dataset, group, run), so run *i*'s 1-shot support is the
first element of its 5-shot draw. That is the licence for the cross-shot pairing and it is
**verified from the records, never assumed** (`registry.nested_shot_evidence`, S16's
rule, re-earned by S15 at 5 shots).

**Cells.** `base_5_tta`, `crown_all_5_tta`, `odd_cf_all_5_tta` per dataset (S17c's
naming), next to the parent's `base_tta`, `crown_all_tta`, `odd_cf_all_tta` (copied into
the S19 record from the parent's receipted test zips; not re-run).

**Pre-registered reads** (`s19_contract.reads`; every one paired per support seed, 10,000
resamples, seed 0; unadjusted across reads and datasets):

1. *At 5-shot:* each wing − base_5_tta; odd_cf_all_5_tta − crown_all_5_tta. Ordinary
   within-protocol pairing on identical 5-shot draws.
2. *Shot lift per arm:* arm_5_tta − arm_tta through `compare(nested_shots=True)`, on the
   same adapter bytes (delta-norm pin, rel 1e-4, to the parent training record and to the
   1-shot cell).
3. *Interaction* (`s19_contract.interaction`): per seed, (wing_5 − base_5) − (wing_1 −
   base_1), paired bootstrap. Positive = the fine-tune's gain grows with supports; zero =
   it holds; negative = it shrinks. The same difference-in-differences between the wings,
   (odd_cf_all_5 − crown_all_5) − (odd_cf_all_1 − crown_all_1), asks whether odd_cf_all's
   1-shot cost is preserved, amplified, or erased by five supports.

**Priors (bands, written before the run; not gates).** SAM 2 5-shot references are
unpaired and old: S6 beetle base 5-shot 73.16 (S17c's pin), S8d Major `occl_tta` 78.07
(summary-only). S15's SAM 3 shot lifts were +10.95 ✱ beetle (arch), +1.21 / +0.81 ✱
butterfly. Expect base_5_tta ≈ 73–75 beetle, ≈ 77–78 butterfly; crown_all − base at
5-shot anywhere in −0.5 … +2.0 — the forward-only null (S17c) predicts ≈ 0, the all-leg
1-shot margin predicts it holds; odd_cf_all − crown_all at 5-shot −0.5 … +0.2. The
interaction CIs are the read; no sign is promised.

**What would falsify what.** Interaction for crown_all significantly negative and the
5-shot gain's CI including zero → the all-leg gain is a 1-shot property (as the
forward-only one was) and SAM 2 5-shot belongs to the base. Interaction ≈ 0 with the
5-shot gain ✱ → the fine-tune transfers; crown_all becomes the SAM 2 5-shot candidate
(a recipe change still needs the user's rule amended). Wing difference-in-differences
≈ 0 → the wings stay equivalent; significantly positive → the centre target's 1-shot cost
was a single-reference artefact; negative → five references amplify it.

**Gates** (`check_gate.py s19_screen`, `s19`; `occl.s19_contract.evidence_check`):
screen = the parent binding (S18b identity, both wing adapters complete and hash-pinned,
the three 1-shot cells receipted, 20 × groups runs each, paired draws, adapter provenance
by `occl.s18b.arm` + manifest + delta norm) and the frozen plan reproduced; confirm = the
three 5-shot cells complete (shots 5, 20 runs, five support ids each, same selection as
the 1-shot cells, paired draws), the same adapter bytes on both shot counts, prefix
nesting verified per arm, and every read computable. Neither gate demands a sign.

## Execution

`scripts/run_s19_stage.py` (reuses the S18b graph runner, fan/merge builders and bundle
shape; `confirm_graph` gained a `shot` argument, default 1, so S18b/S18c graphs are
unchanged): per dataset `plan` in the driver process before any GPU work (binds the
parent, copies its manifest, verifies the dataset cache still reproduces it, writes
`parent.json` and the frozen `selection.json`; identity = digest of the usual (manifest,
source, checkpoint, torch, dtype) payload + parent identity + stage `s19` + shot 5), then
the screen gate on the on-disk record, then the sliced 5-shot TTA fans (`run_arms.py
--shot 5`, base + both wings on one image encode per slice, banked in the shared run bank
under the 5-shot key — nothing from the 1-shot bank is reused or overwritten) and one
merge per cell (`run_eval.py --shot 5 --no-precompute`). Bundle `results/s19.zip`
(record with the 5-shot cells and the parent's 1-shot cell records; the 5-shot test
zips). Read: `scripts/report_s19.py` → [S19_RESULTS.md](S19_RESULTS.md).

Batch: the S18b qsub with `OCCL_STAGE=s19`, root `results/s19_work_v1`, BUILD ≥ 23:

```bash
ssh hbridges@scc1.bu.edu 'bash /projectnb/bu-mlb/hbridges/sst/occl/oc-ccl-improvements/scripts/scc_submit_s18b.sh all 8 1 s19'
```

Sizing (estimates, not guarantees): the butterfly 5-shot TTA fan is the critical path —
25 slices (5 groups × 5 slices of 4 runs) carrying three arms and six frames per episode. S15
measured 21–24 min per 5-shot TTA butterfly slice with two arms; S18c measured 21.5 min
mean / 29.1 max per 1-shot slice carrying eight arms; so ≈ 30–40 min per slice here,
≈ 3.5–4.5 h on four cards, plus 5 beetle slices (≈ 12 min) and 6 merges (≤ 2 min): ≈ 4–5 h
of wall. One 8 h allocation (driver budget 455 min) covers it; `all 8 2 s19` adds a
chained resume as insurance (the second job exits at once if the first finished). The
driver's stall limit is 40 min (a 5-shot butterfly TTA slice logs per run, ≈ 8 min apart).

Local round trip performed 2026-09-23: a synthetic record (prefix-nested draws, pinned
adapters, known gains) passes `s19_screen` and `s19`, and `report_s19.py` recovers the
planted shot lifts and interactions exactly (`tests/test_s19.py`).

## Limitations

One adapter per wing per dataset (no training replicates; the 1-shot cells were fitted
once). Twelve intervals across two datasets, unadjusted. The 5-shot base references from
S6/S8d are unpaired and were produced by older code paths; the paired reads inside S19 do
not depend on them. Beetle's 5-shot pool is small (79 canonical images), so five supports
overlap heavily across seeds and the paired resolution there is lower than on butterfly.

## Landed result (2026-09-24)

One batch job (`all 8 1 s19`, root `s19_work_v1`): 36 jobs, all complete, **4.07 h of wall**
on 4 × L40S (butterfly 5-shot TTA slices 35.9 min mean / 45.5 max, beetle 10.5 min, merges
≤ 2.1 min — inside the 30–40 min per-slice estimate). Both gates PASS
([s19_gate.txt](s19_gate.txt)); prefix nesting verified on every (group, run) pair; the
same adapter bytes on both shot counts. Read: [S19_RESULTS.md](S19_RESULTS.md). Every
prior band held (base_5_tta beetle 75.15 sits just above the 73–75 band).

| TTA, 20 paired seeds | butterfly 1-shot → 5-shot | beetle 1-shot → 5-shot |
| :-- | :-- | :-- |
| base | 75.18 → **77.23** (lift +2.04 ✱) | 63.86 → **75.15** (lift +11.29 ✱) |
| crown_all | 77.16 → **78.53** (lift +1.37 ✱) | 64.94 → **75.50** (lift +10.56 ✱) |
| odd_cf_all | 77.02 → **78.46** (lift +1.45 ✱) | 64.86 → **75.47** (lift +10.61 ✱) |
| crown_all − base | +1.98 ✱ → **+1.30 [+1.21, +1.39] ✱** | +1.08 ✱ → **+0.35 [−0.05, +0.74]** |
| interaction (5-shot gain − 1-shot gain) | **−0.68 [−0.92, −0.45] ✱** | −0.73 [−1.47, +0.03] |
| odd_cf_all − crown_all | −0.14 ✱ → **−0.06 [−0.11, −0.02] ✱** | −0.08 → −0.03 [−0.16, +0.11] |
| wing interaction | +0.08 [−0.02, +0.17] | +0.05 [−0.11, +0.19] |

**Butterfly: the all-leg gain transfers, shrunk by a third.** Five supports lift base more
than they lift the adapters (+2.04 vs +1.37/+1.45), so the fine-tune's margin falls from
+1.98 to +1.30 — a significant negative interaction — but the 5-shot margin itself is
sharp (paired std ≈ 0.2) and far from zero. `crown_all_5_tta` **78.53** is the highest
butterfly number in the programme: +0.46 over S8d's summary-only Major 78.07 (SAM 2, old
code path, unpaired) and +0.58 over S15's SAM 3 5-shot `base_tta` 77.95 (different family,
unpaired; SAM 3's own arch adapter cost −0.48 ✱ there). Neither pre-registered corner
exactly: not "holds" (interaction ≠ 0) and not "1-shot property" (5-shot gain ✱).

**Beetle: the all-leg gain is a 1-shot property.** The 5-shot margin +0.35 [−0.05, +0.74]
includes zero and the interaction is borderline (−0.73 [−1.47, +0.03]): the +1.08 ✱ 1-shot
gain does not survive five supports, matching S17c/S17d's forward-only 5-shot null. SAM 2
5-shot beetle sits at 75.15–75.50, 3.4 below the SAM 3 5-shot champion 78.91 (unpaired).

**The wings stay equivalent.** odd_cf_all's 1-shot cost halves on butterfly (−0.14 → −0.06,
still ✱ on the tight 5-shot pairing) and is null on beetle; both wing interactions include
zero. Five references neither erase nor amplify the centre target's cost; it is not a
single-reference artefact, just small.

**Disposition.** No recipe promotion (the user's rule stands: `sam2_crown` forward-only
default). For SAM 2 at 5-shot the all-leg adapter is worth +1.30 ✱ on butterfly and nothing
measurable on beetle; the SAM 2 5-shot butterfly candidate, if one is wanted, is
`crown_all`. The open levers remain outside SAM 2: the all-leg port to SAM 3 at 1-shot
(control: beetle arch_tta 69.88 / 70.21) and at 5-shot (where arch cost butterfly −0.48 ✱
while helping beetle +0.97), and a second support.
