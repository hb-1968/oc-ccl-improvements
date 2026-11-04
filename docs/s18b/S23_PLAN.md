> Status update, 2026-11-04: [S23 results](S23_RESULTS.md) record completed
> screen and confirmation gates. The preregistration below retains its original
> pre-run status and decisions. Gate completion does not imply improvement.

# S23 — thin-structure shape terms + ventral-reference oversampling (SAM 2, butterfly, 1-/5-shot)

Status: **pre-registered 2026-10-05; implemented and unit-tested; not run.** A SAM 3 port follows only for a lever
that clears its band.

- Contract: `src/occl/s23_contract.py`.
- Loss: `src/occl/finetune/shape.py`.
- Driver: `scripts/run_s23_stage.py`.
- Gates: `s23_screen`, `s23`.
- Report: `scripts/report_s23.py` → `docs/s18b/S23_RESULTS.md`.

## Why

After S21 and S22, the (m×p)×m ventral deficit sits in two structures that no support choice reaches:

| Ventral structure | IoU on SAM 2 and SAM 3, with or without POOL |
| :-- | --: |
| hindwing-costal streak (traits 5/8) | ≈ 24 |
| basal spots (traits 6/9) | ≈ 36–48 |

The CPU probe ([MXP_DIAGNOSIS.md](MXP_DIAGNOSIS.md) §4; the probe JSON's area ratios) shows how the streak fails:
- from dorsal supports it is **over-painted**, at 2.6–4.1× its area;
- from ventral supports it is **misplaced**: streak-sized, about 1.5× the area, but IoU 25 against an area-only
  ceiling of about 60;
- it is **never missed** (0 of 32 empty predictions).

The user asked (2026-10-05) for the queued stage: loss modifications that target such traits.

**Pure OC-CCL stands** (user decision, 2026-10-02). Every new term acts on the cycle's **closing frame**, against
the **reference's own labels**.

**Honest prior: null.**
- S21's CLEAN (clean anchors) did not move the underside.
- S17 found the closing head absorbs closing-frame objectives.
- Only 13 of 870 fit references are label-consistent ventral masks.

## Levers (each alone, plus the stack)

All three arms are refits of S21's `crown_all_mref_lc` recipe, on S21's CLEAN manifest (`7573573b904a`) with S21's
calibration rows:
- MREF, refs = 5, all legs;
- objectness 10, excl zerocat, arch scopes;
- lr 1e-5, one epoch.

The parent is `crown_all_mref_lc` itself.

1. **SHAPE** (`crown_all_mref_lc_shape`): two terms on the closing frame, added to the anchor with weights
   `SHAPE = {boundary: 1.0, tversky: 1.0}`:
   - **Boundary.** The mean outside-distance of the predicted probability mass:
     Σ φ⁺·σ(z) / (Σ σ(z) + 1), with φ⁺ the distance outside the reference outline in units of 32 px, clipped
     at 8 (256 px). It punishes misplacement in proportion to distance and is bounded in [0, 8] for every trait
     size. **Deviation from Kervadec et al. 2019, decided before any run:** the original signed, area-normalised
     sum exploded for thin traits on real masks (4,100 on a ventral (m×p)×m reference against 684 on a dorsal one,
     at σ = 0.5 everywhere), so it would have dominated the loss and the gradient clip. Measured on real 1024-px
     masks, the bounded form gives:

     | Prediction | Ventral reference | Dorsal reference |
     | :-- | --: | --: |
     | exact | 0.31 | 0.06 |
     | uninformative | 7.0 | 6.9 |

   - **Tversky.** 1 − (TP+1) / (TP + 0.7·FP + 0.3·FN + 1): over-painting costs more than missing.
   - **Thin weighting.** Per channel, w = clip(40 px / width, 1, 4), where width = 2 × the 90th-percentile inside
     distance at 1024 px. Measured on a ventral (m×p)×m reference: the streak channels get 3.6 and 2.2, the spots
     4.0 and 1.7, the rays 4.0, and the bands and patches 1.0.
   - **Cost:** about 180 ms of CPU per step (distance transforms at 1024²).
2. **OVS** (`crown_all_mref_lc_ovs`): oversampling by **repetition**.
   - Every fit reference whose mask is a complete thin-streak ventral annotation (`conventions.ventral_strip`)
     gets `k = 7` extra cycle pairs: 12 in (m×p)×m and 1 in malleti, so 13 references and 91 extra steps
     (870 → 961).
   - Each extra pair uses a seeded same-group fit query and is inserted at a seeded position
     (`s23_contract.ovs_schedule`). It copies its reference's manifest row.
   - Repetition, not loss weighting: 21 % of the parent's steps already hit the gradient clip (max-norm 1), which
     would blunt a weight.
   - Anchor mode only: the extra pairs have no frozen-teacher calibration rows.
3. **SHAPE + OVS** (`crown_all_mref_lc_shape_ovs`): both.

All constants were fixed before any S23 fit, from the diagnosis' measured widths and the parent's gradient
statistics; none was tuned.

## Cells and reads

- **Test cells:** the three arms × {1-shot, 5-shot}, TTA, 20 seeds, `canonical_placement` +
  `--support-pool label_consistent` (the standing best support protocol): **6 cells**.
  - Each pairs with S21's `crown_all_mref_lc_lcpool[_5]_tta` on **identical draws** (same pool, same seeds), so
    `registry.compare` applies.
  - Each cell's GPU type is recorded (`hardware`). Every arm-vs-parent read reports `hardware_matched`, the S22
    lesson: L40S vs A40 bf16 is not bit-reproducible.
- **Memorization probe:** the three arms' 5-shot curated cells (runs 0–4), paired per episode with S21's
  `crown_all_mref_lc` probe cell.
- **Reads** (`s23_contract.reads`, `report_s23.py`):
  - each arm vs the parent: Major, (m×p)×m, malleti, lativitta, best-of-3, and vs base;
  - stack additivity;
  - **the (m×p)×m ventral side by trait group, paired per support seed** from `per_query.csv`: 5/8 streak,
    6/9 spots, 2/3 band, 7/10 rays, 1/4 patch;
  - **lativitta rays 7/8**, the harm monitor;
  - per subspecies;
  - the probe.
- **CIs:** 10,000 resamples over support seeds, seed 0, unadjusted. Many reads, so read the bands, not single stars.

## Pre-registered bands (never gates)

| Read | Band / criterion |
| :-- | :-- |
| **PRIMARY: (m×p)×m ventral traits 5/8, 5-shot, SHAPE or stack vs parent** | success = **≥ +2.0 ✱**; prior: null (\|Δ\| < 1). **Falsifier of the premise: ≤ 0** |
| (m×p)×m ventral traits 6/9, 5-shot | same criterion (secondary target) |
| Major vs parent, 5-shot | −0.10 to +0.15 |
| (m×p)×m vs parent, 5-shot | −0.3 to +0.5 |
| **lativitta rays 7/8 under SHAPE (harm monitor)** | −0.5 to −3. Tversky's false-positive weight tightens masks, and lativitta's TEST rays are loose envelopes ([LATIVITTA_DIAGNOSIS.md](LATIVITTA_DIAGNOSIS.md)) |
| lativitta rays under OVS | ±0.5 |
| cyrbia, notabilis × lativitta | ±0.3 |
| (m×p)×m ventral (all traits) under OVS | 0 to +1 |
| 1-shot reads | non-significant |
| Stack additivity | reported, no band |
| Cut-out occupancy vs parent | within ±3 pp |
| Hardware | matched to the parent cells (A40); a mismatch is disclosed, not gated |

## Gates

`s23_screen` (after the fits) and `s23` (everything), both `s23_contract.evidence_check`:

- **Parent:** the S21 parent is bound (CLEAN manifest, calibration, parent fit and validation, pooled parent and
  base cells, parent probe).
- **Training:** each arm's history reproduces its schedule. For OVS this is re-derived from the recorded strip
  ids, k and seed, with the strip ids required to be fit ids of the filtered groups. The shape terms must be
  recorded every step; the extra references must sit inside the group's CLEAN fit split; the lever metadata must
  match the arm.
- **Validation and selection:** validations cover the CLEAN split; eligibility vs the parent is reproduced and
  recorded, not enforced.
- **Cells:** the 6 pooled cells sit on the parent's draws and are pinned by ‖ΔW‖; the 3 probes cover the parent
  probe's episodes; every read computes.

A pass is completion, not a win. No selection and no promotion: `sam2_crown` stands.

## Execution

```bash
ssh hbridges@scc1.bu.edu 'bash /projectnb/bu-mlb/hbridges/sst/occl/oc-ccl-improvements/scripts/scc_submit_s18b.sh all 48 1 s23 2'
```

- **Inputs:** the S21 root (read-only), `data/occl-cache-delta.zip` and the BUILD 28 code layer. 40 GB per GPU.
- **Wall-time estimate:** three fits of about 13–15 min each, run in parallel with validations, then 3 arms ×
  2 shots of pooled fans and the 5-shot probe: about 6–9 h on four cards.
- **Pull, gate and read:**

```bash
mkdir -p results/inbox && ssh hbridges@scc1.bu.edu 'cd /projectnb/bu-mlb/hbridges/sst/occl/results && ls -la s23.zip >&2 && tar cf - s23.zip' | tar xf - -C results/inbox/ && .venv/bin/python scripts/check_gate.py s23 --bundle results/inbox/s23.zip --no-file && .venv/bin/python scripts/report_s23.py --bundle results/inbox/s23.zip
```

## Local verification (2026-10-05)

- `pytest tests/` passed: 430 tests, including `tests/test_s23.py`, which covers:
  - **shape terms:** thin weights; exact < over-painted; cost growing with misplacement distance; gradient flow;
    empty channels skipped;
  - **the step hook:** terms added only for SHAPE arms, and OVS leaving the step untouched;
  - **the OVS schedule:** fit rows verbatim, k extras per reference, seeded, pair indices;
  - **the real training loop on a stub model:** steps follow the schedule, each step reads its own pair, and the
    OVS record is written;
  - **the contract:** tampering refused, planted effects recovered, additivity, the hardware flag.
- `test_s18b`'s resume test, which exercises the default training order, still passes unchanged.
- **Real data:**
  - 13 ventral strip references in the CLEAN fit (12 (m×p)×m, 1 malleti);
  - schedule 961 = 870 + 91;
  - shape terms on real 1024-px reference masks, with values and weights as above.
- **Real-model SHAPE smoke (CPU, 36 GB address-space cap,
  `scratchpad smoke23/shape_step.py`).**
  - **Setup:** one cycle step through the real SAM 2-large tracker with arch-scope LoRA and a backward pass, on a
    real CLEAN-fit ventral strip (m×p)×m reference (17957). It used one reference and 3 channels (the streak traits
    5/8 plus trait 1).
  - **Result:** anchor 0.054, boundary 0.013, Tversky 0.050, total 0.227, finite gradients on all 182 LoRA tensors.
    The shape terms sit at the anchor's scale.
  - **Why it was reduced:** the full five-reference, 10-channel step cannot run locally. The 36 GB cap stopped it
    at allocation, and one uncapped attempt in S21 ran the 46 GB workstation out of memory. The five-reference
    path is S19b's unchanged code.
  - **So:** the first full SHAPE/OVS steps run on the SCC, and the screen gate checks every history row (schedule,
    shape terms finite) before any test cell.

## Limitations

- **Closing-frame-only supervision** is S23's premise and its likely limit: S17's absorption finding and S21's CLEAN
  null.
- **13 ventral strip references are few.**
- **The ventral strip classification** uses S21's test-informed thresholds.
- **The boundary term is a bounded variant**, not Kervadec's exact loss (see above).
- **The thin-weight reference** (40 px) and the term weights (1.0) are a priori, not swept: one setting, one read.

## Landed result — 2026-10-06 (gates `s23_screen` PASS, `s23` PASS; first run)

Full tables: [S23_RESULTS.md](S23_RESULTS.md).

**Hardware confound (measured, disclosed).**
- All S23 cells and fits ran on **NVIDIA L40S**; the S21 parent cells they pair with ran on **A40**.
- S22 measured the L40S vs A40 bf16 gap on identical SAM 3 inputs: mean shift ≤ 0.024, per-run up to 0.7.
- So every arm-vs-parent read carries a small hardware term on top of the refit.
- The three arms share hardware with **each other**, so the clean lever reads are the within-S23 contrasts: SHAPE =
  stack − OVS, and OVS = stack − SHAPE.

### Lever effects on matched hardware (5-shot; 1-shot in the results file)

| Effect | Major | (m×p)×m | ventral 5/8 streak | ventral 6/9 spots | ventral 2/3 band | lativitta rays |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| SHAPE (stack − OVS) | −0.11 [−0.15, −0.07] ✱ | −0.19 [−0.36, −0.00] ✱ | **+0.03 [−0.33, +0.43]** | −1.05 [−1.59, −0.48] ✱ | +1.06 [+0.30, +1.92] ✱ | −0.94 ✱ |
| OVS (stack − SHAPE) | +0.04 [+0.01, +0.07] ✱ | +0.20 [+0.07, +0.33] ✱ | +0.54 [+0.14, +0.91] ✱ | −0.32 [−0.88, +0.22] | **+3.92 [+3.04, +4.87] ✱** | −0.16 ✱ |

### Each arm vs the S21 parent (crosses hardware), 5-shot, label-consistent pool

| Arm | mIoU | Major | ventral 5/8 | ventral 6/9 | ventral 2/3 | lativitta rays |
| :-- | --: | :-- | :-- | :-- | :-- | :-- |
| parent `crown_all_mref_lc` (S21, A40) | 78.76 | | 25.61 | 39.79 | 69.16 | 48.22 |
| SHAPE | 78.62 | −0.13 ✱ | +0.32 ✱ | −1.00 ✱ | +0.26 | −1.13 ✱ |
| OVS | 78.77 | +0.01 | +0.82 ✱ | −0.27 | +3.12 ✱ | −0.35 ✱ |
| SHAPE + OVS | 78.66 | −0.09 ✱ | +0.86 ✱ | −1.32 ✱ | +4.18 ✱ | −1.29 ✱ |

Further reads:
- Stack additivity is +0.03 [−0.02, +0.07].
- Memorization occupancy vs parent: +0.61 ✱, +0.19 and +0.74 ✱ pp, all within the ±3 pp band.
- Validation on the CLEAN split: 82.10, 82.19 and 82.04 against the parent's 82.21. All three fits are eligible.

### Against the bands

| Read | Band | Result |
| :-- | :-- | :-- |
| **PRIMARY: ventral 5/8, 5-shot** | success ≥ +2 ✱; falsifier ≤ 0 | **Success not met by any arm.** SHAPE's own effect on matched hardware is **+0.03 (falsifier met: the shape terms do not reach the streak)**. The best arm-vs-parent read is the stack's +0.86 ✱, which is OVS's |
| Ventral 6/9 | ≥ +2 ✱ | **missed, and the opposite way:** SHAPE −1.05 ✱ |
| Major, 5-shot | −0.10 to +0.15 | OVS +0.01 ✓; stack −0.09 ✓; **SHAPE −0.13 ✗** (below) |
| (m×p)×m | −0.3 to +0.5 | all inside |
| lativitta rays under SHAPE | −0.5 to −3 (harm) | **−1.13 ✱, as predicted.** The false-positive weight tightens rays that the test labels draw loosely |
| lativitta rays under OVS | ±0.5 | −0.35 ✓ |
| cyrbia, notabilis × lativitta | ±0.3 | inside (largest +0.19 notabilis under SHAPE) |
| OVS ventral (all traits) | 0 to +1 | +0.66 ✓ |
| 1-shot | non-significant | **missed**: SHAPE −0.15 ✱ Major at 1-shot, plus several ✱ subgroup reads |
| Memorization | ±3 pp | ✓ |

### Reading

- **Closing-frame shape supervision does not reach the ventral streak.** On matched hardware SHAPE moves it by +0.03.
  It also costs Major (−0.11 ✱), the basal spots (−1.05 ✱) and lativitta's rays (−0.94 ✱). The cycle anchors only
  on the reference, and the reference's streak is already reconstructed (closing anchor about 0.05). So the terms
  have almost nothing to correct where they act, and their side effects land elsewhere. This is S17's absorption
  finding again, applied to shape terms.
- **Oversampling the 13 label-consistent ventral references helps the underside a little:**
  - the ventral band 2/3 gains +3.9 ✱;
  - the streak gains +0.5 ✱;
  - (m×p)×m gains +0.2 ✱;
  - Major is flat-to-slightly-positive (+0.04 ✱ matched; +0.01 vs parent across hardware).
- **The streak (about 26) and the spots (about 39) remain out of reach under pure OC-CCL.**
- **Nothing is promoted.** The butterfly best stays at 78.75–78.77 (S21 POOL `crown_all_mref` and its CLEAN/OVS
  equivalents, all within noise of each other).
- **Process lesson:** the stage driver should pin the GPU type to the parent's, or re-score the parent on the same
  allocation. Recording it (new in S23) is what made the confound visible.
