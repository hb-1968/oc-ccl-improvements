# Technical Q&A, S16

Implementation-level answers for `Hunter_SST_second_reference.pptx`, at the
level of functions, formulas and thresholds. Backup slides 18 and 19 point
here.

**Delta convention.** mIoU deltas are quoted in points first and as a
percentage of the baseline they were measured against second: `+5.17 (+8.1%)`
is 5.17 mIoU points, which is 8.1% of the 63.86 it was added to. A percentage
is never a share of a total unless the text says so.

**Source of every number.** `results/s16.zip` for the cells and the two
headline rows; the README rows named in `make_s16_figures.py` for the banked
levels (63.20, 73.16, 70.30, 71.78, 75.18, 63.86, 69.88, 65.2). The figure
script asserts the transcribed constants against the bundle before it draws,
so a stale constant fails the build rather than reaching a slide.

---

## 1. What the implicit second reference actually is

`src/occl/adapters/tta.py`.

`rotate_episode(ep)` returns the same episode with every support rotated 180
degrees, image and label map together, via `np.rot90(x, 2)`. The query is never
touched: what matters is the support's orientation relative to the query, and
rotating the support avoids rotating the prediction back afterwards. Support
ids get the suffix `@r180`, because the embedding cache is keyed by the id
string and a rotated support served the upright image's features would fail
silently.

The pick is `confidence(logits)`:

```python
top = logits.max(axis=0)
fg = top > 0.0
return float(top[fg].mean()) if bool(fg.any()) else float("-inf")
```

Mean top-channel margin over the pixels the model claims. A prediction
propagated from a mismatched orientation is scattered and weakly positive; one
from a matching orientation is compact and strongly positive. An empty
prediction returns `-inf` and always loses. No ground truth is consulted, so
the choice leaks nothing.

`OrientationTTAAdapter` is a wrapper around the tracker adapter rather than a
change to it, so `SSTSam2Adapter` stays byte-stable against every previously
recorded number. It counts `n_rotated_chosen / n_total`, which is the fire rate
on slide 15.

## 2. What the explicit second reference actually is

The paper's own multi-shot construction: one 3-frame video `{x0, x1, xq}`, both
supports as conditioning frames, the query last. Selection is memory attention's
job, and S6b measured that the bank selects rather than blends.

SAM 2's `max_cond_frames_in_attn` is -1, meaning all conditioning frames are
attended, so both supports are genuinely in play
(`sam2/modeling/sam2_base.py`). This is worth stating because the commissioned
SAM 3 checkpoint ships with that same setting configured to 4
(`sam3_multiplex_base.py`, `++trainer.model.max_cond_frames_in_attn=4`), while
the SAM 3 tracker's own default is also -1. It is a config attribute rather
than weights, so it can be raised, but it is a trap at higher shot counts if
nobody looks. It does not arise here.

## 3. Where the draws come from

`seeding.sample_supports` and `curation.stratified_supports` both slice
`[:shot]` from one seeded permutation of the dataset's canonical pool.
`derive_seed(dataset, group, shot, run_idx)` is a blake2b digest that does take
a shot field; the nesting comes from its callers, which pass a literal `0`
there rather than the actual shot count. `sample_supports` carries the comment
"Do not 'fix' it to `shot`", because that literal is exactly what makes 1-shot
a strict prefix of the 2-shot and 5-shot draws instead of a separate
randomisation.

Consequence, and the reason S16 is runnable at all: run i's 1-shot support is
the first element of run i's 2-shot draw. The per-seed difference therefore
still cancels the shared support's difficulty, which is what makes the row a
paired comparison rather than two independent means.

Butterfly used `canonical_placement`, beetle `canonical_balanced`, both 20
seeds, 5 groups for butterfly and 1 for beetle, hence 100 and 20 (group, run)
pairs.

## 4. Why `registry.compare` refuses the row, and what unlocks it

The row crosses `protocol.shots` and `protocol.support_set_hash`. Both are
comparability fields, so the default refusal is correct: `support_set_hash`
legitimately differs across shot counts, so the field that normally guards
pairing cannot stand guard here.

The one sanctioned route is `compare(..., nested_shots=True)`, which calls
`nested_shot_evidence(treatment, baseline)` before masking anything:

```python
def _draws(rec):
    return {(r["group"], int(r["run_idx"])): [str(x) for x in r["support_ids"]]
            for r in (rec.get("runs") or []) if r.get("support_ids") is not None}

bad = [k for k in shared
       if len(dt[k]) != kt or len(db[k]) != kb or dt[k][:kb] != db[k]]
```

It re-reads the per-run `support_ids` off both records, requires treatment
shots > baseline shots, requires at least one shared (group, run), and raises
`ValueError` if the prefix property fails on any pair. Summary-only records are
refused outright, because a record with no draw carries nothing to verify.

Only then does it mask `NESTED_SHOT_FIELDS = ("protocol.shots",
"protocol.support_set_hash")`, and the proof travels in the result as
`nested_shots = {treatment_shots, baseline_shots, runs_checked}`. The recorded
rows carry `runs_checked` 100 and 20.

Naming a protocol field in `ignore_fields` by hand is still refused. The
relaxation has to arrive with its evidence. The `model.tta` toggle the row also
crosses is declared separately in `ignored_fields`, with both values recorded.

The nesting was verified on the real canonical pools before any GPU time was
spent: 100 of 100 butterfly and 20 of 20 beetle pairs nested, and every beetle
2-shot draw spans 2 distinct species strata. The gate to combine schema is
exercised end to end on synthetic cells in
`tests/test_s16_gate_roundtrip.py`, including the refusal on non-nested draws.

## 5. The metric, exactly

`METRIC_SPEC = {name: iou_per_trait_macro, assignment: argmax_label,
reduction: macro_over_gt_traits}`.

Per query: resize logits to the query's native resolution, argmax across trait
channels, background wherever the top channel is not positive, IoU per
ground-truth trait, mean over the traits present in that image's ground truth.
Then the mean over queries, then over the 20 runs. A trait the model never
claims scores 0 and still counts, so a missed antenna costs as much as a
botched elytra.

Four readings are recorded per cell and all four are in the bundle:

| reading | what it changes | butterfly TTA / 2-shot | beetle TTA / 2-shot |
| :-- | :-- | :-- | :-- |
| plain | the headline rule above | 75.18 / 71.48 | 63.86 / 69.04 |
| independent | per-trait threshold rather than argmax | 75.23 / 71.73 | 64.94 / 70.27 |
| best-of-3 | independent overlap *and* boundary tolerance | 76.72 / 73.23 | 65.45 / 70.75 |
| oracle | best assignment of the proposed regions | 75.65 / 75.35 | 66.93 / 71.73 |

The second reading is `miou_boundary.independent_best3`, which is what
`combine_s16.py` pairs on, so it differs from the plain headline in two ways at
once: argmax is replaced by an independent per-trait threshold, and the
boundary gets the paper's plus or minus one pixel tolerance. Both readings give
the same sign and both are significant, so the conclusion does not depend on
either one, but it is a two-factor change and should be described as one.

The oracle reading is recorded per cell and is not paired, so oracle
differences quoted anywhere in this deck are differences rather than tested
effects.

## 6. What the oracle ceiling means, and why slide 14 matters

The oracle reading assigns each proposed region to whichever trait scores best.
It is therefore an upper bound on what assignment alone could fix, and the gap
between plain and oracle is assignment error while the ceiling itself is a
property of what the model proposed.

Beetle's ceiling moved 66.93 to 71.73, +4.80 (+7.2%), when the second specimen
entered the bank. That is a change in the proposals, not in the assignment.
Butterfly's is flat within a third of a point, 75.65 to 75.35. Neither figure
carries a paired test, since the oracle reading is not one of the paired
metrics.

S14sam3's arch arm moved the same axis on beetle by training, 66.76 to 71.08.
Here it moved by data instead. S14sec has no arch arm, so name S14sam3 when
making this comparison.

## 7. The anchors, and what they would have caught

The two 1-shot TTA cells resume the banked S11 ledger cells rather than being
re-measured. The evaluator fingerprint is identical, so the run costs zero
inference and the values come back bit for bit: every recorded field matches
S11 to all 16 significant digits. The gate pins them to 0.7518 and 0.6386 with
a tolerance of 0.005, that is 0.5 mIoU points.

That tolerance is written for the case where the cell is genuinely re-run: the
plain path is deterministic to about 1e-4, but TTA's confidence pick can flip
on near-ties, so 0.5 is generous to noise and still catches the slips that
matter. The named failure it guards against is TTA being silently off, which
would put roughly 70.30 in a 75.18 cell, and which has happened twice in this
project when s5c ran as plain s5b.

What the anchors therefore certify is the wiring of this stage: the right cells
in the right slots, drawn from the right pool, merged under the right labels.
They are not an independent reproduction of TTA, and the only genuinely new
measurements in S16 are the two 2-shot cells, which have no anchor of their own
and are guarded instead by the dropped-frame floor in section 8.

## 8. Disqualifiers written before the run

Gate FAIL, debug before interpreting:

- beetle 2-shot at or below the banked 1-shot plain level 63.20, or butterfly
  2-shot below 69.80 (70.30 minus 0.5). A second conditioning frame cannot make
  the tracker worse unless it is being dropped, so this is the S6 "frames not
  dropped" check one shot down. It is unpaired against the ledger by design,
  because the plain 1-shot cell was deliberately not re-run.

Recorded but not gated:

- beetle 2-shot above 73.16, the measured 5-shot level, would mean the draw is
  wrong, most likely two strata collapsing to one, rather than that 2-shot is
  unusually good. Measured 69.04, so this stayed quiet.

## 9. The seven gate predicates, per dataset

1. Both cells on the dataset's canonical protocol, cell names matching their
   shot counts and TTA flags, 20 of 20 seeds each.
2. The 1-shot TTA anchor reproduces the S11 ledger within 0.5.
3. The headline carries its per-run prefix-nesting proof, with
   `runs_checked` equal to 20 times the number of groups.
4. The headline declares the `model.tta` toggle it crosses.
5. 2-shot clears the banked 1-shot level, so no conditioning frame is dropped.
6. The paper's best-of-3 ran alongside the plain metric.
7. The headline is recorded with its confidence interval and seed count.

Seven per dataset, 14 of 14 PASS, landed 2026-09-01.

## 10. Statistics

Paired bootstrap over the 20 seeds, 10,000 resamples, seed 0, computed by
`registry.compare`. Significance means the 95% interval excludes 0.

| dataset | delta | CI95 | seeds improved | paired std |
| :-- | --: | :-- | --: | --: |
| butterfly Major | -3.71 | [-4.48, -2.69] | 2 of 20 | 2.10 |
| beetle | +5.17 | [+2.63, +7.77] | 15 of 20 | 6.05 |

Butterfly's 2 of 20 with a tight interval and beetle's 15 of 20 with a wide one
are consistent with the cell spreads: butterfly's 2-shot cell has std 0.30 and
beetle's has 4.28.

## 11. Cost

Two `fanout` passes per dataset. Butterfly about 1.1 hours, dominated by the
TTA cell's double decode; 2-shot adds one conditioning frame at roughly 1.2
times the plain cost. Beetle about 0.3 hours. Both inside one A100 session, and
less when the TTA cells resume from the Drive bank, since an identical
evaluator fingerprint means zero inference.

Splittable per dataset via `S16_DATASETS`; the merge pulls a missing dataset's
combine from Drive. Gate: `check_gate.py s16` on the merged bundle, or
`--bundle s16_<dataset>.zip` for a split session.

## 12. What S16 does not claim

- It does not measure whether TTA and an explicit second shot compose. Four
  runs, two cells per dataset, no `{1,2}-shot` by `{plain, TTA}` square. The
  square was cut to these four runs by decision on 2026-09-01, and the cut
  single-lever deltas remain reconstructible against S11's banked plain cells.
- It does not say 2-shot beats the trained stack. On beetle 69.04 at two labels
  is within noise of 69.88 at one label plus training, which is a statement
  about label cost. Nor is a second label the largest lever measured on beetle
  in absolute terms: S6's 1 to 5 shot lift is +9.96 on the same stock model.
- The +5.84 and +1.18 quoted against banked 1-shot levels are unpaired
  cross-stage subtractions, since the plain 1-shot cells were deliberately not
  re-run. They carry no confidence interval, and the gate says so in the same
  words.
- It does not generalise across datasets. The two results have opposite signs,
  each by the mechanism that dataset's residual named.
- It says nothing about butterfly's absolute gap to the paper's Major numbers,
  which is a separate open item.

---

## Short answers

- **Why is TTA "implicit"?** The rotated support is a second reference that
  never enters the memory bank and is chosen per episode by a confidence rule
  instead of by attention.
- **Why does TTA fire on 29.7% of beetle episodes but pay only +0.66?** Firing
  means the rotated pass was more confident. It adds an orientation, not a
  species, and beetle's residual is species identity.
- **Why does 8.1% firing pay +4.88 on Major?** The cohort it is presumed to be
  reaching is cyrbia's rotated queries, 37% of that subspecies' test specimens,
  which score 0.48 against 74.95 for the upright ones, and which nothing
  support-side reaches. Only the aggregate fire rate is recorded, so the
  per-episode correspondence is an inference, not a measurement.
- **Is the butterfly band miss a failure?** It is a miss of 0.19 on the
  2-shot side, and the direction and significance were both predicted. The
  cause is measured: the shot curve saturates.
- **Why 20 seeds?** It is the project's standing protocol, and it is what the
  banked cells the anchors pin to were run with.
- **Could the two cells differ in anything other than the second reference?**
  Both are base SAM 2 with no adapters, the same image size, dtype and
  normalisation, the same canonical pool, the same seeds, and the same
  evaluator. The comparison records every field it masks.
