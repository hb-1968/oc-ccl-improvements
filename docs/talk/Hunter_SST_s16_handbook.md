---
title: "S16 talk handbook"
subtitle: "Speaking guide and technical Q&A for Hunter_SST_second_reference.pptx"
author: "Hunter Bridges"
---

The deck is 19 slides: 17 presented, 2 backup. Part I is the speaking guide,
one entry per slide, giving the time to spend and what the slide has to
establish. Part II answers the implementation questions at the level of
functions, formulas and thresholds, and is what backup slides 18 and 19 point
at.

\newpage


# Part I. Speaking guide

For `Hunter_SST_second_reference.pptx`, 19 slides: 17 presented, 2 backup.
The timings below sum to about 14.5 minutes, so cut slide 5 to land nearer 13
and leave room for questions in a 20 minute slot.

These are directions, not a script. Each entry gives the time to spend, the
number of clicks, what the slide has to establish, and what to leave alone.
Numbers are quoted here only where saying the wrong one would break the
argument.

**Delta convention.** Every mIoU delta is quoted in points first and as a
percentage of the baseline it was measured against second, so `+5.17 (+8.1%)`
means 5.17 mIoU points, which is 8.1% of the 63.86 it was added to. The
percentage is never a share of a total.

---

### Part I. The question (slides 1 to 4, about 2.5 minutes)

#### 1. Title

20 seconds, no clicks.

State the question in one sentence: a second reference can be the same
specimen rotated, or a different specimen labelled, and this stage measures
which one pays. Say that the answer is not the same on the two datasets, and
that this was predicted before the run rather than after.

#### 2. One labelled specimen, then a query

45 seconds, 4 clicks.

Only enough setup to make the budget visible. Support in, its trait masks,
query, prediction, ground truth. If the audience saw the main talk, click
through to the last line and say only that one labelled mask is the entire
supervision budget.

#### 3. Two ways to spend a second reference

70 seconds, 3 clicks.

This is the core distinction, so spend the time.

- Click 1, implicit: the same specimen turned 180 degrees, carried outside the
  memory bank, chosen per episode by a confidence rule over the logits.
- Click 2, explicit: a different individual, inside the bank as one 3-frame
  video, chosen by memory attention.
- Click 3: the costs, said out loud. Zero labels and two forward passes
  against one label and one forward pass.

The sentence to land: since S5c every headline arm has used the implicit
version, and the explicit one at the same inference cost had never been
measured.

#### 4. The shot axis had a hole at two

35 seconds, 2 clicks.

Point at the gap. The project has 1-shot and 5-shot on both datasets and never
2. The dashed segment is an interpolation nobody measured.

---

### Part II. How it was run (slides 5 to 8, about 3.5 minutes)

#### 5. What one mIoU number means here

40 seconds, 3 clicks. First candidate for cutting if the audience sat through
the main talk.

Say the two rules that matter: argmax across trait channels after resizing to
the query, and a trait the model never claims scores 0 and still counts in the
mean.

#### 6. Four runs, two of them anchors

40 seconds, 3 clicks.

Four cells, base SAM 2, no adapters, 20 seeds each. Be precise about what the
TTA cells are: they resume the banked S11 ledger cells from the Drive bank,
same evaluator fingerprint and therefore zero inference, and the gate pins them
to 75.18 and 63.86 within 0.5. That certifies the wiring of the merge, not TTA
itself. Say plainly that the two 2-shot cells are the only new measurement.

If asked why an anchor is worth anything if it is a resume: it catches a
mis-wired merge, a mislabelled cell, or the wrong pool, which is the failure
mode that produced a wrong number in this project before.

#### 7. Why this comparison is allowed

70 seconds, 3 clicks. This is the implementation question people ask, so give
it the time.

- Click 1: the row crosses two comparability fields, `protocol.shots` and
  `protocol.support_set_hash`, so `registry.compare` refuses it by default.
- Click 2: the licence is a property of the draws, not of the two records. Both
  cells slice a prefix of one seeded permutation, so run i's 1-shot support is
  the first element of its 2-shot pair and the shared support's difficulty
  still cancels in the per-seed difference.
- Click 3: the proof is read back off the per-run `support_ids` and travels
  inside the row, 100 butterfly pairs and 20 beetle pairs. Naming a protocol
  field in `ignore_fields` is still refused.

If asked how the pairing could have been wrong, that is backup slide 19 and
Q&A section 4.

#### 8. Two predictions, opposite signs

60 seconds, 3 clicks. Do not skip this. It is what makes the next slide a test
rather than a report.

Read the bands off the figure before showing anything measured, then give one
mechanism sentence each:

- Butterfly, band -3.9 to -4.6: the defect is query orientation, 37% of
  cyrbia's test specimens are pinned upside down, and no upright second
  specimen reaches them.
- Beetle, band +2 to +5.3: the 5-shot mechanism is species diversity, and a
  rotated copy of one specimen adds no species.

---

### Part III. The result (slides 9 and 10, about 1.5 minutes)

#### 9. The mirror image lands, both signs

70 seconds, 4 clicks. Let the figure sit for a beat before speaking.

Both predicted signs, both significant, gate PASS 14 of 14. Quote butterfly
-3.71 (-4.9%) and beetle +5.17 (+8.1%).

Say both band misses honestly. Butterfly is 0.19 short of its band edge, and
the miss is on the 2-shot side rather than the TTA side. Beetle landed at the
top of its headline band and past the top of its pre-registered share band,
58.6% against 35 to 55%. The two datasets missed in opposite directions, which
is worth saying out loud rather than reporting only the one that undershot.
Slides 11 to 13 are the explanation; do not defend anything here.

#### 10. The sign survives the other reading

30 seconds, 2 clicks. Short.

Under the paper's full reading, which swaps argmax for an independent per-trait
threshold and adds the boundary best-of-3, the headlines are -3.49 and +5.30,
both still significant. Say that it changes two things at once and neither sign
moves.

---

### Part IV. Why (slides 11 to 15, about 4.5 minutes)

#### 11. Major's shot curve is flat by two

50 seconds, 3 clicks.

2-shot came in at 71.48 against a predicted 70.6 to 71.3. The explanation is
saturation: +1.18 of the measured +1.48 five-shot lift, which is 80% of it,
arrives with the first added specimen. That is the number to say. Add that the
+1.18 is against the banked 1-shot level and is therefore unpaired, because the
plain 1-shot cell was deliberately not re-run.

#### 12. The second specimen bought spread

50 seconds, 3 clicks.

The second support did something real, just not to the level: std 2.10 to 0.30,
a drop of 86%, the tightest cell measured at 1 or 2 shots, and the level still
3.71 below TTA. If someone knows the 5-shot cells, S6's Major 5-shot sits at
0.15, so say "at this budget" rather than claiming a project record. Tie it to what S5b and S6 said support-side levers do on Major, and then
say the conclusion plainly: the residual is query-side.

#### 13. Beetle earns the lift on shot two

45 seconds, 3 clicks.

+5.84 (+9.2%) over the banked 63.20, unpaired for the same reason as slide 11,
which is 58.6% of the whole 1 to 5 shot lift from one added specimen. Concave
as predicted, but past the top of the pre-registered 35 to 55% share band, so
report it as an overshoot rather than as a clean confirmation. Mention that the
5-shot level 73.16 stays above it, so the draw-suspicion check stays quiet.

#### 14. It changed what gets proposed

60 seconds, 3 clicks. Give this one room; it is the most interesting slide.

The beetle oracle ceiling moved 66.93 to 71.73, which is +4.80 (+7.2%). The
ceiling is what the model could score if every proposed region were assigned
to the right trait, so a ceiling that moves means the proposals changed, not
just the assignment. Butterfly's is flat within a third of a point, 75.65 to
75.35. Say once that no paired test was run on the oracle reading, so these are
differences rather than tested effects.

If the audience was at the S14 talk, add one sentence: S14sam3's arch arm moved
this same axis on beetle by training, 66.76 to 71.08, and here it moved by
data.

#### 15. TTA is an orientation shot only

60 seconds, 3 clicks. This is the mechanism sentence of the whole stage.

On beetle the rotated support is kept in 29.7% of episodes and buys +0.66
(+1.0%). The explicit specimen is worth about nine times that. On Major it
fires in 8.1% of episodes and is worth +4.88 (+6.9%). Fires often and pays
little against fires rarely and pays a lot.

Be careful with the last step: the cohort those Major firings are presumed to
be reaching is cyrbia's 37% rotated queries, which score 0.48 against 74.95 for
the upright ones, but the per-episode correspondence is inferred rather than
recorded. Say "presumed" if you say it at all.

---

### Part V. What it costs and what it means (slides 16 and 17, about 2 minutes)

#### 16. What the second label actually buys

55 seconds, 3 clicks.

One extra labelled mask on stock SAM 2 puts beetle at 69.04, within noise of
the project's champion at one label, 69.88, which is a model swap plus a
trained adapter plus TTA. On SST's own label axis the cheapest training-free
point above the paper's 1-shot OC-CCL row of 65.2 is now 2 labels.

Say training-free, not cheapest outright: the trained stack reaches 69.88 at
one label and is on the same figure. This is a statement about label cost, not
about the value of the trained stack, and it is one dataset.

#### 17. Three things this changes

60 seconds, 4 clicks. One line per click, then leave it up for questions.

Land the mirror image first, then the framing point about asking what a
technique adds rather than what it resembles, then the cost. Stop talking after
the third card.

---

### Backup

#### 18. Every cell, as recorded

Not presented. Jump here for any per-cell number, spread, or confidence
interval.

#### 19. What the gate checks, per dataset

Not presented. Jump here when asked how the run was guarded or what would have
made it fail. Seven predicates per dataset, 14 of 14 PASS.

---

### If you are short on time

Cut in this order:

1. Slide 5, the metric, if the audience has seen it.
2. Slide 10, the second metric, folded into one sentence on slide 9.
3. Slide 13, beetle concavity, folded into one sentence on slide 14.

Do not cut slide 8. Without the pre-registration, slide 9 is a report rather
than a test, and the whole talk loses its point.

### Questions to expect

- **"Is 2-shot just better, then?"** On beetle at a two-label budget, yes, and
  by more than any other single lever available there. Not in absolute terms:
  5-shot is +9.96 on the same stock model, at five labels. On butterfly, no.
  The answer is dataset-dependent by mechanism, which is the finding.
- **"Why not run the full square?"** The `{1,2}-shot` by `{plain, TTA}` square
  was cut to these four runs by decision on 2026-09-01. The cut single-lever
  deltas remain reconstructible against S11's banked plain cells.
- **"How do you know the pairing is legitimate?"** Slide 7 and Q&A section 4.
  The evidence is recomputed from the per-run draws at combine time and stored
  in the row.
- **"Would TTA and 2-shot compose?"** Not measured here. S16 tests which
  second reference pays, not whether they stack. Say so plainly.
- **"Does this change the deployment recommendation?"** On beetle it adds a
  no-training option at two labels. It does not retire the trained stack,
  which still holds the best number at one label.


\newpage


# Part II. Technical Q&A

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

### 1. What the implicit second reference actually is

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

### 2. What the explicit second reference actually is

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

### 3. Where the draws come from

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

### 4. Why `registry.compare` refuses the row, and what unlocks it

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

### 5. The metric, exactly

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

### 6. What the oracle ceiling means, and why slide 14 matters

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

### 7. The anchors, and what they would have caught

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

### 8. Disqualifiers written before the run

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

### 9. The seven gate predicates, per dataset

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

### 10. Statistics

Paired bootstrap over the 20 seeds, 10,000 resamples, seed 0, computed by
`registry.compare`. Significance means the 95% interval excludes 0.

| dataset | delta | CI95 | seeds improved | paired std |
| :-- | --: | :-- | --: | --: |
| butterfly Major | -3.71 | [-4.48, -2.69] | 2 of 20 | 2.10 |
| beetle | +5.17 | [+2.63, +7.77] | 15 of 20 | 6.05 |

Butterfly's 2 of 20 with a tight interval and beetle's 15 of 20 with a wide one
are consistent with the cell spreads: butterfly's 2-shot cell has std 0.30 and
beetle's has 4.28.

### 11. Cost

Two `fanout` passes per dataset. Butterfly about 1.1 hours, dominated by the
TTA cell's double decode; 2-shot adds one conditioning frame at roughly 1.2
times the plain cost. Beetle about 0.3 hours. Both inside one A100 session, and
less when the TTA cells resume from the Drive bank, since an identical
evaluator fingerprint means zero inference.

Splittable per dataset via `S16_DATASETS`; the merge pulls a missing dataset's
combine from Drive. Gate: `check_gate.py s16` on the merged bundle, or
`--bundle s16_<dataset>.zip` for a split session.

### 12. What S16 does not claim

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

### Short answers

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
