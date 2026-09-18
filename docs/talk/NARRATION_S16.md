# Speaking guide, S16 deck

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

## Part I. The question (slides 1 to 4, about 2.5 minutes)

### 1. Title

20 seconds, no clicks.

State the question in one sentence: a second reference can be the same
specimen rotated, or a different specimen labelled, and this stage measures
which one pays. Say that the answer is not the same on the two datasets, and
that this was predicted before the run rather than after.

### 2. One labelled specimen, then a query

45 seconds, 4 clicks.

Only enough setup to make the budget visible. Support in, its trait masks,
query, prediction, ground truth. If the audience saw the main talk, click
through to the last line and say only that one labelled mask is the entire
supervision budget.

### 3. Two ways to spend a second reference

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

### 4. The shot axis had a hole at two

35 seconds, 2 clicks.

Point at the gap. The project has 1-shot and 5-shot on both datasets and never
2. The dashed segment is an interpolation nobody measured.

---

## Part II. How it was run (slides 5 to 8, about 3.5 minutes)

### 5. What one mIoU number means here

40 seconds, 3 clicks. First candidate for cutting if the audience sat through
the main talk.

Say the two rules that matter: argmax across trait channels after resizing to
the query, and a trait the model never claims scores 0 and still counts in the
mean.

### 6. Four runs, two of them anchors

40 seconds, 3 clicks.

Four cells, base SAM 2, no adapters, 20 seeds each. Be precise about what the
TTA cells are: they resume the banked S11 ledger cells from the Drive bank,
same evaluator fingerprint and therefore zero inference, and the gate pins them
to 75.18 and 63.86 within 0.5. That certifies the wiring of the merge, not TTA
itself. Say plainly that the two 2-shot cells are the only new measurement.

If asked why an anchor is worth anything if it is a resume: it catches a
mis-wired merge, a mislabelled cell, or the wrong pool, which is the failure
mode that produced a wrong number in this project before.

### 7. Why this comparison is allowed

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

### 8. Two predictions, opposite signs

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

## Part III. The result (slides 9 and 10, about 1.5 minutes)

### 9. The mirror image lands, both signs

70 seconds, 4 clicks. Let the figure sit for a beat before speaking.

Both predicted signs, both significant, gate PASS 14 of 14. Quote butterfly
-3.71 (-4.9%) and beetle +5.17 (+8.1%).

Say both band misses honestly. Butterfly is 0.19 short of its band edge, and
the miss is on the 2-shot side rather than the TTA side. Beetle landed at the
top of its headline band and past the top of its pre-registered share band,
58.6% against 35 to 55%. The two datasets missed in opposite directions, which
is worth saying out loud rather than reporting only the one that undershot.
Slides 11 to 13 are the explanation; do not defend anything here.

### 10. The sign survives the other reading

30 seconds, 2 clicks. Short.

Under the paper's full reading, which swaps argmax for an independent per-trait
threshold and adds the boundary best-of-3, the headlines are -3.49 and +5.30,
both still significant. Say that it changes two things at once and neither sign
moves.

---

## Part IV. Why (slides 11 to 15, about 4.5 minutes)

### 11. Major's shot curve is flat by two

50 seconds, 3 clicks.

2-shot came in at 71.48 against a predicted 70.6 to 71.3. The explanation is
saturation: +1.18 of the measured +1.48 five-shot lift, which is 80% of it,
arrives with the first added specimen. That is the number to say. Add that the
+1.18 is against the banked 1-shot level and is therefore unpaired, because the
plain 1-shot cell was deliberately not re-run.

### 12. The second specimen bought spread

50 seconds, 3 clicks.

The second support did something real, just not to the level: std 2.10 to 0.30,
a drop of 86%, the tightest cell measured at 1 or 2 shots, and the level still
3.71 below TTA. If someone knows the 5-shot cells, S6's Major 5-shot sits at
0.15, so say "at this budget" rather than claiming a project record. Tie it to what S5b and S6 said support-side levers do on Major, and then
say the conclusion plainly: the residual is query-side.

### 13. Beetle earns the lift on shot two

45 seconds, 3 clicks.

+5.84 (+9.2%) over the banked 63.20, unpaired for the same reason as slide 11,
which is 58.6% of the whole 1 to 5 shot lift from one added specimen. Concave
as predicted, but past the top of the pre-registered 35 to 55% share band, so
report it as an overshoot rather than as a clean confirmation. Mention that the
5-shot level 73.16 stays above it, so the draw-suspicion check stays quiet.

### 14. It changed what gets proposed

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

### 15. TTA is an orientation shot only

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

## Part V. What it costs and what it means (slides 16 and 17, about 2 minutes)

### 16. What the second label actually buys

55 seconds, 3 clicks.

One extra labelled mask on stock SAM 2 puts beetle at 69.04, within noise of
the project's champion at one label, 69.88, which is a model swap plus a
trained adapter plus TTA. On SST's own label axis the cheapest training-free
point above the paper's 1-shot OC-CCL row of 65.2 is now 2 labels.

Say training-free, not cheapest outright: the trained stack reaches 69.88 at
one label and is on the same figure. This is a statement about label cost, not
about the value of the trained stack, and it is one dataset.

### 17. Three things this changes

60 seconds, 4 clicks. One line per click, then leave it up for questions.

Land the mirror image first, then the framing point about asking what a
technique adds rather than what it resembles, then the cost. Stop talking after
the third card.

---

## Backup

### 18. Every cell, as recorded

Not presented. Jump here for any per-cell number, spread, or confidence
interval.

### 19. What the gate checks, per dataset

Not presented. Jump here when asked how the run was guarded or what would have
made it fail. Seven predicates per dataset, 14 of 14 PASS.

---

## If you are short on time

Cut in this order:

1. Slide 5, the metric, if the audience has seen it.
2. Slide 10, the second metric, folded into one sentence on slide 9.
3. Slide 13, beetle concavity, folded into one sentence on slide 14.

Do not cut slide 8. Without the pre-registration, slide 9 is a report rather
than a test, and the whole talk loses its point.

## Questions to expect

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
