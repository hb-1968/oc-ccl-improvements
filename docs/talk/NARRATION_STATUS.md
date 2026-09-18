# Speaking guide, status deck

For `Hunter_SST_status.pptx`, 30 slides: 28 presented, 2 backup. The timings
below sum to about 24 minutes of talking, which is the wrong number for a
20-minute slot on purpose: the cut list at the end takes it to 19. Cut before
you start, not while you are up there.

Directions, not a script. Each entry gives the time, the clicks, what the slide
has to establish, and what to leave for questions.

**Delta convention.** mIoU deltas are in points, and a percentage beside one is
a percentage of the baseline it was measured against. A star means the 95%
interval excludes zero.

**What this talk assumes.** That the audience saw the techniques talk. It does
not assume they saw anything since.

---

## Part I. Frame (slides 1 to 4, about 3 minutes)

### 1. Title

25 seconds, no clicks.

Say what kind of talk this is: a status report with a roadmap at the end. Six
stages landed since the last time, one recipe came out of them, one mechanism
question is still open. Do not preview numbers here.

### 2. Where that talk left it

45 seconds, 5 clicks. Recap only.

Four lines, then the three questions that were open. If they were at the last
talk, click through the first three and spend the time on the questions, since
those are what the next twenty minutes answer.

### 3. What landed since that talk

40 seconds, 3 clicks.

This is the map. Read the verdict column once, out loud, and move. Say that S18
is built and waiting for a session, so it will come back at the end as the next
move rather than as a result.

### 4. The scoreboard, as of today

60 seconds, 3 clicks. Let them read it before you speak.

Beetle passes both shot counts and now exceeds the paper at both. Major fails
both. That split organises the whole talk: the beetle half is about how far it
went, the Major half is about why nothing moves it.

---

## Part II. The label axis (slides 5 and 6, about 2 minutes)

### 5. Two ways to spend a second reference

45 seconds, 3 clicks.

Only the distinction and the two costs. Implicit: the same specimen rotated,
outside the bank, chosen by a confidence rule, free. Explicit: a second
individual inside the bank, one label.

### 6. Which one pays flips by dataset

55 seconds, 3 clicks.

Both signs were pre-registered from each dataset's own residual, and both
landed. The line to land is the practical one: two labels on stock SAM 2 puts
beetle at 69.04, within noise of the trained champion at one label.

If asked which is better in general: neither, and that is the finding.

---

## Part III. The champion at five shots (slides 7 to 10, about 3.5 minutes)

### 7. The champion at five shots

70 seconds, 3 clicks.

Bands first, then the landing, then Major. Name the two independent anchors
that produced the beetle band, the silhouette score and the 1-shot oracle
ceiling plus the measured shot mechanism, because that is what makes a landing
inside the band worth quoting at all.

Then Major: 77.47, below its band, and beaten by its own base.

### 8. Every cell, both datasets

45 seconds, 3 clicks.

Do not read the cells. Say two things: on beetle the adapter is the top of the
stack, and on Major the honest number is 77.95 with no adapter at all. Then the
backbone edge, which is real on both datasets.

### 9. The adapter is beetle-specific

40 seconds, 3 clicks.

Same adapter, same seeds, opposite signs, and unanimous in both directions:
20 of 20 seeds up on beetle, 0 of 20 on Major. Set up the next slide by saying
the beetle contribution halved from +1.96 at one shot to +0.97 at five.

### 10. Five labels close the pot

60 seconds, 3 clicks. Give this room; it is the bridge into the whole S17
branch.

Define the ceiling once: what the same predictions would score if every
proposed region were assigned to the right trait. At one shot there is a pot.
At five there is none, so there is nothing left for an assignment-side loss to
win. The pot is a function of the label budget, not of the objective.

---

## Part IV. Why every SAM 2 loss arm was null (slides 11 to 14, about 3.5 min)

### 11. One epoch is ten bad pairs

40 seconds, 3 clicks.

This was measured on the base model over all 120 training pairs before any arm
was designed. Ten pairs carry more than half the epoch. Say that the older
merged-silhouette recipe has no such tail, so the tail is a property of the
per-trait loss.

### 12. What they are charging for

70 seconds, 3 clicks. The core mechanism of the talk.

Three beats: SAM 2 replaces an absent trait's mask with a constant of −1024;
per-trait BCE then charges 1024 times the trait's area, and that identity holds
on all twenty of the worst channels; the constant has no gradient, so 62% of
the loss value is inert and nothing in the objective can reach a channel the
model did not find.

That last sentence explains every null loss arm in the project's SAM 2 history.
Say it slowly.

### 13. The cycle's own images

60 seconds, 4 clicks. Real images from the bundle, one training pair.

Point at the missing antennae in the forward prediction. Then at the closing
image, where the leg channel has taken them over. Then the trained arm, where
the closing image improves by about six points on this pair while the forward
image does not move.

### 14. The reverse leg absorbs it all

55 seconds, 3 clicks.

Every arm lowers its own objective, and every arm does it on the closing leg.
Then butterfly as the control: there the forward leg does move, which is why
OC-CCL ever paid on that dataset. The asymmetry is a property of the training
pairs, not of the loss.

---

## Part V. The repairs (slides 15 to 21, about 5 minutes)

### 15. Three faults, three repairs

50 seconds, 4 clicks. One line per click, and stress that all three live inside
the fine-tune: no model-code change, no extra labels, no change to the
evaluator.

### 16. The repairs compose

45 seconds, 2 clicks.

The ladder. The number to say is +0.69 with TTA on 19 of 20 seeds, and the
phrase is that it is the first beetle gain on SAM 2 that is not a reshuffle of
channels the model already found.

### 17. The ceiling moves, finally

40 seconds, 3 clicks. Short.

First movement of the oracle ceiling on this family. It says the fine-tune
changed what gets proposed, not how proposals are labelled.

### 18. The prior is free up to 30x

35 seconds, 3 clicks.

Say the falsified prediction out loud: a cost at weight 30 was pre-registered
and never appeared, so the curve is a plateau.

### 19. Scope subsumes the weight

45 seconds, 3 clicks.

Two levers, each real alone, composing below their sum. The outcome was
pre-registered as one of two values and it landed on the second. 65.04 against
65.03 for the scope alone, +0.01 on the same seeds across two sessions.

### 20. The fence, made differentiable

55 seconds, 3 clicks. The cleanest negative in the branch.

Exact at base weights, gradient into the score head turned from 0 to 0.566 on a
real pair, and the result is still nothing: 1 of 19 fence channels flipped on
its own, and −0.13 on top of the prior. The loss is now honest about the fence
and does not want it.

### 21. None of it reaches five shots

40 seconds, 3 clicks.

Three adapters, no 5-shot effect. Offer the mechanism, that at five supports a
trait missing from one reference is present on another, and flag it as
unmeasured.

---

## Part VI. The output (slides 22 and 23, about 2 minutes)

### 22. sam2_crown, established

55 seconds, 3 clicks.

Read the five components. Leave the evidence column for questions. The sentence
that matters is that it is pinned in the registry, re-derivable from a record's
own fields, and is the SAM 2 fine-tune from here on.

### 23. Where SAM 2 stops

50 seconds, 3 clicks.

65.04, 0.16 short of the paper's own OC-CCL row, 4.8 below the SAM 3 champion.
Every axis inside the family is measured and closed. That last line is the
handoff into the roadmap.

---

## Part VII. Where it goes (slides 24 to 28, about 4.5 minutes)

### 24. What S18 asks

70 seconds, 3 clicks.

Lead with the negative result, because it rules out the obvious story: the mask
is not being carried, its similarity to the reference sits at chance. Then the
route that is left, the continuous prompt, which hands the reverse leg a full
logit map the evaluator never scores. Then what S18 does about it: binarise the
prompt in value, keep the gradient.

If asked why this matters: it is the difference between the cycle learning to
track and the cycle learning to pass a note to itself.

### 25. The port that is left

55 seconds, 4 clicks.

One session. Both repairs were written against per-object logit stacks so they
could transfer. Say that the 5-shot result predicts a 1-shot-only gain, and
that this is worth pre-registering before the run rather than after.

### 26. What the headline hides

60 seconds, 4 clicks.

Do not defend the number. State it: the best Major arm reads 1,727 labelled
masks to choose one reference. State the defensible version, curation as a
stand-in for human judgement. Then the test, and what each outcome would mean.
The last line is the one to land.

### 27. Major's residual, after all that

55 seconds, 3 clicks.

Everything tried, against the gap. Rotation handling is the only lever that
ever moved this dataset. Two model families at two shot counts leave it where
it was, so what is left is protocol or annotation, not capacity.

### 28. Three sentences

45 seconds, 4 clicks. One per click, then stop.

---

## Backup

### 29. The gates behind this talk

Not presented. Jump here for a verdict or a landing date, or when asked what is
still owed.

### 30. The cells this talk quotes

Not presented. Jump here for any level.

---

## Cutting to 20 minutes

Cut in this order. The first three take roughly four minutes out.

1. **Slide 8** (the S15 cells): fold into slide 7 as one sentence, "the honest
   Major number is 77.95 with no adapter".
2. **Slide 18** (the weight) and **slide 19** (the scope): replace with one
   sentence on slide 22, "the weight is free and the scope subsumes it".
3. **Slide 17** (the ceiling): fold into slide 16's footer.
4. **Slide 11** (the tail): open slide 12 with "ten pairs carry half the epoch"
   instead.

Do not cut slides 10, 12, 24 or 26. Slide 10 is why the branch exists, 12 is
the mechanism, 24 is the next run, and 26 is the question someone will ask if
you do not.

## Questions to expect

- **"So is the fine-tuning work a dead end?"** No, but it is bounded: it is
  worth +1.18 on SAM 2 at one shot, and the same axis on SAM 3 is worth +1.96.
  The parts that transfer are the two repairs, not the objective.
- **"Why keep the objectness weight if the scope subsumes it?"** Because it
  flips 16 of the 19 fence channels rather than 12 on the training pairs, at no
  measured cost. The recipe keeps it for that, not for a test-set delta.
- **"Is 78.91 on beetle a real reproduction?"** It is above the paper's 5-shot
  row by 4.7 and above their OC-CCL row by 3.3, on the paper's protocol with
  20 seeds, gate PASS 18 of 18. What it is not is the same model.
- **"Why has Major not moved?"** Every support-side, inference-side, loss-side
  and model-side lever has been measured. Rotation handling is worth 4 to 5,
  the rest is flat. The remaining candidate is the synthesized split, which is
  a data question.
- **"What would you do with one more session?"** S18, because it is built, the
  smoke is green, and it decides whether the objective is learning to track or
  learning to carry information across the cycle.
- **"Are the 5-shot cross-family rows paired?"** Two of them are not: the S8d
  comparisons are differences of means with no interval, because S8d shipped a
  summary without per-run banking. Say so before someone finds it.
