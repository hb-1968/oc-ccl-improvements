---
title: "Talk handbook"
subtitle: "Speaking guide and technical Q&A for Hunter_SST_techniques.pptx"
author: "Hunter Bridges"
---

The deck is 34 slides: 31 presented, 3 backup. Part I is the speaking guide, one
entry per slide, giving the time to spend and what the slide has to establish.
Part II answers the implementation questions at the level of functions, formulas
and thresholds, and is what backup slides 32 to 34 point at.

\newpage


# Part I. Speaking guide

Directions for delivering `Hunter_SST_techniques.pptx`, 34 slides (31 presented, 3 backup). Not a script. Each
entry gives the time to spend, what the slide has to establish, the numbers to say out
loud, and what to leave for questions. Total as laid out is ≈22 minutes; the cut-list at
the end brings it to 20.

Conventions: `[click]` counts the build steps, so the cue column matches the deck. "Say"
means the number should be spoken, not just shown. Deltas are quoted twice, in mIoU
points and as a percentage of the arm they were added to, because a point means different
things at 55 and at 77; say whichever fits the sentence, and the ledger slide shows both. "Skip unless asked" marks material
that is defensible but not load-bearing.

---

### Part I. The baseline and the metric (slides 1 to 4, ≈3:05)

**1 · Title (35s).** Establish what the project is: a reproduction of Static Segmentation
by Tracking, plus an attempt to improve the loss it fine-tunes with. Set the frame for
the whole talk, which is a ledger: one baseline, seven techniques, each measured the same
way. Flag that the last section changes the model rather than the method, because that is
where the ordering reverses.

**2 · The task (50s, 6 clicks).** Walk the five panels left to right, one click each,
naming what each one is. Two things to establish: the method turns segmentation into
two-frame tracking, and the render is real project output rather than a diagram. Land on
the final click, which is the failure: the query is pinned the other way up and the
prediction inherits the reference's layout. Say that four of the seven techniques exist
because of this one behaviour.

**3 · The metric (45s, 8 clicks).** Define the unit before any number is spoken, because
every figure after this is in it. Build the five steps, then the worked example, then the
semantics card. The line to land is that traits are averaged, not pixels, so a hair-thin
antenna counts as much as the elytra. Say 0.655 and point at the zero: that trait was
present in the ground truth and never claimed, and it costs the image 0.13 on its own.
Mention that a trait predicted but absent from the ground truth contributes no term of its
own, and hurts only by taking pixels from traits that are present.

**4 · The premise (55s, 4 clicks).** Say all four numbers: 55.96 with spread 7.90 against
a target of 61.9 ± 3.7, and 68.54 ± 3.15 against 81.0 ± 1.0. Then the protocol line,
which is the methodological claim of the talk: twenty seeded draws, the same draws for
both arms, paired per seed, a bootstrap interval, and a band recorded before the run.
Close on the spread, and give the reason it is treated as a result: it measures
sensitivity to which reference was drawn, so a matched mean with double the variance is
not a reproduction.

---

### Part II. Support-side techniques (slides 5 to 9, ≈4:10)

**5 · Curation (55s, 5 clicks).** One click per step. The point to emphasise is the
threshold discipline: 55.6 degrees is the training pool's own circular standard
deviation, so it is a property of how specimens are pinned rather than a value chosen for
its result. Mention that a test pins it. The train-only step is worth ten seconds because
a reviewer will otherwise ask about leakage.

**6 · The draw (45s, 3 clicks).** Frame it as a defect that curation itself introduced.
The instructive part is the middle click: stratified random sampling fails here because
twenty draws over twelve species is a coupon-collector problem, and three species went
unused when measured. Round-robin is the fix. Do not spend time on the seeding details.

**7 · What those two bought (55s, 2 clicks).** Say 55.96, 58.54, 63.20, which is +7.24
or +12.9%, and the spread sequence 7.90, 5.77, 3.89 against the target's 3.70. Then the per-trait breakdown, which
is what makes this mechanistic rather than an average that moved: head 43.6 to 55.9 for
curation, pronotum plus 8.7 for balancing. Each fix moved the trait its own diagnostic
predicted.

**8 · Placement curation (50s, 4 clicks).** Establish that butterflies have a different
defect from beetles, so the same rule takes a different statistic. Say the result, 68.54
to 70.30, which is +1.90 or +2.8%, and note that the spread change, 3.15 to 1.36, matters
more than the mean. Close
on the disjointness of the two criteria, which is the evidence that both mechanisms are
real.

**9 · More references (45s, 3 clicks).** Contrast the two datasets: beetle +9.96, which
is +15.8%, against butterfly +1.48, which is +2.1%. Explain the asymmetry rather than just reporting it. Averaging over
five references fixes defects whose cause is the reference draw, so it fixes beetle and
not butterfly. Note the ninefold spread collapse on butterfly as the signature of a
defect that is invariant to reference choice. This is the transition into Part III.

---

### Part III. The query-side defect (slides 10 to 11, ≈1:30)

**10 · The flip (50s, 3 clicks).** Say 37 percent, 74.95 upright, 0.48 inverted, and that
the weighted combination reproduces the measured 47.4 to two decimals. The strongest
evidence is the second click, the copy baseline scoring zero with no model involved,
which makes it a data property. Third click explains why nothing reference-side moved it.

**11 · The general form (40s, 2 clicks).** One correlation, minus 0.87 over five hundred
images, and the quartile spread from 82.8 down to 50.8. Establish two things: the top
quartile already exceeds the target, and the test set cannot be curated. Skip unless
asked: only 46 percent of the displacement is a global offset, so cropping would not fix
it.

---

### Part IV. Orientation TTA (slides 12 to 16, ≈4:15)

This is the centre of the talk. Do not rush it, and do not cut it.

**12 · Section title (15s).** One sentence of framing: the largest single gain, and the
least machinery.

**13 · Mechanism, part 1 (60s, 6 clicks).** The left column builds the two passes, the
right column the three properties. Emphasise that image and mask rotate together, that
only the reference rotates, and that the rotated variant needs its own cache key.
Mention that the bottom-left panel is the project's own render rotated by the same
operation the code applies, so nothing here is an illustration.

**14 · Mechanism, part 2 (60s, 5 clicks).** Read the criterion off the code block rather
than paraphrasing it. Two properties deserve a sentence each: no ground truth is
consulted, so the protocol stays one-shot, and caching both orientations keeps the cost at
two passes rather than ten. Close on the limitation, which is that it selects between two
predictions and cannot improve either.

**15 · Results (60s, 2 clicks).** Say +5.44 (+7.6%) and +4.88 (+6.9%) on butterfly, then
+2.04 (+2.8%) on beetle at five-shot and +0.66 (+1.0%) at one-shot. On the right panel,
say 47.40 to 73.27, a gain of +25.87 or +54.6%, against a ceiling of 74.95, which is 94
percent of the deficit. Note that the silhouette
score rises with it, so this is re-segmentation rather than better assignment. Second
click is the variance cost. Skip unless asked: the rotated-chosen counter postdates this
bundle, so the recovery mechanism is consistent with but not proven to be the flipped
cohort.

**16 · The 2×2 (50s, 3 clicks).** Establish the second, reference-side fix and that
attention selects rather than blends. Then the square, all against the same 71.78 base:
+4.02 (+5.6%), +5.44 (+7.6%), +4.80 together, an interaction of −4.66 or −6.5%. The procedural lesson is worth stating explicitly, that a one-dimensional
sweep would have called both additive and shipped the worse arm.

---

### Part V. The loss (slides 17 to 23, ≈5:00)

**17 · Section title (15s).** Mark the shift: everything before this is inference-time,
this is the only technique that changes weights.

**18 · The cycle (55s, 6 clicks).** Walk the four legs with the panels. Establish the
appeal, which is that supervision comes from a mask you already have, so it can train on
unlabelled queries.

**19 · Two preconditions (55s, 4 clicks).** The differentiability fix first: thresholding
severs the graph, the reparameterisation reconstructs the same values on a linear path,
and the forward-to-reverse ratio goes from 0.00 to 26, and to 351 per-trait. Then the
supervision axis, and the law it implies: each objective pays only where it has room,
which predicts butterfly gains and beetle does not.

**20 · SCNP (60s, 4 clicks).** Cover the transform, then why it was a candidate here (the
cycle supplies a genuine one-hot target), then both attempts. Say minus 0.09 significant
on the merged form, and that the module's own documentation pre-registered beetle as the
detrimental case before the run. For the composition, say the prediction was plus 0.5 to
plus 2 and the measurement was minus 0.01, inside the recorded band. If time is short,
compress the transform and keep the pre-registration.

**21 · Every loss variant (55s, 2 clicks).** Do not enumerate the arms; the figure does
that. Establish the pattern on beetle, name the two arms that hurt badly (full
fine-tuning at minus 3.95, the margin fold at minus 3.62), and say why the margin fold
failed, which is that training the statistic reshapes the masks TTA chooses between.
Note explicitly that this slate is on SAM 2, because Part VI reverses it.

**22 · The pot (50s, 2 clicks).** The correction to the earlier headroom figure: the
quoted number was a correlate, the measured ceiling is 2.65 on beetle and 0.38 on
butterfly. Then the sentence the last section depends on, that the remainder is regions
the tracker never proposes. Flag it as something to remember.

**23 · Scoring (35s, 2 clicks).** Three readings of the same predictions: 63.20, 64.69,
65.20. The point is that two points of spread exist with no model change, which is why
every fine-tune delta was re-measured under all three. Mention that the model-swap deltas
are larger and do move between readings, and that both are quoted. First candidate for
cutting.

---

### Part VI. The model swaps and close (slides 24 to 31, ≈7:15)

**24 · The ledger (60s, 1 click).** Pause and let the audience read before speaking. Every
row carries both readings, the mIoU delta and the same gain as a percentage of the arm it
was added to; the percentages are what make a +1.59 on beetle and a +1.90 on butterfly
comparable. Then
give the shape: on SAM 2 everything that paid changed what the model is shown. Point out
the two largest negatives as compositions assumed safe, the three new positive rows as the
model swap, and the bottom row as the other model's headline feature. Leave this slide up
during questions.

**25 · Where the numbers land (40s, 2 clicks).** Beetle passes both gates, butterfly does
not, and roughly five of butterfly's last six points sit below the silhouette ceiling.
That is the argument for changing the model rather than adding method.

**26 · The two swaps (50s, 4 clicks).** Introduce both families and what was held fixed.
The cross-family pairing discipline is worth one sentence for this audience: the draws are
model-independent and hash-pinned, the registry refuses such comparisons by default, and
this stage uses one whitelisted relaxation that records what it ignored. Then the new arm,
LoRA on the memory attention.

**27 · SeC-4B (70s, 3 clicks).** Read the deployment row first: −2.79 on butterfly and
−3.02 on beetle, which is −3.6% and −4.7%. Note its oracle pot is in the same range, so
the proposal boundary did not move. Then the concept injection, −5.92 and −6.23, which is
−8.6% and −10.0%, with 0 and 1 of 20 seeds improving. Report the falsified pre-registration directly: the native gate was
predicted quiet, and it fires on 0.585 of butterfly episodes and 0.869 of beetle ones, so
the gate-off cells are the charitable reading. Close on the one positive, the loss stack
at plus 0.60 and plus 0.58.

**28 · SAM 3 (75s, 2 clicks).** Four rungs, every one significant: +2.14 (+3.4%), +0.99
(+1.5%), +1.59 (+2.4%), +1.96 (+2.9%). Then the headline, 64.12 to 69.88, which is +5.77
or +9.0% against the SAM 2 stack, and 70.56 under the paper's own full metric. Second click is butterfly as the control, flat at 76.5 to 76.6,
which is the third independent confirmation that its residual is protocol rather than
capacity.

**29 · Why it worked (65s, 3 clicks).** This is the analytical payoff, so give it room.
SAM 3's own assignment pot is smaller than SAM 2's, 1.80 against 2.65, yet the stack and
the arch arm delivered 3.55 points, or +5.4% on the arm they were added to, which is only
possible if the ceiling moved. It did, 66.76 to
71.08. Connect it back to slide 21 explicitly. Add that the loss arms were written on
per-object logit stacks so they could transfer, which is why the port was cheap.

**30 · What is next (55s, 3 clicks).** Present the projection as a projection. The bands
come from measured components, two independent caps agree near 81, and there is a
pre-registered disqualifier at 72 for the five-shot base. Mention the conditioning-frame
cap of four as a known trap that is configuration rather than weights. Second candidate
for cutting; it is the most likely thing to be asked about anyway.

**31 · Close (45s, 4 clicks).** One line per click. The through-line to land is that the
ordering reversed: on SAM 2 the objective was the wrong place to spend effort, on SAM 3
the objective plus the proposal-forming arm is most of the gain, and butterfly's residual
survived both swaps.

---

### Backup slides (32 to 34)

Not presented. They exist for the implementation questions, which is what got asked last
time. Jump to them rather than describing the code from memory.

- **32, canonicalization geometry.** The three criteria as computed: the directed
  head-to-tail centroid vector, the centroid displacement against a per-trait median
  reference, and completeness, plus the 180 degree test used to build strata.
- **33, pool to draw.** `canonical_pool` to `draw_supports` to the seeding, with the
  round-robin rule, the orientation-diverse proportion, and the support-set hash.
- **34, what guards it.** The tests that pin the threshold, the leakage check, the
  dispatch check, and the list of fields recorded in every bundle.

`TECHNICAL_QA.md` is the long form, organised by question. Sections 1 to 3 back slide 32,
4 to 8 back slide 33, 9 to 12 back slide 34. Its last section is a set of one-line answers
for questions asked in passing.

Two places in the main talk are worth an explicit offer to go deeper, because they are
where the question usually comes from: slide 5 (the threshold, where "is that tuned?" is
the natural follow-up) and slide 8 (placement, which is the least familiar of the three
criteria to an audience).

### Cutting to 20 minutes

In order:

1. **Slide 23**, scoring, 35s. The point survives as one sentence on slide 24.
2. **Slide 30**, the projection, 55s. Keep it as a backup for questions.
3. **Slide 11**, displacement quartiles, 40s, if slide 10 landed clearly.
4. **Slide 6**, the round-robin, 45s. Mention it in one line on slide 5.

Do not cut 3, 13 to 15, 20, or 28 to 29.

### Questions to expect

**Why does the loss work on SAM 3 when it was null on SAM 2?** The binding constraint
changed. On SAM 2 every loss arm was bounded by a 2.65-point assignment pot, about 4% of
the score. The arch arm
is not an assignment method: it trains the module that forms proposals, and the oracle
ceiling moved 4.3 points, which is the direct evidence.

**Is the cross-family comparison legitimate?** The support draws are model-independent by
construction and hash-pinned, so both families saw the same twenty references. The
registry refuses cross-family pairings by default; this stage used one whitelisted
relaxation, and every row records the fields ignored and both their values.

**Why is the confidence criterion trustworthy?** It is not, entirely. That is the variance
cost on slide 15, and on one family and dataset combination it goes negative, SeC on
beetle at minus 1.80. It is right often enough to be worth 5.4 points net on butterfly.
Improving the criterion is open headroom.

**Is curating the reference pool consistent with a one-shot protocol?** Curation reads
training masks only, never test. The uncomfortable part is the accounting: the best
butterfly arm reads about 1,700 labelled masks to choose one reference. The defensible
version is that curation stands in for human judgement, and a human does not need 1,700
masks. It is testable by shrinking the pool until the drawn set stops changing.

**Could the SAM 2 nulls have been an implementation fault?** The gradient measurement on
slide 19, the training diagnostics, and an exact checkpoint round-trip all say the
machinery works, and the same machinery produces significant positives on SAM 3.


\newpage


# Part II. Technical Q&A

Answers for the "how did you actually implement this" questions, at the level of
functions, formulas, thresholds and guards. Ordered so the first eight sections cover
canonicalization, which is what backup slides 32 to 34 show. Section 10b covers the metric
on slide 3. Every claim here names the
file it lives in, so you can open the source if pressed.

**How deltas are quoted.** Every gain appears twice: in mIoU points, and as a percentage
of the arm it was added to. The percentage is the honest comparator across datasets,
because a point is worth more at 55 than at 77. So orientation TTA's +5.44 on butterfly is
+7.6% of the 71.78 it was added to, while the loss stack's +1.59 on SAM 3 beetle is +2.4%
of 66.33. Both readings come from the same paired per-seed measurement; nothing is
rescaled.

Source layout referenced throughout:

```
src/occl/curation.py     the criteria, the pools, the draws
src/occl/seeding.py      seeds, prefix nesting, support-set hashing
src/occl/adapters/tta.py orientation TTA
src/occl/finetune/       the cycle loss and its arms
src/occl/registry.py     comparability fields and paired comparison
tests/test_curation.py   the guards named below
```

---

### 1. How is a specimen's orientation measured?

`curation.body_axis_angle(labelmap, head_id, tail_id)`.

Take the centroid of the head trait and the centroid of the tail trait, then the signed
angle of the vector between them:

```python
hy, hx = np.nonzero(labelmap == head_id)
ty, tx = np.nonzero(labelmap == tail_id)
angle = np.degrees(np.arctan2(ty.mean() - hy.mean(), tx.mean() - hx.mean()))
```

Returns `None` if either part is absent, and that specimen is rejected by the `no_axis`
counter rather than silently treated as canonical.

**Why a directed trait-pair vector and not PCA on the silhouette?** Because the failure
being filtered is a 180 degree inversion, and a principal axis is undirected: a specimen
and its inversion have the same principal axis, so PCA cannot see the defect at all. The
directed vector differs by exactly 180 degrees between them. There is also a measurement
reason: on spread butterfly wings the silhouette is nearly isotropic, so a principal axis
is unstable. Measured with the trait-pair vector, butterfly orientation is extremely
well defined (R between 0.993 and 0.9996, circular std 1.6 to 6.9 degrees), which is how
we established that butterflies are not mis-oriented and need a different criterion.

**Which two traits?** The dataset declares them as `axis_parts`. Datasets without that
attribute skip the criterion entirely, and the bundle records `criteria_applied` so a
skipped criterion is visible rather than implied.

---

### 2. Where does the 55.6 degree threshold come from?

It is the training pool's own circular standard deviation, computed on the same angles
the filter is about to test:

```python
R      = hypot(mean(sin θ), mean(cos θ))      # resultant length, 0 to 1
std    = degrees(sqrt(-2 * ln R))             # circular std
accept = angular_deviation(angle, circular_mean(train)) <= std
```

`circular_mean` and `circular_resultant` are used rather than an arithmetic mean because
angles wrap: the arithmetic mean of -179 and +179 is 0, which is the opposite direction
from the truth. `angular_deviation` wraps the difference into [0, 180].

For beetle this evaluates to 55.6 degrees. It is a property of how the specimens happen
to be pinned, not a number chosen to hit a target. Two things enforce that:

- `tests/test_curation.py::test_threshold_is_the_a_priori_training_circular_std` pins the
  rule, so a later edit cannot quietly substitute a tuned constant.
- The bundle records `threshold_deg`, `train_circular_mean_deg`, `train_resultant_R` and
  a literal `threshold_rule` string, so an auditor can recompute it.

If the arm had missed its target, the project's stated position was to report the miss
rather than move the threshold.

---

### 3. What exactly does placement curation measure, and why is it not orientation?

`curation.trait_centroids` plus `curation.placement_displacement`.

1. For each specimen, compute the centroid of every present trait.
2. Build a reference configuration from the training split: the **median** x and median y
   per trait, taken independently. Median rather than mean so a handful of badly placed
   specimens cannot drag the reference toward themselves.
3. For a candidate, take the mean Euclidean distance between its centroids and the
   reference's, over the traits the two configurations share. Returns `None` if they
   share none.
4. Accept if that displacement is at most `train mean + 1 train std` of the same
   quantity. Same discipline as the orientation threshold: derived from the training
   distribution, recorded in the bundle as `placement_threshold_px`.

**Why a separate criterion?** A specimen can be perfectly upright and still be in the
wrong corner of the sheet. Angular deviation is zero in that case and displacement is
large, and it is the displacement that breaks mask propagation, because the tracker
carries the reference's spatial layout onto the query. The numbers: on cyrbia the
correlation between run mIoU and support-query displacement is -0.988, against the -0.710
orientation signal that motivated the beetle arm.

**Why butterflies need it and beetles do not.** Cambridge butterfly frames are dissected
wings laid loosely on a sheet. Cyrbia's trait centroids scatter with a 219.4 px standard
deviation against 36 to 60 px for the other four subspecies, and splitting cyrbia by
background reveals two imaging protocols filed under one name: green-background frames
scatter the wings at 209.9 px, neutral frames sit at 27.7 px like everything else.

---

### 4. How is the 180 degree cohort identified?

`curation.orientation_of(centroids, reference, size)`, used to build strata rather than to
filter.

Compare the candidate's centroid configuration against two things: the reference
configuration, and the reference **mirrored about the image centre** (`size - x`,
`size - y`). Whichever is closer in mean distance wins, giving `"upright"` or `"rot180"`.

One implementation detail worth stating if asked, because it was a real bug: `size` is the
mask's own extent, `max(lm.shape)`, not a hardcoded 1024. Butterfly masks are natively
1024 square, but beetle crops are 172 to 431 px. With 1024 hardcoded, the mirrored
reference lands outside the image for beetle and every specimen scores as upright,
turning the criterion into a silent no-op. Pinned by
`test_orientation_is_measured_against_the_masks_own_extent`.

The reference is built from the **training** split only, and the whole map is memoised per
dataset instance because `orientation_diverse_supports` calls it once per run with an
identical pool. Without the memo that is roughly 15 minutes of single-threaded mask
reading per stage on Butterfly Major.

---

### 5. How does the pool become a draw?

```
canonical_pool(ds, group, criteria)   ->  (pool_ids, stats)     train masks only
draw_supports(ds, group, pool, selection, shot, run_idx)        single dispatch
  |- canonical_balanced      -> stratified_supports(...)
  |- orientation_diverse     -> orientation_diverse_supports(...)
  '- everything else         -> seeding.sample_supports(...)
```

`canonical_pool` applies completeness first (cheapest), then orientation, then placement,
counting rejections by reason. It raises if the pool comes out empty rather than falling
back to the unfiltered split.

`draw_supports` exists because this dispatch was duplicated in three places and the copies
drifted on the `orientation_diverse` arm: `protocol.build_run` used the orientation-diverse
draw while `build_cache.required_butterfly` and `evaluator.distinct_images` fell through to
uniform sampling. The consequences, in the order they bite: the cache is pre-built from a
support set the run never asks for; the preflight reports the wrong images as missing, so
topping up from its own error message does not fix it; and `precompute` warms embeddings
no episode uses while every real support misses the cache.

---

### 6. How is the sampling made reproducible and paired?

`seeding.derive_seed(dataset, group, shot, run_idx)` is a blake2b digest of a
human-readable key string, reduced to 31 bits. No global RNG state is involved anywhere.

Two properties do the work:

- **Model independence.** The seed depends only on dataset, group and run index, never on
  the adapter, dtype or device. Run 7 of Butterfly Major draws the same reference for
  pretrained SAM 2, every OC-CCL variant, and SAM 3. That is what makes the comparisons
  paired, which matters because support difficulty is the dominant variance component and
  cancels exactly when you difference per seed.
- **Prefix nesting.** `sample_supports` passes a literal `0` in the seed's `shot` field and
  slices `[:shot]` from one permutation, so the 20 one-shot references are exactly the
  first elements of the 20 five-shot sets. The 1-shot to 5-shot comparison is therefore
  paired too. Pinned by `test_balanced_selection_preserves_prefix_nesting`.

`sorted()` before permuting is load-bearing: filesystem ordering is not stable across
machines.

`seeding.support_set_hash` hashes the **resolved** IDs per run, not the seed, because two
runs can share a seed and still draw different references if one image failed to download
and was filtered from `train_ids`. That hash gates whether two results may appear in the
same table.

---

### 7. Why round-robin rather than stratified random sampling?

`curation.stratified_supports`. Run `r` takes stratum `(r + k) mod S` for the k-th
reference, and the member within that stratum is drawn by the run's seed.

The non-obvious part: stratified **random** sampling does not fix the coverage problem at
this sample size. Twenty independent draws over twelve species is a coupon-collector
problem, and measured, it still left three species unused, 9 of 12. Equal probability is
not equal representation at n = 20. Round-robin gives 12 of 12 with usage 1 to 2, keeps
prefix nesting, and makes a 5-shot draw span five distinct species.

It raises if `shot` exceeds the number of strata rather than silently repeating one.

Guard: `test_round_robin_covers_every_species_where_random_sampling_does_not`.

---

### 8. How is the orientation-diverse draw constructed?

`curation.orientation_diverse_supports`. Split the pool by the stratum map from section 4,
then take `n_rot = round(shot * rotated_fraction_of_pool)`, clamped to `[1, shot - 1]`, so
both orientations are always represented at 5-shot. Members within each orientation are
drawn by the run's seed in a deterministic class order, and the set is topped up from the
whole pool if a class is short, keeping the count exact.

The proportion follows the **train** prior (cyrbia: 34 percent rotated, so 2 of 5), never
the test distribution, which would be leakage. Where a group has only one orientation the
function degenerates to the ordinary draw, so it is a no-op everywhere except cyrbia.

---

### 9. Is any of this leaking test information?

Three separate answers:

- Every reference criterion is computed from `ds.train_ids(group)`. Never test.
- `test_pool_is_estimated_from_train_only` perturbs `ds.test_ids` and asserts the pool is
  unchanged. Note where the memoisation lives: `support_pool` caches, `canonical_pool`
  stays pure, precisely so that the leakage test calls an uncached function and cannot
  pass vacuously.
- Estimating on train costs nothing anyway: train circular mean 67.1 degrees against test
  56.8, concentration R 0.625 against 0.614.

The uncomfortable part of the accounting is not leakage but **label budget**: the best
butterfly arm reads about 1,700 training masks to choose one reference, against a headline
of one labelled image. The defensible framing is that curation stands in for a human
curator's judgement and a human does not need 1,700 masks; the test is to shrink the pool
until the drawn set stops changing. That is written down as future work, not claimed.

---

### 10. How do you stop two incompatible arms being compared?

`registry.COMPARABILITY_FIELDS` includes `protocol.support_selection`, so
`registry.compare()` raises on a uniform-versus-canonical pair: they draw different
references by construction, and differencing them measures the draw rather than the
technique. `data.interp`, `data.cache_format` and `model.tta` are in the same set, each
added after a near miss where an arm would otherwise have been classified as comparable.

For the cross-family comparisons in the model swap, there is exactly one sanctioned
relaxation: `compare(..., ignore_fields=("model.family", "model.image_size"))`, which pins
`support_set_hash` equality and embeds the ignored fields and both their values in every
row it produces. A cross-family number that cannot say it is one does not appear.

Guards: `test_registry_refuses_to_pair_the_two_canonical_arms`,
`test_registry_refuses_to_pair_uniform_against_canonical`.

---

### 10b. How is mIoU computed here, exactly?

`metrics.score_episode` is the only entry point, and nothing else in the package computes
IoU. The order of operations is fixed and enforced structurally: **resize logits, then
argmax, then IoU**. Argmax first would nearest-neighbour a label map and change the answer.

1. Resize the (C, H, W) logit stack bilinearly to the ground truth's own resolution
   (`iou_space="gt_native"`, the default). The alternative, resizing the label map up to
   the model's resolution, is available as `"model"` for repo compatibility and is not
   what we report.
2. `logits_to_labelmap` gives each pixel its highest-scoring trait, and background where
   that best logit is at most 0. That is the paper's "highest score label" overlap rule,
   and it is equivalent to appending a constant-zero background channel without
   materialising it. It holds because SAM 2 and SAM 3 mask logits are calibrated so that
   greater than zero means inside the object. Ties go to the lowest channel index, so the
   result is deterministic given a fixed `class_ids` order.
3. `per_image_miou` takes IoU per trait **present in this image's ground truth**, then the
   unweighted mean over those traits. Background is never a term.
4. The run score is the mean over images; the reported number is the mean over 20 runs,
   with the spread reported beside it.

**Semantically**: 1.00 means every pixel of every trait is correct. Because the mean is
over traits rather than pixels, a hair-thin antenna counts as much as the elytra, which is
why beetle scores lower than its silhouette quality suggests.

Three edge cases, each decided deliberately:

- A trait predicted but **absent from the ground truth** contributes no term. It still
  hurts indirectly, because argmax gave it pixels that present traits wanted.
- A trait **present in the ground truth but never prompted**, because the drawn reference
  did not contain it, scores 0 and is counted. That is a real failure of the one-shot
  draw and a main driver of the run-to-run spread. Excluding it would shrink our spread
  below the paper's.
- A trait in the reference but **empty in this ground truth** is excluded, symmetric with
  the first case.

`iou_binary`, the merged-foreground IoU, is computed alongside for continuity with the
upstream harness, but it is not the paper's metric and is never the headline. It is what
the talk calls the silhouette score.

Worked example, slide 3, from `s4c.zip`'s `per_query.csv`, beetle
`impunctatus/beetle_1166_231`, run 0: elytra 0.925, pronotum 0.916, head 0.802, legs
0.630, antenna 0.000. The image scores the mean of those five, 0.655. The antenna alone
costs 0.13.

---

### 11. How is TTA implemented, and how is it kept honest?

`adapters/tta.py`, a wrapper around the tracker adapter rather than a modification of it,
so every previously recorded number stays byte-stable.

- `rotate_episode` applies `np.rot90(x, 2)` to the reference image and its labelmap
  together, leaves the query untouched, and suffixes the support IDs with `@r180`.
- The suffix is the embedding-cache key. Without it the second pass is served the upright
  image's features, silently. `test_tta_precompute_caches_both_orientations` pins it.
- The decision is `confidence(logits) = top[top > 0].mean()` where `top = logits.max(0)`,
  returning `-inf` when nothing is claimed so an empty prediction always loses. No ground
  truth is consulted, which is what keeps the protocol one-shot.
  `test_tta_picks_the_more_confident_orientation_without_ground_truth` pins that.
- `precompute` is served dynamically through `__getattr__` rather than declared as a
  method, so `hasattr(wrapper, "precompute")` mirrors the inner adapter exactly. A
  class-level method made the wrapper always claim the capability, which crashed on an
  adapter without one.

---

### 12. What is actually recorded per run?

Every bundle carries the curation `stats` verbatim: `criteria`, `criteria_applied`,
`pool_size`, `train_size`, `rejected` counts by reason, `axis_parts`, `expected_traits`,
`threshold_deg`, `train_circular_mean_deg`, `train_resultant_R`, `threshold_rule`, and for
butterfly `placement_threshold_px` and `train_placement_mean_px`.

The point of recording the rule string as well as the value is that the approximation
stays auditable after the fact. Curation approximates a human judgement the source paper
never defines, so all three support arms are always reported rather than only the curated
one.

---

### Short answers, if the question is asked in passing

- **"Is the threshold tuned?"** No. It is the training pool's own circular standard
  deviation, recorded per run and pinned by a test.
- **"Why not PCA for the axis?"** A principal axis is undirected and cannot distinguish a
  specimen from its 180 degree rotation, which is the exact defect being filtered.
- **"Why median for the placement reference?"** So a few badly placed specimens cannot
  drag the reference toward themselves.
- **"Does curation touch the test split?"** No, and a test asserts the pool is unchanged
  when the test split is perturbed.
- **"How do you know the 20 draws are the same across models?"** The seed is a hash of
  dataset, group and run index only, and the resolved IDs are hashed per run into
  `support_set_hash`, which gates pairing.
- **"Does TTA see labels?"** No. The choice is the mean top logit over claimed pixels.
- **"Is mIoU averaged over pixels or traits?"** Traits, unweighted, over the traits present
  in each image's ground truth. A missed antenna costs as much as a missed elytra.
- **"What happens to a trait the reference never showed the model?"** It scores 0 and is
  counted, because that is a real failure of the one-shot draw.
