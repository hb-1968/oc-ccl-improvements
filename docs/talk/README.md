# The 20-minute talk

**`Hunter_SST_techniques.pptx` is the current deck** — 34 slides (31 presented,
3 backup), ~22 min as written
(the script's cut-list lands it at 20), click-built, and current as of the S14sam3 /
S14sec bundles.
`Hunter_SST_talk_handbook.pdf` combines the two documents below into one printable
file. `TECHNICAL_QA.md` answers the "how did you implement this" questions at the level of
functions, formulas, thresholds and guards, ordered so sections 1–3 back slide 32, 4–8
back slide 33, and 9–12 back slide 34. Read it before presenting; it is what you reach for
when a question goes past the slide.

`NARRATION.md` is the speaking guide, not a script: per slide it gives the time to spend,
what the slide has to establish, which numbers to say out loud, and what to leave for
questions. The deck's presenter notes carry the same directions in one short paragraph
each. `Hunter_SST_techniques.pdf` is the deck
flattened (all build steps visible at once) for handout or projector fallback.

`Hunter_SST_OC-CCL_findings.*` is the earlier cut, kept for reference. It told the
reproduction story and spent two slides on paper-vs-release discrepancies; the current
deck drops that thread entirely and walks the techniques instead.

## What the deck argues

One baseline — uniform supports, no technique, Beetle 55.96 ± 7.90 and Butterfly Major
68.54 ± 3.15 — then each technique added to it and measured the same way: 20 seeded
draws, the same draws for both arms, paired per seed, bootstrap CI, band frozen before
the run. Slide 24 is the ledger those measurements add up to, and slide 3 defines the unit
they are all in: mIoU per trait present in an image's ground truth, averaged over traits,
then images, then the 20 runs.

Structure:

| slides | part |
| :-- | :-- |
| 1–2 | the task |
| 3 | **the metric** — how one image is scored, and what a number means |
| 4 | the baseline everything else is measured against |
| 5–9 | support-side techniques: curation, round-robin balancing, placement, more shots |
| 10–11 | the query-side defect that support-side techniques cannot reach |
| 12–16 | **orientation TTA** — mechanism (2 slides), results, and the redundancy warning |
| 17–19 | the loss: the cycle and its two preconditions |
| 20 | SCNP — the borrowed topology term, tried twice, closed twice |
| 21–22 | every loss variant, and the oracle ceiling that explains them |
| 23–25 | scoring as a technique, the ledger, where the numbers land |
| 26–29 | **the model swaps** — SeC-4B (doesn't pay), SAM 3 (does), and why: the ceiling moved |
| 30–31 | the pre-registered 5-shot projection, close |
| 32–34 | **backup, not presented** — how canonicalization is implemented, and what guards it |

The last act is the S14 result. SeC-4B loses under the deployment stack (−2.79 ✱ /
−3.02 ✱) and its concept injection costs ~6 points; SAM 3 pays on beetle in four
composing rungs (backbone +2.14 ✱, TTA +0.99 ✱, our loss stack +1.59 ✱, the new
architecture arm +1.96 ✱), taking beetle's best 1-shot number from 64.12 to **69.88**
(+5.77 ✱). The mechanism slide is the one that matters: SAM 3's assignment pot is
*smaller* than SAM 2's (1.80 vs 2.65), so the gains are not assignment-bounded — the
oracle ceiling itself moved, 66.76 → 71.08, because the arch arm trains the
proposal-forming module. Butterfly is flat across both families, confirming its residual
is protocol, not capacity.

TTA gets the deepest treatment: `rotate_episode` (image and labelmap rotated together,
query untouched, its own `@r180` cache key), the decision rule (`top[top > 0].mean()`,
−inf when nothing is claimed, no ground truth consulted), what it costs (2× inference,
kept at 2× rather than ~10× by caching both orientations), what it bought (+5.44 / +4.88
butterfly, +2.04 beetle at 5-shot, cyrbia 47.40 → 73.27), and what it costs in variance
(cyrbia per-run spread 6.81 → 10.89).

## Where the content comes from

Nothing is drawn or invented. Every figure is either

* **a render this project already produced** — `results/s3/overlay.png`, cropped into its
  five panels by `make_talk_figures.py::f11_overlay_panels`, and rotated 180° for the TTA
  mechanism slide by `f14_rotated_panels`, which is the same `np.rot90(·, 2)` the adapter
  applies; or
* **a plot of numbers lifted live from a landed bundle** in `results/` and
  `results/inbox/` — and where a README constant and a bundle value both exist, the script
  `assert`s they agree before plotting (`close()`).

Shapes are used only to frame, number and point. There is no generated imagery.

## The documents as PDF

`./docs_to_pdf.sh` renders everything to PDF beside the sources, via pandoc and xelatex.
xelatex rather than pdflatex because the documents contain the significance mark, arrows,
degree signs and a true minus sign. Re-run it after editing any markdown here.

**`Hunter_SST_talk_handbook.pdf` is the one to print**: the speaking guide and the
technical Q&A in a single document, with a contents page that lists every slide entry and
every question. `make_handbook.py` builds it by demoting each source document's headings
one level under a part heading, so the two stay authored separately and combine on
render. The individual `NARRATION.pdf` and `TECHNICAL_QA.pdf` are still produced if you
want one without the other.

## House style

The deck copies the visual system of `Hunter_Bridges_Introduction.pptx`:

* **Courier New throughout**, at every level. Titles 30pt bold, eyebrow 12pt bold
  uppercase with 2pt tracking, body 11 to 14pt.
* **Black on white, with one deliberate exception.** Photographs go through luminance
  (`make_talk_figures.py::_to_grey`). **Label maps keep their colour**, because the hue
  *is* the trait identity: head, pronotum, elytra, antennae and legs are only
  distinguishable by it, and a grey ramp cannot carry five categories at slide scale.
  The rule the audience can infer: colour on a slide always means "this encodes a
  trait". Charts, chrome and type stay monochrome.
* **Square boxes**, white fill, 1.75pt black outline. No rounded corners, no tints, no
  shadows.
* **A full-width black rule under every title**, at y=1.62in, exactly as the reference
  deck sets it. Nothing may sit on or above it; `build_deck.py` clamps figures, pictures
  and cards to y ≥ 1.95in so this cannot regress.
* Charts follow: black fill for the subject, mid grey for a comparator, white with a
  black edge for a negative, hatching for a pre-registered band. Sign reads from which
  side of the zero line a bar sits on, not from colour.

Courier is about 25% wider than a proportional face, so the type scale was rebuilt for
it and titles are kept to one line. `check_overlaps.py` measures in DejaVu Sans Mono for
this reason.

Slide text is plain and technical: statements of fact with the numbers carrying the
weight, no aphorisms, and **no em-dashes anywhere** (slides, figures, presenter notes or
the speaking guide). `retone.py` records the rewrite that established this, and
`geo_fix.py` the geometry pass that followed the restyle. Check with:

```bash
python -c "from pptx import Presentation; d=Presentation('Hunter_SST_techniques.pptx'); \
print(sum('—' in sh.text_frame.text for s in d.slides for sh in s.shapes if sh.has_text_frame))"
```

## Rebuilding

```bash
cd docs/talk
python make_talk_figures.py --out figures            # figures from results/
python build_deck.py                                 # writes the .pptx + anim_spec.json
python add_animations.py Hunter_SST_techniques.pptx anim_spec.json \
       Hunter_SST_techniques_anim.pptx               # injects the <p:timing> builds
```

```bash
python check_overlaps.py Hunter_SST_techniques_anim.pptx deck.pdf   # layout sweep
```

`check_overlaps.py` is the layout regression check — run it after any edit. It measures
each text block's real rendered height with font metrics, then flags text that spills
onto the shape below it, text crossing a card boundary, and (from the exported PDF's word
boxes) any two lines whose ink actually overlaps. One entry is a known false positive and stays in the
output: slide 18's "prompt back" label sitting above its arrow, verified by eye. The `edge` rows are the 0.33in eyebrow labels, which is
deliberate.

`python-pptx` has no animation API, so `add_animations.py` writes the OOXML timing tree
directly: one fade entrance per shape, grouped into clicks by `anim_spec.json`
(`{slide_index: [[shape_id, ...], ...]}`). Slides 7, 21 and 25 build a chart by stacking
two pixel-identical PNGs and revealing the later one — which is why those figures are
saved with `tight=False`.

## When S15 lands

The 5-shot run on the champion is the next thing that changes this deck.

0. Backup slides 32–34 are static text and do not need rebuilding unless
   `src/occl/curation.py` changes; if it does, re-read it against `TECHNICAL_QA.md`.
1. **Slide 30** (`f22_projection.png`) — the projection bands become measurements. Edit
   `make_talk_figures.py::f22_projection` to plot the landed values beside the bands
   rather than replacing them; a projection that was written down and then scored is
   worth more than a bar.
2. **Slide 28** — if 5-shot beats it, the "beetle's best 1-shot number" line becomes a
   best-overall line.
3. **Slide 24** — a new ledger row ("more shots on SAM 3"), which will re-sort the figure.
4. **Slide 31** — the closing lines stand unless the projection misses its band.

`f10_s14.png` (the pre-registration bands) is no longer on a slide — the results
superseded it. It stays in `figures/` in case the write-up wants the pre-registration
shown beside the outcome.

If a bundle disagrees with the pins the deck quotes (76.52 / 64.12 for the SAM 2 crown,
69.88 / 76.64 for the SAM 3 arms), the gate fails before the deck is wrong:
`check_gate.py s14sam3 --bundle s14sam3_<ds>.zip`.

---

# The S16 deck: `Hunter_SST_second_reference.pptx`

A separate, shorter talk on one stage: which second reference pays, the implicit
one (orientation TTA's rotated support) or the explicit one (a real 2-shot). It
does not replace `Hunter_SST_techniques.pptx` and shares none of its files
beyond the animation injector and the layout checker.

19 slides, 17 presented and 2 backup, about 13 to 14 minutes.

## Files

| file | what it is |
| :-- | :-- |
| `make_s16_figures.py` | every figure, from `results/s16.zip` and the named README rows, into `figures_s16/` |
| `build_s16_deck.py` | the deck, writing `anim_spec_s16.json` alongside it |
| `s16_factcheck_fixes.py` | the recorded corrections from the fact-check pass |
| `verify_s16.py` | checks every decimal and percentage in the deck PDF and both markdown files against the record |
| `NARRATION_S16.md` | speaking guide, directions per slide |
| `TECHNICAL_QA_S16.md` | implementation answers, what backup slides 18 and 19 point at |
| `make_handbook_s16.py`, `docs_to_pdf_s16.sh` | the combined handbook and the PDF renders |

## Rebuild

```
python make_s16_figures.py
python build_s16_deck.py
python add_animations.py Hunter_SST_second_reference.pptx \
    anim_spec_s16.json Hunter_SST_second_reference.pptx
soffice --headless --convert-to pdf --outdir . Hunter_SST_second_reference.pptx
python check_overlaps.py Hunter_SST_second_reference.pptx \
    Hunter_SST_second_reference.pdf
./docs_to_pdf_s16.sh
python verify_s16.py
```

Same house style as the main deck: Courier New, black on white, the rule at
1.62in, colour only where a label map's hue carries trait identity, and no
generated imagery. The only photographs are panels of `results/s3/overlay.png`,
with the implicit arm's second panel being that support rotated 180 degrees.

## Claims this deck deliberately qualifies

The fact-check pass caught six overreaches inherited from the README's S16
result section, and the deck now states each the careful way. If the README is
edited later, keep these:

1. The two 1-shot TTA cells **resume** S11's banked cells (identical evaluator
   fingerprint, zero inference, bit-identical values). They certify the wiring
   of the merge, not TTA. The 2-shot cells are the only new measurement.
2. `best-of-3` is `miou_boundary.independent_best3`, so it changes the overlap
   rule **and** the boundary tolerance, not the boundary alone.
3. Beetle's share of the 1 to 5 shot lift, 58.6%, is **past** the top of its
   pre-registered 35 to 55% band. Butterfly undershot, beetle overshot.
4. 0.30 is the tightest cell **at 1 or 2 shots**; S6's Major 5-shot cells sit at
   0.15.
5. Two labels is the cheapest **training-free** point above the paper's 65.2
   row. S14sam3's arch_tta is above it at one label.
6. A second label is the strongest **per-label** lever on beetle; 5-shot is
   larger in absolute terms (+9.96).

Also: the oracle reading is not paired, so ceiling moves are differences rather
than tested effects, and +5.84 and +1.18 are unpaired against banked 1-shot
levels because those cells were deliberately not re-run.
