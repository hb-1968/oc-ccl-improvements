#!/usr/bin/env python3
"""Build the S16 deck: which second reference pays, the implicit one or the
explicit one.

Separate from Hunter_SST_techniques.pptx and built in the same house style.
Figures come from make_s16_figures.py (project data only). Click-builds are
injected afterwards by add_animations.py, driven by anim_spec_s16.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

FIG = Path("figures_s16")
OUT = Path("Hunter_SST_second_reference.pptx")

INK = RGBColor(0x00, 0x00, 0x00)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = INK
SLATE = INK

HEAD = "Courier New"
BODY = "Courier New"

W, H = 13.333, 7.5
ANIM: dict[int, list[list[int]]] = {}

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


# ── primitives, lifted from build_deck.py so the two decks stay identical ───
def slide():
    return prs.slides.add_slide(BLANK)


def txt(s, text, x, y, w, h, size=13, color=INK, bold=False, font=BODY,
        align=PP_ALIGN.LEFT, italic=False, space_after=6, anchor=MSO_ANCHOR.TOP,
        line_spacing=1.0):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
        r.font.color.rgb = color
    return tb


def title(s, text, size=30, y=0.86, rule=True):
    t = txt(s, text, 0.7, y, W - 1.4, 0.7, size=size, bold=True, font=HEAD,
            color=INK)
    if rule:
        r = shape(s, MSO_SHAPE.RECTANGLE, 0.72, 1.62, 12.0, 0.03, fill=INK)
        return t, r
    return t


def kicker(s, text, y=0.5):
    tb = txt(s, text.upper(), 0.7, y, W - 1.4, 0.32, size=12, bold=True,
             font=HEAD, color=INK)
    for p in tb.text_frame.paragraphs:
        for r in p.runs:
            r.font._rPr.set("spc", "201")
    return tb


def pic(s, name, x, y, w=None, h=None):
    y = max(y, 1.95)
    im = Image.open(FIG / name)
    ar = im.width / im.height
    if w is None:
        w = h * ar
    if h is None:
        h = w / ar
    return s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y),
                                Inches(w), Inches(h))


def figure(s, name, top, bottom, max_w=11.6):
    top = max(top, 1.95)
    im = Image.open(FIG / name)
    ar = im.width / im.height
    h = bottom - top
    w = h * ar
    if w > max_w:
        w, h = max_w, max_w / ar
    return pic(s, name, (W - w) / 2, top, w=w), top + h


def card(s, x, y, w, h):
    y = max(y, 1.95)
    c = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                           Inches(w), Inches(h))
    c.fill.solid()
    c.fill.fore_color.rgb = PAPER
    c.line.color.rgb = INK
    c.line.width = Pt(1.75)
    c.shadow.inherit = False
    c.text_frame.text = ""
    return c


def shape(s, kind, x, y, w, h, fill=INK, line=None):
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(2.5)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def arrow(s, x, y, w=0.5, h=0.3):
    return shape(s, MSO_SHAPE.RIGHT_ARROW, x, y, w, h, fill=INK)


def ids(*shapes):
    out = []
    for sh in shapes:
        if isinstance(sh, tuple):
            out.extend(x.shape_id for x in sh)
        else:
            out.append(sh.shape_id)
    return out


def anim(s, *click_groups):
    ANIM[prs.slides.index(s)] = [list(g) for g in click_groups]


# ── 1 · title ───────────────────────────────────────────────────────────────
s = slide()
txt(s, "THE SECOND REFERENCE", 0.7, 1.5, 11.9, 0.95, size=44, bold=True,
    font=HEAD, color=INK)
txt(s, "Rotate the support, or label another specimen",
    0.72, 2.55, 11.9, 0.5, size=18, color=INK)
for i, lab in enumerate(("implicit: rotate\nthe one you have",
                         "explicit: label\na second specimen",
                         "the answer flips\nbetween datasets")):
    x = 0.7 + i * 4.07
    card(s, x, 3.6, 3.77, 1.5)
    txt(s, lab, x, 4.05, 3.77, 0.8, size=14, bold=True, font=HEAD, color=INK,
        align=PP_ALIGN.CENTER)
txt(s, "S16 / base SAM 2 / 4 runs / 20 seeds / Hunter Bridges",
    0.72, 6.5, 11.9, 0.4, size=13, color=INK)

# ── 2 · the task ────────────────────────────────────────────────────────────
s = slide()
kicker(s, "THE TASK")
title(s, "One labelled specimen, then a query")
labels = ["support", "its trait masks", "query", "prediction", "ground truth"]
files = ["g1_support.png", "g1_support_mask.png", "g1_query.png",
         "g1_prediction.png", "g1_ground_truth.png"]
xs = [0.7, 2.45, 4.47, 7.24, 10.01]
panels = []
for f, x, lab in zip(files, xs, labels):
    p = pic(s, f, x, 2.06, h=2.4)
    w = p.width / 914400
    t = txt(s, lab, x, 1.74, w, 0.3, size=11.5, bold=True, color=INK,
            align=PP_ALIGN.CENTER)
    panels.append((p, t))
a1 = arrow(s, 4.13, 2.9, 0.32, 0.3)
a2 = arrow(s, 6.9, 2.9, 0.32, 0.3)
f2 = txt(s, "One labelled mask is the whole supervision budget. Everything in "
            "this talk is about spending a second one, or not.",
         0.7, 5.1, 11.9, 0.6, size=15, color=INK)
anim(s, ids(*panels[:2]), ids(a1, *panels[2:4]), ids(a2, panels[4]), ids(f2))

# ── 3 · the two second references ───────────────────────────────────────────
s = slide()
kicker(s, "THE QUESTION")
title(s, "Two ways to spend a second reference")

lft = txt(s, "IMPLICIT  (orientation TTA)", 0.7, 1.95, 5.6, 0.32, size=14,
          bold=True, font=HEAD, color=INK)
p1 = pic(s, "g1_support.png", 0.7, 2.4, h=1.9)
p2 = pic(s, "g1_support_rot.png", 2.35, 2.4, h=1.9)
l1 = txt(s, "the same specimen, turned 180 degrees, kept outside the memory "
            "bank and picked per episode by a confidence rule",
         0.7, 4.45, 5.6, 0.9, size=13, color=INK)
c1 = txt(s, "cost: 0 labels, 2 forward passes", 0.7, 5.45, 5.6, 0.32, size=14,
         bold=True, color=INK)

rgt = txt(s, "EXPLICIT  (a real 2-shot)", 7.0, 1.95, 5.6, 0.32, size=14,
          bold=True, font=HEAD, color=INK)
p3 = pic(s, "g1_support.png", 7.0, 2.4, h=1.9)
p4 = pic(s, "g1_query.png", 8.65, 2.4, h=1.9)
l2 = txt(s, "a second individual from the same pool, inside the bank as one "
            "3-frame video, selected by memory attention",
         7.0, 4.45, 5.6, 0.9, size=13, color=INK)
c2 = txt(s, "cost: 1 label, 1 forward pass", 7.0, 5.45, 5.6, 0.32, size=14,
         bold=True, color=INK)

div = shape(s, MSO_SHAPE.RECTANGLE, 6.62, 1.95, 0.02, 3.9, fill=INK)
foot = txt(s, "Panels are the project's own S3 render. Since S5c every "
              "headline arm has used the implicit version. The explicit one had "
              "never been measured at any budget, and it is the cheaper of the "
              "two at inference.",
           0.7, 6.1, 11.9, 0.8, size=14, color=INK)
anim(s, ids(lft, p1, p2, l1, c1), ids(div, rgt, p3, p4, l2, c2), ids(foot))

# ── 4 · the hole in the shot axis ───────────────────────────────────────────
s = slide()
kicker(s, "WHAT WAS ALREADY MEASURED")
title(s, "The shot axis had a hole at two")
f, bot = figure(s, "g2_ladder_1.png", 1.98, 5.5)
foot = txt(s, "The project has 1-shot and 5-shot on both datasets, never 2. "
              "The dashed line is an interpolation nobody measured.",
           0.7, 5.8, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(foot))

# ── 5 · the metric ──────────────────────────────────────────────────────────
s = slide()
kicker(s, "THE GRADING RULE")
title(s, "What one mIoU number means here")
f, bot = figure(s, "g10_metric.png", 1.98, 5.35)
f1 = txt(s, "Per image: resize logits to the query, argmax over trait "
            "channels, IoU per ground-truth trait, then the mean over traits.",
         0.7, 5.65, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "A trait the model never claims scores 0 and still counts, so a "
            "missed antenna costs as much as a botched elytra.",
         0.7, 6.35, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 6 · the design ──────────────────────────────────────────────────────────
s = slide()
kicker(s, "DESIGN")
title(s, "Four runs, two of them anchors")
f, bot = figure(s, "g11_design.png", 1.98, 5.35)
f1 = txt(s, "The two TTA cells resume the banked S11 cells, and the gate "
            "pins them to 75.18 and 63.86 within 0.5.",
         0.7, 5.6, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "That certifies the wiring of the merge, not TTA itself. The "
            "two 2-shot cells are the only new measurement here.",
         0.7, 6.3, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 7 · the pairing licence ─────────────────────────────────────────────────
s = slide()
kicker(s, "HOW IT IS IMPLEMENTED")
title(s, "Why this comparison is allowed")
f, bot = figure(s, "g8_nesting.png", 1.98, 5.15)
f1 = txt(s, "The row crosses protocol.shots and protocol.support_set_hash, "
            "both comparability fields, so registry.compare refuses it.",
         0.7, 5.35, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "The licence is a property of the draws: both cells slice a "
            "prefix of one seeded permutation, so the 1-shot support is the "
            "first element of the 2-shot pair and the shared support's "
            "difficulty still cancels.",
         0.7, 6.0, 11.9, 1.0, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 8 · the pre-registration ────────────────────────────────────────────────
s = slide()
kicker(s, "WRITTEN BEFORE THE RUN")
title(s, "Two predictions, opposite signs")
f, bot = figure(s, "g3_mirror_1.png", 1.98, 5.0)
f1 = txt(s, "Butterfly, band -3.9 to -4.6: Major's defect is query "
            "orientation, and 37% of cyrbia's test specimens are pinned upside "
            "down, so no upright second specimen reaches it.",
         0.7, 5.2, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "Beetle, band +2 to +5.3: the 5-shot mechanism is species "
            "diversity, and a rotated copy of one specimen adds no species.",
         0.7, 6.15, 11.9, 0.7, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 9 · the result ──────────────────────────────────────────────────────────
s = slide()
kicker(s, "RESULT")
title(s, "The mirror image lands, both signs")
f, bot = figure(s, "g3_mirror_2.png", 1.98, 5.15)
f1 = txt(s, "butterfly 75.18 vs 71.48, -3.71 (-4.9%), significant, 0.19 short "
            "of the band edge",
         0.7, 5.4, 11.9, 0.5, size=16, bold=True, color=INK,
         align=PP_ALIGN.CENTER)
f2 = txt(s, "beetle 63.86 vs 69.04, +5.17 (+8.1%), significant, at the top of "
            "its band",
         0.7, 5.95, 11.9, 0.5, size=16, bold=True, color=INK,
         align=PP_ALIGN.CENTER)
f3 = txt(s, "Gate PASS 14 of 14. The anchors resume S11's cells; the "
            "2-shot cells are new.",
         0.7, 6.55, 11.9, 0.5, size=14, color=INK, align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(f1), ids(f2), ids(f3))

# ── 10 · both readings ──────────────────────────────────────────────────────
s = slide()
kicker(s, "ROBUSTNESS")
title(s, "The sign survives the other reading")
f, bot = figure(s, "g9_readings.png", 1.98, 5.35)
foot = txt(s, "Under the paper's full reading, independent overlap plus "
              "boundary best-of-3, the headlines are -3.49 and +5.30, both "
              "still significant. That reading changes two things at once, and "
              "neither sign moves.",
           0.7, 5.65, 11.9, 0.8, size=15, color=INK)
anim(s, ids(f), ids(foot))

# ── 11 · butterfly saturates ────────────────────────────────────────────────
s = slide()
kicker(s, "BUTTERFLY MAJOR")
title(s, "Major's shot curve is flat by two")
f, bot = figure(s, "g2_ladder_2.png", 1.98, 5.4)
f1 = txt(s, "2-shot came in at 71.48 against a predicted 70.6 to 71.3, so the "
            "band miss is on the 2-shot side.",
         0.7, 5.6, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "+1.18 of the measured +1.48 five-shot lift, 80% of it, arrives "
            "with the first added specimen. Unpaired against the banked level.",
         0.7, 6.25, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 12 · butterfly variance ─────────────────────────────────────────────────
s = slide()
kicker(s, "BUTTERFLY MAJOR")
title(s, "The second specimen bought spread")
f, bot = figure(s, "g4_variance.png", 1.98, 5.4)
f1 = txt(s, "std 2.10 to 0.30, a drop of 86%, the tightest cell measured at "
            "1 or 2 shots. The level stayed 3.71 below TTA.",
         0.7, 5.65, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "That is what S5b and S6 said support-side levers do on Major. The "
            "residual is query-side, and support-side levers do not reach it.",
         0.7, 6.3, 11.9, 0.7, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 13 · beetle concavity ───────────────────────────────────────────────────
s = slide()
kicker(s, "BEETLE")
title(s, "Beetle earns the lift on shot two")
f, bot = figure(s, "g12_concave.png", 1.98, 5.1)
f1 = txt(s, "+5.84 (+9.2%) over the banked 63.20, unpaired, which is 58.6% "
            "of the whole 1 to 5 shot lift from one added specimen.",
         0.7, 5.35, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "Concave as predicted, but past the top of the pre-registered "
            "35 to 55% share band. Beetle overshot where butterfly undershot.",
         0.7, 6.0, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 14 · the ceiling moved ──────────────────────────────────────────────────
s = slide()
kicker(s, "BEETLE")
title(s, "It changed what gets proposed")
f, bot = figure(s, "g5_oracle.png", 1.98, 5.3)
f1 = txt(s, "beetle oracle ceiling 66.93 to 71.73, +4.80 (+7.2%). The second "
            "specimen changed the proposals, not just the assignment.",
         0.7, 5.55, 11.9, 0.7, size=15, color=INK)
f2 = txt(s, "Butterfly's ceiling is flat within a third of a point, 75.65 to "
            "75.35, with no paired test on the oracle reading. S14sam3's arch "
            "arm moved this axis by training; here it moved by data.",
         0.7, 6.25, 11.9, 0.7, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 15 · the mechanism ──────────────────────────────────────────────────────
s = slide()
kicker(s, "MECHANISM")
title(s, "TTA is an orientation shot only")
f, bot = figure(s, "g6_orientation.png", 1.98, 5.2)
f1 = txt(s, "On beetle the rotated support is kept in 29.7% of episodes and "
            "buys +0.66 (+1.0%). The explicit specimen is worth about 9 times "
            "that, because a rotated copy adds no species.",
         0.7, 5.4, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "On Major it fires in 8.1% of episodes and is worth +4.88 (+6.9%), "
            "because those are the rotated queries nothing else reaches.",
         0.7, 6.3, 11.9, 0.7, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 16 · the label budget ───────────────────────────────────────────────────
s = slide()
kicker(s, "WHAT IT COSTS")
title(s, "What the second label actually buys")
f, bot = figure(s, "g7_labels.png", 1.98, 5.35)
f1 = txt(s, "One extra labelled mask on stock SAM 2 puts beetle at 69.04, "
            "within noise of the project's champion at one label, 69.88, which "
            "is a model swap plus a trained adapter plus TTA.",
         0.7, 5.55, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "On SST's own label axis the cheapest training-free point above "
            "the paper's 1-shot OC-CCL row of 65.2 is now 2 labels.",
         0.7, 6.45, 11.9, 0.7, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 17 · what it suggests ───────────────────────────────────────────────────
s = slide()
kicker(s, "WHAT IT SUGGESTS")
title(s, "Three things this changes")
lines = [
    ("The two datasets are mirror images, each by the mechanism its own "
     "residual named. Neither result generalises to the other."),
    ("Ask what a technique adds, not how many references it looks like. TTA "
     "adds an orientation, not a specimen."),
    ("A second label is the strongest per-label lever measured on beetle, "
     "and it needs no training at all. 5-shot is still larger."),
]
groups = []
for i, line in enumerate(lines):
    y = 2.05 + i * 1.45
    c = card(s, 0.7, y, 11.9, 1.2)
    n = txt(s, f"{i + 1:02d}", 1.0, y + 0.15, 0.6, 0.35, size=13, bold=True,
            font=HEAD, color=INK)
    t = txt(s, line, 1.75, y + 0.2, 10.6, 0.85, size=15, color=INK)
    groups.append(ids(c, n, t))
foot = txt(s, "S16, gate PASS 14 of 14, landed 2026-09-01. Every number here "
              "is in results/s16.zip.",
           0.7, 6.65, 11.9, 0.5, size=13, color=INK)
anim(s, *groups, ids(foot))

# ── 18 · backup: the cells ──────────────────────────────────────────────────
s = slide()
kicker(s, "BACKUP")
title(s, "Every cell, as recorded")
rows = [
    ("", "1-shot + TTA", "2-shot plain", "headline"),
    ("butterfly plain", "75.18 +/- 2.10", "71.48 +/- 0.30", "-3.71 (-4.9%)"),
    ("butterfly best-of-3", "76.72", "73.23", "-3.49"),
    ("butterfly oracle", "75.65", "75.35", "-0.30"),
    ("beetle plain", "63.86 +/- 4.32", "69.04 +/- 4.28", "+5.17 (+8.1%)"),
    ("beetle best-of-3", "65.45", "70.75", "+5.30"),
    ("beetle oracle", "66.93", "71.73", "+4.80"),
]
xs = [0.75, 4.3, 7.3, 10.3]
for i, row in enumerate(rows):
    y = 2.0 + i * 0.62
    for x, cellv in zip(xs, row):
        txt(s, cellv, x, y, 3.4, 0.4, size=14, bold=(i == 0), color=INK)
    if i == 0:
        shape(s, MSO_SHAPE.RECTANGLE, 0.75, y + 0.42, 11.8, 0.02, fill=INK)
txt(s, "20 seeds per cell. CIs on the plain rows: butterfly [-4.48, -2.69], "
       "2 of 20 seeds improved; beetle [+2.63, +7.77], 15 of 20. The oracle "
       "rows are differences, with no paired test run on that reading.",
    0.75, 6.55, 11.9, 0.5, size=13, color=INK)

# ── 19 · backup: the gate ───────────────────────────────────────────────────
s = slide()
kicker(s, "BACKUP")
title(s, "What the gate checks, per dataset")
checks = [
    "both cells on the dataset's canonical protocol, 20 of 20 seeds each",
    "the 1-shot TTA anchor reproduces the S11 ledger within 0.5",
    "the headline carries its per-run prefix-nesting proof",
    "the headline declares the model.tta toggle it crosses",
    "2-shot clears the banked 1-shot level, so no conditioning frame is dropped",
    "the paper's best-of-3 ran alongside the plain metric",
    "the headline itself is recorded with its CI and seed count",
]
for i, c in enumerate(checks):
    y = 2.0 + i * 0.6
    txt(s, f"{i + 1:02d}", 0.75, y, 0.6, 0.4, size=14, bold=True, font=HEAD,
        color=INK)
    txt(s, c, 1.5, y, 11.0, 0.4, size=14, color=INK)
txt(s, "Seven per dataset, 14 of 14 PASS. Naming a protocol field in "
       "ignore_fields is still refused; the relaxation must arrive with its "
       "evidence.",
    0.75, 6.4, 11.9, 0.7, size=13, color=INK)

# ── presenter notes: directions, not a script ───────────────────────────────
DIRECTIONS = [
    "Title. 20 seconds. Name the question in one sentence: a second reference "
    "can be the same specimen rotated, or a different specimen labelled, and "
    "this stage measures which one pays.",
    "45s, 4 clicks. Only enough setup to make the budget clear. One labelled "
    "mask in, one query out. Do not re-teach the task if they saw the main "
    "talk; go straight to the last line.",
    "70s, 3 clicks. The core distinction, so spend the time. Left: outside the "
    "bank, chosen per episode by confidence. Right: inside the bank, chosen by "
    "memory attention. Then the two costs, said out loud.",
    "35s, 2 clicks. Point at the gap. Say plainly that shot 2 was never run "
    "before this stage.",
    "40s, 3 clicks. Only if the audience has not seen the metric slide. Say "
    "the argmax rule and the zero-for-unclaimed rule. First candidate for "
    "cutting.",
    "40s, 3 clicks. Four runs. Emphasise that half of them exist to catch a "
    "protocol slip rather than to produce a number.",
    "70s, 3 clicks. This is the implementation question people ask. Say why "
    "compare refuses the row, then why the draws license it, then that the "
    "proof is embedded in the row rather than asserted.",
    "60s, 3 clicks. Read the bands off the figure before showing anything "
    "measured. Say both mechanisms in one sentence each. Do not skip this: it "
    "is what makes the next slide a test rather than a report.",
    "70s, 4 clicks. Let the figure sit before speaking. Both predicted signs, "
    "both significant. Mention the 0.19 miss honestly and say it is on the "
    "2-shot side, which the next slides explain.",
    "30s, 2 clicks. Short. Two metrics, same signs, same conclusion.",
    "50s, 3 clicks. Explain the miss: the curve saturates. 80% of the 5-shot "
    "lift by shot 2 is the number to say.",
    "50s, 3 clicks. The second specimen did something real, just not to the "
    "level. Tie it back to what S5b and S6 predicted about support-side "
    "levers on Major.",
    "45s, 3 clicks. Concave as pre-registered. 59% of the lift on the first "
    "added specimen.",
    "60s, 3 clicks. Give this room. The ceiling moved on beetle and did not on "
    "butterfly, which separates proposals from assignment. Connect to S14's "
    "arch arm if they were at that talk.",
    "60s, 3 clicks. The mechanism sentence of the whole stage. Fires often and "
    "pays little on beetle, fires rarely and pays a lot on Major.",
    "55s, 3 clicks. The practical result. Two labels, no training, no TTA, is "
    "within noise of the trained champion at one label.",
    "60s, 4 clicks. One line per click. Land the mirror image, then the "
    "framing point about what a technique adds, then the cost. Leave up for "
    "questions.",
    "BACKUP, not presented. Jump here for any per-cell number, spread, or CI.",
    "BACKUP, not presented. Jump here when asked how the run was guarded or "
    "what would have made it fail.",
]
assert len(DIRECTIONS) == len(prs.slides._sldIdLst), (
    f"{len(DIRECTIONS)} notes for {len(prs.slides._sldIdLst)} slides")
for sld, note in zip(prs.slides, DIRECTIONS):
    sld.notes_slide.notes_text_frame.text = note

prs.save(str(OUT))
Path("anim_spec_s16.json").write_text(json.dumps(ANIM, indent=1))
print(f"wrote {OUT} · {len(prs.slides._sldIdLst)} slides")
print("clicks per slide:", {k + 1: len(v) for k, v in ANIM.items()})
