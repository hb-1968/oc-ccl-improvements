#!/usr/bin/env python3
"""Build the 20-minute deck: the techniques, and which of them work.

Figures come from make_talk_figures.py (project data only). Click-builds are
injected afterwards by add_animations.py, driven by anim_spec.json.
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

FIG = Path("figures")
OUT = Path("Hunter_SST_techniques.pptx")

INK = RGBColor(0x00, 0x00, 0x00)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
#: The reference deck is strictly black on white. These aliases keep the slide
#: code readable without reintroducing colour.
COPPER = INK
TEAL = INK
CRIMSON = INK
SLATE = INK
MUTED = INK
WASH = PAPER
PALE = INK

HEAD = "Courier New"
BODY = "Courier New"
MONO = "Courier New"

W, H = 13.333, 7.5
ANIM: dict[int, list[list[int]]] = {}

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


# ── primitives ───────────────────────────────────────────────────────────────
def slide(dark=False):
    """One slide. `dark` is kept as a no-op so callers need not change."""
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


def title(s, text, dark=False, size=30, y=0.86, rule=True):
    """Eyebrow, title, and the full-width rule the reference deck puts under it."""
    t = txt(s, text, 0.7, y, W - 1.4, 0.7, size=size, bold=True, font=HEAD,
            color=INK)
    if rule:
        r = shape(s, MSO_SHAPE.RECTANGLE, 0.72, 1.62, 12.0, 0.03, fill=INK)
        return t, r
    return t


def kicker(s, text, y=0.5, color=None):
    tb = txt(s, text.upper(), 0.7, y, W - 1.4, 0.32, size=12, bold=True,
             font=HEAD, color=INK)
    for p in tb.text_frame.paragraphs:            # tracked out, as the reference is
        for r in p.runs:
            r.font._rPr.set("spc", "201")
    return tb


def pic(s, name, x, y, w=None, h=None):
    y = max(y, 1.95)                           # never under the title rule
    im = Image.open(FIG / name)
    ar = im.width / im.height
    if w is None:
        w = h * ar
    if h is None:
        h = w / ar
    return s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y),
                                Inches(w), Inches(h))


def figure(s, name, top, bottom, max_w=11.6):
    """Scale a figure to fit the band [top, bottom], centred. Returns (pic, y)."""
    top = max(top, 1.95)                       # never under the title rule
    im = Image.open(FIG / name)
    ar = im.width / im.height
    h = bottom - top
    w = h * ar
    if w > max_w:
        w, h = max_w, max_w / ar
    return pic(s, name, (W - w) / 2, top, w=w), top + h


def card(s, x, y, w, h, fill=PAPER, line=None):
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


def arrow(s, x, y, w=0.5, h=0.3, fill=INK):
    return shape(s, MSO_SHAPE.RIGHT_ARROW, x, y, w, h, fill=fill)


def step(s, n, x, y, color=None):
    """A bold index. The reference deck uses type, not ornament."""
    t = txt(s, f"{n:02d}", x, y + 0.02, 0.5, 0.35, size=11.5, bold=True,
            font=HEAD, color=INK)
    return (t,)


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


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ── 1 · title ────────────────────────────────────────────────────────────────
s = slide()
txt(s, "SEVEN TECHNIQUES", 0.7, 1.5, 11.9, 0.95, size=46, bold=True, font=HEAD,
    color=INK)
txt(s, "Measured against one baseline / SAM 2 and SAM 3",
    0.72, 2.55, 11.9, 0.5, size=18, color=INK)
for i, lab in enumerate(("Support-side curation", "Orientation TTA",
                         "Loss and model swaps")):
    x = 0.7 + i * 4.07
    card(s, x, 3.6, 3.77, 1.35)
    txt(s, lab, x, 4.15, 3.77, 0.4, size=14, bold=True, font=HEAD, color=INK,
        align=PP_ALIGN.CENTER)
txt(s, "A findings talk / about 20 minutes / Hunter Bridges",
    0.72, 6.5, 11.9, 0.4, size=13, color=INK)

# ── 2 · the task ─────────────────────────────────────────────────────────────
s = slide()
kicker(s, "THE TASK")
title(s, "The task")
labels = ["support", "its trait masks", "query", "prediction", "ground truth"]
files = ["f11_p1_support.png", "f11_p2_support_mask.png", "f11_p3_query.png",
         "f11_p4_prediction.png", "f11_p5_ground_truth.png"]
xs = [0.7, 2.45, 4.47, 7.24, 10.01]
panels = []
for f, x, lab in zip(files, xs, labels):
    p = pic(s, f, x, 2.06, h=2.4)
    w = p.width / 914400
    t = txt(s, lab, x, 1.74, w, 0.3, size=11.5, bold=True, color=INK,
            align=PP_ALIGN.CENTER)
    panels.append((p, t))
ar1 = arrow(s, 4.13, 2.9, 0.32, 0.3)
cap = txt(s, "one 2-frame video  ·  masks propagate from frame 0 to frame 1", 0.85, 5.45, 11.6,
          0.5, size=14, color=MUTED, align=PP_ALIGN.CENTER)
bad = txt(s, "the query is pinned 180°, and the prediction follows the support's layout",
          0.85, 6.15, 11.6, 0.6, size=17, bold=True, color=CRIMSON,
          font=HEAD, align=PP_ALIGN.CENTER)
anim(s, ids(panels[0], panels[1]), ids(ar1, panels[2]), ids(panels[3]),
     ids(panels[4]), ids(cap), ids(bad))
notes(s, "≈60s. A real render from the S3 bundle, not an illustration. Walk "
         "left to right, then land on the failure: the model carries the "
         "support's spatial layout across, so an upside-down query gets every "
         "trait on its mirror-image partner.")

# ── 3 · the metric ──────────────────────────────────────────────────────────
s = slide()
kicker(s, "THE UNIT EVERY NUMBER IS IN")
title(s, "How a score is computed")
steps = [
    "resize logits to GT resolution",
    "each pixel takes its best-scoring trait",
    "best logit <= 0 means background",
    "IoU per trait present in this GT",
    "image score = mean over those traits",
]
groups = []
for i, text in enumerate(steps):
    y = 1.95 + i * 0.5
    n = txt(s, f"{i + 1}", 0.85, y, 0.3, 0.35, size=12, bold=True, color=COPPER)
    t = txt(s, text, 1.25, y, 5.5, 0.4, size=12, font=MONO, color=INK)
    groups.append(ids(n, t))
sem = card(s, 0.85, 4.25, 5.95, 2.3, fill=WASH)
sh1 = txt(s, "what a number means", 1.2, 4.45, 5.3, 0.4, size=14, bold=True,
          font=HEAD, color=TEAL)
sh2 = txt(s, "1.00 is every pixel of every trait correct.\n"
             "Traits are averaged, not pixels, so a hair-thin antenna counts "
             "as much as the elytra.\n"
             "A ground-truth trait the model never claims scores 0.",
          1.0, 5.15, 5.3, 1.3, size=11.5, color=INK, line_spacing=1.3)
f = pic(s, "f23_metric.png", 6.95, 2.35, w=5.85)
cap = txt(s, "one real query: beetle_1166_231, run 0",
          6.95, 1.98, 5.85, 0.3, size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
foot = txt(s, "then the mean over the images in a run, and the mean over 20 runs",
           0.85, 6.6, 11.9, 0.5, size=14, color=INK, align=PP_ALIGN.CENTER)
anim(s, *groups, ids(cap, f), ids(sem, sh1, sh2), ids(foot))

# ── 3 · the premise: the starting score, and how each technique is judged ────
s = slide()
kicker(s, "THE PREMISE")
title(s, "The baseline")
c1 = card(s, 0.85, 1.85, 5.6, 2.75)
b1 = txt(s, "55.96", 1.2, 2.05, 4.9, 1.1, size=62, bold=True, font=HEAD,
         color=SLATE)
b1b = txt(s, "± 7.90", 4.35, 2.45, 1.9, 0.5, size=26, bold=True, color=COPPER)
l1 = txt(s, "Beetle, 1-shot, uniform supports", 1.2, 3.3, 5.1, 0.5,
         size=13, color=INK)
l1b = txt(s, "target: 61.9 ± 3.7", 1.2, 3.85, 5.1, 0.4, size=12.5, bold=True,
          color=CRIMSON)
c2 = card(s, 6.9, 1.85, 5.6, 2.75)
b2 = txt(s, "68.54", 7.25, 2.05, 4.9, 1.1, size=62, bold=True, font=HEAD,
         color=SLATE)
b2b = txt(s, "± 3.15", 10.4, 2.45, 1.9, 0.5, size=26, bold=True, color=COPPER)
l2 = txt(s, "Butterfly Major, 1-shot, uniform supports", 7.25, 3.3,
         5.1, 0.5, size=13, color=INK)
l2b = txt(s, "target: 81.0 ± 1.0", 7.25, 3.85, 5.1, 0.4, size=12.5, bold=True,
          color=CRIMSON)
rule = txt(s, "how each technique is judged", 0.85, 5.05, 11.9, 0.45,
           size=14, bold=True, color=INK, align=PP_ALIGN.CENTER)
r1 = txt(s, "20 draws / same draws both arms / paired per seed / CI / "
            "band frozen first",
         0.85, 5.6, 11.9, 0.5, size=15, color=TEAL, bold=True,
         align=PP_ALIGN.CENTER)
r2 = txt(s, "run-to-run spread is reported as a result, not an error bar", 0.85, 6.35, 11.9, 0.5,
         size=14, italic=True, color=COPPER, align=PP_ALIGN.CENTER)
anim(s, ids(c1, b1, b1b, l1, l1b), ids(c2, b2, b2b, l2, l2b),
     ids(rule, r1), ids(r2))
notes(s, "≈60s. This is the baseline everything is measured against: no "
         "curation, no orientation handling, no fine-tune. Note both numbers "
         "are short of target and — more diagnostic — the beetle spread is "
         "2.1× the target's. A matched mean with a doubled variance is not a "
         "reproduction.")

# ── 4 · technique 1: support curation ────────────────────────────────────────
s = slide()
kicker(s, "TECHNIQUE 1  ·  SUPPORT-SIDE")
title(s, "Curating the reference")
steps = [
    ("Measure the body axis", "directed vector between two trait centroids"),
    ("Take the threshold from the data", "the train pool's own circular std, "
     "55.6°, not tuned"),
    ("Estimate on train only", "no test orientation may touch the pool"),
    ("Require every trait present", "the second half of the criterion"),
]
groups = []
for i, (h1, b) in enumerate(steps):
    y = 2.0 + i * 1.15
    (n,) = step(s, i + 1, 0.85, y)
    t1 = txt(s, h1, 1.55, y + 0.03, 4.6, 0.8, size=15, bold=True, font=HEAD,
             color=INK, line_spacing=1.1)
    t2 = txt(s, b, 6.4, y + 0.06, 6.1, 1.0, size=12.5, color=MUTED,
             line_spacing=1.15)
    groups.append(ids(n, t1, t2))
foot = txt(s, "120 candidate references   →   a pool of 79",
           0.85, 6.5, 11.9, 0.6, size=16, bold=True, font=HEAD, color=TEAL,
           align=PP_ALIGN.CENTER)
anim(s, *groups, ids(foot))
notes(s, "≈75s. The paper samples its reference 'with canonical shapes and "
         "visually clear traits' and never defines the criterion, so this is "
         "an approximation of a human judgement — which is why all arms are "
         "always reported, not just this one. The threshold discipline is the "
         "part to stress: it is a property of how specimens are pinned, not a "
         "value chosen because it produced a good number.")

# ── 5 · technique 2: balancing the draw ──────────────────────────────────────
s = slide()
kicker(s, "TECHNIQUE 2  ·  SUPPORT-SIDE")
title(s, "Fixing the draw")
prob = txt(s, "the filter keeps 10/10 of one species and 4/10 of three others",
           0.85, 1.8, 11.9, 0.5, size=15, color=INK, align=PP_ALIGN.CENTER)
f, fb = figure(s, "f17_coverage.png", 2.72, 5.4, max_w=11.0)
mech = txt(s, "run r starts at species r mod 12", 0.85, 5.88, 11.9, 0.5,
           size=16, bold=True, font=HEAD, color=TEAL, align=PP_ALIGN.CENTER)
anim(s, ids(prob), ids(f), ids(mech))
notes(s, "≈55s. The instructive part: stratified *random* sampling does not "
         "work here. Twenty independent draws over twelve species is coupon "
         "collector — measured, it still left three species unused. Equal "
         "probability is not equal representation at n=20.")

# ── 6 · result: beetle ──────────────────────────────────────────────────────
s = slide()
kicker(s, "WHAT THOSE TWO BOUGHT  ·  BEETLE")
title(s, "What those two bought")
b1 = pic(s, "f3_beetle_ladder_3.png", 2.5, 1.75, w=8.3)
b2 = pic(s, "f3_beetle_ladder_5.png", 2.5, 1.75, w=8.3)
foot = txt(s, "+7.24 mIoU (+12.9%)      head 43.6 → 55.9      "
              "spread 7.90 → 3.89",
           0.85, 6.45, 11.9, 0.5, size=15, bold=True, font=HEAD, color=TEAL,
           align=PP_ALIGN.CENTER)
anim(s, ids(b2), ids(foot))
notes(s, "≈65s. Two clicks: first the three 1-shot bars, then 5-shot and TTA "
         "for context. The per-trait breakdown is what makes this a mechanism "
         "rather than a moved average — each fix moved exactly the trait its "
         "diagnostic predicted.")

# ── 7 · technique 3: placement curation ─────────────────────────────────────
s = slide()
kicker(s, "TECHNIQUE 3  ·  SUPPORT-SIDE")
title(s, "Placement curation")
c1 = card(s, 0.7, 1.98, 5.85, 2.8)
t1 = txt(s, "Butterflies are not mis-oriented", 1.2, 1.98, 5.1, 0.5, size=15,
         bold=True, font=HEAD, color=TEAL)
t1b = txt(s, "Pinned wings are already canonical.\nThey lie loosely on a "
             "sheet, so placement varies rather than angle.", 1.2, 2.6, 5.1,
          1.8, size=13, color=INK, line_spacing=1.3)
c2 = card(s, 6.75, 1.98, 5.85, 2.8)
t2 = txt(s, "Change the statistic, keep the rule", 7.35, 1.95, 4.85, 0.5,
         size=14, bold=True, font=HEAD, color=TEAL)
t2b = txt(s, "Centroid displacement from the train median.\nThreshold = "
             "mean + 1 std, the same discipline.", 7.35, 2.6, 4.85, 1.8,
          size=13, color=INK, line_spacing=1.3)
res = txt(s, "68.54 → 70.30      +1.90 (+2.8%)",
          0.85, 4.95, 11.9, 0.7,
          size=34, bold=True, font=HEAD, color=INK, align=PP_ALIGN.CENTER)
why = txt(s, "the two criteria fired on disjoint subspecies, each where its "
              "diagnostic predicted", 0.85, 5.8, 11.9, 0.6, size=14,
          italic=True, color=COPPER, align=PP_ALIGN.CENTER)
anim(s, ids(c1, t1, t1b), ids(c2, t2, t2b), ids(res), ids(why))
notes(s, "≈65s. Placement explains cyrbia (+3.56, 160 px displacement, zero "
         "incomplete supports); completeness explains notabilis and (m×p)×m "
         "(+3.18, +2.78, ~30 px displacement but 1 and 2 incomplete supports). "
         "The two subspecies with neither defect did not move.")

# ── 8 · technique 4: more shots ─────────────────────────────────────────────
s = slide()
kicker(s, "TECHNIQUE 4  ·  SUPPORT-SIDE")
title(s, "More references")
f, fb = figure(s, "f15_shots.png", 1.78, 5.2, max_w=9.4)
foot1 = txt(s, "beetle +9.96 (+15.8%)        butterfly +1.48 (+2.1%)",
            0.85, 5.5, 11.9, 0.5, size=15, bold=True, color=INK,
            align=PP_ALIGN.CENTER)
foot2 = txt(s, "butterfly gains 1.48; its defect is on the query side", 0.85, 6.25, 11.9,
            0.6, size=17, bold=True, font=HEAD, color=CRIMSON,
            align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot1), ids(foot2))
notes(s, "≈55s. Also note the spread collapse on butterfly, 1.36 → 0.15, "
         "nine-fold and below the target's. Five supports make the answer "
         "stable without making it right — the signature of a defect that is "
         "invariant to support choice.")

# ── 9 · the query-side defect ───────────────────────────────────────────────
s = slide()
kicker(s, "WHY SUPPORT-SIDE STOPS")
title(s, "The query-side defect")
f = pic(s, "f4_cyrbia_split.png", 0.75, 1.8, w=8.1)
note = txt(s, "0.00 is what the copy baseline\nscores on those specimens,\n"
              "with no model involved", 9.15, 2.6, 3.6, 1.6, size=13,
           color=INK, bold=True, line_spacing=1.35)
foot = txt(s, "curation +3.56   ·   five shots +2.85   ·   the level did not "
              "move", 0.75, 6.5, 11.9, 0.6, size=14, italic=True, color=INK,
           align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(note), ids(foot))
notes(s, "≈60s. cyrbia is two populations: one scoring at target level, one "
         "scoring zero. 0.63 × 74.95 + 0.37 × 0.48 = 47.4, its measured score "
         "to two decimals.")

# ── 10 · and it generalises ─────────────────────────────────────────────────
s = slide()
kicker(s, "THE GENERAL FORM OF IT")
title(s, "Query arrangement decides the score")
f = pic(s, "f5_displacement.png", 1.85, 1.75, w=9.6)
foot = txt(s, "the top quartile already beats the target; the deficit is in "
              "the tail", 0.75, 6.6, 11.9, 0.5, size=14, bold=True,
           color=INK, align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot))
notes(s, "≈50s. Only 46% of that displacement is a global offset, and the "
         "residual after centring is the *stronger* predictor — so cropping or "
         "centring each specimen would remove less than half of the weaker "
         "half. The fix has to act on the comparison, not on the framing.")

# ── 11 · section: TTA ───────────────────────────────────────────────────────
s = slide(dark=True)
kicker(s, "TECHNIQUE 5", y=2.4)
txt(s, "ORIENTATION TTA", 0.7, 2.6, 11.9, 1.0, size=40, bold=True, font=HEAD,
    color=INK)
txt(s, "The largest single gain in the project, and the least machinery.\n"
       "No training and no extra labels.", 0.7, 3.9, 11.9, 1.0, size=14,
    color=INK, line_spacing=1.4)
notes(s, "≈15s. Test-time augmentation: no training, no extra labels, no "
         "change to the tracker. Three slides: what it does, how it decides, "
         "what it bought.")

# ── 12 · TTA mechanism, part 1 ──────────────────────────────────────────────
s = slide()
kicker(s, "HOW IT WORKS  ·  1 OF 2")
title(s, "TTA, how it works: 1 of 2")
up_i = pic(s, "f11_p1_support.png", 1.0, 2.15, h=1.95)
up_m = pic(s, "f11_p2_support_mask.png", 2.35, 2.15, h=1.95)
up_l = txt(s, "pass 1, as given", 1.0, 1.8, 2.7, 0.3, size=12, bold=True,
           color=MUTED, align=PP_ALIGN.CENTER)
ro_i = pic(s, "f14_rot_support.png", 1.0, 4.45, h=1.95)
ro_m = pic(s, "f14_rot_mask.png", 2.35, 4.45, h=1.95)
ro_l = txt(s, "pass 2, support rotated 180°", 0.75, 4.12, 3.25, 0.3,
           size=11.5, bold=True, color=COPPER, align=PP_ALIGN.CENTER)
q_i = pic(s, "f11_p3_query.png", 5.3, 3.25, h=2.35)
q_l = txt(s, "the query, never touched", 4.75, 2.9, 3.4, 0.3, size=12,
          bold=True, color=MUTED, align=PP_ALIGN.CENTER)
a1 = arrow(s, 4.35, 3.55, 0.42, 0.3)
a2 = arrow(s, 4.35, 5.05, 0.42, 0.3)
pts = [
    ("Image and mask rotate together", "pixel-aligned, or the mask is useless"),
    ("Only the support rotates", "its orientation relative to the query is "
     "what matters"),
    ("The rotated variant is cached apart", "@r180, or pass 2 gets the "
     "wrong embeddings"),
]
groups = []
for i, (h1, b) in enumerate(pts):
    y = 2.0 + i * 1.68
    (n,) = step(s, i + 1, 8.02, y)
    t1 = txt(s, h1, 8.66, y + 0.02, 4.1, 0.7, size=13, bold=True, font=HEAD,
             color=INK, line_spacing=1.1)
    t2 = txt(s, b, 8.66, y + 0.78, 4.1, 0.9, size=11.5, color=INK,
             line_spacing=1.15)
    groups.append(ids(n, t1, t2))
anim(s, ids(up_i, up_m, up_l), ids(a1, q_i, q_l), ids(ro_i, ro_m, ro_l, a2),
     *groups)
notes(s, "≈70s. Left column is the project's own S3 render; the bottom row is "
         "that same support rotated, which is exactly what rotate_episode "
         "does — np.rot90 twice on image and labelmap together. The wrapper "
         "composes the tracker rather than modifying it, so every previously "
         "recorded number stays byte-stable.")

# ── 13 · TTA mechanism, part 2 ──────────────────────────────────────────────
s = slide()
kicker(s, "HOW IT WORKS  ·  2 OF 2")
title(s, "TTA, how it works: 2 of 2")
code = txt(s, "top = logits.max(axis=0)          # best trait score per pixel\n"
              "fg  = top > 0                     # the pixels it claims\n"
              "confidence = top[fg].mean()       # −inf if it claims nothing",
           0.85, 1.68, 8.7, 1.7, size=12, font=MONO, color=INK,
           line_spacing=1.3, space_after=3)
why = txt(s, "mismatched orientation → scattered and weak\n"
             "matching orientation → compact and strong",
          0.85, 3.42, 8.7, 0.9, size=14, color=INK, line_spacing=1.35)
c1 = card(s, 0.85, 4.78, 5.7, 1.25, fill=WASH, line=TEAL)
t1 = txt(s, "No ground truth is consulted", 1.2, 5.02, 5.0, 0.4, size=14,
         bold=True, font=HEAD, color=TEAL)
t1b = txt(s, "the choice comes from logits alone", 1.2, 5.5, 5.0, 0.4,
          size=12.5, color=INK)
c2 = card(s, 6.8, 4.78, 5.7, 1.25, fill=WASH, line=TEAL)
t2 = txt(s, "2× inference, not 10×", 7.15, 5.02, 5.0, 0.4, size=14, bold=True,
         font=HEAD, color=TEAL)
t2b = txt(s, "both orientations cached, so pass 2 is a hit", 7.15, 5.5,
          5.0, 0.4, size=12.5, color=INK)
cost = txt(s, "it chooses between two predictions; it cannot improve either one", 0.85, 6.45, 11.9, 0.6, size=14, italic=True, color=INK,
           align=PP_ALIGN.CENTER)
anim(s, ids(code), ids(why), ids(c1, t1, t1b), ids(c2, t2, t2b), ids(cost))
notes(s, "≈70s. This is the whole decision rule — mean margin over claimed "
         "pixels, and an empty prediction always loses. Worth stressing to a "
         "reviewer: it is label-free, which is why it stays inside the "
         "one-shot protocol rather than smuggling supervision in.")

# ── 14 · TTA results ────────────────────────────────────────────────────────
s = slide()
kicker(s, "WHAT IT BOUGHT")
title(s, "What TTA bought")
f, fb = figure(s, "f12_tta.png", 1.65, 5.9, max_w=11.5)
foot = txt(s, "cyrbia 47.40 → 73.27 (+54.6%).  cost: its spread rises "
              "6.81 → 10.89",
           0.85, 6.3, 11.9, 0.6, size=14, color=INK, align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot))
notes(s, "≈70s. The 94% recovery is the number to say out loud: cyrbia goes "
         "47.40 → 73.27 against the 74.95 its never-flipped specimens score, "
         "which is the ceiling this intervention can reach by construction. "
         "And the silhouette score rises ten points with it, so this is "
         "re-segmentation, not luckier trait assignment. One caveat: the "
         "rotated-chosen counter postdates that bundle, so we cannot yet say "
         "it picked exactly the flipped 37%.")

# ── 15 · TTA vs the support-side rotation fix ───────────────────────────────
s = slide()
kicker(s, "TECHNIQUE 6, AND A WARNING")
title(s, "Two fixes, one defect")
f, fb = figure(s, "f6_rotation_2x2.png", 1.62, 5.6, max_w=8.3)
foot1 = txt(s, "technique 6 is support-side: the five references span both "
               "orientations, and the memory bank selects rather than blends",
            0.75, 5.85, 11.9, 0.6, size=14, color=INK,
            align=PP_ALIGN.CENTER)
foot2 = txt(s, "best arm: TTA on placement-curated references, 77.22 "
               "(+7.6%)",
            0.75, 6.55, 11.9, 0.6, size=16, bold=True, font=HEAD,
            color=CRIMSON, align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot1), ids(foot2))
notes(s, "≈60s. This is the second time in the project that two techniques "
         "assumed independent were not. The lesson is procedural: measure the "
         "square, not the line — a one-dimensional sweep would have reported "
         "both as additive and shipped the worse arm.")

# ── 16 · section: the loss ──────────────────────────────────────────────────
s = slide(dark=True)
kicker(s, "TECHNIQUE 7", y=2.4)
txt(s, "CHANGING THE LOSS", 0.7, 2.6, 11.9, 1.0, size=40, bold=True,
    font=HEAD, color=INK)
txt(s, "OC-CCL, a cycle-consistency fine-tune.\nThe most machinery in the "
       "project.", 0.7, 3.9, 11.9, 1.0, size=14, color=INK, line_spacing=1.4)
notes(s, "≈15s. Everything so far has been inference-time. This is the one "
         "technique that changes the weights.")

# ── 17 · the cycle, mechanically ────────────────────────────────────────────
s = slide()
kicker(s, "HOW IT WORKS")
title(s, "The cycle, mechanically")
p_sup = pic(s, "f11_p1_support.png", 1.15, 2.35, h=1.8)
p_mask = pic(s, "f11_p2_support_mask.png", 2.35, 2.35, h=1.8)
l_sup = txt(s, "1 / labeled reference", 1.0, 2.02, 2.7, 0.3, size=11.5, bold=True,
            color=MUTED, align=PP_ALIGN.CENTER)
a1 = arrow(s, 4.05, 3.1, 0.5, 0.32)
p_q = pic(s, "f11_p3_query.png", 4.85, 2.35, h=1.8)
p_pred = pic(s, "f11_p4_prediction.png", 6.9, 2.35, h=1.8)
l_q = txt(s, "2 / propagate to the unlabeled query", 4.85, 2.02, 4.05, 0.3,
          size=11.5, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
a2 = arrow(s, 9.15, 3.1, 0.5, 0.32)
p_back = pic(s, "f11_p2_support_mask.png", 10.0, 2.35, h=1.8)
l_back = txt(s, "3 / prompt back, reconstruct", 9.7, 2.02, 2.9, 0.3, size=11.5,
             bold=True, color=MUTED, align=PP_ALIGN.CENTER)
loss = txt(s, "loss  =  BCE + Dice   between the reconstruction and the "
               "reference's own labels", 0.85, 4.5, 11.9, 0.5, size=15,
           bold=True, font=HEAD, color=TEAL, align=PP_ALIGN.CENTER)
pts = txt(s, "no new labels required; the reference's own mask is the target",
          0.85, 5.55, 11.9, 0.6, size=15, italic=True, color=COPPER,
          align=PP_ALIGN.CENTER)
anim(s, ids(p_sup, p_mask, l_sup), ids(a1, p_q, l_q), ids(p_pred),
     ids(a2, p_back, l_back), ids(loss), ids(pts))
notes(s, "≈70s. Four legs in one pass: encode the reference, prompt it with "
         "its ground-truth masks, propagate to the query, then prompt the "
         "reference frame back with the query prediction and reconstruct. "
         "Panels are the project's own render standing in for each tensor.")

# ── 18 · the two conditions ─────────────────────────────────────────────────
s = slide()
kicker(s, "TWO THINGS IT NEEDS TO BE ABLE TO LEARN")
title(s, "Two preconditions")
c1 = card(s, 0.7, 1.98, 5.85, 4.0)
h1 = txt(s, "1 · the closing prompt", 1.2, 1.95, 5.0, 0.45, size=15, bold=True,
         font=HEAD, color=INK)
code = txt(s, "(pred > 0).float()", 1.2, 2.5, 5.0, 0.4, size=14, font=MONO,
           color=CRIMSON, bold=True)
t1 = txt(s, "severs the graph, so nothing downstream reaches the forward leg.",
         1.2, 3.0, 5.0, 1.0, size=12.5, color=INK, line_spacing=1.2)
fix = txt(s, "(logits + 10) / 20", 1.2, 4.05, 5.0, 0.4, size=14, font=MONO,
          color=TEAL, bold=True)
t1b = txt(s, "same logits, linear path.\nforward ÷ reverse gradient: "
             "0.00 → 26×, and 351× per-trait.",
          1.2, 4.5, 5.0, 1.1, size=12.5, color=INK, line_spacing=1.25)
c2 = card(s, 6.75, 1.98, 5.85, 4.0)
h2 = txt(s, "2 · what it supervises", 7.25, 1.95, 4.9, 0.45, size=15,
         bold=True, font=HEAD, color=INK)
t2 = txt(s, "One merged channel supervises the silhouette.\n"
            "C trait channels supervise assignment.\n\n"
            "Each pays where its own objective has room.",
         7.25, 2.55, 4.9, 2.2, size=13, color=INK, line_spacing=1.3)
law = txt(s, "butterfly silhouette had room → +0.85 to +1.25        "
             "beetle silhouette at 90.8 → no effect",
          0.85, 6.15, 11.9, 0.6, size=14, bold=True, color=INK,
          align=PP_ALIGN.CENTER)
anim(s, ids(c1, h1, code, t1), ids(fix, t1b), ids(c2, h2, t2), ids(law))
notes(s, "≈70s. Both conditions were implementation work before any result "
         "existed. The second is the more useful idea for the write-up: the "
         "objective you write down decides which dataset can possibly "
         "benefit, and we measured that in both directions.")

# ── 19 · SCNP, the borrowed term ────────────────────────────────────────────
s = slide()
kicker(s, "A TERM BORROWED FROM ANOTHER PAPER")
title(s, "SCNP")
c1 = card(s, 0.85, 1.9, 5.7, 1.72, fill=WASH)
h1 = txt(s, "the transform", 1.2, 2.1, 5.0, 0.4, size=14, bold=True,
         font=HEAD, color=INK)
m1 = txt(s, "same class  →  min-pool the neighbours\n"
            "other class  →  max-pool the neighbours", 1.2, 2.63, 5.1, 0.9,
         size=12.5, color=TEAL, bold=True, line_spacing=1.35)
c2 = card(s, 6.9, 1.9, 5.6, 1.72, fill=WASH)
h2 = txt(s, "why it should have paid here", 7.25, 2.1, 4.9, 0.4, size=14,
         bold=True, font=HEAD, color=INK)
m2 = txt(s, "an isolated false positive is the worst neighbour of up to nine "
            "pixels", 7.25, 2.63, 4.9, 0.9, size=12.5, color=INK,
         line_spacing=1.25)
f, fb = figure(s, "f18_scnp.png", 3.62, 6.5, max_w=11.9)
foot = txt(s, "tried twice: on the merged silhouette, then on the per-trait cycle "
              "where “same class” means same trait",
           0.85, 6.5, 11.9, 0.6, size=14, italic=True, color=INK,
           align=PP_ALIGN.CENTER)
anim(s, ids(c1, h1, m1), ids(c2, h2, m2), ids(f), ids(foot))
notes(s, "≈75s. Valverde et al., CVPR 2026 — a topology term for segmentation. "
         "It needs a genuine one-hot target, which most self-supervised losses "
         "lack and the cycle happens to supply: the closing mask is scored "
         "against the reference's own committed labelmap. First attempt sat on "
         "the merged silhouette, where 'same class' means only foreground, and "
         "the module's own docstring pre-registered beetle as the paper's "
         "detrimental small-structure analogue — which is what it measured, "
         "−0.09 significant, with the worst TTA interaction of any arm. Second "
         "attempt composed it with the per-trait cycle, where same-class "
         "finally means same-trait and the term has cross-trait bleed to bite "
         "on. That is where the docstring predicted +0.5 to +2 on beetle. It "
         "measured −0.01. Both deltas sit inside the ±0.3 band that was "
         "written down as 'no effect', so the term is closed in every form we "
         "could give it — and the butterfly +0.04 is significant and "
         "irrelevant, four hundredths of a point.")

# ── 20 · every loss variant ─────────────────────────────────────────────────
s = slide()
kicker(s, "EVERY LOSS VARIANT WE RAN  ·  ON SAM 2")
title(s, "Every loss variant")
f1, _ = figure(s, "f7_forest_8.png", 1.5, 6.35, max_w=8.6)
f2, fb = figure(s, "f7_forest_13.png", 1.5, 6.35, max_w=8.6)
foot = txt(s, "LoRA and full fine-tuning · 1 and 10 epochs · five learning "
              "rates · five loss terms · two datasets",
           0.75, 6.55, 11.9, 0.5, size=12.5, color=MUTED,
           align=PP_ALIGN.CENTER)
anim(s, ids(f2), ids(foot))
notes(s, "≈75s. Beetle first: every arm null or negative, four independent "
         "measurements of the same small negative. Then butterfly, where the "
         "silhouette objective pays about a point. The two arms that hurt "
         "badly are worth naming — full fine-tuning at −3.95, and the "
         "margin fold at −3.62, which damaged the very stack it was designed "
         "to protect by shifting the calibration TTA decides with.")

# ── 21 · why nothing more was there ─────────────────────────────────────────
s = slide()
kicker(s, "WHY")
title(s, "The pot they competed for")
f, fb = figure(s, "f8_oracle.png", 1.92, 5.9, max_w=9.6)
foot = txt(s, "the rest is regions the tracker never proposes, which no loss on "
              "proposed logits reaches",
           0.75, 6.05, 11.9, 0.6, size=14, bold=True, color=INK,
           align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot))
notes(s, "≈60s. This is the measurement that should have come first. For most "
         "of the project the quoted headroom was silhouette minus mIoU — a "
         "correlate at r = 0.86, not a ceiling. Measured properly it is six "
         "times smaller, and it explains every null at once.")

# ── 22 · scoring is a technique too ─────────────────────────────────────────
s = slide()
kicker(s, "AND ONE THAT ISN'T A MODEL CHANGE AT ALL")
title(s, "Three defensible scorings")
f, fb = figure(s, "f16_readings.png", 1.75, 5.9, max_w=9.4)
foot = txt(s, "every fine-tune delta in this talk moved by less than 0.3 "
              "under all three readings", 0.75, 6.1, 11.9, 0.6, size=14,
           color=INK, align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot))
notes(s, "≈50s. Worth saying that the boundary tolerance — one-pixel dilate "
         "and erode, best of three — is enormous on beetle's thin traits and "
         "marginal on wing patches, which is why it is recorded beside the "
         "plain metric rather than replacing it.")

# ── 23 · the ledger ─────────────────────────────────────────────────────────
s = slide()
kicker(s, "THE LEDGER")
title(s, "The ledger")
f, fb = figure(s, "f13_ledger.png", 1.45, 7.1, max_w=11.6)
anim(s, ids(f))
notes(s, "≈60s. On SAM 2 the shape was clean: everything that paid changed "
         "what the model is shown — which reference, how many, which "
         "orientation — and nothing that changed the objective cleared noise "
         "on beetle. The two largest negatives are compositions that were "
         "assumed safe. The three new entries in the middle are the last act: "
         "a newer backbone, our loss stack on it, and the architecture arm — "
         "all positive, all on beetle. The bottom row is the other new model, "
         "whose headline feature cost six points. Leave this up for questions.")

# ── 24 · where it lands ─────────────────────────────────────────────────────
s = slide()
kicker(s, "WHERE THAT LEAVES THE NUMBERS")
title(s, "Where the numbers land")
g1 = pic(s, "f1_gates_2.png", 1.15, 1.75, w=11.0)
g2 = pic(s, "f1_gates_4.png", 1.15, 1.75, w=11.0)
foot = txt(s, "~5 of butterfly's last 6 points sit below the silhouette ceiling",
           0.85, 6.4, 11.9, 0.6, size=14, bold=True, color=INK,
           align=PP_ALIGN.CENTER)
anim(s, ids(g2), ids(foot))
notes(s, "≈55s. Diamonds are ours, circles the target, bars the run-to-run "
         "spread. Beetle passes at both shot counts on both statistics. The "
         "butterfly bars are as tight as the target's — the protocol behaves; "
         "the level is what is missing.")

# ── 25 · the two swaps ──────────────────────────────────────────────────────
s = slide()
kicker(s, "THE LAST TECHNIQUE  ·  CHANGE THE MODEL")
title(s, "The two swaps")
c1 = card(s, 0.7, 1.98, 5.85, 2.6)
t1 = txt(s, "SeC-4B", 1.2, 2.12, 5.2, 0.6, size=30, bold=True, font=HEAD,
         color=TEAL)
t1b = txt(s, "A concept LVLM steering a retrained SAM 2.1\ntracker: identity "
             "reasoning, not pixel memory.", 1.0, 3.0, 5.3, 1.4,
          size=12, color=INK, line_spacing=1.35)
c2 = card(s, 6.75, 1.98, 5.85, 2.6)
t2 = txt(s, "SAM 3", 7.35, 2.12, 4.8, 0.6, size=30, bold=True, font=HEAD,
         color=TEAL)
t2b = txt(s, "The same tracker design one generation on,\nso every technique "
             "here ports unchanged.", 7.05, 3.0, 5.3, 1.4, size=12,
          color=INK, line_spacing=1.35)
same = txt(s, "same 20 draws   ·   same protocol   ·   the winning stack on top",
           0.85, 4.85, 11.9, 0.6, size=15, bold=True, color=TEAL,
           align=PP_ALIGN.CENTER)
extra = txt(s, "one new arm: LoRA on the memory attention, the module that "
               "forms proposals", 0.7, 5.8, 11.9, 0.5, size=14, color=INK,
            align=PP_ALIGN.CENTER)
anim(s, ids(c1, t1, t1b), ids(c2, t2, t2b), ids(same), ids(extra))
notes(s, "≈55s. Cross-family comparisons are exactly what the comparability "
         "key exists to refuse, so they pass through one sanctioned relaxation "
         "and every row records the fields it had to ignore. The support draws "
         "are model-independent by construction and hash-pinned, so the "
         "pairing is legitimate. The arch arm is the one genuinely new idea in "
         "this stage — everything else is a port.")

# ── 26 · SeC-4B ─────────────────────────────────────────────────────────────
s = slide()
kicker(s, "RESULT  ·  SeC-4B")
title(s, "SeC-4B: the swap does not pay")
f, fb = figure(s, "f20_sec.png", 1.68, 6.05, max_w=11.2)
foot1 = txt(s, "concept injection costs 5.92 and 6.23 mIoU, which is 8.6% "
               "and 10.0% of what it was added to",
            0.75, 6.15, 11.9, 0.6, size=14, color=CRIMSON, bold=True,
            align=PP_ALIGN.CENTER)
foot2 = txt(s, "the loss stack still pays on it: +0.60 (+0.8%) and +0.58 "
               "(+1.0%), above its value on SAM 2",
            0.75, 6.75, 11.9, 0.6, size=14, color=TEAL, bold=True,
            align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot1), ids(foot2))
notes(s, "≈75s. Read the deployment row first: about three points below SAM 2 "
         "on both datasets. Its oracle pot matches SAM 2's, so the retrained "
         "encoder does not propose the thin structures either — same boundary, "
         "different weights. Then the concept injection, which is the whole "
         "reason to try this model, and it is the largest single negative in "
         "the project. A pre-registered expectation was also falsified here: "
         "we predicted SeC's native gate would stay quiet on this benchmark "
         "because its histograms cannot see a 180° flip. The flip part held, "
         "but the gate fires on 59% of butterfly episodes and 87% of beetle "
         "ones — two different specimens clear its threshold — so native SeC "
         "would sit below even the 'off' cells we report. Reporting off is the "
         "charitable reading. The one good news is the last line: the loss "
         "stack transfers, and transfers better here than on SAM 2.")

# ── 27 · SAM 3 ──────────────────────────────────────────────────────────────
s = slide()
kicker(s, "RESULT  ·  SAM 3")
title(s, "SAM 3: the swap pays")
f, fb = figure(s, "f19_sam3.png", 1.72, 6.2, max_w=11.2)
foot = txt(s, "beetle's best 1-shot number: 64.12 → 69.88, "
              "+5.77 ✱ (+9.0%) vs the SAM 2 stack",
           0.75, 6.35, 11.9, 0.6, size=16, bold=True, font=HEAD, color=COPPER,
           align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot))
notes(s, "≈80s. Four rungs on beetle, every one significant, and they "
         "compose: the backbone is worth +2.14 raw, TTA +0.99 on top, the "
         "crowned loss stack +1.59 — its largest contribution on any family — "
         "and the architecture arm +1.96 on top of that. Butterfly is the "
         "control: everything lands at 76.5 to 76.6, statistically "
         "indistinguishable from the SAM 2 crown, which is the third "
         "independent confirmation that Major's residual is a protocol and "
         "layout story rather than model capacity. Two model families have now "
         "left it untouched.")

# ── 28 · the mechanism ──────────────────────────────────────────────────────
s = slide()
kicker(s, "WHY IT WORKED  ·  AND WHY THAT MATTERS")
title(s, "The ceiling itself moved")
f, fb = figure(s, "f21_ceiling.png", 1.75, 5.75, max_w=10.4)
foot1 = txt(s, "the arch arm trains the memory attention, the module that decides "
               "what is proposed",
            0.75, 5.95, 11.9, 0.6, size=14, color=INK,
            align=PP_ALIGN.CENTER)
foot2 = txt(s, "so it reaches error mass that a loss on proposed logits cannot",
            0.75, 6.6, 11.9, 0.6, size=16, bold=True, font=HEAD, color=COPPER,
            align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot1), ids(foot2))
notes(s, "≈70s. This is the slide that closes the loop with the oracle. On "
         "SAM 2 the assignment pot was 2.65 points and every loss arm was "
         "bounded by it. SAM 3's own pot is smaller still — 1.80 — yet the "
         "stack plus arch delivered three and a half points, which is only "
         "possible if the ceiling moved rather than the assignment improving. "
         "It did: 66.76 to 71.08. Training the proposal-forming module changes "
         "what the tracker offers in the first place. That is the one door the "
         "S13 result said no objective on SAM 2's logits could open, and it is "
         "why the loss programme's portable design mattered — the arms were "
         "written on per-object logit stacks precisely so they could move.")

# ── 29 · what is projected next ─────────────────────────────────────────────
s = slide()
kicker(s, "WHAT IS PRE-REGISTERED NEXT")
title(s, "What is pre-registered next")
f, fb = figure(s, "f22_projection.png", 1.75, 5.8, max_w=11.2)
foot = txt(s, "bands assembled from measured components: shot lift, TTA's growth "
              "with shot, the stack, the arch arm",
           0.75, 5.95, 11.9, 0.6, size=14, color=INK, align=PP_ALIGN.CENTER)
foot2 = txt(s, "a central landing sits ≈ +3 over the paper's best beetle row",
            0.75, 6.55, 11.9, 0.6, size=14, bold=True, color=COPPER,
            align=PP_ALIGN.CENTER)
anim(s, ids(f), ids(foot), ids(foot2))
notes(s, "≈60s. Assembled from measured components only and written down "
         "before the run, in the same discipline as every band in this talk. "
         "Two independent caps agree at about 81 — an oracle projection and a "
         "silhouette anchor. There is also a pre-registered disqualifier: if "
         "beetle's 5-shot base lands below 72, conditioning frames are being "
         "dropped and I stop and debug the plumbing rather than tune. That "
         "trap is real — the commissioned model attends to at most four "
         "conditioning frames, which at 5-shot would silently ignore one "
         "support. It is a config attribute, not weights.")

# ── 30 · close ──────────────────────────────────────────────────────────────
s = slide(dark=True)
kicker(s, "IN ONE LINE EACH")
title(s, "Where it lands")
lines = [
    ("On SAM 2, every technique that paid changed what the model is shown, "
     "not what it optimises.", PAPER),
    ("Orientation TTA is the largest inference-time gain: +5.44, or +7.6%, "
     "label-free and untrained.", TEAL),
    ("On SAM 3 that ordering reverses: the loss stack and the arch arm are "
     "+3.55 of beetle's +5.77, which is +9.0%.", COPPER),
    ("They moved the ceiling rather than the assignment. Butterfly's "
     "residual survived both swaps.", PAPER),
]
groups = []
for i, (l, c) in enumerate(lines):
    t = txt(s, l, 0.72, 2.1 + i * 1.15, 11.9, 1.0, size=15,
            bold=(i in (1, 2)), font=HEAD, color=INK, line_spacing=1.25)
    groups.append(ids(t))
anim(s, *groups)
notes(s, "≈45s. The honest summary is that the ordering flipped when the model "
         "changed: on SAM 2 the objective was the wrong place to spend effort, "
         "and on SAM 3 the same objective — plus the arm that trains what gets "
         "proposed — is most of the gain. Close on the fourth line and take "
         "questions with the ledger slide back up.")

# ── 31 · backup: canonicalization geometry ──────────────────────────────────
s = slide()
kicker(s, "BACKUP  ·  IMPLEMENTATION")
title(s, "Canonicalization, implemented")
cols = [
    ("orientation", TEAL, [
        ("body_axis_angle(lm, head, tail)", MONO),
        ("atan2(tail_cy - head_cy,", MONO),
        ("      tail_cx - head_cx)", MONO),
        ("directed, so a 180° rotation is a", BODY),
        ("different angle, not the same axis", BODY),
        ("accept if deviation", MONO),
        ("   <= circular_std(train)", MONO),
    ]),
    ("placement", TEAL, [
        ("trait_centroids(lm, expected)", MONO),
        ("reference = per-trait MEDIAN", BODY),
        ("over the train split", BODY),
        ("displacement = mean distance over", BODY),
        ("the traits both configs share", BODY),
        ("accept if disp", MONO),
        ("   <= train mean + 1 std", MONO),
    ]),
    ("completeness", TEAL, [
        ("len(trait_ids(lm)) == expected", MONO),
        ("", BODY),
        ("the 180° test, used for strata:", BODY),
        ("orientation_of() compares against", BODY),
        ("the reference and the reference", BODY),
        ("mirrored about the image centre,", BODY),
        ("at the mask's own extent", BODY),
    ]),
]
groups = []
for i, (head, col, lines) in enumerate(cols):
    x = 0.72 + i * 4.03
    c = card(s, x, 1.98, 3.85, 4.05)
    h = txt(s, head, x + 0.28, 2.18, 3.4, 0.4, size=14, bold=True, font=HEAD,
            color=col)
    body = []
    yy = 2.72
    for text, font in lines:
        if text:
            body.append(txt(s, text, x + 0.28, yy, 3.6, 0.35,
                            size=10.5 if font == MONO else 14,
                            font=font, color=INK if font == MONO else MUTED))
        yy += 0.40 if font == MONO else 0.34
    groups.append(ids(c, h, *body))
foot = txt(s, "circular_std = sqrt(-2 ln R), the train pool's own spread",
           0.75, 6.2, 11.9, 0.4, size=13, color=INK, align=PP_ALIGN.CENTER)
foot2 = txt(s, "every threshold is recorded per run and pinned by a test",
            0.75, 6.7, 11.9, 0.4, size=13, italic=True, color=INK,
            align=PP_ALIGN.CENTER)
anim(s, *groups, ids(foot, foot2))

# ── 32 · backup: pool to draw ───────────────────────────────────────────────
s = slide()
kicker(s, "BACKUP  ·  IMPLEMENTATION")
title(s, "From pool to draw")
rows = [
    ("canonical_pool(ds, group, criteria)",
     "reads TRAIN masks only, returns (pool_ids, stats); stats carry the "
     "thresholds and the rejection counts into the bundle"),
    ("draw_supports(ds, group, pool, selection, shot, run)",
     "the single dispatch point; three call sites had drifted before it existed"),
    ("derive_seed(dataset, group, shot=0, run)",
     "blake2b of the key string, no global RNG; shot is pinned to 0 so 1-shot "
     "is a strict prefix of 5-shot"),
    ("stratified_supports(...)",
     "run r takes stratum (r + k) mod S, member drawn by the run's seed; "
     "systematic coverage rather than equal probability"),
    ("orientation_diverse_supports(...)",
     "n_rot = round(shot × rotated fraction of the train pool), clamped to "
     "[1, shot-1]; degenerates to the ordinary draw where one orientation exists"),
    ("support_set_hash(resolved ids)",
     "sha1 of the IDs actually drawn per run; two arms may only be paired if "
     "this matches"),
]
groups = []
for i, (code, note) in enumerate(rows):
    y = 2.0 + i * 0.8
    t1 = txt(s, code, 0.8, y, 5.35, 0.4, size=11.5, font=MONO, color=TEAL)
    t2 = txt(s, note, 6.35, y - 0.04, 6.2, 0.8, size=11, color=INK,
             line_spacing=1.15)
    groups.append(ids(t1, t2))
foot = txt(s, "memoised above canonical_pool, never inside it, so the leakage "
              "test cannot pass vacuously",
           0.7, 6.9, 11.9, 0.4, size=12.5, italic=True, color=INK)
anim(s, *groups, ids(foot))

# ── 33 · backup: what guards it ─────────────────────────────────────────────
s = slide()
kicker(s, "BACKUP  ·  IMPLEMENTATION")
title(s, "What keeps it honest")
tests = [
    ("threshold is the a priori train circular std", "so it cannot be retuned"),
    ("pool is unchanged when the test split is perturbed", "no leakage"),
    ("orientation is measured against the mask's own extent",
     "beetle crops are 172 to 431 px"),
    ("balanced draws preserve prefix nesting", "1-shot vs 5-shot stays paired"),
    ("the support draw is dispatched in exactly one place",
     "the test discovers callers rather than listing them"),
    ("registry refuses to pair two different support arms",
     "protocol.support_selection is a comparability field"),
]
groups = []
for i, (name, why) in enumerate(tests):
    x = 0.8 + (i % 2) * 6.2
    y = 2.0 + (i // 2) * 1.28
    d = shape(s, MSO_SHAPE.OVAL, x, y + 0.1, 0.16, 0.16, fill=TEAL)
    t1 = txt(s, name, x + 0.4, y, 5.5, 0.5, size=12.5, color=INK,
             line_spacing=1.15)
    t2 = txt(s, why, x + 0.4, y + 0.62, 5.5, 0.4, size=11.5, color=INK)
    groups.append(ids(d, t1, t2))
rec = txt(s, "recorded per run: criteria, criteria_applied, pool_size, "
             "train_size, rejections by reason, threshold_deg, "
             "train_circular_mean_deg, train_resultant_R, threshold_rule, "
             "placement_threshold_px",
          0.7, 5.9, 11.9, 0.9, size=12, color=INK, line_spacing=1.25)
foot = txt(s, "the approximation is auditable after the fact, which is the point",
           0.8, 6.75, 11.9, 0.5, size=13, italic=True, color=COPPER)
anim(s, *groups, ids(rec), ids(foot))

# ── presenter notes: short directions, matching NARRATION.md ────────────────
DIRECTIONS = [
 "35s. Frame the talk as a ledger: one baseline, seven techniques, each "
 "measured the same way. Flag that the last section changes the model, not "
 "the method.",
 "50s, 6 clicks. Name each panel as it lands. Establish that this is real "
 "project output, not a diagram. End on the failure: the query is inverted "
 "and the prediction inherits the reference layout. Four of seven techniques "
 "exist because of it.",
 "45s, 8 clicks. Define the unit before any number is spoken. Build the five "
 "steps, then the worked example, then the semantics card. The line to land: "
 "traits are averaged, not pixels, so a missed antenna costs as much as a "
 "missed elytra. Say 0.655 and point at the zero.",
 "55s, 4 clicks. Say all four numbers. Then the protocol line. Close on the "
 "spread and why it counts as a result: it measures sensitivity to the draw, "
 "so a matched mean with double the variance is not a reproduction.",
 "55s, 5 clicks. One click per step. Emphasise the threshold discipline: "
 "55.6 degrees is the train pool's own circular std, pinned by a test. Give "
 "the train-only step ten seconds; otherwise you get a leakage question.",
 "45s, 3 clicks. A defect curation itself introduced. The middle click is "
 "the instructive one: stratified random fails at n=20, three species went "
 "unused. Skip the seeding details.",
 "55s, 2 clicks. Say the three levels and the three spreads. Then the "
 "per-trait breakdown: head 43.6 to 55.9 for curation, pronotum +8.7 for "
 "balancing. Each fix moved the trait its diagnostic predicted.",
 "50s, 4 clicks. Same rule, different statistic, because butterflies have a "
 "different defect. Say 68.54 to 70.30 and note the spread matters more. "
 "Close on the disjointness of the two criteria.",
 "45s, 3 clicks. Contrast +9.96 and +1.48, then explain the asymmetry: five "
 "references fix defects whose cause is the draw. The ninefold spread "
 "collapse on butterfly is the signature of a draw-invariant defect.",
 "50s, 3 clicks. Say 37 percent, 74.95, 0.48, and that the weighting "
 "reproduces 47.4. The copy baseline at zero is the strongest evidence. "
 "Third click explains why nothing reference-side moved it.",
 "40s, 2 clicks. One correlation and the quartile spread. Establish that the "
 "top quartile already exceeds the target and the test set cannot be "
 "curated. Skip unless asked: only 46 percent is a global offset.",
 "15s. One sentence of framing: largest single gain, least machinery.",
 "60s, 6 clicks. Left column builds the two passes, right column the three "
 "properties. Note the bottom-left panel is the project's own render under "
 "the same rotation the code applies.",
 "60s, 5 clicks. Read the criterion off the code. One sentence each for the "
 "two properties: no ground truth, and caching keeps it at two passes. Close "
 "on the limitation, that it selects and cannot improve.",
 "60s, 2 clicks. Say the four deltas, then 47.40 to 73.27 against a 74.95 "
 "ceiling, 94 percent of the deficit. The silhouette rising with it makes "
 "this re-segmentation. Second click is the variance cost.",
 "50s, 3 clicks. Establish the reference-side fix and that attention selects "
 "rather than blends. Then the square and the interaction. State the "
 "procedural lesson: a one-dimensional sweep would have shipped the worse arm.",
 "15s. Mark the shift: everything before this is inference-time, this is the "
 "only technique that changes weights.",
 "55s, 6 clicks. Walk the four legs with the panels. Establish the appeal: "
 "supervision from a mask you already have, so it trains on unlabelled "
 "queries.",
 "55s, 4 clicks. Differentiability first, with the 0.00 to 26 and 351 "
 "ratios. Then the supervision axis and the law it implies: each objective "
 "pays only where it has room.",
 "60s, 4 clicks. Transform, then why it was a candidate here, then both "
 "attempts. Say the pre-registration: beetle was named as the detrimental "
 "case, and the composition was predicted at +0.5 to +2 and measured at "
 "-0.01. If short, compress the transform and keep the pre-registration.",
 "55s, 2 clicks. Do not enumerate the arms. Give the pattern, name the two "
 "that hurt, and say why the margin fold failed. State that this slate is on "
 "SAM 2, because the next section reverses it.",
 "50s, 2 clicks. The correction: the quoted headroom was a correlate, the "
 "measured ceiling is 2.65 and 0.38. Then the sentence the last section "
 "depends on, about regions never proposed. Flag it.",
 "35s, 2 clicks. Three readings of the same predictions. Two points of "
 "spread with no model change. Note the model-swap deltas move between "
 "readings and both are quoted. First candidate for cutting.",
 "60s. Pause and let them read. Then the shape: on SAM 2 everything that "
 "paid changed what the model is shown. Point out the two compositions, the "
 "three new positive rows, and the bottom row. Leave this up for questions.",
 "40s, 2 clicks. Beetle passes both gates, butterfly does not, and five of "
 "its last six points sit below the silhouette ceiling. That is the argument "
 "for changing the model.",
 "50s, 4 clicks. Both families and what was held fixed. One sentence on the "
 "cross-family pairing discipline: draws are model-independent and "
 "hash-pinned, and the one relaxation records what it ignored. Then the new "
 "arm.",
 "70s, 3 clicks. Deployment row first, then the oracle pot being unchanged. "
 "Concept injection at about -6 with 0 and 1 of 20 seeds. Report the "
 "falsified pre-registration directly, with the fire rates. Close on the "
 "loss stack still paying.",
 "75s, 2 clicks. Four rungs, every one significant, then 64.12 to 69.88 and "
 "70.56 under the paper's own metric. Second click is butterfly as the "
 "control, flat, which is the third confirmation that its residual is "
 "protocol.",
 "65s, 3 clicks. Give this room. Smaller pot, larger gain, so the ceiling "
 "moved: 66.76 to 71.08. Connect back to the earlier slide explicitly. Add "
 "that the arms were written on per-object logit stacks so they could "
 "transfer.",
 "55s, 3 clicks. Present the projection as a projection. Bands from measured "
 "components, two caps near 81, a disqualifier at 72. Mention the "
 "conditioning-frame cap as configuration, not weights. Second candidate for "
 "cutting.",
 "45s, 4 clicks. One line per click. Land the reversal: on SAM 2 the "
 "objective was the wrong place to spend effort, on SAM 3 it is most of the "
 "gain, and butterfly's residual survived both swaps.",
 "BACKUP, not presented. Jump here for how the criteria are computed. Three "
 "columns: the directed axis, the centroid displacement, and completeness "
 "plus the 180 degree test. See TECHNICAL_QA.md sections 1 to 3.",
 "BACKUP, not presented. Jump here for the pool to draw pipeline and the "
 "seeding. See TECHNICAL_QA.md sections 4 to 6.",
 "BACKUP, not presented. Jump here when asked how any of it is guarded, or "
 "what is recorded per run. See TECHNICAL_QA.md sections 7 and 8.",
]
assert len(DIRECTIONS) == len(prs.slides._sldIdLst), (
    f"{len(DIRECTIONS)} notes for {len(prs.slides._sldIdLst)} slides")
for sld, note in zip(prs.slides, DIRECTIONS):
    sld.notes_slide.notes_text_frame.text = note

prs.save(str(OUT))
Path("anim_spec.json").write_text(json.dumps(ANIM, indent=1))
print(f"wrote {OUT} · {len(prs.slides._sldIdLst)} slides")
print("clicks per slide:", {k + 1: len(v) for k, v in ANIM.items()})
