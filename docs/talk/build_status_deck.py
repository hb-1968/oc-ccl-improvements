#!/usr/bin/env python3
"""Build the status deck: what landed since the techniques talk, and where it goes.

House style shared with Hunter_SST_techniques.pptx and
Hunter_SST_second_reference.pptx. Figures come from make_status_figures.py
(project data only). Click-builds are injected afterwards by add_animations.py,
driven by anim_spec_status.json.
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

FIG = Path("figures_status")
OUT = Path("Hunter_SST_status.pptx")

INK = RGBColor(0x00, 0x00, 0x00)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)

HEAD = "Courier New"
BODY = "Courier New"

W, H = 13.333, 7.5
ANIM: dict[int, list[list[int]]] = {}

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


# ── primitives, identical to the other two decks ────────────────────────────
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


def flat(gs):
    """One click out of several card groups."""
    return [i for g in gs for i in g]


def anim(s, *click_groups):
    ANIM[prs.slides.index(s)] = [list(g) for g in click_groups]


def pair0_scores():
    """The forward and closing mIoU of training pair 0, base and arch arms.

    Read from s17.zip's own `pairs.csv`, which is the table the panel PNGs on
    that slide were written beside.
    """
    import csv
    import io
    import zipfile

    out = {}
    src = Path("../../results/s17.zip")
    with zipfile.ZipFile(src) as z:
        for r in csv.DictReader(io.TextIOWrapper(z.open("beetle/pairs.csv"))):
            if r["train_step"] == "0" and r["arm"] in ("base", "arch"):
                out[r["arm"]] = {"fwd": 100 * float(r["fwd_miou"]),
                                 "cls": 100 * float(r["cls_miou"]),
                                 "found": r["fwd_found"]}
    assert set(out) == {"base", "arch"}, out
    assert out["base"]["found"] == "1;1;1;0;1", out["base"]["found"]
    return out


def rows_card(s, lines, top=2.05, h=0.95, gap=0.18, size=14.5, num=True):
    """A stack of outlined cards, one per line. Returns the click groups."""
    groups = []
    for i, line in enumerate(lines):
        y = top + i * (h + gap)
        c = card(s, 0.7, y, 11.9, h)
        parts = [txt(s, line, 1.65 if num else 1.05, y + 0.16, 10.4, h - 0.3,
                     size=size, color=INK)]
        if num:
            parts.append(txt(s, f"{i + 1:02d}", 1.0, y + 0.18, 0.6, 0.35,
                             size=13, bold=True, font=HEAD, color=INK))
        groups.append(ids(c, *parts))
    return groups


# ── 1 · title ───────────────────────────────────────────────────────────────
s = slide()
txt(s, "WHERE THE PROJECT STANDS", 0.7, 1.45, 11.9, 0.95, size=40, bold=True,
    font=HEAD, color=INK)
txt(s, "Six stages since the last talk, one established recipe, one open "
       "mechanism",
    0.72, 2.5, 11.9, 0.5, size=17, color=INK)
for i, lab in enumerate(("beetle now passes\nboth shot counts",
                         "SAM 2's fine-tuning\nprogram is closed",
                         "Major's residual is\nnot about the model")):
    x = 0.7 + i * 4.07
    card(s, x, 3.55, 3.77, 1.5)
    txt(s, lab, x, 4.0, 3.77, 0.8, size=14, bold=True, font=HEAD, color=INK,
        align=PP_ALIGN.CENTER)
txt(s, "S15, S16, S17, S17b, S17c, S17d / S18 built and waiting / Hunter Bridges",
    0.72, 6.5, 11.9, 0.4, size=13, color=INK)

# ── 2 · where the last talk left it ─────────────────────────────────────────
s = slide()
kicker(s, "RECAP")
title(s, "Where that talk left it")
g = rows_card(s, [
    "Seven techniques measured on SAM 2. The ones that paid all changed what "
    "the model is shown, not what it optimises.",
    "The model swap paid on beetle: SAM 3 plus the architecture-adapted arm "
    "took the best 1-shot number from 64.12 to 69.88.",
    "Every loss arm on SAM 2 was null, and nobody could say why.",
    "Butterfly Major's residual survived curation, both rotation fixes, "
    "5-shot, every loss and two model families.",
], top=2.05, h=1.05, gap=0.17)
foot = txt(s, "Three questions were open: is a second labelled specimen worth "
              "more than a rotated one, what does the champion do at five "
              "shots, and why is every SAM 2 loss arm null.",
           0.7, 6.55, 11.9, 0.8, size=14, color=INK)
anim(s, *g, ids(foot))

# ── 3 · what landed since ───────────────────────────────────────────────────
s = slide()
kicker(s, "SINCE THEN")
title(s, "What landed since that talk")
entries = [
    ("S16", "which second reference pays", "PASS 14/14"),
    ("S15", "the champion at five shots", "PASS 18/18"),
    ("S17", "what the fine-tune does to its own pairs", "PASS 20/20"),
    ("S17b", "the repairs, and whether they compose", "PASS 20/20"),
    ("S17c", "the rest of the branch, in one session", "PASS 11/11"),
    ("S17d", "the last cell, and the sentinel itself", "PASS 11/11"),
    ("S18", "what may cross the centre of the cycle", "built, not yet run"),
]
groups = []
for i, (tag, what, verdict) in enumerate(entries):
    y = 1.98 + i * 0.66
    c = card(s, 0.7, y, 11.9, 0.56)
    a = txt(s, tag, 1.0, y + 0.13, 1.1, 0.35, size=14, bold=True, font=HEAD,
            color=INK)
    b = txt(s, what, 2.3, y + 0.13, 7.2, 0.35, size=14, color=INK)
    d = txt(s, verdict, 9.6, y + 0.13, 2.8, 0.35, size=14, bold=True,
            color=INK, align=PP_ALIGN.RIGHT)
    groups.append(ids(c, a, b, d))
anim(s, flat(groups[:2]), flat(groups[2:6]), groups[6])

# ── 4 · the scoreboard ──────────────────────────────────────────────────────
s = slide()
kicker(s, "WHERE THE NUMBERS ARE")
title(s, "The scoreboard, as of today")
f, _ = figure(s, "k1_scoreboard.png", 1.98, 5.4)
f1 = txt(s, "Beetle reproduces at both shot counts and now exceeds the paper "
            "at both. Butterfly Major does not reproduce at either.",
         0.7, 5.6, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "Everything in this talk is either one of those four numbers "
            "moving, or the reason one of them will not.",
         0.7, 6.3, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 5 · S16: the two second references ──────────────────────────────────────
s = slide()
kicker(s, "S16")
title(s, "Two ways to spend a second reference")
lft = txt(s, "IMPLICIT  (orientation TTA)", 0.7, 1.95, 5.6, 0.32, size=14,
          bold=True, font=HEAD, color=INK)
p1 = pic(s, "k20_support.png", 0.7, 2.4, h=1.9)
p2 = pic(s, "k20_support_rot.png", 2.35, 2.4, h=1.9)
l1 = txt(s, "the same specimen turned 180 degrees, outside the memory bank, "
            "picked per episode by a confidence rule",
         0.7, 4.45, 5.6, 0.8, size=13, color=INK)
c1 = txt(s, "cost: 0 labels", 0.7, 5.3, 5.6, 0.32, size=14, bold=True,
         color=INK)
rgt = txt(s, "EXPLICIT  (a real 2-shot)", 7.0, 1.95, 5.6, 0.32, size=14,
          bold=True, font=HEAD, color=INK)
p3 = pic(s, "k20_support.png", 7.0, 2.4, h=1.9)
p4 = pic(s, "k20_query.png", 8.65, 2.4, h=1.9)
l2 = txt(s, "a second individual from the same pool, inside the bank, "
            "selected by memory attention",
         7.0, 4.45, 5.6, 0.8, size=13, color=INK)
c2 = txt(s, "cost: 1 label", 7.0, 5.3, 5.6, 0.32, size=14, bold=True,
         color=INK)
div = shape(s, MSO_SHAPE.RECTANGLE, 6.62, 1.95, 0.02, 3.7, fill=INK)
foot = txt(s, "Every headline arm since S5c has used the implicit version. The "
              "explicit one had never been measured.",
           0.7, 5.9, 11.9, 0.6, size=15, color=INK)
anim(s, ids(lft, p1, p2, l1, c1), ids(div, rgt, p3, p4, l2, c2), ids(foot))

# ── 6 · S16 result ──────────────────────────────────────────────────────────
s = slide()
kicker(s, "S16 RESULT")
title(s, "Which one pays flips by dataset")
f, _ = figure(s, "k2_s16.png", 1.98, 5.1)
f1 = txt(s, "Both signs were pre-registered, from each dataset's own residual: "
            "Major's defect is query orientation, beetle's is species "
            "diversity, and a rotated copy adds no species.",
         0.7, 5.3, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "The practical read: two labels on stock SAM 2 puts beetle at "
            "69.04, within noise of the trained champion at one label.",
         0.7, 6.25, 11.9, 0.7, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 7 · S15: the projection and the landing ─────────────────────────────────
s = slide()
kicker(s, "S15")
title(s, "The champion at five shots")
f1, _ = figure(s, "k3_s15_band_1.png", 1.98, 4.9)
f2, _ = figure(s, "k3_s15_band_2.png", 1.98, 4.9)
t1 = txt(s, "Bands written before the run, from two independent anchors: the "
            "silhouette score and the 1-shot oracle ceiling plus the measured "
            "shot mechanism.",
         0.7, 5.1, 11.9, 0.9, size=15, color=INK)
t2 = txt(s, "beetle 78.91, inside. Gate PASS 18 of 18, and +3.3 above the "
            "paper's best beetle row.",
         0.7, 6.05, 11.9, 0.6, size=15, bold=True, color=INK)
t3 = txt(s, "Major 77.47, below its band, and beaten by its own base.",
         0.7, 6.6, 11.9, 0.5, size=15, color=INK)
anim(s, ids(f1, t1), ids(f2, t2), ids(t3))

# ── 8 · the S15 cells ───────────────────────────────────────────────────────
s = slide()
kicker(s, "S15")
title(s, "Every cell, both datasets")
f, _ = figure(s, "k4_s15_cells.png", 1.98, 5.4)
f1 = txt(s, "On beetle the adapter is the top of the stack. On Major the "
            "honest number is 77.95 with no adapter at all.",
         0.7, 5.6, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "The backbone edge is real on both: SAM 3's plain weights beat SAM "
            "2's by +3.06 on beetle and +0.52 on Major.",
         0.7, 6.25, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 9 · the adapter is beetle-specific ──────────────────────────────────────
s = slide()
kicker(s, "S15")
title(s, "The adapter is beetle-specific")
f, _ = figure(s, "k5_s15_arch.png", 1.98, 5.1)
f1 = txt(s, "Same adapter, same protocol, same seeds. On beetle it pays on "
            "every seed. On Major it costs on every seed.",
         0.7, 5.35, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "At 1 shot the beetle contribution was +1.96. At 5 shots it is "
            "+0.97, and the next slide is why.",
         0.7, 6.0, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 10 · the assignment pot closes ──────────────────────────────────────────
s = slide()
kicker(s, "S15")
title(s, "Five labels close the pot")
f, _ = figure(s, "k6_pot.png", 1.98, 5.35)
f1 = txt(s, "The hatched bar is what the same predictions would score if every "
            "proposed region were assigned to the right trait.",
         0.7, 5.55, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "At five supports that gap is gone, so there is nothing left for "
            "an assignment loss to win. The pot is a function of the label "
            "budget, not of the objective.",
         0.7, 6.2, 11.9, 0.9, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 11 · the anatomy: one epoch is ten pairs ────────────────────────────────
s = slide()
kicker(s, "S17 ANATOMY")
title(s, "One epoch is ten bad pairs")
f, _ = figure(s, "k7_tail.png", 1.98, 5.2)
f1 = txt(s, "Measured on the base model over all 120 beetle training pairs, "
            "before any arm was designed.",
         0.7, 5.4, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "The merged-silhouette recipe has no such tail. So the question "
            "became what those ten pairs are charging for.",
         0.7, 6.05, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 12 · the sentinel ───────────────────────────────────────────────────────
s = slide()
kicker(s, "S17 ANATOMY")
title(s, "What they are charging for")
f, _ = figure(s, "k8_sentinel.png", 1.98, 5.25)
f1 = txt(s, "When SAM 2 decides a trait is absent it replaces the mask with a "
            "constant of -1024. Per-trait BCE then charges 1024 times the "
            "trait's area, on all 20 of the worst channels.",
         0.7, 5.45, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "62% of the loss value is that constant, and it has no gradient. "
            "Nothing in the objective can reach a channel the model did not "
            "find.",
         0.7, 6.4, 11.9, 0.8, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 13 · the cycle's own images ─────────────────────────────────────────────
s = slide()
kicker(s, "S17")
title(s, "The cycle's own images")
PAIR0 = pair0_scores()          # the same pair the panels were drawn from
labels = [
    ("forward leg, on the query",
     f"base {PAIR0['base']['fwd']:.2f}, trained {PAIR0['arch']['fwd']:.2f}"),
    ("closing leg, back on the reference",
     f"base {PAIR0['base']['cls']:.2f}"),
    ("closing leg, trained arm",
     f"{PAIR0['arch']['cls']:.2f}"),
]
files = ["k10_base_fwd.png", "k10_base_cls.png", "k10_arch_cls.png"]
xs = [1.1, 5.1, 8.9]
panels = []
for f_, x, (lab, num) in zip(files, xs, labels):
    p = pic(s, f_, x, 2.35, h=2.7)
    w = p.width / 914400
    t = txt(s, lab, x - 0.4, 2.0, w + 0.8, 0.3, size=12, bold=True, color=INK,
            align=PP_ALIGN.CENTER)
    n = txt(s, num, x - 0.4, 5.12, w + 0.8, 0.3, size=12.5, bold=True,
            color=INK, align=PP_ALIGN.CENTER)
    panels.append((p, t, n))
f1 = txt(s, "One real training pair, from the bundle. The antennae are missing "
            "from the forward prediction and taken over by the leg channel in "
            "the closing one.",
         0.7, 5.45, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "Training moves the closing image. It does not move the forward "
            "one, which is the image the evaluator scores.",
         0.7, 6.35, 11.9, 0.7, size=15, color=INK)
anim(s, ids(panels[0]), ids(panels[1]), ids(panels[2], f1), ids(f2))

# ── 14 · absorption ─────────────────────────────────────────────────────────
s = slide()
kicker(s, "S17 RESULT")
title(s, "The reverse leg absorbs it all")
f, _ = figure(s, "k9_absorb.png", 1.98, 5.3)
f1 = txt(s, "Every arm lowers its own objective, and every arm does it on the "
            "closing leg. The correlation between the two legs is 0.01 to "
            "0.31.",
         0.7, 5.5, 11.9, 0.8, size=15, color=INK)
f2 = txt(s, "On butterfly the forward leg does move, +1.39, which is why "
            "OC-CCL pays there and not on beetle. The asymmetry is a property "
            "of the training pairs.",
         0.7, 6.3, 11.9, 0.9, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 15 · three faults, three repairs ────────────────────────────────────────
s = slide()
kicker(s, "S17b")
title(s, "Three faults, three repairs")
g = rows_card(s, [
    "The closing head absorbs the objective. Repair: put the adapters on the "
    "forward legs only, so the closing frame cannot learn.",
    "The sentinel cuts unfound channels out of the graph. Repair: a term on "
    "the forward objectness score, which is upstream of it.",
    "Exclusivity, kept from the earlier slate: the one closing-loss change "
    "that paid on its own.",
], top=2.1, h=1.15, gap=0.22)
foot = txt(s, "All three are inside the fine-tune. No model-code change, no "
              "extra labels, no change to the evaluator.",
           0.7, 6.3, 11.9, 0.6, size=15, color=INK)
anim(s, *g, ids(foot))

# ── 16 · the ladder ─────────────────────────────────────────────────────────
s = slide()
kicker(s, "S17b AND S17c")
title(s, "The repairs compose")
f, _ = figure(s, "k11_ladder_1.png", 1.98, 5.4)
f1 = txt(s, "+0.69 with TTA on 19 of 20 seeds for the composition, the first "
            "beetle fine-tuning gain on SAM 2 that is not a reshuffle of "
            "channels the model already found.",
         0.7, 5.6, 11.9, 0.9, size=15, color=INK)
anim(s, ids(f), ids(f1))

# ── 17 · the ceiling moves ──────────────────────────────────────────────────
s = slide()
kicker(s, "S17b")
title(s, "The ceiling moves, finally")
f, _ = figure(s, "k12_ceiling.png", 1.98, 5.2)
f1 = txt(s, "The ceiling is what the proposals allow. On SAM 2 it had not "
            "moved for any loss arm in the project.",
         0.7, 5.45, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "It moves here, which says the fine-tune changed what gets "
            "proposed rather than how the proposals are labelled.",
         0.7, 6.1, 11.9, 0.7, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 18 · the weight ─────────────────────────────────────────────────────────
s = slide()
kicker(s, "S17c")
title(s, "The prior is free up to 30x")
f, _ = figure(s, "k13_weight.png", 1.98, 5.3)
f1 = txt(s, "The pre-registration expected a cost at 30. It never appeared: "
            "the curve is a plateau, not a peak.",
         0.7, 5.5, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "The prior pushes on the objectness score, which is exactly the "
            "quantity the sentinel reads.",
         0.7, 6.15, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 19 · scope subsumes weight ──────────────────────────────────────────────
s = slide()
kicker(s, "S17c AND S17d")
title(s, "Scope subsumes the weight")
f, _ = figure(s, "k14_scope.png", 1.98, 5.3)
f1 = txt(s, "Two levers, each real on its own, and the composition lands below "
            "their sum. Pre-registered as either +0.71 if they add or +0.44 if "
            "the scope subsumes the weight.",
         0.7, 5.5, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "65.04 against 65.03 for the scope alone, +0.01 on the same seeds "
            "across two sessions.",
         0.7, 6.45, 11.9, 0.6, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 20 · the fence, made differentiable ─────────────────────────────────────
s = slide()
kicker(s, "S17d")
title(s, "The fence, made differentiable")
f, _ = figure(s, "k15_soft.png", 1.98, 5.0)
f1 = txt(s, "Replacing SAM 2's hard sentinel with a straight-through surrogate "
            "is exact at base weights and turns the score head's gradient from "
            "0 into 0.566 on a real pair.",
         0.7, 5.25, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "It then flips 1 of 19 fence channels on its own, and costs "
            "-0.13 on top of the prior. The loss is now honest about the fence "
            "and does not want it.",
         0.7, 6.2, 11.9, 0.9, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 21 · nothing reaches five shots ─────────────────────────────────────────
s = slide()
kicker(s, "S17c AND S17d")
title(s, "None of it reaches five shots")
f, _ = figure(s, "k16_fiveshot.png", 1.98, 5.1)
f1 = txt(s, "Three SAM 2 adapters have now been measured at five shots and "
            "none of them moves the number.",
         0.7, 5.35, 11.9, 0.6, size=15, color=INK)
f2 = txt(s, "Consistent with S15: at five supports a trait absent from one "
            "reference is present on another, so the fence the fine-tune "
            "repairs is not standing. That reading is not itself measured.",
         0.7, 6.0, 11.9, 0.9, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 22 · the recipe ─────────────────────────────────────────────────────────
s = slide()
kicker(s, "THE OUTPUT")
title(s, "sam2_crown, established")
parts = [
    ("forward-only legs", "removes the absorption"),
    ("objectness prior, weight 10", "reaches past the sentinel"),
    ("exclusivity", "the one closing-loss change that paid"),
    ("arch scopes", "puts the proposal-forming module in scope"),
    ("the S11 carrier", "per-trait, lr 1e-5, one epoch, r=32"),
]
groups = []
for i, (name, why) in enumerate(parts):
    y = 2.0 + i * 0.78
    c = card(s, 0.7, y, 11.9, 0.66)
    a = txt(s, name, 1.0, y + 0.18, 4.6, 0.35, size=14.5, bold=True,
            font=HEAD, color=INK)
    b = txt(s, why, 5.9, y + 0.18, 6.4, 0.35, size=14.5, color=INK)
    groups.append(ids(c, a, b))
foot = txt(s, "Pinned in the registry and re-derivable from a record's own "
              "fields. Every future SAM 2 fine-tune uses it, and a stage that "
              "departs from it says so in its brief.",
           0.7, 6.15, 11.9, 0.9, size=14, color=INK)
anim(s, flat(groups[:3]), flat(groups[3:]), ids(foot))

# ── 23 · where SAM 2 stops ──────────────────────────────────────────────────
s = slide()
kicker(s, "THE FAMILY IS CLOSED")
title(s, "Where SAM 2 stops")
f, _ = figure(s, "k11_ladder_2.png", 1.98, 5.4)
f1 = txt(s, "65.04 with TTA, +1.18 over base, oracle ceiling 67.92. That is "
            "0.16 short of the paper's own OC-CCL row and 4.8 below the SAM 3 "
            "champion.",
         0.7, 5.6, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "Every remaining axis inside the family is measured and closed. "
            "What is left is outside it.",
         0.7, 6.5, 11.9, 0.6, size=15, bold=True, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 24 · S18 ────────────────────────────────────────────────────────────────
s = slide()
kicker(s, "NEXT, BUILT AND WAITING")
title(s, "What S18 asks")
f, _ = figure(s, "k17_s18.png", 1.98, 5.2)
f1 = txt(s, "The cycle is satisfied without the forward mask improving. The "
            "mask itself is not being carried: its similarity to the reference "
            "sits at chance.",
         0.7, 5.4, 11.9, 0.9, size=15, color=INK)
f2 = txt(s, "What is left is the prompt. It hands the reverse leg a full "
            "continuous logit map, a channel the evaluator never scores. S18 "
            "binarises it and keeps the gradient.",
         0.7, 6.3, 11.9, 0.9, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 25 · the port ───────────────────────────────────────────────────────────
s = slide()
kicker(s, "AFTER S18")
title(s, "The port that is left")
g = rows_card(s, [
    "Carry the two repairs that worked, forward-only legs and the objectness "
    "prior, to SAM 3, where the same memory-attention arm already sits at "
    "69.88.",
    "The parts are family-agnostic by construction: both were written against "
    "per-object logit stacks so they could transfer.",
    "The 5-shot result says to expect this at one shot only, and that is a "
    "prediction worth writing down before the run.",
], top=2.15, h=1.2, gap=0.24)
foot = txt(s, "Cost is one session. The alternative on the same axis is a "
              "second support, which S16 already priced at one label.",
           0.7, 6.3, 11.9, 0.7, size=15, color=INK)
anim(s, *g, ids(foot))

# ── 26 · the label budget ───────────────────────────────────────────────────
s = slide()
kicker(s, "THE UNCOMFORTABLE AXIS")
title(s, "What the headline hides")
f, _ = figure(s, "k18_labels.png", 1.98, 4.7)
f1 = txt(s, "SST's claim is about labels. The best Major arm reads 1,727 "
            "labelled masks to choose which single reference to use.",
         0.7, 4.95, 11.9, 0.7, size=15, color=INK)
f2 = txt(s, "The defensible version is that curation stands in for human "
            "judgement, and a human does not need 1,727 masks. That is "
            "testable offline: shrink the pool and find where the drawn "
            "support stops changing.",
         0.7, 5.7, 11.9, 1.0, size=15, color=INK)
f3 = txt(s, "If it converges at about 20, the headline becomes 20 labels, "
            "which is a different paper.",
         0.7, 6.75, 11.9, 0.5, size=15, bold=True, color=INK)
anim(s, ids(f), ids(f1), ids(f2), ids(f3))

# ── 27 · Major's residual ───────────────────────────────────────────────────
s = slide()
kicker(s, "THE OPEN GAP")
title(s, "Major's residual, after all that")
f, _ = figure(s, "k19_major.png", 1.98, 5.5)
f1 = txt(s, "Rotation handling is worth 4 to 5 points and is the only lever "
            "that ever moved this dataset. Everything else lands at zero.",
         0.7, 5.7, 11.9, 0.7, size=15, color=INK)
f2 = txt(s, "Two model families at two shot counts leave it where it was, so "
            "the remaining candidate is the protocol or the annotations, not "
            "capacity.",
         0.7, 6.4, 11.9, 0.8, size=15, color=INK)
anim(s, ids(f), ids(f1), ids(f2))

# ── 28 · close ──────────────────────────────────────────────────────────────
s = slide()
kicker(s, "WHERE IT GOES")
title(s, "Three sentences")
g = rows_card(s, [
    "Beetle is done as a reproduction: it exceeds the paper at one shot and "
    "at five, and the label-cheapest routes to it are now priced.",
    "SAM 2's objective has been taken apart and put back together, and it "
    "tops out 4.8 below what a model swap already gives.",
    "The next three moves are S18's mechanism, the port of the two repairs to "
    "SAM 3, and the label-budget audit that decides what the headline can "
    "say.",
], top=2.2, h=1.25, gap=0.26)
foot = txt(s, "Every number in this talk traces to a bundle in results/ and a "
              "gate that passed.",
           0.7, 6.55, 11.9, 0.6, size=14, color=INK)
anim(s, *g, ids(foot))

# ── 29 · backup: the gates ──────────────────────────────────────────────────
s = slide()
kicker(s, "BACKUP")
title(s, "The gates behind this talk")
rows = [
    ("S15", "champion at 5-shot, both datasets", "PASS 18/18", "2026-09-13"),
    ("S16", "implicit vs explicit second reference", "PASS 14/14", "2026-09-01"),
    ("S17", "the cycle's own images, 5 arms", "PASS 20/20", "2026-09-16"),
    ("S17b", "forward-only legs, objectness", "PASS 20/20", "2026-09-16"),
    ("S17c", "weight, lr, epochs, scope, link", "PASS 11/11", "2026-09-17"),
    ("S17d", "scope x weight, the soft sentinel", "PASS 11/11", "2026-09-17"),
    ("S18", "the palindrome's bottleneck", "built, smoke green", "pending"),
]
xs = [0.75, 2.0, 7.6, 10.7]
for i, row in enumerate(rows):
    y = 2.0 + i * 0.62
    for x, cellv in zip(xs, row):
        txt(s, cellv, x, y, 5.4, 0.4, size=14, bold=(i < 0), color=INK)
txt(s, "Owed, and named in the README: the s11, s13 and s14ft bundles for "
       "their formal gates, which sit on Drive rather than in results/inbox.",
    0.75, 6.6, 11.9, 0.8, size=13, color=INK)

# ── 30 · backup: the numbers ────────────────────────────────────────────────
s = slide()
kicker(s, "BACKUP")
title(s, "The cells this talk quotes")
rows = [
    ("beetle 1-shot, SAM 2 base", "63.20 plain", "63.86 + TTA"),
    ("beetle 1-shot, the recipe", "63.71 plain", "65.04 + TTA"),
    ("beetle 1-shot, SAM 3 champion", "", "69.88 + TTA"),
    ("beetle 5-shot, SAM 2 base", "73.09 plain", "75.16 + TTA"),
    ("beetle 5-shot, SAM 3 champion", "", "78.91 + TTA"),
    ("Major 1-shot, best", "", "76.64 + TTA"),
    ("Major 5-shot, SAM 3 base", "72.30 plain", "77.95 + TTA"),
    ("Major 5-shot, best (SAM 2)", "", "78.07 + TTA"),
]
for i, (a, b, c) in enumerate(rows):
    y = 1.98 + i * 0.56
    txt(s, a, 0.75, y, 6.0, 0.4, size=14, color=INK)
    txt(s, b, 7.0, y, 2.6, 0.4, size=14, color=INK)
    txt(s, c, 9.9, y, 2.6, 0.4, size=14, bold=True, color=INK)
txt(s, "20 seeds per cell. Every cell and every paired delta is in the stage "
       "bundle it came from.",
    0.75, 6.6, 11.9, 0.5, size=13, color=INK)

# ── presenter notes: directions, not a script ───────────────────────────────
DIRECTIONS = [
    "Title. 25 seconds. Frame the talk as a status report with a roadmap: six "
    "stages landed, one recipe came out of them, one mechanism question is "
    "still open. Do not preview the numbers.",
    "45s, 5 clicks. Recap only. If they were at the last talk, click through "
    "the first three and land on the three open questions.",
    "40s, 3 clicks. The map of the talk. Say the verdict column out loud once, "
    "then move; the detail is coming.",
    "60s, 3 clicks. Let them read the figure. Beetle passes both, Major fails "
    "both, and that split organises everything after this slide.",
    "45s, 3 clicks. Only the distinction and the two costs. If short, skip to "
    "the footer line.",
    "55s, 3 clicks. Both signs pre-registered, both landed. The label line is "
    "the one to land.",
    "70s, 3 clicks. Bands first, then the landing, then Major. Name the two "
    "independent anchors, because that is what makes the beetle landing worth "
    "quoting.",
    "45s, 3 clicks. The cells. Do not read them all: say the adapter's place "
    "on beetle and the no-adapter number on Major.",
    "40s, 3 clicks. Same adapter, opposite signs, every seed. Set up the next "
    "slide by saying the contribution halved from 1 shot to 5.",
    "60s, 3 clicks. Give this room. Define the ceiling once, then say the pot "
    "is a function of the label budget. This is the bridge from S15 into the "
    "whole S17 branch.",
    "40s, 3 clicks. The measurement that started the branch. Ten pairs, more "
    "than half the epoch.",
    "70s, 3 clicks. The core mechanism of the talk. Say the -1024 constant, "
    "the 1024-times-area arithmetic, and then the consequence: no gradient, so "
    "no loss can reach an unfound channel.",
    "60s, 4 clicks. Real images from the bundle. Point at the missing "
    "antennae, then at the leg channel taking them over in the closing image, "
    "then say what training moves and what it does not.",
    "55s, 3 clicks. The verdict of S17. Absorption on every arm, and butterfly "
    "as the control that shows why OC-CCL ever paid.",
    "50s, 4 clicks. One line per repair. Emphasise that all three are inside "
    "the fine-tune.",
    "45s, 2 clicks. The ladder. Say the composition number and the phrase "
    "about it not being a reshuffle.",
    "40s, 3 clicks. Short. First ceiling movement on this family.",
    "35s, 3 clicks. The plateau, and the falsified prediction. Say that the "
    "prediction was wrong out loud.",
    "45s, 3 clicks. Sub-additive, and pre-registered as one of two outcomes. "
    "The measured answer is the second one.",
    "55s, 3 clicks. The cleanest negative result in the branch. Exact at base "
    "weights, gradient turned on, and the loss still does not want the fence.",
    "40s, 3 clicks. Three adapters, no 5-shot effect. Flag the mechanism "
    "sentence as unmeasured.",
    "55s, 3 clicks. The deliverable. Read the five components, not the "
    "evidence column; leave that for questions.",
    "50s, 3 clicks. Where the family stops, with the champion on the same "
    "axis. This is the handoff into the roadmap.",
    "70s, 3 clicks. The mechanism question. Say the similarity-at-chance "
    "result first, because it rules out the obvious story, then the prompt as "
    "the only route left.",
    "55s, 4 clicks. The port. One session, and a prediction to pre-register.",
    "60s, 4 clicks. The uncomfortable number. Do not defend it; state the test "
    "that would settle it and what each outcome would mean.",
    "55s, 3 clicks. Everything that has failed to move Major, against the gap. "
    "End on protocol or annotation, not capacity.",
    "45s, 4 clicks. Three sentences, one per click. Stop talking after the "
    "third.",
    "BACKUP, not presented. Jump here for a gate verdict or a landing date.",
    "BACKUP, not presented. Jump here for any level this talk quoted.",
]
assert len(DIRECTIONS) == len(prs.slides._sldIdLst), (
    f"{len(DIRECTIONS)} notes for {len(prs.slides._sldIdLst)} slides")
for sld, note in zip(prs.slides, DIRECTIONS):
    sld.notes_slide.notes_text_frame.text = note

prs.save(str(OUT))
Path("anim_spec_status.json").write_text(json.dumps(ANIM, indent=1))
print(f"wrote {OUT} · {len(prs.slides._sldIdLst)} slides")
print("clicks per slide:", {k + 1: len(v) for k, v in ANIM.items()})
