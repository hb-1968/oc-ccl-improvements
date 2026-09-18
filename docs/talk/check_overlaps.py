#!/usr/bin/env python3
"""Geometry sweep for the deck: text-on-text collisions, edge bleed, and
text boxes that render taller than the gap under them.

Two independent passes:

  PDF pass  — `pdftotext -bbox` gives every word's rendered box. Words whose
              boxes intersect but belong to different lines are a real visual
              collision (LibreOffice renders what PowerPoint will).
  PPTX pass — python-pptx geometry: estimate each text block's rendered height
              from font size, wrap width and line count, then check it against
              the next shape below it and against the slide edges.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from pptx import Presentation
from pptx.util import Emu

EMU = 914400.0
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.40                      # in
CARDLIKE = ("Rounded Rectangle", "Rectangle", "Oval", "Arrow")


# ── pass 1 · rendered-word collisions from the PDF ───────────────────────────
def pdf_pass(pdf: Path):
    out = subprocess.run(["pdftotext", "-bbox-layout", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    root = ET.fromstring(out)
    ns = {"h": "http://www.w3.org/1999/xhtml"}
    problems = []
    for pno, page in enumerate(root.iter("{http://www.w3.org/1999/xhtml}page"), 1):
        pw = float(page.get("width")), float(page.get("height"))
        words = []
        for line in page.iter("{http://www.w3.org/1999/xhtml}line"):
            lbox = (float(line.get("xMin")), float(line.get("yMin")),
                    float(line.get("xMax")), float(line.get("yMax")))
            txt = " ".join(w.text or "" for w in line)
            words.append((lbox, txt.strip()))
        # line-vs-line intersection with a tolerance for normal leading
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                (a, ta), (b, tb) = words[i], words[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > 2 and oy > 2:          # >2pt overlap in both axes
                    ha = a[3] - a[1]
                    if oy > 0.35 * ha:         # more than a descender's worth
                        problems.append(
                            (pno, "text/text", round(oy, 1), ta[:46], tb[:46]))
        # slide-edge bleed
        for (a, ta) in words:
            if not ta:
                continue
            l = a[0] / pw[0] * SLIDE_W
            r = SLIDE_W - a[2] / pw[0] * SLIDE_W
            t = a[1] / pw[1] * SLIDE_H
            btm = SLIDE_H - a[3] / pw[1] * SLIDE_H
            m = min(l, r, t, btm)
            if m < MARGIN:
                problems.append((pno, "edge", round(m, 2), ta[:46], ""))
    return problems


# ── pass 2 · declared geometry from the pptx ─────────────────────────────────
from PIL import ImageFont                                    # noqa: E402
import matplotlib                                            # noqa: E402

_FONT_PATH = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"
#: DejaVu Sans Mono and Courier New have nearly identical advance widths.
_WIDTH_FUDGE = 1.00
_font_cache: dict[int, "ImageFont.FreeTypeFont"] = {}


def _font(px: int):
    if px not in _font_cache:
        _font_cache[px] = ImageFont.truetype(str(_FONT_PATH), px)
    return _font_cache[px]


def _wrapped_lines(text: str, pt: float, width_in: float) -> int:
    """How many lines this paragraph really takes, by measured glyph widths."""
    if not text.strip():
        return 1
    f = _font(max(6, int(round(pt))))
    limit = width_in * 72 / _WIDTH_FUDGE
    lines, cur = 1, ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if f.getlength(trial) <= limit or not cur:
            cur = trial
        else:
            lines += 1
            cur = word
    return lines


def est_text_height(shape) -> float:
    """Rendered height of a text frame, in inches, from measured glyph widths."""
    width_in = shape.width / EMU
    total = 0.0
    for p in shape.text_frame.paragraphs:
        runs = p.runs
        if not runs:
            total += 0.14                      # a blank spacer paragraph
            continue
        pt = max((r.font.size.pt if r.font.size else 18) for r in runs)
        text = "".join(r.text for r in runs)
        ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
        total += _wrapped_lines(text, pt, width_in) * pt * 1.20 * ls / 72.0
        total += (p.space_after.pt / 72.0) if p.space_after else 0.0
    return total


def pptx_pass(deck: Path):
    prs = Presentation(str(deck))
    problems = []
    for idx, sld in enumerate(prs.slides, 1):
        boxes = []
        for sh in sld.shapes:
            geo = (sh.left / EMU, sh.top / EMU,
                   (sh.left + sh.width) / EMU, (sh.top + sh.height) / EMU)
            kind = "text" if sh.has_text_frame and sh.text_frame.text.strip() \
                else ("pic" if sh.shape_type == 13 else "shape")
            label = (sh.text_frame.text.replace("\n", " ")[:44]
                     if kind == "text" else sh.name)
            h = est_text_height(sh) if kind == "text" else (geo[3] - geo[1])
            boxes.append({"geo": geo, "kind": kind, "label": label,
                          "real_bottom": geo[1] + h, "name": sh.name})
        full_bleed = [b for b in boxes
                      if b["geo"][2] - b["geo"][0] > SLIDE_W - 0.2]
        for i, b in enumerate(boxes):
            if b in full_bleed or b["kind"] != "text":
                continue
            brect = (b["geo"][0], b["geo"][1], b["geo"][2], b["real_bottom"])
            for j, o in enumerate(boxes):
                if i == j or o in full_bleed or o["kind"] == "text":
                    continue
                # a text box sitting inside a card is that card's own content
                inside = (brect[0] >= o["geo"][0] - 0.05
                          and brect[2] <= o["geo"][2] + 0.05
                          and brect[1] >= o["geo"][1] - 0.05)
                if inside:
                    continue
                ox = min(brect[2], o["geo"][2]) - max(brect[0], o["geo"][0])
                oy = min(brect[3], o["geo"][3]) - max(brect[1], o["geo"][1])
                if ox > 0.15 and oy > 0.06:
                    problems.append((idx, "text-over-shape", round(oy, 2),
                                     b["label"], o["name"]))
        # a card must contain the text that sits on it
        for b in boxes:
            if b["kind"] != "text":
                continue
            for o in boxes:
                if not o["name"].startswith("Rounded"):
                    continue
                inside = (b["geo"][0] >= o["geo"][0] - 0.02
                          and b["geo"][2] <= o["geo"][2] + 0.02
                          and b["geo"][1] >= o["geo"][1] - 0.02)
                if inside and b["real_bottom"] > o["geo"][3] - 0.05:
                    problems.append((idx, "card-overflow",
                                     round(b["real_bottom"] - o["geo"][3], 2),
                                     b["label"], o["name"]))
    return problems


if __name__ == "__main__":
    deck, pdf = Path(sys.argv[1]), Path(sys.argv[2])
    rows = pptx_pass(deck) + pdf_pass(pdf)
    if not rows:
        print("clean — no collisions, no edge bleed")
    for r in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"slide {r[0]:2d}  {r[1]:<14} {r[2]:>6}  {r[3]!r} | {r[4]!r}")
