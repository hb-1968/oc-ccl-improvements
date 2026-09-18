#!/usr/bin/env python3
"""Apply the fact-check corrections to the S16 deck and figures.

Recorded as a script so the edits are reviewable rather than hand-typed into a
binary. Each entry is one finding from the check of the deck against
results/s16.zip, the README and the source.
"""

from __future__ import annotations

missed: list[str] = []


def patch(path, pairs):
    s = open(path).read()
    for a, b in pairs:
        if a in s:
            s = s.replace(a, b, 1)
        else:
            missed.append(f"{path}: {a[:70]}")
    open(path, "w").write(s)


# ── 1 · the anchors resume S11's cells; they are not an independent re-run ──
patch("build_s16_deck.py", [
    ('''f1 = txt(s, "The two TTA cells are re-runs of banked ledger cells. The gate "
            "pins them to 75.18 and 63.86 within 0.5.",''',
     '''f1 = txt(s, "The two TTA cells resume the banked S11 cells, and the gate "
            "pins them to 75.18 and 63.86 within 0.5.",'''),
    ('''f2 = txt(s, "A protocol slip, TTA silently off or the wrong supports, cannot "
            "then masquerade as a result.",''',
     '''f2 = txt(s, "That certifies the wiring of the merge, not TTA itself. The "
            "two 2-shot cells are the only new measurement here.",'''),
    # slide 9 footer
    ('''f3 = txt(s, "Gate PASS 14 of 14. Both anchors re-landed on the ledger exactly.",''',
     '''f3 = txt(s, "Gate PASS 14 of 14. The anchors resume S11's cells; the "
            "2-shot cells are new.",'''),
])

# ── 2 · best-of-3 is the paper's full reading, not boundary tolerance alone ──
patch("build_s16_deck.py", [
    ('''title(s, "The sign survives the other metric")''',
     '''title(s, "The sign survives the other reading")'''),
    ('''foot = txt(s, "Under the paper's boundary-tolerant best-of-3 the headlines are "
              "-3.49 and +5.30, both still significant. Neither reading is "
              "doing the work.",''',
     '''foot = txt(s, "Under the paper's full reading, independent overlap plus "
              "boundary best-of-3, the headlines are -3.49 and +5.30, both "
              "still significant. That reading changes two things at once, and "
              "neither sign moves.",'''),
])

# ── 3 · beetle overshot its pre-registered share band ───────────────────────
patch("build_s16_deck.py", [
    ('''f1 = txt(s, "+5.84 (+9.2%) over the banked 63.20, which is 59% of the whole 1 "
            "to 5 shot lift, from one added specimen.",''',
     '''f1 = txt(s, "+5.84 (+9.2%) over the banked 63.20, unpaired, which is 58.6% "
            "of the whole 1 to 5 shot lift from one added specimen.",'''),
    ('''f2 = txt(s, "Concave, as pre-registered. The 5-shot level 73.16 stays above, "
            "so the draw-suspicion check stays quiet.",''',
     '''f2 = txt(s, "Concave as predicted, but past the top of the pre-registered "
            "35 to 55% share band. Beetle overshot where butterfly undershot.",'''),
])

# ── 4 · not the tightest cell in the project ────────────────────────────────
patch("build_s16_deck.py", [
    ('''f1 = txt(s, "std 2.10 to 0.30, a drop of 86%, and the tightest cell in the "
            "project. The level stayed 3.71 below TTA.",''',
     '''f1 = txt(s, "std 2.10 to 0.30, a drop of 86%, the tightest cell measured at "
            "1 or 2 shots. The level stayed 3.71 below TTA.",'''),
])

# ── 5 · cheapest TRAINING-FREE point above the paper's row ──────────────────
patch("build_s16_deck.py", [
    ('''f2 = txt(s, "On SST's own label axis the cheapest point above the paper's "
            "1-shot OC-CCL row of 65.2 is now 2 labels, no training, no TTA.",''',
     '''f2 = txt(s, "On SST's own label axis the cheapest training-free point above "
            "the paper's 1-shot OC-CCL row of 65.2 is now 2 labels.",'''),
])

# ── 6 · strongest per-label lever, not strongest lever ──────────────────────
patch("build_s16_deck.py", [
    ('''    ("A second label is the strongest single lever measured on beetle, and it "
     "needs no training at all."),''',
     '''    ("A second label is the strongest per-label lever measured on beetle, "
     "and it needs no training at all. 5-shot is still larger."),'''),
])

# ── 9 · the two arms do not cost the same at inference ──────────────────────
patch("build_s16_deck.py", [
    ('''foot = txt(s, "Panels are the project's own S3 render. Since S5c every "
              "headline arm has used the implicit version. The explicit one at "
              "the same inference cost had never been measured.",''',
     '''foot = txt(s, "Panels are the project's own S3 render. Since S5c every "
              "headline arm has used the implicit version. The explicit one had "
              "never been measured at any budget, and it is the cheaper of the "
              "two at inference.",'''),
])

# ── 10, 11, 12 · flat within a third of a point; name S14sam3; say unpaired ─
patch("build_s16_deck.py", [
    ('''f2 = txt(s, "Butterfly's ceiling did not move, 75.65 to 75.35. Same axis "
            "S14's arch arm moved by training, here moved by data.",''',
     '''f2 = txt(s, "Butterfly's ceiling is flat within a third of a point, 75.65 to "
            "75.35, with no paired test on the oracle reading. S14sam3's arch "
            "arm moved this axis by training; here it moved by data.",'''),
    ('''f2 = txt(s, "+1.18 of the measured +1.48 five-shot lift, which is 80% of it, "
            "arrives with the first added specimen.",''',
     '''f2 = txt(s, "+1.18 of the measured +1.48 five-shot lift, 80% of it, arrives "
            "with the first added specimen. Unpaired against the banked level.",'''),
])

# ── backup slide: the oracle rows are differences, not paired tests ─────────
patch("build_s16_deck.py", [
    ('''txt(s, "20 seeds per cell. CIs: butterfly [-4.48, -2.69], 2 of 20 seeds "
       "improved; beetle [+2.63, +7.77], 15 of 20.",''',
     '''txt(s, "20 seeds per cell. CIs on the plain rows: butterfly [-4.48, -2.69], "
       "2 of 20 seeds improved; beetle [+2.63, +7.77], 15 of 20. The oracle "
       "rows are differences, with no paired test run on that reading.",'''),
])

# ── figures: name the anchor as a resume, and the reading as two-factor ─────
patch("make_s16_figures.py", [
    ('''    tags = [["anchor: pinned to 75.18", "new"],
            ["anchor: pinned to 63.86", "new"]]''',
     '''    tags = [["resumed from S11, pinned", "new measurement"],
            ["resumed from S11, pinned", "new measurement"]]'''),
    ('''    ax.set_xlabel("headline under the plain metric (grey) and the paper's "
                  "best-of-3 (white)")''',
     '''    ax.set_xlabel("headline under the plain metric (grey) and the paper's "
                  "independent + best-of-3 reading (white)")'''),
])

print("missed:", len(missed))
for m in missed:
    print("   MISS:", m)
