#!/usr/bin/env python3
"""Figures for the S14-era talk deck.

Every number here is either (a) lifted live from a landed bundle in `results/`, or
(b) a constant transcribed from README.md with the section named beside it — and
where both exist, the script ASSERTS they agree before plotting. Nothing is drawn
that the record does not contain.

    python make_talk_figures.py --repo /path/oc-ccl-improvements --out figures/
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# ── palette: the reference deck is black on white, so the plots are too ──────
INK = "#000000"
MUTED = "#000000"
GRID = "#D6D6D6"
#: Three fills carry every distinction colour used to: solid black for the
#: subject, mid grey for a comparator, light grey for a band or a correlate.
COPPER = "#000000"
TEAL = "#000000"
CRIMSON = "#000000"          # markers stay black; bar fills carry the sign
HOLLOW = "#FFFFFF"           # a negative bar: white fill, black edge
SLATE = "#8C8C8C"
SLATE2 = "#C4C4C4"
PAPER_LINE = "#000000"

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["DejaVu Sans Mono"],
    "font.size": 16,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlesize": 17,
    "axes.linewidth": 1.4,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "patch.edgecolor": INK,
    "patch.linewidth": 1.4,
    "hatch.linewidth": 1.2,
})

REPO = Path("/sessions/wonderful-pensive-shannon/mnt/oc-ccl-improvements")
OUT = Path("figures")


def bundle(name: str):
    for d in (REPO / "results", REPO / "results" / "inbox"):
        p = d / name
        if p.exists():
            with zipfile.ZipFile(p) as z:
                return json.loads(z.read("record.json"))
    return None


def close(a, b, tol=0.03):
    assert abs(a - b) < tol, f"record disagrees with README: {a} vs {b}"


#: value labels sit on a white plate so gridlines never run through them.
PLATE = dict(facecolor="white", edgecolor="none", pad=1.5)
HATCH = "////"          # for pre-registered bands


def _edged(ax):
    """Give every patch a black edge, which is what makes a white fill legible."""
    for p in ax.patches:
        p.set_edgecolor(INK)
        p.set_linewidth(1.4)


def tidy(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    if xgrid:
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, lw=1.1)


def save(fig, name, tight=True):
    for ax in fig.get_axes():
        _edged(ax)
    OUT.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.18)
    else:                      # layered builds must be pixel-identical to stack
        fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("  wrote", name)


# ── F1 · the four reproduction gates ─────────────────────────────────────────
def f1_gates(layers=True):
    # README "Where the gates stand"
    rows = [
        ("Beetle 1-shot", 61.9, 3.7, 63.20, 3.89, True),
        ("Beetle 5-shot", 74.2, 2.9, 73.16, 2.31, True),
        ("Butterfly 1-shot", 81.0, 1.0, 70.30, 1.36, False),
        ("Butterfly 5-shot", 83.4, 0.4, 71.78, 0.15, False),
    ]
    for n_show in ([2, 4] if layers else [4]):
        fig, ax = plt.subplots(figsize=(12.8, 5.2))
        y = np.arange(len(rows))[::-1]
        for i, (lab, pm, ps, mm, ms, ok) in enumerate(rows):
            yy = y[i]
            if i >= n_show:
                continue
            col = TEAL if ok else CRIMSON
            ax.plot([pm, mm], [yy, yy], color=MUTED, lw=2.5, zorder=1)
            ax.errorbar(pm, yy, xerr=ps, fmt="o", ms=15, color=INK,
                        ecolor=INK, elinewidth=2.5, capsize=7, zorder=3)
            ax.errorbar(mm, yy, xerr=ms, fmt="D", ms=14, color=col,
                        ecolor=col, elinewidth=2.5, capsize=7, zorder=3)
            ax.text(95.5, yy, "reproduces" if ok else f"{mm - pm:+.1f}",
                    va="center", ha="right", color=col, fontsize=22,
                    fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=22)
        ax.set_xlim(50, 96)
        ax.set_ylim(-0.7, 3.9)
        ax.set_xticks([50, 55, 60, 65, 70, 75, 80, 85])
        ax.set_xlabel("mIoU  (20 seeded runs, ± run-to-run std)")
        ax.plot([], [], "o", color=INK, ms=13, label="paper")
        ax.plot([], [], "D", color=TEAL, ms=13, label="measured here")
        ax.legend(loc="lower left", frameon=False, fontsize=19, ncol=2,
                  bbox_to_anchor=(0.0, -0.42))
        tidy(ax)
        fig.subplots_adjust(left=0.30, right=0.99, top=0.97, bottom=0.30)
        save(fig, f"f1_gates_{n_show}.png", tight=False)


# ── F2 · what each technique bought on Butterfly Major ───────────────────────
def f2_major_ladder():
    # README S5/S5b/S5c/S6/S6b/S6c/S8d/S11
    bars = [
        ("uniform\n1-shot", 68.54, 3.15, "one"),
        ("placement\n1-shot", 70.30, 1.36, "one"),
        ("+ TTA\n1-shot", 75.18, 2.10, "one"),
        ("+ OC-CCL\n1-shot", 76.52, 2.13, "one"),
        ("5-shot", 71.78, 0.15, "five"),
        ("+ orient-div\n5-shot", 75.80, 0.68, "five"),
        ("+ TTA\n5-shot", 77.22, 0.15, "five"),
        ("+ OC-CCL\n5-shot", 78.07, 0.23, "five"),
    ]
    s11 = bundle("s11.zip")
    if s11:
        c = s11["datasets"]["butterfly_major"]["cells"]["pertrait_tta"]
        close(100 * c["occl_miou"], 76.52)
        close(100 * c["base_miou"], 75.18)
    s8d = bundle("s8d_butterfly_major.zip")
    if s8d:
        close(100 * s8d["cells"]["tta"]["occl_miou"], 78.07)
        close(100 * s8d["cells"]["tta"]["base_miou"], 77.22)

    for n_show in (3, 4, 8):
        fig, ax = plt.subplots(figsize=(13.2, 6.2))
        x = np.arange(len(bars))
        for i, (lab, v, s, shot) in enumerate(bars):
            if i >= n_show:
                continue
            col = SLATE if shot == "one" else TEAL
            ax.bar(i, v, 0.72, color=col)
            ax.errorbar(i, v, yerr=s, fmt="none", ecolor=INK, elinewidth=2,
                        capsize=6)
            ax.text(i, v + s + 0.9, f"{v:.1f}", ha="center", fontsize=20,
                    fontweight="bold", bbox=PLATE)
        ax.axhline(81.0, ls="--", lw=2.5, color=PAPER_LINE)
        ax.text(7.45, 81.4, "paper 1-shot 81.0", ha="right", color=PAPER_LINE,
                fontsize=19)
        ax.axhline(83.4, ls="--", lw=2.5, color=PAPER_LINE, alpha=0.55)
        ax.text(7.45, 83.8, "paper 5-shot 83.4", ha="right", color=PAPER_LINE,
                fontsize=19, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars], fontsize=17)
        ax.set_ylim(60, 87)
        ax.set_ylabel("mIoU")
        tidy(ax, xgrid=False)
        ax.grid(axis="y", color=GRID, lw=1.1)
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.17)
        save(fig, f"f2_major_ladder_{n_show}.png", tight=False)


# ── F3 · Beetle: curation is the whole 1-shot story ──────────────────────────
def f3_beetle_ladder():
    bars = [("uniform", 55.96, 7.90), ("canonical", 58.54, 5.77),
            ("balanced", 63.20, 3.89), ("5-shot", 73.16, 2.31),
            ("+ TTA", 75.20, 2.77)]
    for n_show in (3, 5):
        fig, ax = plt.subplots(figsize=(11.6, 6.2))
        for i, (lab, v, s) in enumerate(bars):
            if i >= n_show:
                continue
            ax.bar(i, v, 0.7, color=SLATE if i < 3 else TEAL)
            ax.errorbar(i, v, yerr=s, fmt="none", ecolor=INK, elinewidth=2,
                        capsize=7)
            ax.text(i, v + s + 0.9, f"{v:.1f}", ha="center", fontsize=20,
                    fontweight="bold", bbox=PLATE)
            ax.text(i, 46.2, f"±{s:.2f}", ha="center", fontsize=18,
                    color=COPPER, fontweight="bold", bbox=PLATE)
        ax.axhline(61.9, ls="--", lw=2.5, color=PAPER_LINE, xmax=0.775)
        ax.text(4.78, 61.9, "paper\n1-shot\n61.9", ha="left", va="center",
                color=PAPER_LINE, fontsize=17)
        ax.axhline(74.2, ls="--", lw=2.5, color=PAPER_LINE, alpha=0.55,
                   xmax=0.775)
        ax.text(4.78, 74.2, "paper\n5-shot\n74.2", ha="left", va="center",
                color=PAPER_LINE, fontsize=17, alpha=0.85)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=13)
        ax.set_xlim(-0.6, 6.0)
        ax.set_ylim(45, 82)
        ax.set_ylabel("mIoU")
        ax.set_title("± is the run-to-run spread over 20 seeded supports",
                     fontsize=19, color=COPPER, fontweight="bold", pad=14)
        tidy(ax, xgrid=False)
        ax.grid(axis="y", color=GRID, lw=1.1)
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.11, right=0.99, top=0.90, bottom=0.15)
        save(fig, f"f3_beetle_ladder_{n_show}.png", tight=False)


# ── F4 · cyrbia is two populations ───────────────────────────────────────────
def f4_cyrbia():
    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    ax.bar([0], [74.95], 0.62, color=TEAL)
    ax.bar([1], [0.48], 0.62, color=CRIMSON)
    ax.bar([2.35], [47.40], 0.62, color=SLATE)
    for x, v, lab in ((0, 74.95, "upright\n63% of test set"),
                      (1, 0.48, "pinned 180°\n37% of test set"),
                      (2.35, 47.40, "cyrbia, measured")):
        ax.text(x, v + 1.8, f"{v:.2f}", ha="center", fontsize=24,
                fontweight="bold", bbox=PLATE)
        ax.text(x, -6.5, lab, ha="center", fontsize=19, va="top")
    ax.text(1.62, 62, "0.63 × 74.95\n+ 0.37 × 0.48\n= 47.4", fontsize=20,
            ha="center", color=COPPER, fontweight="bold")
    ax.set_xlim(-0.6, 2.95)
    ax.set_ylim(0, 88)
    ax.set_xticks([])
    ax.set_ylabel("mIoU")
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "f4_cyrbia_split.png")


# ── F5 · query arrangement decides the score ─────────────────────────────────
def f5_displacement():
    q = [("7.7 – 23.1 px", 82.83), ("23.1 – 40.5 px", 75.08),
         ("40.5 – 70.5 px", 73.02), ("70.5 – 464.6 px", 50.83)]
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    cols = [INK, SLATE, SLATE2, HOLLOW]
    for i, (lab, v) in enumerate(q):
        ax.barh(3 - i, v, 0.66, color=cols[i])
        ax.text(v + 1.4, 3 - i, f"{v:.1f}", va="center", ha="left",
                fontsize=19, fontweight="bold", bbox=PLATE)
    ax.axvline(81.0, ls="--", lw=2.5, color=PAPER_LINE, zorder=5)
    ax.text(81.0, 3.72, "paper 81.0", color=PAPER_LINE, fontsize=16,
            va="bottom", ha="center", bbox=PLATE)
    ax.set_yticks(range(4))
    ax.set_yticklabels([q[3][0], q[2][0], q[1][0], q[0][0]], fontsize=19)
    ax.set_xlabel("mIoU of the query quartile")
    ax.set_xlim(0, 96)
    ax.set_ylabel("query displacement\nfrom the train median", fontsize=19)
    ax.set_title("corr(mIoU, query displacement) = -0.866, n = 500.  "
                 "dashed line: the target, 81.0",
                 fontsize=14, color=INK, fontweight="bold", pad=14)
    tidy(ax)
    save(fig, "f5_displacement.png")


# ── F6 · two rotation fixes, one defect ──────────────────────────────────────
def f6_rotation():
    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    labs = ["placement\nsupports", "orientation-diverse\nsupports"]
    plain = [71.78, 75.80]
    tta = [77.22, 76.58]
    x = np.arange(2)
    ax.bar(x - 0.2, plain, 0.38, color=SLATE, label="no TTA")
    ax.bar(x + 0.2, tta, 0.38, color=TEAL, label="+ TTA")
    for xi, v in zip(x - 0.2, plain):
        ax.text(xi, v + 0.5, f"{v:.2f}", ha="center", fontsize=21,
                fontweight="bold", bbox=PLATE)
    for xi, v in zip(x + 0.2, tta):
        ax.text(xi, v + 0.5, f"{v:.2f}", ha="center", fontsize=21,
                fontweight="bold", bbox=PLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=20)
    ax.set_ylim(66, 82)
    ax.set_ylabel("mIoU")
    ax.legend(frameon=False, fontsize=19, loc="upper left")
    ax.set_title("support-side +4.02   ·   query-side +5.44   ·   both +4.80"
                 "   ·   interaction −4.66",
                 fontsize=19, color=COPPER, fontweight="bold", pad=14)
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "f6_rotation_2x2.png")


# ── F7 · every OC-CCL delta measured ─────────────────────────────────────────
def f7_forest():
    R = []  # (label, delta, lo, hi, dataset)
    s8c = bundle("s8c_beetle.zip")
    s8d_b, s8d_m = bundle("s8d_beetle.zip"), bundle("s8d_butterfly_major.zip")
    s85, s10 = bundle("s8_5a.zip"), bundle("results/s10.zip") or bundle("s10.zip")
    s11, s12, s13 = bundle("s11.zip"), bundle("s12.zip"), bundle("s13.zip")

    def add(lab, d, lo, hi, ds):
        R.append((lab, d, lo, hi, ds))

    add("S8c   LoRA, 1-shot", -0.05, -0.15, 0.03, "beetle")
    if s8d_b:
        c = s8d_b["cells"]["tta"]
        add("S8d   5-shot + TTA", 100 * c["paired_delta"],
            100 * c["paired_ci95"][0], 100 * c["paired_ci95"][1], "beetle")
        close(100 * c["paired_delta"], -0.40)
    if s85:
        c = s85["datasets"]["beetle"]["cells"]["plain"]
        add("S8.5a   SST's full fine-tune", 100 * c["paired_delta"],
            100 * c["paired_ci95"][0], 100 * c["paired_ci95"][1], "beetle")
        close(100 * c["paired_delta"], -3.95, tol=0.05)
        c = s85["datasets"]["butterfly_major"]["cells"]["plain"]
        add("S8.5a   SST's full fine-tune", 100 * c["paired_delta"],
            100 * c["paired_ci95"][0], 100 * c["paired_ci95"][1], "butterfly")
    add("S8.5a2   SST's literal config", -0.42, -1.20, 0.44, "beetle")
    add("S10   the PAPER's method", 0.10, -0.16, 0.37, "beetle")
    add("S11   per-trait + TTA", 0.26, -0.32, 0.80, "beetle")
    add("S12   + SCNP", -0.09, -0.31, 0.09, "beetle")
    add("S13   + exclusivity CE", 0.25, -0.06, 0.61, "beetle")
    if s8d_m:
        c = s8d_m["cells"]["tta"]
        add("S8d   5-shot + TTA", 100 * c["paired_delta"],
            100 * c["paired_ci95"][0], 100 * c["paired_ci95"][1], "butterfly")
        close(100 * c["paired_delta"], 0.85)
    if s11:
        c = s11["datasets"]["butterfly_major"]["cells"]["pertrait_tta"]
        add("S11   per-trait + TTA", 100 * c["paired_delta"],
            100 * c["paired_ci95"][0], 100 * c["paired_ci95"][1], "butterfly")
    add("S13   margin fold", -3.62, -4.43, -2.89, "butterfly")

    beetle = [r for r in R if r[4] == "beetle"]
    butter = [r for r in R if r[4] == "butterfly"]
    order = beetle + [None] + butter          # None = spacer row for the group tag
    n_beetle = len(beetle)

    for n_show in (n_beetle, len(order)):
        fig, ax = plt.subplots(figsize=(13.4, 7.6))
        y = np.arange(len(order))[::-1]
        for i, row in enumerate(order):
            if i >= n_show or row is None:
                continue
            lab, d, lo, hi, ds = row
            col = TEAL if lo > 0 else (CRIMSON if hi < 0 else MUTED)
            ax.plot([lo, hi], [y[i], y[i]], color=col, lw=4, solid_capstyle="round")
            ax.plot(d, y[i], "o", ms=15, color=col)
        ax.axvline(0, color=INK, lw=2)
        ax.axvline(3.3, ls="--", lw=3, color=PAPER_LINE)
        ax.text(3.3, len(order) - 0.15, "paper: +3.3", ha="center",
                va="bottom", color=PAPER_LINE, fontsize=16, fontweight="bold",
                bbox=PLATE)
        ax.set_yticks(y)
        ax.set_yticklabels([("" if r is None else r[0]) for r in order],
                           fontsize=18)
        ax.set_xlim(-5.15, 4.0)
        ax.set_ylim(-0.8, len(order) - 0.1)
        ax.set_xlabel("paired ΔmIoU, OC-CCL − base  (95% CI over 20 seeds)")
        ax.text(-4.65, y[0] + 0.55, "BEETLE", fontsize=19, color=COPPER,
                fontweight="bold")
        if n_show > n_beetle:
            ax.text(-4.65, y[n_beetle] - 0.12, "BUTTERFLY MAJOR", fontsize=19,
                    color=COPPER, fontweight="bold")
        tidy(ax)
        fig.subplots_adjust(left=0.26, right=0.99, top=0.97, bottom=0.12)
        save(fig, f"f7_forest_{n_show}.png", tight=False)


# ── F8 · the ceiling that retired the story ──────────────────────────────────
def f8_oracle():
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.barh([1.0], [15.58], 0.5, color=MUTED)
    ax.barh([0.35], [2.65], 0.5, color=CRIMSON)
    ax.barh([-0.75], [1.13], 0.5, color=MUTED)
    ax.barh([-1.4], [0.38], 0.5, color=CRIMSON)
    for y, v in ((1.0, 15.58), (0.35, 2.65), (-0.75, 1.13), (-1.4, 0.38)):
        ax.text(v + 0.45, y, f"+{v:.2f}", va="center", fontsize=19,
                fontweight="bold", bbox=PLATE)
    ax.set_yticks([1.0, 0.35, -0.75, -1.4])
    ax.set_yticklabels(["silhouette − mIoU\n(a correlate)",
                        "oracle − argmax\n(the measured ceiling)",
                        "silhouette − mIoU",
                        "oracle − argmax"], fontsize=17)
    ax.set_xlim(0, 20.5)
    ax.set_xlabel("points of assignment headroom")
    ax.text(20.3, 1.38, "BEETLE", fontsize=20, color=COPPER, fontweight="bold",
            ha="right", va="center")
    ax.text(20.3, -0.37, "BUTTERFLY", fontsize=20, color=COPPER,
            fontweight="bold", ha="right", va="center")
    tidy(ax)
    save(fig, "f8_oracle.png")


# ── F9 · one metric, and every number locks together ─────────────────────────
def f9_metric():
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    pts = [("their base", 61.9, PAPER_LINE, 1),
           ("their OC-CCL", 65.2, PAPER_LINE, 1),
           ("our base, our metric", 63.20, TEAL, -1),
           ("our base, their metric", 65.20, TEAL, -1)]
    ax.plot([60.8, 66.4], [0, 0], color=MUTED, lw=2)
    for lab, v, col, side in pts:
        ax.plot([v, v], [0, 0.42 * side], color=col, lw=3)
        ax.plot(v, 0, "o", ms=16, color=col)
        ax.text(v, 0.52 * side, f"{v:.2f}" if v % 1 else f"{v:.1f}",
                ha="center", fontsize=26, fontweight="bold", color=col,
                va="bottom" if side > 0 else "top")
        ax.text(v, 0.95 * side, lab, ha="center", fontsize=18, color=INK,
                va="bottom" if side > 0 else "top")
    ax.annotate("", xy=(65.2, 1.42), xytext=(61.9, 1.42),
                arrowprops=dict(arrowstyle="<->", color=COPPER, lw=3))
    ax.text(63.55, 1.5, "their +3.3", ha="center", color=COPPER, fontsize=22,
            fontweight="bold")
    ax.set_xlim(60.8, 66.6)
    ax.set_ylim(-2.0, 2.1)
    ax.set_yticks([])
    ax.set_xlabel("Beetle mIoU, 1-shot")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "f9_metric.png")


# ── F10 · S14, pre-registered and pending ────────────────────────────────────
def f10_s14():
    rows = [
        ("P1   SAM 3 base  vs  SAM 2 base", -2, 2, 1, 8),
        ("P3   SeC off  vs  SAM 2 base", -2, 2, None, None),
        ("P4   SeC concept forced  vs  off", -0.5, 0.5, 0.5, 2),
        ("P5   + our loss stack  vs  own TTA base", -0.3, 0.3, 0.5, 2),
        ("P7   HEADLINE  vs  SAM 2 deployment stack", None, None, 2, 8),
    ]
    fig, ax = plt.subplots(figsize=(13.4, 6.2))
    y = np.arange(len(rows))[::-1]
    for i, (lab, plo, phi, olo, ohi) in enumerate(rows):
        if plo is not None:
            ax.barh(y[i], phi - plo, 0.52, left=plo, color="white", hatch=HATCH,
                    zorder=2)
        if olo is not None:
            ax.barh(y[i], ohi - olo, 0.52, left=olo, color="#DDDDDD", zorder=1)
        ax.plot(9.3, y[i], marker="o", ms=16, mfc="white", mec=MUTED, mew=2.5)
    ax.text(9.3, len(rows) - 0.42, "result", ha="center", fontsize=17,
            color=MUTED)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", hatch=HATCH, edgecolor=INK,
                             label="prior band"),
                       Patch(facecolor="#DDDDDD", edgecolor=INK,
                             label="optimist band")],
              frameon=False, fontsize=18, loc="lower left",
              bbox_to_anchor=(0.0, -0.30), ncol=2)
    ax.axvline(0, color=INK, lw=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=18)
    ax.set_xlim(-3.4, 10.2)
    ax.set_xlabel("paired ΔmIoU vs the named baseline  (bands frozen before the run)")
    tidy(ax)
    save(fig, "f10_s14.png")


# ── F11 · the failure, from the project's own render ─────────────────────────
def f11_overlay_panels():
    """Split the project's own S3 render into panels.

    The deck is otherwise black and white, but the label maps keep their colour:
    the hue is the trait identity, and collapsing it to grey destroys the one
    thing those panels exist to show. Photographs go through luminance, so
    colour on a slide always means "this encodes a trait".
    """
    src = REPO / "results" / "s3" / "overlay.png"
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).mean(axis=2)
    colmin = a.min(axis=0)
    ink_cols = colmin < 245
    runs, start = [], None
    for i, v in enumerate(ink_cols):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > 40:
                runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(ink_cols)))
    print("  overlay panel columns:", runs)
    names = ["p1_support", "p2_support_mask", "p3_query", "p4_prediction",
             "p5_ground_truth"]
    OUT.mkdir(parents=True, exist_ok=True)
    for (x0, x1), nm in zip(runs, names):
        pad = 8
        crop = im.crop((max(0, x0 - pad), 96, min(im.width, x1 + pad), im.height))
        labelled = ("mask" in nm or "prediction" in nm or "truth" in nm)
        # Label maps keep their colour: the hue IS the trait identity, and a
        # grey ramp cannot carry it at slide scale. Photographs go to luminance,
        # so colour on a slide always means "this encodes a trait".
        if not labelled:
            crop = _to_grey(crop, labelled=False)
        crop = _trim(crop)
        crop.save(OUT / f"f11_{nm}.png")
        print("  wrote", f"f11_{nm}.png", crop.size)


def _trim(im: "Image.Image") -> "Image.Image":
    """Drop the render's pale background rows so panels share a bottom edge."""
    a = np.asarray(im.convert("L"))
    rows = np.where(a.min(axis=1) < 235)[0]
    cols = np.where(a.min(axis=0) < 235)[0]
    if len(rows) and len(cols):
        im = im.crop((int(cols[0]), int(rows[0]),
                      int(cols[-1]) + 1, int(rows[-1]) + 1))
    return im


def _to_grey(im: "Image.Image", labelled: bool) -> "Image.Image":
    """Photographs: luminance. Label maps: one distinct grey per label."""
    arr = np.asarray(im).astype(np.float32)
    if not labelled:
        g = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        return Image.fromarray(g.astype(np.uint8)).convert("RGB")

    flat = arr.reshape(-1, 3).astype(np.uint8)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    cols = cols[np.argsort(-counts)][:12]              # the label palette
    lum = 0.299 * cols[:, 0] + 0.587 * cols[:, 1] + 0.114 * cols[:, 2]
    order = np.argsort(lum)
    levels = np.linspace(25, 235, len(cols)).astype(np.uint8)
    lut = {tuple(int(c) for c in cols[o]): int(levels[i])
           for i, o in enumerate(order)}
    out = np.zeros(arr.shape[:2], np.uint8)
    q = flat.reshape(arr.shape[:2], 3) if False else flat.reshape(*arr.shape[:2], 3)
    for rgb, lvl in lut.items():
        m = (q[..., 0] == rgb[0]) & (q[..., 1] == rgb[1]) & (q[..., 2] == rgb[2])
        out[m] = lvl
    unmapped = out == 0
    if unmapped.any():                                  # anti-aliased edges
        g = (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
             + 0.114 * arr[..., 2]).astype(np.uint8)
        out[unmapped] = g[unmapped]
    return Image.fromarray(out).convert("RGB")


# ── F12 · what TTA is worth, and what it costs ───────────────────────────────
def f12_tta():
    """Every measured TTA delta (README S5c, S6c, S8d, S11) + the cyrbia recovery."""
    #: (label, delta, the arm TTA was added to)
    arms = [("Butterfly\n5-shot", 5.44, 71.78), ("Butterfly\n1-shot", 4.88, 70.30),
            ("Beetle\n5-shot", 2.04, 73.16), ("Beetle\n1-shot", 0.66, 63.20)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.9),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    for i, (lab, d, base) in enumerate(arms):
        ax.barh(3 - i, d, 0.62, color=TEAL if d > 1 else SLATE)
        ax.text(d + 0.14, 3 - i, f"+{d:.2f}  (+{100 * d / base:.1f}%)",
                va="center", fontsize=16, fontweight="bold", bbox=PLATE)
    ax.set_yticks(range(4))
    ax.set_yticklabels([a[0] for a in arms][::-1], fontsize=18)
    ax.set_xlim(0, 9.4)
    ax.set_xlabel("ΔmIoU from TTA alone")
    tidy(ax)

    ax2.bar([0], [47.40], 0.55, color=CRIMSON)
    ax2.bar([1], [73.27], 0.55, color=TEAL)
    ax2.axhline(74.95, ls="--", lw=2.5, color=INK)
    ax2.text(1.44, 97, "74.95: what cyrbia's never-flipped\n"
                       "specimens score, i.e. the ceiling\n"
                       "for this intervention",
             ha="right", va="top", fontsize=12, color=INK)
    ax2.text(0, 48.8, "47.40", ha="center", fontsize=20, fontweight="bold")
    ax2.text(1, 68.5, "73.27", ha="center", fontsize=20, fontweight="bold",
             color="white")
    ax2.text(0.5, 40.0, "+25.87  (+54.6%)", ha="center", fontsize=16,
             fontweight="bold", bbox=PLATE)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["cyrbia\nbefore", "cyrbia\n+ TTA"], fontsize=18)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("mIoU")
    ax2.set_title("recovers 94% of the flip deficit", fontsize=18,
                  color=COPPER, fontweight="bold", pad=12)
    tidy(ax2, xgrid=False)
    ax2.grid(axis="y", color=GRID, lw=1.1)
    ax2.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "f12_tta.png")


# ── F13 · the ledger: what worked, what didn't ───────────────────────────────
def f13_ledger():
    #: (label, delta, the arm it was added to, dataset). The base is what makes
    #: the relative gain computable rather than typed.
    rows = [
        ("more shots (1 → 5)", 9.96, 63.20, "beetle"),
        ("support curation", 7.24, 55.96, "beetle"),
        ("orientation TTA", 5.44, 71.78, "butterfly"),
        ("support curation", 1.90, 68.54, "butterfly"),
        ("orientation-diverse supports", 4.02, 71.78, "butterfly"),
        ("per-trait OC-CCL", 1.33, 75.18, "butterfly"),
        ("more shots (1 → 5)", 1.48, 70.30, "butterfly"),
        ("per-trait OC-CCL", 0.26, 63.86, "beetle"),
        ("SCNP", -0.09, 63.20, "beetle"),
        ("merged OC-CCL", -0.40, 75.20, "beetle"),
        ("overlap penalty", -0.17, 64.12, "beetle"),
        ("both rotation fixes together", -4.66, 71.78, "butterfly"),
        ("full fine-tuning", -3.95, 63.20, "beetle"),
        ("confidence-margin fold", -3.62, 76.52, "butterfly"),
        ("SAM 3 backbone, raw", 2.14, 63.20, "beetle"),
        ("our loss stack on SAM 3", 1.59, 66.33, "beetle"),
        ("arch arm on SAM 3", 1.96, 67.92, "beetle"),
        ("SeC-4B swap, deployed", -3.02, 64.12, "beetle"),
        ("SeC concept injection", -6.23, 62.31, "beetle"),
    ]
    rows = sorted(rows, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(13.6, 8.6))
    y = np.arange(len(rows))
    for i, (lab, d, base, ds) in enumerate(rows):
        ax.barh(i, d, 0.66, color=INK if d > 0 else "white")
        ax.text(d + (0.18 if d > 0 else -0.18), i,
                f"{d:+.2f}  ({100 * d / base:+.1f}%)",
                va="center", ha="left" if d > 0 else "right", fontsize=15,
                fontweight="bold", bbox=PLATE)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}   ·  {ds}" for lab, d, base, ds in rows],
                       fontsize=16)
    ax.axvline(0, color=INK, lw=2)
    ax.set_xlim(-12.8, 14.5)
    ax.set_xlabel("ΔmIoU against the arm it was added to, and the same gain as a percentage of it")
    tidy(ax)
    save(fig, "f13_ledger.png")


# ── F14 · the rotated support, from the project's own render ─────────────────
def f14_rotated_panels():
    for src, dst in (("f11_p1_support.png", "f14_rot_support.png"),
                     ("f11_p2_support_mask.png", "f14_rot_mask.png")):
        im = Image.open(OUT / src)
        im.rotate(180).save(OUT / dst)
        print("  wrote", dst)


# ── F15 · shots: the same change, two datasets ───────────────────────────────
def f15_shots():
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    pairs = [("Beetle", 63.20, 73.16, TEAL), ("Butterfly Major", 70.30, 71.78,
                                              SLATE)]
    for i, (lab, one, five, col) in enumerate(pairs):
        ax.plot([0, 1], [one, five], "-o", lw=5, ms=17, color=col)
        ax.text(-0.06, one, f"{one:.2f}", ha="right", va="center", fontsize=21,
                fontweight="bold")
        ax.text(1.06, five, f"{five:.2f}", ha="left", va="center", fontsize=21,
                fontweight="bold")
        ax.text(0.5, (one + five) / 2 + (1.5 if i == 0 else -1.5),
                f"{five - one:+.2f}", ha="center", fontsize=19,
                fontweight="bold", color=INK, bbox=PLATE)
        ax.text(1.42, five, lab, va="center", fontsize=19, color=col,
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1-shot", "5-shot"], fontsize=21)
    ax.set_xlim(-0.35, 2.3)
    ax.set_ylim(60, 78)
    ax.set_ylabel("mIoU")
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "f15_shots.png")


# ── F16 · one prediction, four defensible scorings ───────────────────────────
def f16_readings():
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    vals = [("argmax\nover logits", 63.20, SLATE),
            ("per-trait\nbinarisation", 64.69, SLATE),
            ("+ 1-px boundary\ntolerance", 65.20, TEAL)]
    for i, (lab, v, col) in enumerate(vals):
        ax.bar(i, v, 0.62, color=col)
        ax.text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=24,
                fontweight="bold", bbox=PLATE)
    ax.set_xticks(range(3))
    ax.set_xticklabels([v[0] for v in vals], fontsize=17)
    ax.set_xlim(-0.75, 2.75)
    ax.set_ylim(60, 67.0)
    ax.set_ylabel("Beetle mIoU, same predictions")
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "f16_readings.png")


# ── F17 · why round-robin, not stratified random ─────────────────────────────
def f17_coverage():
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    for row, (lab, n, col) in enumerate([("stratified random", 9, MUTED),
                                         ("systematic round-robin", 12, TEAL)]):
        for k in range(12):
            filled = k < n
            ax.add_patch(plt.Rectangle((k * 1.0, -row * 1.3), 0.82, 0.9,
                                       facecolor=col if filled else "white",
                                       edgecolor=col if filled else GRID,
                                       lw=2.5))
        ax.text(12.4, -row * 1.3 + 0.45, f"{n} / 12 species used",
                va="center", fontsize=21, color=col, fontweight="bold")
        ax.text(-0.35, -row * 1.3 + 0.45, lab, va="center", ha="right",
                fontsize=19)
    ax.set_xlim(-5.0, 17.2)
    ax.set_ylim(-1.5, 1.1)
    ax.axis("off")
    ax.set_title("20 draws over 12 species: equal probability is not equal "
                 "representation", fontsize=19, color=COPPER, fontweight="bold",
                 pad=6)
    save(fig, "f17_coverage.png")


# ── F18 · SCNP, in both forms it was tried ───────────────────────────────────
def f18_scnp():
    """S11 (SCNP as the cycle's own loss) and S12 (SCNP added to the per-trait
    cycle, best-of-3 headline). Values from the landed s11/s12 bundles."""
    s11, s12 = bundle("s11.zip"), bundle("s12.zip")
    left = [("butterfly", 1.29, (1.18, 1.41)), ("beetle", -0.09, (-0.16, -0.02))]
    right = [("butterfly", 0.04, (0.02, 0.06)), ("beetle", -0.01, (-0.03, 0.01))]
    if s12:
        c = s12["datasets"]["butterfly_major"]["cells"]["ptscnp"]
        close(100 * c["paired_delta_best3"], 0.04, tol=0.02)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.2))
    for ax, rows, ttl, lim in (
            (a1, left, "on its own, against no fine-tune", 1.7),
            (a2, right, "added to the per-trait cycle", 0.42)):
        for i, (lab, d, ci) in enumerate(rows):
            col = TEAL if ci[0] > 0 else (CRIMSON if ci[1] < 0 else MUTED)
            ax.plot(ci, [i, i], color=col, lw=5, solid_capstyle="round")
            ax.plot(d, i, "o", ms=17, color=col)
            off = 0.10 * lim if d >= 0 else -0.10 * lim
            ax.text(d + off, i + 0.26, f"{d:+.2f}",
                    ha="left" if d >= 0 else "right", fontsize=17,
                    fontweight="bold", color=INK, bbox=PLATE)
        ax.axvline(0, color=INK, lw=2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=16)
        ax.set_ylim(-1.15, 1.6)
        ax.set_xlim(-lim, lim)
        ax.set_title(ttl, fontsize=19, color=INK, pad=14)
        tidy(ax)
    a2.axvspan(-0.3, 0.3, facecolor="#EBEBEB", edgecolor="none", zorder=0)
    a2.text(0, -0.72, "pre-registered no-effect band", ha="center",
            fontsize=13, color=INK, fontweight="bold", bbox=PLATE)
    a1.set_xlabel("ΔmIoU")
    a2.set_xlabel("ΔmIoU, best-of-3")
    fig.subplots_adjust(wspace=0.28)
    save(fig, "f18_scnp.png")


# ── F19 · SAM 3: the beetle ladder, and butterfly parity ─────────────────────
def f19_sam3():
    r = bundle("s14sam3.zip")
    b = r["datasets"]["beetle"]["cells"]
    m = r["datasets"]["butterfly_major"]["cells"]
    rungs = [("SAM 2\ncrown", 100 * b["sam2_pertrait_tta"]["miou"], SLATE, None),
             ("SAM 3\nbase", 100 * b["base"]["miou"], TEAL, 2.14),
             ("+ TTA", 100 * b["base_tta"]["miou"], TEAL, 0.99),
             ("+ loss\nstack", 100 * b["stack_tta"]["miou"], TEAL, 1.59),
             ("+ arch\narm", 100 * b["arch_tta"]["miou"], COPPER, 1.96)]
    close(rungs[0][1], 64.12)
    close(rungs[-1][1], 69.88, tol=0.05)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.0),
                                  gridspec_kw={"width_ratios": [1.75, 1]})
    for i, (lab, v, col, d) in enumerate(rungs):
        ax.bar(i, v, 0.68, color=col)
        ax.text(i, v + 0.55, f"{v:.2f}", ha="center", fontsize=17,
                fontweight="bold", bbox=PLATE)
        if d:
            prev = rungs[i - 1][1]
            ax.text(i, v + 1.35, f"{d:+.2f}*  ({100 * d / prev:+.1f}%)",
                    ha="center", fontsize=13, fontweight="bold", bbox=PLATE)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([r[0] for r in rungs], fontsize=18)
    ax.set_ylim(62, 73.2)
    ax.set_ylabel("mIoU")
    ax.set_title("BEETLE: the rungs compose",
                 fontsize=19, color=INK, pad=12)
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)

    pairs = [("SAM 2\ncrown", 100 * m["sam2_pertrait_tta"]["miou"], SLATE),
             ("SAM 3\n+ stack", 100 * m["stack_tta"]["miou"], TEAL),
             ("SAM 3\n+ arch", 100 * m["arch_tta"]["miou"], TEAL)]
    for i, (lab, v, col) in enumerate(pairs):
        ax2.bar(i, v, 0.6, color=col)
        ax2.text(i, v + 0.14, f"{v:.2f}", ha="center", fontsize=19,
                 fontweight="bold", bbox=PLATE)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([p[0] for p in pairs], fontsize=17)
    ax2.set_ylim(74, 78.8)
    ax2.set_title("BUTTERFLY: parity", fontsize=19, color=MUTED,
                  pad=12)
    tidy(ax2, xgrid=False)
    ax2.grid(axis="y", color=GRID, lw=1.1)
    ax2.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "f19_sam3.png")


# ── F20 · SeC-4B: the swap that does not pay ─────────────────────────────────
def f20_sec():
    r = bundle("s14sec.zip")
    rows = []
    for lab, key in (("the swap itself\n(base vs SAM 2)", "base_vs_sam2_base"),
                     ("concept injection\n(SeC's headline feature)",
                      "forced_vs_off"),
                     ("orientation TTA", "tta_vs_base"),
                     ("our loss stack", "stack_vs_base_tta"),
                     ("DEPLOYMENT\nvs the SAM 2 crown", "stack_vs_sam2_crown")):
        rows.append((lab,
                     r["datasets"]["butterfly_major"]["pairings"][key],
                     r["datasets"]["beetle"]["pairings"][key]))
    close(100 * rows[-1][1]["delta"], -2.79, tol=0.05)

    fig, ax = plt.subplots(figsize=(13.4, 6.4))
    y = np.arange(len(rows))[::-1]
    for i, (lab, bfly, beet) in enumerate(rows):
        for p, off, mk in ((bfly, 0.17, "o"), (beet, -0.17, "s")):
            d = 100 * p["delta"]
            lo, hi = (100 * c for c in p["ci95"])
            col = TEAL if lo > 0 else (CRIMSON if hi < 0 else MUTED)
            ax.plot([lo, hi], [y[i] + off] * 2, color=col, lw=4,
                    solid_capstyle="round")
            ax.plot(d, y[i] + off, mk, ms=14, color=col)
    ax.axvline(0, color=INK, lw=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=18)
    ax.set_xlim(-8.2, 6.5)
    ax.set_xlabel("paired ΔmIoU  (circle = butterfly, square = beetle)")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    tidy(ax)
    save(fig, "f20_sec.png")


# ── F21 · the ceiling itself moved ───────────────────────────────────────────
def f21_ceiling():
    r = bundle("s14sam3.zip")
    b = r["datasets"]["beetle"]["cells"]
    cols = [("SAM 2 crown", 100 * b["sam2_pertrait_tta"]["miou"],
             100 * b["sam2_pertrait_tta"]["miou_oracle"], SLATE),
            ("SAM 3 + arch", 100 * b["arch_tta"]["miou"],
             100 * b["arch_tta"]["miou_oracle"], COPPER)]
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    for i, (lab, score, oracle, col) in enumerate(cols):
        ax.barh(i, score, 0.5, color=col)
        ax.plot([oracle, oracle], [i - 0.32, i + 0.32], color=INK, lw=4)
        ax.text(score - 0.4, i, f"{score:.2f}", va="center", ha="right",
                fontsize=22, fontweight="bold", color="white")
        ax.text(oracle + 0.25, i, f"ceiling {oracle:.2f}", va="center",
                fontsize=19, color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([c[0] for c in cols], fontsize=20)
    ax.set_xlim(60, 76)
    ax.set_ylim(-0.65, 1.75)
    ax.set_xlabel("Beetle mIoU, 1-shot")
    ax.annotate("", xy=(71.08, 1.52), xytext=(66.76, 1.52),
                arrowprops=dict(arrowstyle="-|>", color=COPPER, lw=3))
    ax.text(68.9, 1.62, "the ceiling moved +4.3", ha="center", fontsize=19,
            color=COPPER, fontweight="bold")
    tidy(ax)
    save(fig, "f21_ceiling.png")


# ── F22 · S15, the 5-shot projection ─────────────────────────────────────────
def f22_projection():
    """Pre-registered bounds for 5-shot on the champion — README S15 brief."""
    fig, ax = plt.subplots(figsize=(12.8, 5.0))
    rows = [("Beetle, 1-shot today", 69.88, None, None, COPPER),
            ("Beetle, 5-shot projected", 78.7, 76.8, 81.3, TEAL),
            ("Butterfly, 1-shot today", 76.64, None, None, COPPER),
            ("Butterfly, 5-shot projected", 78.75, 78.0, 79.5, TEAL)]
    y = np.arange(len(rows))[::-1]
    for i, (lab, c, lo, hi, col) in enumerate(rows):
        if lo is not None:
            ax.barh(y[i], hi - lo, 0.42, left=lo, facecolor="white", hatch=HATCH)
            ax.plot(c, y[i], "D", ms=15, color=col)
            ax.text(c, y[i] + 0.34, f"≈ {c:.1f}", ha="center", fontsize=19,
                    color=col, fontweight="bold")
        else:
            ax.plot(c, y[i], "o", ms=15, color=col)
            ax.text(c, y[i] + 0.34, f"{c:.2f}", ha="center", fontsize=19,
                    color=col, fontweight="bold")
    ax.axvline(75.6, ls="--", lw=2.5, color=PAPER_LINE)
    ax.text(75.2, -0.62, "the paper's best beetle row  75.6", ha="right",
            color=PAPER_LINE, fontsize=17)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=18)
    ax.set_xlim(66, 84)
    ax.set_ylim(-0.9, 3.6)
    ax.set_xlabel("mIoU")
    tidy(ax)
    save(fig, "f22_projection.png")


# ── F23 · one image, scored ──────────────────────────────────────────────────
def f23_metric():
    """A real scored query, lifted from S4c's per_query.csv.

    beetle impunctatus/beetle_1166_231, run 0. Trait ids 1-5 are head, pronotum,
    elytra, antenna, legs.
    """
    import csv
    import io

    names = {1: "head", 2: "pronotum", 3: "elytra", 4: "antenna", 5: "legs"}
    per = {}
    for d in (REPO / "results", REPO / "results" / "inbox"):
        f = d / "s4c.zip"
        if not f.exists():
            continue
        with zipfile.ZipFile(f) as z:
            for r in csv.DictReader(io.TextIOWrapper(z.open("per_query.csv"))):
                if r["run_idx"] == "0" and r["query_id"] == "impunctatus/beetle_1166_231":
                    per[int(r["trait_id"])] = float(r["iou"])
        break
    assert len(per) == 5, f"expected 5 traits, got {sorted(per)}"
    mean = sum(per.values()) / len(per)
    close(100 * mean, 65.45, tol=0.2)

    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    order = [3, 2, 1, 5, 4]                       # largest trait at the top
    for i, t in enumerate(order):
        v = per[t]
        col = CRIMSON if v < 0.05 else (TEAL if v > 0.7 else SLATE)
        ax.barh(i, v, 0.62, color=col)
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=21,
                fontweight="bold", bbox=PLATE)
    ax.axvline(mean, ls="--", lw=3, color=COPPER)
    ax.text(mean + 0.015, 4.62, f"image mIoU = {mean:.3f}", color=COPPER,
            fontsize=20, fontweight="bold")
    ax.set_yticks(range(5))
    ax.set_yticklabels([names[t] for t in order], fontsize=20)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.6, 5.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("IoU against this image's own ground truth")
    tidy(ax)
    save(fig, "f23_metric.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    REPO = Path(args.repo)
    OUT = Path(args.out)
    f1_gates()
    f2_major_ladder()
    f3_beetle_ladder()
    f4_cyrbia()
    f5_displacement()
    f6_rotation()
    f7_forest()
    f8_oracle()
    f9_metric()
    f10_s14()
    f11_overlay_panels()
    f12_tta()
    f13_ledger()
    f14_rotated_panels()
    f15_shots()
    f16_readings()
    f17_coverage()
    f18_scnp()
    f19_sam3()
    f20_sec()
    f21_ceiling()
    f22_projection()
    f23_metric()
    print("done")
