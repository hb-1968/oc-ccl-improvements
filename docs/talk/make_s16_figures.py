#!/usr/bin/env python3
"""Figures for the S16 talk: which second reference pays.

Same discipline as make_talk_figures.py. Every number is either lifted live
from results/s16.zip (or another landed bundle), or transcribed from a named
README row and ASSERTED against the bundle before anything is drawn.

    python make_s16_figures.py --repo /path/oc-ccl-improvements --out figures_s16/
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

# ── palette: black on white, as the deck is ─────────────────────────────────
INK = "#000000"
MUTED = "#000000"
GRID = "#D6D6D6"
COPPER = "#000000"
TEAL = "#000000"
CRIMSON = "#000000"
HOLLOW = "#FFFFFF"           # a negative bar: white fill, black edge
SLATE = "#8C8C8C"
SLATE2 = "#C4C4C4"

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
OUT = Path("figures_s16")

PLATE = dict(facecolor="white", edgecolor="none", pad=1.5)
HATCH = "////"


def bundle(name: str):
    for d in (REPO / "results", REPO / "results" / "inbox"):
        p = d / name
        if p.exists():
            with zipfile.ZipFile(p) as z:
                return json.loads(z.read("record.json"))
    raise SystemExit(f"missing bundle: {name}")


def close(a, b, tol=0.03):
    assert abs(a - b) < tol, f"record disagrees with README: {a} vs {b}"


def _edged(ax):
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
    else:                      # layered builds must stack pixel-for-pixel
        fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("  wrote", name)


# ── the record, read once ───────────────────────────────────────────────────
S16 = bundle("s16.zip")
BF = S16["datasets"]["butterfly_major"]
BE = S16["datasets"]["beetle"]


def cell(d, name, key="miou"):
    v = d["cells"][name][key]
    return 100 * v if key in ("miou", "std", "miou_best3", "miou_oracle",
                              "miou_independent", "iou_binary") else v


def pair(d, key="delta"):
    p = d["pairings"]["second_shot_vs_tta"]
    if key in ("delta", "delta_best3", "paired_std"):
        return 100 * p[key]
    if key in ("ci95", "ci95_best3"):
        return [100 * x for x in p[key]]
    return p[key]


# banked levels these figures lean on, each with the README row that holds it
BANKED = {
    "beetle_1shot": 63.20,        # "Where the gates stand" · S4 beetle 1-shot
    "beetle_5shot": 73.16,        # "Where the gates stand" · S6 beetle 5-shot
    "beetle_tta": 63.86,          # S11 ledger, and the S16 anchor
    "bf_1shot": 70.30,            # "Where the gates stand" · S5 Major 1-shot
    "bf_5shot": 71.78,            # "Where the gates stand" · S6 Major 5-shot
    "bf_tta": 75.18,              # S5c, and the S16 anchor
    "beetle_champion": 69.88,     # S14sam3 arch_tta deployment row
    "paper_occl_beetle": 65.2,    # paper table, 1-shot + OC-CCL
}

# the two anchors are pinned to the ledger by the gate; check that here too
close(cell(BF, "s1_tta"), BANKED["bf_tta"], tol=0.5)
close(cell(BE, "s1_tta"), BANKED["beetle_tta"], tol=0.5)


def pct(delta, base):
    return f"({100 * delta / base:+.1f}%)"


# ── G1 · the two second references, from the project's own render ───────────
def g1_panels():
    """Split S3's overlay into panels; rotate the support for the implicit arm.

    Photographs go to luminance. The label map keeps its colour, because the
    hue is the trait identity and grey cannot carry it at slide scale.
    """
    src = REPO / "results" / "s3" / "overlay.png"
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).mean(axis=2)
    ink_cols = a.min(axis=0) < 245
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
    names = ["support", "support_mask", "query", "prediction", "ground_truth"]
    assert len(runs) == 5, f"expected 5 panels, got {len(runs)}"
    OUT.mkdir(parents=True, exist_ok=True)
    for (x0, x1), nm in zip(runs, names):
        pad = 8
        crop = im.crop((max(0, x0 - pad), 96, min(im.width, x1 + pad), im.height))
        if nm in ("support", "query"):
            crop = _to_grey(crop)
        crop = _trim(crop)
        crop.save(OUT / f"g1_{nm}.png")
        print("  wrote", f"g1_{nm}.png", crop.size)
    for s, d in (("g1_support.png", "g1_support_rot.png"),
                 ("g1_support_mask.png", "g1_support_mask_rot.png")):
        Image.open(OUT / s).rotate(180).save(OUT / d)
        print("  wrote", d)


def _trim(im: "Image.Image") -> "Image.Image":
    a = np.asarray(im.convert("L"))
    rows = np.where(a.min(axis=1) < 235)[0]
    cols = np.where(a.min(axis=0) < 235)[0]
    if len(rows) and len(cols):
        im = im.crop((int(cols[0]), int(rows[0]),
                      int(cols[-1]) + 1, int(rows[-1]) + 1))
    return im


def _to_grey(im: "Image.Image") -> "Image.Image":
    arr = np.asarray(im).astype(np.float32)
    g = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return Image.fromarray(g.astype(np.uint8)).convert("RGB")


# ── G2 · the shot ladder, with shot 2 filled in ─────────────────────────────
def g2_ladder():
    """1-shot and 5-shot were banked; 2-shot is what S16 measured."""
    b2, f2 = cell(BE, "s2"), cell(BF, "s2")
    for n_show in (1, 2):
        fig, ax = plt.subplots(figsize=(11.8, 5.6))
        fig.subplots_adjust(left=0.10, right=0.80, top=0.95, bottom=0.16)
        series = [
            # label, banked ys, the new 2-shot value, colour, label offsets
            ("beetle", [BANKED["beetle_1shot"], BANKED["beetle_5shot"]], b2,
             INK, (-1.5, 1.6), -1.7),
            ("butterfly Major", [BANKED["bf_1shot"], BANKED["bf_5shot"]], f2,
             SLATE, (1.3, -1.6), 1.3),
        ]
        for lab, ys, y2, col, dy_end, dy_new in series:
            if n_show == 1:
                ax.plot([1, 5], ys, "--o", lw=2.5, ms=15, color=col,
                        markerfacecolor=HOLLOW, markeredgewidth=2.2)
            else:
                ax.plot([1, 2, 5], [ys[0], y2, ys[1]], "-o", lw=4, ms=15,
                        color=col)
                ax.plot([2], [y2], "o", ms=22, color=col)
                ax.text(2, y2 + dy_new, f"{y2:.2f}", ha="center", fontsize=20,
                        fontweight="bold", bbox=PLATE)
            for x, y, dy in zip([1, 5], ys, dy_end):
                ax.text(x, y + dy, f"{y:.2f}", ha="center", fontsize=17,
                        color=col, bbox=PLATE)
            ax.text(5.25, ys[1] + (0.55 if col == INK else -0.55), lab,
                    va="center", fontsize=18, color=col, fontweight="bold")
        ax.set_xticks([1, 2, 5])
        ax.set_xticklabels(["1 shot", "2 shots", "5 shots"], fontsize=19)
        ax.set_xlim(0.6, 5.2)
        ax.set_ylim(60, 78)
        ax.set_ylabel("mIoU")
        tidy(ax, xgrid=False)
        ax.grid(axis="y", color=GRID, lw=1.1)
        ax.set_axisbelow(True)
        save(fig, f"g2_ladder_{n_show}.png", tight=False)


# ── G3 · the mirror image: both headlines against their bands ───────────────
def g3_mirror():
    rows = [
        # label, band lo, band hi, delta, ci lo, ci hi, seeds improved, base
        ("butterfly\nMajor", -4.6, -3.9, pair(BF), *pair(BF, "ci95"),
         pair(BF, "n_seeds_improved"), cell(BF, "s1_tta")),
        ("beetle", 2.0, 5.3, pair(BE), *pair(BE, "ci95"),
         pair(BE, "n_seeds_improved"), cell(BE, "s1_tta")),
    ]
    close(rows[0][3], -3.71, tol=0.02)
    close(rows[1][3], 5.17, tol=0.02)
    for n_show in (1, 2):
        fig, ax = plt.subplots(figsize=(12.4, 5.2))
        fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.20)
        for i, (lab, lo, hi, d, cl, ch, imp, base) in enumerate(rows):
            y = 1 - i
            ax.barh(y, hi - lo, 0.52, left=lo, color=HOLLOW, hatch=HATCH,
                    edgecolor=INK, zorder=1)
            if n_show == 2:
                ax.errorbar(d, y, xerr=[[d - cl], [ch - d]], fmt="D", ms=17,
                            color=INK, ecolor=INK, elinewidth=3, capsize=8,
                            zorder=3)
                ax.text(d, y + (0.40 if i == 0 else -0.44),
                        f"{d:+.2f} {pct(d, base)}   {imp}/20 seeds",
                        ha="center", fontsize=18, fontweight="bold", bbox=PLATE)
        ax.axvline(0, color=INK, lw=2.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([rows[1][0], rows[0][0]], fontsize=19)
        ax.set_ylim(-0.75, 1.75)
        ax.set_xlim(-6.2, 8.6)
        ax.set_xlabel("2-shot plain  minus  1-shot + TTA   (mIoU)")
        ax.text(-5.9, 1.62, "implicit shot wins", fontsize=16, style="italic")
        ax.text(8.3, 1.62, "explicit shot wins", fontsize=16, style="italic",
                ha="right")
        tidy(ax)
        save(fig, f"g3_mirror_{n_show}.png", tight=False)


# ── G4 · what the second specimen did to the spread ─────────────────────────
def g4_variance():
    rows = [
        ("butterfly  1-shot + TTA", cell(BF, "s1_tta", "std")),
        ("butterfly  2-shot plain", cell(BF, "s2", "std")),
        ("beetle  1-shot + TTA", cell(BE, "s1_tta", "std")),
        ("beetle  2-shot plain", cell(BE, "s2", "std")),
    ]
    close(rows[0][1], 2.10, tol=0.02)
    close(rows[1][1], 0.30, tol=0.02)
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    for i, (lab, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, 0.6, color=(SLATE if i % 2 else HOLLOW))
        ax.text(v + 0.09, y, f"{v:.2f}", va="center", fontsize=20,
                fontweight="bold", bbox=PLATE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=16)
    ax.set_xlim(0, 5.4)
    ax.set_xlabel("run-to-run std over 20 seeds (mIoU)")
    tidy(ax)
    save(fig, "g4_variance.png")


# ── G5 · the ceiling: what the model even proposes ──────────────────────────
def g5_oracle():
    rows = [
        ("beetle\n1-shot + TTA", cell(BE, "s1_tta"), cell(BE, "s1_tta", "miou_oracle")),
        ("beetle\n2-shot plain", cell(BE, "s2"), cell(BE, "s2", "miou_oracle")),
        ("butterfly\n1-shot + TTA", cell(BF, "s1_tta"), cell(BF, "s1_tta", "miou_oracle")),
        ("butterfly\n2-shot plain", cell(BF, "s2"), cell(BF, "s2", "miou_oracle")),
    ]
    close(rows[0][2], 66.93, tol=0.05)
    close(rows[1][2], 71.73, tol=0.05)
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    for i, (lab, got, ceil) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, ceil, 0.58, color=HOLLOW, hatch=HATCH, edgecolor=INK,
                zorder=1)
        ax.barh(y, got, 0.58, color=SLATE, zorder=2)
        ax.text(got - 0.6, y, f"{got:.2f}", va="center", ha="right",
                fontsize=17, fontweight="bold", color="white", zorder=3)
        ax.text(ceil + 0.5, y, f"ceiling {ceil:.2f}", va="center", fontsize=17,
                fontweight="bold", bbox=PLATE)
    ax.annotate("", xy=(rows[1][2], 2.42), xytext=(rows[0][2], 2.42),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=INK))
    d = rows[1][2] - rows[0][2]
    ax.text((rows[0][2] + rows[1][2]) / 2, 2.56,
            f"{d:+.2f} {pct(d, rows[0][2])}", ha="center", fontsize=17,
            fontweight="bold", bbox=PLATE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=15)
    ax.set_xlim(0, 95)
    ax.set_ylim(-0.7, 3.75)
    ax.set_xlabel("mIoU: achieved (solid) inside the oracle ceiling (hatched)")
    tidy(ax)
    save(fig, "g5_oracle.png")


# ── G6 · TTA is an orientation shot, not a shot ─────────────────────────────
def g6_orientation():
    """How often the rotated support is kept, against what it is worth."""
    fires = [100 * BF["cells"]["s1_tta"]["tta_rotated_frac"],
             100 * BE["cells"]["s1_tta"]["tta_rotated_frac"]]
    close(fires[0], 8.06, tol=0.05)
    close(fires[1], 29.67, tol=0.05)
    worth = [BANKED["bf_tta"] - BANKED["bf_1shot"],      # S5c: +4.88
             BANKED["beetle_tta"] - BANKED["beetle_1shot"]]   # S11: +0.66
    close(worth[0], 4.88, tol=0.02)
    close(worth[1], 0.66, tol=0.02)
    labs = ["butterfly Major", "beetle"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    fig.subplots_adjust(wspace=0.42, left=0.09, right=0.97, bottom=0.16)
    for ax, vals, ttl, xmax, fmt in (
            (axes[0], fires, "episodes that keep the rotated support", 42,
             "{:.1f}%"),
            (axes[1], worth, "what TTA is worth at 1 shot (mIoU)", 6.6,
             "{:+.2f}")):
        for i, v in enumerate(vals):
            ax.barh(1 - i, v, 0.5, color=(HOLLOW if i == 0 else SLATE))
            ax.text(v + xmax * 0.03, 1 - i, fmt.format(v), va="center",
                    fontsize=19, fontweight="bold", bbox=PLATE)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labs[::-1], fontsize=16)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(ttl, fontsize=15)
        tidy(ax)
    save(fig, "g6_orientation.png")


# ── G7 · the label-budget axis ──────────────────────────────────────────────
def g7_labels():
    """Beetle mIoU against the thing SST actually spends: labelled masks."""
    pts = [
        (1, BANKED["paper_occl_beetle"], "paper 1-shot + OC-CCL", "o", HOLLOW),
        (1, BANKED["beetle_1shot"], "ours, plain", "o", HOLLOW),
        (1, cell(BE, "s1_tta"), "ours, + TTA", "o", SLATE),
        (1, BANKED["beetle_champion"], "SAM 3 + adapter + TTA", "s", INK),
        (2, cell(BE, "s2"), "2 shots, stock SAM 2", "D", INK),
        (5, BANKED["beetle_5shot"], "5 shots, stock SAM 2", "o", SLATE),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    fig.subplots_adjust(left=0.09, right=0.72, top=0.95, bottom=0.16)
    ax.axhline(cell(BE, "s2"), ls="--", lw=2, color=INK, zorder=1)
    for x, y, lab, mk, fc in pts:
        ax.plot(x, y, mk, ms=17, color=INK, markerfacecolor=fc,
                markeredgewidth=2.2, zorder=3)
        left = x < 5
        ax.text(x + (0.14 if left else -0.16), y,
                f"{y:.2f}   {lab}", va="center", fontsize=15,
                ha="left" if left else "right", bbox=PLATE, zorder=4)
    ax.set_xticks([1, 2, 5])
    ax.set_xticklabels(["1 label", "2 labels", "5 labels"], fontsize=18)
    ax.set_xlim(0.8, 5.3)
    ax.set_ylim(62, 75)
    ax.set_ylabel("beetle mIoU")
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "g7_labels.png")


# ── G8 · why the row can be paired at all ───────────────────────────────────
def g8_nesting():
    """One seeded permutation; the 1-shot draw is the prefix of the 2-shot draw."""
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.15, 4.4)
    ax.axis("off")
    ids = [f"perm[{k}]" for k in range(5)]
    ax.text(0.1, 3.9, "run i's seeded permutation of the canonical pool",
            fontsize=16)
    for k, sid in enumerate(ids):
        x = 0.1 + k * 2.1
        ax.add_patch(plt.Rectangle((x, 3.0), 1.9, 0.66, fill=False, lw=2.0))
        ax.text(x + 0.95, 3.33, sid, ha="center", va="center", fontsize=15)
    for row, (lab, take) in enumerate((("1-shot + TTA draw", 1),
                                       ("2-shot plain draw", 2))):
        y = 1.75 - row * 1.15
        ax.text(0.1 + take * 2.1 + 0.25, y + 0.33, lab, fontsize=16,
                fontweight="bold", va="center")
        for k, sid in enumerate(ids[:take]):
            x = 0.1 + k * 2.1
            ax.add_patch(plt.Rectangle((x, y), 1.9, 0.66, fill=True,
                                       facecolor=SLATE2, lw=2.0,
                                       edgecolor=INK))
            ax.text(x + 0.95, y + 0.33, sid, ha="center", va="center",
                    fontsize=15)
    ax.annotate("", xy=(1.05, 2.45), xytext=(1.05, 2.98),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=INK))
    ax.annotate("", xy=(1.05, 1.30), xytext=(1.05, 1.72),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=INK))
    nb, nf = (pair(BE, "nested_shots")["runs_checked"],
              pair(BF, "nested_shots")["runs_checked"])
    ax.text(0.1, -0.55,
            f"prefix verified per (group, run):  "
            f"{nf} butterfly pairs, {nb} beetle pairs", fontsize=15,
            va="center",
            bbox=dict(facecolor="white", edgecolor=INK, lw=1.6, pad=8))
    save(fig, "g8_nesting.png")


# ── G9 · the headline under two metrics ─────────────────────────────────────
def g9_readings():
    rows = [
        ("butterfly Major", pair(BF), pair(BF, "delta_best3")),
        ("beetle", pair(BE), pair(BE, "delta_best3")),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    w = 0.34
    for i, (lab, plain, best3) in enumerate(rows):
        y = 1 - i
        ax.barh(y + w / 1.7, plain, w, color=SLATE)
        ax.barh(y - w / 1.7, best3, w, color=HOLLOW)
        for v, yy in ((plain, y + w / 1.7), (best3, y - w / 1.7)):
            off = 0.28 if v > 0 else -0.28
            ax.text(v + off, yy, f"{v:+.2f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=18,
                    fontweight="bold", bbox=PLATE)
    ax.axvline(0, color=INK, lw=2.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([rows[1][0], rows[0][0]], fontsize=17)
    ax.set_ylim(-0.8, 1.8)
    ax.set_xlim(-6.0, 7.6)
    ax.set_xlabel("headline under the plain metric (grey) and the paper's "
                  "independent + best-of-3 reading (white)")
    tidy(ax)
    save(fig, "g9_readings.png")


# ── G10 · what one mIoU number is, on one real query ────────────────────────
def g10_metric():
    """A real scored query, lifted from S4c's per_query.csv (beetle run 0)."""
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
                if (r["run_idx"] == "0"
                        and r["query_id"] == "impunctatus/beetle_1166_231"):
                    per[int(r["trait_id"])] = float(r["iou"])
        break
    assert len(per) == 5, f"expected 5 traits, got {sorted(per)}"
    mean = sum(per.values()) / len(per)
    close(100 * mean, 65.45, tol=0.2)

    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    order = [3, 2, 1, 5, 4]
    for i, t in enumerate(order):
        v = per[t]
        ax.barh(i, v, 0.62, color=(HOLLOW if v < 0.05 else SLATE))
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=20,
                fontweight="bold", bbox=PLATE)
    ax.axvline(mean, ls="--", lw=3, color=INK)
    ax.text(mean + 0.015, 4.62, f"image mIoU = {mean:.3f}", fontsize=19,
            fontweight="bold")
    ax.set_yticks(range(5))
    ax.set_yticklabels([names[t] for t in order], fontsize=19)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.6, 5.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("IoU against this image's own ground truth")
    tidy(ax)
    save(fig, "g10_metric.png")


# ── G11 · the design: four runs ─────────────────────────────────────────────
def g11_design():
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    cols = ["1 shot + TTA", "2 shots, plain"]
    rows = ["butterfly Major", "beetle"]
    vals = [[cell(BF, "s1_tta"), cell(BF, "s2")],
            [cell(BE, "s1_tta"), cell(BE, "s2")]]
    tags = [["resumed from S11, pinned", "new measurement"],
            ["resumed from S11, pinned", "new measurement"]]
    for j, c in enumerate(cols):
        ax.text(4.75 + j * 3.7, 4.25, c, fontsize=17, fontweight="bold",
                ha="center")
    for i, r in enumerate(rows):
        y = 2.35 - i * 1.75
        ax.text(0.1, y + 0.55, r, fontsize=16, fontweight="bold")
        for j in range(2):
            x = 3.1 + j * 3.7
            ax.add_patch(plt.Rectangle((x, y), 3.3, 1.15, fill=False, lw=2.2))
            ax.text(x + 1.65, y + 0.72, f"{vals[i][j]:.2f}", ha="center",
                    fontsize=22, fontweight="bold")
            ax.text(x + 1.65, y + 0.26, tags[i][j], ha="center", fontsize=11.5)
    ax.text(0.1, 0.15, "base SAM 2, no adapters, 20 seeds each, "
                       "canonical protocol per dataset", fontsize=14)
    save(fig, "g11_design.png")


# ── G12 · where the 1 → 5 shot lift is actually earned ──────────────────────
def g12_concave():
    """Beetle: how much of the measured +9.96 arrives with the first added shot."""
    one, five = BANKED["beetle_1shot"], BANKED["beetle_5shot"]
    two = cell(BE, "s2")
    total = five - one
    first, rest = two - one, five - two
    close(total, 9.96, tol=0.02)          # S6: 5-shot > 1-shot, beetle +9.96
    fig, ax = plt.subplots(figsize=(12.2, 4.2))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.28)
    ax.barh(0, first, 0.55, color=SLATE)
    ax.barh(0, rest, 0.55, left=first, color=HOLLOW, hatch=HATCH,
            edgecolor=INK)
    ax.text(first / 2, 0, f"shot 2\n{first:+.2f}  ({100 * first / total:.1f}%)",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="white")
    ax.text(first + rest / 2, 0,
            f"shots 3 to 5\n{rest:+.2f}  ({100 * rest / total:.1f}%)",
            ha="center", va="center", fontsize=18, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=4))
    for x, lab in ((0, f"1 shot\n{one:.2f}"), (first, f"2 shots\n{two:.2f}"),
                   (total, f"5 shots\n{five:.2f}")):
        ax.plot([x, x], [-0.42, 0.42], color=INK, lw=2.0, zorder=3)
        ax.text(x, -0.62, lab, ha="center", va="top", fontsize=16)
    ax.set_xlim(-0.4, total + 0.4)
    ax.set_ylim(-1.15, 0.7)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"beetle: the measured 1 → 5 shot lift, {total:+.2f} mIoU",
                 fontsize=17)
    save(fig, "g12_concave.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    REPO = Path(args.repo)
    OUT = Path(args.out)
    g1_panels()
    g2_ladder()
    g3_mirror()
    g4_variance()
    g5_oracle()
    g6_orientation()
    g7_labels()
    g8_nesting()
    g9_readings()
    g10_metric()
    g11_design()
    g12_concave()
    print("done")
