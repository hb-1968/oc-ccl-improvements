#!/usr/bin/env python3
"""Figures for the status talk: everything that landed since the techniques deck.

Same discipline as the earlier figure scripts. Every number is lifted live from a
landed bundle in `results/` (s15, s16, s17, s17b, s17c, s17d and the two probe
JSONs), or transcribed from a named README row and ASSERTED against the bundle
before anything is drawn.

    python make_status_figures.py --repo /path/oc-ccl-improvements --out figures_status/
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

# ── palette: black on white, as the decks are ───────────────────────────────
INK = "#000000"
GRID = "#D6D6D6"
HOLLOW = "#FFFFFF"
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
OUT = Path("figures_status")

PLATE = dict(facecolor="white", edgecolor="none", pad=1.5)
HATCH = "////"

#: trait identity keeps its colour; everything else is black on white.
TRAIT_RGB = {
    1: (215, 48, 39),      # head
    2: (252, 141, 89),     # pronotum
    3: (69, 117, 180),     # elytra
    4: (26, 152, 80),      # antenna
    5: (145, 84, 168),     # legs
}
TRAIT_NAME = {1: "head", 2: "pronotum", 3: "elytra", 4: "antenna", 5: "legs"}


def bundle(name: str):
    for d in (REPO / "results", REPO / "results" / "inbox"):
        p = d / f"{name}.zip"
        if p.exists():
            with zipfile.ZipFile(p) as z:
                return json.loads(z.read("record.json"))
    raise SystemExit(f"missing bundle: {name}")


def jload(name: str):
    p = REPO / "results" / name
    if not p.exists():
        raise SystemExit(f"missing file: {p}")
    return json.loads(p.read_text())


def close(a, b, tol=0.03):
    assert abs(a - b) < tol, f"record disagrees with README: {a} vs {b}"


def tidy(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xgrid:
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, lw=1.1)


def _edged(ax):
    for p in ax.patches:
        p.set_edgecolor(INK)
        p.set_linewidth(1.4)


def save(fig, name, tight=True):
    for ax in fig.get_axes():
        _edged(ax)
    OUT.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.18)
    else:
        fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("  wrote", name)


def pct(delta, base):
    return f"({100 * delta / base:+.1f}%)"


# ── the records, read once ──────────────────────────────────────────────────
S15 = bundle("s15")
S16 = bundle("s16")
S17 = bundle("s17")
S17B = bundle("s17b")
S17C = bundle("s17c")
S17D = bundle("s17d")
ANAT = jload("s17probe_beetle_base_anatomy.json")
GRAD = jload("s17d_sentinel_gradient.json")


def cell15(ds, name, key="miou"):
    v = S15["datasets"][ds]["cells"][name][key]
    return None if v is None else 100 * v


def pair15(ds, name, key="delta"):
    p = S15["datasets"][ds]["pairings"][name]
    if key in ("delta", "paired_std"):
        return 100 * p[key]
    if key == "ci95":
        return [100 * x for x in p[key]]
    return p[key]


def ev(rec, name, key="miou"):
    c = rec["datasets"]["beetle"]["eval"]["cells"][name]
    v = c.get(key)
    return None if v is None else 100 * v


def evpair(rec, name, key="delta"):
    p = rec["datasets"]["beetle"]["eval"]["pairings"][name]
    if key == "delta":
        return 100 * p["delta"]
    if key == "ci95":
        return [100 * x for x in p["ci95"]]
    return p.get(key)


def arm(rec, name):
    return rec["datasets"]["beetle"]["results"][name]


#: levels banked before this talk's window, each with the README row it sits in
BANKED = {
    "beetle_sam2_1shot": 63.20,       # gates table, S4
    "beetle_sam2_5shot": 73.16,       # gates table, S6
    "bf_sam2_1shot": 70.30,           # gates table, S5
    "bf_sam2_5shot": 71.78,           # gates table, S6
    "beetle_champion_1shot": 69.88,   # S14sam3 arch_tta
    "bf_best_1shot": 76.64,           # S14sam3 stack_tta
    "bf_best_5shot": 78.07,           # S8d, SAM 2 5-shot + TTA + OC-CCL
    "beetle_crown_sam2": 64.12,       # S11 crown, SAM 2
    "paper_beetle_1shot": 61.9,
    "paper_beetle_1shot_occl": 65.2,
    "paper_beetle_5shot": 74.2,
    "paper_beetle_5shot_occl": 75.6,
    "paper_bf_1shot": 81.0,
    "paper_bf_5shot": 83.4,
    "major_curated_masks": 1727,      # "what is left", item 2
}

# the bundles have to agree with the levels the talk quotes
close(cell15("beetle", "sam2_base"), BANKED["beetle_sam2_5shot"], tol=0.02)
close(cell15("butterfly_major", "sam2_base"), BANKED["bf_sam2_5shot"], tol=0.02)
close(ev(S17D, "base"), BANKED["beetle_sam2_1shot"], tol=0.05)


# ── K1 · the scoreboard against the paper ───────────────────────────────────
def k1_scoreboard():
    rows = [
        ("beetle 1-shot", BANKED["beetle_champion_1shot"],
         BANKED["paper_beetle_1shot"], "S14sam3 arch_tta", True),
        ("beetle 5-shot", cell15("beetle", "arch_tta"),
         BANKED["paper_beetle_5shot"], "S15 arch_tta", True),
        ("Major 1-shot", BANKED["bf_best_1shot"],
         BANKED["paper_bf_1shot"], "S14sam3 stack_tta", False),
        ("Major 5-shot", BANKED["bf_best_5shot"],
         BANKED["paper_bf_5shot"], "S8d", False),
    ]
    close(rows[1][1], 78.91, tol=0.02)
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.16)
    for i, (lab, got, target, who, ok) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, got, 0.56, color=(SLATE if ok else HOLLOW))
        ax.plot([target, target], [y - 0.36, y + 0.36], color=INK, lw=3.4)
        d = got - target
        ax.text(got + 0.6, y, f"{got:.2f}   {d:+.2f} vs target",
                va="center", fontsize=15, fontweight="bold", bbox=PLATE)
        ax.text(1.5, y, who, va="center", fontsize=13, color="white"
                if ok else INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=16)
    ax.set_xlim(0, 100)
    ax.set_xlabel("mIoU. bar = our best, vertical rule = the paper's number")
    tidy(ax)
    save(fig, "k1_scoreboard.png")


# ── K2 · S16, in one figure ─────────────────────────────────────────────────
def k2_s16():
    rows = []
    for tag, ds in (("butterfly Major", "butterfly_major"), ("beetle", "beetle")):
        p = S16["datasets"][ds]["pairings"]["second_shot_vs_tta"]
        rows.append((tag, 100 * p["delta"], [100 * x for x in p["ci95"]],
                     p["n_seeds_improved"],
                     100 * S16["datasets"][ds]["cells"]["s1_tta"]["miou"]))
    close(rows[0][1], -3.71, tol=0.02)
    close(rows[1][1], 5.17, tol=0.02)
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.92, bottom=0.22)
    for i, (lab, d, ci, imp, base) in enumerate(rows):
        y = 1 - i
        ax.errorbar(d, y, xerr=[[d - ci[0]], [ci[1] - d]], fmt="D", ms=16,
                    color=INK, ecolor=INK, elinewidth=3, capsize=8)
        ax.text(d, y + (0.34 if i == 0 else -0.40),
                f"{d:+.2f} {pct(d, base)}   {imp}/20 seeds", ha="center",
                fontsize=17, fontweight="bold", bbox=PLATE)
    ax.axvline(0, color=INK, lw=2.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([rows[1][0], rows[0][0]], fontsize=17)
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlim(-6.2, 8.6)
    ax.set_xlabel("2-shot plain  minus  1-shot + TTA   (mIoU)")
    ax.text(-5.9, 1.55, "rotate the support", fontsize=15, style="italic")
    ax.text(8.3, 1.55, "label a second specimen", fontsize=15, style="italic",
            ha="right")
    tidy(ax)
    save(fig, "k2_s16.png")


# ── K3 · S15 against its pre-registered bands ───────────────────────────────
def k3_s15_band():
    # README "S15 brief": beetle central 78.7, band 76.8-81.3; butterfly 78-79.5
    rows = [("beetle", 76.8, 81.3, cell15("beetle", "arch_tta")),
            ("butterfly\nMajor", 78.0, 79.5, cell15("butterfly_major", "arch_tta"))]
    close(rows[0][3], 78.91, tol=0.02)
    close(rows[1][3], 77.47, tol=0.02)
    for n_show in (1, 2):
        fig, ax = plt.subplots(figsize=(12.4, 4.4))
        fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.22)
        for i, (lab, lo, hi, got) in enumerate(rows):
            y = 1 - i
            ax.barh(y, hi - lo, 0.5, left=lo, color=HOLLOW, hatch=HATCH,
                    edgecolor=INK)
            if n_show == 2:
                ax.plot([got], [y], "D", ms=17, color=INK, zorder=3)
                inside = lo <= got <= hi
                ax.text(got, y - 0.40,
                        f"{got:.2f}  {'inside the band' if inside else 'below the band'}",
                        ha="center", fontsize=16, fontweight="bold", bbox=PLATE)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([rows[1][0], rows[0][0]], fontsize=16)
        ax.set_ylim(-0.75, 1.7)
        ax.set_xlim(74.5, 83)
        ax.set_xlabel("5-shot mIoU of the champion, against the band written "
                      "before the run")
        tidy(ax)
        save(fig, f"k3_s15_band_{n_show}.png", tight=False)


# ── K4 · the S15 cells ──────────────────────────────────────────────────────
def k4_s15_cells():
    labs = ["base", "base + TTA", "champion\n(arch + TTA)"]
    keys = ["base", "base_tta", "arch_tta"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    fig.subplots_adjust(wspace=0.3, left=0.09, right=0.97, bottom=0.2, top=0.88)
    for ax, ds, ttl, ref in (
            (axes[0], "beetle", "beetle", BANKED["beetle_sam2_5shot"]),
            (axes[1], "butterfly_major", "butterfly Major",
             BANKED["bf_sam2_5shot"])):
        vals = [cell15(ds, k) for k in keys]
        for i, v in enumerate(vals):
            ax.bar(i, v, 0.62, color=(INK if i == 2 else SLATE))
            ax.text(i, v + 0.35, f"{v:.2f}", ha="center", fontsize=16,
                    fontweight="bold")
        ax.axhline(ref, ls="--", lw=2.2, color=INK)
        ax.text(2.45, ref, f"SAM 2\n5-shot\n{ref:.2f}", va="center",
                fontsize=12.5, ha="right")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labs, fontsize=12.5)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(68, 82)
        ax.set_title(ttl, fontsize=16)
        ax.set_ylabel("5-shot mIoU" if ds == "beetle" else "")
        tidy(ax, xgrid=False)
        ax.grid(axis="y", color=GRID, lw=1.1)
        ax.set_axisbelow(True)
    save(fig, "k4_s15_cells.png")


# ── K5 · the adapter is beetle-specific ─────────────────────────────────────
def k5_s15_arch():
    rows = [("beetle", pair15("beetle", "arch_vs_base_tta"),
             pair15("beetle", "arch_vs_base_tta", "ci95"),
             pair15("beetle", "arch_vs_base_tta", "n_seeds_improved")),
            ("butterfly\nMajor", pair15("butterfly_major", "arch_vs_base_tta"),
             pair15("butterfly_major", "arch_vs_base_tta", "ci95"),
             pair15("butterfly_major", "arch_vs_base_tta", "n_seeds_improved"))]
    close(rows[0][1], 0.97, tol=0.02)
    close(rows[1][1], -0.48, tol=0.02)
    fig, ax = plt.subplots(figsize=(12.2, 4.2))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.9, bottom=0.24)
    for i, (lab, d, ci, imp) in enumerate(rows):
        y = 1 - i
        ax.barh(y, d, 0.5, color=(SLATE if d > 0 else HOLLOW))
        ax.errorbar(d, y, xerr=[[d - ci[0]], [ci[1] - d]], fmt="none",
                    ecolor=INK, elinewidth=2.6, capsize=7)
        ax.text(d + (0.08 if d > 0 else -0.08), y,
                f"{d:+.2f}, {imp}/20 seeds", va="center",
                ha="left" if d > 0 else "right", fontsize=16,
                fontweight="bold", bbox=PLATE)
    ax.axvline(0, color=INK, lw=2.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([rows[1][0], rows[0][0]], fontsize=16)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(-1.4, 1.8)
    ax.set_xlabel("the trained adapter, against the same model without it "
                  "(5-shot, mIoU)")
    tidy(ax)
    save(fig, "k5_s15_arch.png")


# ── K6 · the assignment pot closes as shots rise ────────────────────────────
def k6_pot():
    rows = [
        ("1 shot, base", cell15("beetle", "base_1"),
         cell15("beetle", "base_1", "miou_oracle")),
        ("1 shot, champion", cell15("beetle", "arch_tta_1"),
         cell15("beetle", "arch_tta_1", "miou_oracle")),
        ("5 shots, base", cell15("beetle", "base"),
         cell15("beetle", "base", "miou_oracle")),
        ("5 shots, champion", cell15("beetle", "arch_tta"),
         cell15("beetle", "arch_tta", "miou_oracle")),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.95, bottom=0.18)
    for i, (lab, got, ceil) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, ceil, 0.55, color=HOLLOW, hatch=HATCH, edgecolor=INK)
        ax.barh(y, got, 0.55, color=SLATE)
        ax.text(got - 0.5, y, f"{got:.2f}", va="center", ha="right",
                fontsize=15, fontweight="bold", color="white")
        ax.text(ceil + 0.4, y, f"ceiling {ceil:.2f}   pot {ceil - got:+.2f}",
                va="center", fontsize=15, fontweight="bold", bbox=PLATE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=15)
    ax.set_xlim(55, 92)
    ax.set_xlabel("beetle mIoU: achieved (solid) inside the oracle ceiling "
                  "(hatched)")
    tidy(ax)
    save(fig, "k6_pot.png")


# ── K7 · the epoch is a tail ────────────────────────────────────────────────
def k7_tail():
    L = ANAT["base"]["loss"]
    shares = [(5, L["top5_share"]), (10, L["top10_share"]), (20, L["top20_share"])]
    close(100 * L["top10_share"], 55.7, tol=0.3)
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.2)
    for i, (n, s) in enumerate(shares):
        y = len(shares) - 1 - i
        ax.barh(y, 100 * s, 0.55, color=SLATE)
        ax.text(100 * s + 1.2, y, f"{100 * s:.0f}% of the epoch's loss",
                va="center", fontsize=17, fontweight="bold", bbox=PLATE)
    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels([f"worst {n} pairs" for n, _ in shares][::-1],
                       fontsize=16)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"share of one epoch, over all {ANAT['base']['n_pairs']} "
                  f"beetle training pairs")
    ax.set_title(f"median loss {L['median']:.2f}, p90 {L['p90']:.1f}, "
                 f"max {L['max']:.1f}", fontsize=15)
    tidy(ax)
    save(fig, "k7_tail.png")


# ── K8 · the sentinel, on one real pair ─────────────────────────────────────
def k8_sentinel():
    hard = GRAD["gates"]["hard"]
    soft = GRAD["gates"]["ste20"]
    ch = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.18)
    w = 0.38
    for i, c in enumerate(ch):
        h, s = hard["bce_per_channel"][i], soft["bce_per_channel"][i]
        ax.bar(i - w / 1.9, h, w, color=SLATE)
        ax.bar(i + w / 1.9, s, w, color=HOLLOW)
        ax.text(i - w / 1.9, h + 0.4, f"{h:.2f}", ha="center", fontsize=13,
                fontweight="bold")
        ax.text(i + w / 1.9, s + 0.4, f"{s:.2f}", ha="center", fontsize=13)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{TRAIT_NAME[c]}\n{'absent' if c == 4 else 'found'}"
                        for c in ch], fontsize=13)
    ax.set_ylim(0, 21)
    ax.set_ylabel("per-channel BCE")
    ax.set_title("one real training pair: SAM 2's gate (grey) against the "
                 "straight-through gate (white)", fontsize=14)
    ax.text(0.5, 17.5, f"gradient into the score head\n"
                       f"{hard['score_head_grad_norm']:.1f} under SAM 2's gate, "
                       f"{soft['score_head_grad_norm']:.3f} under the surrogate",
            fontsize=14, bbox=dict(facecolor="white", edgecolor=INK, lw=1.6,
                                   pad=7))
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "k8_sentinel.png")


# ── K9 · the reverse leg absorbs the objective ──────────────────────────────
def k9_absorb():
    arms = [("carrier", "carrier"), ("found_mask", "found_mask"),
            ("excl", "excl"), ("arch", "arch")]
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.18)
    w = 0.38
    for i, (lab, key) in enumerate(arms):
        a = arm(S17, key)
        f = 100 * a["forward"]["delta"]["mean_delta"]
        c = 100 * a["closing"]["delta"]["mean_delta"]
        sig_f = a["forward"]["delta"]["significant"]
        sig_c = a["closing"]["delta"]["significant"]
        ax.bar(i - w / 1.9, f, w, color=HOLLOW)
        ax.bar(i + w / 1.9, c, w, color=SLATE)
        ax.text(i - w / 1.9, f + 0.25, f"{f:+.2f}{'*' if sig_f else ''}",
                ha="center", fontsize=13, fontweight="bold")
        ax.text(i + w / 1.9, c + 0.25, f"{c:+.2f}{'*' if sig_c else ''}",
                ha="center", fontsize=13, fontweight="bold")
    close(100 * arm(S17, "carrier")["closing"]["delta"]["mean_delta"], 3.74,
          tol=0.05)
    close(100 * arm(S17, "arch")["closing"]["delta"]["mean_delta"], 7.26,
          tol=0.05)
    ax.axhline(0, color=INK, lw=2.2)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=15)
    ax.set_ylim(-1.5, 8.6)
    ax.set_ylabel("mIoU change on the training pairs")
    ax.set_title("forward leg, the eval's own direction (white)     "
                 "closing leg, the loss's own target (grey)", fontsize=14)
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "k9_absorb.png")


# ── K10 · the cycle's own images, from the bundle ───────────────────────────
def k10_maps():
    """Forward and closing label maps for one real training pair, base vs arch.

    Label maps keep their colour: the hue is the trait identity. The pair is
    step 0 of the beetle epoch, the same pair the sentinel probe used.
    """
    want = {
        "k10_base_fwd.png": "beetle/maps/base/fwd/0000.png",
        "k10_base_cls.png": "beetle/maps/base/cls/0000.png",
        "k10_arch_cls.png": "beetle/maps/arch/cls/0000.png",
        "k10_arch_fwd.png": "beetle/maps/arch/fwd/0000.png",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(REPO / "results" / "s17.zip") as z:
        for dst, src in want.items():
            with z.open(src) as fh:
                lab = np.array(Image.open(fh))
            rgb = np.full(lab.shape + (3,), 255, np.uint8)
            for k, col in TRAIT_RGB.items():
                rgb[lab == k] = col
            Image.fromarray(rgb).save(OUT / dst)
            print("  wrote", dst, rgb.shape[:2])


# ── K11 · the SAM 2 ladder since the last talk ──────────────────────────────
def k11_ladder():
    rows = [
        ("base", ev(S17D, "base_tta"), None),
        ("carrier (the S11 crown's loss)", ev(S17C, "carrier_tta"),
         evpair(S17C, "carrier_tta_vs_base_tta")),
        ("+ exclusivity", ev(S17C, "excl_tta"),
         evpair(S17C, "excl_tta_vs_base_tta")),
        ("+ forward-only legs, objectness", ev(S17C, "comp_tta"),
         evpair(S17C, "comp_tta_vs_base_tta")),
        ("+ objectness weight 10", ev(S17C, "comp_w10_tta"),
         evpair(S17C, "comp_w10_tta_vs_base_tta")),
        ("+ arch scopes  (the recipe)", ev(S17D, "comp_arch_w10_tta"),
         evpair(S17D, "comp_arch_w10_tta_vs_base_tta")),
    ]
    close(rows[-1][1], 65.04, tol=0.02)
    for n_show in (1, 2):
        fig, ax = plt.subplots(figsize=(12.4, 5.4))
        fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.16)
        for i, (lab, v, d) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.barh(y, v, 0.58, color=(INK if i == len(rows) - 1 else SLATE))
            txt = f"{v:.2f}"
            if d is not None:
                txt += f"   {d:+.2f} vs base"
            ax.text(v + 0.06, y, txt, va="center", fontsize=14.5,
                    fontweight="bold", bbox=PLATE)
        if n_show == 2:
            ax.axvline(BANKED["beetle_champion_1shot"], ls="--", lw=2.4,
                       color=INK)
            ax.text(BANKED["beetle_champion_1shot"] - 0.08, 0.1,
                    f"SAM 3 champion {BANKED['beetle_champion_1shot']:.2f} ",
                    ha="right", fontsize=14, fontweight="bold", bbox=PLATE)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=12.5)
        ax.set_xlim(62.5, 70.6)
        ax.set_xlabel("beetle 1-shot mIoU, with TTA, 20 seeds")
        tidy(ax)
        save(fig, f"k11_ladder_{n_show}.png", tight=False)


# ── K12 · the oracle ceiling on SAM 2 ───────────────────────────────────────
def k12_ceiling():
    rows = [("base", ev(S17D, "base_tta", "miou_oracle")),
            ("S17b composition", ev(S17B, "fwdonly_objness_excl_tta", "miou_oracle")),
            ("+ arch scopes (S17c)", ev(S17C, "comp_arch_tta", "miou_oracle")),
            ("the recipe (S17d)", ev(S17D, "comp_arch_w10_tta", "miou_oracle"))]
    close(rows[0][1], 66.99, tol=0.15)
    close(rows[-1][1], 67.92, tol=0.02)
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.2)
    for i, (lab, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, 0.55, color=(INK if i == len(rows) - 1 else SLATE))
        ax.text(v + 0.03, y, f"{v:.2f}", va="center", fontsize=16,
                fontweight="bold", bbox=PLATE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=13.5)
    ax.set_xlim(66.0, 68.6)
    ax.set_xlabel("beetle oracle ceiling with TTA: what the proposals allow")
    tidy(ax)
    save(fig, "k12_ceiling.png")


# ── K13 · the objectness weight is a free prior ─────────────────────────────
def k13_weight():
    arms = [("1", "comp"), ("3", "comp_w3"), ("10", "comp_w10"),
            ("30", "comp_w30")]
    base_found = arm(S17C, "comp")["found_rate"]["base"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    fig.subplots_adjust(wspace=0.32, left=0.09, right=0.97, bottom=0.2, top=0.88)
    ax = axes[0]
    ax.axhline(100 * base_found, ls="--", lw=2.2, color=INK)
    ax.text(3.35, 100 * base_found - 0.35, f"base {100 * base_found:.1f}",
            ha="right", fontsize=13)
    for i, (lab, key) in enumerate(arms):
        v = 100 * arm(S17C, key)["found_rate"]["ft"]
        ax.bar(i, v, 0.6, color=SLATE)
        ax.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=14,
                fontweight="bold")
    ax.set_ylim(96.5, 99.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([a[0] for a in arms], fontsize=15)
    ax.set_xlabel("objectness weight")
    ax.set_title("channels the forward leg finds, %", fontsize=14)
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)

    ax = axes[1]
    for i, (lab, key) in enumerate(arms):
        d = evpair(S17C, f"{key}_tta_vs_base_tta")
        ci = evpair(S17C, f"{key}_tta_vs_base_tta", "ci95")
        ax.errorbar(i, d, yerr=[[d - ci[0]], [ci[1] - d]], fmt="D", ms=14,
                    color=INK, ecolor=INK, elinewidth=2.6, capsize=7)
        ax.text(i + 0.12, d, f"{d:+.2f}", va="center", fontsize=14,
                fontweight="bold", bbox=PLATE)
    ax.axhline(0, color=INK, lw=2.0)
    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(0, 1.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([a[0] for a in arms], fontsize=15)
    ax.set_xlabel("objectness weight")
    ax.set_title("test mIoU against base, with TTA", fontsize=14)
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "k13_weight.png")


# ── K14 · scope subsumes weight ─────────────────────────────────────────────
def k14_scope():
    rows = [
        ("composition", evpair(S17D, "comp_tta_vs_base_tta"),
         evpair(S17D, "comp_tta_vs_base_tta", "ci95")),
        ("+ weight 10 (S17c)", evpair(S17C, "comp_w10_tta_vs_base_tta"),
         evpair(S17C, "comp_w10_tta_vs_base_tta", "ci95")),
        ("+ arch scopes (S17c)", evpair(S17C, "comp_arch_tta_vs_base_tta"),
         evpair(S17C, "comp_arch_tta_vs_base_tta", "ci95")),
        ("both (S17d)", evpair(S17D, "comp_arch_w10_tta_vs_base_tta"),
         evpair(S17D, "comp_arch_w10_tta_vs_base_tta", "ci95")),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.2)
    for i, (lab, d, ci) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, d, 0.55, color=(INK if i == 3 else SLATE))
        ax.errorbar(d, y, xerr=[[d - ci[0]], [ci[1] - d]], fmt="none",
                    ecolor=INK, elinewidth=2.6, capsize=7)
        ax.text(ci[1] + 0.06, y, f"{d:+.2f}", va="center", fontsize=16,
                fontweight="bold", bbox=PLATE)
    add = rows[1][1] + rows[2][1] - rows[0][1]
    ax.plot([add, add], [-0.4, 0.4], color=INK, lw=3.0, ls=":")
    ax.text(add, -0.62, f"if the two added: {add:+.2f}", ha="center",
            fontsize=13)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=13.5)
    ax.set_xlim(0, 2.4)
    ax.set_ylim(-0.95, 3.6)
    ax.set_xlabel("beetle 1-shot mIoU against base, with TTA")
    tidy(ax)
    save(fig, "k14_scope.png")


# ── K15 · the differentiable sentinel is a null lever ───────────────────────
def k15_soft():
    rows = [
        ("no prior, soft gate", evpair(S17D, "fwdonly_excl_soft_tta_vs_comp_tta"),
         evpair(S17D, "fwdonly_excl_soft_tta_vs_comp_tta", "ci95")),
        ("prior + soft gate", evpair(S17D, "comp_soft_tta_vs_comp_tta"),
         evpair(S17D, "comp_soft_tta_vs_comp_tta", "ci95")),
        ("recipe + soft gate",
         evpair(S17D, "comp_arch_w10_soft_tta_vs_comp_tta")
         - evpair(S17D, "comp_arch_w10_tta_vs_comp_tta"), None),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.22)
    for i, (lab, d, ci) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, d, 0.5, color=HOLLOW)
        if ci:
            ax.errorbar(d, y, xerr=[[d - ci[0]], [ci[1] - d]], fmt="none",
                        ecolor=INK, elinewidth=2.6, capsize=7)
        ax.text(d - 0.03, y, f"{d:+.2f}", va="center", ha="right", fontsize=16,
                fontweight="bold", bbox=PLATE)
    ax.axvline(0, color=INK, lw=2.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=13)
    ax.set_xlim(-0.55, 0.25)
    ax.set_xlabel("mIoU against the same arm with SAM 2's own gate, with TTA")
    tidy(ax)
    save(fig, "k15_soft.png")


# ── K16 · nothing transfers to 5-shot ───────────────────────────────────────
def k16_fiveshot():
    rows = [
        ("S17c composition", evpair(S17C, "comp_tta_vs_base_tta"),
         evpair(S17C, "comp_5_tta_vs_base_5_tta")),
        ("the recipe", evpair(S17D, "comp_arch_w10_tta_vs_base_tta"),
         evpair(S17D, "comp_arch_w10_5_tta_vs_base_5_tta")),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.9, bottom=0.2)
    w = 0.34
    for i, (lab, one, five) in enumerate(rows):
        y = 1 - i
        ax.barh(y + w / 1.6, one, w, color=SLATE)
        ax.barh(y - w / 1.6, five, w, color=HOLLOW)
        ax.text(one + 0.04, y + w / 1.6, f"{one:+.2f}  at 1 shot", va="center",
                fontsize=14, fontweight="bold", bbox=PLATE)
        ax.text(five + 0.04, y - w / 1.6, f"{five:+.2f}  at 5 shots, ns",
                va="center", fontsize=14, bbox=PLATE)
    ax.axvline(0, color=INK, lw=2.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([rows[1][0], rows[0][0]], fontsize=15)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(-0.3, 2.0)
    ax.set_xlabel("beetle mIoU against the same base at that shot count, "
                  "with TTA")
    tidy(ax)
    save(fig, "k16_fiveshot.png")


# ── K17 · S18: the side channel that is left ────────────────────────────────
def k17_s18():
    """The filed pre-session reads for S18 (README 'S18 brief')."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    fig.subplots_adjust(wspace=0.36, left=0.11, right=0.97, bottom=0.2, top=0.86)
    ax = axes[0]
    rows = [("beetle", 0.127, 0.124), ("butterfly", 0.226, 0.224)]
    w = 0.34
    for i, (lab, got, chance) in enumerate(rows):
        ax.bar(i - w / 1.9, got, w, color=SLATE)
        ax.bar(i + w / 1.9, chance, w, color=HOLLOW)
        ax.text(i - w / 1.9, got + 0.006, f"{got:.3f}", ha="center", fontsize=13,
                fontweight="bold")
        ax.text(i + w / 1.9, chance + 0.006, f"{chance:.3f}", ha="center",
                fontsize=13)
    ax.set_xticks(range(2))
    ax.set_xticklabels([r[0] for r in rows], fontsize=15)
    ax.set_ylim(0, 0.30)
    ax.set_title("forward mask's similarity to the reference mask\n"
                 "(grey) against chance (white)", fontsize=13)
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)

    ax = axes[1]
    modes = [("binary\nmask", 0.0), ("plain\nSTE", 0.1),
             ("memory\nSTE", 117.3), ("continuous\nprompt", 56.4)]
    for i, (lab, v) in enumerate(modes):
        ax.bar(i, max(v, 0.02), 0.6, color=(SLATE if i >= 2 else HOLLOW))
        ax.text(i, max(v, 0.02) * 1.25, f"{v:g}", ha="center", fontsize=13,
                fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1200)
    ax.set_xticks(range(4))
    ax.set_xticklabels([m[0] for m in modes], fontsize=12.5)
    ax.set_title("forward-leg gradient on one channel of one real pair,\n"
                 "by what crosses the centre of the cycle", fontsize=13)
    tidy(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, lw=1.1)
    ax.set_axisbelow(True)
    save(fig, "k17_s18.png")


# ── K18 · the label budget ──────────────────────────────────────────────────
def k18_labels():
    rows = [("the headline", 1), ("what the best Major arm reads",
                                  BANKED["major_curated_masks"])]
    fig, ax = plt.subplots(figsize=(12.2, 3.8))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.9, bottom=0.24)
    for i, (lab, v) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, 0.5, color=(SLATE if i else HOLLOW))
        ax.text(v * 1.15, y, f"{v:,} labelled mask{'' if v == 1 else 's'}",
                va="center", fontsize=17, fontweight="bold", bbox=PLATE)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 12000)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([rows[1][0], rows[0][0]], fontsize=14)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlabel("labelled masks, log scale")
    tidy(ax)
    save(fig, "k18_labels.png")


# ── K19 · what has already failed to move Major ─────────────────────────────
def k19_major():
    """Every lever tried against Major's residual, with the README row for each."""
    rows = [
        ("support curation (S5b)", 0.0),
        ("orientation TTA (S5c)", 4.88),
        ("rotation 2x2 (S6c)", 0.0),
        ("5-shot (S6)", 1.48),
        ("every loss arm (S8 to S13)", 0.85),
        ("SeC-4B swap (S14sec)", 0.0),
        ("SAM 3 swap (S14sam3)", 0.12),
        ("SAM 3 at 5-shot (S15)", -0.48),
    ]
    gap = BANKED["paper_bf_5shot"] - BANKED["bf_best_5shot"]
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.92, bottom=0.16)
    for i, (lab, d) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, d, 0.55, color=(SLATE if d > 0.5 else HOLLOW))
        ax.text(max(d, 0) + 0.12, y, f"{d:+.2f}", va="center", fontsize=14,
                fontweight="bold", bbox=PLATE)
    ax.axvline(gap, ls="--", lw=2.6, color=INK)
    ax.text(gap - 0.15, len(rows) - 1.2, f"the gap: {gap:.2f} ", ha="right",
            fontsize=15, fontweight="bold", bbox=PLATE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=13)
    ax.set_xlim(-1.2, 6.2)
    ax.set_xlabel("mIoU moved on butterfly Major")
    tidy(ax)
    save(fig, "k19_major.png")



# ── K20 · the two second references, panels from the project's own render ───
def k20_panels():
    """Reuse the S16 deck's panels: same source render, same treatment."""
    import shutil
    src = Path("figures_s16")
    want = ["g1_support.png", "g1_support_rot.png", "g1_query.png",
            "g1_support_mask.png"]
    if not all((src / w).exists() for w in want):
        raise SystemExit("run make_s16_figures.py first: " + str(src))
    OUT.mkdir(parents=True, exist_ok=True)
    for w in want:
        shutil.copyfile(src / w, OUT / w.replace("g1_", "k20_"))
        print("  wrote", w.replace("g1_", "k20_"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    REPO = Path(args.repo)
    OUT = Path(args.out)
    k1_scoreboard()
    k2_s16()
    k3_s15_band()
    k4_s15_cells()
    k5_s15_arch()
    k6_pot()
    k7_tail()
    k8_sentinel()
    k9_absorb()
    k10_maps()
    k11_ladder()
    k12_ceiling()
    k13_weight()
    k14_scope()
    k15_soft()
    k16_fiveshot()
    k17_s18()
    k18_labels()
    k19_major()
    k20_panels()
    print("done")
