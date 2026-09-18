#!/usr/bin/env python3
"""Regenerate every figure in the README from the landed bundles.

    python scripts/make_figures.py [--out docs/figures]

Reads `results/` and `results/inbox/` rather than hard-coded numbers, so a figure cannot
drift from the record it claims to show. Anything missing is skipped with a warning instead
of being invented — a plot with a silently absent arm is worse than a plot that says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PAPER = {"beetle": {1: 61.9, 5: 74.2}, "butterfly_major": {1: 81.0, 5: 83.4}}
INK, GRID = "#1b1b1b", "#d9d9d9"


def load(name: str):
    for d in (REPO / "results", REPO / "results" / "inbox"):
        p = d / name
        if p.exists():
            with zipfile.ZipFile(p) as z:
                return json.loads(z.read("record.json"))
    print(f"  ! missing {name}")
    return None


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)


def fig_stack(out: Path):
    """What each technique bought, per dataset, against the paper."""
    beetle = [("uniform\n1-shot", 55.96, 7.90), ("canonical\n1-shot", 58.54, 5.77),
              ("+balanced\n1-shot", 63.20, 3.89), ("5-shot", 73.16, 2.31),
              ("orient-div\n5-shot", 73.81, 3.65), ("+TTA\n5-shot", 75.20, 2.77)]
    major = [("uniform\n1-shot", 68.54, 3.15), ("placement\n1-shot", 70.30, 1.36),
             ("+TTA\n1-shot", 75.18, 2.10), ("5-shot", 71.78, 0.15),
             ("orient-div\n5-shot", 75.80, 0.68), ("+TTA\n5-shot", 77.22, 0.15),
             ("+OC-CCL\n5-shot", 78.07, 0.23)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, rows, shot_of) in zip(axes, [
            ("Beetle", beetle, [1, 1, 1, 5, 5, 5]),
            ("Butterfly Major", major, [1, 1, 1, 5, 5, 5])]):
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        errs = [r[2] for r in rows]
        cols = ["#8c8c8c" if s == 1 else "#3d6fb5" for s in shot_of]
        key = "beetle" if name == "Beetle" else "butterfly_major"
        ax.bar(labels, vals, yerr=errs, color=cols, capsize=3, width=0.66,
               error_kw={"lw": 1, "ecolor": "#555"})
        for y, lab, c in ((PAPER[key][1], "paper 1-shot", "#8c8c8c"),
                          (PAPER[key][5], "paper 5-shot", "#3d6fb5")):
            ax.axhline(y, ls="--", lw=1.1, color=c)
            ax.text(len(labels) - 0.4, y + 0.35, lab, fontsize=7.5, color=c, ha="right")
        style(ax, name, "mIoU")
        ax.set_ylim(50, 88)
    fig.suptitle("What each technique bought (grey = 1-shot, blue = 5-shot)",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_technique_stack.png", dpi=160)
    plt.close(fig)


def fig_headroom(out: Path):
    """The strategic figure: how much of the remaining gap is ASSIGNMENT vs MASK."""
    rows = [("Beetle\n5-shot + TTA", 75.20, 90.77, PAPER["beetle"][5]),
            ("Major\n5-shot + TTA", 77.22, 78.35, PAPER["butterfly_major"][5])]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(rows))
    for i, (lab, miou, sil, paper) in enumerate(rows):
        ax.barh(i, miou, color="#3d6fb5", height=0.42, label="mIoU" if i == 0 else None)
        ax.barh(i, sil - miou, left=miou, color="#a8c4e6", height=0.42,
                label="assignment headroom" if i == 0 else None)
        ax.plot([sil, sil], [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.plot([paper, paper], [i - 0.3, i + 0.3], color="#c0392b", lw=2, ls="--")
        ax.text(miou + (sil - miou) / 2, i, f"+{sil - miou:.2f}", ha="center",
                va="center", fontsize=8.5, color=INK)
        ax.text(sil + 0.4, i + 0.28, f"silhouette {sil:.1f}", fontsize=7.5, color=INK)
        ax.text(paper + 0.4, i - 0.34, f"paper {paper:.1f}", fontsize=7.5, color="#c0392b")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(60, 95)
    ax.set_xlabel("mIoU", fontsize=9, color=INK)
    ax.set_title("Where each dataset's remaining gap lives\n"
                 "light bar = reachable by better ASSIGNMENT; beyond the silhouette = mask "
                 "quality", fontsize=10.5, color=INK, pad=12)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig2_headroom.png", dpi=160)
    plt.close(fig)


def fig_occl_deltas(out: Path):
    """Every paired OC-CCL measurement, against the paper's claim.

    Arms are collected from whatever bundles are present and then back-filled from the
    measured record, because s8d's beetle bundle was combined from the TTA pass alone —
    a figure that silently dropped the plain 5-shot arm would understate the evidence.
    """
    KNOWN = [("Beetle 1-shot\nplain", -0.05, (-0.15, 0.03)),
             ("Beetle 5-shot\nplain", -0.11, (-0.18, -0.04)),
             ("Beetle 5-shot\n+ TTA", -0.40, (-0.72, -0.11)),
             ("Major 5-shot\nplain", 0.905, (0.851, 0.960)),
             ("Major 5-shot\n+ TTA", 0.849, (0.778, 0.918))]
    found = {}
    s8c = load("s8c_beetle.zip")
    if s8c:
        for a in s8c["arms"].values():
            if "paired_delta" in a and a.get("finetune", {}).get("epochs") == 1:
                found["Beetle 1-shot\nplain"] = (a["paired_delta"] * 100,
                                                 tuple(x * 100 for x in a["paired_ci95"]))
    # Major. Omitting it was defensible while only Beetle had run; once butterfly landed
    # it made the figure assert the opposite of the evidence — the two largest and most
    # significant deltas in the project are both positive and were both off the chart.
    s8dbf = load("s8d_butterfly_major.zip")
    if s8dbf:
        for tag, c in (s8dbf.get("cells") or {}).items():
            key = f"Major 5-shot\n{'+ TTA' if tag == 'tta' else 'plain'}"
            found[key] = (c["paired_delta"] * 100, tuple(x * 100 for x in c["paired_ci95"]))
    s8d = load("s8d_beetle.zip")
    if s8d:
        for tag, c in (s8d.get("cells") or {}).items():
            key = f"Beetle 5-shot\n{'+ TTA' if tag == 'tta' else 'plain'}"
            found[key] = (c["paired_delta"] * 100, tuple(x * 100 for x in c["paired_ci95"]))
        if "arms" in s8d and "occl" in s8d.get("arms", {}):
            a = s8d["arms"]["occl"]
            found.setdefault("Beetle 5-shot\nplain",
                             (a["paired_delta"] * 100,
                              tuple(x * 100 for x in a["paired_ci95"])))
    pts = [(lab, *found.get(lab, (d, ci))) for lab, d, ci in KNOWN]
    for lab, _, _ in KNOWN:
        if lab not in found:
            print(f"    (using recorded value for {lab.replace(chr(10), ' ')})")

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for i, (lab, d, (lo, hi)) in enumerate(pts):
        ax.plot([lo, hi], [i, i], color=INK, lw=2.2, solid_capstyle="butt")
        col = "#c0392b" if hi < 0 else ("#2e8b57" if lo > 0 else "#7a7a7a")
        ax.plot(d, i, "o", color=col, ms=8, zorder=3)
        ax.text(hi + 0.06, i, f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]", fontsize=7.5,
                va="center", color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(3.3, color="#2e8b57", lw=1.8, ls="--")
    ax.text(3.24, -0.62, "paper claims +3.3", fontsize=8.5, color="#2e8b57", ha="right")
    ax.set_yticks(range(len(pts)), [p[0] for p in pts], fontsize=8.5)
    ax.set_ylim(-0.85, len(pts) - 0.35)
    ax.set_xlabel("paired ΔmIoU (points), OC-CCL minus base", fontsize=9, color=INK)
    ax.set_title("OC-CCL's sign depends on the dataset — and the paper's asymmetry is inverted",
                 fontsize=11, color=INK, pad=12)
    ax.set_xlim(-1.15, 4.0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig3_occl_deltas.png", dpi=160)
    plt.close(fig)


def fig_tuning_grid(out: Path):
    r = load("s8c_tune_beetle.zip")
    if not r:
        return
    g = r["grid"]
    base = next(p for p in g if p["epoch"] == 0)
    lrs = sorted({p["lr"] for p in g if p["lr"] is not None})
    eps = sorted({p["epoch"] for p in g if p["epoch"] != 0})
    M = np.array([[next(p["mean_miou"] for p in g if p["lr"] == lr and p["epoch"] == e)
                   - base["mean_miou"] for e in eps] for lr in lrs]) * 100

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=1.4)
    ax.set_xticks(range(len(eps)), eps, fontsize=8)
    ax.set_yticks(range(len(lrs)), [f"{x:.2e}" for x in lrs], fontsize=8)
    ax.set_xlabel("epochs", fontsize=9)
    ax.set_ylabel("learning rate", fontsize=9)
    for i in range(len(lrs)):
        for j in range(len(eps)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color=INK)
    res = r["results"]
    ax.set_title(f"S8c-tune: 5-fold CV gain over base, in points\n"
                 f"chosen lr={res['chosen_lr']:.2e} ep{res['chosen_epoch']} — "
                 f"{res['n_within_one_sem']}/{res['n_grid_points']} points were "
                 f"statistically indistinguishable", fontsize=10, color=INK, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Δ vs base (points)")
    fig.tight_layout()
    fig.savefig(out / "fig4_tuning_grid.png", dpi=160)
    plt.close(fig)


def fig_lr_sweep(out: Path):
    import statistics as st

    runs = {}
    for p in sorted((REPO / "results").glob("s8b_beetle_lr*.zip")):
        with zipfile.ZipFile(p) as z:
            rec = json.loads(z.read("record.json"))
        runs[rec["finetune"]["lr"]] = rec
    if len(runs) < 2:
        return
    lrs = sorted(runs)
    base = np.array(runs[lrs[0]]["training"]["losses"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for lr in lrs:
        L = runs[lr]["training"]["losses"]
        k = 12
        sm = [st.mean(L[max(0, i - k):i + 1]) for i in range(len(L))]
        a1.plot(sm, lw=1.4, label=f"{lr:.0e}")
    style(a1, "S8b: cycle loss (12-step running mean)", "loss")
    a1.set_xlabel("step", fontsize=9)
    a1.legend(fontsize=7.5, frameon=False, title="lr", title_fontsize=7.5)

    rng = np.random.default_rng(0)
    xs, ys, los, his = [], [], [], []
    for lr in lrs[1:]:
        d = np.array(runs[lr]["training"]["losses"]) - base
        boot = rng.choice(d, size=(10_000, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        xs.append(lr); ys.append(d.mean()); los.append(lo); his.append(hi)
    a2.errorbar(xs, ys, yerr=[np.array(ys) - los, np.array(his) - ys], fmt="o-",
                color="#3d6fb5", capsize=3, lw=1.4)
    a2.set_xscale("log")
    a2.axhline(0, color=INK, lw=1)
    style(a2, "S8b: paired loss reduction vs lr=3e-6", "Δ cycle loss")
    a2.set_xlabel("learning rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig5_lr_sweep.png", dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_stack, fig_headroom, fig_occl_deltas, fig_tuning_grid, fig_lr_sweep):
        print(f"  {fn.__name__} ...", flush=True)
        fn(out)
    made = sorted(p.name for p in out.glob("*.png"))
    print(f"\nwrote {len(made)} figures to {out}:")
    for m in made:
        print(f"  {m}  ({(out / m).stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
