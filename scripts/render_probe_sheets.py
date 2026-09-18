#!/usr/bin/env python3
"""S17 — contact sheets and summary figures from a training-pair probe bundle. CPU, no torch.

    python scripts/render_probe_sheets.py --bundle results/s17.zip --dataset beetle \
        --select most_degraded --n 6 --out-dir docs/figures/s17

The GPU side (`scripts/probe_train_pairs.py`) banked every predicted label map as a
uint8 PNG and every score in `pairs.csv`; the images themselves live in the local
cache and the GT masks in the SST checkout. So any training pair can be looked at here,
after the fact, on a machine with no GPU — sorted by whatever the question is.

One row per pair, eight panels:

    x0 | y0 (GT of x0) | x | GT(x) | forward base | forward ft | closing base | closing ft

Forward maps are in GT-native space (the eval metric's), closing maps in model space
(1024², the loss's). Label colours follow `run_eval.py::render_overlay` (nipy_spectral,
vmin 0, vmax = the dataset's largest trait id, nearest) so a trait keeps its hue across
every figure in the project.

`--select`:  spread          worst / median / best pair by the BASE arm's forward mIoU
             most_improved   largest ft − base delta on `--by`
             most_degraded   smallest (most negative) delta on `--by`
             worst_base      lowest base value on `--by`
             best_base       highest base value on `--by`
             random          seeded draw
`--by`:      fwd (forward mIoU, default) | cls (closing mIoU) | loss (the cycle loss;
             "improved" = loss DOWN)

Two summary figures per dataset ride along (`--figures`): the per-pair forward Δ vs
closing Δ scatter (the correlation the brief pre-registers) and per-trait forward Δ
bars with their bootstrap CIs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.datasets.beetle import PART_NAMES, Beetle  # noqa: E402
from occl.datasets.butterfly_major import TRAIT_VOCAB, ButterflyMajor  # noqa: E402
from occl.finetune.probe import parse_per_trait  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402

SELECTIONS = ("spread", "most_improved", "most_degraded", "worst_base", "best_base", "random")
BY = {"fwd": ("fwd_miou", +1), "cls": ("cls_miou", +1), "loss": ("loss", -1)}

# dataviz reference palette: categorical slots 1/2 for base/ft, diverging blue/orange
# around a neutral zero for deltas, text in ink — never a series colour.
C_BASE, C_FT, C_NEUTRAL, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#9a9992", "#0b0b0b", "#52514e"
C_POS, C_NEG = "#2a78d6", "#eb6834"


class Bundle:
    """A probe, per-dataset or merged S17 bundle, read uniformly."""

    def __init__(self, path, dataset: str | None):
        self.z = zipfile.ZipFile(path)
        self.record = json.loads(self.z.read("record.json"))
        names = set(self.z.namelist())
        if "datasets" in self.record:                     # merged
            if dataset is None:
                raise SystemExit(f"--dataset is required for a merged bundle "
                                 f"({sorted(self.record['datasets'])})")
            self.record = self.record["datasets"][dataset]
            self.prefix = f"{dataset}/"
        else:
            self.prefix = ""
        self.dataset = self.record["dataset"]
        if dataset is not None and self.dataset != dataset:
            raise SystemExit(f"bundle is {self.dataset!r}, not {dataset!r}")
        self.recipe = self.record.get("recipe") or {}
        if self.recipe.get("augmentation"):
            raise SystemExit("this bundle's recipe rotated x0 — the renderer shows the "
                             "unrotated reference and would mislabel y0; not implemented")
        if f"{self.prefix}pairs.csv" not in names:
            raise SystemExit(f"no {self.prefix}pairs.csv in the bundle")
        self.pairs = list(csv.DictReader(io.StringIO(
            self.z.read(f"{self.prefix}pairs.csv").decode())))
        self.deltas = (list(csv.DictReader(io.StringIO(
            self.z.read(f"{self.prefix}deltas.csv").decode())))
            if f"{self.prefix}deltas.csv" in names else None)
        arms = self.record.get("arm_order") or (self.record.get("probe") or {}).get("arm_order")
        self.base = self.record.get("base_arm") or next(
            a for a in arms if (self.record["arms"][a]["path"] is None))
        base_arms = self.record.get("base_arms") or [self.base]
        self.adapters = self.record.get("adapter_arms") or [a for a in arms if a not in base_arms]
        # S17d: an adapter trained under the soft sentinel pairs against the base variant
        # under the same gate (`base_of`); the hard-gate base otherwise
        self.base_of = dict(self.record.get("base_of") or {})
        # a base-only probe (the "before" picture alone): rows are ranked on the base
        # arm's own levels and the sheet has no ft columns
        self.base_only = not self.adapters

    def base_for(self, arm: str) -> str:
        return self.base_of.get(arm, self.base)

    def labelmap(self, rel: str) -> np.ndarray:
        import cv2

        buf = np.frombuffer(self.z.read(f"{self.prefix}{rel}"), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.int32)

    def rows(self, arm: str) -> dict[int, dict]:
        return {int(r["train_step"]): r for r in self.pairs if r["arm"] == arm}

    def delta_rows(self, arm: str) -> list[dict]:
        """One row per pair with base/ft/delta on every `--by` key — from deltas.csv when
        the combine wrote it, else joined here from pairs.csv (a bare probe bundle)."""
        if self.deltas is not None:
            return [r for r in self.deltas if r["arm"] == arm]
        b = self.rows(self.base if self.base_only else self.base_for(arm))
        a = b if self.base_only else self.rows(arm)
        out = []
        for s in sorted(set(b) & set(a)):
            row = {"arm": arm, "train_step": s, "group": b[s]["group"], "ref": b[s]["ref"],
                   "query": b[s]["query"]}
            for key in ("fwd_miou", "cls_miou", "loss"):
                bv, av = _f(b[s][key]), _f(a[s][key])
                row[f"base_{key}"], row[f"ft_{key}"] = bv, av
                # base-only: "delta" is the base level itself, so most_degraded == worst_base
                row[f"d_{key}"] = ((av - bv) if None not in (av, bv) else None) \
                    if not self.base_only else bv
            out.append(row)
        return out


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def select(rows: list[dict], how: str, by: str, n: int, seed: int = 0) -> list[dict]:
    key, sign = BY[by]
    scored = [r for r in rows if _f(r.get(f"d_{key}")) is not None
              and _f(r.get(f"base_{key}")) is not None]
    if not scored:
        raise SystemExit("no pair has both arms scored on that key")
    if how == "spread":
        s = sorted(scored, key=lambda r: _f(r["base_fwd_miou"]))
        if n >= len(s):
            return s
        idx = [int(round(i * (len(s) - 1) / (n - 1))) for i in range(n)] if n > 1 else [0]
        return [s[i] for i in idx]
    if how == "most_improved":
        return sorted(scored, key=lambda r: -sign * _f(r[f"d_{key}"]))[:n]
    if how == "most_degraded":
        return sorted(scored, key=lambda r: sign * _f(r[f"d_{key}"]))[:n]
    if how == "worst_base":
        return sorted(scored, key=lambda r: sign * _f(r[f"base_{key}"]))[:n]
    if how == "best_base":
        return sorted(scored, key=lambda r: -sign * _f(r[f"base_{key}"]))[:n]
    if how == "random":
        rng = np.random.default_rng(seed)
        return [scored[int(i)] for i in rng.permutation(len(scored))[:n]]
    raise SystemExit(f"unknown selection {how!r}")


def default_out_dir(bundle: "Bundle", arm: str) -> Path:
    """docs/figures/<stage>, plus /<arm> when the bundle carries more than one adapter.

    S17b and S17c both once landed in docs/figures/s17/ and overwrote the S17 sheets; the
    stage comes from the bundle's own record now, never from a constant."""
    stage = str(bundle.record.get("stage") or "probe")
    out = REPO / "docs" / "figures" / stage
    if not bundle.base_only and len(bundle.adapters) > 1:
        out = out / arm
    return out


def n_classes(ds) -> int:
    """The dataset's largest trait id — the shared colour scale's vmax."""
    return max(PART_NAMES) if ds.name == "beetle" else max(TRAIT_VOCAB.values())


def trait_names(ds) -> dict:
    return PART_NAMES if ds.name == "beetle" else {}


def _close_note(row) -> str:
    """S18: a closing leg on the ROTATED reference (`close_aug`) is drawn in its own
    frame — the y0 panel is the unrotated target, so say so in the title."""
    if row.get("close_aug") not in (None, ""):
        return f" (x0 rotated {float(row.get('close_angle') or 0):+.1f}°)"
    return ""


def render_sheet(bundle: Bundle, ds, arm: str, picked: list[dict], out_path: Path,
                 by: str, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_label = bundle.base if bundle.base_only else bundle.base_for(arm)
    base_rows = bundle.rows(base_label)
    ft_rows = base_rows if bundle.base_only else bundle.rows(arm)
    n_cls = n_classes(ds)
    lab_kw = {"cmap": "nipy_spectral", "vmin": 0, "vmax": n_cls, "interpolation": "nearest"}
    n_panels = 6 if bundle.base_only else 8
    fig, axes = plt.subplots(len(picked), n_panels, figsize=(3.75 * n_panels, 4.4 * len(picked)),
                             squeeze=False, gridspec_kw={"hspace": 0.5, "wspace": 0.06})
    for r, sel in enumerate(picked):
        s = int(sel["train_step"])
        b, a = base_rows[s], ft_rows[s]
        ref, query = b["ref"], b["query"]
        gt_ref, gt_q = ds.load_labelmap(ref), ds.load_labelmap(query)
        panels = [
            (ds.load_image(ref), None, f"x0 {ref}"),
            (None, gt_ref, "y0 — GT of x0"),
            (ds.load_image(query), None, f"x {query}"),
            (None, gt_q, "GT of x (unused by the loss)"),
            (None, bundle.labelmap(b["fwd_map"]) if b["fwd_map"] else None,
             f"forward {base_label} — {100 * _f(b['fwd_miou']):.1f}"),
            (None, bundle.labelmap(a["fwd_map"]) if a["fwd_map"] else None,
             f"forward {arm} — {100 * _f(a['fwd_miou']):.1f} "
             f"({100 * (_f(a['fwd_miou']) - _f(b['fwd_miou'])):+.1f})"),
            (None, bundle.labelmap(b["cls_map"]) if b["cls_map"] else None,
             f"closing {base_label} — {100 * _f(b['cls_miou']):.1f}, loss {_f(b['loss']):.3f}"
             + _close_note(b)),
            (None, bundle.labelmap(a["cls_map"]) if a["cls_map"] else None,
             f"closing {arm} — {100 * _f(a['cls_miou']):.1f}, loss {_f(a['loss']):.3f}"
             + _close_note(a)),
        ]
        if bundle.base_only:
            panels = [panels[i] for i in (0, 1, 2, 3, 4, 6)]
        for ax, (img, lab, ttl) in zip(axes[r], panels):
            if img is not None:
                ax.imshow(img)
            elif lab is not None:
                ax.imshow(lab, **lab_kw)
            else:
                ax.text(0.5, 0.5, "no map saved", ha="center", va="center", color=C_MUTED,
                        transform=ax.transAxes)
            ax.set_title(ttl, fontsize=9.5, color=C_INK)
            ax.axis("off")
        pt = lambda rw, k: "  ".join(f"t{t}:{100 * v:.0f}"                       # noqa: E731
                                     for t, v in sorted(parse_per_trait(rw[k]).items()))
        found = "".join("●" if c == "1" else "○" for c in b["fwd_found"].split(";"))
        bce_c = b.get("bce_per_channel", "")
        if bundle.base_only:
            caption = (f"step {s} · group {b['group']} · channels {b['class_ids']} · found {found}"
                       f"   forward per-trait: {pt(b, 'fwd_per_trait')}\n"
                       f"closing per-trait: {pt(b, 'cls_per_trait')}"
                       + (f"   ·   BCE per channel: {bce_c.replace(';', '  ')}" if bce_c else ""))
        else:
            caption = (f"step {s} · group {b['group']} · channels {b['class_ids']} · found(base) "
                       f"{found}   forward per-trait  {base_label}: {pt(b, 'fwd_per_trait')}   |   "
                       f"{arm}: {pt(a, 'fwd_per_trait')}\n"
                       f"closing per-trait  {base_label}: {pt(b, 'cls_per_trait')}   |   {arm}: "
                       f"{pt(a, 'cls_per_trait')}")
        axes[r][0].text(0, -0.1, caption, transform=axes[r][0].transAxes, fontsize=8.5,
                        va="top", color=C_MUTED)
    fig.suptitle(title, fontsize=12, color=C_INK, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_NEUTRAL)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def render_scatter(rows: list[dict], out_path: Path, arm: str, dataset: str, results: dict | None):
    """Per-pair forward Δ vs closing Δ — the correlation the brief pre-registers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(100 * _f(r["d_fwd_miou"]), 100 * _f(r["d_cls_miou"]), int(r["train_step"]))
           for r in rows if _f(r.get("d_fwd_miou")) is not None
           and _f(r.get("d_cls_miou")) is not None]
    if not pts:
        return None
    x, y, steps = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
                   [p[2] for p in pts])
    corr = (results or {}).get("corr_fwd_cls_delta")
    if corr is None and len(x) > 2 and x.std() and y.std():
        corr = float(np.corrcoef(x, y)[0, 1])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    _style(ax)
    ax.axhline(0, color=C_NEUTRAL, linewidth=1)
    ax.axvline(0, color=C_NEUTRAL, linewidth=1)
    ax.scatter(x, y, s=34, color=C_BASE, alpha=0.75, edgecolors="white", linewidths=0.8)
    order = np.argsort(x)
    for i in list(order[:3]) + list(order[-3:]):
        ax.annotate(f"step {steps[i]}", (x[i], y[i]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color=C_MUTED)
    ax.set_xlabel(f"forward mIoU Δ ({arm} − base), points", color=C_INK)
    ax.set_ylabel(f"closing mIoU Δ ({arm} − base), points", color=C_INK)
    ax.set_title(f"{dataset}: per training pair, forward vs closing leg — "
                 f"r = {corr:.2f}" if corr is not None else f"{dataset}: forward vs closing Δ",
                 fontsize=11, color=C_INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_trait_bars(results: dict, out_path: Path, arm: str, dataset: str, ds):
    """Per-trait forward Δ with the bootstrap CI — positive blue, negative orange, zero
    line neutral; the mean printed beside each bar (selective direct labels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = (results or {}).get("forward", {}).get("per_trait") or {}
    if not per:
        return None
    names = trait_names(ds)
    traits = sorted(per, key=int)
    vals = np.array([100 * per[t]["delta"]["mean_delta"] for t in traits])
    lo = np.array([100 * per[t]["delta"]["ci95"][0] for t in traits])
    hi = np.array([100 * per[t]["delta"]["ci95"][1] for t in traits])
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(traits) + 1.6))
    _style(ax)
    ypos = np.arange(len(traits))
    ax.axvline(0, color=C_NEUTRAL, linewidth=1)
    ax.barh(ypos, vals, height=0.55, color=[C_POS if v >= 0 else C_NEG for v in vals])
    ax.errorbar(vals, ypos, xerr=[vals - lo, hi - vals], fmt="none", ecolor=C_INK,
                elinewidth=1, capsize=2)
    span = max(float(hi.max() - min(lo.min(), 0.0)), 1e-6)
    for yv, v, h in zip(ypos, vals, hi):
        ax.text(max(h, v, 0.0) + 0.03 * span, yv, f"{v:+.2f}", va="center", ha="left",
                fontsize=8.5, color=C_INK)
    ax.set_xlim(min(lo.min(), 0.0) - 0.05 * span, hi.max() + 0.22 * span)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"t{t} {names.get(int(t), '')}".strip() + f"  (n={per[t]['n']})"
                        for t in traits], fontsize=9, color=C_INK)
    ax.invert_yaxis()
    ax.set_xlabel(f"forward IoU Δ per trait ({arm} − base), points, 95% CI", color=C_INK)
    ax.set_title(f"{dataset}: which traits the fine-tune moved on its own pairs",
                 fontsize=11, color=C_INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--cache-root", default=str(REPO / "oc-ccl-data" / "cache-png"))
    ap.add_argument("--arm", default=None, help="adapter arm label (default: the first)")
    ap.add_argument("--select", default="spread", choices=SELECTIONS)
    ap.add_argument("--by", default="fwd", choices=list(BY))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--group", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="default docs/figures/<bundle stage>[/<arm> when the bundle has "
                         "several adapters] — a later stage never overwrites an earlier one's sheets")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--figures", action="store_true",
                    help="also the scatter and per-trait bars for this dataset")
    args = ap.parse_args()

    bundle = Bundle(args.bundle, args.dataset)
    arm = args.arm or (bundle.base if bundle.base_only else bundle.adapters[0])
    if not bundle.base_only and arm not in bundle.adapters:
        raise SystemExit(f"arm {arm!r} not in {bundle.adapters}")
    if bundle.base_only and args.figures:
        print("  base-only bundle: no deltas, the summary figures are skipped")
        args.figures = False
    ds = {"beetle": Beetle, "butterfly_major": ButterflyMajor}[bundle.dataset](
        cache=ImageCache(args.cache_root))
    rows = bundle.delta_rows(arm)
    if args.group:
        rows = [r for r in rows if r["group"] == args.group]
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(bundle, arm)
    results = (bundle.record.get("results") or {}).get(arm)

    written = []
    if not args.no_sheets:
        picked = select(rows, args.select, args.by, args.n, args.seed)
        tag = f"{bundle.dataset}_{args.select}_{args.by}" + (f"_{args.group}" if args.group else "")
        title = (f"{bundle.dataset} — training pairs, {args.select.replace('_', ' ')} by "
                 f"{args.by} ({bundle.base + ' only' if bundle.base_only else arm + ' vs ' + bundle.base_for(arm)}); "
                 f"forward = x0→x scored on x's GT, closing = x→x0 scored on y0")
        written.append(render_sheet(bundle, ds, arm, picked, out_dir / f"{tag}.png",
                                    args.by, title))
        print("  pairs: " + ", ".join(
            f"step {r['train_step']} ({100 * _f(r['base_fwd_miou']):.1f}"
            f"→{100 * _f(r['ft_fwd_miou']):.1f})" for r in picked))
    if args.figures:
        p = render_scatter(rows, out_dir / f"{bundle.dataset}_fwd_vs_cls_delta.png", arm,
                           bundle.dataset, results)
        if p:
            written.append(p)
        p = render_trait_bars(results, out_dir / f"{bundle.dataset}_per_trait_delta.png", arm,
                              bundle.dataset, ds)
        if p:
            written.append(p)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
