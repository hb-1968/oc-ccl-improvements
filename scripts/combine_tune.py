#!/usr/bin/env python3
"""Pick (learning rate, epochs) from the per-fold grids — S8c-tune.

    python scripts/combine_tune.py --out s8c_tune.zip tune_f0.zip tune_f1.zip ...

The choice is the grid point with the best **mean validation mIoU across folds**, never the
best in any single fold: with 24 validation images per fold, a per-fold argmax selects that
fold's noise.

**The winner is not taken at face value.** Every point within one standard error of the
argmax is statistically indistinguishable from it, and with a grid of ~50 points the argmax
is very likely to be a lucky draw from that set — the more points, the more likely. So the
tie-break is explicit and stated in advance: among points inside one SEM of the best,
prefer **fewer epochs** first (less compute, less drift from the base weights), then the
**lower learning rate** (same reason). That is a bias toward the conservative end of a
plateau, not toward the largest number.

If epoch 0 survives the tie-break, no configuration beat the base model on held-out data.
That is a result about OC-CCL, not a tuning failure, and it is reported as one.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.bundle import export_bundle  # noqa: E402


def key_of(point: dict) -> tuple:
    """(lr, epoch). Epoch 0 is the base model and has no learning rate."""
    return (None if point["epoch"] == 0 else point["lr"], point["epoch"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folds", nargs="+")
    ap.add_argument("--out", default="s8c_tune.zip")
    args = ap.parse_args()

    recs = []
    for p in args.folds:
        with zipfile.ZipFile(p) as z:
            recs.append(json.loads(z.read("record.json")))

    grids = {tuple(r["finetune"]["lrs"]) for r in recs}
    if len(grids) != 1:
        raise SystemExit(f"folds disagree on the lr grid: {sorted(grids)}")

    by_point: dict[tuple, list[float]] = {}
    for r in recs:
        for c in r["curve"]:
            by_point.setdefault(key_of(c), []).append(c["miou"])
    n_folds = len(recs)
    ragged = [k for k, v in by_point.items() if len(v) != n_folds]
    if ragged:
        raise SystemExit(f"{len(ragged)} grid point(s) missing from some folds, e.g. "
                         f"{ragged[:4]} — an incomplete point would be compared on fewer "
                         "folds than the rest")

    # PAIRED against the base, fold by fold. Every point is scored on the same five folds
    # and fold difficulty dominates — the per-fold base spans 58.2 to 62.1, a 3.9-point
    # spread, against between-point differences of ~0.2. Comparing means and their
    # independent SEMs throws that away; differencing within a fold cancels it exactly.
    # Same argument as `registry.compare`, one level down: folds here, support seeds there.
    # Measured on the first real grid, pairing tightened the uncertainty 0.65 -> 0.37.
    base_per_fold = by_point[(None, 0)]
    base_mean = st.mean(base_per_fold)

    def paired(k):
        """(mean gain vs base, sem of that gain). Zero for the base itself."""
        d = [a - b for a, b in zip(by_point[k], base_per_fold)]
        if k == (None, 0):
            return 0.0, 0.0
        return st.mean(d), (st.stdev(d) / len(d) ** 0.5) if len(d) > 1 else 0.0

    gains = {k: paired(k) for k in by_point}
    stats = {k: (st.mean(v), (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0)
             for k, v in by_point.items()}

    print(f"\n{n_folds}-fold CV over {len(stats)} grid points, "
          f"{recs[0]['cv']['val_size']} val images per fold")
    print(f"lr grid: {[f'{x:.2e}' for x in recs[0]['finetune']['lrs']]}, "
          f"epochs 1..{recs[0]['finetune']['epochs']}\n")

    # mean mIoU as a grid: rows = lr, cols = epoch
    lrs = sorted({k[0] for k in stats if k[0] is not None})
    eps = sorted({k[1] for k in stats if k[1] != 0})
    print(f"paired gain vs base ({100 * base_mean:.2f}), in points")
    print("      lr \\ ep " + "".join(f"{e:>7d}" for e in eps))
    for lr in lrs:
        cells = "".join(
            f"{100 * gains[(lr, e)][0]:>7.2f}" if (lr, e) in gains else "      ·"
            for e in eps)
        print(f"{lr:12.2e} {cells}")

    # Selection on the PAIRED gain, not the raw mean.
    best = max(gains, key=lambda k: gains[k][0])
    best_gain, best_sem = gains[best]
    within = [k for k, (m, _) in gains.items() if m >= best_gain - best_sem]
    # Fewer epochs first, then lower lr. Epoch 0 (base) sorts first on both counts.
    choice = min(within, key=lambda k: (k[1], k[0] if k[0] is not None else -1))
    flat = choice != best

    n_positive = sum(1 for k, (m, s_) in gains.items()
                     if k != (None, 0) and m - s_ > 0)
    print(f"\npoints whose paired gain clears its own sem: {n_positive} of {len(gains) - 1}"
          "  <- does OC-CCL help at all")
    print(f"argmax: lr={best[0]}, epoch {best[1]} -> {100 * best_gain:+.2f} "
          f"+- {100 * best_sem:.2f} paired")
    print(f"within one paired sem of it: {len(within)} of {len(gains)} points"
          "  <- can we tell WHICH hyperparameters")
    if flat:
        print(f"-> plateau; taking the cheapest point inside it: lr={choice[0]}, "
              f"epoch {choice[1]} at {100 * stats[choice][0]:.2f} "
              f"({100 * (stats[choice][0] - base_mean):+.2f} vs base)")
    else:
        print(f"-> choosing lr={choice[0]}, epoch {choice[1]}")
    if choice[1] == 0:
        print("-> NOTE the BASE model survived the tie-break: no (lr, epoch) beat it on "
              "held-out data.\n   That is a finding about OC-CCL, not a tuning failure.")

    record = {
        "stage": "s8c_tune",
        "dataset": recs[0]["dataset"],
        "cv": {"k": recs[0]["cv"]["k"], "n_folds": n_folds,
               "val_size": recs[0]["cv"]["val_size"],
               "folds": [r["cv"]["fold"] for r in recs]},
        "finetune": recs[0]["finetune"],
        "protocol": recs[0]["protocol"],
        "grid": [{"lr": k[0], "epoch": k[1], "mean_miou": stats[k][0], "sem": stats[k][1],
                  "paired_gain": gains[k][0], "paired_sem": gains[k][1],
                  "per_fold": by_point[k]} for k in sorted(
                      gains, key=lambda k: (k[1], k[0] if k[0] is not None else -1))],
        "results": {
            "chosen_lr": choice[0], "chosen_epoch": choice[1],
            "argmax_lr": best[0], "argmax_epoch": best[1],
            "curve_is_flat": flat,
            "n_within_one_sem": len(within),
            "n_grid_points": len(stats),
            "chosen_miou": stats[choice][0],
            "base_miou": base_mean,
            "gain_over_base": gains[choice][0],
            "gain_sem": gains[choice][1],
            "n_points_helping": n_positive,
            "argmax_gain": best_gain,
        },
    }
    dest = export_bundle("s8c_tune", record, out=args.out)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
