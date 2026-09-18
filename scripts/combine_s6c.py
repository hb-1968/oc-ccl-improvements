#!/usr/bin/env python3
"""The rotation 2x2 on Butterfly Major at 5-shot — S6c.

    python scripts/combine_s6c.py --out s6c.zip \
        canonical_placement=a.zip orientation_diverse=b.zip

Rotation has been fixed twice on Major and never together:

* **query-side** — orientation TTA predicts with the supports as given and again rotated
  180 degrees, keeping whichever the model is more confident about (+4.88 at 1-shot);
* **support-side** — orientation-diverse supports draw the 5-set to span both orientation
  strata, so the memory bank holds an upright and a rotated exemplar (+4.01 at 5-shot).

Both land ~+4.5 on the same defect. Whether that is one effect found twice or two effects
that add is the question, and only the full 2x2 separates them: a single cell cannot tell
"TTA adds nothing" from "these two are redundant".

The no-TTA cells are already measured, so this fills the two TTA cells and reports the
**interaction** — the amount by which the combined gain exceeds the sum of the parts. A
negative interaction is the expected outcome and means the mechanisms overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.bundle import export_bundle  # noqa: E402

#: Measured 5-shot Major, PNG cache. `results/inbox/s6_butterfly_major.zip` and
#: `results/s6b_butterfly_major.zip`. Constants rather than inputs because these are
#: settled numbers and re-running them to fill a table would be pure cost.
NO_TTA = {"canonical_placement": (0.7178, 0.0015),      # S6
          "orientation_diverse": (0.7580, 0.0068)}      # S6b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", metavar="selection=bundle.zip")
    ap.add_argument("--out", default="s6c.zip")
    args = ap.parse_args()

    tta = {}
    for spec in args.cells:
        sel, _, path = spec.partition("=")
        with zipfile.ZipFile(path) as z:
            rec = json.loads(z.read("record.json"))
        got = rec.get("protocol", {}).get("support_selection")
        if got != sel:
            raise SystemExit(f"{Path(path).name} says support_selection={got!r}, "
                             f"labelled {sel!r} — refusing to file it in the wrong cell")
        if not str(rec.get("adapter", "")).endswith("+tta_rot180"):
            raise SystemExit(f"{Path(path).name} adapter is {rec.get('adapter')!r} — "
                             "this cell must be the TTA arm")
        tta[sel] = rec

    print("\nButterfly Major, 5-shot — rotation 2x2 (mIoU)\n")
    print(f"{'supports':>22s} {'no TTA':>14s} {'+ TTA':>14s} {'TTA effect':>12s}")
    print("-" * 66)
    rows = {}
    for sel in ("canonical_placement", "orientation_diverse"):
        base_m, base_s = NO_TTA[sel]
        if sel not in tta:
            print(f"{sel:>22s} {100 * base_m:7.2f}±{100 * base_s:5.2f} {'—':>14s}")
            continue
        r = tta[sel]["results"]
        eff = r["miou"] - base_m
        rows[sel] = {"no_tta": base_m, "no_tta_std": base_s,
                     "tta": r["miou"], "tta_std": r["std"], "tta_effect": eff,
                     "iou_binary": r.get("iou_binary"), "headroom": r.get("headroom")}
        print(f"{sel:>22s} {100 * base_m:7.2f}±{100 * base_s:5.2f} "
              f"{100 * r['miou']:7.2f}±{100 * r['std']:5.2f} {100 * eff:+12.2f}")

    out = {"stage": "s6c", "dataset": "butterfly_major",
           "protocol": {"shots": 5, "n_runs": 20}, "cells": rows}

    if len(rows) == 2:
        p, d = rows["canonical_placement"], rows["orientation_diverse"]
        supports_effect = d["no_tta"] - p["no_tta"]        # +4.01, support-side alone
        tta_effect = p["tta"] - p["no_tta"]                # query-side alone
        both = d["tta"] - p["no_tta"]                      # combined, over the same base
        interaction = both - (supports_effect + tta_effect)
        print(f"\nover the same base ({100 * p['no_tta']:.2f}):")
        print(f"  support-side alone (orientation-diverse) {100 * supports_effect:+6.2f}")
        print(f"  query-side alone   (TTA)                 {100 * tta_effect:+6.2f}")
        print(f"  both                                     {100 * both:+6.2f}")
        print(f"  INTERACTION (both - sum of parts)        {100 * interaction:+6.2f}")
        # 0.005 = half a POINT, because every value here is a fraction. Comparing the
        # fraction against `0.5` made a -4.66 point interaction read as "additive" — the
        # one field in this record a reader would take at face value without recomputing.
        if interaction < -0.005:
            verdict = ("redundant — the two fixes address the same defect and do not add; "
                       "rotation is handled and the remaining gap is elsewhere")
        elif interaction > 0.005:
            verdict = ("synergistic — combined exceeds the sum, so each was leaving "
                       "something the other catches")
        else:
            verdict = "additive — the two operate on separable parts of the defect"
        print(f"  -> {verdict}")
        best = max(("canonical_placement", "orientation_diverse"),
                   key=lambda k: rows[k]["tta"])
        best_v = max(rows[best]["tta"], d["no_tta"])
        print(f"\nbest Major arm now: {100 * best_v:.2f} (paper 83.4, gap "
              f"{100 * (0.834 - best_v):.2f})")
        if rows[best].get("headroom") is not None:
            print(f"  silhouette {100 * rows[best]['iou_binary']:.2f}, assignment headroom "
                  f"{100 * rows[best]['headroom']:+.2f} — the rest is mask quality")
        out["results"] = {"supports_effect": supports_effect, "tta_effect": tta_effect,
                          "both": both, "interaction": interaction, "verdict": verdict,
                          "best_miou": best_v, "gap_to_paper": 0.834 - best_v}

    dest = export_bundle("s6c", out, out=args.out)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
