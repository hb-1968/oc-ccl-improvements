#!/usr/bin/env python3
"""S16: TTA as an implicit second shot — one dataset's {1-shot+TTA, 2-shot plain} pair.

    python scripts/combine_s16.py --out s16_beetle.zip s1_tta=a.zip s2=b.zip

    python scripts/combine_s16.py --merge --out s16.zip \
        butterfly_major=s16_butterfly_major.zip beetle=s16_beetle.zip

The premise: orientation TTA is analogous to an IMPLICIT 2-shot — the rotated support is
a second reference (same specimen, second orientation) carried outside the memory bank
and selected by a confidence rule over the logits. Explicit 2-shot puts a second real
specimen INSIDE the bank (one 3-frame video ``{x0, x1, xq}``, the paper's own multi-shot
construction) and lets memory attention do the selecting. Same tracker (base SAM 2, no
adapters), same canonical protocol, the same 20 seeds; in SST's own currency the
implicit reference is free and the explicit one costs a labelled mask.

One paired row, under the plain metric and the paper's best-of-3:

  second_shot_vs_tta   s2 vs s1_tta   THE HEADLINE: implicit or explicit second reference

The row crosses `protocol.shots` AND `protocol.support_set_hash`, so it goes through the
one sanctioned route — `registry.compare(nested_shots=True)` — which verifies from every
per-run record that run i's 1-shot support is the first of its 2-shot pair before it
masks anything, and embeds that proof in the row (it also declares the `model.tta` mask
it crosses). A combine that cannot prove the nesting refuses rather than shipping a
delta whose pairing is a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl import registry  # noqa: E402
from occl.bundle import export_bundle  # noqa: E402

#: label -> (shots, tta): the implicit second reference and the explicit one.
CELLS = {"s1_tta": (1, True), "s2": (2, False)}
HEADLINE = "second_shot_vs_tta"


def _lift(rec, block, key):
    vals = [r[block][key] for r in rec.get("runs", [])
            if (r.get(block) or {}).get(key) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _tta_frac(rec):
    vals = [r["tta"]["rotated_chosen"] / r["tta"]["episodes"]
            for r in rec.get("runs", []) if (r.get("tta") or {}).get("episodes")]
    return float(sum(vals) / len(vals)) if vals else None


def _cell(label, rec):
    shots, tta = CELLS[label]
    return {
        "shots": shots, "tta": tta,
        "miou": rec["results"]["miou"], "std": rec["results"]["std"],
        "n_seeds": (rec.get("protocol") or {}).get("n_runs"),
        "support_set_hash": (rec.get("protocol") or {}).get("support_set_hash"),
        "miou_independent": _lift(rec, "miou_variants", "independent"),
        "miou_best3": _lift(rec, "miou_boundary", "independent_best3"),
        "miou_oracle": _lift(rec, "miou_variants", "oracle"),
        "miou_argmax_no_background": _lift(rec, "miou_variants",
                                           "argmax_no_background"),
        "tta_rotated_frac": _tta_frac(rec),
        "iou_binary": rec["results"].get("iou_binary"),
    }


def _pair(t, b, name):
    kw = {"ignore_fields": ("model.tta",), "nested_shots": True}
    try:
        cmp = registry.compare(t, b, **kw)
    except ValueError as e:
        raise SystemExit(f"{name}: cannot pair: {e}") from e
    try:
        cmp3 = registry.compare(t, b, metric="miou_boundary.independent_best3", **kw)
    except ValueError:
        cmp3 = None
    row = {
        "delta": cmp["mean_delta"], "ci95": cmp["ci95"],
        "significant": cmp["significant"], "n_seeds": cmp["n_seeds"],
        "n_seeds_improved": cmp["n_seeds_improved"],
        "paired_std": cmp["paired_std"],
        "delta_best3": cmp3["mean_delta"] if cmp3 else None,
        "ci95_best3": cmp3["ci95"] if cmp3 else None,
        "significant_best3": cmp3["significant"] if cmp3 else None,
        "ignored_fields": cmp.get("ignored_fields"),
        "nested_shots": cmp.get("nested_shots"),
    }
    star = "*" if cmp["significant"] else ""
    b3 = (f" | best3 {100 * cmp3['mean_delta']:+.2f}"
          f" [{100 * cmp3['ci95'][0]:+.2f},{100 * cmp3['ci95'][1]:+.2f}]"
          f"{'*' if cmp3['significant'] else ''}" if cmp3 else "")
    print(f"{name:>20s}: {100 * cmp['mean_delta']:+.2f} "
          f"[{100 * cmp['ci95'][0]:+.2f},{100 * cmp['ci95'][1]:+.2f}]{star}"
          f"  {cmp['n_seeds_improved']}/{cmp['n_seeds']} seeds{b3}"
          f"  (nested: {row['nested_shots']['runs_checked']} runs verified)")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", metavar="label=bundle.zip")
    ap.add_argument("--stage", default="s16")
    ap.add_argument("--merge", action="store_true",
                    help="labels are DATASET names of per-dataset s16 combines; "
                         "emit the stage bundle {datasets: {...}}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.merge:
        per_ds = {}
        for spec in args.cells:
            label, _, path = spec.partition("=")
            with zipfile.ZipFile(path) as z:
                per_ds[label] = json.loads(z.read("record.json"))
        for ds, r in per_ds.items():
            if r.get("dataset") != ds:
                raise SystemExit(f"{ds}: record is for {r.get('dataset')!r}")
        record = {
            "stage": args.stage, "datasets": per_ds,
            "results": {ds: {"headline": r["results"]["headline"],
                             "headline_plain": r["results"]["headline_plain"]}
                        for ds, r in per_ds.items()},
        }
        dest = export_bundle(args.stage, record, out=args.out)
        print(f"wrote {dest}")
        return 0

    recs = {}
    for spec in args.cells:
        label, _, path = spec.partition("=")
        with zipfile.ZipFile(path) as z:
            recs[label] = json.loads(z.read("record.json"))
    if set(recs) != set(CELLS):
        raise SystemExit(f"cells {sorted(recs)} != expected {sorted(CELLS)}")

    # Check the RECORDS against the labels, never the labels alone: a plain cell filed
    # as _tta, or a 1-shot cell filed as s2, would silently become the headline.
    ds = recs["s2"]["dataset"]
    sel = recs["s2"]["protocol"]["support_selection"]
    for k, r in recs.items():
        shots, tta = CELLS[k]
        got_shots = (r.get("protocol") or {}).get("shots")
        if got_shots != shots:
            raise SystemExit(f"cell {k!r} has protocol.shots {got_shots!r}, expected {shots}")
        has_tta = str(r.get("adapter", "")).endswith("+tta_rot180")
        if has_tta != tta:
            raise SystemExit(f"cell {k!r} adapter {r.get('adapter')!r} does not match "
                             f"its {'_tta' if tta else 'plain'} label")
        if r.get("dataset") != ds:
            raise SystemExit(f"cell {k!r} is {r.get('dataset')!r}, not {ds!r}")
        if r["protocol"].get("support_selection") != sel:
            raise SystemExit(f"cell {k!r} uses {r['protocol'].get('support_selection')!r} "
                             f"supports; both cells must share {sel!r}")
        # Base weights only. An adapter on either cell changes the question from "which
        # second reference" to "which second reference on which weights".
        if r.get("finetune") is not None:
            raise SystemExit(f"cell {k!r} carries a finetune block — s16 is base weights")

    print(f"\n{ds}, {sel} — TTA as an implicit second shot\n")
    for k in CELLS:
        r = recs[k]["results"]
        print(f"  {k:>7s}: {100 * r['miou']:6.2f} ± {100 * r['std']:5.2f}")
    print()

    pairings = {HEADLINE: _pair(recs["s2"], recs["s1_tta"], HEADLINE)}

    h = pairings[HEADLINE]
    winner = ("the EXPLICIT second support" if h["delta"] > 0
              else "the IMPLICIT one (TTA)")
    print(f"\nHEADLINE 2-shot plain vs 1-shot TTA: {100 * h['delta']:+.2f} "
          f"[{100 * h['ci95'][0]:+.2f},{100 * h['ci95'][1]:+.2f}]"
          f"{'*' if h['significant'] else ''} -> {winner} wins on {ds}"
          + ("" if h["significant"] else " (not significant)"))

    best = max(recs, key=lambda k: recs[k]["results"]["miou"])
    record = {
        "stage": args.stage, "dataset": ds,
        # The shared protocol; per-cell shots live in `cells`, and the hashes that
        # legitimately differ across shot counts travel there too.
        "protocol": {k: v for k, v in recs["s2"]["protocol"].items()
                     if k not in ("shots", "support_set_hash")},
        "model": recs["s2"].get("model"),
        "cells": {k: _cell(k, r) for k, r in recs.items()},
        "pairings": pairings,
        "results": {
            "headline": h["delta_best3"],
            "headline_plain": h["delta"],
            "headline_significant": h["significant"],
            "best_cell": best,
            "best_miou": recs[best]["results"]["miou"],
            "miou": {k: r["results"]["miou"] for k, r in recs.items()},
        },
    }
    dest = export_bundle(args.stage, record, out=args.out)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
