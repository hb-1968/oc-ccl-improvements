#!/usr/bin/env python3
"""One dataset x one model family — S14's five-pairing combine.

    python scripts/combine_s14.py --family sam3 --out s14sam3_beetle.zip \
        base=a.zip base_tta=b.zip stack_tta=c.zip \
        sam2_base=<banked s11 base> sam2_base_tta=<banked s11 base_tta> \
        sam2_pertrait_tta=<banked s11 occl_pertrait_tta>

    python scripts/combine_s14.py --family sec --out s14sec_beetle.zip \
        off=... forced=... off_tta=... stack_tta=... sam2_base=... \
        sam2_base_tta=... sam2_pertrait_tta=...

Within-family pairings use `registry.compare` as every stage since S8d has; the
CROSS-FAMILY rows are the comparison `comparability_key` exists to refuse, and they
go through the one sanctioned relaxation — `compare(..., ignore_fields=
("model.family", "model.image_size"))` — which pins `support_set_hash` equality (the
draws are model-independent by construction) and embeds the ignored fields and both
their values in every row it produces. A cross-family number that cannot say it is
one cannot appear in this record.

Pairings emitted:
  1. base vs sam2_base            (cross) — did the model upgrade pay, raw
  2. base_tta vs base             (within) — TTA's value on the new family
  3. stack_tta vs base_tta        (within) — the stack's contribution (toggle)
  4. stack_tta vs sam2_pertrait_tta (cross) — THE DEPLOYMENT VERDICT
  5. sec only: forced vs off      (within) — the concept delta
plus base_tta vs sam2_base_tta (cross) for the TTA-stack base read. For sec, "base"
duties are performed by the `off` cell (`forced` is a diagnostic arm, never a
comparator base). Every cell lifts oracle / argmax_no_background / best3 /
rotated_frac exactly as S13's combiner does.
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

CROSS = ("model.family", "model.image_size")

EXPECTED = {
    # arch_tta = the architecture-adapted arm: ported scopes + `transformer`
    # (sam3's memory attention) — a strict scope superset, so arch_vs_stack is
    # attributable to that one addition.
    "sam3": {"base", "base_tta", "stack_tta", "arch_tta",
             "sam2_base", "sam2_base_tta", "sam2_pertrait_tta"},
    "sec": {"off", "forced", "off_tta", "stack_tta",
            "sam2_base", "sam2_base_tta", "sam2_pertrait_tta"},
}


def _lift(rec, block, key):
    vals = [r[block][key] for r in rec.get("runs", [])
            if (r.get(block) or {}).get(key) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _tta_frac(rec):
    vals = [r["tta"]["rotated_chosen"] / r["tta"]["episodes"]
            for r in rec.get("runs", []) if (r.get("tta") or {}).get("episodes")]
    return float(sum(vals) / len(vals)) if vals else None


def _cell(rec):
    return {
        "miou": rec["results"]["miou"], "std": rec["results"]["std"],
        "miou_independent": _lift(rec, "miou_variants", "independent"),
        "miou_best3": _lift(rec, "miou_boundary", "independent_best3"),
        "miou_oracle": _lift(rec, "miou_variants", "oracle"),
        "miou_argmax_no_background": _lift(rec, "miou_variants",
                                           "argmax_no_background"),
        "tta_rotated_frac": _tta_frac(rec),
        "iou_binary": rec["results"].get("iou_binary"),
    }


def _pair(t, b, name, cross=False, ignore=()):
    kw = ({"ignore_fields": CROSS} if cross
          else {"ignore_fields": tuple(ignore)} if ignore else {})
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
        "delta_best3": cmp3["mean_delta"] if cmp3 else None,
        "ci95_best3": cmp3["ci95"] if cmp3 else None,
        "significant_best3": cmp3["significant"] if cmp3 else None,
        "cross_family": cross,
        "ignored_fields": cmp.get("ignored_fields"),
    }
    star = "*" if cmp["significant"] else ""
    b3 = (f" | best3 {100 * cmp3['mean_delta']:+.2f}"
          f" [{100 * cmp3['ci95'][0]:+.2f},{100 * cmp3['ci95'][1]:+.2f}]"
          f"{'*' if cmp3['significant'] else ''}" if cmp3 else "")
    tag = " (CROSS-FAMILY)" if cross else ""
    print(f"{name:>28s}: {100 * cmp['mean_delta']:+.2f} "
          f"[{100 * cmp['ci95'][0]:+.2f},{100 * cmp['ci95'][1]:+.2f}]{star}{b3}{tag}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", metavar="label=bundle.zip")
    ap.add_argument("--family", required=True, choices=["sam3", "sec"])
    ap.add_argument("--stage", default=None,
                    help="default s14<family>")
    ap.add_argument("--merge", action="store_true",
                    help="labels are DATASET names of per-dataset s14 combines; "
                         "emit the stage bundle {datasets: {...}}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stage = args.stage or f"s14{args.family}"

    if args.merge:
        per_ds = {}
        for spec in args.cells:
            label, _, path = spec.partition("=")
            with zipfile.ZipFile(path) as z:
                per_ds[label] = json.loads(z.read("record.json"))
        for ds, r in per_ds.items():
            if r.get("family") != args.family:
                raise SystemExit(f"{ds}: family {r.get('family')!r} != "
                                 f"{args.family!r}")
        record = {
            "stage": stage, "family": args.family, "datasets": per_ds,
            "results": {ds: {"headline": r["results"]["headline"],
                             "headline_plain": r["results"]["headline_plain"]}
                        for ds, r in per_ds.items()},
        }
        dest = export_bundle(stage, record, out=args.out)
        print(f"wrote {dest}")
        return 0

    recs = {}
    for spec in args.cells:
        label, _, path = spec.partition("=")
        with zipfile.ZipFile(path) as z:
            recs[label] = json.loads(z.read("record.json"))
    want = EXPECTED[args.family]
    if set(recs) != want:
        raise SystemExit(f"cells {sorted(recs)} != expected {sorted(want)}")

    # label sanity: sam2_* cells must be family sam2; the family cells its own
    for k, r in recs.items():
        fam = (r.get("model") or {}).get("family")
        want_fam = "sam2" if k.startswith("sam2_") else args.family
        if fam != want_fam:
            raise SystemExit(f"cell {k!r} has model.family {fam!r}, "
                             f"expected {want_fam!r}")
        wants_tta = k.endswith("_tta")
        has_tta = str(r.get("adapter", "")).endswith("+tta_rot180")
        if wants_tta != has_tta:
            raise SystemExit(f"cell {k!r} adapter {r.get('adapter')!r} does not "
                             f"match its _tta suffix")

    base = recs["base" if args.family == "sam3" else "off"]
    base_tta = recs["base_tta" if args.family == "sam3" else "off_tta"]
    ds = base["dataset"]
    print(f"\n{ds} — {args.family} vs the SAM 2 ledger\n")

    pairings = {
        "base_vs_sam2_base": _pair(base, recs["sam2_base"],
                                   "base vs sam2_base", cross=True),
        "base_tta_vs_sam2_base_tta": _pair(base_tta, recs["sam2_base_tta"],
                                           "base_tta vs sam2_base_tta", cross=True),
        # model.tta sits in the comparability key, so the TTA-toggle pairing
        # declares it (whitelisted, embedded in the result) — the first butterfly
        # combine died right here on the bare compare's refusal.
        "tta_vs_base": _pair(base_tta, base, "base_tta vs base",
                             ignore=("model.tta",)),
        "stack_vs_base_tta": _pair(recs["stack_tta"], base_tta,
                                   "stack_tta vs base_tta"),
        "stack_vs_sam2_crown": _pair(recs["stack_tta"], recs["sam2_pertrait_tta"],
                                     "stack_tta vs sam2_pertrait_tta", cross=True),
    }
    if args.family == "sec":
        pairings["forced_vs_off"] = _pair(recs["forced"], recs["off"],
                                          "forced vs off")
    if args.family == "sam3":
        # the architecture-adaptation delta (within-family, both TTA, scope
        # superset) and its own deployment read against the SAM 2 crown
        pairings["arch_vs_stack"] = _pair(recs["arch_tta"], recs["stack_tta"],
                                          "arch_tta vs stack_tta")
        pairings["arch_vs_sam2_crown"] = _pair(
            recs["arch_tta"], recs["sam2_pertrait_tta"],
            "arch_tta vs sam2_pertrait_tta", cross=True)

    cells = {k: _cell(r) for k, r in recs.items()}
    fam_model = (recs["stack_tta"].get("model") or {})
    print(f"\noracle ceiling ({args.family} base): "
          f"{100 * (cells['base' if args.family == 'sam3' else 'off']['miou_oracle'] or 0):.2f} "
          f"vs argmax {100 * cells['base' if args.family == 'sam3' else 'off']['miou']:.2f}")

    record = {
        "stage": stage, "dataset": ds, "family": args.family,
        "protocol": base["protocol"],
        "model": fam_model,
        "finetune": recs["stack_tta"].get("finetune"),
        "occl": recs["stack_tta"].get("occl"),
        "sec": recs.get("forced", {}).get("model") if args.family == "sec" else None,
        "cells": cells,
        "pairings": pairings,
        "results": {
            "headline": pairings["stack_vs_sam2_crown"]["delta_best3"],
            "headline_plain": pairings["stack_vs_sam2_crown"]["delta"],
            "best_cell": max(recs, key=lambda k: recs[k]["results"]["miou"]),
        },
    }
    dest = export_bundle(stage, record, out=args.out)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
