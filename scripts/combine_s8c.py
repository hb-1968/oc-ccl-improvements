#!/usr/bin/env python3
"""Merge S8c's per-arm bundles into one record with paired deltas.

    python scripts/combine_s8c.py --out s8c.zip base=a.zip lora_lr1e-05=b.zip ...

The arms differ in **weights and nothing else** — same dataset, protocol, seeds, supports,
pixel pipeline and metric — so `registry.compare` pairs them per support seed and
bootstraps the mean difference. Support difficulty is the dominant variance component on
Beetle (S4a's 20 runs span 35.5-64.9 mIoU on the same model), and it cancels exactly in a
paired comparison. That is what makes a +2 to +3 point effect resolvable at n=20 at all;
comparing the arms' means alone would drown it.

`compare()` refuses records with differing comparability keys, which is the check that the
arms really are otherwise identical. `finetune` is deliberately outside
`COMPARABILITY_FIELDS` (see `registry.py`) precisely so a base and a fine-tuned run CAN
pair — do not "fix" that.
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


def load(path: str) -> dict:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("record.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", metavar="label=bundle.zip")
    ap.add_argument("--out", default="s8c.zip")
    ap.add_argument("--baseline", default="base")
    args = ap.parse_args()

    recs = {}
    for spec in args.arms:
        if "=" not in spec:
            raise SystemExit(f"expected label=bundle.zip, got {spec!r}")
        label, path = spec.split("=", 1)
        recs[label] = load(path)
    if args.baseline not in recs:
        raise SystemExit(f"no {args.baseline!r} arm among {sorted(recs)}")

    base = recs[args.baseline]
    print(f"baseline {args.baseline}: {100 * base['results']['miou']:.2f} "
          f"+- {100 * base['results']['std']:.2f}  "
          f"(key {registry.comparability_key(base)})")

    out: dict = {
        "stage": "s8c",
        "dataset": base["dataset"],
        "protocol": base["protocol"],
        "metric": base.get("metric"),
        "data": base.get("data"),
        "model": base.get("model"),
        "baseline": args.baseline,
        "arms": {},
        "results": {},
    }

    deltas = []
    for label, rec in recs.items():
        entry = {
            "miou": rec["results"]["miou"],
            "std": rec["results"]["std"],
            "comparability_key": registry.comparability_key(rec),
            "track": registry.classify(rec)[1],
            "deviations": registry.classify(rec)[0],
            "finetune": rec.get("finetune"),
            "occl": rec.get("occl"),
        }
        if label != args.baseline:
            try:
                cmp = registry.compare(rec, base)
            except ValueError as e:
                # Loud, not skipped. An unpaired arm is the whole stage failing, and a
                # record that quietly reports means instead would read as a result.
                raise SystemExit(
                    f"arm {label!r} cannot be paired against {args.baseline!r}: {e}"
                ) from e
            entry |= {"paired_delta": cmp["mean_delta"],
                      "paired_ci95": cmp["ci95"],
                      "significant": cmp["significant"],
                      "n_paired_seeds": cmp["n_seeds"],
                      "n_seeds_improved": cmp["n_seeds_improved"],
                      "delta_vs_base_mean": rec["results"]["miou"] - base["results"]["miou"]}
            deltas.append((label, entry))
        out["arms"][label] = entry

    print(f"\n{'arm':>14s} {'mIoU':>15s} {'paired delta':>26s} {'track':>10s}")
    print("-" * 70)
    print(f"{args.baseline:>14s} {100 * base['results']['miou']:7.2f}"
          f"+-{100 * base['results']['std']:5.2f} {'—':>26s} "
          f"{out['arms'][args.baseline]['track']:>10s}")
    for label, e in deltas:
        lo, hi = e["paired_ci95"]
        mark = "" if lo < 0 < hi else " *"
        print(f"{label:>14s} {100 * e['miou']:7.2f}+-{100 * e['std']:5.2f} "
              f"{100 * e['paired_delta']:+8.2f} [{100 * lo:+.2f},{100 * hi:+.2f}]{mark:>2s} "
              f"{e['track']:>10s}")
    print("\n* = 95% CI excludes zero.")

    # The gate reads the BEST arm, but every arm is recorded — reporting only the winner
    # of a sweep overstates the result, the same reason all three support-selection arms
    # are always reported.
    if deltas:
        best = max(deltas, key=lambda kv: kv[1]["paired_delta"])
        out["results"] = {
            "best_arm": best[0],
            "paired_delta": best[1]["paired_delta"],
            "paired_ci95": best[1]["paired_ci95"],
            "miou": best[1]["miou"],
            "std": best[1]["std"],
            "baseline_miou": base["results"]["miou"],
            "baseline_std": base["results"]["std"],
        }
        print(f"\nbest arm: {best[0]} at {100 * best[1]['paired_delta']:+.2f} pts")

    dest = export_bundle("s8c", out, out=args.out)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
