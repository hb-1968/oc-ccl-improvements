#!/usr/bin/env python3
"""Stack S8d's per-dataset paired comparisons into one record.

    python scripts/combine_s8d.py --out s8d.zip beetle=s8d_beetle.zip \
        butterfly_major=s8d_butterfly_major.zip

Each input is a `combine_s8d_cells` record: the {base, OC-CCL} x {no TTA, TTA} square for
one dataset, each OC-CCL arm paired against its own base. This stacks the two datasets and
states the joint reading, because the interesting question is not any single delta but
whether they agree — across datasets, and across inference stacks.

**Deliberately not pooled.** The two datasets have different trait vocabularies, different
support-selection arms and different baselines, so a combined "mean delta" would be an
average of unlike things. `registry.comparability_key` already refuses to pair across them;
this script does not route around that.
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

#: S8c, 1-shot Beetle at (1e-5, 1 epoch) — the reference this stage extends to 5-shot.
ONE_SHOT_DELTA = {"beetle": -0.0005}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", metavar="name=bundle.zip")
    ap.add_argument("--stage", default="s8d",
                    help="the record must name the stage that produced it. Hardcoding\n"
                         "         s8d made every s8_5a bundle claim to be s8d, which is the\n"
                         "         same class of error as a record misstating its own epochs.")
    ap.add_argument("--out", default="s8d.zip")
    args = ap.parse_args()

    recs = {}
    for spec in args.datasets:
        name, _, path = spec.partition("=")
        with zipfile.ZipFile(path) as z:
            recs[name] = json.loads(z.read("record.json"))

    print(f"\n{'dataset':>18s} {'stack':>7s} {'base':>15s} {'OC-CCL':>15s} "
          f"{'paired delta (CI95)':>26s} {'seeds':>7s}")
    print("-" * 94)
    out_ds, verdicts = {}, []
    for name, r in recs.items():
        for tag, c in r["cells"].items():
            lo, hi = c["paired_ci95"]
            print(f"{name:>18s} {tag:>7s} {100 * c['base_miou']:7.2f}±{100 * c['base_std']:5.2f} "
                  f"{100 * c['occl_miou']:7.2f}±{100 * c['occl_std']:5.2f} "
                  f"{100 * c['paired_delta']:+8.2f} [{100 * lo:+.2f},{100 * hi:+.2f}]"
                  f"{'*' if c['significant'] else ' '} "
                  f"{c['n_seeds_improved']:>3d}/{c['n_seeds']}")
            verdicts.append(lo > 0)
        out_ds[name] = {"cells": r["cells"], "results": r["results"],
                        "best_cell": r["results"]["best_cell"],
                        "best_miou": r["results"]["best_miou"]}
    print("\n* = 95% CI excludes zero.")
    n_help = sum(1 for v in verdicts if v)
    print(f"\nbest cell per dataset:")
    for name, d in out_ds.items():
        print(f"  {name:>18s} {d['best_cell']:>10s} {100 * d['best_miou']:7.2f}")
    if n_help == len(verdicts) and verdicts:
        print("\n-> OC-CCL improves 5-shot in every cell tested.")
    elif n_help:
        print(f"\n-> OC-CCL improves {n_help} of {len(verdicts)} cells — its effect depends "
              "on the dataset or the inference stack, not a general gain.")
    else:
        print("\n-> OC-CCL improves NO cell at 5-shot, with or without TTA. Combined with "
              "S8c's 1-shot null\n   on Beetle, the published loss has no measurable effect "
              "on this harness at either shot\n   count or either inference stack. That is "
              "the baseline S9 modifies: no effect to preserve.")

    first = next(iter(recs.values()))
    record = {
        "stage": args.stage,
        # From the records, not hardcoded: the literal {"shots": 5} here made s8_5a — a
        # 1-shot stage — claim 5-shot, the same class of error as the --stage fix above.
        "protocol": first.get("protocol"),
        "finetune": first.get("finetune"),
        "occl": first.get("occl"),
        "datasets": out_ds,
        "results": {
            "n_datasets": len(out_ds),
            "n_cells": len(verdicts),
            "n_improved": n_help,
            "any_significant_gain": bool(n_help),
            # Per dataset and per stack, so a gain that exists only with TTA is visible
            # rather than averaged away.
            "deltas": {name: {tag: c["paired_delta"] for tag, c in r["cells"].items()}
                       for name, r in recs.items()},
            "interactions": {name: r["results"].get("interaction")
                             for name, r in recs.items()},
            "best": {name: {"cell": d["best_cell"], "miou": d["best_miou"]}
                     for name, d in out_ds.items()},
        },
    }
    dest = export_bundle(args.stage, record, out=args.out)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
