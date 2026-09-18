#!/usr/bin/env python3
"""Pair two run_eval cell bundles per seed — an ad-hoc `registry.compare` from the shell.

    python scripts/pair_cells.py comp=s17b_beetle_fwdonly_objness_excl_tta.zip \
        excl=s17b_beetle_excl_tta.zip [--ignore model.tta]

The first cell is the treatment, the second the baseline; the labels are free text.

The combines pair every arm against base and the control; any other pairing (S17b's
deployment read, `fwdonly_objness_excl` vs `excl`) is this, on the per-cell bundles the
cluster keeps under $PROJECT/results/<stage>_<ds>_<cell>.zip.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs=2, metavar="label=bundle.zip",
                    help="treatment first, baseline second; the labels are free text "
                         "(they name the two sides in the output)")
    ap.add_argument("--ignore", nargs="*", default=(),
                    help="comparability fields to relax (model.tta for a TTA-vs-plain pair)")
    args = ap.parse_args()
    labels, recs = [], []
    for spec in args.cells:
        label, _, path = spec.partition("=")
        if not path:                       # a bare path: label it by its file name
            label, path = Path(label).stem, label
        with zipfile.ZipFile(path) as z:
            recs.append(json.loads(z.read("record.json")))
        labels.append(label)
    t, b = recs
    print(f"{labels[0]} vs {labels[1]}")
    kw = {"ignore_fields": tuple(args.ignore)} if args.ignore else {}
    for metric in ("miou", "miou_boundary.independent_best3"):
        try:
            c = registry.compare(t, b, metric=metric, **kw)
        except ValueError as e:
            print(f"{metric}: cannot pair: {e}")
            continue
        print(f"{metric:>32s}: {100 * t['results']['miou']:.2f} vs {100 * b['results']['miou']:.2f} "
              f"| paired {100 * c['mean_delta']:+.2f} [{100 * c['ci95'][0]:+.2f},{100 * c['ci95'][1]:+.2f}]"
              f"{'*' if c['significant'] else ''} {c['n_seeds_improved']}/{c['n_seeds']} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
