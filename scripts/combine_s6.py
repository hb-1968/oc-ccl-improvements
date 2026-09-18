#!/usr/bin/env python3
"""Combine S6's two per-dataset bundles into the one bundle the gate expects.

    python scripts/combine_s6.py

S6 runs one dataset per Colab session, so it produces two bundles. The gate is written
against a *merged* record — `results.{beetle,butterfly_major}.{miou,std,miou_1shot,
per_group}` — because its load-bearing check is cross-dataset: **5-shot > 1-shot on
both**, which is the signal that conditioning frames are not being silently dropped.

`miou_1shot` is pulled from each dataset's *matching* 1-shot arm, not from whatever
1-shot bundle happens to be lying around. Support selection must agree or the comparison
is meaningless: beetle 5-shot uses `canonical_balanced` (S4c) and butterfly uses
`canonical_placement` (S5b), so those are the bundles read, and the script refuses if the
selections do not line up.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.bundle import export_bundle  # noqa: E402

#: dataset -> (5-shot bundle, its 1-shot counterpart with the SAME support selection)
SOURCES = {
    "beetle": ("s6_beetle.zip", ["s4c.zip"]),
    "butterfly_major": ("s6_butterfly_major.zip", ["s5b.zip"]),
}


def _record(path: Path) -> dict | None:
    try:
        with zipfile.ZipFile(path) as z:
            return json.load(io.TextIOWrapper(z.open("record.json")))
    except Exception:
        return None


def read(name: str) -> dict | None:
    """Find a bundle in results/inbox/ or results/ and return its record."""
    for root in (REPO / "results" / "inbox", REPO / "results"):
        p = root / name
        if p.exists():
            return _record(p)
    return None


def find(dataset: str, shots: int, preferred: str) -> tuple[dict, str] | None:
    """Locate a bundle by its CONTENT, falling back from the expected filename.

    Runs launched from an older notebook wrote `s6.zip` for whichever dataset they
    happened to evaluate, so relying on the filename means a mis-rename silently feeds
    the wrong dataset into the merge. Matching on `dataset` + `shots` inside the record
    cannot be fooled that way. The merged output is skipped explicitly (`dataset:
    "both"`), so this can never read its own previous result.
    """
    rec = read(preferred)
    if rec is not None:
        return rec, preferred
    for root in (REPO / "results" / "inbox", REPO / "results"):
        for p in sorted(root.glob("*.zip")) if root.exists() else []:
            r = _record(p)
            if (r and r.get("dataset") == dataset
                    and r.get("protocol", {}).get("shots") == shots):
                return r, p.name
    return None


def main() -> int:
    merged: dict = {}
    env_like: dict = {}
    for ds, (five_name, one_names) in SOURCES.items():
        found = find(ds, 5, five_name)
        if found is None:
            raise SystemExit(
                f"no 5-shot {ds} bundle found. Run S6 with DATASET_5SHOT={ds!r} and drop "
                f"the bundle in results/inbox/ (any name — it is matched on content, but "
                f"{five_name} is the conventional one)."
            )
        five, src = found
        one = next((r for n in one_names if (r := read(n)) is not None), None)
        if one is None:
            raise SystemExit(f"missing the 1-shot counterpart for {ds}: {one_names}")

        sel5 = five["protocol"]["support_selection"]
        sel1 = one["protocol"]["support_selection"]
        if sel5 != sel1:
            raise SystemExit(
                f"{ds}: 5-shot used {sel5!r} but the 1-shot bundle used {sel1!r}. "
                "A 5-vs-1 comparison across different support selections is meaningless "
                "— they draw different supports by construction."
            )
        if five["protocol"]["shots"] != 5 or one["protocol"]["shots"] != 1:
            raise SystemExit(f"{ds}: expected shots 5 and 1, got "
                             f"{five['protocol']['shots']} and {one['protocol']['shots']}")

        merged[ds] = {
            "miou": five["results"]["miou"],
            "std": five["results"]["std"],
            "miou_1shot": one["results"]["miou"],
            "std_1shot": one["results"]["std"],
            "support_selection": sel5,
            "per_group": five["results"]["per_group"],
        }
        env_like = env_like or {k: five.get(k) for k in ("metric", "data", "model")}
        print(f"  {ds:16s} 1-shot {100 * one['results']['miou']:6.2f} -> "
              f"5-shot {100 * five['results']['miou']:6.2f} "
              f"({sel5}, {five['protocol']['n_runs']} runs)   [{src}]")

    record = {
        "stage": "s6", "dataset": "both",
        "adapter": "sst_sam2",
        **{k: v for k, v in env_like.items() if v is not None},
        "protocol": {"shots": 5, "n_runs": 20,
                     "support_selection": {d: v["support_selection"]
                                           for d, v in merged.items()}},
        "results": merged,
    }
    out = REPO / "results" / "inbox" / "s6.zip"
    export_bundle("s6", record, out)
    print(f"\nwrote {out}\nnext:  python scripts/check_gate.py s6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
