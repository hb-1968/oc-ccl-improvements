#!/usr/bin/env python3
"""Back-fill ``data.cache_format`` on a LEGACY ledger bundle, with the evidence recorded.

    python scripts/backfill_ledger.py results/inbox/s6_beetle.zip
    python scripts/backfill_ledger.py results/inbox/s6_beetle.zip --manifest oc-ccl-data/cache/manifest.jsonl

`data.cache_format` joined `COMPARABILITY_FIELDS` when the lossless cache landed, so
that a JPEG-era butterfly run can never be paired with a PNG-era one. Bundles written
BEFORE the field existed carry nothing there, and `registry.compare` then refuses them
against any modern record — correctly, for butterfly, whose pixels changed between the
eras. It is the wrong answer for a dataset whose images were stored losslessly in EVERY
era: the field is absent because the schema was younger, not because the pixels differ.

This script closes that gap the only honest way: it reads the era's cache manifest, and
only if every entry for the bundle's dataset is lossless does it write
``data.cache_format = "png"`` — alongside ``data.cache_format_backfilled`` naming the
manifest, the entry count, the formats found and the date, so the record says what was
inferred and from what. A dataset with any lossy entry is refused. The original bundle
is kept next to the result as ``<name>.legacy.zip``.

First use (2026-09-13): ``s6_beetle.zip`` (2026-07-31, the README's 5-shot beetle
reproduction, 73.16) as S15's SAM 2 5-shot anchor. Beetle: 180/180 PNG in both
``oc-ccl-data/cache`` and ``oc-ccl-data/cache-png``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

LOSSLESS = {"png", "webp"}


def manifest_formats(manifest: Path, dataset: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(manifest) as fh:
        for line in fh:
            e = json.loads(line)
            if e.get("dataset") == dataset:
                counts[e.get("fmt")] = counts.get(e.get("fmt"), 0) + 1
    return counts


def backfill(record: dict, manifest: Path) -> dict:
    """Return the back-filled record, or raise if the evidence does not license it."""
    data = record.get("data") or {}
    if data.get("cache_format") is not None:
        raise ValueError(f"data.cache_format is already {data['cache_format']!r} — nothing to do")
    ds = record.get("dataset")
    counts = manifest_formats(manifest, ds)
    if not counts:
        raise ValueError(f"{manifest} has no entries for dataset {ds!r}")
    lossy = {f: n for f, n in counts.items() if f not in LOSSLESS}
    if lossy:
        raise ValueError(
            f"{ds}: {lossy} lossy entries in {manifest} — this dataset's pixels DID differ "
            "between eras; the field cannot be inferred, it must stay absent")
    fmt = "png" if counts.get("png") else next(iter(counts))
    out = json.loads(json.dumps(record))
    out.setdefault("data", {})
    out["data"]["cache_format"] = fmt
    out["data"]["cache_format_backfilled"] = {
        "from_manifest": str(manifest), "dataset": ds, "entries": sum(counts.values()),
        "formats": sorted(counts), "date": dt.date.today().isoformat(),
        "reason": "record predates the data.cache_format comparability field; every "
                  "cached image of this dataset was lossless in every era",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("--manifest", default=str(REPO / "oc-ccl-data" / "cache" / "manifest.jsonl"),
                    help="the ERA's cache manifest (default: the pre-lossless cache, which "
                         "is what a legacy bundle read)")
    args = ap.parse_args()
    src = Path(args.bundle)
    manifest = Path(args.manifest)
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        record = json.loads(z.read("record.json"))
        others = {n: z.read(n) for n in names if n != "record.json"}
    out = backfill(record, manifest)
    legacy = src.with_name(src.stem + ".legacy.zip")
    if not legacy.exists():
        shutil.copy(src, legacy)
    with zipfile.ZipFile(src, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("record.json", json.dumps(out, indent=2, default=str))
        for n, b in others.items():
            z.writestr(n, b)
    bf = out["data"]["cache_format_backfilled"]
    print(f"{src}: data.cache_format <- {out['data']['cache_format']!r} "
          f"({bf['entries']} {bf['dataset']} entries, formats {bf['formats']}, "
          f"from {manifest}); original kept at {legacy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
