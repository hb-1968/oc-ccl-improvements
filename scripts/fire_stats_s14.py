#!/usr/bin/env python3
"""Recompute SeC's native would-fire statistic over the exact S14 protocol draws.

The fan-out workers computed this per episode (`adapters/sec.bhattacharyya_hsv`,
accumulated on the adapter), but per-run records bank only `tta` adapter stats —
banking concept stats would be an evaluator fingerprint break, deferred — and the
merge process resumes every run with zero inference, so the landed bundle carries
``native_fire_rate: None``. The statistic is a pure CPU function of dataset pixels
and the seeded draws, both of which live on this machine, so the gate's evidence is
recomputed here and PINNED to the bundle by ``support_set_hash`` equality: if the
reconstructed draws differ from the ones the workers scored, the gate must not read
this file.

    python scripts/fire_stats_s14.py --cache-root ~/Projects/oc-ccl-data/cache-png

Writes results/s14sec_fire_stats.json, where check_gate's s14sec would-fire
predicate looks when the record's own statistic is None.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.adapters.sec import SCENE_CHANGE_THRESHOLD, bhattacharyya_hsv  # noqa: E402
from occl.datasets.beetle import Beetle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402
from occl.protocol import build_episodes, build_run  # noqa: E402
from occl.seeding import support_set_hash  # noqa: E402

# The S14 protocol, exactly as the notebook ran it (S11_SEL): each dataset's own
# best 1-shot selection.
PROTOCOLS = {
    "butterfly_major": (ButterflyMajor, "canonical_placement"),
    "beetle": (Beetle, "canonical_balanced"),
}
SHOT, N_RUNS = 1, 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", default=str(REPO / "results" / "s14sec_fire_stats.json"))
    args = ap.parse_args()

    cache = ImageCache(args.cache_root)
    out = {}
    for name, (cls, selection) in PROTOCOLS.items():
        ds = cls(cache=cache)
        groups = list(ds.groups())
        # the evaluator hashes the FIRST group's draws, keyed by run_idx
        supports = {run_idx: build_run(ds, groups[0], SHOT, run_idx, selection)[0]
                    for run_idx in range(N_RUNS)}
        dists = []
        t0 = time.time()
        for group in groups:
            for run_idx in range(N_RUNS):
                for ep in build_episodes(ds, group, SHOT, run_idx, selection):
                    dists.append(bhattacharyya_hsv(ep.support_images[-1],
                                                   ep.query_image))
        fired = sum(1 for d in dists if d > SCENE_CHANGE_THRESHOLD)
        out[name] = {
            "support_set_hash": support_set_hash(supports),
            "selection": selection,
            "groups": groups,
            "n_episodes": len(dists),
            "threshold": SCENE_CHANGE_THRESHOLD,
            "native_fire_rate": fired / len(dists) if dists else None,
            "would_fire": fired,
            "bhattacharyya_mean": sum(dists) / len(dists) if dists else None,
            "bhattacharyya_max": max(dists) if dists else None,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"{name}: {len(dists)} episodes, fire-rate "
              f"{out[name]['native_fire_rate']:.4f} "
              f"(mean {out[name]['bhattacharyya_mean']:.4f}, "
              f"max {out[name]['bhattacharyya_max']:.4f}, "
              f"hash {out[name]['support_set_hash']}) "
              f"in {out[name]['elapsed_s']:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
