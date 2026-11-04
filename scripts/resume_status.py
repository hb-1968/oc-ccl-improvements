#!/usr/bin/env python3
"""Inspect completed records and missing work in a resume directory.

Read status only; this command does not restart workers.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def parse(name: str):
    """`ds__group__shotN__sel__adapter__fp__runN.json` -> parts, or None if unrecognised."""
    stem = name[:-5] if name.endswith(".json") else name
    bits = stem.split("__")
    if len(bits) < 7 or not bits[-1].startswith("run"):
        return None
    ds, group, shot, sel, adapter, fp, run = (
        bits[0], "__".join(bits[1:-5]), bits[-5], bits[-4], bits[-3], bits[-2], bits[-1])
    if not shot.startswith("shot"):
        return None
    try:
        return ds, group, int(shot[4:]), sel, adapter, fp, int(run[3:])
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("resume_dir")
    ap.add_argument("--expect-runs", type=int, default=20)
    ap.add_argument("--miou", action="store_true",
                    help="also read each JSON and show the mean mIoU of what is banked")
    args = ap.parse_args()

    root = Path(args.resume_dir)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    files = sorted(root.glob("*.json"))
    arms: dict[tuple, dict[str, set[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    scores: dict[tuple, list[float]] = collections.defaultdict(list)
    skipped = 0
    for f in files:
        p = parse(f.name)
        if p is None:
            skipped += 1
            continue
        ds, group, shot, sel, adapter, fp, run = p
        arms[(ds, shot, sel, adapter, fp)][group].add(run)
        if args.miou:
            try:
                scores[(ds, shot, sel, adapter, fp)].append(
                    json.loads(f.read_text())["miou"])
            except Exception:  # noqa: BLE001 — a half-written file is not fatal here
                pass

    print(f"\n{len(files)} JSON files in {root}"
          + (f" ({skipped} unrecognised)" if skipped else ""))
    if not arms:
        print("  nothing recognisable — is this the right directory?")
        return 1

    print(f"\n{'dataset':>16s} {'shot':>4s} {'selection':>20s} {'adapter':>20s} "
          f"{'fp':>9s} {'complete':>9s}  missing")
    print("-" * 108)
    for key in sorted(arms, key=lambda k: (k[0], k[1], k[2], k[3])):
        ds, shot, sel, adapter, fp = key
        by_group = arms[key]
        done = sum(len(v) for v in by_group.values())
        want = args.expect_runs * len(by_group)
        gaps = []
        for g, runs in sorted(by_group.items()):
            miss = sorted(set(range(args.expect_runs)) - runs)
            if miss:
                gaps.append(f"{g}:{len(miss)}")
        mark = "" if done == want else "  <-"
        line = (f"{ds:>16s} {shot:>4d} {sel:>20s} {adapter:>20s} {fp:>9s} "
                f"{done:>4d}/{want:<4d}{mark} {' '.join(gaps) if gaps else ''}")
        print(line)
        if args.miou and scores.get(key):
            v = scores[key]
            print(f"{'':>16s} banked mean mIoU {100 * sum(v) / len(v):.2f} over {len(v)} runs")

    print("\ngroups seen per arm:")
    for key in sorted(arms, key=lambda k: (k[0], k[1])):
        print(f"  {key[0]}/{key[1]}-shot/{key[3]}: {sorted(arms[key])}")
    print("\nAn arm at n/n is fully banked — re-running it does zero inference and only "
          "merges.\nA missing arm has to be computed; a partial one resumes from where it "
          "stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
