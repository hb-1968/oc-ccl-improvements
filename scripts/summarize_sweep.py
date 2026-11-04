#!/usr/bin/env python3
"""Summarize saved learning-rate sweep arms and their matched comparisons."""

from __future__ import annotations

import json
import statistics as st
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl import paths  # noqa: E402


def load_runs(pattern: str = "s8b_*.zip") -> dict[float, dict]:
    runs: dict[float, dict] = {}
    for root in (paths.results_root(), paths.inbox_root()):
        for p in sorted(Path(root).glob(pattern)):
            with zipfile.ZipFile(p) as z:
                rec = json.loads(z.read("record.json"))
            rec["_path"] = p.name
            runs[rec["finetune"]["lr"]] = rec
    return runs


def main() -> int:
    runs = load_runs()
    if len(runs) < 2:
        print(f"need >= 2 sweep bundles, found {len(runs)}")
        return 1
    lrs = sorted(runs)

    print(f"\nS8b sweep — {len(lrs)} arms, {runs[lrs[0]]['dataset']}, "
          f"{runs[lrs[0]]['training']['steps']} steps each\n")
    print(f"{'lr':>9s} {'||dW||':>8s} {'loss mean':>10s} {'h1':>8s} {'h2':>8s} "
          f"{'h2-h1':>8s}   {'bundle'}")
    print("-" * 78)
    for lr in lrs:
        t = runs[lr]["training"]
        L = t["losses"]
        h = len(L) // 2
        print(f"{lr:9.0e} {t['lora_delta_norm']:8.3f} {st.mean(L):10.4f} "
              f"{st.mean(L[:h]):8.4f} {st.mean(L[h:]):8.4f} "
              f"{st.mean(L[h:]) - st.mean(L[:h]):+8.4f}   {runs[lr]['_path']}")

    # Identical start, or nothing below is interpretable.
    zeros = {round(runs[lr]["training"]["losses"][0], 6) for lr in lrs}
    ok = len(zeros) == 1
    print(f"\nstep-0 loss across arms: {sorted(zeros)} — "
          + ("identical, so the arms differ only in lr" if ok
             else "DIFFER: something other than lr varied; the comparison below is void"))
    if not ok:
        return 1

    base_lr = lrs[0]
    base = np.array(runs[base_lr]["training"]["losses"])
    rng = np.random.default_rng(0)
    h = len(base) // 2
    print(f"\npaired per-step difference vs lr={base_lr:.0e}, 10k bootstrap "
          f"(negative = lower loss):")
    print(f"{'lr':>9s} {'full epoch':>26s} {'second half':>26s}")
    print("-" * 64)
    for lr in lrs[1:]:
        cells = []
        for a, b in ((np.array(runs[lr]['training']['losses']), base),
                     (np.array(runs[lr]['training']['losses'][h:]), base[h:])):
            d = a - b
            boot = rng.choice(d, size=(10_000, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mark = " " if lo < 0 < hi else "*"
            cells.append(f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]{mark}")
        print(f"{lr:9.0e} {cells[0]:>26s} {cells[1]:>26s}")
    print("\n* = 95% CI excludes zero.")
    print("\nA lower CYCLE loss is not a better mIoU. The cycle reconstructs a merged "
          "foreground\nsilhouette; the metric scores per-trait assignment. Only S8c's "
          "paired evaluation\nsettles whether any of this transfers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
