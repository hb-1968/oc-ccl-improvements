#!/usr/bin/env python3
"""S5-DIAG — isolate why Butterfly Major lands 12.5 points below the paper.

    python scripts/run_diagnostic.py --cache-root /content/cache --checkpoint ckpt.pt \
        --groups cyrbia malleti --n-runs 2 --out s5diag.zip

Context. S5a measured **68.54 ± 3.15** against the paper's **81.0 ± 1.0**. Three
explanations have already been eliminated *offline*, so this script does not re-test them:

* **Aggregation** is not the gap. All four readings of "averaged over traits in images"
  (per-image macro, per-trait macro, pooled, and skip-unpredictable) agree within 0.5
  points on both datasets.
* **Support curation** is not the gap. Fixing all three trait-incomplete supports is worth
  +0.8, and stratum-matching supports to the test composition is worth +0.08.
* **The data path is correct.** The model-free copy baseline reproduces the paper's
  Table A3 on *both* datasets -- Major **21.04 ± 2.98** vs **20.9 ± 10.9**, Beetle
  **12.09** vs **13.2 ± 6.7**. Masks, splits, trait vocabularies, label-map resize and
  aggregation are therefore all verified against a published number.

That leaves the model path, and exactly three live hypotheses:

  H1  ASSIGNMENT RULE. The paper never states how overlapping trait predictions are
      resolved. We argmax; SST's own `per_trait_iou` binarises each trait independently.
      Independent binarisation is never lower per-trait, so it is the natural suspect for
      numbers that sit high. `--groups` x `--n-runs` here scores BOTH from identical
      logits, so the comparison is exact rather than inferred.

  H2  IMAGE RESAMPLING. The copy baseline validates masks but never touches images, so it
      cannot clear this. We downsample ~5.3x (5472 -> 1024) with `cv2.resize`'s default
      INTER_LINEAR, faithful to SST's `read_image`, but that is a known aliasing case and
      the failing subspecies are exactly the ones whose wings are small and dark. Pass
      `--cache-root` pointing at an INTER_AREA-derived cache to test it; everything else
      is held fixed.

  H3  SAM 2 INVOCATION. Recorded, not tested: image_size, num_maskmem,
      apply_postprocessing, non-overlap constraint, dtype. A mismatch here would depress
      every subspecies uniformly.

The slice is deliberately two subspecies at the extremes -- cyrbia (44.36, worst) and
malleti (79.12, near paper level). A cause that explains the gap must move cyrbia much
more than malleti; a cause that moves both equally is a global bug instead.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.bundle import download, export_bundle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402
from occl.metrics import ASSIGNMENT_RULES, assignment_variants, score_episode  # noqa: E402
from occl.protocol import build_episodes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--groups", nargs="*", default=["cyrbia", "malleti"])
    ap.add_argument("--n-runs", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="s5diag.zip")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--resample-tag", default="INTER_LINEAR",
                    help="records which cache was used, so H2 arms cannot be confused")
    args = ap.parse_args()

    import torch

    from occl.adapters.sst_sam2 import SSTSam2Adapter

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = ButterflyMajor(cache=ImageCache(args.cache_root))
    adapter = SSTSam2Adapter(checkpoint=args.checkpoint)
    adapter.build(device=device, dtype=None)

    # H3: capture the invocation exactly, so a config drift can never be argued about
    # later. `config_report` is the same dict the S3 gate asserts against, extended with
    # the two SAM 2 flags that bear directly on the overlap rule.
    m = adapter.model
    config = {
        **adapter.config_report(),
        "non_overlap_masks": bool(getattr(m, "non_overlap_masks", False)),
        "binarize_mask_from_pts_for_mem_enc": bool(
            getattr(m, "binarize_mask_from_pts_for_mem_enc", False)),
        "device": device,
        "resample": args.resample_tag,
    }
    print(json.dumps(config, indent=2))

    rows = []
    t0 = time.time()
    for group in args.groups:
        for run_idx in range(args.n_runs):
            for ep in build_episodes(ds, group, shot=1, run_idx=run_idx,
                                     support_selection="uniform"):
                logits = adapter.predict(ep)
                gt = ds.load_labelmap(ep.query_id)
                base = score_episode(logits, ep.class_ids, gt)
                if base["miou"] is None:
                    continue
                var = assignment_variants(logits, ep.class_ids, gt)
                rows.append({"group": group, "run_idx": run_idx, "query_id": ep.query_id,
                             "support_id": ep.support_ids[0],
                             "iou_binary": base["iou_binary"], **var})
            print(f"  {group} run {run_idx}: {len(rows)} episodes, {time.time()-t0:.0f}s")

    # Aggregate exactly as the real evaluator does: macro over images, then over runs.
    def agg(group, rule):
        per_run = []
        for r in range(args.n_runs):
            v = [x[rule] for x in rows if x["group"] == group and x["run_idx"] == r]
            if v:
                per_run.append(float(np.mean(v)))
        return per_run

    print(f"\n{'group':10s} " + "".join(f"{r:>22s}" for r in ASSIGNMENT_RULES) + f"{'iou_binary':>13s}")
    summary: dict = {}
    for g in args.groups:
        cells = []
        for rule in ASSIGNMENT_RULES:
            v = agg(g, rule)
            summary.setdefault(g, {})[rule] = {"mean": float(np.mean(v)), "per_run": v}
            cells.append(f"{100*np.mean(v):21.2f}")
        b = float(np.mean([x["iou_binary"] for x in rows if x["group"] == g]))
        summary[g]["iou_binary"] = b
        print(f"{g:10s} " + "".join(cells) + f"{100*b:12.2f}")

    print("\nHow to read this:")
    print("  independent >> argmax    -> H1 confirmed; the paper used per-trait "
          "binarisation and our argmax is the deviation.")
    print("  oracle      >> argmax    -> regions ARE found but mislabelled: an assignment "
          "failure, and support curation is back on the table.")
    print("  oracle      ~= argmax    -> the regions were never segmented; neither the "
          "overlap rule nor curation can recover it. Look at H2/H3.")
    print("  cyrbia and malleti move together -> global bug, not a subspecies effect.")

    record = {
        "dataset": "butterfly_major", "stage": "s5diag",
        "protocol": {"shots": 1, "n_runs": args.n_runs, "groups": args.groups,
                     "support_selection": "uniform"},
        "model": config,
        "results": {"by_group": summary, "n_episodes": len(rows)},
        "reference": {"s5a_measured": 68.54, "paper": 81.0,
                      "copy_baseline_ours": 21.04, "copy_baseline_paper": 20.9},
    }

    csv_path = REPO / "per_query_s5diag.csv"
    cols = ["group", "run_idx", "query_id", "support_id", "iou_binary", *ASSIGNMENT_RULES]
    with open(csv_path, "w", newline="") as fh:
        import csv as _csv

        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    bundle = Path(args.out)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle = export_bundle("s5diag", record, bundle,
                           extra_files={"per_query.csv": csv_path},
                           dtype=str(getattr(adapter, "dtype", None)))
    print(f"\nwrote {bundle.resolve()}")
    if not args.no_download:
        download(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
