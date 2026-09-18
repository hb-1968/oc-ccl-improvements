#!/usr/bin/env python3
"""S14 commissioning — fail in seconds, before a session spends GPU-hours.

    python scripts/commission_s14.py --family sam3 --checkpoint /path/sam3.pt
    python scripts/commission_s14.py --family sec  --checkpoint /path/SeC-4B

Builds the family's adapter STRICTLY, asserts the gate-shaped commissioning facts
(geometry pins, 0/0 load, autocast exited, config-bug patched), and runs one
synthetic 2-frame episode twice — once cold, once against the embedding cache — and
asserts the logits are bit-identical (the S3 `cache_check` predicate, re-earned on
the new mechanism). Prints the config_report so the session log carries it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["sam3", "sec"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shots", type=int, default=1,
                    help="supports in the synthetic episode; >1 additionally asserts "
                         "that memory attention sees EVERY support at the query frame "
                         "(S15: sam3's builder caps conditioning frames at 4)")
    args = ap.parse_args()

    import numpy as np

    from occl.datasets.base import Episode

    if args.family == "sam3":
        from occl.adapters.sam3 import Sam3TrackerAdapter

        adapter = Sam3TrackerAdapter(checkpoint=args.checkpoint)
    else:
        from occl.adapters.sec import SeCAdapter

        adapter = SeCAdapter(checkpoint=args.checkpoint, concept_policy="off")
    adapter.build(device=args.device)

    rep = adapter.config_report()
    print(json.dumps({k: v for k, v in rep.items() if k != "decode"}, indent=1))
    assert rep["load_missing_keys"] == 0 and rep["load_unexpected_keys"] == 0, \
        "loose load"
    if args.family == "sam3":
        assert (rep["image_size"], rep["mask_input_size"],
                rep["num_maskmem"]) == (1008, 1152, 7)
        assert rep["global_autocast_exited"] is True
    else:
        assert rep["num_maskmem"] == 22

    rng = np.random.default_rng(0)
    sups = [rng.integers(0, 255, (400, 400, 3), np.uint8) for _ in range(args.shots)]
    qry = rng.integers(0, 255, (400, 400, 3), np.uint8)
    lms = []
    for k in range(args.shots):
        lm = np.zeros((400, 400), np.int32)
        lm[50 + 8 * k:150 + 8 * k, 50:150] = 1
        lm[250:350, 200:300] = 3
        lms.append(lm)
    sids = tuple(f"s{k}" for k in range(args.shots))
    ep = Episode(query_id="q", support_ids=sids, support_images=tuple(sups),
                 support_labelmaps=tuple(lms), query_image=qry, class_ids=(1, 3))

    seen = []
    if args.family == "sam3" and args.shots > 1:
        # Spy on the tracker's conditioning-frame selection: at the query frame
        # it must return ALL `shots` supports. The builder's cap of 4 would
        # silently pass 4 of 5 here and every 5-shot number downstream would be
        # a 4-shot number wearing a 5-shot label.
        import sam3.model.sam3_tracker_base as _tb

        _orig = _tb.select_closest_cond_frames

        def _spy(frame_idx, cond, cap, *a, **k):
            out = _orig(frame_idx, cond, cap, *a, **k)
            seen.append((int(frame_idx), len(cond), len(out[0]), cap))
            return out

        _tb.select_closest_cond_frames = _spy
        cap = rep.get("max_cond_frames_in_attn")
        assert cap == -1 or (isinstance(cap, int) and cap >= args.shots + 1), \
            f"max_cond_frames_in_attn={cap!r}: a {args.shots}-shot episode would drop supports"

    cold = adapter.predict(ep)          # encodes through the predict-side path
    if seen:
        at_query = [s for s in seen if s[0] == args.shots]
        assert at_query and all(n_sel == n_cond == args.shots
                                for _, n_cond, n_sel, _ in at_query), \
            f"memory attention did not see all {args.shots} supports at the query: {seen}"
        print(f"conditioning frames at the query: {args.shots}/{args.shots} attended "
              f"(cap {rep.get('max_cond_frames_in_attn')})")
    if hasattr(adapter, "precompute"):
        adapter.clear_embeddings()
        adapter.precompute([*zip(sids, sups), ("query:q", qry)])
        warm = adapter.predict(ep)      # served from the precompute-side path
        assert np.array_equal(cold, warm), \
            "cached predictions differ from uncached — the feature cache lies"
        print(f"cache identity: OK ({adapter.cache_report()})")
    assert np.isfinite(cold[cold > -1e3]).all()
    print(f"logits {cold.shape}, ch maxes "
          f"{[round(float(c.max()), 2) for c in cold]}")
    print(f"COMMISSION {args.family.upper()}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
