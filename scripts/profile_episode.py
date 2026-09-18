#!/usr/bin/env python3
"""Where an episode's time actually goes — the instrument the throughput anomaly needed.

    python scripts/profile_episode.py --dataset butterfly_major --shot 5 \
        --cache-root <cache> --group malleti --episodes 12

Butterfly at 5-shot ran at **2.95 s/image through `fanout` (S6b)** and **~13 s/image
through the s8d launcher**, on the same card, same protocol, same 100 episodes a run. Three
candidate causes were identified by reading the code; this measures them instead, because
the last time a number here was inferred rather than measured it was wrong by 2.5x in one
direction and 2x in the other.

What it reports, per episode:

* **frame prep** — SST's `load_video_frames`, run inside `init_state`. A PIL round-trip and
  a float64 divide per frame; measured off-GPU at 145 ms for 6 frames against 48 ms for 2.
* **`_get_image_feature` calls, and the bytes they move.** The embedding cache skips the
  image *encoder* but the patch still re-uploads the raw 12.6 MB frame on every call, and
  **every caller in SAM 2 discards it** (`_, _, current_vision_feats, ...` in all three of
  `_run_single_frame_inference`, `_run_memory_encoder` and `_get_empty_mask_ptr`). The call
  count is `1 + shot*classes + shot + 1`, so butterfly at 5-shot x 10 traits pays 57 of
  them — 718 MB of pageable host-to-device per episode — against 13 at 1-shot.
* **prompting** (`add_new_mask`), **propagation**, and **scoring**.

Then it A/Bs the two `init_state` placement flags. Both are pure memory-placement choices —
`offload_video_to_cpu` decides whether frames are normalised on CPU or GPU, and
`offload_state_to_cpu` whether the memory bank round-trips to host between frames — so the
logits must come back **bit-identical**, and that is asserted rather than assumed. If they
do, `--gpu-frames` on `run_eval.py`/`run_arms.py` turns the redundant upload into a no-op.

Cheap on purpose: a dozen episodes, one group, no bundle. Run it before commissioning
anything long.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.bundle import download, export_bundle  # noqa: E402
from occl.datasets.beetle import Beetle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402
from occl.protocol import build_episodes  # noqa: E402

DATASETS = {"beetle": Beetle, "butterfly_major": ButterflyMajor}


class Probe:
    """Counts and times what happens inside one `predict`, by wrapping the model.

    Wraps rather than edits: `adapters/sst_sam2.py` is in `evaluator._FINGERPRINTED`, so a
    line changed there invalidates every per-run JSON ever banked. A profiler must not cost
    the project its resume cache.
    """

    def __init__(self, adapter):
        import torch

        self.torch = torch
        self.adapter = adapter
        self.model = adapter.model
        self.reset()
        self._orig_gif = self.model._get_image_feature
        self._orig_init = self.model.init_state
        self._orig_mask = self.model.add_new_mask
        self._orig_prop = self.model.propagate_in_video

    def reset(self):
        self.n_gif = 0
        self.h2d_bytes = 0
        self.t = {"frame_prep": 0.0, "get_image_feature": 0.0, "add_new_mask": 0.0,
                  "propagate": 0.0}

    def _sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def install(self):
        torch, model = self.torch, self.model

        def gif(inference_state, frame_idx, batch_size):
            self.n_gif += 1
            img = inference_state["images"][frame_idx]
            # Only a host tensor costs a transfer; on-device frames make the patch free.
            if not str(img.device).startswith("cuda"):
                self.h2d_bytes += img.numel() * 4      # .float() upload, fp32
            self._sync()
            t0 = time.perf_counter()
            out = self._orig_gif(inference_state, frame_idx, batch_size)
            self._sync()
            self.t["get_image_feature"] += time.perf_counter() - t0
            return out

        def init_state(*a, **kw):
            self._sync()
            t0 = time.perf_counter()
            gif0 = self.t["get_image_feature"]
            out = self._orig_init(*a, **kw)
            self._sync()
            # `init_state` warms frame 0 on its way out, so its wall time is frame prep PLUS
            # one `_get_image_feature`. Subtract what `gif` already charged, or the same
            # milliseconds appear in two rows and the breakdown oversums.
            self.t["frame_prep"] += (time.perf_counter() - t0
                                     - (self.t["get_image_feature"] - gif0))
            return out

        def add_new_mask(*a, **kw):
            # Same nesting subtraction as `init_state`: `add_new_mask` reaches
            # `_run_single_frame_inference`, which calls `_get_image_feature`. Charging both
            # in full double-counts, and the rows then oversum the episode — the first
            # profile reported 2089 ms of parts inside an 1803 ms episode, which invites
            # exactly the wrong conclusion about where the time is.
            self._sync()
            t0 = time.perf_counter()
            gif0 = self.t["get_image_feature"]
            out = self._orig_mask(*a, **kw)
            self._sync()
            self.t["add_new_mask"] += (time.perf_counter() - t0
                                       - (self.t["get_image_feature"] - gif0))
            return out

        def propagate(*a, **kw):
            # try/finally, not a plain trailing statement: `predict` BREAKS out of this
            # generator at the query frame, so the loop never falls through and anything
            # after it would never run. The first version of this reported 0.0 ms of
            # propagation for every episode, which reads as "propagation is free" rather
            # than as an instrumentation bug — exactly the failure a profiler must not have.
            self._sync()
            t0 = time.perf_counter()
            gif0 = self.t["get_image_feature"]
            try:
                for item in self._orig_prop(*a, **kw):
                    yield item
            finally:
                self._sync()
                self.t["propagate"] += (time.perf_counter() - t0
                                        - (self.t["get_image_feature"] - gif0))

        model._get_image_feature = gif
        model.init_state = init_state
        model.add_new_mask = add_new_mask
        model.propagate_in_video = propagate

    def remove(self):
        self.model._get_image_feature = self._orig_gif
        self.model.init_state = self._orig_init
        self.model.add_new_mask = self._orig_mask
        self.model.propagate_in_video = self._orig_prop


def run_config(ds, adapter, group, shot, selection, episodes, label,
               offload_video, offload_state):
    """Time `episodes` episodes under one placement configuration. Returns (stats, logits)."""
    import torch

    adapter.offload_video_to_cpu = offload_video
    adapter.offload_state_to_cpu = offload_state

    probe = Probe(adapter)
    probe.install()
    per_ep, first = [], None
    try:
        torch.cuda.reset_peak_memory_stats()
        for n, ep in enumerate(build_episodes(ds, group, shot, 0, selection)):
            if n >= episodes:
                break
            probe._sync()
            t0 = time.perf_counter()
            logits = adapter.predict(ep)
            probe._sync()
            per_ep.append(time.perf_counter() - t0)
            if first is None:
                first = logits
    finally:
        probe.remove()

    n = max(1, len(per_ep))
    # The warm-up episode carries one-off allocator growth; report the median too.
    stats = {
        "label": label,
        "episodes": len(per_ep),
        "mean_s": sum(per_ep) / n,
        "median_s": statistics.median(per_ep) if per_ep else 0.0,
        "gif_calls": probe.n_gif / n,
        "h2d_mb": probe.h2d_bytes / n / 1e6,
        "peak_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        **{k: v / n for k, v in probe.t.items()},
    }
    stats["unaccounted"] = stats["median_s"] - sum(probe.t[k] / n for k in probe.t)
    return stats, first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="butterfly_major", choices=sorted(DATASETS))
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--group", default=None, help="default: the dataset's first group")
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--support-selection", default="canonical_placement")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--runs-per-group", type=int, default=20,
                    help="only used to project the stage total from the measurement")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="profile.zip",
                    help="bundle path. A timing that survives only as notebook scrollback "
                         "is a timing that gets retyped from memory later — this project "
                         "has been wrong by 2.5x in both directions doing exactly that.")
    ap.add_argument("--no-download", action="store_true",
                    help="skip the Colab download; the notebook does it from the kernel, "
                         "since files.download() cannot work from a subprocess")
    args = ap.parse_args()

    # Say what we are about to do BEFORE importing torch or touching the GPU. The first
    # version printed nothing until after `precompute`, so any failure before that — a
    # missing checkpoint, an uncached image, a bad group name — surfaced in the notebook as
    # a bare `CalledProcessError: returned non-zero exit status 1` with no output above it
    # to say which. A diagnostic tool that cannot report its own failures is the wrong shape.
    print(f"profile: {args.dataset}/{args.group or '(first group)'}  shot={args.shot}  "
          f"sel={args.support_selection}  episodes={args.episodes}", flush=True)
    print(f"  cache      {args.cache_root}", flush=True)
    print(f"  checkpoint {args.checkpoint}", flush=True)

    import numpy as np

    from occl.adapters.sst_sam2 import SSTSam2Adapter
    from occl.evaluator import distinct_images

    cache = ImageCache(args.cache_root)
    ds = DATASETS[args.dataset](cache=cache)
    groups = list(ds.groups())
    if args.group is not None and args.group not in groups:
        raise SystemExit(
            f"\n{args.dataset} has no group {args.group!r}.\n"
            f"  groups: {groups}\n"
            "  Butterfly subspecies names contain spaces and parentheses; quote them."
        )
    group = args.group or groups[0]

    # Which images this profile needs, resolved through the SAME dispatch `evaluate` uses.
    # run 0 only: the profile is of the steady state, not of the encoder.
    keys = distinct_images(ds, group, args.shot, 1, args.support_selection, 0, 1)

    # Preflight, copied in spirit from `run_eval.py`. Curation changes which supports get
    # drawn, so a cache built for one arm does not cover another — and without this the miss
    # surfaces only after the 898 MB checkpoint has loaded, as a FileNotFoundError from deep
    # inside `precompute`.
    missing = [k for k in keys if not cache.has(ds.name, k)]
    if missing:
        raise SystemExit(
            f"\n{len(missing)} of {len(keys)} image(s) needed by {args.dataset}/{group} at "
            f"{args.shot}-shot / {args.support_selection} are not in the cache at "
            f"{args.cache_root}.\n"
            f"  first few: {missing[:8]}\n\n"
            "Either point --cache-root at the cache the evaluation stages use, or profile a "
            "configuration the cache covers:\n"
            f"  --shot 1                       (fewer supports drawn)\n"
            f"  --support-selection uniform    (no curation filter)\n"
        )
    print(f"  {len(keys)} distinct images, all cached", flush=True)

    if not Path(args.checkpoint or "").is_file():
        raise SystemExit(f"\ncheckpoint not found: {args.checkpoint!r}")

    print("  building SAM 2 ...", flush=True)
    adapter = SSTSam2Adapter(checkpoint=args.checkpoint)
    adapter.build(device=args.device)
    print(f"  built on {args.device}, dtype {adapter.dtype}", flush=True)

    t0 = time.perf_counter()
    rep = adapter.precompute((k, ds.load_image(k)) for k in keys)
    t_pre = time.perf_counter() - t0
    print(f"\n{args.dataset}/{group}  shot={args.shot}  sel={args.support_selection}", flush=True)
    print(f"precompute: {rep['n_cached']} embeddings, {rep['total_gb']:.2f} GB, "
          f"{t_pre:.1f} s ({1000 * t_pre / max(1, rep['n_cached']):.0f} ms/image)")
    print(f"  -> {1000 * rep['bytes_per_image'] / 1e6:.2f} MB per cached image "
          f"(planner assumes 10.49)")

    configs = [
        ("offload both (current default)", True, True),
        ("frames on GPU", False, True),
        ("frames + state on GPU", False, False),
    ]
    results, logits = [], {}
    for label, ov, os_ in configs:
        st, lg = run_config(ds, adapter, group, args.shot, args.support_selection,
                            args.episodes, label, ov, os_)
        results.append(st)
        logits[label] = lg
        print(f"\n--- {label} ---", flush=True)
        print(f"  {st['median_s']:.3f} s/episode (median), {st['mean_s']:.3f} mean, "
              f"peak {st['peak_gb']:.2f} GB")
        print(f"  frame prep (load_video_frames) {1000 * st['frame_prep']:8.1f} ms", flush=True)
        print(f"  _get_image_feature x{st['gif_calls']:.0f}         "
              f"{1000 * st['get_image_feature']:8.1f} ms   "
              f"({st['h2d_mb']:.0f} MB host->device)")
        print(f"  add_new_mask                   {1000 * st['add_new_mask']:8.1f} ms", flush=True)
        print(f"  propagate_in_video             {1000 * st['propagate']:8.1f} ms", flush=True)
        # The rows are disjoint (nested `_get_image_feature` time is subtracted from its
        # callers), so they must not exceed the episode. Print the remainder rather than
        # asserting: scoring, prompt-mask construction and the resize live there, and a
        # NEGATIVE remainder is the signal that the subtraction has broken again.
        print(f"  ---- unaccounted               {1000 * st['unaccounted']:8.1f} ms"
              + ("   <- NEGATIVE: the breakdown double-counts, do not trust the rows"
                 if st["unaccounted"] < 0 else ""), flush=True)

    # Placement must not change the arithmetic. If it does, none of this is usable.
    base = logits[configs[0][0]]
    print("\n--- equivalence against the current default ---", flush=True)
    equivalence, ok = [], True
    for label, _ov, _os in configs[1:]:
        same = bool(np.array_equal(base, logits[label]))
        diff = float(np.abs(base - logits[label]).max())
        ok &= same
        equivalence.append({"label": label, "identical": same, "max_abs_diff": diff})
        print(f"  {label:32s} identical={same}  max|diff|={diff:.3e}", flush=True)
    if not ok:
        print("  WARNING: a placement flag changed the logits. It must not — investigate "
              "before using --gpu-frames for anything that will be reported.")

    ref = results[0]["median_s"]
    n_groups = len(ds.groups())
    total_eps = args.runs_per_group * n_groups * len(ds.test_ids(group))
    print(f"\n--- projection: {args.runs_per_group} runs x {n_groups} group(s) x "
          f"{len(ds.test_ids(group))} images = {total_eps} episodes, 1 arm ---")
    for st in results:
        speed = ref / st["median_s"] if st["median_s"] else 0
        print(f"  {st['label']:32s} {st['median_s']:6.3f} s/ep  "
              f"{total_eps * st['median_s'] / 3600:6.2f} worker-hours  ({speed:.2f}x)")
    print("\nDivide worker-hours by the worker count for wall clock. `plan_workers` sizes "
          "that from\nVRAM; `peak` above is ONE worker with one group cached.")

    # ── the bundle ───────────────────────────────────────────────────────────
    #
    # `fixed_gb` is the transferable number: peak residency minus THIS run's cache. The raw
    # peak is not transferable and must not be — the profiler warms run 0 only, so its cache
    # is far smaller than a real worker's slice, and feeding the peak back into
    # `vram_calibration.json` would under-size the fixed cost and re-introduce the
    # over-commit that motivated this script. Recorded for a human to move into `PEAK_GB`;
    # deliberately NOT written to the calibration file. An unrepresentative peak written
    # automatically is exactly how 0.963 GB — a merge pass that did zero inference — became
    # the planner's idea of a worker.
    cache_gb_here = rep["n_cached"] * rep["bytes_per_image"] / 1e9
    for st, (_label, ov, os_) in zip(results, configs):
        st["offload_video_to_cpu"] = ov
        st["offload_state_to_cpu"] = os_
        st["fixed_gb"] = round(max(0.0, st["peak_gb"] - cache_gb_here), 3)

    record = {
        "stage": "profile",
        "dataset": ds.name,
        "group": group,
        "shot": args.shot,
        "support_selection": args.support_selection,
        "episodes": args.episodes,
        "embedding_cache": {
            "n_cached": rep["n_cached"],
            "bytes_per_image": rep["bytes_per_image"],
            # The planner hardcodes EMBED_MB_PER_IMAGE = 10.49, derived from the S4/S5
            # bundles. Recording the measured value lets that constant be CHECKED rather
            # than trusted, which is the whole difference between the two.
            "mb_per_image": round(rep["bytes_per_image"] / 1e6, 3),
            "total_gb": rep["total_gb"],
            "precompute_s": round(t_pre, 2),
        },
        "configs": results,
        "equivalence": equivalence,
        "placement_changes_logits": not ok,
        "projection": {
            "episodes_total": total_eps,
            "runs_per_group": args.runs_per_group,
            "n_groups": n_groups,
            "worker_hours_by_config": {
                st["label"]: round(total_eps * st["median_s"] / 3600, 3) for st in results
            },
        },
        # A single-worker measurement. Thread-pool thrash and VRAM over-commit are
        # CONTENTION effects and cannot appear here; confirming those needs an N-worker run.
        "workers": 1,
        "model": adapter.config_report(),
    }

    bundle = export_bundle("profile", record, Path(args.out),
                           dtype=str(getattr(adapter, "dtype", None)))
    print(f"\nwrote {bundle.resolve()}", flush=True)
    if not args.no_download:
        download(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
