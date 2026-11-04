#!/usr/bin/env python3
"""Evaluate several adapter arms while sharing image embeddings.

Keep per-arm weight identities and banked-run keys separate. Shared encodings
avoid repeating the frozen image encoder for each arm.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.adapters.sst_sam2 import SSTSam2Adapter  # noqa: E402
from occl.datasets.beetle import Beetle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.evaluator import evaluate  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402


def resolve_lora(spec: str) -> Path:
    """Accept the S8b bundle or the raw .pt, same as run_eval."""
    p = Path(spec)
    if p.suffix != ".zip":
        return p
    with zipfile.ZipFile(p) as z:
        if "lora.pt" not in z.namelist():
            raise SystemExit(f"{p.name} has no lora.pt (contains {z.namelist()})")
        out = Path(tempfile.mkdtemp()) / "lora.pt"
        out.write_bytes(z.read("lora.pt"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beetle", choices=["beetle", "butterfly_major"])
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--support-selection", default="canonical_balanced")
    ap.add_argument("--resume-dir", required=True)
    ap.add_argument("--run-start", type=int, default=0)
    ap.add_argument("--run-end", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--arms", action="append", required=True,
                    metavar="label=path", help="empty path means the base model")
    ap.add_argument("--adapter", default="sst_sam2",
                    choices=["sst_sam2", "sam3", "sec"],
                    help="model family (S14) — copy_baseline has no weights to arm")
    ap.add_argument("--concept-policy", choices=["off", "forced"], default="off",
                    help="sec only; see run_eval.py")
    ap.add_argument("--gpu-frames", action="store_true",
                    help="keep video frames and tracking state on the GPU; see "
                         "run_eval.py --gpu-frames and scripts/profile_episode.py")
    ap.add_argument("--tta", action="store_true",
                    help="wrap in OrientationTTAAdapter — every arm is then scored with "
                         "test-time augmentation. Run once with and once without to get "
                         "the {weights} x {tta} square.")
    args = ap.parse_args()

    import torch

    from occl.finetune.lora import lora_delta_norm, reset_lora
    from occl.finetune.train import load_finetune, restore_base

    cache = ImageCache(args.cache_root) if args.cache_root else ImageCache()
    ds = {"beetle": Beetle, "butterfly_major": ButterflyMajor}[args.dataset](cache=cache)

    arms = []
    for spec in args.arms:
        label, _, path = spec.partition("=")
        arms.append((label, path or None))
    if not any(p is None for _, p in arms):
        print("  NOTE: no base arm given; deltas will have no reference")
    # Base first, before any adapter is attached.
    arms.sort(key=lambda kv: kv[1] is not None)

    if args.adapter == "sst_sam2":
        adapter = SSTSam2Adapter(checkpoint=args.checkpoint,
                                 offload_video_to_cpu=not args.gpu_frames,
                                 offload_state_to_cpu=not args.gpu_frames)
    elif args.adapter == "sam3":
        from occl.adapters.sam3 import Sam3TrackerAdapter

        adapter = Sam3TrackerAdapter(checkpoint=args.checkpoint)
    elif args.adapter == "sec":
        from occl.adapters.sec import SeCAdapter

        adapter = SeCAdapter(checkpoint=args.checkpoint,
                             concept_policy=args.concept_policy)
    else:
        raise SystemExit(f"unknown adapter {args.adapter!r}")
    adapter.build(device=args.device)
    inner = adapter
    if args.tta:
        # Wrap AFTER build. The wrapper delegates `precompute` to the inner adapter and
        # caches both orientations, so the arms still share one encode — and `adapter.name`
        # becomes `sst_sam2+tta_rot180`, which is what keeps the TTA per-run JSON in a
        # different resume slug from the plain arms.
        from occl.adapters.tta import OrientationTTAAdapter

        adapter = OrientationTTAAdapter(adapter)
        print("  test-time augmentation: support orientation (upright vs rot180)")
    print(f"arms: {[a for a, _ in arms]} | run range [{args.run_start}:{args.run_end}]"
          f" | adapter {adapter.name}")

    applied = False
    base_state: dict | None = None      # full fine-tune only; see restore_base
    method = "lora"

    def set_weights(label, path):
        """Point the adapter at this arm's weights and stamp `weights_id` for the slug.

        The base arm must **undo** whatever the previous arm loaded, not merely relabel it.
        This loop is group-outer/arm-inner, so on every group after the first the base arm
        runs against a model that still has the OC-CCL adapters attached from the previous
        group — scored as fine-tuned, filed under `:base`, and then used as the reference
        the OC-CCL delta is measured against. The delta would collapse toward zero for
        reasons having nothing to do with OC-CCL.

        `reset_lora` rather than surgery on the module tree: `LoRALinear` is an exact no-op
        at `B = 0`, so a reset model is bit-identical to one that never had adapters, and
        the assertion below is what says so rather than assuming it.

        Latent until now, which is why the banked results survive: `_assign` gives a worker
        several groups only when workers < groups, and every run so far had one group per
        worker (beetle has exactly one; butterfly ran at 5 and 8 workers against 5 groups).
        At 4, 6 or 7 workers on butterfly it would have fired silently.
        """
        nonlocal applied, base_state, method
        if path is None:
            if applied:
                if method == "full":
                    # No adapters to zero — the base weights themselves were overwritten,
                    # so they have to be put back from the snapshot taken at load time.
                    delta = restore_base(inner.model, base_state)
                    what = f"restored {len(base_state)} tensors"
                else:
                    n = reset_lora(inner.model)
                    delta = lora_delta_norm(inner.model)
                    what = f"reset {n} adapters"
                if delta != 0.0:
                    raise SystemExit(
                        f"[{label}] {what} but ||dW|| == {delta:.3e}, not 0. "
                        "The base arm would be scored with fine-tuned weights."
                    )
                print(f"  base arm: {what}, ||dW|| = 0", flush=True)
            inner.weights_id = f"{Path(inner.checkpoint).name}:base"
            return 0.0
        pt = resolve_lora(path)
        rep = load_finetune(inner.model, pt, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        method = rep["method"]
        if rep["base_state"] is not None:
            base_state = rep["base_state"]
        if not applied:
            applied = True
            print(f"  {method} fine-tune loaded: "
                  + (f"{rep['n_tensors']} tensors" if method == "full"
                     else f"{rep['modules']} LoRA modules (r={rep['r']}, "
                          f"alpha={rep['alpha']})"), flush=True)
        if rep["mismatches"]:
            raise SystemExit(f"[{label}] {rep['mismatches']} key mismatches — refusing to "
                             "score a model that is part fine-tuned and part base")
        delta = rep["delta_norm"]
        if delta == 0.0:
            raise SystemExit(f"[{label}] ||dW|| == 0: the weights are unchanged from base, "
                             "so this would score the BASE model as fine-tuned")
        inner.weights_id = f"{Path(inner.checkpoint).name}:{method}{delta:.8f}"
        return delta

    # GROUP outer, ARM inner. The arms share an image encoder so they must share one
    # encode — but *groups* must not, or a worker owning several of Butterfly Major's five
    # subspecies would hold every group's embeddings at once (~8 GB) instead of one
    # (~1.7 GB). That per-group free is exactly what `evaluate` does by default and what
    # `keep_embeddings` suppresses, so the loop has to reinstate it at the right level.
    groups = args.groups or ds.groups()
    results = {}
    for gi, group in enumerate(groups):
        for label, path in arms:
            t0 = time.time()
            delta = set_weights(label, path)
            out = evaluate(ds, adapter, shot=args.shot, n_runs=args.n_runs,
                           groups=[group], run_start=args.run_start,
                           run_end=args.run_end, resume_dir=Path(args.resume_dir),
                           precompute=True, support_selection=args.support_selection,
                           keep_embeddings=True)
            out.pop("_rows", None)
            results.setdefault(label, []).append(out["results"]["miou"])
            # Peak VRAM, per (group, arm). `run_eval.py` has recorded this since s5c but
            # `run_arms.py` never has — so s8c, s8c_tune and s8d, the three most expensive
            # stages in the project, are the only ones that cannot size themselves from
            # measurement. `plan_group_workers` has been projecting 8.73 GB a worker for
            # butterfly 5-shot + TTA where the real figure is nearer 5, which costs half the
            # card. Same "written later, missing what the original had" shape as the thread
            # caps and the calibration harvest.
            #
            # `fixed` is the transferable part: subtract THIS worker's own cache, because
            # residency scales with the slice width and the raw peak does not transfer.
            peak = cache_gb = 0.0
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_reserved() / 1e9
                # Guarded (S14): a family without a separable embedding cache has
                # neither report; the log line then shows zeros rather than crashing.
                _crep = getattr(inner, "cache_report", None)
                cache_gb = ((_crep().get("total_gb") or 0.0) if _crep else 0.0)
            _cstats = getattr(inner, "cache_stats", {"encoded": "?", "hits": "?"})
            print(f"[{group}][{label}] mIoU {100 * out['results']['miou']:.2f} "
                  f"(||dW||={delta:.3f}) in {time.time() - t0:.0f}s "
                  f"| cache {_cstats['encoded']} encoded, "
                  f"{_cstats['hits']} hits "
                  f"| VRAM peak {peak:.2f} GB, cache {cache_gb:.2f} GB, "
                  f"fixed {max(0.0, peak - cache_gb):.2f} GB", flush=True)
        if hasattr(adapter, "clear_embeddings"):
            adapter.clear_embeddings()          # bound peak VRAM to one group
    results = {k: sum(v) / len(v) for k, v in results.items()}

    if hasattr(adapter, "clear_embeddings"):
        adapter.clear_embeddings()
    print("\n" + json.dumps({k: round(100 * v, 2) for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
