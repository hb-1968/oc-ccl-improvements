#!/usr/bin/env python3
"""Select fine-tuning settings within the training split.

Use training-only folds; test results do not select the epoch count.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.adapters.sst_sam2 import SSTSam2Adapter  # noqa: E402
from occl.bundle import export_bundle  # noqa: E402
from occl.datasets.beetle import Beetle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.datasets.folds import CVFold, fold_report  # noqa: E402
from occl.evaluator import evaluate  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beetle", choices=["beetle", "butterfly_major"])
    ap.add_argument("--group", default="all")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4],
                    help="learning-rate grid. S8b bracketed the optimum inside "
                         "[1e-5, 1e-4]: 3e-5->1e-4 was the steepest gain and 1e-4->3e-4 "
                         "halved it, so refine in there rather than extending the range.")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--prompt-mode", default="logit_reparam")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=10,
                    help="inner-loop runs. Fewer than the outer 20 — this selects a "
                         "hyperparameter, it does not report a result.")
    ap.add_argument("--support-selection", default="canonical_balanced")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default="s8c_tune_fold.zip")
    args = ap.parse_args()

    import torch

    from occl.finetune.lora import (apply_lora, load_lora_state_dict, lora_delta_norm)
    from occl.finetune.train import train_lora

    cache = ImageCache(args.cache_root) if args.cache_root else ImageCache()
    base_ds = {"beetle": Beetle, "butterfly_major": ButterflyMajor}[args.dataset](cache=cache)
    ds = CVFold(base_ds, k=args.k, fold=args.fold, group=args.group)
    import occl as _occl
    # Stamped into every worker log. When a run fails in a way a fix was
    # supposed to prevent, the first question is which code actually ran —
    # and inferring it from which log lines are absent is guesswork.
    print(f"occl build {_occl.BUILD} | {__file__}", flush=True)
    print(ds)

    # The guard, asserted rather than assumed: the real test split must be unreachable.
    real_test = set(base_ds.test_ids(args.group)
                    if _takes_group(base_ds.test_ids) else base_ds.test_ids())
    seen = set(ds.train_ids()) | set(ds.test_ids())
    leaked = seen & real_test
    assert not leaked, (
        f"LEAK: {len(leaked)} real test images reachable from fold {args.fold}; "
        "a hyperparameter chosen here would be chosen on the evaluation set")
    print(f"  leak check: 0 of {len(real_test)} real test images reachable")

    adapter = SSTSam2Adapter(checkpoint=args.checkpoint)
    dtype = torch.float32 if args.device == "cpu" else None
    adapter.build(device=args.device, dtype=dtype)
    t0 = time.time()

    def score() -> float:
        """Evaluate the model as currently weighted, on this fold's validation split."""
        out = evaluate(ds, adapter, shot=args.shot, n_runs=args.n_runs,
                       groups=[args.group], resume_dir=None, precompute=True,
                       support_selection=args.support_selection, keep_embeddings=True)
        out.pop("_rows", None)
        return out["results"]["miou"]

    # phase 1: TRAIN every grid point. No inference happens in this phase.
    #
    # Train-then-score, never interleaved. SAM 2 memoises its rotary encoding on the
    # attention modules, so a scoring pass leaves an inference tensor behind that the next
    # backward pass cannot use — `detach_inference_caches` now repairs that, but not
    # interleaving means it never arises. It is also the better order for the embedding
    # cache: phase 2 then encodes once and reuses it across the whole grid.
    trainings, points, train_s = [], [], 0.0
    for lr in args.lrs:
        ckpt_dir = Path(f"/tmp/s8c_tune_f{args.fold}_lr{lr:g}")
        _t = time.time()
        # `train_lora` resets the adapters in place when they already exist, so each grid
        # point starts from the same initial state rather than from its predecessor's
        # solution — verified equal-in, equal-out by tests/test_finetune.py.
        report = train_lora(adapter.model, ds, lr=lr, epochs=args.epochs,
                            group=args.group, lora_r=args.lora_r,
                            lora_alpha=args.lora_alpha, prompt_mode=args.prompt_mode,
                            memory_reset=not args.no_reset, device=args.device,
                            dtype=dtype, max_steps=args.max_steps,
                            checkpoint_dir=ckpt_dir, log_every=0)
        assert report["lora_delta_norm_before"] == 0.0, (
            f"lr={lr:g} started from a non-zero ||dW|| — the previous grid point leaked in")
        train_s += time.time() - _t
        trainings.append({k: v for k, v in report.items() if k != "losses"})
        points += [(lr, c) for c in report["checkpoints"]]
        print(f"  trained lr={lr:g}: {args.epochs} epochs, "
              f"||dW||={report['lora_delta_norm']:.3f}, "
              f"{report['elapsed_s']:.0f}s", flush=True)

    # phase 2: SCORE. The base first, before any adapter is loaded.
    # Embeddings are kept throughout: the image encoder is never adapted, so ONE encode
    # serves every point on the grid.
    t_score = time.time()
    adapter.weights_id = f"{Path(adapter.checkpoint).name}:base"
    base_miou = score()
    curve = [{"lr": None, "epoch": 0, "miou": base_miou, "delta_norm": 0.0}]
    print(f"\n  base (epoch 0): val mIoU {100 * base_miou:.2f}\n")

    for lr, c in points:
        blob = torch.load(c["path"], map_location="cpu", weights_only=False)
        mism = load_lora_state_dict(adapter.model, blob["lora"])
        assert mism == 0, f"lr={lr:g} epoch {c['epoch']}: {mism} key mismatches"
        delta = lora_delta_norm(adapter.model)
        adapter.weights_id = f"{Path(adapter.checkpoint).name}:lora{delta:.8f}"
        m = score()
        curve.append({"lr": lr, "epoch": c["epoch"], "miou": m, "delta_norm": delta,
                      "train_loss": c["loss_mean"]})
        print(f"  lr={lr:8.1e} epoch {c['epoch']:2d}: val mIoU {100 * m:.2f} "
              f"({100 * (m - base_miou):+.2f} vs base), ||dW||={delta:.3f}", flush=True)
    score_s = time.time() - t_score

    best = max(curve, key=lambda c: c["miou"])
    record = {
        "stage": "s8c_tune",
        "dataset": base_ds.name,
        "cv": {"k": args.k, "fold": args.fold,
               "fit_size": len(ds.train_ids()), "val_size": len(ds.test_ids()),
               "folds": fold_report(base_ds, args.k, args.group)},
        "finetune": {"method": "lora", "lrs": list(args.lrs), "epochs": args.epochs,
                     "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                              "scopes": ["memory_encoder", "sam_mask_decoder"]}},
        "occl": {"prompt_mode": args.prompt_mode, "memory_reset": not args.no_reset,
                 "cycle_len": 2, "mask_mode": "binary"},
        "protocol": {"shots": args.shot, "n_runs": args.n_runs,
                     "support_selection": args.support_selection},
        "trainings": trainings,
        "curve": curve,
        "results": {"best_lr": best["lr"], "best_epoch": best["epoch"],
                    "best_miou": best["miou"], "base_miou": base_miou,
                    "gain": best["miou"] - base_miou},
        "timing": {"train_s": round(train_s, 1), "score_s": round(score_s, 1),
                   "total_s": round(time.time() - t0, 1)},
    }
    print(f"\nfold {args.fold}: best lr={best['lr']} epoch {best['epoch']} at "
          f"{100 * best['miou']:.2f} ({100 * (best['miou'] - base_miou):+.2f} vs base) "
          f"over {len(curve)} grid points")
    dest = export_bundle("s8c_tune", record, out=args.out, dtype=str(adapter.dtype))
    print(f"wrote {dest}")
    return 0


def _takes_group(fn) -> bool:
    import inspect

    try:
        return bool(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
