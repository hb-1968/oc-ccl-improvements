#!/usr/bin/env python3
"""Run per-reference anchored OC-CCL fine-tunes for S10.

Each reference has its own adapter and recorded support identity. This is a
separate experimental reading from one shared dataset-wide fine-tune.
"""

from __future__ import annotations

import argparse
import hashlib
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
from occl.evaluator import (distinct_images, evaluate,  # noqa: E402
                            run_fingerprint, write_per_query_csv)
from occl.finetune.cycle import PROMPT_MODES  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402


def wanted_config(args, anchor: str) -> dict:
    """What an adapter file must have been trained with to be reusable for this arm."""
    return {
        "epochs": args.epochs, "lr": args.lr, "method": "lora",
        "prompt_mode": args.prompt_mode, "normalization": args.normalization,
        "train_mode": args.train_mode, "memory_reset": not args.no_reset,
        "mask_mode": args.mask_mode, "reference_id": anchor,
        "aug_max_deg": args.rotation_aug_deg,
    }


def meta_config(meta: dict) -> dict:
    """The same fields read back out of a checkpoint's own metadata."""
    occ = meta.get("occl") or {}
    aug = occ.get("augmentation") or {}
    return {
        "epochs": meta.get("epochs"), "lr": meta.get("lr"),
        "method": meta.get("method"),
        "prompt_mode": occ.get("prompt_mode"),
        "normalization": occ.get("normalization"),
        "train_mode": occ.get("train_mode"), "memory_reset": occ.get("memory_reset"),
        "mask_mode": occ.get("mask_mode"), "reference_id": occ.get("reference_id"),
        "aug_max_deg": aug.get("max_deg"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beetle", choices=["beetle", "butterfly_major"])
    ap.add_argument("--group", default=None)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shot", type=int, default=1,
                    help="only 1 is implemented: the anchor is THE support of a 1-shot "
                         "run. A 5-shot anchored arm needs a defined multi-reference "
                         "cycle the paper does not describe.")
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--run-start", type=int, default=0)
    ap.add_argument("--run-end", type=int, default=None)
    ap.add_argument("--support-selection", default="canonical_balanced")
    ap.add_argument("--resume-dir", required=True)
    ap.add_argument("--adapter-dir", required=True,
                    help="where per-run adapters live; existing files are validated "
                         "against the wanted config and reused, mismatches retrain")
    # Training axes — same vocabulary as run_finetune.py.
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--prompt-mode", default="logit_reparam", choices=list(PROMPT_MODES))
    ap.add_argument("--normalization", default="imagenet", choices=["imagenet", "unit"])
    ap.add_argument("--train-mode", action="store_true")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--mask-mode", default="per_trait", choices=["merged", "per_trait"],
                    help="default per_trait: this script exists for the paper's protocol")
    ap.add_argument("--rotation-aug-deg", type=float, default=None)
    ap.add_argument("--rotation-aug-rule", default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap steps per fine-tune — smoke tests only")
    ap.add_argument("--gpu-frames", action="store_true")
    ap.add_argument("--stage", default="s10_anchored")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-bundle-adapters", action="store_true",
                    help="leave the N adapter files out of the bundle (~4 MB each)")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    if args.shot != 1:
        ap.error("--shot must be 1 (see help)")

    import torch

    import occl as _occl
    from occl.curation import draw_supports, support_pool
    from occl.finetune.lora import (load_lora_state_dict, lora_delta_norm,
                                    lora_modules, reset_lora)
    from occl.finetune.train import checkpoint_hash, save_lora, train_lora

    print(f"occl build {_occl.BUILD} | {__file__}", flush=True)
    cache = ImageCache(args.cache_root) if args.cache_root else ImageCache()
    ds = {"beetle": Beetle, "butterfly_major": ButterflyMajor}[args.dataset](cache=cache)
    groups = list(ds.groups())
    g = args.group if args.group is not None else (
        groups[0] if len(groups) == 1 else None)
    if g is None:
        ap.error("--group is required on a multi-group dataset")
    run_end = args.n_runs if args.run_end is None else args.run_end
    runs = list(range(args.run_start, run_end))

    pool, _ = support_pool(ds, g, args.support_selection)
    anchors = {i: draw_supports(ds, g, pool, args.support_selection, 1, i)[0]
               for i in runs}
    print(f"anchors ({len(anchors)}): "
          + ", ".join(f"run{i}={a}" for i, a in list(anchors.items())[:4]) + " ...",
          flush=True)

    adapter = SSTSam2Adapter(checkpoint=args.checkpoint,
                             offload_video_to_cpu=not args.gpu_frames,
                             offload_state_to_cpu=not args.gpu_frames)
    dtype = torch.float32 if args.device == "cpu" else None
    print(f"building SAM 2 on {args.device} from {adapter.checkpoint}", flush=True)
    adapter.build(device=args.device, dtype=dtype)
    model = adapter.model

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # phase 1: train (or validate-and-reuse) one adapter per run
    trainings: dict[int, dict] = {}
    adapter_paths: dict[int, Path] = {}
    for i in runs:
        anchor = anchors[i]
        pt = adapter_dir / f"anchored_run{i:02d}.pt"
        adapter_paths[i] = pt
        if pt.exists():
            meta = torch.load(pt, map_location="cpu", weights_only=False).get("meta") or {}
            want, got = wanted_config(args, anchor), meta_config(meta)
            if want == got:
                print(f"  run {i}: reusing {pt.name} (config matches)", flush=True)
                trainings[i] = {"reused": True, "reference_id": anchor}
                continue
            diff = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
            print(f"  run {i}: RETRAINING — {pt.name} config mismatch on {diff}",
                  flush=True)
        t0 = time.time()
        report = train_lora(
            model, ds, lr=args.lr, epochs=args.epochs, group=g, method="lora",
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            train_mode=args.train_mode, normalization=args.normalization,
            prompt_mode=args.prompt_mode, memory_reset=not args.no_reset,
            mask_mode=args.mask_mode, reference_id=anchor,
            rotation_aug_deg=args.rotation_aug_deg,
            device=args.device, dtype=dtype, log_every=0, max_steps=args.max_steps)
        assert report["epochs"] == args.epochs
        meta = {
            "method": "lora", "epochs": args.epochs, "lr": args.lr,
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                     "scopes": ["memory_encoder", "sam_mask_decoder"]},
            "occl": {
                "prompt_mode": args.prompt_mode, "train_mode": args.train_mode,
                "normalization": args.normalization,
                "memory_reset": not args.no_reset, "cycle_len": 2,
                "mask_mode": args.mask_mode,
                "reference": "fixed", "reference_id": anchor, "reference_run": i,
                "augmentation": (None if args.rotation_aug_deg is None else
                                 {"type": "rotation_x0",
                                  "max_deg": args.rotation_aug_deg,
                                  "rule": args.rotation_aug_rule}),
            },
        }
        save_lora(model, pt, meta=meta, method="lora")
        trainings[i] = {
            "reused": False, "reference_id": anchor,
            "delta_norm": report["delta_norm"], "steps": report["steps"],
            "loss_first": report["loss_first"], "loss_last": report["loss_last"],
            "aug": report["aug"], "mask_channels": report["mask_channels"],
            "elapsed_s": report["elapsed_s"],
        }
        print(f"  run {i}: trained on x0={anchor} ||dW||={report['delta_norm']:.4f} "
              f"in {time.time() - t0:.0f}s", flush=True)

    shas = {i: checkpoint_hash(adapter_paths[i]) for i in runs}
    composite = hashlib.sha1("".join(shas[i] for i in runs).encode()).hexdigest()[:8]
    adapter.weights_id = f"{Path(adapter.checkpoint).name}:anchored:{composite}"
    print(f"composite weights_id {adapter.weights_id}", flush=True)

    # Training may have left train-mode or a trained adapter attached; evaluation must
    # start from a defined state every run, and always in eval().
    model.eval()

    # phase 2: encode once, then score run i under adapter i
    keys = distinct_images(ds, g, args.shot, args.n_runs, args.support_selection,
                           args.run_start, run_end)
    print(f"precomputing {len(keys)} embeddings...", flush=True)
    rep = adapter.precompute((k, ds.load_image(k)) for k in keys)
    print(f"  {rep['n_cached']} cached, {rep['total_gb']} GB", flush=True)

    applied: dict[int, dict] = {}
    for i in runs:
        n = reset_lora(model)
        if (z := lora_delta_norm(model)) != 0.0:
            raise SystemExit(f"run {i}: reset {n} adapters but ||dW||={z:.3e}, not 0")
        blob = torch.load(adapter_paths[i], map_location="cpu", weights_only=False)
        mismatches = load_lora_state_dict(model, blob["lora"])
        if mismatches:
            raise SystemExit(f"run {i}: {mismatches} key mismatches loading "
                             f"{adapter_paths[i].name}")
        delta = lora_delta_norm(model)
        if delta == 0.0:
            raise SystemExit(f"run {i}: ||dW|| == 0 after load — would score base as "
                             "fine-tuned")
        if any(name.startswith("image_encoder") for name, _ in lora_modules(model)):
            raise SystemExit("an adapter reaches image_encoder — the shared embedding "
                             "cache would be scoring stale features")
        applied[i] = {"sha": shas[i], "delta_norm": delta,
                      "reference_id": anchors[i]}
        out_i = evaluate(ds, adapter, shot=args.shot, n_runs=args.n_runs, groups=[g],
                         run_start=i, run_end=i + 1,
                         resume_dir=Path(args.resume_dir), precompute=False,
                         support_selection=args.support_selection,
                         keep_embeddings=True, progress=True)
        r = out_i["runs"][0]
        if r["support_ids"] != [anchors[i]]:
            raise SystemExit(
                f"run {i} evaluated with supports {r['support_ids']} but was anchored on "
                f"{anchors[i]} — the draw dispatch and the anchor resolution disagree")
        print(f"  run {i}: mIoU {100 * r['miou']:.2f} (x0={anchors[i]}, "
              f"||dW||={delta:.4f})", flush=True)

    # phase 3: zero-inference merge, guarded
    fp = run_fingerprint(ds, adapter)
    missing = []
    for i in runs:
        slug = (f"{ds.name}__{g.replace(' ', '_').replace('/', '_')}"
                f"__shot{args.shot}__{args.support_selection}__{adapter.name}"
                f"__{fp}__run{i}.json")
        if not (Path(args.resume_dir) / slug).exists():
            missing.append(i)
    if missing:
        raise SystemExit(f"runs {missing} have no banked per-run JSON — the merge pass "
                         "computes nothing and must not fill gaps with whatever weights "
                         "are loaded")
    out = evaluate(ds, adapter, shot=args.shot, n_runs=args.n_runs, groups=[g],
                   run_start=args.run_start, run_end=run_end,
                   resume_dir=Path(args.resume_dir), precompute=False,
                   support_selection=args.support_selection, keep_embeddings=True)
    rows = out.pop("_rows")
    res = out["results"]
    print(f"\n  anchored mIoU {100 * res['miou']:.2f} ± {100 * res['std']:.2f} "
          f"(n={out['protocol']['n_runs']})", flush=True)

    out["model"] = adapter.config_report()
    out["finetune"] = {
        "method": "lora", "epochs": args.epochs, "lr": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "scopes": ["memory_encoder", "sam_mask_decoder"]},
        "anchored": {
            "composite": composite,
            "adapters": [dict(run=i, **applied[i]) for i in runs],
        },
    }
    out["occl"] = {
        "prompt_mode": args.prompt_mode,
        "train_mode": args.train_mode,
        "normalization": args.normalization,
        "memory_reset": not args.no_reset,
        "cycle_len": 2,
        "mask_mode": args.mask_mode,
        "reference": "anchored",
        "reference_id": None,          # per run — see finetune.anchored.adapters
        "reference_run": None,
        "augmentation": (None if args.rotation_aug_deg is None else
                         {"type": "rotation_x0", "max_deg": args.rotation_aug_deg,
                          "rule": args.rotation_aug_rule}),
        "scnp": None,                  # anchored arms never train SCNP
    }
    out["diagnostics"] = {
        "delta_norm": float(min(a["delta_norm"] for a in applied.values())),
        "delta_norms": {str(i): applied[i]["delta_norm"] for i in runs},
        "roundtrip_mismatches": 0,     # enforced per load above, else SystemExit
    }
    out["trainings"] = {str(i): trainings[i] for i in runs}

    csv_path = REPO / f"per_query_{args.stage}.csv"
    write_per_query_csv(rows, csv_path)
    extra = {"per_query.csv": csv_path}
    if not args.no_bundle_adapters:
        for i in runs:
            extra[f"adapters/{adapter_paths[i].name}"] = adapter_paths[i]

    bundle = Path(args.out or f"{args.stage}.zip")
    bundle = export_bundle(args.stage, out, bundle, extra_files=extra,
                           dtype=str(getattr(adapter, "dtype", None)))
    print(f"\nwrote {bundle.resolve()}")
    if not args.no_download:
        from occl.bundle import download

        download(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
