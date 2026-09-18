#!/usr/bin/env python3
"""Run an evaluation stage and emit its gate bundle.

    # CPU sanity check — no GPU, reproduces the paper's Table A3 copy baseline
    python scripts/run_eval.py --stage s3 --adapter copy_baseline --dataset beetle \
        --cache-root ~/Projects/oc-ccl-data/cache --shot 1 --n-runs 20 --local

    # the real thing (Colab)
    python scripts/run_eval.py --stage s4 --adapter sst_sam2 --dataset beetle --shot 1
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl import paths  # noqa: E402
from occl.bundle import download, export_bundle  # noqa: E402
from occl.datasets.beetle import Beetle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.evaluator import evaluate, write_per_query_csv  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402

DATASETS = {"beetle": Beetle, "butterfly_major": ButterflyMajor}


def make_adapter(name: str, checkpoint: str | None, gpu_frames: bool = False,
                 concept_policy: str | None = None):
    if name == "copy_baseline":
        from occl.adapters.copy_baseline import CopyBaselineAdapter

        return CopyBaselineAdapter()
    if name == "sst_sam2":
        from occl.adapters.sst_sam2 import SSTSam2Adapter

        # Constructor arguments, deliberately: `run_fingerprint` hashes the SOURCE of
        # `adapters/sst_sam2.py`, so getting at these through a flag rather than an edit
        # leaves every banked per-run JSON valid. They choose where tensors live, not what
        # is computed with them — `scripts/profile_episode.py` asserts the logits are
        # bit-identical either way.
        return SSTSam2Adapter(checkpoint=checkpoint,
                              offload_video_to_cpu=not gpu_frames,
                              offload_state_to_cpu=not gpu_frames)
    if name == "sam3":
        from occl.adapters.sam3 import Sam3TrackerAdapter

        return Sam3TrackerAdapter(checkpoint=checkpoint)
    if name == "sec":
        from occl.adapters.sec import SeCAdapter

        return SeCAdapter(checkpoint=checkpoint,
                          concept_policy=concept_policy or "off")
    raise SystemExit(f"unknown adapter {name!r}")


def render_overlay(ds, adapter, shot: int, out_path: Path, group: str | None = None,
                   n_episodes: int = 3, rows=None, select: str = "spread") -> Path:
    """Render episodes as support | support mask | query | prediction | GT, one row each.

    The cheapest possible check on a class of bug that no assertion catches: labels
    permuted between traits, a mirrored or transposed mask, or an off-by-one in the trait
    vocabulary all produce perfectly plausible IoU numbers. Look at it once here rather
    than after 10,000 episodes.

    This was beetle-only for a long time, which is exactly backwards: beetle reproduces
    the paper (63.20 vs 61.9 at 1-shot, 73.16 vs 74.2 at 5-shot) while butterfly sits
    ~11 points low at BOTH shot counts. The dataset nobody had eyeballed is the one that
    disagrees, and the one time these images were actually looked at it overturned the
    working model of them entirely — they are dissected wings laid out in quadrants, not
    spread specimens.

    ``group`` picks the subspecies. Default is the first, which for butterfly is cyrbia —
    usually the wrong choice, since cyrbia is the known outlier; pass a mid-scoring group
    like `malleti` to see what the *typical* case looks like. Per-trait IoU is printed
    beneath each row so the picture and the number can be read together.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from occl.metrics import logits_to_labelmap, per_image_miou, resize_logits
    from occl.protocol import build_episodes

    group = group or ds.groups()[0]

    # Which queries to draw. With the run's per-query scores in hand, take the WORST,
    # median and BEST rather than the first N — on a subspecies averaging 47 the first
    # three frames say almost nothing, while the worst one is where a mis-assigned trait
    # or a blown-out extent is plainly visible.
    wanted = None
    if rows and select == "spread":
        per = {}
        for r in rows:
            if r["group"] == group and int(r["run_idx"]) == 0:
                per.setdefault(r["query_id"], []).append(float(r["iou"]))
        scored = sorted(((sum(v) / len(v), q) for q, v in per.items()))
        if len(scored) >= n_episodes:
            idx = [int(round(i * (len(scored) - 1) / (n_episodes - 1)))
                   for i in range(n_episodes)] if n_episodes > 1 else [0]
            wanted = [scored[i][1] for i in idx]
            print("  overlay episodes (worst -> best): "
                  + ", ".join(f"{q} {100 * scored[i][0]:.1f}"
                              for i, q in zip(idx, wanted)))

    eps = []
    for ep in build_episodes(ds, group, shot, 0):
        if wanted is not None:
            if ep.query_id in wanted:
                eps.append(ep)
            if len(eps) == len(wanted):
                break
        else:
            eps.append(ep)
            if len(eps) >= n_episodes:
                break
    eps.sort(key=lambda e: wanted.index(e.query_id) if wanted else 0)

    rows = []
    for ep in eps:
        gt = ds.load_labelmap(ep.query_id)
        logits = adapter.predict(ep)
        pred = logits_to_labelmap(resize_logits(logits, gt.shape), ep.class_ids)
        miou, per_trait = per_image_miou(pred, gt)
        rows.append((ep, gt, pred, miou, per_trait))

    n_cls = max(max(int(max(e.class_ids, default=1)) for e in eps),
                max(int(g.max()) for _, g, _, _, _ in rows), 1)
    lab_kw = {"cmap": "nipy_spectral", "vmin": 0, "vmax": n_cls, "interpolation": "nearest"}

    fig, axes = plt.subplots(len(rows), 5, figsize=(20, 4.6 * len(rows)), squeeze=False)
    for r, (ep, gt, pred, miou, per_trait) in enumerate(rows):
        panels = [
            (ep.support_images[0], None, f"support {ep.support_ids[0]}"),
            (None, ep.support_labelmaps[0], "support mask"),
            (ep.query_image, None, f"query {ep.query_id}"),
            (None, pred, f"prediction — mIoU {100 * (miou or 0):.1f}"),
            (None, gt, "ground truth"),
        ]
        for ax, (img, lab, title) in zip(axes[r], panels):
            ax.imshow(img if img is not None else lab,
                      **({} if img is not None else lab_kw))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        axes[r][3].set_xlabel("")
        per = "  ".join(f"t{t}:{100 * v:.0f}" for t, v in sorted(per_trait.items()))
        axes[r][0].text(0, -0.08, f"per-trait IoU  {per}", transform=axes[r][0].transAxes,
                        fontsize=9, va="top")
    fig.suptitle(f"{ds.name} / {group} — {shot}-shot, classes {list(eps[0].class_ids)}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def verify_embedding_cache(ds, adapter, shot: int) -> dict:
    """Assert the embedding cache is numerically exact, not merely close.

    The cache replaces ~95% of the model's compute, so if it drifted at all, every number
    downstream would be quietly wrong in a way no gate would catch. Verified bit-identical
    on CPU during development; re-checked here on the actual GPU and dtype, because bf16
    accumulation order is a plausible way for that to stop being true.
    """
    import numpy as np

    from occl.protocol import build_episodes

    group = ds.groups()[0]
    ep = next(build_episodes(ds, group, shot, 0))
    adapter.clear_embeddings()
    uncached = adapter.predict(ep)
    adapter.precompute((k, ds.load_image(k))
                       for k in {ep.query_id, *ep.support_ids})
    cached = adapter.predict(ep)
    adapter.clear_embeddings()

    identical = bool(np.array_equal(uncached, cached))
    maxdiff = float(np.abs(uncached - cached).max())
    print(f"  embedding cache: identical={identical} maxdiff={maxdiff:.3e}")
    if not identical:
        print("  WARNING: cached and uncached predictions differ — investigate before "
              "trusting any number from this run.")
    return {"identical": identical, "max_abs_diff": maxdiff}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--adapter", default="sst_sam2",
                    choices=["sst_sam2", "copy_baseline", "sam3", "sec"])
    ap.add_argument("--concept-policy", choices=["off", "forced"], default="off",
                    help="S14, sec only: whether the LVLM concept embedding is "
                         "injected on every episode (`forced` — the primary arm; the "
                         "native HSV-histogram gate is blind to 180-degree flips and "
                         "same-species pairs, so `native` would silently measure a "
                         "retuned SAM 2.1) or never (`off` — the companion arm, which "
                         "still records the would-fire statistics per episode)")
    ap.add_argument("--dataset", default="beetle", choices=sorted(DATASETS))
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--lora", default=None,
                    help="S8c: adapter weights from run_finetune.py. Applied to the base "
                         "checkpoint before scoring; loading is strict.")
    ap.add_argument("--lora-r", type=int, default=None,
                    help="override the rank recorded in the adapter file")
    ap.add_argument("--lora-alpha", type=float, default=None)
    # Carried into the record so `registry.classify` can judge the FINE-TUNE, not just the
    # evaluation. They describe how the adapters were produced, which the adapter file's
    # own metadata also records — these let a mislabelled run be caught rather than assumed.
    ap.add_argument("--lora-epochs", type=int, default=1)
    ap.add_argument("--lora-prompt-mode", default="logit_reparam",
                    choices=["logit_reparam", "hard_threshold"])
    ap.add_argument("--lora-no-reset", action="store_true")
    # A1 and A5 must reach the SCORING record too, not only the training one. Without them
    # `occl_unit` and `occl_train` land in a combined bundle with identical `occl` blocks,
    # distinguishable only by ||dW|| and a filename — so the record cannot say what was
    # varied, which is the same defect as `finetune: None` one layer up.
    ap.add_argument("--lora-normalization", default="imagenet",
                    choices=["imagenet", "unit"],
                    help="A1, as the checkpoint was TRAINED. Scoring always uses the "
                         "inference pipeline; this records the training condition.")
    ap.add_argument("--lora-train-mode", action="store_true",
                    help="A5: the checkpoint was trained under `model.train()`.")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--run-start", type=int, default=0,
                    help="first run index this worker owns (parallel fan-out)")
    ap.add_argument("--run-end", type=int, default=None,
                    help="one past the last run index; default n_runs. Workers with "
                         "disjoint (groups, run-range) share one --resume-dir safely, "
                         "since the slug carries both. Beetle has a single universal "
                         "group, so the run range is its ONLY parallel axis.")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--resume-dir", default=None,
                    help="per-run JSON lands here; point at Drive to survive a session end")
    ap.add_argument("--local", action="store_true",
                    help="write the bundle straight to results/inbox/ and skip the download")
    ap.add_argument("--overlay", action="store_true",
                    help="render episodes (support/mask/query/pred/GT) for visual check")
    ap.add_argument("--overlay-group", default=None,
                    help="which group to render. Defaults to the first, which for "
                         "butterfly is cyrbia — usually the WRONG choice, since cyrbia is "
                         "the known outlier. Pass a mid-scoring group to see the typical case.")
    ap.add_argument("--overlay-episodes", type=int, default=3,
                    help="rows in the overlay; one episode each")
    ap.add_argument("--overlay-select", default="spread", choices=["spread", "first"],
                    help="spread: worst / median / best scoring query, which is where "
                         "failure modes are visible. first: the first N episodes.")
    ap.add_argument("--verify-cache", action="store_true",
                    help="assert cached embeddings reproduce uncached predictions exactly")
    ap.add_argument("--no-precompute", action="store_true",
                    help="disable the embedding cache (much slower; for A/B only)")
    ap.add_argument("--gpu-frames", action="store_true",
                    help="keep video frames and tracking state on the GPU instead of "
                         "offloading to host. The embedding cache skips the image encoder "
                         "but still re-uploads a 12.6 MB frame on every "
                         "`_get_image_feature` call (57 of them per butterfly 5-shot "
                         "episode) that SAM 2 then discards; on-device frames make that a "
                         "no-op and move normalisation off the CPU. Placement only — "
                         "verify with scripts/profile_episode.py before reporting from it.")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation over support orientation: predict with "
                         "the support as given and rotated 180 deg, keep the more "
                         "confident. Targets the 37%% of cyrbia queries that are upside "
                         "down and score ~0. Modified track; ~2x inference.")
    ap.add_argument("--support-selection", choices=["uniform", "canonical", "canonical_balanced", "canonical_placement", "orientation_diverse"],
                    default="uniform",
                    help="canonical approximates the paper's reference curation: "
                         "train-only orientation + trait completeness (occl.curation)")
    ap.add_argument("--out", default=None,
                    help="bundle path (default: <stage>.zip in the cwd)")
    ap.add_argument("--no-download", action="store_true",
                    help="skip the Colab download; the notebook does it from the kernel, "
                         "since files.download() cannot work from a subprocess")
    args = ap.parse_args()

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    cache = ImageCache(args.cache_root)
    ds = DATASETS[args.dataset](cache=cache)

    # Preflight: does the cache hold every image this arm will ask for?
    #
    # Curation filters the support *pool*, and `sample_supports` permutes whatever pool it
    # is given, so removing even one image reshuffles every draw. A cache built for
    # `uniform` is missing 87 of the images the placement-curated 1-shot arm wants. Without
    # this check that surfaces as a mid-run cache miss after the 898 MB checkpoint load and
    # a chunk of A100 time; with it, it fails in seconds and says exactly what to run.
    from occl.evaluator import distinct_images

    missing = []
    for group in list(args.groups or ds.groups()):
        for image_id in distinct_images(ds, group, args.shot, args.n_runs,
                                        args.support_selection,
                                        args.run_start, args.run_end):
            if not cache.has(ds.name, image_id):
                missing.append(image_id)
    if missing:
        raise SystemExit(
            f"\n{len(missing)} image(s) required by --support-selection "
            f"{args.support_selection} are not in the cache at {args.cache_root}.\n"
            f"  first few: {missing[:8]}\n\n"
            "Curation changes which supports get drawn, so a cache built for another arm "
            "does not cover this one. Top it up locally (no compute units, resumable):\n\n"
            f"  python scripts/build_cache.py --cache-root <cache> --originals-root <originals> \\\n"
            f"      --datasets {args.dataset} --selections {args.support_selection}\n\n"
            "then re-upload the cache. Omit --selections to cover every arm at once."
        )

    adapter = make_adapter(args.adapter, args.checkpoint, gpu_frames=args.gpu_frames,
                           concept_policy=args.concept_policy)
    adapter.build(device=device)
    if args.gpu_frames:
        print("  frames and tracking state kept on the GPU (--gpu-frames)")

    # S8c: score a fine-tuned model. The adapters are a few MB stored separately from the
    # 2.3 GB base checkpoint, so they are applied here rather than shipped as a full
    # checkpoint. Loading is STRICT — a partial load would score a model that is part
    # fine-tuned and part base, and every number after it would be uninterpretable.
    lora_report = None
    if args.lora:
        import torch

        from occl.finetune.lora import apply_lora, load_lora_state_dict, lora_delta_norm

        # Accept either the raw .pt or the S8b bundle that carries it, so the artifact you
        # downloaded is directly usable and nobody has to unzip by hand and guess which
        # file inside is the right one.
        lora_src = Path(args.lora)
        if lora_src.suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(lora_src) as z:
                if "lora.pt" not in z.namelist():
                    raise SystemExit(
                        f"{lora_src.name} has no lora.pt — it predates the fix that puts "
                        f"the adapters in the bundle (contains: {z.namelist()}). Re-run "
                        f"S8b at this learning rate; it takes ~45 s."
                    )
                extracted = Path(tempfile.mkdtemp()) / "lora.pt"
                extracted.write_bytes(z.read("lora.pt"))
                want_sha = json.loads(z.read("record.json")).get(
                    "finetune", {}).get("lora_sha")
            from occl.finetune.train import checkpoint_hash

            got_sha = checkpoint_hash(extracted)
            if want_sha and got_sha != want_sha:
                raise SystemExit(
                    f"{lora_src.name}: lora.pt hashes {got_sha}, record says {want_sha}. "
                    "The adapters do not match the training run that describes them."
                )
            lora_src = extracted
            print(f"  LoRA extracted from {Path(args.lora).name} (sha {got_sha})")

        # One loader for both methods — see `finetune.train.load_finetune`. The method is
        # read from the checkpoint's own metadata, never inferred from the flag name: a
        # full fine-tune pushed through the LoRA path would mismatch on every key.
        from occl.finetune.train import load_finetune

        rep = load_finetune(adapter.model, lora_src,
                            lora_r=args.lora_r or 32, lora_alpha=args.lora_alpha or 64.0)
        if rep["mismatches"]:
            raise SystemExit(
                f"{args.lora}: {rep['mismatches']} key mismatch(es) loading a "
                f"{rep['method']} checkpoint. Scoring would mix fine-tuned and base "
                f"weights."
            )
        delta = rep["delta_norm"]
        if delta == 0.0:
            raise SystemExit(
                f"{args.lora}: ||dW|| is exactly 0 — the weights are unchanged from base, "
                "so this would score the BASE model while labelling it fine-tuned."
            )
        lora_report = {"path": str(args.lora), "method": rep["method"],
                       "delta_norm": delta, "roundtrip_mismatches": rep["mismatches"],
                       "sha": None, "meta": rep.get("meta") or {}}
        if rep["method"] == "full":
            lora_report["n_tensors"] = rep["n_tensors"]
            print(f"  full fine-tune: {rep['n_tensors']} tensors, ||dW||={delta:.6f}")
        else:
            lora_report |= {"r": rep["r"], "alpha": rep["alpha"],
                            "modules": rep["modules"]}
            print(f"  LoRA: {rep['modules']} modules, r={rep['r']} "
                  f"alpha={rep['alpha']}, ||dW||={delta:.6f}")

    # Feeds `evaluator.run_fingerprint`, and it is load-bearing: S8c's arms differ ONLY in
    # weights, so without this they share a resume slug and the later arms silently return
    # the earlier one's cached per-run JSON. `||dW||` rather than the file sha, so two
    # different files with identical adapters correctly share a cache and a re-trained
    # adapter never collides with its predecessor.
    # The copy baseline has no checkpoint — it broke here the day this stamping was
    # added and nobody re-ran it until the boundary-tolerance check needed it.
    _ckpt = getattr(adapter, "checkpoint", None)
    _ckpt_name = Path(_ckpt).name if _ckpt else adapter.name
    adapter.weights_id = (
        f"{_ckpt_name}:base" if lora_report is None
        else f"{_ckpt_name}:"
             f"{lora_report['method']}{lora_report['delta_norm']:.8f}"
    )

    if args.tta:
        from occl.adapters.tta import OrientationTTAAdapter

        adapter = OrientationTTAAdapter(adapter)
        print("  test-time augmentation: support orientation (upright vs rot180)")

    print(f"{args.stage}: {args.dataset} | {args.adapter} | shot={args.shot} | "
          f"{args.n_runs} runs | supports={args.support_selection} | device={device}")
    print(f"  config: {adapter.config_report()}")

    sdpa = None
    if args.verify_cache:
        from occl.diagnostics import sdpa_backend_report
        sdpa = sdpa_backend_report(device, getattr(adapter, 'dtype', None))
        if sdpa.get('available'):
            print(f"  sdpa on {sdpa['gpu']} sm_{sdpa['capability'][0]}{sdpa['capability'][1]} "
                  f"(sam2 flash gate: {sdpa['sam2_use_flash_attn']})")
            for phase in ('forward', 'backward'):
                for shape, res in sdpa.get(phase, {}).items():
                    print(f"    {phase:8s} {shape:30s} "
                          + '  '.join(f'{b}={v if v=="ok" else "FAIL"}' for b, v in res.items()))

    cache_check = None
    if args.verify_cache and hasattr(adapter, "precompute"):
        cache_check = verify_embedding_cache(ds, adapter, args.shot)

    overlay_path = None
    out = evaluate(ds, adapter, shot=args.shot, n_runs=args.n_runs, groups=args.groups,
                   run_start=args.run_start, run_end=args.run_end,
                   resume_dir=Path(args.resume_dir) if args.resume_dir else None,
                   precompute=not args.no_precompute,
                   support_selection=args.support_selection)
    rows = out.pop("_rows")

    r = out["results"]
    print(f"\n  mIoU {100 * r['miou']:.2f} +- {100 * r['std']:.2f} "
          f"(sem {100 * r['sem']:.2f}, n={out['protocol']['n_runs']})")
    print(f"  model-space mIoU {100 * r['miou_model_space']:.2f}")
    for g, v in r["per_group"].items():
        print(f"    {g:34s} {100 * v['miou']:6.2f} +- {100 * v['std']:5.2f}")

    if args.overlay:
        # Rendered AFTER evaluation so the run's own per-query scores pick the episodes.
        # Showing the first three is close to useless on a subspecies scoring 47: the
        # informative frames are the worst and best cases, which is where a wrong-trait
        # assignment or an extent blow-out actually shows.
        overlay_path = REPO / f"overlay_{args.stage}.png"
        render_overlay(ds, adapter, args.shot, overlay_path,
                       group=args.overlay_group, n_episodes=args.overlay_episodes,
                       rows=rows, select=args.overlay_select)
        print(f"  overlay -> {overlay_path}  (EYEBALL THIS before trusting the numbers)")

    csv_path = REPO / f"per_query_{args.stage}.csv"
    write_per_query_csv(rows, csv_path)

    out["model"] = adapter.config_report()
    # Actual VRAM this worker used. The fan-out planner has been sizing workers from a
    # hand-maintained table whose numbers came from peak totals reported mid-precompute —
    # which overstated steady state by ~2.5x and left most of the card idle. Recording it
    # makes the table measured instead of guessed.
    try:
        import torch

        if torch.cuda.is_available():
            out["vram"] = {
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                "capacity_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
            print(f"  worker VRAM peak: {out['vram']['peak_reserved_gb']:.2f} GB reserved "
                  f"/ {out['vram']['peak_allocated_gb']:.2f} GB allocated")
    except Exception:  # noqa: BLE001 — instrumentation must never fail a run
        pass
    if args.overlay:
        # Which dataset/group the human actually looked at. `overlay_reviewed` alone is
        # ambiguous once more than one dataset gets a smoke stage — a beetle review must
        # not read as clearance for butterfly.
        out["overlay"] = {"dataset": args.dataset,
                          "group": args.overlay_group or ds.groups()[0],
                          "shot": args.shot, "n_episodes": args.overlay_episodes}
    if lora_report is not None:
        # `finetune` and `occl` must reach the record, or `registry.classify` cannot see
        # that a fine-tune happened at all and a fine-tuned run classifies exactly like the
        # base — the same crack `data.interp`, `data.cache_format` and `model.tta` each
        # fell through. `classify` folds in FAITHFUL_FINETUNE whenever `finetune` is
        # present, so these blocks are what make an unfaithful fine-tune visible.
        m = lora_report
        # The checkpoint's own metadata is the source of truth for what produced it —
        # `save_lora` stamps the full finetune+occl blocks — and the flags are fallbacks
        # for pre-axis checkpoints that carry no meta. A record populated from flags alone
        # can claim a configuration the training never used; the meta cannot.
        meta = m["meta"]
        meta_occl = meta.get("occl") or {}
        out["finetune"] = dict(
            {"method": m["method"], "epochs": meta.get("epochs", args.lora_epochs),
             "lora_path": m["path"], "lora_sha": m["sha"]},
            **({"frozen": ["image_encoder"], "n_tensors": m["n_tensors"],
                "delta_norm": m["delta_norm"]} if m["method"] == "full" else
               {"lora": {"r": m["r"], "alpha": m["alpha"],
                         "scopes": (meta.get("lora") or {}).get(
                             "scopes", ["memory_encoder", "sam_mask_decoder"]),
                         "modules": m["modules"], "delta_norm": m["delta_norm"]}}))
        out["occl"] = {
            "prompt_mode": meta_occl.get("prompt_mode", args.lora_prompt_mode),
            "memory_reset": meta_occl.get("memory_reset", not args.lora_no_reset),
            "normalization": meta_occl.get("normalization", args.lora_normalization),
            "train_mode": meta_occl.get("train_mode", args.lora_train_mode),
            "cycle_len": 2,
            # "binary" is the pre-axis name for the merged silhouette; new checkpoints
            # say "merged" or "per_trait" for themselves.
            "mask_mode": meta_occl.get("mask_mode", "binary"),
            "reference": meta_occl.get("reference", "derangement"),
            "reference_id": meta_occl.get("reference_id"),
            "reference_run": meta_occl.get("reference_run"),
            "augmentation": meta_occl.get("augmentation"),
            "scnp": meta_occl.get("scnp"),
            # S13 knobs, re-lifted from the checkpoint's own metadata like scnp — a
            # scoring record that cannot show the deviation would classify an S13 arm
            # as a plain per-trait fine-tune.
            "excl": meta_occl.get("excl"),
            "xovl": meta_occl.get("xovl"),
            "bshape": meta_occl.get("bshape"),
            "margin": meta_occl.get("margin"),
            # S17 knobs, same contract: the scoring record names the arm's objective
            "found_mask": meta_occl.get("found_mask"),
            "bce_w": meta_occl.get("bce_w"),
            "objness": meta_occl.get("objness"),
            "lora_legs": meta_occl.get("lora_legs"),
            "soft_sentinel": meta_occl.get("soft_sentinel"),
            "close_aug": meta_occl.get("close_aug"),
            "recipe": meta_occl.get("recipe"),
        }
        out["diagnostics"] = dict(out.get("diagnostics") or {}, **{
            "delta_norm": m["delta_norm"],
            "roundtrip_mismatches": m["roundtrip_mismatches"],
        })
    if cache_check:
        out["cache_check"] = cache_check
    if sdpa:
        from occl.diagnostics import allow_all_kernels_state
        sdpa["allow_all_kernels_after_run"] = allow_all_kernels_state()
        out["sdpa"] = sdpa
    # S3 only. Deliberately not auto-set: a human has to look at the overlay, and a flag
    # the code can flip on its own would certify nothing.
    out["overlay_reviewed"] = False
    extra = {"per_query.csv": csv_path}
    if overlay_path:
        extra["overlay.png"] = overlay_path

    if args.local:
        bundle = paths.inbox_root() / f"{args.stage}.zip"
    else:
        bundle = Path(args.out or f"{args.stage}.zip")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle = export_bundle(args.stage, out, bundle, extra_files=extra,
                           dtype=str(getattr(adapter, "dtype", None)))
    print(f"\nwrote {bundle.resolve()}")
    if args.local:
        print(f"next:  python scripts/check_gate.py {args.stage}")
    elif not args.no_download:
        download(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
