#!/usr/bin/env python3
"""Replay a fine-tune's training pairs with base and adapted weights.

Record forward and closing predictions/losses on the same pairs. These are
training-pair diagnostics, separate from held-out test performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.adapters.sst_sam2 import SSTSam2Adapter  # noqa: E402
from occl.bundle import export_bundle  # noqa: E402
from occl.datasets.base import Episode, union_class_ids  # noqa: E402
from occl.datasets.beetle import Beetle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.finetune.cycle import PROMPT_MODES  # noqa: E402
from occl.finetune.probe import (CARRIER_RECIPE, FORMULAS, OBJECTIVES,  # noqa: E402
                                 REPLAY_PROMPT_MODES, base_arm_label, base_variant_of,
                                 cycle_probe, cyclepairs_kwargs, fmt_per_trait,
                                 loss_parity, loss_variants, macro_over, objective_of,
                                 oc_ccl_kwargs_from_recipe, own_loss, own_prompt_is_hard,
                                 recipe_from_lora_meta, replay_checks, replay_subset,
                                 score_leg, strip_objective, write_labelmap_png,
                                 y0_labelmap)
from occl.metrics import resize_logits  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402

CSV_FIELDS = [
    "dataset", "group", "train_step", "ref", "query", "class_ids", "n_channels", "arm",
    "aug_angle",
    "fwd_miou", "fwd_miou_prompted", "fwd_iou_binary", "fwd_n_gt_traits",
    "fwd_n_prompted_present", "fwd_per_trait", "fwd_found", "fwd_found_count",
    "fwd_claimed_frac",
    "cls_miou", "cls_iou_binary", "cls_per_trait", "cls_claimed_frac",
    "loss", "bce", "dice", "ce", "dice_per_channel", "bce_per_channel",
    "fwd_obj", "cls_obj", "lora_legs", "soft_sentinel", "fwd_fp_frac", "fwd_n_absent_prompted",
    *[f"loss_{o}" for o in OBJECTIVES if o != "plain"],
    # S18: the arm's own prompt mode / closing-end rotation, and the closing leg under
    # the BINARISED forward logits (the steganography column)
    "prompt_mode", "close_aug", "close_angle",
    "cls_miou_hard", "cls_iou_binary_hard", "cls_per_trait_hard", "cls_claimed_frac_hard",
    "cls_obj_hard", "loss_hard", "bce_hard", "dice_hard",
    *[f"loss_{o}_hard" for o in FORMULAS if o != "plain"],
    "fwd_map", "cls_map", "cls_map_hard",
]


def build_dataset(name: str, cache_root):
    if not cache_root:
        raise SystemExit("--cache-root is required: the pixel cache lives outside the repo")
    return {"beetle": Beetle,
            "butterfly_major": ButterflyMajor}[name](cache=ImageCache(cache_root))


def resolve_lora(path: str) -> Path:
    """A raw `.pt`, or a training bundle whose `lora.pt` is extracted (run_arms.py)."""
    p = Path(path)
    if p.suffix != ".zip":
        return p
    with zipfile.ZipFile(p) as z:
        if "lora.pt" not in z.namelist():
            raise SystemExit(f"{p.name} has no lora.pt (contains {z.namelist()})")
        out = Path(tempfile.mkdtemp()) / "lora.pt"
        out.write_bytes(z.read("lora.pt"))
    return out


def read_record(path) -> dict:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("record.json"))


def group_index(ds) -> dict[str, str]:
    """image id -> group, over the train split (a pair's group is its reference's)."""
    out = {}
    for g in ds.groups():
        try:
            ids = ds.train_ids(g)
        except TypeError:
            ids = ds.train_ids()
        for i in ids:
            out[i] = g
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="beetle", choices=["beetle", "butterfly_major"])
    ap.add_argument("--groups", nargs="*", default=None,
                    help="probe only pairs whose reference is in these groups (global "
                         "step 0 is always included — it carries the replay proof)")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    ap.add_argument("--family", default="sam2", choices=["sam2"],
                    help="S17 probes SAM 2; the recipe machinery is family-neutral but "
                         "the sam3 tracker has not been exercised here")
    ap.add_argument("--arms", action="append", required=True, metavar="label=path",
                    help="empty path means the base model; repeatable")
    ap.add_argument("--recipe-from", default=None,
                    help="adapter .pt whose meta defines the replay (required with any "
                         "adapter arm; every adapter arm must share it)")
    ap.add_argument("--train-record", action="append", default=None, metavar="[label=]zip",
                    help="a fine-tune bundle (s17ft_*.zip); repeatable, one per adapter arm "
                         "as label=zip. Enables the replay checks and the step-0 loss "
                         "parity proof under EACH arm's own objective")
    ap.add_argument("--epoch", type=int, default=0, help="which epoch's pair order")
    # commissioning path: no adapter yet, so the carrier recipe comes from flags
    ap.add_argument("--mask-mode", default=None, choices=["merged", "per_trait"])
    ap.add_argument("--normalization", default=None, choices=["imagenet", "unit", "half"])
    ap.add_argument("--prompt-mode", default=None, choices=list(PROMPT_MODES))
    ap.add_argument("--no-reset", action="store_true",
                    help="reuse the forward leg's memory in the reverse leg (A7 off)")
    ap.add_argument("--max-pairs", type=int, default=None, help="smoke: first N pairs")
    ap.add_argument("--steps", default=None,
                    help="probe only these global train steps (comma list) — a local read "
                         "of chosen pairs; step 0 is always included")
    ap.add_argument("--eval-parity", type=int, default=1,
                    help="per group, how many pairs also go through adapter.predict on "
                         "the same 1-shot episode (recorded, never gated): the "
                         "training-leg vs inference-path gap on identical images")
    ap.add_argument("--save-maps", default="all", choices=["all", "none"])
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--stage", default="s17probe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch

    import occl as _occl
    from occl.finetune.lora import lora_delta_norm, reset_lora
    from occl.finetune.train import _make_tracker, checkpoint_hash, load_finetune
    from occl.finetune.pairs import CyclePairs

    print(f"occl build {_occl.BUILD} | {__file__}", flush=True)
    t_start = time.time()
    ds = build_dataset(args.dataset, args.cache_root)
    out_path = Path(args.out or f"{args.stage}_{ds.name}.zip")

    # arms
    arms: list[tuple[str, str | None]] = []
    for spec in args.arms:
        label, _, path = spec.partition("=")
        if not label:
            ap.error(f"--arms {spec!r}: needs label=path")
        arms.append((label, path or None))
    if len({a for a, _ in arms}) != len(arms):
        ap.error("arm labels must be unique")
    if not any(p is None for _, p in arms):
        ap.error("a base arm (label=) is required: it is the 'before' every delta reads")
    if sum(1 for _, p in arms if p is None) != 1:
        ap.error("exactly one base arm: the soft-gate base variants are derived from the "
                 "adapters' own recipes, never passed")
    arms.sort(key=lambda kv: kv[1] is not None)          # base first, always
    if any(p is not None for _, p in arms) and not args.recipe_from:
        ap.error("--recipe-from is required when an adapter arm is given: the replay "
                 "must be read from the checkpoint's own metadata, never from flags")
    # each adapter's OWN base-variant axes — the soft gate (S17d) and the closing-end
    # rotation (S18) — and its own prompt mode, read from its meta up front: the base
    # weights are probed once per distinct (gate, closing end), and those variants must
    # be scheduled before any adapter is attached
    arm_var: dict[str, tuple[float | None, float | None]] = {}
    arm_prompt: dict[str, str] = {}
    for label, path in arms:
        if path is None:
            arm_var[label] = (None, None)
            arm_prompt[label] = REPLAY_PROMPT_MODES["continuous"]
            continue
        meta = (torch.load(resolve_lora(path), map_location="cpu",
                           weights_only=False).get("meta") or {})
        _r = recipe_from_lora_meta(meta)
        arm_var[label] = base_variant_of(_r)
        arm_prompt[label] = _r["prompt_mode"]
        if _r["prompt_mode"] not in PROMPT_MODES:
            ap.error(f"[{label}] unknown prompt_mode {_r['prompt_mode']!r} in the adapter meta")
    base_label = next(a for a, p in arms if p is None)
    variants = sorted({v for v in arm_var.values() if v != (None, None)},
                      key=lambda v: (v[0] or 0.0, v[1] or 0.0))
    base_variants = [(base_arm_label(base_label, *v), None) for v in variants]
    for (lbl, _), v in zip(base_variants, variants):
        if lbl in arm_var:
            ap.error(f"arm label {lbl!r} collides with a derived base variant")
        arm_var[lbl] = v
        arm_prompt[lbl] = REPLAY_PROMPT_MODES["continuous"]
    arms = [(base_label, None), *base_variants, *[(a, p) for a, p in arms if p is not None]]
    base_of = {a: base_arm_label(base_label, *arm_var[a]) for a, p in arms if p is not None}
    arm_gate = {a: v[0] for a, v in arm_var.items()}          # the soft gate, per arm

    # the recipe
    if args.recipe_from:
        blob = torch.load(resolve_lora(args.recipe_from), map_location="cpu",
                          weights_only=False)
        recipe = recipe_from_lora_meta(blob.get("meta") or {})
        recipe_source = str(args.recipe_from)
        for flag in ("mask_mode", "normalization", "prompt_mode"):
            v = getattr(args, flag)
            if v is not None and v != recipe[flag]:
                ap.error(f"--{flag.replace('_', '-')} {v!r} contradicts the adapter's "
                         f"recipe ({recipe[flag]!r}); drop the flag")
        if args.no_reset and recipe["memory_reset"]:
            ap.error("--no-reset contradicts the adapter's recipe (memory_reset=True)")
        del blob
    else:
        recipe = dict(CARRIER_RECIPE)
        recipe_source = "flags"
        if args.mask_mode:
            recipe["mask_mode"] = args.mask_mode
        if args.normalization:
            recipe["normalization"] = args.normalization
        if args.prompt_mode:
            recipe["prompt_mode"] = args.prompt_mode
        if args.no_reset:
            recipe["memory_reset"] = False
    if recipe["family"] != args.family:
        raise SystemExit(f"adapter family {recipe['family']!r} != --family {args.family!r}")
    if recipe.get("reference") == "fixed" and not recipe.get("reference_id"):
        raise SystemExit("recipe says reference=fixed but carries no reference_id")
    # The step runs under the PLAIN objective for every arm — the legs are knob-free —
    # and every S17 objective is scored post hoc from the closing logits
    # (`loss_variants`), so arms of different objectives share one replay. S18: the
    # shared replay's prompt is the CONTINUOUS one for every arm, and the binarised
    # closing rides along in the same call (`hard_replay`); an arm's own prompt mode
    # only decides which of the two columns is its own.
    oc_kw = oc_ccl_kwargs_from_recipe(strip_objective(recipe))
    oc_kw["prompt_mode"] = REPLAY_PROMPT_MODES["continuous"]
    print(f"recipe ({recipe_source}): mask={recipe['mask_mode']} "
          f"norm={recipe['normalization']} prompt={recipe['prompt_mode']} "
          f"reset={recipe['memory_reset']} ref={recipe['reference']} "
          f"aug={recipe['augmentation']} legs={oc_kw} | objective of the adapter: "
          f"{objective_of(recipe)}; scored post hoc: {OBJECTIVES}; closing replayed "
          f"under {REPLAY_PROMPT_MODES}", flush=True)

    # fine-tune records: label=zip per adapter arm (a bare zip serves the replay checks)
    train_recs: dict[str, dict] = {}
    for spec in args.train_record or []:
        label, sep, path = spec.partition("=")
        if not sep:
            label, path = "", spec
        rec = read_record(path)
        if not rec.get("training"):
            raise SystemExit(f"{path} has no training block")
        rec["_path"] = path
        train_recs[label] = rec
    training = next(iter(train_recs.values()))["training"] if train_recs else None

    # model
    adapter = SSTSam2Adapter(checkpoint=args.checkpoint)
    dtype = torch.float32 if args.device == "cpu" else None
    print(f"building {args.family} on {args.device} from {adapter.checkpoint}", flush=True)
    adapter.build(device=args.device, dtype=dtype)
    model = adapter.model
    tracker = _make_tracker(model, args.family)
    cuda = str(args.device).startswith("cuda")
    autocast_dtype = torch.bfloat16 if cuda else torch.float32

    # the pairs: dataset-wide, exactly as train_lora built them
    pair_kw = cyclepairs_kwargs(recipe)
    pair_kw.pop("close_aug_deg", None)             # per arm, below
    pairs = CyclePairs(ds, None, epoch=args.epoch, **pair_kw)
    # S18: one CyclePairs per distinct closing-end rotation, for `closing(i)` only —
    # the same pairs (asserted), seeded per pair exactly as the arm's training was
    close_pairs: dict[float, CyclePairs] = {}
    for _soft, _cdeg in set(arm_var.values()):
        if _cdeg is not None and _cdeg not in close_pairs:
            close_pairs[_cdeg] = CyclePairs(ds, None, epoch=args.epoch, close_aug_deg=_cdeg,
                                            **pair_kw)
            assert close_pairs[_cdeg].pairs == pairs.pairs
    replay = replay_checks(ds, pairs, training, recipe["mask_size"])
    replay |= {"epoch": args.epoch, "recipe_source": recipe_source}
    grp = group_index(ds)
    indices = list(range(len(pairs)))
    if args.groups:
        want = set(args.groups)
        unknown = want - set(ds.groups())
        if unknown:
            raise SystemExit(f"unknown group(s) {sorted(unknown)}; have {list(ds.groups())}")
        indices = [i for i in indices if grp.get(pairs.pairs[i][0]) in want]
    if args.max_pairs is not None:
        indices = indices[:args.max_pairs]
    if args.steps:
        want_steps = {int(x) for x in args.steps.split(",") if x.strip()}
        bad = sorted(want_steps - set(range(len(pairs))))
        if bad:
            raise SystemExit(f"--steps {bad} outside 0..{len(pairs) - 1}")
        indices = [i for i in indices if i in want_steps]
    if 0 not in indices:
        indices = [0] + indices              # the replay proof lives on global step 0
    replay |= {"n_probed": len(indices), "train_steps": indices,
               "groups": args.groups or list(ds.groups()), "max_pairs": args.max_pairs,
               "steps": args.steps}
    print(f"{len(pairs)} training pairs rebuilt ({pairs.reference_mode}); probing "
          f"{len(indices)} with {len(arms)} arm(s): {[a for a, _ in arms]}", flush=True)
    for k, v in replay["checks"].items():
        print(f"  replay check {k}: {v}")

    # which pairs also get the inference-path read
    parity_idx: set[int] = set()
    if args.eval_parity:
        seen: dict[str, int] = {}
        for i in indices:
            g = grp.get(pairs.pairs[i][0], "all")
            if seen.get(g, 0) < args.eval_parity:
                parity_idx.add(i)
                seen[g] = seen.get(g, 0) + 1

    # arm switching (run_arms.py semantics)
    applied = False
    arm_info: dict[str, dict] = {}
    arm_legs: dict[str, str] = {}        # which weights each arm's CLOSING legs run under

    def set_weights(label: str, path: str | None) -> float:
        nonlocal applied
        if path is None:
            if applied:
                n = reset_lora(model)
                delta = lora_delta_norm(model)
                if delta != 0.0:
                    raise SystemExit(f"[{label}] reset {n} adapters but ||dW|| == {delta:.3e}, "
                                     "not 0 — the base arm would run fine-tuned weights")
                print(f"  base arm: reset {n} adapters, ||dW|| = 0", flush=True)
            arm_info[label] = {"path": None, "delta_norm": 0.0, "lora_sha": None,
                               "soft_sentinel": arm_gate[label],
                               "close_aug": arm_var[label][1],
                               "prompt_mode": arm_prompt[label]}
            arm_legs[label] = "all"
            return 0.0
        pt = resolve_lora(path)
        rep = load_finetune(model, pt, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        applied = True
        if rep["method"] != "lora":
            raise SystemExit(f"[{label}] {rep['method']} fine-tune: the probe resets "
                             "adapters between arms and cannot restore full weights")
        if rep["mismatches"]:
            raise SystemExit(f"[{label}] {rep['mismatches']} key mismatches — refusing to "
                             "probe a model that is part fine-tuned and part base")
        if rep["delta_norm"] == 0.0:
            raise SystemExit(f"[{label}] ||dW|| == 0: the weights are unchanged from base")
        meta = rep["meta"] or {}
        if args.recipe_from and replay_subset(recipe_from_lora_meta(meta)) != replay_subset(recipe):
            raise SystemExit(f"[{label}] this adapter's recipe differs from --recipe-from; "
                             "every adapter arm must share one replay")
        arm_info[label] = {"path": str(path), "lora_sha": checkpoint_hash(pt),
                           "delta_norm": rep["delta_norm"], "mismatches": rep["mismatches"],
                           "modules": rep["modules"], "r": rep["r"], "alpha": rep["alpha"],
                           "scopes": list((meta.get("lora") or {}).get("scopes") or []),
                           "objective": objective_of(recipe_from_lora_meta(meta)),
                           "lora_legs": recipe_from_lora_meta(meta).get("lora_legs") or "all",
                           "soft_sentinel": recipe_from_lora_meta(meta).get("soft_sentinel"),
                           "close_aug": base_variant_of(recipe_from_lora_meta(meta))[1],
                           "prompt_mode": recipe_from_lora_meta(meta)["prompt_mode"]}
        if (arm_info[label]["soft_sentinel"], arm_info[label]["close_aug"]) != arm_var[label]:
            raise SystemExit(f"[{label}] variant read at load "
                             f"({arm_info[label]['soft_sentinel']}, {arm_info[label]['close_aug']}) "
                             f"!= read up front ({arm_var[label]})")
        # the arm's own closing-leg setting (S17b fwdonly): its objective saw the closing
        # frame under the BASE weights, so its replay must too
        arm_legs[label] = arm_info[label]["lora_legs"]
        print(f"  {label}: {rep['modules']} LoRA modules, ||dW|| = {rep['delta_norm']:.6f}",
              flush=True)
        return rep["delta_norm"]

    # the probe
    maps_dir = Path(tempfile.mkdtemp(prefix="s17maps_"))
    rows: list[dict] = []
    parity: list[dict] = []
    gt_of = ds.load_labelmap          # never cached: butterfly masks are 4 MB each

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for label, path in arms:
        t_arm = time.time()
        set_weights(label, path)
        # the arm's own no-object gate (S17d): hard (None) or soft K, on the tracker
        tracker.soft_sentinel = arm_gate[label]
        if arm_gate[label] is not None:
            print(f"  [{label}] soft sentinel K = {arm_gate[label]:g}", flush=True)
        _cdeg = arm_var[label][1]
        if _cdeg is not None:
            print(f"  [{label}] closing end rotated, up to ±{_cdeg:g}°", flush=True)
        for n_done, i in enumerate(indices):
            x0, xu, y0, (ref, query) = pairs[i]
            log = pairs.step_log[-1]
            class_ids = tuple(log["class_ids"] or (1,))       # merged: one foreground channel
            x0d, xud, y0d = x0.to(args.device), xu.to(args.device), y0.to(args.device)
            close_kw, close_angle, y0_target = {}, "", y0
            if _cdeg is not None:
                x0c, y0c, crec = close_pairs[_cdeg].closing(i)
                close_kw = {"x0_close": x0c.to(args.device), "y0_close": y0c.to(args.device)}
                close_angle, y0_target = crec["close_angle"], y0c
            res = cycle_probe(tracker, x0d, xud, y0d, autocast_dtype=autocast_dtype,
                              hard_replay=True,
                              **{**oc_kw, "lora_legs": arm_legs[label], **close_kw})
            gt_q = gt_of(query)
            if recipe["mask_mode"] == "merged":
                gt_q = (gt_q > 0).astype(np.int32)
            fwd = score_leg(res.forward, class_ids, gt_q, "gt_native")
            y0_lab = y0_labelmap(y0_target, class_ids)
            cls = score_leg(res.closing, class_ids, y0_lab, "model")
            variants = loss_variants(res.closing, y0_target, fwd["found"],
                                     fwd_obj=res.parts.get("fwd_obj_score"))
            # the steganography column: the same closing legs on the BINARISED prompt
            cls_h = score_leg(res.closing_hard, class_ids, y0_lab, "model")
            variants_h = loss_variants(res.closing_hard, y0_target, fwd["found"],
                                       fwd_obj=res.parts.get("fwd_obj_score"))
            fwd_map = cls_map = cls_map_hard = ""
            if args.save_maps == "all":
                fwd_map = f"maps/{label}/fwd/{i:04d}.png"
                cls_map = f"maps/{label}/cls/{i:04d}.png"
                cls_map_hard = f"maps/{label}/cls_hard/{i:04d}.png"
                write_labelmap_png(maps_dir / fwd_map, fwd["labelmap"])
                write_labelmap_png(maps_dir / cls_map, cls["labelmap"])
                write_labelmap_png(maps_dir / cls_map_hard, cls_h["labelmap"])
            parts = res.parts
            rows.append({
                "dataset": ds.name, "group": grp.get(ref, "all"), "train_step": i,
                "ref": ref, "query": query,
                "class_ids": " ".join(str(c) for c in class_ids),
                "n_channels": len(class_ids), "arm": label,
                "aug_angle": log.get("aug_angle", ""),
                "fwd_miou": fwd["miou"], "fwd_miou_prompted": fwd["miou_prompted"],
                "fwd_iou_binary": fwd["iou_binary"], "fwd_n_gt_traits": fwd["n_gt_traits"],
                "fwd_n_prompted_present": fwd["n_prompted_present"],
                "fwd_per_trait": fmt_per_trait(fwd["per_trait"]),
                "fwd_found": ";".join("1" if f else "0" for f in fwd["found"]),
                "fwd_found_count": fwd["found_count"],
                "fwd_claimed_frac": fwd["claimed_frac"],
                "fwd_fp_frac": fwd["fp_frac"], "fwd_n_absent_prompted": fwd["n_absent_prompted"],
                "cls_miou": cls["miou"], "cls_iou_binary": cls["iou_binary"],
                "cls_per_trait": fmt_per_trait(cls["per_trait"]),
                "cls_claimed_frac": cls["claimed_frac"],
                "loss": res.loss, "bce": parts.get("bce"), "dice": parts.get("dice"),
                "ce": parts.get("ce", ""),
                "dice_per_channel": ";".join(f"{v:.4f}"
                                             for v in parts.get("dice_per_channel") or []),
                "bce_per_channel": ";".join(f"{v:.4f}"
                                            for v in parts.get("bce_per_channel") or []),
                "fwd_obj": ";".join(f"{v:.3f}" for v in parts.get("fwd_obj_score") or []),
                "cls_obj": ";".join(f"{v:.3f}" for v in parts.get("cls_obj_score") or []),
                "lora_legs": arm_legs[label],
                "soft_sentinel": ("" if arm_gate[label] is None else arm_gate[label]),
                **{f"loss_{o}": ("" if variants[o] is None else variants[o])
                   for o in OBJECTIVES if o != "plain"},
                "prompt_mode": arm_prompt[label],
                "close_aug": ("" if _cdeg is None else _cdeg), "close_angle": close_angle,
                "cls_miou_hard": cls_h["miou"], "cls_iou_binary_hard": cls_h["iou_binary"],
                "cls_per_trait_hard": fmt_per_trait(cls_h["per_trait"]),
                "cls_claimed_frac_hard": cls_h["claimed_frac"],
                "cls_obj_hard": ";".join(f"{v:.3f}" for v in res.cls_obj_hard or []),
                "loss_hard": variants_h["plain"], "bce_hard": variants_h["bce"],
                "dice_hard": variants_h["dice"],
                **{f"loss_{o}_hard": variants_h[o] for o in FORMULAS if o != "plain"},
                "fwd_map": fwd_map, "cls_map": cls_map, "cls_map_hard": cls_map_hard,
            })
            if i in parity_idx:
                # The inference path on the SAME 1-shot episode. `predict` runs under
                # inference_mode with SAM 2's video predictor (memory re-encoded from the
                # consolidated low-res masks, bf16 memory features) — the two paths are
                # known to differ; this records BY HOW MUCH on identical images.
                lm_ref = gt_of(ref)
                ep = Episode(query_id=query, support_ids=(ref,),
                             support_images=(ds.load_image(ref),),
                             support_labelmaps=(lm_ref,),
                             query_image=ds.load_image(query),
                             class_ids=union_class_ids([lm_ref]))
                ev = adapter.predict(ep)
                adapter.clear_embeddings()
                shared = [c for c in class_ids if c in ep.class_ids]
                d = []
                for c in shared:
                    a = res.forward[class_ids.index(c)]
                    b = ev[ep.class_ids.index(c)]
                    if a.shape != b.shape:
                        b = resize_logits(b[None], a.shape)[0]
                    d.append(np.abs(a - b))
                ev_s = score_leg(ev, ep.class_ids, gt_of(query), "gt_native")
                parity.append({
                    "arm": label, "train_step": i, "ref": ref, "query": query,
                    "class_ids_train": list(class_ids), "class_ids_eval": list(ep.class_ids),
                    "max_abs_diff": float(max(x.max() for x in d)) if d else None,
                    "mean_abs_diff": float(np.mean([x.mean() for x in d])) if d else None,
                    "labelmap_agreement": (float((fwd["labelmap"] == ev_s["labelmap"]).mean())
                                           if fwd["labelmap"].shape == ev_s["labelmap"].shape
                                           else None),
                    "fwd_miou_train_leg": fwd["miou"], "fwd_miou_eval_path": ev_s["miou"],
                })
            if n_done % 20 == 0 or n_done == len(indices) - 1:
                print(f"  [{label}] {n_done + 1}/{len(indices)} step {i} "
                      f"{ref} -> {query}: fwd {100 * (fwd['miou'] or 0):.1f} "
                      f"cls {100 * (cls['miou'] or 0):.1f} (hard "
                      f"{100 * (cls_h['miou'] or 0):.1f}) loss {res.loss:.4f}", flush=True)
        arm_rows = [r for r in rows if r["arm"] == label]
        print(f"[{label}] {len(arm_rows)} pairs in {time.time() - t_arm:.0f}s | "
              f"fwd {100 * (macro_over(arm_rows, 'fwd_miou') or 0):.2f} "
              f"(prompted {100 * (macro_over(arm_rows, 'fwd_miou_prompted') or 0):.2f}) | "
              f"cls {100 * (macro_over(arm_rows, 'cls_miou') or 0):.2f} "
              f"(hard {100 * (macro_over(arm_rows, 'cls_miou_hard') or 0):.2f}) | "
              f"loss {macro_over(arm_rows, 'loss'):.4f}", flush=True)

    # the replay proof, under EACH arm's own objective
    # At global step 0 LoRA's B is zero, so the base arm's closing logits are the ones
    # every training run scored first; each record's `loss_first` must equal the base
    # row's column for that record's objective. One rel err per record, the max gated.
    base0 = next(r for r in rows if r["arm"] == base_label and r["train_step"] == 0)
    # NOT `parity`: that name is the eval-path list above, written into the record as
    # `eval_parity` — shadowing it here emptied that list on the cluster (first S17 session).
    replay_parity: dict[str, dict] = {}
    for label, rec in train_recs.items():
        rec_recipe = recipe_from_lora_meta({**(rec.get("finetune") or {}),
                                            "occl": rec.get("occl") or {}})
        obj = objective_of(rec_recipe)
        # the record's own gate and closing end: its first loss was computed under them,
        # so the base row it is proven against must be the base variant under the same
        # (gate, closing end) — and under its own prompt VALUE (the `_hard` columns
        # for an ste/hard-trained record)
        rec_base = base_arm_label(base_label, *base_variant_of(rec_recipe))
        rec_hard = own_prompt_is_hard(rec_recipe)
        own0 = own_loss(next(r for r in rows if r["arm"] == rec_base and r["train_step"] == 0),
                        obj, hard=rec_hard)
        if own0 is None:
            replay_parity[label or "_"] = {
                "objective": obj, "loss_first_train": rec["training"].get("loss_first"),
                "loss_first_probe_base": None, "loss_first_rel_err": None,
                "note": "objective not scored post hoc"}
            continue
        pv = loss_parity(own0, rec["training"].get("loss_first"))
        replay_parity[label or "_"] = {"objective": obj, **pv, "record": rec["_path"],
                                       "base_arm": rec_base,
                                       "column": "hard" if rec_hard else "continuous",
                                       "lora_sha": (rec.get("finetune") or {}).get("lora_sha"),
                                       "delta_norm": rec["training"].get("delta_norm")}
        print(f"  loss parity at step 0 [{label or '-'} / {obj} / "
              f"{'hard' if rec_hard else 'continuous'} prompt vs {rec_base}]: probe "
              f"{pv['loss_first_probe_base']:.8f} vs trained {pv['loss_first_train']:.8f} "
              f"(rel err {pv['loss_first_rel_err']:.2e})", flush=True)
    replay["parity"] = replay_parity
    replay["prompt_modes"] = dict(REPLAY_PROMPT_MODES)
    errs = [v["loss_first_rel_err"] for v in replay_parity.values()
            if v.get("loss_first_rel_err") is not None]
    first = next(iter(replay_parity.values()), None)
    replay |= {"loss_first_probe_base": (first or {}).get("loss_first_probe_base", base0["loss"]),
               "loss_first_train": (first or {}).get("loss_first_train"),
               "loss_first_rel_err": (max(errs) if errs else None)}

    # summary + bundle
    summary = {}
    for label, _ in arms:
        ar = [r for r in rows if r["arm"] == label]
        found = [int(r["fwd_found_count"]) / max(int(r["n_channels"]), 1) for r in ar]
        summary[label] = {
            "n": len(ar),
            "objective": arm_info[label].get("objective", "plain"),
            **{f"loss_{o}": macro_over([{"v": r[f"loss_{o}"]} for r in ar
                                        if r[f"loss_{o}"] != ""], "v")
               for o in OBJECTIVES if o != "plain"},
            "fwd_miou": macro_over(ar, "fwd_miou"),
            "fwd_miou_prompted": macro_over(ar, "fwd_miou_prompted"),
            "fwd_iou_binary": macro_over(ar, "fwd_iou_binary"),
            "fwd_found_rate": float(np.mean(found)) if found else None,
            "cls_miou": macro_over(ar, "cls_miou"),
            "cls_miou_hard": macro_over(ar, "cls_miou_hard"),
            "stego_gap": macro_over([{"v": r["cls_miou"] - r["cls_miou_hard"]} for r in ar
                                     if None not in (r["cls_miou"], r["cls_miou_hard"])], "v"),
            "loss": macro_over(ar, "loss"), "bce": macro_over(ar, "bce"),
            "dice": macro_over(ar, "dice"), "loss_hard": macro_over(ar, "loss_hard"),
            "prompt_mode": arm_prompt[label], "close_aug": arm_var[label][1],
        }
    peak = torch.cuda.max_memory_reserved() / 1e9 if cuda else 0.0
    record = {
        "stage": args.stage, "dataset": ds.name, "family": args.family,
        "model": adapter.config_report(),
        "recipe": recipe, "objectives_scored": list(OBJECTIVES),
        "arms": arm_info, "arm_order": [a for a, _ in arms],
        "base_arm": base_label,
        "base_arms": [a for a, p in arms if p is None],       # base + one variant per (gate, closing end)
        "base_of": base_of,                                    # adapter -> its base arm
        "arm_prompt_mode": arm_prompt,                         # S18: each arm's own prompt value
        "replay": replay,
        "summary": summary,
        "eval_parity": parity,          # list: one entry per (arm, parity pair)
        "pairs": rows,
        "probe": {"complete": True, "elapsed_s": round(time.time() - t_start, 1),
                  "peak_vram_gb": round(peak, 3), "dtype": str(autocast_dtype),
                  "device": args.device, "save_maps": args.save_maps},
        "data": {"cache_format": ds.cache.policy(ds.name).fmt,
                 "source_preference": ds.cache.source_preference},
    }
    assert isinstance(record["eval_parity"], list) and isinstance(record["replay"]["parity"], dict)
    if args.eval_parity:
        assert len(record["eval_parity"]) >= 1, "eval parity was requested but none was recorded"
    csv_path = maps_dir / "pairs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    extra = {"pairs.csv": csv_path}
    for p in sorted(maps_dir.rglob("*.png")):
        extra[str(p.relative_to(maps_dir))] = p
    part = out_path.with_name(out_path.name + ".part")
    export_bundle(args.stage, record, out=part, dtype=str(autocast_dtype), extra_files=extra)
    os.replace(part, out_path)
    shutil.rmtree(maps_dir, ignore_errors=True)
    adapter.close()
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(rows)} rows, {len(extra) - 1} maps)")
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
