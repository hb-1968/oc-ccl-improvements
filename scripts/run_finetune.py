#!/usr/bin/env python3
"""Run OC-CCL fine-tuning and record its configuration and adapter weights.

The established SAM 2 recipe is sam2_crown; a stage brief must declare deviations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl import paths  # noqa: E402
from occl.adapters.sst_sam2 import SSTSam2Adapter  # noqa: E402
from occl.bundle import export_bundle  # noqa: E402
from occl.datasets.beetle import Beetle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.finetune.cycle import PROMPT_MODES  # noqa: E402
from occl.finetune.train import diagnostics, save_lora, train_lora  # noqa: E402
from occl.imagecache import ImageCache  # noqa: E402
from occl.registry import RECIPES, expand_recipe  # noqa: E402


def build_dataset(name: str, cache_root):
    # `paths.cache_root()` never existed — the old fallback was an AttributeError
    # waiting for the first caller to omit the flag (S13 CPU smoke found it; Colab
    # always passes --cache-root). Fail naming the flag instead.
    if not cache_root:
        raise SystemExit(
            "--cache-root is required: the pixel cache lives outside the repo "
            "(locally e.g. ~/Projects/oc-ccl-data/cache-png; Colab passes LOCAL_CACHE)")
    return {"beetle": Beetle,
            "butterfly_major": ButterflyMajor}[name](cache=ImageCache(cache_root))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beetle", choices=["beetle", "butterfly_major"])
    ap.add_argument("--group", default=None,
                    help="restrict the fine-tune to one group. Default is ALL "
                         "groups — Butterfly Major has no 'all' group and "
                         "asking for one raises FileNotFoundError.")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1,
                    help="Spec pins 1 (A4). >1 is a deviation and classifies as modified.")
    ap.add_argument("--train-mode", action="store_true",
                    help="A5: call `model.train()` during the cycle, as SST does "
                         "(oc_ccl.py:260,293), instead of holding `eval()`. Inert under "
                         "LoRA; under a FULL fine-tune it activates dropout and lets "
                         "normalisation layers update running stats that ship in the "
                         "checkpoint.")
    ap.add_argument("--recipe", default=None, choices=sorted(RECIPES),
                    help="a named, established configuration (registry.RECIPES): "
                         "`sam2_crown` = the SAM 2 fine-tune every future run uses "
                         "(S17d; fwdonly + objness w10 + excl zerocat + the arch scopes, "
                         "per-trait, lr 1e-5, one epoch). Its flags are spliced in front "
                         "of the command line, so a flag given explicitly still wins — "
                         "and the record's `occl` block, not the name, is what a gate "
                         "reads (registry.is_sam2_crown_finetune)")
    ap.add_argument("--scope-preset", default="faithful",
                    choices=["faithful", "sam3_arch", "sam2_arch"],
                    help="LoRA scope set: 'faithful' = the family's A2.2 reading; "
                         "'sam3_arch' (sam3 only) adds `transformer` — the memory "
                         "attention, the S14sam3 architecture-adapted arm; "
                         "'sam2_arch' (sam2 only) adds `memory_attention`, the same "
                         "addition on SAM 2 — the S17 arch arm")
    ap.add_argument("--family", default="sam2", choices=["sam2", "sam3", "sec"],
                    help="S14: model family. Selects the adapter build, the "
                         "differentiable tracker, the LoRA scope strings "
                         "(FAITHFUL_SCOPES_BY_FAMILY), and the geometry/normalization "
                         "defaults (sam3: 1008/1152, half-norm; sec: sam2's, rooted "
                         "at the grounding encoder with the LVLM dropped).")
    ap.add_argument("--normalization", default=None,
                    choices=["imagenet", "unit", "half"],
                    help="A1: `imagenet` builds the same tensor inference does (verified "
                         "identical). `unit` is SST's `/255` alone "
                         "(butterfly_dataset.py:138-139), which trains on [0,1] and scores "
                         "on ~[-2.1,+2.6].")
    ap.add_argument("--method", default="lora", choices=["full", "lora"],
                    help="`full` is SST's OWN method — freeze the image encoder, train the "
                         "remaining ~11.7M parameters (src/sst/oc_ccl.py:237-246) — and is "
                         "what registry.FAITHFUL_FINETUNE now requires, because LoRA appears "
                         "in SST only as an ablation axis defaulting to off. The default "
                         "stays `lora` so every S8 command still reproduces S8 exactly; "
                         "S8.5 passes --method full explicitly. Flipping a default would "
                         "make an old command silently mean a new experiment.")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--prompt-mode", default="logit_reparam", choices=list(PROMPT_MODES),
                    help="hard_threshold reproduces SST's defect A2, for the ablation; "
                         "ste_threshold (S18) is the release's binary prompt VALUE with "
                         "the reparam's gradient — the closing frame receives the argmax "
                         "mask only, closing the sub-threshold side channel")
    ap.add_argument("--no-reset", action="store_true",
                    help="A7: reuse the forward leg's memory instead of resetting it")
    ap.add_argument("--mask-mode", default="merged", choices=["merged", "per_trait"],
                    help="What the cycle supervises. `merged` is the RELEASE's silhouette "
                         "((mask>0), butterfly_dataset.py:121); `per_trait` is the PAPER's "
                         "y ∈ {0,1}^{W×H×C} (§3; §3.3 contrasts the single-channel y0*), "
                         "one SAM 2 object per trait. The default stays `merged` so every "
                         "S8/S8.5 command reproduces exactly; S10 passes per_trait "
                         "explicitly.")
    ap.add_argument("--reference-run", type=int, default=None,
                    help="The paper's protocol (A2.2: 'one fixed labeled image as x0'): "
                         "fix the reference to the support that evaluation run N draws "
                         "(via curation.draw_supports, so training anchors on exactly the "
                         "image run N scores with), queries iterate over the rest. "
                         "Default None keeps the all-references derangement.")
    ap.add_argument("--reference-id", default=None,
                    help="explicit fixed reference image id (alternative to "
                         "--reference-run)")
    ap.add_argument("--reference-selection", default="canonical_balanced",
                    help="support-selection arm used to resolve --reference-run")
    ap.add_argument("--rotation-aug-deg", type=float, default=None,
                    help="A2.2, Beetle only: 'a small degree of random 2D rotation "
                         "augmentation on the labeled example'. Uniform(-θ,+θ) per step, "
                         "reference image+labelmap jointly, query untouched. The paper "
                         "does not quantify θ; pass it under a recorded rule "
                         "(--rotation-aug-rule).")
    ap.add_argument("--rotation-aug-rule", default=None,
                    help="the a-priori rule the θ value came from, recorded verbatim "
                         "(e.g. 'half the beetle train pool circular std: 55.6/2')")
    ap.add_argument("--scnp-width", type=int, default=0,
                    help="S9a: transform the closing-mask loss with Same-Class Neighbor "
                         "Penalization at this window width (3 = the paper's and the "
                         "tested value; 0 = off). Merged mask only — the per-trait "
                         "composition is deliberately guarded off in cycle.py.")
    ap.add_argument("--scnp-no-joint", action="store_true",
                    help="apply SCNP to BCE and Dice separately instead of jointly "
                         "(joint is the default and the offline-tested form)")
    ap.add_argument("--excl-mode", choices=["off", "aggregate", "zerocat"],
                    default="off",
                    help="S13: replace the closing-frame BCE with a softmax CE over "
                         "[background, trait channels]. `zerocat` (constant-0 "
                         "background) is the eval decision rule verbatim and the arm "
                         "setting; `aggregate` is the STM/Cutie soft-aggregation form. "
                         "Composition with --scnp-width is refused.")
    ap.add_argument("--xovl-weight", type=float, default=0.0,
                    help="S13: weight of the label-free pairwise overlap penalty "
                         "mean(P_i * sum_j!=i P_j) (SAM-MT). 0 = off; the arm runs 1.0 "
                         "(unit-weight rule, a-priori, never tuned).")
    ap.add_argument("--xovl-where", choices=["query", "both"], default="query",
                    help="where the overlap penalty lands: the forward-leg query "
                         "prediction (the tensor no GT can reach — default) or "
                         "additionally the closing frame")
    ap.add_argument("--bshape-weight", type=float, default=0.0,
                    help="S13: weight of the Kervadec signed-distance gross-shape term "
                         "(1-px dead zone, per-channel max normalization). 0 = off; "
                         "the arm runs 1.0 (unit-weight rule).")
    ap.add_argument("--bshape-schedule", default="const",
                    help="weight schedule for bshape. Only 'const' is implemented — "
                         "Kervadec's per-epoch ramp is degenerate at the spec's "
                         "epochs=1; the string is recorded so a future ramp is a value "
                         "change, not a schema change.")
    ap.add_argument("--margin-weight", type=float, default=0.0,
                    help="S13: weight of the orientation-contrastive confidence margin "
                         "added to the cycle loss — hinge demanding the as-given leg's "
                         "claimed-pixel confidence beat the 180-rotated leg's by m. "
                         "0 = off; the ptmargin arm runs 1.0.")
    ap.add_argument("--margin-m", type=float, default=1.0,
                    help="the margin m, a-priori |SAM2_OUT_BIAS|/10 = 1.0; pass its "
                         "rule via --margin-rule")
    ap.add_argument("--margin-only", action="store_true",
                    help="S13: train the margin term ALONE — no reconstruction, no "
                         "reverse legs (cycle.margin_step). --margin-weight is ignored; "
                         "the mechanism-isolation arm.")
    ap.add_argument("--margin-rule", default=None,
                    help="the a-priori rule m came from, recorded verbatim (the "
                         "--rotation-aug-rule pattern)")
    ap.add_argument("--found-mask", action="store_true",
                    help="S17 arm: drop from the closing loss every channel the forward "
                         "leg did not find on the query (cycle.oc_ccl_step found_mask) — "
                         "the channels S10 proved carry zero forward-leg gradient and the "
                         "120-pair probe measured carrying 65%% of the epoch's BCE")
    ap.add_argument("--bce-weight", type=float, default=1.0,
                    help="S17 arm `nobce` = 0: weight of the BCE term (1.0 = the paper's "
                         "BCE + Dice); Dice is bounded per channel where BCE is linear in "
                         "|logit| on a confidently wrong region")
    ap.add_argument("--lora-legs", choices=["all", "forward", "forward_link"], default="all",
                    help="S17b arm `fwdonly`: run the reverse init and the closing frame "
                         "with the adapters switched off, so the closing head cannot learn "
                         "to read x0 (cycle.oc_ccl_step lora_legs); `forward_link` (S17c) "
                         "switches them off for the closing frame only")
    ap.add_argument("--objness", action="store_true",
                    help="S17b arm `objness`: add BCE(forward objectness score, present) "
                         "per prompted channel — the score the no-object sentinel keys on")
    ap.add_argument("--objness-weight", type=float, default=1.0,
                    help="the objectness term's weight (S17b: 1.0, the first a-priori "
                         "value; S17c sweeps a pre-declared grid {3, 10, 30} after 1.0 "
                         "flipped 6 of 19 fence channels)")
    ap.add_argument("--soft-sentinel", type=float, default=None, metavar="K",
                    help="S17d arm `soft`: replace SAM 2's hard −1024 no-object gate on the "
                         "propagated frames of the cycle by mask − K·(1 − σ(score)) and "
                         "the soft pointer mix (tracker.soft_sentinel_heads) — the "
                         "closing loss then reaches the fence channels' masks AND scores. "
                         "A-priori K = 20 (SAM 2's own ±10 prompt-logit span). Training "
                         "path only; the evaluator's hard gate is untouched")
    ap.add_argument("--close-aug-deg", type=float, default=None, metavar="DEG",
                    help="S18 arm `perturb`: close the cycle on the reference under its "
                         "own independently seeded rotation, uniform(-DEG, +DEG) per step "
                         "(CyclePairs.closing); the opening x0 is untouched. A-priori 10")
    ap.add_argument("--skip-grad-diagnostics", action="store_true",
                    help="skip the post-training gradient report (CPU smoke tests only — "
                         "the s10ft gate requires it)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap steps per epoch — for smoke tests, not for a real arm")
    ap.add_argument("--diagnostics-only", action="store_true")
    ap.add_argument("--lora-out", default=None, help="where to write the adapter weights")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="write adapters after EVERY epoch here — one training run then\n                          yields the whole epochs curve for s8c_tune")
    ap.add_argument("--out", default="s8.zip")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args(expand_recipe(sys.argv[1:]))

    import torch

    import occl as _occl
    # Stamped into every worker log. When a run fails in a way a fix was
    # supposed to prevent, the first question is which code actually ran —
    # and inferring it from which log lines are absent is guesswork.
    print(f"occl build {_occl.BUILD} | {__file__}", flush=True)
    ds = build_dataset(args.dataset, args.cache_root)

    if args.reference_run is not None and args.reference_id is not None:
        ap.error("--reference-run and --reference-id are mutually exclusive")
    if args.bshape_weight and args.bshape_schedule != "const":
        ap.error(f"--bshape-schedule {args.bshape_schedule!r} is not implemented — "
                 "only 'const' exists (the record carries the string, not the code)")
    if args.margin_only and args.margin_weight:
        ap.error("--margin-only trains the margin ALONE; --margin-weight is the "
                 "composite arm — pass one or the other")
    reference_id = args.reference_id
    if args.reference_run is not None:
        # THE dispatcher, never a fourth copy of the draw (see curation.draw_supports).
        # shot=1: the anchor is the 1-shot support of run N, so an anchored fine-tune
        # trains on exactly the image its evaluation run prompts with.
        from occl.curation import draw_supports, support_pool

        groups = list(ds.groups())
        g = args.group if args.group is not None else (
            groups[0] if len(groups) == 1 else None)
        if g is None:
            ap.error("--reference-run needs --group on a multi-group dataset")
        pool, _ = support_pool(ds, g, args.reference_selection)
        reference_id = draw_supports(ds, g, pool, args.reference_selection,
                                     1, args.reference_run)[0]
        print(f"reference: run {args.reference_run} of {args.reference_selection} "
              f"-> {reference_id}", flush=True)
    # S14: family resolution — geometry and normalization defaults follow the family
    # unless explicitly overridden, and the scope strings come from the shared table.
    from occl.finetune.lora import (FAITHFUL_SCOPES_BY_FAMILY, SAM2_ARCH_SCOPES,
                                    SAM3_ARCH_SCOPES)

    _fam = args.family
    if args.normalization is None:
        args.normalization = "half" if _fam == "sam3" else "imagenet"
    _image_size = 1008 if _fam == "sam3" else 1024
    _mask_size = 1152 if _fam == "sam3" else None
    if args.scope_preset == "sam3_arch":
        if _fam != "sam3":
            raise SystemExit("--scope-preset sam3_arch is a sam3 arm "
                             f"(got --family {_fam})")
        _scopes = SAM3_ARCH_SCOPES
    elif args.scope_preset == "sam2_arch":
        if _fam != "sam2":
            raise SystemExit("--scope-preset sam2_arch is a sam2 arm "
                             f"(got --family {_fam})")
        _scopes = SAM2_ARCH_SCOPES
    else:
        _scopes = FAITHFUL_SCOPES_BY_FAMILY[_fam]

    def _build_family_adapter():
        if _fam == "sam2":
            return SSTSam2Adapter(checkpoint=args.checkpoint)
        if _fam == "sam3":
            from occl.adapters.sam3 import Sam3TrackerAdapter

            return Sam3TrackerAdapter(checkpoint=args.checkpoint)
        from occl.adapters.sec import SeCAdapter

        # Training never injects language (the pinned S14 recipe), so the concept
        # policy is irrelevant here; the LVLM is dropped right after build.
        return SeCAdapter(checkpoint=args.checkpoint, concept_policy="off")

    adapter = _build_family_adapter()
    dtype = torch.float32 if args.device == "cpu" else None
    print(f"building {_fam} on {args.device} from {adapter.checkpoint}", flush=True)
    adapter.build(device=args.device, dtype=dtype)
    model = adapter.model
    if _fam == "sec":
        # The cycle trains the grounding encoder only; ~7.6 GB of frozen LVLM has no
        # business resident during training.
        adapter.drop_lvlm()
        # Recipe parity with the crown: sam2/sam3 train fp32 master weights under
        # train_lora's bf16 autocast (sst build never casts weights), but
        # from_pretrained leaves the grounding encoder bf16 — the released eval
        # precision, kept for eval. AdamW at lr 1e-5 over bf16 masters silently
        # loses updates below the mantissa (ulp at LoRA scale ~1e-2 is ~4e-5), a
        # GPU-only stall no CPU smoke can see (CPU builds are already fp32).
        model = model.float()

    record: dict = {
        "dataset": ds.name,
        "finetune": dict(
            {"method": args.method, "epochs": args.epochs, "lr": args.lr,
             "family": _fam, "image_size": _image_size,
             "mask_size": _mask_size or _image_size},
            **({"frozen": ["image_encoder"]} if args.method == "full" else
               {"lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                         "scopes": list(_scopes)}})),
        "occl": {
            "prompt_mode": args.prompt_mode,
            "train_mode": args.train_mode,
            "normalization": args.normalization,
            "memory_reset": not args.no_reset,
            "cycle_len": 2,
            # `merged` was recorded as "binary" before the mask axis existed; the two
            # words name the same release-shaped silhouette and both deviate from the
            # paper's `per_trait`.
            "mask_mode": args.mask_mode,
            "reference": "fixed" if reference_id is not None else "derangement",
            "reference_id": reference_id,
            "reference_run": args.reference_run,
            "augmentation": (None if args.rotation_aug_deg is None else
                             {"type": "rotation_x0",
                              "max_deg": args.rotation_aug_deg,
                              "rule": args.rotation_aug_rule}),
            "scnp": (None if not args.scnp_width else
                     {"width": args.scnp_width, "joint": not args.scnp_no_joint}),
            # S13 knobs — null when off, so every pre-S13 record classifies unchanged
            # and any non-null value names its own deviation (the scnp pattern).
            "excl": (None if args.excl_mode == "off" else {"mode": args.excl_mode}),
            "xovl": (None if not args.xovl_weight else
                     {"weight": args.xovl_weight, "where": args.xovl_where}),
            "bshape": (None if not args.bshape_weight else
                       {"weight": args.bshape_weight,
                        "schedule": args.bshape_schedule,
                        "dead_zone": 1, "normalize": "chan_max"}),
            "margin": (None if not (args.margin_weight or args.margin_only) else
                       {"m": args.margin_m,
                        "weight": (None if args.margin_only else args.margin_weight),
                        "only": args.margin_only,
                        "statistic": "hard_claim_mean_top_logit", "form": "hinge",
                        "consistent": "as_given", "rule": args.margin_rule}),
            # S17 knobs — the same null-when-off contract as the S13 knobs.
            "found_mask": ({"rule": "any_logit_gt_0"} if args.found_mask else None),
            "bce_w": (None if args.bce_weight == 1.0 else args.bce_weight),
            # S17b knobs, same contract
            "objness": ({"weight": args.objness_weight, "target": "present_for_prompted"}
                        if args.objness else None),
            "lora_legs": (None if args.lora_legs == "all" else args.lora_legs),
            # S17d knob, same contract
            "soft_sentinel": args.soft_sentinel,
            # S18 knob, same contract (prompt_mode above is the other S18 axis)
            "close_aug": (None if args.close_aug_deg is None else
                          {"type": "rotation_x0_close", "max_deg": args.close_aug_deg}),
            # the named recipe the command line invoked, if any (a label; the fields
            # above are the proof — registry.is_sam2_crown_finetune re-derives it)
            "recipe": args.recipe,
        },
        "model": adapter.config_report(),
        "data": {
            "source_preference": ds.cache.source_preference,
            "interp": ds.cache.interp,
            "cache_format": ds.cache.policy(ds.name).fmt,
            "normalization": "imagenet",
        },
    }

    _margin_m = (args.margin_m if (args.margin_weight or args.margin_only) else None)
    if args.diagnostics_only:
        print("measuring the forward-leg gradient under both prompt modes...", flush=True)
        diag = diagnostics(model, ds, group=args.group, device=args.device, dtype=dtype,
                           mask_mode=args.mask_mode, margin_m=_margin_m,
                           family=_fam, image_size=_image_size,
                           mask_size=_mask_size, normalization=args.normalization,
                           soft_sentinel=args.soft_sentinel)
        record["diagnostics"] = diag
        record["stage"] = "s8a"
        print(json.dumps(diag, indent=2))
        out = export_bundle("s8a", record, out=args.out, dtype=str(adapter.dtype))
        print(f"wrote {out}")
        return 0

    if args.epochs != 1:
        print(f"  NOTE: --epochs {args.epochs} deviates from the spec's 1 — see the "
              "comment in registry.FAITHFUL_FINETUNE; the record carries the true value")

    report = train_lora(
        model, ds, lr=args.lr, epochs=args.epochs, group=args.group,
        method=args.method, train_mode=args.train_mode,
        normalization=args.normalization, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, prompt_mode=args.prompt_mode,
        memory_reset=not args.no_reset, device=args.device, dtype=dtype,
        mask_mode=args.mask_mode, reference_id=reference_id,
        rotation_aug_deg=args.rotation_aug_deg,
        scnp_width=args.scnp_width, scnp_joint=not args.scnp_no_joint,
        excl_mode=args.excl_mode, xovl_weight=args.xovl_weight,
        xovl_where=args.xovl_where, bshape_weight=args.bshape_weight,
        margin_weight=args.margin_weight, margin_m=args.margin_m,
        margin_only=args.margin_only,
        found_mask=args.found_mask, bce_w=args.bce_weight,
        lora_legs=args.lora_legs, obj_weight=(args.objness_weight if args.objness else 0.0),
        soft_sentinel=args.soft_sentinel, close_aug_deg=args.close_aug_deg,
        family=_fam, image_size=_image_size, mask_size=_mask_size,
        lora_scopes=_scopes,
        max_steps=args.max_steps, checkpoint_dir=args.checkpoint_dir,
    )
    # The record must not be able to claim a hyperparameter the run did not use. This
    # exact mismatch shipped once: the CLI took --epochs 10, wrote `epochs: 10`, and
    # trained a single epoch.
    assert report["epochs"] == args.epochs, (
        f"trained {report['epochs']} epochs but --epochs said {args.epochs}")
    record["training"] = report
    record["diagnostics"] = {
        "delta_norm": report["delta_norm"],
        "delta_norm_before": report["delta_norm_before"],
    }
    if not args.skip_grad_diagnostics:
        # The s10ft gate's mechanism evidence, per arm bundle: gradient reaches the
        # forward leg under `logit_reparam` (per channel, when per-trait) and is exactly
        # 0.0 under `hard_threshold`. Measured on the just-trained weights — the
        # structural claims are weight-independent, and this is the model the arm ships.
        print("measuring the forward-leg gradient under both prompt modes...", flush=True)
        record["diagnostics"] |= diagnostics(
            model, ds, group=args.group, device=args.device, dtype=dtype,
            mask_mode=args.mask_mode, margin_m=_margin_m,
            family=_fam, image_size=_image_size,
            mask_size=_mask_size, normalization=args.normalization,
            soft_sentinel=args.soft_sentinel)

    lora_out = args.lora_out or str(Path(args.out).with_suffix("")) + "_lora.pt"
    # `occl` too: the prompt mode is the difference between "the forward leg trains" and
    # "it does not" (defect A2), and a checkpoint that cannot say which one produced it
    # can be silently reused for the other. That is exactly the reuse trap S8.5 hit.
    digest = save_lora(model, lora_out,
                       meta={**record["finetune"], "occl": record["occl"]},
                       method=args.method)
    record["finetune"]["lora_path"] = str(lora_out)
    record["finetune"]["lora_sha"] = digest

    # Round-trip immediately, into a FRESH adapter set, so the number in the record is a
    # real load and not an assertion about one. A checkpoint that cannot be reloaded is
    # worthless however well it trained.
    from occl.finetune.lora import apply_lora, load_lora_state_dict, lora_delta_norm
    from occl.finetune.train import load_trainable_state_dict

    probe = _build_family_adapter()
    probe.build(device="cpu", dtype=torch.float32)
    blob = torch.load(lora_out, map_location="cpu", weights_only=False)
    if args.method == "full":
        # No adapters to attach; the tensors overwrite the base weights in place. `||dW||`
        # is measured against the FRESHLY BUILT probe, which is the base checkpoint — so
        # this is the same quantity `full_delta_norm` reported during training, recomputed
        # independently rather than copied.
        theta0 = {k: v.clone() for k, v in probe.model.state_dict().items()}
        mismatches = load_trainable_state_dict(probe.model, blob["full"])
        reloaded = float(sum(((p.detach() - theta0[n]) ** 2).sum()
                             for n, p in probe.model.named_parameters()
                             if n in blob["full"]) ** 0.5)
    else:
        apply_lora(probe.model, r=args.lora_r, alpha=args.lora_alpha, scopes=_scopes)
        mismatches = load_lora_state_dict(probe.model, blob["lora"])
        reloaded = lora_delta_norm(probe.model)
    record["diagnostics"]["roundtrip_mismatches"] = mismatches
    record["diagnostics"]["delta_norm_reloaded"] = reloaded
    probe.close()
    print(f"round-trip: {mismatches} mismatches | "
          f"||dW|| {report['delta_norm']:.6f} -> {reloaded:.6f}")

    record["stage"] = "s8b"
    # The adapters travel INSIDE the bundle. They are the entire point of this stage — S8c
    # cannot run without them — and writing them beside the zip left them on the Colab VM,
    # which is destroyed with the session. At ~4.3 MB fp32 for 1.1M parameters they are
    # smaller than most per_query.csv payloads already shipped this way.
    out = export_bundle("s8b", record, out=args.out, dtype=str(adapter.dtype),
                        extra_files={"lora.pt": lora_out})   # name kept: every loader
                                                             # already looks for lora.pt
    print(f"wrote {out} (includes lora.pt, {Path(lora_out).stat().st_size / 1e6:.2f} MB)")
    if not args.no_download:
        from occl.bundle import download

        download(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
