"""S17 — the cycle on its own training pairs, both legs, before and after fine-tuning.

Every OC-CCL number the project reports is a test-split number. The fine-tune itself is
observed through its loss curve and `gradient_report` on ONE pair. Nobody had looked at what
the cycle says about the images it trains on: the query side ``x`` of each ``(x0 -> x ->
x0)`` pair is "unlabeled" to the loss, but every one has a GT mask on disk — `CyclePairs`
simply never calls ``ds.load_labelmap(query_id)``.

This module replays the exact training pairs through `cycle.oc_ccl_step` under
``no_grad`` and scores what the step computes and then discards:

* the **forward leg** — the model's prediction on ``x`` prompted by ``x0``'s GT, scored
  against ``x``'s held-out GT in GT-native space (the evaluation metric);
* the **closing leg** — the prediction on ``x0`` prompted by the forward leg's own output,
  scored against ``y0`` (the literal target the loss supervised) in model space, plus the
  loss value itself.

Deliberately OUTSIDE `evaluator._FINGERPRINTED`: nothing here touches a banked per-run
record. Torch is imported lazily so the combine and the sheet renderer stay torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..metrics import (BACKGROUND, logits_to_labelmap, resize_logits,
                       score_episode)

#: The S11 carrier — what `run_finetune.py` records when no axis is overridden. The probe's
#: commissioning path (no adapter yet, nothing to read a recipe from) falls back to these;
#: any adapter arm reads its recipe from the checkpoint's own meta instead.
CARRIER_RECIPE = {
    "mask_mode": "per_trait", "normalization": "imagenet",
    "prompt_mode": "logit_reparam", "memory_reset": True,
    "excl": None, "scnp": None, "xovl": None, "bshape": None, "margin": None,
    "found_mask": None, "bce_w": None, "objness": None, "lora_legs": None,
    "soft_sentinel": None, "close_aug": None,
    "augmentation": None, "reference": "derangement", "reference_id": None,
    "family": "sam2", "image_size": 1024, "mask_size": 1024,
}

#: What decides the PAIRS and the LEGS — two adapters sharing these can be probed in one
#: run: the legs are knob-free (pinned by test), only the loss differs.
LEG_KEYS = ("mask_mode", "normalization", "memory_reset", "augmentation",
            "reference", "reference_id", "family", "image_size", "mask_size")
#: What decides the OBJECTIVE on the closing mask. The probe scores every pair under
#: every S17 objective post hoc (`loss_variants`), so an arm's disqualifier reads its
#: own objective and the carrier's plain loss is the common yardstick.
LOSS_KEYS = ("excl", "scnp", "xovl", "bshape", "margin", "found_mask", "bce_w", "objness")
#: `lora_legs` (S17b `fwdonly`) is neither: it changes which weights the CLOSING legs run
#: under, so the probe replays each adapter's legs under its own setting — the base arm and
#: every adapter still share the pairs and the forward leg's weights-on setting.
#: `soft_sentinel` (S17d) is per-arm too, and unlike `lora_legs` it changes the BASE
#: model's legs (the gate is on the tracker, not the weights): the probe runs one base
#: variant per distinct value (`base_arm_label`) and pairs each adapter against the base
#: under ITS gate — the replay proof at step 0 needs the same gate on both sides.
#: `prompt_mode` and `close_aug` (S18) are per-arm too. The probe's shared replay runs
#: every arm's closing legs under the CONTINUOUS prompt (`logit_reparam`) AND, in the same
#: call, under the binary one (`hard_threshold`, the `*_hard` columns) — so an arm trained
#: with `ste_threshold` / `ste_memory` (whose value IS the binary prompt; they differ in
#: gradient only) reads its own closing off the `_hard` columns (`own_prompt_is_hard`)
#: and every arm's steganography gap is
#: `cls_miou − cls_miou_hard`. `close_aug` changes the closing frame itself, so it is a
#: base-variant axis like the soft gate (`base_arm_label`).
PER_ARM_KEYS = ("lora_legs", "soft_sentinel", "prompt_mode", "close_aug")
#: The two prompt values every arm's closing legs are replayed under.
REPLAY_PROMPT_MODES = {"continuous": "logit_reparam", "hard": "hard_threshold"}
#: The closing-loss FORMULAS `loss_variants` scores post hoc, plus the raw objectness term
#: (`objterm`, unweighted). An adapter's objective is a formula, optionally
#: ``+obj<w>``: its own loss on a pair is ``formula + w × objterm`` (`own_loss`), so the
#: objectness weight (S17c sweeps it) never needs a column of its own.
FORMULAS = ("plain", "found_mask", "nobce", "found_mask_nobce", "excl")
OBJECTIVES = FORMULAS + ("objterm",)


@dataclass
class ProbeResult:
    """One pair through the cycle. Logits are fp32 numpy ``(C, H, W)``.

    ``closing_hard`` (S18) is the closing leg re-run with the forward logits binarised
    before the reverse init — the steganography read; ``None`` unless asked for."""

    forward: np.ndarray
    closing: np.ndarray
    loss: float
    parts: dict = field(default_factory=dict)
    closing_hard: np.ndarray | None = None
    cls_obj_hard: list | None = None


def cycle_probe(tracker, x0, xu, y0, *, autocast_dtype=None, hard_replay: bool = False,
                **oc_ccl_kwargs) -> ProbeResult:
    """`oc_ccl_step` under ``no_grad``, with the forward leg captured.

    ``oc_ccl_kwargs`` are the step's own knobs (`oc_ccl_kwargs_from_recipe`); the capture
    dict is the one thing added, so the closing mask and loss are bit-identical to a
    training step at the same weights (pinned by test on the stub tracker).

    ``autocast_dtype`` mirrors `train_lora`: bf16 on CUDA, disabled elsewhere — the probe
    must see the tensors the loss saw, not an fp32 re-reading of them.

    ``hard_replay`` (S18) asks the step for the closing leg under the BINARISED forward
    logits as well (`oc_ccl_step`'s ``capture["hard_replay"]``): same backbones, same
    weights, the argmax mask as the only thing crossing the centre of the palindrome.
    """
    import torch

    from .cycle import oc_ccl_step

    cuda = x0.is_cuda
    dtype = autocast_dtype or (torch.bfloat16 if cuda else torch.float32)
    capture: dict = {"hard_replay": True} if hard_replay else {}
    with torch.no_grad(), torch.autocast(device_type="cuda" if cuda else "cpu",
                                         dtype=dtype, enabled=cuda):
        loss, closing, parts = oc_ccl_step(tracker, x0, xu, y0, capture=capture,
                                           **oc_ccl_kwargs)
    fwd = capture["forward"]
    hard = capture.get("closing_hard")
    hobj = capture.get("cls_obj_hard")
    return ProbeResult(
        forward=fwd.detach().float().cpu().numpy()[:, 0],
        closing=closing.detach().float().cpu().numpy()[:, 0],
        loss=float(loss),
        parts={k: v for k, v in parts.items()},
        closing_hard=(None if hard is None else hard.detach().float().cpu().numpy()[:, 0]),
        cls_obj_hard=(None if hobj is None
                      else [round(float(v), 4) for v in hobj.detach().flatten()]),
    )


def y0_labelmap(y0, class_ids) -> np.ndarray:
    """The closing target as an integer label map in model space.

    ``y0`` is `pairs.trait_mask_stack`'s ``(C, 1, S, S)`` stack — channels are disjoint by
    construction (one NEAREST-resized labelmap split by trait), so painting them back in
    order is lossless. Built from the tensor rather than re-loading the reference labelmap
    so the target is exact under the rotation augmentation for free.
    """
    y = y0.detach().float().cpu().numpy()[:, 0] if hasattr(y0, "detach") else np.asarray(y0)[:, 0]
    if y.shape[0] != len(class_ids):
        raise ValueError(f"y0 has {y.shape[0]} channels but class_ids has {len(class_ids)}")
    lab = np.full(y.shape[-2:], BACKGROUND, dtype=np.int32)
    for c, t in enumerate(class_ids):
        lab[y[c] > 0.5] = int(t)
    return lab


def score_leg(logits: np.ndarray, class_ids, gt_lab: np.ndarray,
              iou_space: str = "gt_native") -> dict:
    """Score one leg's ``(C, H, W)`` logits against a label map.

    `metrics.score_episode` is the evaluation rule (argmax, macro over GT-present traits, a
    GT trait absent from the support scores 0 — the one-shot draw's failure, counted). On a
    training pair that rule mixes two things the probe wants apart, so two more readings
    ride along:

    * ``miou_prompted`` — macro over the GT traits that ARE in ``class_ids`` (the traits
      ``x0`` carried). A trait the reference never had is not a model failure on the
      forward leg; it is the pair protocol.
    * ``found`` — per channel, ``(logits[c] > 0).any()``: the `per_channel_query_found`
      statistic of `cycle.gradient_report`. README S10: a channel the model claims nowhere
      on the query gets EXACTLY zero forward-leg gradient, so the fine-tune cannot reach it
      through that pair. Recorded per pair here so the bound can be read at scale.

    ``labelmap`` is the argmax map in the scoring space, for the bundle's PNGs.
    """
    scored = score_episode(logits, class_ids, gt_lab, iou_space=iou_space)
    lg = resize_logits(logits, gt_lab.shape) if iou_space == "gt_native" else logits
    gt = gt_lab if iou_space == "gt_native" else _resize_lab(gt_lab, logits.shape[-2:])
    pred = logits_to_labelmap(lg, class_ids)
    prompted = {t: v for t, v in scored["per_trait"].items() if t in set(int(c) for c in class_ids)}
    found = [bool((logits[c] > 0.0).any()) for c in range(logits.shape[0])]
    # S17c: the objectness term's cost — pixels claimed for a prompted trait the query's
    # GT does not contain at all (a false positive the macro mean never scores; under
    # argmax it can only steal pixels from the traits that ARE there)
    gt_present = {int(v) for v in np.unique(gt) if int(v) != BACKGROUND}
    absent = [int(c) for c in class_ids if int(c) not in gt_present]
    fp_frac = float(np.isin(pred, absent).mean()) if absent else 0.0
    return {
        "miou": scored["miou"],
        "miou_prompted": (float(np.mean(list(prompted.values()))) if prompted else None),
        "per_trait": scored["per_trait"],
        "iou_binary": scored["iou_binary"],
        "n_gt_traits": scored["n_traits"],
        "n_prompted_present": len(prompted),
        "found": found,
        "found_count": int(sum(found)),
        "claimed_frac": float((pred != BACKGROUND).mean()),
        "n_absent_prompted": len(absent),
        "fp_frac": fp_frac,
        "iou_space": iou_space,
        "labelmap": pred,
        "gt": gt,
    }


def _resize_lab(lab, size):
    from ..metrics import resize_labelmap

    return resize_labelmap(lab, tuple(size))


# ── the recipe: one place mapping a checkpoint's meta onto the replay ────────


def recipe_from_lora_meta(meta: dict) -> dict:
    """What `run_finetune.py` wrote into the adapter (`save_lora` meta = the record's
    ``finetune`` block + ``occl``) -> the knobs `CyclePairs` and `oc_ccl_step` need.

    Read from the checkpoint's OWN metadata, never from flags, so a replay cannot drift
    from what trained (the S8.5 reuse trap, in the other direction).
    """
    occ = meta.get("occl") or {}
    if not occ:
        raise ValueError("adapter meta has no 'occl' block — not a run_finetune.py "
                         "checkpoint (per-epoch ep<N>.pt files carry none either)")
    fam = meta.get("family", "sam2")
    size = int(meta.get("image_size") or (1008 if fam == "sam3" else 1024))
    msize = int(meta.get("mask_size") or size)
    return {
        "mask_mode": occ.get("mask_mode", "merged"),
        "normalization": occ.get("normalization", "imagenet"),
        "prompt_mode": occ.get("prompt_mode", "logit_reparam"),
        "memory_reset": bool(occ.get("memory_reset", True)),
        "excl": occ.get("excl"), "scnp": occ.get("scnp"), "xovl": occ.get("xovl"),
        "bshape": occ.get("bshape"), "margin": occ.get("margin"),
        "found_mask": occ.get("found_mask"), "bce_w": occ.get("bce_w"),
        "objness": occ.get("objness"), "lora_legs": occ.get("lora_legs"),
        "soft_sentinel": (None if occ.get("soft_sentinel") is None
                          else float(occ["soft_sentinel"])),
        "close_aug": occ.get("close_aug"),
        "augmentation": occ.get("augmentation"),
        "reference": occ.get("reference", "derangement"),
        "reference_id": occ.get("reference_id"),
        "family": fam, "image_size": size, "mask_size": msize,
        "epochs": meta.get("epochs"), "lr": meta.get("lr"),
        "lora": meta.get("lora"), "method": meta.get("method", "lora"),
    }


def replay_subset(recipe: dict) -> dict:
    """The keys that decide the replay (pairs + legs) — what adapter arms must share to
    be probed in one run. Loss knobs, `lr`, `epochs`, `lora` may differ."""
    return {k: recipe.get(k) for k in LEG_KEYS}


def objective_of(recipe: dict) -> str:
    """Which of `OBJECTIVES` an adapter trained on, from its recipe's loss knobs.

    ``other:<knobs>`` for anything `loss_variants` does not compute (the S13 xovl /
    bshape / margin folds, SCNP) — recorded, and the combine refuses to read its
    disqualifier off a column that does not exist."""
    fm = recipe.get("found_mask") is not None
    nobce = recipe.get("bce_w") == 0
    excl = (recipe.get("excl") or {}).get("mode")
    obj = recipe.get("objness")
    others = [k for k in ("scnp", "xovl", "bshape", "margin") if recipe.get(k) is not None]
    if others or (excl and excl != "zerocat") or (recipe.get("bce_w") not in (None, 0)) \
            or (obj is not None and not (float(obj.get("weight", 0)) > 0)):
        return "other:" + ",".join(
            f"{k}={recipe.get(k)}" for k in LOSS_KEYS if recipe.get(k) is not None)
    if excl and (fm or nobce):
        return "other:excl+"
    if excl:
        formula = "excl"
    elif fm and nobce:
        formula = "found_mask_nobce"
    elif fm:
        formula = "found_mask"
    elif nobce:
        formula = "nobce"
    else:
        formula = "plain"
    return formula + (f"+obj{float(obj['weight']):g}" if obj is not None else "")


def split_objective(objective: str) -> tuple[str, float]:
    """``"excl+obj10"`` -> ``("excl", 10.0)``; a bare formula -> weight 0."""
    formula, _, w = objective.partition("+obj")
    return formula, (float(w) if w else 0.0)


def own_prompt_is_hard(recipe: dict) -> bool:
    """Whether an arm's OWN closing leg saw the binary prompt (S18 `ste_threshold`, or
    the release's `hard_threshold`) — its own readings then live in the `_hard` columns."""
    return recipe.get("prompt_mode") in ("ste_threshold", "ste_memory", "hard_threshold")


def own_loss(row: dict, objective: str, hard: bool = False) -> float | None:
    """An arm's own objective on one probe row: ``formula column + w × objterm``.

    ``row`` is a pairs.csv row (strings) or a record row (numbers); ``None`` when the
    formula is not one the probe scored or the objectness term is missing. ``hard``
    reads the closing formula off the `_hard` columns (S18: the arm's closing leg saw
    the binary prompt); the objectness term is on the forward frame and has no
    hard variant."""
    formula, w = split_objective(objective)
    if formula not in FORMULAS:
        return None
    col = "loss" if formula == "plain" else f"loss_{formula}"
    if hard:
        col += "_hard"
    v = row.get(col)
    if v in (None, ""):
        return None
    v = float(v)
    if w:
        ot = row.get("loss_objterm")
        if ot in (None, ""):
            return None
        v += w * float(ot)
    return v


def strip_objective(recipe: dict) -> dict:
    """The same replay under the PLAIN objective — what the probe runs the step with."""
    return {**recipe, **{k: None for k in LOSS_KEYS}}


def cyclepairs_kwargs(recipe: dict) -> dict:
    """The `CyclePairs(...)` keyword arguments a recipe implies."""
    aug = recipe.get("augmentation")
    return {
        "size": recipe["image_size"],
        "mask_size": recipe["mask_size"],
        "normalization": recipe["normalization"],
        "mask_mode": recipe["mask_mode"],
        "reference_id": recipe.get("reference_id"),
        "rotation_aug_deg": (float(aug["max_deg"]) if aug else None),
        "close_aug_deg": close_aug_deg(recipe),
    }


def close_aug_deg(recipe: dict) -> float | None:
    """The S18 closing-end rotation an adapter trained under, else None."""
    ca = recipe.get("close_aug")
    return float(ca["max_deg"]) if ca else None


def oc_ccl_kwargs_from_recipe(recipe: dict) -> dict:
    """The `oc_ccl_step(...)` keyword arguments a recipe implies.

    The margin-only arm is refused: it never ran a closing leg (`cycle.margin_step`), so
    there is nothing to replay.
    """
    m = recipe.get("margin")
    if m and m.get("only"):
        raise ValueError("margin-only arms have no closing leg to probe")
    kw = {"prompt_mode": recipe["prompt_mode"],
          "memory_reset": recipe["memory_reset"]}
    if recipe.get("scnp"):
        kw |= {"scnp_width": int(recipe["scnp"]["width"]),
               "scnp_joint": bool(recipe["scnp"].get("joint", True))}
    if recipe.get("excl"):
        kw["excl_mode"] = recipe["excl"]["mode"]
    if recipe.get("xovl"):
        kw |= {"xovl_weight": float(recipe["xovl"]["weight"]),
               "xovl_where": recipe["xovl"].get("where", "query")}
    if recipe.get("bshape"):
        kw["bshape_weight"] = float(recipe["bshape"]["weight"])
    if m:
        kw |= {"margin_weight": float(m["weight"]), "margin_m": float(m["m"])}
    if recipe.get("found_mask") is not None:
        kw["found_mask"] = True
    if recipe.get("bce_w") is not None:
        kw["bce_w"] = float(recipe["bce_w"])
    if recipe.get("objness") is not None:
        kw["obj_weight"] = float(recipe["objness"]["weight"])
    if recipe.get("lora_legs"):
        kw["lora_legs"] = recipe["lora_legs"]
    return kw


def tracker_kwargs_from_recipe(recipe: dict) -> dict:
    """The tracker-level knobs a recipe implies (`_make_tracker(..., **kw)`): the soft
    sentinel lives on the tracker, not in `oc_ccl_step`."""
    return {"soft_sentinel": recipe.get("soft_sentinel")}


def base_arm_label(base: str, soft_sentinel: float | None,
                   close_aug: float | None = None) -> str:
    """The base-arm label an adapter pairs against: ``base`` under the hard gate on the
    plain closing frame, ``base_soft<K>`` under the soft gate (S17d), ``base_close<D>``
    when the closing end is rotated by up to D degrees (S18), both suffixes when both —
    one probed base per distinct (gate, closing-end) combination."""
    out = base
    if soft_sentinel is not None:
        out += f"_soft{float(soft_sentinel):g}"
    if close_aug is not None:
        out += f"_close{float(close_aug):g}"
    return out


def base_variant_of(recipe: dict) -> tuple[float | None, float | None]:
    """``(soft_sentinel, close_aug_deg)`` — the base-variant axes of an arm's recipe."""
    return recipe.get("soft_sentinel"), close_aug_deg(recipe)


def objness_term(fwd_obj: list[float] | None) -> float | None:
    """``BCE(forward objectness, 1)`` over the prompted channels — the `objness` term."""
    if not fwd_obj:
        return None
    import torch
    import torch.nn.functional as F

    z = torch.tensor([float(v) for v in fwd_obj])
    return float(F.binary_cross_entropy_with_logits(z, torch.ones_like(z)))


def loss_variants(closing: np.ndarray, y0, found: list[bool],
                  fwd_obj: list[float] | None = None) -> dict[str, float | None]:
    """Every S17 objective on one closing prediction, post hoc, fp32 on CPU.

    ``closing`` is the probe's ``(C, H, W)`` logits, ``y0`` the ``(C, 1, S, S)`` target
    stack, ``found`` the forward leg's per-channel claim. The formulas are the ones
    `cycle.oc_ccl_step` trains with (`bce_loss`, `dice_loss`, its found-masked means,
    `excl.excl_ce` zerocat) — pinned bit-identical on the stub tracker — so an arm's
    training `loss_first` must equal its column at step 0 (the per-arm replay proof).
    """
    import torch
    import torch.nn.functional as F

    from .cycle import bce_loss, dice_loss
    from .excl import excl_ce

    pred = torch.from_numpy(np.asarray(closing, dtype=np.float32))[:, None]
    tgt = y0.detach().float().cpu() if hasattr(y0, "detach") else torch.from_numpy(np.asarray(y0, np.float32))
    if tgt.shape[-2:] != pred.shape[-2:]:
        tgt = F.interpolate(tgt, size=pred.shape[-2:], mode="nearest")
    with torch.no_grad():
        bce = float(bce_loss(pred, tgt))
        dice = float(dice_loss(pred, tgt))
        C = pred.shape[0]
        w = torch.tensor([1.0 if f else 0.0 for f in found])
        n_found = float(w.sum())
        bce_c = torch.stack([F.binary_cross_entropy_with_logits(pred[c], tgt[c]) for c in range(C)])
        pr, tg = torch.sigmoid(pred).flatten(1), tgt.flatten(1)
        dice_c = 1.0 - (2.0 * (pr * tg).sum(dim=1) + 1.0) / (pr.sum(dim=1) + tg.sum(dim=1) + 1.0)
        bce_f = float((w * bce_c).sum() / n_found) if n_found else 0.0
        dice_f = float((w * dice_c).sum() / n_found) if n_found else 0.0
        ce = float(excl_ce(pred, tgt, "zerocat"))
    return {"plain": bce + dice, "found_mask": bce_f + dice_f, "nobce": dice,
            "found_mask_nobce": dice_f, "excl": ce + dice,
            "objterm": objness_term(fwd_obj), "bce": bce, "dice": dice}


# ── the replay proof ──────────────────────────────────────────────────────────


def mask_channel_histogram(ds, pairs, n: int, mask_size: int) -> dict[str, int]:
    """``{str(C): count}`` over the first ``n`` pairs — what `train_lora` records as
    ``training.mask_channels``, recomputed the same way (`trait_mask_stack` on the
    reference, so a trait lost to the NEAREST resize is lost here too)."""
    from .pairs import trait_mask_stack

    hist: dict[str, int] = {}
    for ref, _q in pairs.pairs[:n]:
        _y, ids = trait_mask_stack(ds.load_labelmap(ref), mask_size)
        k = str(len(ids))
        hist[k] = hist.get(k, 0) + 1
    return hist


def replay_checks(ds, pairs, training: dict | None, mask_size: int) -> dict:
    """Does the rebuilt `CyclePairs` reproduce the run the training report describes?

    ``training`` is ``record["training"]`` of the fine-tune bundle (`train_lora`'s report).
    A `--max-steps` smoke has ``steps_per_epoch < train_images``; that is ``steps_capped``,
    recorded rather than raised, and the histogram is compared over the steps that ran.
    The loss-parity number (`loss_first`) is filled in by the caller once the base arm has
    scored global step 0 — it is the actual proof; these are its preconditions.
    """
    n_pairs = len(pairs)
    n_images = sum(a for a, _ in pairs.coverage.values())
    out = {
        "n_pairs": n_pairs, "train_images": n_images,
        "train_images_total": sum(t for _, t in pairs.coverage.values()),
        "coverage": {g: list(v) for g, v in pairs.coverage.items()},
        "reference": pairs.reference_mode,
        "train_record": training is not None,
    }
    if training is None:
        out["checks"] = {"steps_per_epoch_matches": None, "train_images_matches": None,
                         "mask_channels_matches": None, "steps_capped": None}
        return out
    spe = int(training["steps_per_epoch"])
    capped = spe < n_pairs
    hist_train = training.get("mask_channels")
    hist_here = (mask_channel_histogram(ds, pairs, spe, mask_size)
                 if pairs.mask_mode == "per_trait" else None)
    out |= {
        "steps_per_epoch": spe,
        "train_images_recorded": training.get("train_images"),
        "mask_channels_recorded": hist_train,
        "mask_channels_replayed": hist_here,
        "loss_first_train": training.get("loss_first"),
        "checks": {
            "steps_per_epoch_matches": (spe == n_pairs) or capped,
            "train_images_matches": training.get("train_images") == n_images,
            "mask_channels_matches": (hist_train == hist_here),
            "steps_capped": capped,
        },
    }
    return out


def loss_parity(loss_probe_base: float, loss_first_train: float | None) -> dict:
    """The replay proof: at global step 0 LoRA's B is still zero, so the base arm's loss
    on pair 0 must equal the first recorded training loss up to autocast noise."""
    if loss_first_train is None:
        return {"loss_first_probe_base": loss_probe_base, "loss_first_train": None,
                "loss_first_rel_err": None}
    rel = abs(loss_probe_base - loss_first_train) / max(abs(loss_first_train), 1e-12)
    return {"loss_first_probe_base": loss_probe_base,
            "loss_first_train": float(loss_first_train),
            "loss_first_rel_err": float(rel)}


# ── bundle helpers ────────────────────────────────────────────────────────────


def write_labelmap_png(path, lab: np.ndarray) -> None:
    """uint8 PNG of an integer label map. Refuses labels > 255 (none exist: beetle 5,
    butterfly <= 10) rather than wrapping them."""
    import cv2

    lab = np.asarray(lab)
    if lab.size and int(lab.max()) > 255:
        raise ValueError(f"label map has ids up to {int(lab.max())} — does not fit uint8")
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), lab.astype(np.uint8)):
        raise OSError(f"could not write {path}")


def read_labelmap_png(path) -> np.ndarray:
    import cv2

    lab = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if lab is None:
        raise FileNotFoundError(path)
    return lab.astype(np.int32)


def fmt_per_trait(d: dict) -> str:
    """``"1:0.7100;3:0.4000"`` — one CSV cell per leg, parseable back by `parse_per_trait`."""
    return ";".join(f"{int(t)}:{v:.4f}" for t, v in sorted(d.items()))


def parse_per_trait(s: str) -> dict[int, float]:
    if not s:
        return {}
    return {int(k): float(v) for k, v in (kv.split(":") for kv in s.split(";"))}


def macro_over(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None
