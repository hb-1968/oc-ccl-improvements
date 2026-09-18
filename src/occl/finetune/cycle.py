"""The open-close cycle and its loss — ported from `SST/src/sst/oc_ccl.py:153-234`.

The cycle: prompt the reference with its GT mask, propagate to the query, then prompt the
query with *its own prediction* and propagate back. The closing mask is supervised against
the reference's GT. Cycle-consistency needs no query labels, which is the whole appeal.

**Defect A2 — the prompt is not differentiable.** `oc_ccl.py:209` closes the cycle with

    soft_query_mask = (pred_query_mask > 0).float()

A step function has zero derivative almost everywhere, so `d(loss)/d(forward leg)` is
exactly zero: the reverse leg trains and the forward leg does not. Half the cycle is
decorative. This is the defect the S8 gate's `forward_leg_rel_grad > 1e-3` check exists to
prove fixed.

**The fix is exact, not an approximation.** `sam2_base.py:417-419` turns a prompt mask into
logits with fixed constants:

    out_scale, out_bias = 20.0, -10.0
    high_res_masks = mask_inputs_float * out_scale + out_bias

so feeding ``m = (logits + 10) / 20`` makes SAM 2 reconstruct ``logits`` **identically** —
``((l + 10) / 20) * 20 - 10 == l``. The forward value is unchanged; only the path becomes
linear and differentiable. That is why this is `logit_reparam` and not "a soft threshold":
there is no temperature to tune and no approximation error to bound.

The hard-threshold path is kept reachable, because a test that asserts the gradient is
non-zero under the fix proves nothing unless it can also show it was zero before.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: `sam2/modeling/sam2_base.py:417`. Read from SAM 2's own constants rather than restated,
#: so a future SAM 3 that changes them breaks loudly here instead of silently rescaling the
#: prompt.
SAM2_OUT_SCALE = 20.0
SAM2_OUT_BIAS = -10.0

PROMPT_MODES = ("logit_reparam", "hard_threshold", "ste_threshold", "ste_memory")


def logits_to_prompt(logits: torch.Tensor, mode: str = "logit_reparam") -> torch.Tensor:
    """Mask logits -> a `mask_inputs` prompt.

    ``logit_reparam`` is the exact inverse of SAM 2's own mask->logit map, so the round
    trip is the identity and gradients flow. ``hard_threshold`` is SST's original, retained
    only so tests can demonstrate the gradient it kills.

    ``ste_threshold`` (S18) is the release's VALUE with the reparam's GRADIENT — a
    straight-through estimator: the prompt the reverse init sees is the binary mask
    (bit-identical to ``hard_threshold``, so SAM 2 reconstructs ±10 logits, its own
    training distribution), and the backward pass is ``logit_reparam``'s (1/20 into
    every logit). Defect A2's value was never the defect; its gradient was. What the
    value decides is WHAT MAY CROSS the centre of the palindrome: under ``logit_reparam``
    the closing frame receives the full continuous logit map — 1024² floats per trait,
    which S17b/c measured the fine-tune using as a side channel (closing +2.4 ✱ on
    channels whose forward mask did not move; the (3e-5, 3 ep) corner's closing +9.7 ✱ at
    forward −0.6). Under the STE only the argmax mask crosses — exactly what the
    evaluator scores — so any "encoding" of y₀ into the forward output costs forward
    accuracy directly. ``hard + (soft - soft.detach())`` is exactly ``hard`` in value
    (the two soft terms cancel bit for bit) and has ``soft``'s graph.

    ``ste_memory`` (S18) is the same prompt tensor; the difference is downstream, in
    `oc_ccl_step`: the tracker's `memory_ste` context is opened around the reverse init,
    so the memory encoder's mask input keeps the binary path's VALUE and takes the
    continuous logits' GRADIENT. Measured on the real model, the plain ``ste_threshold``
    leaves the forward leg only the pointer route (forward/reverse per channel 0.1 vs
    19–449 under ``logit_reparam``): SAM 2's memory reads ``sigmoid(20·m − 10)``, which
    is saturated at a binary ``m`` — in this architecture the sub-threshold channel and
    the memory-route gradient are one and the same quantity. ``ste_memory`` separates
    them: the release's value with the paper's gradient.
    """
    if mode == "logit_reparam":
        return (logits - SAM2_OUT_BIAS) / SAM2_OUT_SCALE
    if mode == "hard_threshold":
        return (logits > 0).float()
    if mode in ("ste_threshold", "ste_memory"):
        hard = (logits > 0).to(logits.dtype)
        soft = (logits - SAM2_OUT_BIAS) / SAM2_OUT_SCALE
        return hard + (soft - soft.detach())
    raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got {mode!r}")


def _memory_ste_context(tracker, prompt_mode: str, logits):
    """The `memory_ste` context for the reverse init under ``ste_memory``; a no-op
    context otherwise. Refuses a tracker that cannot wrap its memory encoder."""
    from contextlib import nullcontext

    if prompt_mode != "ste_memory":
        return nullcontext()
    if not hasattr(tracker, "memory_ste"):
        raise ValueError("ste_memory needs a tracker that exposes memory_ste "
                         "(tracker.DifferentiableSAM2Tracker)")
    return tracker.memory_ste(logits)


def bce_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unchanged from `oc_ccl.py:153`."""
    return F.binary_cross_entropy_with_logits(pred_logits, target)


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """Unchanged from `oc_ccl.py:158`."""
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


#: S13 margin fold — recorded verbatim so the run record names the statistic and form.
MARGIN_STATISTIC = "hard_claim_mean_top_logit"
MARGIN_FORM = "hinge"


def claimed_confidence(logits: torch.Tensor) -> torch.Tensor:
    """TTA's decision statistic (`adapters/tta.py:confidence`), values-differentiable.

    ``mean over claimed pixels (top-channel logit > 0) of the top-channel logit`` —
    identical in value to the numpy eval statistic on any non-empty claim (pinned by
    the bridge test). The claim mask itself comes from ``>``, which has no ``grad_fn``,
    so the set selection is structurally detached — the same subgradient the model
    already trains through in ``amax`` over channels.

    fp32 before any reduction: a 1024x1024 sum in bf16 loses effectively the whole
    mantissa. An empty claim returns exactly 0.0 — finite where eval uses ``-inf``, so
    no inf/nan can reach the backward; the direction is still right (an empty
    inconsistent leg lets the hinge push the consistent one above m, an empty
    consistent leg contributes no gradient on that side).
    """
    top = logits.float().amax(dim=0)
    w = (top > 0).float()
    return (top * w).sum() / w.sum().clamp(min=1.0)


def orientation_margin(tracker, backbone_query, x0, y0, pred_query_mask, n_obj: int,
                       margin_m: float):
    """Hinge demanding the as-given leg's claimed confidence beat the 180-rotated leg's.

    Label-free: it compares two self-generated predictions and consults no GT, so it
    keeps OC-CCL's semi-supervised character — and it is the one term that CONSTRUCTS a
    180 degree support-query relation, which the ordinary cycle never does (the coherent
    theory for why every cycle-supervision arm measured ~null on Beetle's flip cohort).

    The rotated support chain is fully independent: `tracker.track_with_mask` with no
    ``output_dict`` on an init frame starts a fresh memory context. 180 degrees is
    ``torch.flip`` on both spatial dims for image and mask together — exact, no
    interpolation. The features cannot be flipped instead of re-encoded: Hiera is not
    rotation-equivariant, so the extra encode is unavoidable. ``as_given`` is the
    consistent orientation BY ASSUMPTION (train poses are canonical); the assumption is
    recorded in the run config, and the pose-prior shortcut it risks is the stage's
    pre-registered disqualifier, instrumented by the recorded ``rotated_chosen`` rate.

    A hinge rather than softplus: it vanishes once the ordering is satisfied by ``m``,
    so the term cannot keep distorting calibration after it has done its job — bounded
    interference is the answer to the S8d silent-pick-degradation failure. Uniform
    logit inflation adds the same constant to both confidences and cancels in the
    difference (pinned by test), so it is not a descent direction.
    """
    x0r = torch.flip(x0, dims=(-2, -1))
    y0r = torch.flip(y0, dims=(-2, -1))
    backbone_rot = tracker.encode_image(x0r)
    _rot_init, od_rot = tracker.track_with_mask(backbone_rot, y0r, is_init_frame=True,
                                                n_objects=n_obj)
    rot_out, _ = tracker.track_with_mask(backbone_query, mask=None,
                                         is_init_frame=False, output_dict=od_rot,
                                         n_objects=n_obj)
    conf_up = claimed_confidence(pred_query_mask)
    conf_rot = claimed_confidence(rot_out["pred_masks_high_res"])
    loss = F.relu(margin_m - (conf_up - conf_rot))
    with torch.no_grad():
        up_top = pred_query_mask.float().amax(dim=0)
        rot_top = rot_out["pred_masks_high_res"].float().amax(dim=0)
        parts = {"conf_up": float(conf_up), "conf_rot": float(conf_rot),
                 "gap": float(conf_up - conf_rot), "margin": float(loss),
                 "margin_violated": bool(loss > 0),
                 "inconsistent_won": bool(conf_rot > conf_up),
                 "empty_claim_up": bool(not (up_top > 0).any()),
                 "empty_claim_rot": bool(not (rot_top > 0).any())}
    return loss, parts


def overlap_penalty(logits: torch.Tensor) -> torch.Tensor:
    """SAM-MT's pairwise overlap term ``mean(P_i * sum_{j!=i} P_j)`` in O(C*H*W).

    Exactly 0 at C=1 (``S - P == 0``); callers skip it there so the graph and cost stay
    untouched.
    """
    p = torch.sigmoid(logits)
    return (p * (p.sum(dim=0, keepdim=True) - p)).mean()


def oc_ccl_step(tracker, x0, xu, y0, prompt_mode: str = "logit_reparam",
                memory_reset: bool = True, bce_w: float = 1.0, dice_w: float = 1.0,
                scnp_width: int = 0, scnp_joint: bool = True,
                excl_mode: str = "off", xovl_weight: float = 0.0,
                xovl_where: str = "query", bshape_weight: float = 0.0,
                margin_weight: float = 0.0, margin_m: float = 1.0,
                capture: dict | None = None, found_mask: bool = False,
                lora_legs: str = "all", obj_weight: float = 0.0,
                x0_close=None, y0_close=None):
    """One full cycle. Returns ``(loss, pred_closing_mask, parts)``.

    ``scnp_width`` is **S9**, and 0 means off — S8's arms must stay reproducible bit for
    bit, so this cannot change the faithful path by default. At width w > 0 the BCE and Dice
    terms are evaluated on SCNP-transformed logits (`finetune.scnp`), which score every
    pixel by its poorest same-class neighbour.

    Note what this can and cannot reach in its present form: `tracker.track_with_mask`
    is single-object, so `pred_closing_mask` is one merged-foreground channel and SCNP
    sharpens that silhouette's topology only. Trait assignment needs per-trait channels for
    "same class" to mean anything — that is S9b. See `finetune/scnp.py`.

    ``memory_reset`` is defect **A7**. SST hardcodes a reset (`oc_ccl.py:203-206` builds a
    fresh dict for the reverse leg), and `registry.FAITHFUL_FINETUNE` records that as the
    faithful setting — so ``True`` is SST's behaviour, not a change. ``False`` reuses the
    forward leg's memory, which lets the reverse pass attend to the reference frame it is
    trying to reconstruct. That is a shortcut the loss cannot see, and it is exactly why the
    arm is worth measuring rather than assuming.

    **The object dimension.** ``y0`` is ``(C, 1, H, W)`` — C=1 is the release's merged
    silhouette, C>1 is the paper's per-trait cycle (`pairs.trait_mask_stack`), with each
    trait an object in SAM 2's object batch. All four legs run at ``n_objects=C`` in one
    pass; the closing prediction comes back per channel and BCE/Dice already reduce as a
    mean over channels (Dice is per-batch-element then mean), so the loss is the paper's
    "minimise the discrepancy between ŷ₀† and y₀" over all C trait channels at once.

    **S13 knobs** — all default off, so S8-S12 stay reproducible bit for bit (the
    ``scnp_width=0`` discipline). ``excl_mode`` replaces the BCE slot with the
    STM/Cutie exclusivity CE (`finetune.excl`; composition with SCNP raises — both
    transform the same slot). ``xovl_weight`` adds the label-free overlap penalty on
    the forward-leg prediction, the one tensor no GT can reach (``xovl_where="both"``
    adds the closing frame). ``bshape_weight`` adds the signed-distance gross-shape
    term (`finetune.bshape`). ``margin_weight`` adds the orientation-contrastive
    confidence margin — one extra encode + init + forward leg on the 180-rotated
    support in its own memory chain; the margin-ONLY objective is `margin_step`. None
    of these are mirrored in `gradient_report` — see its docstring for why that
    exclusion is load-bearing.

    ``capture`` (S17) — a dict the caller owns; when given, the forward-leg prediction
    ``pred_query_mask`` is stored under ``"forward"`` before the cycle closes. The step
    otherwise discards it: the loss only ever sees the closing mask, which is exactly why
    nobody had looked at what the cycle says about the query. ``None`` is the default and
    the step is bit-for-bit unchanged under it — every S8–S16 arm passes nothing here.

    ``found_mask`` (S17 arm) — drop from the loss every channel the forward leg did not
    FIND on the query (no logit > 0 anywhere: `gradient_report`'s
    ``per_channel_query_found``). Measured on all 120 beetle pairs of the base model
    (README S17): 18 such channel-steps, 3 % of the epoch's channels, carried 65 % of its
    BCE — the reverse leg turns an empty prompt into a confident whole-body blob, the
    independent-sigmoid BCE charges 30–47 for it, and S10 proved the channel's forward-leg
    gradient is exactly zero, so all of that gradient buys is calibration of hidden logits
    under garbage prompts. The pixel and Dice terms become means over the found channels;
    a pair with no found channel contributes a zero loss that keeps the graph (the
    optimiser takes an empty step). Composition with ``scnp`` or ``excl`` raises: both
    transform the joint channel stack and a masked subset is a different objective
    nobody pre-registered. ``bce_w=0`` (the ``nobce`` arm) composes freely.

    ``lora_legs="forward"`` (S17b arm `fwdonly`) — the reverse init and the closing frame
    (legs 3–4) run with every adapter switched off (`lora.lora_disabled`), so the closing
    head is the base model's and cannot learn to read ``x0``: S17 measured that
    absorption (closing gain largest where the forward prompt was worst, forward leg
    unmoved). The only way left to lower the closing loss is a better forward mask; the
    gradient still flows through the base closing modules into the forward leg's
    adapters. ``"all"`` is the default and every S8–S17 arm.

    ``obj_weight`` (S17b arm `objness`) — adds ``obj_weight × BCE(object_score_logits of
    the forward frame, 1)`` per prompted channel: "every trait the reference carries is
    present on the query". The score is what the −1024 sentinel keys on; a channel it
    declares absent gets no mask gradient at all (S17 anatomy), so this is the one term
    that can reach the forward leg's unfound channels. Requires a tracker that exposes
    ``object_score_logits`` (`tracker.DifferentiableSAM2Tracker`); raises otherwise.

    ``x0_close`` / ``y0_close`` (S18 arm `perturb`) — the closing end of the palindrome
    is a PERTURBED copy of the reference (`pairs.CyclePairs.closing`: the same image and
    label map under one independently seeded rotation): leg 4 encodes ``x0_close`` and
    the supervision target is ``y0_close``. The opening reference, the forward leg and
    the reverse init are untouched. Both ``None`` (the default, every S8–S17 arm) is
    byte-identical to closing on ``x0``/``y0``. The channel counts must agree.

    ``capture["hard_replay"] = True`` (S18, the probe's steganography column) — after the
    closing leg, legs 3–4 run AGAIN under ``torch.no_grad()`` with the forward logits
    binarised (``hard_threshold``) before the reverse init, on the same backbones and the
    same weights; the closing logits land in ``capture["closing_hard"]`` (and the head's
    score in ``capture["cls_obj_hard"]``). Closing quality under the continuous prompt
    minus closing quality under the binary one is how much of the cycle's reconstruction
    rides on sub-threshold content of the forward logits — information the evaluator
    never sees. Nothing else in the step reads the flag; without it (and without
    ``capture`` at all) the tracker is called exactly as before.
    """
    n_obj = int(y0.shape[0])
    if lora_legs not in ("all", "forward", "forward_link"):
        raise ValueError(f"lora_legs must be 'all', 'forward' or 'forward_link', got {lora_legs!r}")
    if found_mask and (scnp_width or excl_mode != "off"):
        raise ValueError("found_mask composes with neither scnp nor excl — both act on the "
                         "joint channel stack (one mechanism per arm)")
    if scnp_width and excl_mode != "off":
        raise ValueError(
            "scnp and excl both transform the pixel term; their composition is "
            "undefined and must fail loudly (S12 closed SCNP regardless)")
    if xovl_weight and xovl_where not in ("query", "both"):
        raise ValueError(f"xovl_where must be 'query' or 'both', got {xovl_where!r}")
    # SCNP × per-trait — S12. This composition sat behind a deliberate guard until it
    # had a falsifier design; S12 is that design. The semantics come free from shape:
    # `scnp_logits` transforms each channel against ITS OWN binary target, so
    # "same-class neighbour" means same-TRAIT neighbour — a background-of-trait-k pixel
    # is scored by its strongest z_k neighbour, which is exactly the cross-trait bleed
    # `scnp.py`'s docstring says the merged form could never reach. Per-channel
    # independence is pinned by test (C=2 SCNP loss == mean of the two C=1 SCNP losses).
    # 1. reference, prompted with ground truth
    backbone_ref = tracker.encode_image(x0)
    _ref_out, output_dict = tracker.track_with_mask(backbone_ref, y0, is_init_frame=True,
                                                    n_objects=n_obj)

    # 2. forward leg: propagate to the query
    backbone_query = tracker.encode_image(xu)
    query_out, output_dict_fwd = tracker.track_with_mask(
        backbone_query, mask=None, is_init_frame=False, output_dict=output_dict,
        n_objects=n_obj)
    pred_query_mask = query_out["pred_masks_high_res"]
    if capture is not None:
        capture["forward"] = pred_query_mask
    fwd_obj = query_out.get("object_score_logits")
    if obj_weight and fwd_obj is None:
        raise ValueError("obj_weight needs a tracker that exposes object_score_logits")

    margin_loss = mparts = None
    if margin_weight:
        # S13 margin fold — the rotated support chain, run against the same query.
        # Not mirrored in gradient_report — see its docstring.
        margin_loss, mparts = orientation_margin(
            tracker, backbone_query, x0, y0, pred_query_mask, n_obj, margin_m)

    # 3. close the cycle on the query's OWN prediction — the differentiable link (A2)
    from .lora import lora_disabled

    _model = getattr(tracker, "model", None)
    # S18 `perturb`: the closing frame is a perturbed copy of the reference (one more
    # encode); the target follows it. Both None -> the reference's own backbone and y0.
    if (x0_close is None) != (y0_close is None):
        raise ValueError("x0_close and y0_close come together")
    if y0_close is not None and int(y0_close.shape[0]) != n_obj:
        raise ValueError(f"y0_close has {int(y0_close.shape[0])} channels, y0 has {n_obj}")
    backbone_close = backbone_ref if x0_close is None else tracker.encode_image(x0_close)
    if y0_close is not None:
        y0 = y0_close

    def _close(prompt, mode):
        """Legs 3-4 for one prompt tensor: the reverse init on the query, then the
        closing frame. Factored so the hard replay reruns exactly these calls."""
        if memory_reset:
            rod = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        else:
            rod = {"cond_frame_outputs": dict(output_dict_fwd["cond_frame_outputs"]),
                   "non_cond_frame_outputs": dict(output_dict_fwd["non_cond_frame_outputs"])}
        # "forward": adapters off for the reverse init AND the closing frame (legs 3-4);
        # "forward_link" (S17c): off for the closing frame only — the reverse init's
        # memory encoding of the forward mask (the link) stays trainable, so the arm
        # says which half of the link carries the signal.
        with lora_disabled(_model if lora_legs == "forward" else None):
            # S18 `ste_memory`: the memory straight-through is open for the reverse
            # init only (the one frame whose memory is built from a prompt)
            with _memory_ste_context(tracker, mode, pred_query_mask):
                _q_cond, rod = tracker.track_with_mask(
                    backbone_query, prompt, is_init_frame=True, output_dict=rod,
                    n_objects=n_obj)

            # 4. reverse leg: back to the reference (or its perturbed copy)
            with lora_disabled(_model if lora_legs == "forward_link" else None):
                out, _ = tracker.track_with_mask(
                    backbone_close, mask=None, is_init_frame=False, output_dict=rod,
                    n_objects=n_obj)
        return out

    ref_closing_out = _close(logits_to_prompt(pred_query_mask, prompt_mode), prompt_mode)
    pred_closing_mask = ref_closing_out["pred_masks_high_res"]
    cls_obj = ref_closing_out.get("object_score_logits")
    if capture is not None and capture.get("hard_replay"):
        # S18: the same closing legs with the forward logits BINARISED first — what the
        # closing frame reconstructs from the argmax mask alone (no graph: a read).
        with torch.no_grad():
            hard_out = _close(logits_to_prompt(pred_query_mask.detach(), "hard_threshold"),
                              "hard_threshold")
        capture["closing_hard"] = hard_out["pred_masks_high_res"]
        capture["cls_obj_hard"] = hard_out.get("object_score_logits")

    # 5. supervise the closing mask against the reference GT.
    # NEAREST, not bilinear (S14): this guard never fired in any recorded run (sam2
    # sizes always matched), so the mode change is byte-stable for S8-S13 — but on
    # sam3 it fires EVERY step (prompts must be 1152 = 72*16 for the memory grid,
    # while propagated legs predict at image_size 1008), and a bilinear resize would
    # soften the binary target, silently changing what per-trait/excl supervise.
    if y0.shape[-2:] != pred_closing_mask.shape[-2:]:
        y0 = F.interpolate(y0, size=pred_closing_mask.shape[-2:], mode="nearest")
    if scnp_width:
        from .scnp import scnp_loss

        pixel = scnp_loss(bce_loss, pred_closing_mask, y0, scnp_width, scnp_joint)
        dice = scnp_loss(dice_loss, pred_closing_mask, y0, scnp_width, scnp_joint)
    elif excl_mode != "off":
        # S13 `excl` — the STM/Cutie exclusivity CE replaces the BCE slot (equal to it
        # at C=1 under zerocat, so this is a strict generalisation, not a removal).
        # Dice stays on the independent sigmoids: one mechanism per arm.
        from .excl import excl_ce

        pixel = excl_ce(pred_closing_mask, y0, excl_mode)
        dice = dice_loss(pred_closing_mask, y0)
    elif found_mask:
        # S17 `found_mask`: per-channel terms, averaged over the FOUND channels only.
        # `bce_loss`/`dice_loss` are global means / per-element-then-mean, so at
        # n_found == C both reduce to exactly the plain terms (pinned by test).
        found = (pred_query_mask.detach() > 0).flatten(1).any(dim=1)          # (C,)
        w = found.float().to(pred_closing_mask.dtype)
        n_found = float(w.sum())
        bce_c = torch.stack([F.binary_cross_entropy_with_logits(pred_closing_mask[c], y0[c])
                             for c in range(n_obj)])
        pr = torch.sigmoid(pred_closing_mask).flatten(1)
        tg = y0.flatten(1)
        dice_c = 1.0 - (2.0 * (pr * tg).sum(dim=1) + 1.0) / (pr.sum(dim=1) + tg.sum(dim=1) + 1.0)
        if n_found > 0:
            pixel = (w * bce_c).sum() / n_found
            dice = (w * dice_c).sum() / n_found
        else:
            pixel = pred_closing_mask.sum() * 0.0
            dice = pred_closing_mask.sum() * 0.0
    else:
        pixel = bce_loss(pred_closing_mask, y0)
        dice = dice_loss(pred_closing_mask, y0)
    loss = bce_w * pixel + dice_w * dice
    if excl_mode != "off":
        # `parts["bce"]` stays the PLAIN BCE (detached) so the printed loss line and
        # the curve remain comparable to S8-S12's; the trained term is `parts["ce"]`.
        from .excl import softmax_dice

        with torch.no_grad():
            parts = {"bce": float(bce_loss(pred_closing_mask, y0)),
                     "ce": float(pixel), "excl_mode": excl_mode,
                     "dice": float(dice), "n_objects": n_obj,
                     "dice_softmax": softmax_dice(pred_closing_mask, y0)}
    elif found_mask:
        # `parts["bce"]`/`["dice"]` stay the PLAIN terms so the loss line and curve stay
        # comparable to S8-S16's; the trained terms are the found-masked ones.
        with torch.no_grad():
            parts = {"bce": float(bce_loss(pred_closing_mask, y0)),
                     "dice": float(dice_loss(pred_closing_mask, y0)),
                     "bce_found": float(pixel), "dice_found": float(dice),
                     "found_mask": [bool(f) for f in found], "n_found": int(n_found),
                     "n_objects": n_obj}
    else:
        parts = {"bce": float(pixel.detach()), "dice": float(dice.detach()),
                 "n_objects": n_obj}
    if bce_w != 1.0 or dice_w != 1.0:
        parts |= {"bce_w": float(bce_w), "dice_w": float(dice_w)}
    if n_obj > 1:
        # Per-channel Dice, detached, so the log can show WHICH trait the cycle is
        # failing to carry rather than one blended number. Per-channel BCE too (S17):
        # the pixel term is a mean over ALL channels, so one channel with confident
        # wrong logits over a large region can carry the whole pair's loss, and the
        # blended number cannot say which — the probe decomposes the epoch by channel.
        with torch.no_grad():
            p = torch.sigmoid(pred_closing_mask).flatten(1)
            t = y0.flatten(1)
            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            parts["dice_per_channel"] = [
                round(float(1.0 - (2.0 * i + 1.0) / (u + 1.0)), 4)
                for i, u in zip(inter, union)]
            parts["bce_per_channel"] = [
                round(float(F.binary_cross_entropy_with_logits(
                    pred_closing_mask[c], y0[c])), 4)
                for c in range(n_obj)]
    if scnp_width:
        # The untransformed values too: under `joint` the reported loss is roughly double
        # the S8 figure, and a loss curve that cannot be compared against S8's is a curve
        # nobody can read.
        with torch.no_grad():
            parts |= {"bce_plain": float(bce_loss(pred_closing_mask, y0)),
                      "dice_plain": float(dice_loss(pred_closing_mask, y0)),
                      "scnp_width": scnp_width, "scnp_joint": bool(scnp_joint)}
    if xovl_weight and n_obj > 1:
        # S13 `xovl` — label-free repulsion on the forward leg. Exactly 0 at C=1, so
        # skipped rather than computed. Not mirrored in gradient_report — its docstring.
        xv = overlap_penalty(pred_query_mask)
        if xovl_where == "both":
            xv = xv + overlap_penalty(pred_closing_mask)
        loss = loss + xovl_weight * xv
        parts |= {"xovl": float(xv.detach()), "xovl_where": xovl_where}
    if bshape_weight:
        # S13 `bshape` — placed AFTER the resize guard so phi matches y0's actual
        # resolution; `(y0 > 0.5)` inside makes it immune to the soft-rebind case.
        from .bshape import bshape_loss

        bs = bshape_loss(pred_closing_mask, y0)
        loss = loss + bshape_weight * bs
        parts["bshape"] = float(bs.detach())
    if margin_weight:
        loss = loss + margin_weight * margin_loss
        parts |= mparts | {"margin_weight": margin_weight, "margin_m": margin_m}
    if fwd_obj is not None:
        # the objectness scores, recorded whenever the tracker exposes them (the probe
        # reads them per channel); the `objness` term only when weighted
        parts["fwd_obj_score"] = [round(float(v), 4) for v in fwd_obj.detach().flatten()]
        if cls_obj is not None:
            parts["cls_obj_score"] = [round(float(v), 4) for v in cls_obj.detach().flatten()]
    if obj_weight:
        obj = F.binary_cross_entropy_with_logits(fwd_obj.flatten().float(),
                                                 torch.ones(n_obj, device=fwd_obj.device))
        loss = loss + obj_weight * obj
        parts |= {"obj": float(obj.detach()), "obj_weight": float(obj_weight)}
    if lora_legs != "all":
        parts["lora_legs"] = lora_legs
    if x0_close is not None:
        parts["close_aug"] = True
    return loss, pred_closing_mask, parts


def margin_step(tracker, x0, xu, y0, margin_m: float = 1.0):
    """The margin term ALONE — no reconstruction anywhere. Returns ``(loss, pred, parts)``.

    The cleanest single test of the fold's mechanism: on Beetle every cycle loss
    measures <= 0, so any gain from this objective is selection-sharpening in
    isolation. Training signal exists without a reconstruction term because the
    statistic is a function of the (LoRA'd) mask decoder's outputs on
    memory-conditioned features written by the (LoRA'd) memory encoder — the same path
    the cycle's forward leg already proves live.

    Cost: 3 encodes + 4 track legs — one extra encode over a plain cycle step, and the
    reverse chain is never built.
    """
    n_obj = int(y0.shape[0])
    backbone_ref = tracker.encode_image(x0)
    _ref_out, output_dict = tracker.track_with_mask(backbone_ref, y0,
                                                    is_init_frame=True,
                                                    n_objects=n_obj)
    backbone_query = tracker.encode_image(xu)
    query_out, _ = tracker.track_with_mask(
        backbone_query, mask=None, is_init_frame=False, output_dict=output_dict,
        n_objects=n_obj)
    pred_query_mask = query_out["pred_masks_high_res"]
    loss, parts = orientation_margin(tracker, backbone_query, x0, y0,
                                     pred_query_mask, n_obj, margin_m)
    return loss, pred_query_mask, dict(parts, n_objects=n_obj, margin_m=margin_m)


def gradient_report(tracker, x0, xu, y0, prompt_mode: str = "logit_reparam") -> dict:
    """How much of the loss gradient reaches the forward leg — the S8 gate's A2 evidence.

    **Deliberately excludes every S9+/S13 loss knob (scnp, excl, xovl, bshape,
    margin) — do not "fix" this.** The report exists to prove THE CYCLE carries
    gradient to the forward leg through the closing prompt and the reverse leg.
    Mirroring the query-side terms (xovl, margin) would hand the forward probe a direct
    gradient that never traverses the reverse leg: `forward_to_reverse` would stop
    measuring the cycle mechanism, and the s8a/s10ft "hard-threshold exactly 0.0"
    predicates would silently break. The margin term has its own
    `margin_gradient_report`. The exclusion is pinned by test.

    Measured on the forward leg's own *activation*, not on parameters: the two legs share
    parameters (the same modules run twice), so a non-zero parameter gradient proves only
    that *something* trained, which the reverse leg alone satisfies. The query prediction is
    the one tensor on the forward path only, so gradient arriving there is unambiguous.

    **Reported against a reference, because a bare magnitude is uninterpretable.** The
    closing mask sits on the reverse leg and always receives gradient, in the same units and
    at the same resolution. `forward_to_reverse` is their ratio — scale-free, and the only
    number here that can be argued about. The raw norms are kept so the ratio can be
    checked rather than trusted.

    A relative-to-its-own-norm figure is also returned for continuity with the original S8
    gate field, but it is a poor statistic: `pred_masks_high_res` is 1024x1024 logits of
    magnitude ~10, so its norm is ~1e4 and the ratio is tiny for reasons that have nothing
    to do with whether the fix worked.

    Under ``hard_threshold`` every forward-leg figure is **exactly** 0.0 — not small,
    structurally zero, because `>` returns a tensor with no `grad_fn` at all.

    At C > 1 (per-trait cycle) the report additionally carries
    ``per_channel_forward_to_reverse``: the same ratio taken channel by channel, because a
    healthy mean can hide one dead trait channel.

    **A dead channel can be structural, and the gate must not read it as a wiring bug.**
    Measured on the real model (S10 CPU smoke, beetle pair 0): four channels at 20-449,
    one at exactly 0.0. The zero is the trait the model predicted as absent everywhere on
    the query: its reparam prompt has no value > 0 (no logit > −10), so
    `_use_mask_as_output` sets `lambda_is_obj_appearing = 0.0` and, under
    `fixed_no_obj_ptr: true`, multiplies the differentiable obj_ptr by an exact zero
    before adding the constant `no_obj_ptr`; the memory route stays sigmoid-differentiable
    (`is_mask_from_pts` is False for mask prompts, so the eval-mode binarize never fires)
    but saturates to numerical zero at such logits. So forward-leg gradient for a trait
    exists only on queries where the model claims at least one pixel of it — intrinsic to
    cycle training, not a defect: a cycle cannot teach the forward leg about an object it
    never carried forward. ``per_channel_query_found`` (any logit > 0, the project's
    standard claim rule) is recorded so the gate can require gradient on exactly the
    found channels.
    """
    n_obj = int(y0.shape[0])
    backbone_ref = tracker.encode_image(x0)
    _ref_out, output_dict = tracker.track_with_mask(backbone_ref, y0, is_init_frame=True,
                                                    n_objects=n_obj)
    backbone_query = tracker.encode_image(xu)
    query_out, _fwd = tracker.track_with_mask(
        backbone_query, mask=None, is_init_frame=False, output_dict=output_dict,
        n_objects=n_obj)

    fwd_probe = query_out["pred_masks_high_res"].detach().requires_grad_(True)

    prompt = logits_to_prompt(fwd_probe, prompt_mode)
    reverse = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    with _memory_ste_context(tracker, prompt_mode, fwd_probe):
        _q, reverse = tracker.track_with_mask(backbone_query, prompt, is_init_frame=True,
                                              output_dict=reverse, n_objects=n_obj)
    closing, _ = tracker.track_with_mask(backbone_ref, mask=None, is_init_frame=False,
                                         output_dict=reverse, n_objects=n_obj)

    # The reverse-leg reference. Re-attached as a leaf so both gradients are taken at the
    # same point in the graph and in the same units.
    rev_probe = closing["pred_masks_high_res"]
    pred = rev_probe
    target = y0 if y0.shape[-2:] == pred.shape[-2:] else F.interpolate(
        y0, size=pred.shape[-2:], mode="nearest")  # nearest: see oc_ccl_step step 5
    loss = bce_loss(pred, target) + dice_loss(pred, target)

    grads = torch.autograd.grad(loss, [fwd_probe, rev_probe], allow_unused=True,
                                retain_graph=False)
    fwd_grad, rev_grad = grads
    fwd_n = 0.0 if fwd_grad is None else float(fwd_grad.norm())
    rev_n = 0.0 if rev_grad is None else float(rev_grad.norm())
    act_n = float(fwd_probe.detach().norm())
    query_found = [bool((fwd_probe[c] > 0).any()) for c in range(n_obj)]
    per_channel = per_fwd = per_rev = None
    if n_obj > 1:
        f = (torch.zeros(n_obj) if fwd_grad is None
             else fwd_grad.flatten(1).norm(dim=1).cpu())
        r = (torch.zeros(n_obj) if rev_grad is None
             else rev_grad.flatten(1).norm(dim=1).cpu())
        per_channel = [float(fc / rc) if float(rc) else 0.0 for fc, rc in zip(f, r)]
        per_fwd = [float(v) for v in f]
        per_rev = [float(v) for v in r]
    return {
        "prompt_mode": prompt_mode,
        "n_objects": n_obj,
        "forward_grad_norm": fwd_n,
        "reverse_grad_norm": rev_n,
        "forward_activation_norm": act_n,
        "forward_to_reverse": (fwd_n / rev_n) if rev_n else 0.0,
        "per_channel_forward_to_reverse": per_channel,
        "per_channel_forward_norm": per_fwd,
        "per_channel_reverse_norm": per_rev,
        # Which traits the forward leg actually claimed (> 0 anywhere) on this query —
        # the condition under which a channel is REQUIRED to carry forward gradient.
        "per_channel_query_found": query_found,
        "rel_grad": (fwd_n / act_n) if act_n else 0.0,
    }


def margin_gradient_report(tracker, x0, xu, y0, margin_m: float = 1.0) -> dict:
    """Mechanism evidence for the margin term — deliberately SEPARATE from
    `gradient_report`, which excludes it (see its docstring).

    Both forward legs run with their predictions detached into leaf probes (the
    `gradient_report` trick); the hinge is computed on the probes and differentiated.
    When the hinge is satisfied (gap >= m) both gradients are structurally 0.0 — that
    is an inactive objective, not a wiring bug, so ``hinge_active`` is recorded and the
    gate conditions on it (the `per_channel_query_found` pattern).

    **Every leg runs under `torch.no_grad()` — deliberately, and exactly.** The probes
    are detached leaves, so the leg graphs would contribute NOTHING to the reported
    gradients; building them tracked is pure memory. On the raw model (all params
    requiring grad, as in `--diagnostics-only`) the tracked version tapes three 1024²
    Hiera-L encodes plus four legs and OOM-killed the S13 CPU smoke at 46 GB. Unlike
    `gradient_report`, nothing here needs a leg graph: that report measures
    transmission THROUGH the reverse legs, this one measures the statistic AT the
    probes.
    """
    n_obj = int(y0.shape[0])
    with torch.no_grad():
        backbone_ref = tracker.encode_image(x0)
        _ref_out, output_dict = tracker.track_with_mask(backbone_ref, y0,
                                                        is_init_frame=True,
                                                        n_objects=n_obj)
        backbone_query = tracker.encode_image(xu)
        q, _ = tracker.track_with_mask(backbone_query, mask=None, is_init_frame=False,
                                       output_dict=output_dict, n_objects=n_obj)
        x0r = torch.flip(x0, dims=(-2, -1))
        y0r = torch.flip(y0, dims=(-2, -1))
        backbone_rot = tracker.encode_image(x0r)
        _rot_init, od_rot = tracker.track_with_mask(backbone_rot, y0r,
                                                    is_init_frame=True,
                                                    n_objects=n_obj)
        r, _ = tracker.track_with_mask(backbone_query, mask=None, is_init_frame=False,
                                       output_dict=od_rot, n_objects=n_obj)

    up_probe = q["pred_masks_high_res"].detach().requires_grad_(True)
    rot_probe = r["pred_masks_high_res"].detach().requires_grad_(True)
    conf_up = claimed_confidence(up_probe)
    conf_rot = claimed_confidence(rot_probe)
    loss = F.relu(margin_m - (conf_up - conf_rot))
    g_up, g_rot = torch.autograd.grad(loss, [up_probe, rot_probe], allow_unused=True)
    return {
        "margin_m": margin_m,
        "hinge_active": bool(loss > 0),
        "grad_up_norm": 0.0 if g_up is None else float(g_up.norm()),
        "grad_rot_norm": 0.0 if g_rot is None else float(g_rot.norm()),
        "conf_up": float(conf_up),
        "conf_rot": float(conf_rot),
    }


def forward_leg_rel_grad(tracker, x0, xu, y0, prompt_mode: str = "logit_reparam") -> float:
    """Backwards-compatible scalar: gradient at the forward leg relative to the reverse leg.

    Deliberately `forward_to_reverse` and not the norm-relative figure — see
    `gradient_report`. Exactly 0.0 under `hard_threshold`.
    """
    return gradient_report(tracker, x0, xu, y0, prompt_mode)["forward_to_reverse"]
