"""Differentiable mask-conditioned tracking — ported from `SST/src/sst/oc_ccl.py:64-151`.

SAM 2's `SAM2VideoPredictor` decorates its public entry points with
`@torch.inference_mode()`, which blocks gradients. SST's workaround is to call the
underlying `SAM2Base` methods directly — `forward_image`, `_prepare_backbone_features`,
`track_step` — and that part is sound, so it is carried over essentially unchanged.

What is **not** carried over is `oc_ccl.py`'s `build_model`. The model is built by
`adapters.sst_sam2.SSTSam2Adapter.build()` instead, for three reasons that all bear on
whether an S8 number means anything:

* it loads the checkpoint **strictly**, raising on missing or unexpected keys, where
  `oc_ccl.py:53-57` prints a warning and continues — which would leave randomly
  initialised weights and train against noise;
* it is the same builder the evaluator uses, so the fine-tuned model is architecturally
  identical to the scored one by construction rather than by inspection;
* `model.eval()` is its post-condition (`sst_sam2.py:135`), which is defect **A5** —
  `oc_ccl.py:260,293` calls `model.train()`, leaving dropout active in modules the
  evaluator always runs in eval mode.

The adapter's `_get_image_feature` patch is inert here: it serves the video-predictor
inference path, and training calls `forward_image` directly.
"""

from __future__ import annotations


def _expand_backbone(backbone_out: dict, n: int) -> dict:
    """Expand batch-1 backbone features to ``n`` objects — SAM 2's own mechanism.

    Mirrors `sam2_video_predictor.py::_get_image_feature` (the `.expand(batch_size, ...)`
    block): in SAM 2 the batch dimension IS the object dimension, and every multi-object
    evaluation episode already runs through exactly this expansion. `.expand` returns
    views, so it is free, and the caller's dict is never mutated — the same encoded
    reference is reused across cycle legs.

    The image encoder is upstream of the expansion and sees batch 1 either way, so no
    gradient question arises there (and under LoRA it is frozen besides).
    """
    out = dict(backbone_out)
    out["backbone_fpn"] = [f.expand(n, -1, -1, -1)
                           for f in backbone_out["backbone_fpn"]]
    out["vision_pos_enc"] = [p.expand(n, -1, -1, -1)
                             for p in backbone_out["vision_pos_enc"]]
    return out


def soft_sentinel_heads(model, heads_fn, K: float, *args, **kwargs):
    """`_forward_sam_heads` with the −1024 sentinel replaced by a straight-through gate.

    Runs the original with ``model.pred_obj_scores`` switched off for the call (that flag
    guards exactly two things in `_forward_sam_heads`: the hard mask gate and the hard
    pointer mix — the decoder still predicts the score), then re-applies the gate as

        shift   = K·[score ≤ 0]  +  (K·σ(−score) − stopgrad(K·σ(−score)))
        mask   := mask − shift
        λ       = [score > 0]    +  (σ(score) − stopgrad(σ(score)))
        obj_ptr := λ·obj_ptr + (1 − λ)·no_obj_ptr

    FORWARD VALUE: a present channel is untouched and an absent one sits at ``mask − K``
    instead of −1024 — finite, the decoder's real logits, sigmoid ≈ 0 into the memory as
    before. BACKWARD: the score receives the soft surrogate's gradient (K·σ′(score), ≈ 4
    per unit at the fence's −0.8) from every loss that touches the masks or the pointer.
    A pure soft gate (mask − K·σ(−score) in the forward value too) was the first design
    and the smoke's gate-effect block refused it: a present-but-marginal channel (score
    +0.35) is shifted by ≈ 8 logits, which cost the BASE model −1.4 forward / −5.4 closing
    on three pairs — a deformation of what the model already gets right, confounding
    every read. Straight-through keeps the gradient and drops the deformation.
    Returns the same 7-tuple. See `DifferentiableSAM2Tracker` for why post hoc is exact.
    """
    import torch

    model.pred_obj_scores = False
    try:
        (low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks,
         obj_ptr, object_score_logits) = heads_fn(*args, **kwargs)
    finally:
        model.pred_obj_scores = True
    s = object_score_logits.float()                            # (B, 1)
    absent = (s <= 0).float()
    soft_shift = K * torch.sigmoid(-s)
    shift = (K * absent + (soft_shift - soft_shift.detach()))[:, :, None, None]   # (B,1,1,1)
    low_res_multimasks = low_res_multimasks - shift.to(low_res_multimasks.dtype)
    high_res_multimasks = high_res_multimasks - shift.to(high_res_multimasks.dtype)
    low_res_masks = low_res_masks - shift.to(low_res_masks.dtype)
    high_res_masks = high_res_masks - shift.to(high_res_masks.dtype)
    lam_soft = torch.sigmoid(s)
    lam = ((1.0 - absent) + (lam_soft - lam_soft.detach())).to(obj_ptr.dtype)
    if getattr(model, "fixed_no_obj_ptr", False):
        obj_ptr = lam * obj_ptr
    if getattr(model, "no_obj_ptr", None) is not None:
        obj_ptr = obj_ptr + (1 - lam) * model.no_obj_ptr
    return (low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks,
            obj_ptr, object_score_logits)


class DifferentiableSAM2Tracker:
    """Mask-conditioned `track_step` with gradients intact.

    Also captures the SAM head's **objectness score** (S17b): `track_step` unpacks
    `_forward_sam_heads`' ``object_score_logits`` into ``_`` and drops it, yet that score
    is what the −1024 no-object sentinel keys on (S17 anatomy) — the one quantity a loss
    could push to keep a channel alive. The instance's ``_forward_sam_heads`` and
    ``_use_mask_as_output`` are wrapped to stash their last score; `track_with_mask`
    attaches it to ``current_out["object_score_logits"]`` (``(C, 1)``, differentiable on
    propagated frames; the hard ±10 on init frames). Transparent otherwise, and on this
    instance only — the evaluator's predict path never reads it.

    **The soft sentinel (S17d, ``soft_sentinel=K``).** `sam2_base.py:354-361` replaces the
    whole mask by ``NO_OBJ_SCORE = -1024`` through a hard ``torch.where`` whenever that
    score is ≤ 0: the mask logits leave the graph, the score gets no gradient (a
    threshold), the reverse init reads ``sigmoid(-1024) == 0`` into the memory, and the
    closing loss on such a channel is the constant ``1024 × area`` (README S17 anatomy —
    62 % of the loss value, no gradient). With ``K`` set, the propagated-frame head runs
    UNGATED and the gate is re-applied straight-through on this instance
    (`soft_sentinel_heads`): the forward value is the hard gate with −K in place of
    −1024 (K = 20: SAM 2's own ±10 prompt-logit span), the backward pass is the soft
    gate's (``K·σ(−score)`` and SAM 2's own ``soft_no_obj_ptr`` mix). An absent
    channel's closing loss is then finite, on the decoder's real logits, with gradient
    into the mask decoder AND the score, and the memory the closing frame reads carries
    a gradient to the forward score — so the closing loss can teach the forward leg's
    objectness directly (the route `objness` substitutes with a hand-set prior). A
    per-batch constant commutes with the best-IoU selection and the bilinear upsample,
    so applying it to the head's outputs equals applying it where the hard gate sits.
    Init frames (`_use_mask_as_output`, incl. its inner `_forward_sam_heads` call for the
    pointer) keep SAM 2's own rule; the evaluator's path is untouched (the hard gate
    stays at test — a channel must carry its score across zero to show).

    **The memory straight-through (S18 ``prompt_mode="ste_memory"``, `memory_ste`).** A
    binary prompt on the reverse init reaches the memory as ``sigmoid(20·m − 10)`` =
    ``sigmoid(±10)`` — saturated, ``σ′ = 4.5e-5`` — so the forward leg's gradient through
    the memory route collapses (measured on the real model: forward/reverse per channel
    0.1 under the binary-valued prompt vs 19–449 under the continuous one; only the
    pointer route survives). In SAM 2 the sub-threshold channel and the memory-route
    gradient are the same quantity: the memory reads the prompt only inside the ±10
    band. `memory_ste(logits)` wraps the memory encoder's mask input on this instance,
    for the duration of the context, as ``value(binary path) + (20·σ(l) −
    stopgrad(20·σ(l)))``: the VALUE is exactly what the release's binary prompt produces
    (what crosses the palindrome is the argmax mask), the GRADIENT is `logit_reparam`'s
    memory-route gradient (``20·σ′(l)``). Wrapped once per model instance like the
    heads; inert unless a context is open, so the evaluator's path is untouched.
    """

    def __init__(self, model, soft_sentinel: float | None = None):
        self.model = model
        self.last_obj_score = None
        self.soft_sentinel = soft_sentinel
        self._in_mask_as_output = False
        self._memory_ste_logits = None
        if not getattr(model, "_occl_obj_capture", False):
            # wrapped ONCE per model instance; the wrapper reports to whichever tracker
            # currently owns the model (`_occl_tracker`), so rebuilding a tracker on the
            # same model never stacks wrappers
            def _wrap_heads(fn):
                def wrapped(*a, **k):
                    owner = getattr(model, "_occl_tracker", None)
                    K = getattr(owner, "soft_sentinel", None)
                    if (K is None or getattr(owner, "_in_mask_as_output", False)
                            or not getattr(model, "pred_obj_scores", False)):
                        out = fn(*a, **k)
                    else:
                        out = soft_sentinel_heads(model, fn, float(K), *a, **k)
                    if owner is not None:
                        owner.last_obj_score = out[-1]
                    return out
                return wrapped

            def _wrap_init(fn):
                def wrapped(*a, **k):
                    owner = getattr(model, "_occl_tracker", None)
                    if owner is not None:
                        owner._in_mask_as_output = True
                    try:
                        out = fn(*a, **k)
                    finally:
                        if owner is not None:
                            owner._in_mask_as_output = False
                    if owner is not None:
                        owner.last_obj_score = out[-1]
                    return out
                return wrapped

            def _wrap_mem(fn):
                def wrapped(pix_feat, masks, *a, **k):
                    import torch

                    owner = getattr(model, "_occl_tracker", None)
                    lg = getattr(owner, "_memory_ste_logits", None)
                    if lg is not None:
                        scale = float(getattr(model, "sigmoid_scale_for_mem_enc", 1.0))
                        soft = scale * torch.sigmoid(lg.to(masks.dtype))
                        if soft.shape != masks.shape:
                            raise ValueError(f"memory_ste: logits {tuple(soft.shape)} vs the "
                                             f"memory mask input {tuple(masks.shape)}")
                        masks = masks.detach() + (soft - soft.detach())
                    return fn(pix_feat, masks, *a, **k)
                return wrapped

            model._forward_sam_heads = _wrap_heads(model._forward_sam_heads)
            model._use_mask_as_output = _wrap_init(model._use_mask_as_output)
            mem = getattr(model, "memory_encoder", None)
            if mem is not None:
                mem.forward = _wrap_mem(mem.forward)
            model._occl_obj_capture = True
        model._occl_tracker = self

    def memory_ste(self, logits):
        """Context manager: the memory straight-through on this instance (see the class
        docstring) with ``logits`` — the forward leg's continuous ``(C, 1, H, W)`` logits
        — supplying the gradient; open it around the reverse-init call only."""
        from contextlib import contextmanager

        if getattr(self.model, "memory_encoder", None) is None:
            raise ValueError("memory_ste needs a model with a memory_encoder")

        @contextmanager
        def _ctx():
            self._memory_ste_logits = logits
            try:
                yield
            finally:
                self._memory_ste_logits = None

        return _ctx()

    def encode_image(self, img):
        """``img``: (B, 3, 1024, 1024), already normalised by `finetune.pairs`.

        Note SST's docstring (`oc_ccl.py:85-90`) claims "float32 in [0, 1]" and that "the
        image encoder handles normalization". It does not — SAM 2 normalises in its
        *loader* (`sam2/utils/misc.py:164-165`), which this path bypasses. That mismatch is
        defect **A1**; the fix lives in `pairs.to_model_tensor`.
        """
        return self.model.forward_image(img)

    def prepare_features(self, backbone_out):
        return self.model._prepare_backbone_features(backbone_out)

    def track_with_mask(self, backbone_out, mask, is_init_frame: bool = True,
                        output_dict=None, n_objects: int = 1):
        """One `track_step`. Returns ``(current_out, output_dict)``.

        On an init frame the mask is consumed by `_use_mask_as_output`; on a non-init frame
        the mask must be ``None`` and the prediction comes from the memory bank.

        ``n_objects`` runs C objects through the one pass, with the batch-1 backbone
        features expanded to C views (`_expand_backbone`) — the per-trait cycle. On an init
        frame ``mask`` must then be ``(C, 1, H, W)``, one channel per object, and every
        output (including the memory entries written into ``output_dict``) carries the
        object batch. ``n_objects=1`` is a structural no-op, so every merged-mask arm
        reproduces bit for bit.
        """
        if is_init_frame and mask is not None and mask.shape[0] != n_objects:
            raise ValueError(
                f"mask has {mask.shape[0]} object channel(s) but n_objects={n_objects}")
        if n_objects > 1:
            backbone_out = _expand_backbone(backbone_out, n_objects)
        _, vision_feats, vision_pos_embeds, feat_sizes = self.prepare_features(backbone_out)

        if output_dict is None:
            output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        current_out = self.model.track_step(
            frame_idx=0 if is_init_frame else 1,
            is_init_cond_frame=is_init_frame,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask if is_init_frame else None,
            output_dict=output_dict,
            num_frames=2,
            run_mem_encoder=True,
        )
        if self.last_obj_score is not None:
            current_out["object_score_logits"] = self.last_obj_score
            self.last_obj_score = None
        if is_init_frame:
            output_dict["cond_frame_outputs"][0] = current_out
        return current_out, output_dict
