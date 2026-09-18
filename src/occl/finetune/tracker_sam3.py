"""Differentiable mask-conditioned tracking on SAM 3 — the `tracker.py` diff.

`Sam3TrackerBase` keeps `forward_image` / `_prepare_backbone_features` /
`track_step` with SAM 2's shapes, and `_use_mask_as_output` keeps
`out_scale, out_bias = 20.0, -10.0` verbatim (pinned by a source-grep test over both
checkouts), so `cycle.py` runs UNCHANGED on this tracker — `logit_reparam` stays the
exact inverse it was, and the dead-channel gate contract (`fixed_no_obj_ptr`
behaviour, sigmoid memory route for mask prompts) transfers with it.

The diffs, each verified against `sam3/model/sam3_tracker_base.py` this stage:

* `track_step` gains a positional ``image`` argument. It reaches only
  `_encode_new_memory`, whose `SimpleMaskEncoder` branch never touches it — so this
  tracker passes ``image=None`` and ASSERTS the maskmem backbone is a
  `SimpleMaskEncoder` at construction, making the None legal by checked invariant
  rather than by hope (the multiplex maskmem backbone, which does consume the image,
  belongs to sam3.1's classes and never loads here).
* `_expand_backbone` is imported from `.tracker` unchanged — sam3's own
  `_get_image_feature` uses the identical keys and `.expand` semantics.
* Geometry: images at 1008, masks/predictions at 1152. `cycle.py`'s resize guard is
  inert because `CyclePairs(mask_size=1152)` builds y0 on the prediction grid.
"""

from __future__ import annotations

from .tracker import _expand_backbone


class DifferentiableSAM3Tracker:
    """Mask-conditioned `track_step` with gradients intact, SAM 3 edition."""

    def __init__(self, model):
        from sam3.model.memory import SimpleMaskEncoder

        assert isinstance(model.maskmem_backbone, SimpleMaskEncoder), (
            "sam3 tracker port assumes the plain SimpleMaskEncoder maskmem backbone "
            "(image=None in track_step is only legal on that branch); got "
            f"{type(model.maskmem_backbone).__name__}")
        self.model = model

    def encode_image(self, img):
        """``img``: (B, 3, 1008, 1008), half-normalised by `finetune.pairs`
        (`normalization="half"` -> `adapters.sam3.to_model_tensor_sam3`).

        Runs under ``torch.no_grad()`` + a bf16 autocast — REQUIRED, not an
        optimisation: sam3's ViT MLP calls a fused addmm (`sam3/perflib/fused.py`)
        that raises when grad is enabled AND internally computes in bf16 (sam3's
        released inference behaviour — the predictor normally lives inside a global
        bf16 autocast). The backbone is outside every LoRA scope, so its features
        are constants of the training graph either way; they are cast back to fp32
        on the way out so the track legs run in whatever precision the trainer set.
        """
        import torch

        dev = "cuda" if img.is_cuda else "cpu"
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16):
            out = self.model.forward_image(img)
        out["backbone_fpn"] = [f.float() for f in out["backbone_fpn"]]
        out["vision_pos_enc"] = [p.float() for p in out["vision_pos_enc"]]
        return out

    def prepare_features(self, backbone_out):
        return self.model._prepare_backbone_features(backbone_out)

    def track_with_mask(self, backbone_out, mask, is_init_frame: bool = True,
                        output_dict=None, n_objects: int = 1):
        """One `track_step`. Returns ``(current_out, output_dict)``.

        Same contract as `DifferentiableSAM2Tracker.track_with_mask`; masks are
        ``(C, 1, 1152, 1152)`` on init frames.
        """
        if is_init_frame and mask is not None and mask.shape[0] != n_objects:
            raise ValueError(
                f"mask has {mask.shape[0]} object channel(s) but n_objects={n_objects}")
        if n_objects > 1:
            backbone_out = _expand_backbone(backbone_out, n_objects)
        _, vision_feats, vision_pos_embeds, feat_sizes = \
            self.prepare_features(backbone_out)

        if output_dict is None:
            output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        from ..adapters.sam3 import cuda_identity_on_cpu

        with cuda_identity_on_cpu():
            current_out = self.model.track_step(
                frame_idx=0 if is_init_frame else 1,
                is_init_cond_frame=is_init_frame,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                image=None,      # unused on the SimpleMaskEncoder branch — asserted
                point_inputs=None,
                mask_inputs=mask if is_init_frame else None,
                output_dict=output_dict,
                num_frames=2,
                run_mem_encoder=True,
            )
        if is_init_frame:
            output_dict["cond_frame_outputs"][0] = current_out
            assert len(output_dict["cond_frame_outputs"]) == 1, \
                "init leg produced more than one cond entry — sam3's self-storing leaked"
        return current_out, output_dict
