"""SeC-4B adapter — S14's second port target (OpenIXCLab/SeC, ICLR 2026).

SeC = an InternVL2.5-4B concept module conditioning a stage-1-retrained SAM 2.1-L
"grounding encoder" (num_maskmem=22). Two facts, both verified in its source, shape
this adapter:

1. **The native concept gate cannot SEE the transformation this benchmark tests.**
   `is_scene_change_hsv` (modeling_sec.py) is an HSV-histogram Bhattacharyya test —
   spatially invariant, so a 180°-flipped specimen has a bit-identical histogram:
   whether the gate fires can never depend on the flip. It DOES fire on specimen
   identity: S14's recomputation over the real protocol draws measured a would-fire
   rate of 0.585 (butterfly) / 0.869 (beetle) — support and query are different
   specimens, and that alone clears 0.35 on most episodes, falsifying the
   pre-registered ≈ 0 expectation (scripts/fire_stats_s14.py). The explicit
   ``concept_policy`` is what made the measurement controlled either way:
   ``"forced"`` (one LVLM forward per episode × prompted class, batch-1 as SeC's
   own injection hardwires) and ``"off"`` bracket the native mixture, and S14
   measured forced ≈ 6 points BELOW off on both datasets — so the off cells are
   SeC's most charitable reading here. EVERY ``off`` episode still records the
   would-fire statistic (a bug-compatible replica of their detector,
   PIL-RGB-into-COLOR_BGR2HSV channel swap included).
2. **On a 2-frame episode the grounding encoder degenerates to the SAM 2 graph** —
   the maskmem-22 machinery has nothing to select from — and it keeps
   `out_scale, out_bias = 20.0, -10.0`, so the OC-CCL fine-tune stack runs on
   ``adapter.model`` (= the grounding encoder) with the sam2 scope strings VERBATIM.

The published loader is broken for the published checkpoint (upstream issue #9: an
``==`` where ``=`` was meant leaves the legacy config string in place, and the HF
repo hosts no remote code), so ``build`` patches
``config.grounding_encoder_config = "sec_sam2.1_hiera_l.yaml"`` and imports the
GitHub checkout's ``inference`` package (`paths.bootstrap_sec_syspath`). Pins:
``apply_postprocessing=False`` (no dynamic-multimask override, no fill-hole CUDA
ext, raw per-object logits), ``use_flash_attn=False`` (SDPA/eager — same math, no
20-minute Colab build), strict load via ``output_loading_info`` (any missing or
unexpected key raises — the `sst_sam2.py` law).

Episodes run through a MANUAL inference state (`_init_state_from_tensor`) — the
vendored predictor's `init_state` only accepts a JPEG directory, and temp files
would be a lossy re-encode. Frames go through `sst_sam2.to_model_tensor` (SeC's
tracker is ImageNet-normalised at 1024, same constants — one pixel authority, the
A1 discipline; SeC's own PIL-BICUBIC loader resize is the one deliberate deviation,
recorded as ``frame_interp: "cv2_linear"``).
"""

from __future__ import annotations

from pathlib import Path

from .. import paths
from .base import pick_dtype
from .sam3 import cuda_identity_on_cpu
from .sst_sam2 import to_model_tensor

SCENE_CHANGE_THRESHOLD = 0.35
CONCEPT_PROMPT = ("<image>Please segment the object in the last frame based on "
                  "the object labeled in the first several images.")


def resolve_grounding_config(value: str) -> str:
    """Upstream issue #9: the compat shim wrote ``==`` where ``=`` was meant, so the
    published checkpoint's legacy config string is never rewritten and construction
    dies in an UnboundLocalError. This is the two lines the shim meant to run."""
    if value == "longsam2.1/longsam2.1_hiera_l.yaml":
        return "sec_sam2.1_hiera_l.yaml"
    return value


def bhattacharyya_hsv(img1, img2) -> float:
    """Bug-compatible replica of SeC's scene-change statistic (RGB arrays in,
    `COLOR_BGR2HSV` applied as upstream does to PIL-RGB data)."""
    import cv2
    import numpy as np

    a = cv2.resize(np.asarray(img1), (1024, 1024))
    b = cv2.resize(np.asarray(img2), (1024, 1024))
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2HSV)], [0, 1], None,
                      [60, 80], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2HSV)], [0, 1], None,
                      [60, 80], [0, 180, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


class SeCAdapter:
    """SeC-4B behind the project's adapter contract."""

    def __init__(self, checkpoint: str | None = None,
                 concept_policy: str = "off"):
        if concept_policy not in ("off", "forced"):
            raise ValueError(f"concept_policy must be off|forced, "
                             f"got {concept_policy!r}")
        self.name = f"sec_{concept_policy}"
        self.native_size = 1024
        self.checkpoint = checkpoint          # None -> paths.sec_checkpoint_dir()
        self.concept_policy = concept_policy
        self.sec = None                       # the full SeCModel
        self.model = None                     # = sec.grounding_encoder (SAM 2.1)
        self.dtype = None
        self.device = "cpu"
        self._store: dict = {}
        self._pos_enc = None      # ONE shared copy — content-independent, asserted
        self.cache_stats = {"encoded": 0, "hits": 0}
        self._load_report = {"load_missing_keys": None, "load_unexpected_keys": None}
        self._token_attn_diff = None
        self.concept_fired = 0
        self.concept_expected = 0
        self._fire_distances: list[float] = []
        self._lvlm_dropped = False

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, device: str = "cuda", dtype=None, strict: bool = True):
        import torch

        paths.bootstrap_sec_syspath()
        try:
            from hydra.core.global_hydra import GlobalHydra

            GlobalHydra.instance().clear()
        except Exception:
            pass
        from transformers import AutoTokenizer

        from inference.configuration_sec import SeCConfig
        from inference.modeling_sec import SeCModel

        # The vendored package initializes Hydra at IMPORT time; on a SECOND build in
        # the same process the imports above are cached no-ops, so the clear() above
        # leaves Hydra empty and `compose` dies inside SeCModel.__init__ (found by
        # the round-trip probe in run_finetune — build #2 of every fine-tune run).
        # Re-initialize explicitly, exactly as inference/sam2/__init__.py does.
        from hydra.core.global_hydra import GlobalHydra

        if not GlobalHydra.instance().is_initialized():
            from hydra import initialize_config_module

            initialize_config_module("inference.sam2.configs", version_base="1.2")

        ckpt = self.checkpoint or str(paths.sec_checkpoint_dir())
        config = SeCConfig.from_pretrained(ckpt)
        config.grounding_encoder_config = resolve_grounding_config(
            config.grounding_encoder_config)
        config.apply_postprocessing = False
        model, info = SeCModel.from_pretrained(
            ckpt, config=config, torch_dtype=torch.bfloat16,
            use_flash_attn=False, output_loading_info=True)
        missing = list(info.get("missing_keys", []))
        unexpected = list(info.get("unexpected_keys", []))
        self._load_report = {"load_missing_keys": len(missing),
                             "load_unexpected_keys": len(unexpected)}
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"SeC loaded loosely: {len(missing)} missing (e.g. {missing[:3]}), "
                f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        self.sec = model
        if self.concept_policy == "off":
            # An off-policy instance can never invoke the concept module, so the
            # ~7.6 GB frozen LVLM is dropped BEFORE the device move and never
            # touches the GPU. Order is load-bearing: dropped AFTER `.to(device)`,
            # the freed VRAM stays reserved by this process's caching allocator and
            # is invisible-unusable to sibling eval workers — measured 8.42 GB
            # resident per off worker on the first butterfly fan-out, four of which
            # exhausted the 39.5 GB card and killed the fifth build (OOM) plus a
            # neighbour's SDPA (unknown-error under zero free memory). `forced`
            # builds keep the LVLM; config_report records "dropped" so a record
            # cannot misstate it. preparing_for_generation is safe after the drop:
            # it sets tokenizer/config/scalar attrs only, never the popped modules.
            self.drop_lvlm()
        model.eval().to(device)
        model.preparing_for_generation(tokenizer=tokenizer,
                                       torch_dtype=torch.bfloat16)

        self.model = model.grounding_encoder
        self.device = device
        self.dtype = dtype or pick_dtype(device)
        if self.dtype == torch.float32:
            # preparing_for_generation bf16-casts EVERYTHING; the tracker path runs
            # in our precision (the LVLM stays bf16 — its outputs are cast at
            # injection).
            self.model.float()
        with torch.no_grad():
            # token_attn is a deepcopy of memory_attention at CONSTRUCTION; whether
            # the checkpoint kept them identical is a fact worth the record.
            d = sum((a - b).abs().sum().item() for a, b in zip(
                self.model.token_attn.parameters(),
                self.model.memory_attention.parameters()))
        self._token_attn_diff = float(d)
        self.checkpoint = ckpt
        return self

    def drop_lvlm(self):
        """Free the ~7.6 GB frozen LVLM. The concept module cannot be used
        afterwards — `forced` predicts will raise. Idempotent; off-policy builds
        call it pre-device-move so the LVLM never reaches the GPU."""
        import gc

        import torch

        for attr in ("vision_model", "language_model", "mlp1", "text_hidden_fcs"):
            if hasattr(self.sec, attr):
                self.sec._modules.pop(attr, None)
                setattr(self.sec, attr, None)
        self._lvlm_dropped = True
        gc.collect()
        if torch.cuda.is_available():
            # a post-move caller frees GPU tensors here — RETURN them to the
            # driver; reserved-but-unallocated memory starves sibling workers
            torch.cuda.empty_cache()

    # ── embedding cache (grounding-encoder features, SST pattern) ────────────

    def _tracker_dtype(self):
        import torch  # noqa: F401

        return next(self.model.parameters()).dtype

    def _encode(self, img_tensor):
        import torch

        # EXACTLY the numeric context of the vendored miss path, or cached
        # predictions stop being bit-identical to uncached (the S3 predicate,
        # re-asserted by commission_s14): state images are tracker-dtype (bf16 on
        # GPU), `_get_image_feature` upcasts them `.float()`, and the encode runs
        # inside predict's autocast. Encoding a bf16 input WITHOUT autocast is
        # numerically identical on CPU (autocast disabled, all fp32) but diverged
        # on A100 — which is how the first s14sec commissioning caught it.
        cuda = str(self.device).startswith("cuda")
        with torch.inference_mode(), torch.autocast(
                device_type="cuda" if cuda else "cpu", dtype=self.dtype,
                enabled=cuda):
            out = self.model.forward_image(
                img_tensor.to(self.device, self._tracker_dtype()).float())
        self.cache_stats["encoded"] += 1
        # ONE pos-enc copy for the whole cache (the sst_sam2/sam3 discipline):
        # PositionEmbeddingSine is a function of feature-map shape, not content —
        # asserted, not assumed. Caching it per image is what put forced workers at
        # ~15 GB (105 images x ~44 MB of identical tensors ≈ 4.6 GB each) and
        # OOM'd the 3-wide forced fan-out on the second butterfly session.
        pos = [p.detach() for p in out["vision_pos_enc"]]
        if self._pos_enc is None:
            self._pos_enc = pos
        else:
            for a, b in zip(self._pos_enc, pos):
                assert torch.equal(a, b), \
                    "sec vision_pos_enc varies across images — cannot share one copy"
        return {"backbone_fpn": [f.detach() for f in out["backbone_fpn"]]}

    def precompute(self, pairs) -> dict:
        for key, img in pairs:
            if key in self._store:
                self.cache_stats["hits"] += 1
            else:
                self._store[key] = self._encode(to_model_tensor(img, 1024))
        total = sum(f.numel() * f.element_size()
                    for feats in self._store.values()
                    for f in feats["backbone_fpn"]) / 1e9
        return {"n_cached": len(self._store), "total_gb": round(total, 3)}

    def clear_embeddings(self):
        self._store.clear()
        self._pos_enc = None

    def cache_report(self) -> dict:
        total = sum(f.numel() * f.element_size()
                    for feats in self._store.values()
                    for f in feats["backbone_fpn"]) / 1e9
        return {"n_cached": len(self._store), "total_gb": round(total, 3),
                **self.cache_stats}

    # ── inference ────────────────────────────────────────────────────────────

    def _init_state_from_tensor(self, frames):
        """The vendored `init_state`, minus the JPEG-directory loader: same dict,
        `images` from our own tensors, `video_paths=None` (never consulted — the
        gated loop that reads it is SeCModel's, which this adapter replaces)."""
        import torch
        from collections import OrderedDict

        images = torch.cat(frames).to(self.device, self._tracker_dtype())
        state = {
            "images": images, "num_frames": len(frames),
            "offload_video_to_cpu": False, "offload_state_to_cpu": False,
            "video_height": 1024, "video_width": 1024,
            "device": torch.device(self.device),
            "storage_device": torch.device(self.device),
            "point_inputs_per_obj": {}, "mask_inputs_per_obj": {},
            "cached_features": {}, "constants": {},
            "obj_id_to_idx": OrderedDict(), "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {},
                            "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(),
                                        "non_cond_frame_outputs": set()},
            "tracking_has_started": False, "frames_already_tracked": {},
            "video_paths": None,
        }
        return state

    def _get_feature_into_state(self, state, frame_idx, key=None):
        """Inject stored features for `frame_idx` (the SST cache pattern, applied
        state-side because SeC's predictor keeps the 1-entry cache convention)."""
        if key is not None and key in self._store:
            self.cache_stats["hits"] += 1
            feats = self._store[key]
            state["cached_features"][frame_idx] = (
                state["images"][frame_idx].unsqueeze(0),
                {"backbone_fpn": list(feats["backbone_fpn"]),
                 "vision_pos_enc": list(self._pos_enc)})

    def predict(self, ep):
        import cv2
        import numpy as np
        import torch

        if self.concept_policy == "forced" and self._lvlm_dropped:
            raise RuntimeError("LVLM was dropped — forced concept predicts are gone")
        query_idx = ep.shot
        frames = [to_model_tensor(img, 1024) for img in ep.support_images]
        frames.append(to_model_tensor(ep.query_image, 1024))
        keys = list(ep.support_ids) + [f"query:{ep.query_id}"]

        # the would-fire statistic, recorded on EVERY episode (both policies):
        # support (the "last frame" of SeC's window) vs query
        self._fire_distances.append(
            bhattacharyya_hsv(ep.support_images[-1], ep.query_image))

        lms = [cv2.resize(lm.astype(np.int32), (1024, 1024),
                          interpolation=cv2.INTER_NEAREST)
               for lm in ep.support_labelmaps]

        cuda = str(self.device).startswith("cuda")
        autocast = torch.autocast(device_type="cuda" if cuda else "cpu",
                                  dtype=self.dtype, enabled=cuda)
        size = 1024
        logits = np.full((len(ep.class_ids), size, size), -1e4, np.float32)

        if self.concept_policy == "off":
            # the shim covers the vendored 2.1 predictor's hardcoded `.cuda()` on
            # feature-cache misses — identity on CUDA-less boxes, no-op elsewhere
            with torch.inference_mode(), cuda_identity_on_cpu(), autocast:
                state = self._init_state_from_tensor(frames)
                for idx, key in enumerate(keys):
                    self._get_feature_into_state(state, idx, key)
                for j, lm_r in enumerate(lms):
                    for c in ep.class_ids:
                        m = (lm_r == c)
                        if not m.any():
                            continue
                        self.model.add_new_mask(
                            state, frame_idx=j, obj_id=int(c),
                            mask=torch.from_numpy(m).to(self.device))
                got = obj_ids = None
                for fidx, ids, video_res in self.model.propagate_in_video(state):
                    if fidx == query_idx:
                        got, obj_ids = video_res, ids
                        break
                if got is None:
                    raise RuntimeError("sec propagate never reached the query frame")
                arr = got.float().cpu().numpy().reshape(len(obj_ids), size, size)
                index = {int(o): i for i, o in enumerate(obj_ids)}
                for i, c in enumerate(ep.class_ids):
                    if int(c) in index:
                        logits[i] = arr[index[int(c)]]
            return logits

        # forced: SeC's language injection is hardwired batch-1, so one state per
        # prompted class — the same per-object shape SeC's own VOS harness uses.
        for i, c in enumerate(ep.class_ids):
            m = (lms[-1] == c)  # concept window uses the LAST support, as upstream
            if not any((lm == c).any() for lm in lms):
                continue
            emb = self._concept_embedding(ep.support_images[-1], m, ep.query_image)
            with torch.inference_mode(), cuda_identity_on_cpu(), autocast:
                state = self._init_state_from_tensor(frames)
                for idx, key in enumerate(keys):
                    self._get_feature_into_state(state, idx, key)
                for j, lm_r in enumerate(lms):
                    mj = (lm_r == c)
                    if mj.any():
                        self.model.add_new_mask(
                            state, frame_idx=j, obj_id=int(c),
                            mask=torch.from_numpy(mj).to(self.device))
                self.model.propagate_in_video_preflight(state)
                current_out, pred_masks = self.model._run_single_frame_inference(
                    inference_state=state, output_dict=state["output_dict"],
                    frame_idx=query_idx, batch_size=1, is_init_cond_frame=False,
                    point_inputs=None, mask_inputs=None, reverse=False,
                    run_mem_encoder=True, start_frame_idx=0,
                    language_embd=emb.to(self._tracker_dtype()).unsqueeze(0))
                _, video_res = self.model._get_orig_video_res_output(
                    state, pred_masks)
            self.concept_fired += 1
            logits[i] = video_res.float().cpu().numpy().reshape(size, size)
        self.concept_expected += sum(
            1 for c in ep.class_ids if any((lm == c).any() for lm in lms))
        return logits

    def _concept_embedding(self, support_img, support_mask, query_img):
        """`predict_forward`, reimplemented device-agnostically (upstream hardcodes
        `.cuda()`) on the same members `preparing_for_generation` installs."""
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        sec = self.sec
        frame = np.array(support_img).copy()
        contours, _ = cv2.findContours(support_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        video = [Image.fromarray(frame), Image.fromarray(np.asarray(query_img))]

        pixel_values = torch.stack(
            [sec.transformer(f) for f in video], dim=0).to(sec.torch_dtype)
        tok = (f"{sec.IMG_START_TOKEN}"
               f"{sec.IMG_CONTEXT_TOKEN * sec.patch_token}"
               f"{sec.IMG_END_TOKEN}\n") * len(video)
        text = CONCEPT_PROMPT + "It is [SEG]."
        text = text.replace("<image>", tok.strip())
        input_text = sec.template["INSTRUCTION"].format(
            input=text, round=1, bot_name=sec.bot_name)
        ids = torch.tensor(sec.tokenizer.encode(input_text),
                           device=self.device).unsqueeze(0)
        data = {"input_ids": ids,
                "attention_mask": torch.ones_like(ids, dtype=torch.bool),
                "pixel_values": pixel_values.unsqueeze(0).to(self.device),
                "position_ids": None, "labels": None}
        with torch.no_grad():
            out = sec.forward(data)
            hidden = out.hidden_states[-1][ids == sec.seg_token_idx]
            return sec.text_hidden_fcs(hidden)[0]

    # ── record ───────────────────────────────────────────────────────────────

    def config_report(self) -> dict:
        import numpy as np

        d = np.array(self._fire_distances) if self._fire_distances else None
        return {
            "family": "sec",
            "normalization": "imagenet",
            "image_size": 1024,
            "num_maskmem": int(getattr(self.model, "num_maskmem", 0) or 0)
            if self.model is not None else None,
            "checkpoint": str(self.checkpoint),
            "dtype": str(self.dtype),
            "frame_interp": "cv2_linear",
            "concept_policy": self.concept_policy,
            "concept_fired": self.concept_fired,
            "concept_expected": self.concept_expected,
            "lvlm": "internvl2.5-4b" if not self._lvlm_dropped else "dropped",
            "use_flash_attn": False,
            "apply_postprocessing": False,
            "token_attn_eq_memory_attention": (
                self._token_attn_diff == 0.0
                if self._token_attn_diff is not None else None),
            "native_fire_rate": (float((d > SCENE_CHANGE_THRESHOLD).mean())
                                 if d is not None else None),
            "bhattacharyya_mean": (float(d.mean()) if d is not None else None),
            "bhattacharyya_max": (float(d.max()) if d is not None else None),
            **self._load_report,
        }

    def close(self):
        self.clear_embeddings()
        self.sec = None
        self.model = None
