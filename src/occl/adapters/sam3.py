"""SAM 3 tracker adapter — S14's first port target.

`Sam3TrackerPredictor` (facebookresearch/sam3) is a fork of SAM 2's video predictor:
`init_state` / `add_new_mask` / `propagate_in_video` / `_get_image_feature` survive
with the same shapes, and `_use_mask_as_output` keeps `out_scale, out_bias =
20.0, -10.0` verbatim — which is what lets the whole OC-CCL stack port. This adapter
is the `sst_sam2.py` discipline applied to the new family. Four deliberate departures
from SAM 3's own demo path, each recorded in `config_report`:

1. **Feature injection, not a monkeypatch.** `init_state(cached_features=...)` is
   native here — but its miss path is a 1-entry LRU that EVICTS every prefilled frame
   (`_get_image_feature` overwrites the whole dict) and hard-codes `.cuda()`. So
   `predict` prefills features for EVERY frame and asserts it did; a miss during a
   scored run is a bug, not a slow path.
2. **Masks at 1152, no resample.** `input_mask_size = 1152` (read from the model);
   labelmaps are NEAREST-resized to 1152 once, so `add_new_mask`'s antialiased
   bilinear branch is skipped entirely — the same no-resample discipline as
   `sst_sam2.py`, at the new geometry. `video_height/width` are pinned to 1152 too,
   so `video_res_masks` come back on the same grid.
3. **The constructor's process-global bf16 autocast is exited at build.** SAM 3
   `__enter__`s a `torch.autocast(cuda, bf16)` in `__init__` and never exits;
   left in place it silently converts every later fp32 computation in the process.
4. **`obj_scores` are recorded, never used**: the mask logits already carry the
   no-object gating, and a second threshold would change the argmax rule.

Normalization is **(x/255 − 0.5) / 0.5** — NOT ImageNet; resize stays cv2
INTER_LINEAR (SAM 3's video loader uses INTER_CUBIC — recorded as `resize_interp`,
kept LINEAR so the pixel pipeline matches every recorded number in this project).

Weights: `sam3.pt` only (the 3.1 multiplex checkpoint's keys belong to different
classes). `surgery_map` routes `tracker.*` → the predictor and the shared visual
trunk `detector.backbone.vision_backbone.*` → `backbone.vision_backbone.*`
(model-driven over the `detector.backbone.` prefix — the class attr is
`vision_backbone`, NOT the `visual=` constructor parameter the first surgery
guessed), then raises on ANY missing or unexpected key — the `sst_sam2.py`
strict-load law. `checkpoint="random"` skips
loading for wiring smokes (recorded; never legal in a scored run — the gate pins
load_missing_keys == 0, which a random build reports as None).
"""

from __future__ import annotations

from pathlib import Path

from .. import paths
from .base import pick_dtype

NATIVE_SIZE = 1008


from contextlib import contextmanager  # noqa: E402


@contextmanager
def cuda_identity_on_cpu():
    """sam3 hardcodes ``.cuda()`` in its memory-retrieval path
    (`sam3_tracker_base.py:656,663`) and ``.pin_memory()`` in its temporal
    positional encoding (`:167`) — on a CUDA-less box those calls crash rather than
    degrade. ONLY when torch sees no CUDA at all, both are redirected to identity
    for the duration of the wrapped call; on any GPU machine this context is a pure
    pass-through. On a CPU-only machine there is nothing either call could have
    validly done, so the redirect cannot change a result — it converts a crash into
    the computation sam3 would have done on GPU tensors."""
    import torch

    if torch.cuda.is_available():
        yield
        return
    orig_cuda = torch.Tensor.cuda
    orig_pin = torch.Tensor.pin_memory
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.cuda = orig_cuda
        torch.Tensor.pin_memory = orig_pin


def surgery_map(sd: dict, model_keys) -> dict:
    """sam3.pt -> standalone-tracker state dict. Pure, so the mapping is testable
    without the gated checkpoint.

    The checkpoint is the FULL video model saved verbatim (model_builder loads it
    with no remap), so tracker internals live under ``tracker.*`` and the shared
    visual trunk under ``detector.backbone.vision_backbone.*`` — the class attr is
    ``vision_backbone`` (vl_combiner.py), NOT the ``visual=`` constructor parameter
    name, which is what the first surgery guessed: 464 missing / 0 unexpected on
    first real contact (the prefix matched nothing, everything it did forward was
    right). The backbone half is MODEL-DRIVEN — forward exactly the keys the
    tracker model declares — so the detector's text tower never lands as
    "unexpected", and it stays restricted to the ``detector.backbone.`` prefix so
    a detector-side module whose name coincides with a tracker-root module (the
    detector has its own ``transformer``) can never be grafted in.
    """
    model_keys = set(model_keys)
    mapped = {}
    for k, v in sd.items():
        if k.startswith("tracker."):
            mapped[k[len("tracker."):]] = v
        elif k.startswith("detector.backbone."):
            cand = k[len("detector."):]
            if cand in model_keys:
                mapped[cand] = v
    return mapped


def to_model_tensor_sam3(img, size: int = NATIVE_SIZE, device=None):
    """(H, W, 3) uint8 RGB -> (1, 3, size, size) fp32, half-normalised.

    The single authority for SAM 3's input distribution — `finetune.pairs` imports
    this for the `"half"` normalization branch (the A1 discipline: the inference
    path decides, training follows).
    """
    import cv2
    import torch

    arr = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
    t = t.sub_(0.5).div_(0.5).unsqueeze(0)
    return t.to(device) if device is not None else t


class Sam3TrackerAdapter:
    """SAM 3's plain tracker behind the project's adapter contract."""

    def __init__(self, checkpoint: str | None = None):
        self.name = "sam3"
        self.native_size = NATIVE_SIZE
        self.checkpoint = checkpoint          # None -> paths.sam3_checkpoint()
        self.model = None
        self.dtype = None
        self.device = "cpu"
        self.mask_input_size = None           # read from the model at build
        self._store: dict = {}                # key -> list of fpn tensors
        self._pos_enc = None                  # shared: content-independent
        self.cache_stats = {"encoded": 0, "hits": 0, "misses_in_predict": 0}
        self._load_report = {"load_missing_keys": None, "load_unexpected_keys": None}
        self._autocast_exited = False
        self._obj_score_min = None
        self._obj_score_sum = 0.0
        self._obj_score_n = 0

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, device: str = "cuda", dtype=None, strict: bool = True):
        import torch

        paths.bootstrap_sam3_syspath()
        try:  # families cannot co-build without re-clearing Hydra's global state
            from hydra.core.global_hydra import GlobalHydra

            GlobalHydra.instance().clear()
        except Exception:
            pass
        if not torch.cuda.is_available():
            # sam3's PositionEmbeddingSine precomputes its cache on device="cuda"
            # unconditionally — a torch.compile warm-up, not semantics. On a CPU-only
            # box we skip the precompute; the cache fills lazily on first forward at
            # the right device, bit-identically. GPU behaviour is untouched.
            from sam3.model import position_encoding as _pe

            if not getattr(_pe.PositionEmbeddingSine, "_occl_cpu_patch", False):
                _orig_init = _pe.PositionEmbeddingSine.__init__

                def _cpu_init(s, *a, **kw):
                    kw["precompute_resolution"] = None
                    _orig_init(s, *a, **kw)

                _pe.PositionEmbeddingSine.__init__ = _cpu_init
                _pe.PositionEmbeddingSine._occl_cpu_patch = True
        from sam3.model_builder import build_tracker

        model = build_tracker(apply_temporal_disambiguation=True, with_backbone=True)

        # Departure 4 (S15): attend to EVERY conditioning frame. build_tracker
        # hardcodes max_cond_frames_in_attn=4, so at 5-shot — supports at frames
        # 0-4, the query at 5 — the tracker would silently attend to the 4 closest
        # supports and drop frame 0: the trap the S6 section predicted for "SAM 3"
        # before this port existed, confirmed by S14's commissioned model (cap 4,
        # recorded). SST's SAM 2 runs with -1 (all); match it. At shot <= 4 the
        # selection is identical by construction (select_closest_cond_frames returns
        # everything when len(cond) <= cap), so every 1-shot number is unchanged —
        # though this edit does move the resume fingerprint for sam3 runs (the
        # adapter source is hashed), orphaning the banked 1-shot per-run JSON; both
        # 1-shot bundles are merged and filed, so nothing owed is lost. A config
        # attribute, not weights: config_report records it, the S15 gate pins it.
        model.max_cond_frames_in_attn = -1

        # Departure 3: exit the constructor's process-global bf16 autocast.
        ctx = getattr(model, "bf16_context", None)
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
                self._autocast_exited = True
            except Exception:
                pass
        assert not torch.is_autocast_enabled(), \
            "sam3's process-global autocast is still active after build"

        ckpt = self.checkpoint or str(paths.sam3_checkpoint())
        if ckpt != "random":
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                sd = sd["model"]
            mapped = surgery_map(sd, model.state_dict().keys())
            info = model.load_state_dict(mapped, strict=False)
            missing, unexpected = list(info.missing_keys), list(info.unexpected_keys)
            self._load_report = {"load_missing_keys": len(missing),
                                 "load_unexpected_keys": len(unexpected)}
            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"sam3 weight surgery loaded loosely: {len(missing)} missing "
                    f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
                    f"(e.g. {unexpected[:3]}) — a loose load trains/scores noise")
        self.checkpoint = ckpt

        model.to(device)
        model.eval()
        self.model = model
        self.device = device
        self.dtype = dtype or pick_dtype(device)
        self.mask_input_size = int(model.input_mask_size)
        return self

    # ── embedding cache ──────────────────────────────────────────────────────

    def _encode(self, img_tensor):
        import torch

        cuda = str(self.device).startswith("cuda")
        # bf16 autocast unconditionally: sam3's fused ViT MLP computes in bf16 by
        # construction (its predictor lives inside a global bf16 autocast we exited);
        # features are stored fp32 so downstream legs run in OUR chosen precision.
        with torch.inference_mode(), torch.autocast(
                device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16):
            out = self.model.forward_image(img_tensor.to(self.device))
        fpn = [f.detach().float() for f in out["backbone_fpn"]]
        pos = [p.detach().float() for p in out["vision_pos_enc"]]
        if self._pos_enc is None:
            self._pos_enc = pos
        else:
            # Content-independent by construction (PositionEmbeddingSine) — asserted
            # rather than assumed, since the trunk is new to this project.
            for a, b in zip(self._pos_enc, pos):
                assert torch.equal(a, b), \
                    "sam3 vision_pos_enc varies across images — cannot share one copy"
        self.cache_stats["encoded"] += 1
        return fpn

    def _features_for(self, key, img_tensor):
        if key in self._store:
            self.cache_stats["hits"] += 1
        else:
            self._store[key] = self._encode(img_tensor)
        return {"backbone_fpn": list(self._store[key]),
                "vision_pos_enc": list(self._pos_enc)}

    def precompute(self, pairs) -> dict:
        for key, img in pairs:
            if key not in self._store:
                self._store[key] = self._encode(to_model_tensor_sam3(img))
            else:
                self.cache_stats["hits"] += 1
        total = sum(f.numel() * f.element_size()
                    for feats in self._store.values() for f in feats) / 1e9
        return {"n_cached": len(self._store), "total_gb": round(total, 3)}

    def clear_embeddings(self):
        self._store.clear()

    def cache_report(self) -> dict:
        total = sum(f.numel() * f.element_size()
                    for feats in self._store.values() for f in feats) / 1e9
        return {"n_cached": len(self._store), "total_gb": round(total, 3),
                **self.cache_stats}

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, ep):
        import cv2
        import numpy as np
        import torch

        n_frames = ep.shot + 1
        query_idx = ep.shot
        size = self.mask_input_size

        # Departure 1: full prefill — every frame's features present BEFORE
        # init_state, images as real tensors (the miss path evicts and .cuda()s).
        cached = {}
        keys = list(ep.support_ids) + [f"query:{ep.query_id}"]
        images = list(ep.support_images) + [ep.query_image]
        for idx, (key, img) in enumerate(zip(keys, images)):
            t = to_model_tensor_sam3(img, device=self.device)
            feats = self._features_for(key, t)
            cached[idx] = (t, feats)
        assert all(i in cached and cached[i][1] is not None
                   for i in range(n_frames)), "sam3 predict: incomplete prefill"

        cuda = str(self.device).startswith("cuda")
        with torch.inference_mode(), cuda_identity_on_cpu(), torch.autocast(
                device_type="cuda" if cuda else "cpu", dtype=self.dtype,
                enabled=cuda):
            state = self.model.init_state(
                video_height=size, video_width=size, num_frames=n_frames,
                video_path=None, cached_features=cached,
                offload_state_to_cpu=not cuda)
            for j, lm in enumerate(ep.support_labelmaps):
                lm_r = cv2.resize(lm.astype(np.int32), (size, size),
                                  interpolation=cv2.INTER_NEAREST)
                for c in ep.class_ids:
                    m = (lm_r == c)
                    if not m.any():
                        continue  # empty prompts are skipped, as in sst_sam2
                    self.model.add_new_mask(
                        state, frame_idx=j, obj_id=int(c),
                        mask=torch.from_numpy(m).float().to(self.device))
            masks = obj_ids = obj_scores = None
            for fidx, ids, _low, video_res, scores in self.model.propagate_in_video(
                    state, 0, None, False, tqdm_disable=True,
                    propagate_preflight=True):
                if fidx == query_idx:
                    masks, obj_ids, obj_scores = video_res, ids, scores
                    break
            if masks is None:
                raise RuntimeError("sam3 propagate never reached the query frame")

        got = masks.float().cpu().numpy().reshape(len(obj_ids), size, size)
        if obj_scores is not None:
            s = obj_scores.float().cpu().numpy().reshape(-1)
            self._obj_score_min = (float(s.min()) if self._obj_score_min is None
                                   else min(self._obj_score_min, float(s.min())))
            self._obj_score_sum += float(s.sum())
            self._obj_score_n += len(s)
        index = {int(o): i for i, o in enumerate(obj_ids)}
        logits = np.full((len(ep.class_ids), size, size), -1e4, np.float32)
        for i, c in enumerate(ep.class_ids):
            if int(c) in index:
                logits[i] = got[index[int(c)]]
        # The 1-entry-LRU never fired if the store still covers every frame key.
        self.cache_stats["misses_in_predict"] += sum(
            1 for k in keys if k not in self._store)
        return logits

    # ── record ───────────────────────────────────────────────────────────────

    def config_report(self) -> dict:
        return {
            "family": "sam3",
            "normalization": "half",
            "image_size": NATIVE_SIZE,
            "mask_input_size": self.mask_input_size,
            "num_maskmem": int(getattr(self.model, "num_maskmem", 0) or 0)
            if self.model is not None else None,
            "checkpoint": str(self.checkpoint),
            "dtype": str(self.dtype),
            "resize_interp": "linear",
            "global_autocast_exited": self._autocast_exited,
            "decode": {"dynamic_multimask_via_stability": True,
                       "multimask_output_for_tracking": True},
            "max_cond_frames_in_attn": int(getattr(
                self.model, "max_cond_frames_in_attn", 0) or 0)
            if self.model is not None else None,
            "obj_score_min": self._obj_score_min,
            "obj_score_mean": (self._obj_score_sum / self._obj_score_n
                               if self._obj_score_n else None),
            **self._load_report,
        }

    def close(self):
        self.clear_embeddings()
        self.model = None
