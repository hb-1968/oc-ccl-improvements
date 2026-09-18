"""SAM 2 via SST's vendored `SAM2VideoPredictor`.

This is the paper's configuration: SAM 2.0 hiera-large from the 072824 release, 1024px,
`num_maskmem=7`. `build()` is ported from `experiments/eval_checkpoint.py::build_predictor`
in the SST checkout, whose Hydra `initialize_config_dir` pattern is the robust one -- the
vendored `sam2/__init__.py` calls `initialize_config_module("sam2_configs")` at import
time, which only resolves with `SST/src/sst` on `sys.path`.

Four deliberate departures from `eval_checkpoint.py`, each a correctness fix:

1. **Returns logits, not booleans.** It does `(logits[k] > 0)` inside predict, which
   discards the scores the paper's highest-score-label rule needs.
2. **Loads strictly.** It uses `load_state_dict(..., strict=False)`, which silently
   leaves randomly-initialised weights when keys do not line up -- the same failure that
   invalidated the repo's LoRA ablation rows.
3. **Resizes the label map once with INTER_NEAREST, then slices per class.**
   `eval_all_ablations.py` does `cv2.resize(binary_mask, ...)` at the default
   INTER_LINEAR, fuzzing prompt boundaries.
4. **dtype by compute capability**, rather than a hardcoded bfloat16 that silently
   degrades on pre-Ampere GPUs.

Frames are pre-resized to 1024x1024 with cv2 INTER_LINEAR before being handed over,
matching SST. That is not cosmetic: `_load_img_as_tensor` resizes anything that is not
already at `image_size` using **PIL BICUBIC**, so passing native-resolution arrays would
quietly change the interpolation that produced the paper's numbers.

Embedding cache
---------------
The naive protocol re-encodes the same images relentlessly. Butterfly Major 1-shot is
20 runs x 5 subspecies x 100 queries x 2 frames = **20,000 encoder passes over 957
distinct images**, a ~21x redundancy -- and the hiera-L image encoder is 212.7M of the
model's 224.4M parameters (94.8%), so that redundancy is essentially the whole cost.

`precompute()` encodes each distinct image once and `predict()` serves the rest from
cache, by patching `_get_image_feature`. Patching rather than pre-filling
`inference_state["cached_features"]` is deliberate: `init_state` itself calls
`_get_image_feature(frame_idx=0)` before returning, so a post-hoc fill would still pay
for the support frame every episode.

Two measured facts make the cache cheap:

* `vision_pos_enc` is **bit-identical for every image** (verified across random inputs at
  all three levels) because it is a function of feature-map shape, not content. It is
  computed once and shared, which drops the per-image cost from 52.4 MB to **8.39 MB**
  in bf16 -- 88 MB of the 105 MB fp32 footprint was positional encoding.
* The cached value is batch-1; SAM 2 `.expand()`s it to the object count itself, and
  expand returns a view, so sharing the tensors across episodes is safe.

All 957 butterfly images cache in ~8.0 GB, comfortable on a 40 GB A100. Caching in the
autocast dtype is numerically identical to recomputing, since `forward_image` under
autocast would have produced exactly these tensors.
"""

from __future__ import annotations

import numpy as np

from .. import paths
from ..datasets.base import Episode
from .base import pick_dtype

CONFIG_NAME = "sam2_hiera_l"
TARGET = "sst.segment_anything_2.sam2.sam2_video_predictor.SAM2VideoPredictor"

#: Exactly the constants `sam2/utils/misc.py::load_video_frames` applies, unconditionally,
#: on both its JPEG and in-memory branches. The precompute path must reproduce them or its
#: embeddings would not match what the uncached path produces.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_model_tensor(img, size: int = 1024, device=None):
    """``HxWx3`` uint8 -> the (1, 3, size, size) float tensor SAM 2's encoder expects.

    THE transform, in one place. `precompute` uses it, and so does `finetune.pairs` — which
    is defect **A1**: SST's OC-CCL trainer normalises with `/255` alone
    (`SST/src/sst/butterfly_dataset.py:139-140`) while every inference path subtracts
    ImageNet mean/std (`sam2/utils/misc.py:164-165`), so it trains on a different input
    distribution than it is scored on.

    Kept here rather than in `finetune/` deliberately: the inference path is the authority
    on what the model expects, and a training-side copy is what created the defect. Same
    reasoning as `curation.draw_supports` — several places that must agree, with one place
    that decides.
    """
    import cv2
    import torch

    arr = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    t = ((t - mean) / std).unsqueeze(0)
    return t.to(device) if device is not None else t


class SSTSam2Adapter:
    name = "sst_sam2"
    native_size = 1024

    def __init__(
        self,
        checkpoint=None,
        config_name: str = CONFIG_NAME,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
    ):
        self.checkpoint = str(checkpoint) if checkpoint else str(paths.default_checkpoint())
        self.config_name = config_name
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self.model = None
        self.device = "cuda"
        self.dtype = None
        self.load_report: dict = {}
        # Embedding cache: image_key -> list of batch-1 backbone_fpn tensors.
        self._embed: dict[str, list] = {}
        self._pos: list | None = None  # shared vision_pos_enc (content-independent)
        self._frame_keys: dict[int, str] = {}
        self._orig_get_feature = None
        self.cache_stats = {"hits": 0, "misses": 0, "encoded": 0}

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, device: str = "cuda", dtype=None, strict: bool = True) -> None:
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        sst = paths.bootstrap_sst_syspath()
        self.device = device
        self.dtype = dtype or pick_dtype(device)

        # Clear first: notebook cell re-runs otherwise leave dirty Hydra global state.
        GlobalHydra.instance().clear()
        cfg_dir = str(sst / "src" / "sst" / "sam2_configs")
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
            cfg = compose(config_name=self.config_name,
                          overrides=[f"++model._target_={TARGET}"])
            OmegaConf.resolve(cfg)
            model = instantiate(cfg.model, _recursive_=True)

        sd = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.load_report = {
            "missing_keys": len(missing), "unexpected_keys": len(unexpected),
            "missing_sample": list(missing)[:5], "unexpected_sample": list(unexpected)[:5],
        }
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"checkpoint did not load cleanly: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys. Loading non-strictly here would "
                f"leave randomly-initialised weights and silently score noise.\n"
                f"  missing: {list(missing)[:5]}\n  unexpected: {list(unexpected)[:5]}"
            )

        self.model = model.to(device).eval()
        self._install_feature_patch()

    # ── embedding cache ──────────────────────────────────────────────────────

    def _install_feature_patch(self) -> None:
        """Route `_get_image_feature` through our cache.

        Wrapping instead of pre-filling `inference_state["cached_features"]` is what lets
        the support frame hit the cache too: `init_state` warms frame 0 before it returns,
        so anything filled afterwards arrives too late for that frame.
        """
        model = self.model
        orig = model._get_image_feature
        self._orig_get_feature = orig

        def patched(inference_state, frame_idx, batch_size):
            key = self._frame_keys.get(frame_idx)
            if key is not None and key in self._embed:
                image = inference_state["images"][frame_idx]
                image = image.to(inference_state["device"]).float().unsqueeze(0)
                inference_state["cached_features"] = {
                    frame_idx: (image, {"backbone_fpn": self._embed[key],
                                        "vision_pos_enc": self._pos})
                }
                self.cache_stats["hits"] += 1
            elif key is not None:
                self.cache_stats["misses"] += 1
            return orig(inference_state, frame_idx, batch_size)

        model._get_image_feature = patched

    def precompute(self, items, progress_every: int = 0) -> dict:
        """Encode ``(key, image)`` pairs once and retain their backbone features.

        ``items`` may be any iterable of ``(str, np.ndarray)``; already-cached keys are
        skipped, so calling this per group is idempotent and incremental.
        """
        import torch

        if self.model is None:
            raise RuntimeError("call build() first")
        s = self.native_size
        cuda = str(self.device).startswith("cuda")

        for key, img in items:
            if key in self._embed:
                continue
            # Matches load_video_frames exactly, or cached embeddings would differ from
            # what the uncached path produces. See to_model_tensor.
            t = to_model_tensor(img, s, self.device)
            with torch.inference_mode(), torch.autocast(
                self.device, dtype=self.dtype, enabled=cuda
            ):
                out = self.model.forward_image(t)
            self._embed[key] = [f.detach() for f in out["backbone_fpn"]]
            if self._pos is None:  # identical for every image; keep exactly one copy
                self._pos = [p.detach() for p in out["vision_pos_enc"]]
            self.cache_stats["encoded"] += 1
            n = self.cache_stats["encoded"]
            if progress_every and n % progress_every == 0:
                print(f"    precomputed {n} embeddings", flush=True)
        return self.cache_report()

    def clear_embeddings(self) -> None:
        self._embed.clear()
        self._frame_keys.clear()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def cache_report(self) -> dict:
        per = 0
        if self._embed:
            f = next(iter(self._embed.values()))
            per = sum(t.numel() * t.element_size() for t in f)
        return {"n_cached": len(self._embed), "bytes_per_image": per,
                "total_gb": round(len(self._embed) * per / 1e9, 3), **self.cache_stats}

    def config_report(self) -> dict:
        """Facts the S3 gate asserts against the paper.

        `family` and `normalization` are also read by `registry.classify` — without them
        every record silently lands in the `modified` track on fields that never varied.
        """
        m = self.model
        return {
            "family": "sam2",
            "normalization": "imagenet",  # IMAGENET_MEAN/STD, applied in predict()
            "image_size": int(m.image_size),
            "num_maskmem": int(m.num_maskmem),
            "checkpoint": self.checkpoint,
            "dtype": str(self.dtype),
            "load_missing_keys": self.load_report.get("missing_keys"),
            "load_unexpected_keys": self.load_report.get("unexpected_keys"),
        }

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, ep: Episode) -> np.ndarray:
        import cv2
        import torch

        if self.model is None:
            raise RuntimeError("call build() first")

        s = self.native_size
        frames = [
            cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
            for img in (*ep.support_images, ep.query_image)
        ]
        query_idx = len(ep.support_images)
        # Consulted by the patched _get_image_feature. Set before init_state, which warms
        # frame 0 on its way out.
        self._frame_keys = {j: sid for j, sid in enumerate(ep.support_ids)}
        self._frame_keys[query_idx] = ep.query_id

        state = None
        try:
            with torch.inference_mode(), torch.autocast(
                self.device, dtype=self.dtype, enabled=str(self.device).startswith("cuda")
            ):
                state = self.model.init_state(
                    None,
                    image_inputs=frames,
                    offload_video_to_cpu=self.offload_video_to_cpu,
                    offload_state_to_cpu=self.offload_state_to_cpu,
                    async_loading_frames=False,
                    verbose=False,
                )
                # Once, before prompting. NEVER between frames -- carrying the memory
                # bank forward is the entire method.
                self.model.reset_state(state)

                for j, lm in enumerate(ep.support_labelmaps):
                    lm_r = cv2.resize(lm.astype(np.int32), (s, s),
                                      interpolation=cv2.INTER_NEAREST)
                    for c in ep.class_ids:
                        binary = lm_r == c
                        if not binary.any():
                            continue  # class absent from this support: legal, no prompt
                        self.model.add_new_mask(
                            inference_state=state, frame_idx=j, obj_id=int(c),
                            mask=binary.astype(np.uint8),
                        )

                out = None
                for frame_idx, obj_ids, video_res_masks in self.model.propagate_in_video(
                    state, verbose=False
                ):
                    if frame_idx == query_idx:
                        out = (list(obj_ids), video_res_masks.float().cpu().numpy())
                        break

            if out is None:
                raise RuntimeError(f"propagation never reached query frame {query_idx}")

            obj_ids, masks = out
            masks = masks.reshape(masks.shape[0], *masks.shape[-2:])  # (n_obj, H, W)

            # Reindex to the episode's class order: propagate_in_video does not promise
            # obj_ids come back in insertion order. A class never prompted (absent from
            # every support) gets a constant-negative channel, i.e. "never this label".
            index = {int(o): i for i, o in enumerate(obj_ids)}
            h, w = masks.shape[-2:]
            logits = np.full((len(ep.class_ids), h, w), -1e4, dtype=np.float32)
            for i, c in enumerate(ep.class_ids):
                if int(c) in index:
                    logits[i] = masks[index[int(c)]]
            return logits
        finally:
            self._frame_keys = {}
            if state is not None:
                del state
            # NOTE: no empty_cache() here. It would return the embedding cache's blocks to
            # the allocator every episode and force re-allocation on the next one, which
            # costs more than it saves. Freed once in close()/clear_embeddings() instead.

    def close(self) -> None:
        self.model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
