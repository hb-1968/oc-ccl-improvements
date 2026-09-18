"""SAM 3 port pins — the port's load-bearing assumptions, tested without weights.

The heavyweight properties (strict weight surgery, cache bit-identity, real
gradients) live in the wiring smoke and the Colab commissioning cell; what pytest
pins is everything that can silently drift: the input transform, the A1 bridge into
`pairs`, the 20/-10 constants in BOTH checkouts, and the no-resample mask geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from occl.finetune.lora import (  # noqa: E402
    FAITHFUL_SCOPES, FAITHFUL_SCOPES_BY_FAMILY,
)


def _sam3_root():
    from occl import paths

    try:
        return paths.sam3_root()
    except FileNotFoundError:
        return None


def test_to_model_tensor_sam3_is_half_normalized_1008_fp32():
    from occl.adapters.sam3 import to_model_tensor_sam3
    from occl.adapters.sst_sam2 import to_model_tensor

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (640, 480, 3), np.uint8)
    t = to_model_tensor_sam3(img)
    assert t.shape == (1, 3, 1008, 1008) and t.dtype == torch.float32

    import cv2

    want = cv2.resize(img, (1008, 1008), interpolation=cv2.INTER_LINEAR)
    want = (torch.from_numpy(want).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
    assert torch.allclose(t[0], want, atol=1e-6)
    # and it is NOT the ImageNet transform — the silent-normalization-leak guard
    assert not torch.allclose(t, to_model_tensor(img, 1008), atol=1e-2)


def test_pairs_half_branch_bridges_to_the_sam3_transform():
    """The A1 discipline, re-earned for the new family: training pixels come from the
    inference adapter's own function, element-identically."""
    from occl.adapters.sam3 import to_model_tensor_sam3
    from occl.finetune.pairs import CyclePairs

    cp = CyclePairs.__new__(CyclePairs)
    cp.normalization = "half"
    cp.size = 1008
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (200, 300, 3), np.uint8)
    assert torch.equal(cp._to_tensor(img), to_model_tensor_sam3(img, 1008))


def test_pairs_mask_size_defaults_to_size_and_splits_for_sam3():
    from occl.finetune.pairs import CyclePairs

    cp = CyclePairs.__new__(CyclePairs)
    cp.size = 1024
    cp.mask_size = cp.size  # what __init__(mask_size=None) resolves to
    assert cp.mask_size == 1024

    from occl.finetune.pairs import trait_mask_stack

    lm = np.zeros((100, 100), np.int32)
    lm[10:40, 10:40] = 1
    lm[60:90, 60:90] = 3
    y, ids = trait_mask_stack(lm, 1152)
    assert y.shape == (2, 1, 1152, 1152) and ids == (1, 3)
    assert set(y.unique().tolist()) == {0.0, 1.0}  # NEAREST keeps masks binary,
    # which is what makes add_new_mask's >0.5 threshold exact


def test_scopes_by_family_keeps_sam2_verbatim():
    assert FAITHFUL_SCOPES_BY_FAMILY["sam2"] is FAITHFUL_SCOPES
    assert FAITHFUL_SCOPES_BY_FAMILY["sec"] is FAITHFUL_SCOPES
    assert FAITHFUL_SCOPES_BY_FAMILY["sam3"] == ("maskmem_backbone",
                                                 "sam_mask_decoder")


@pytest.mark.skipif(_sam3_root() is None, reason="no sam3 checkout")
def test_sam3_source_keeps_the_constants_and_geometry_cycle_relies_on():
    """The loud-break promise from cycle.py's docstring, honoured as a grep: if a
    future sam3 (or SST) changes the mask-as-output constants or the mask-resample
    skip, this fails before any number is produced."""
    from occl import paths

    sam3_base = (_sam3_root() / "sam3" / "model"
                 / "sam3_tracker_base.py").read_text()
    assert "out_scale, out_bias = 20.0, -10.0" in sam3_base
    pred = (_sam3_root() / "sam3" / "model"
            / "sam3_tracking_predictor.py").read_text()
    # masks already at input_mask_size skip the antialiased resample entirely —
    # the no-resample discipline our 1152 pre-resize relies on
    assert "if mask_H != self.input_mask_size or mask_W != self.input_mask_size" \
        in pred

    sst = paths.sst_root()
    sam2_base = (sst / "src" / "sst" / "segment_anything_2" / "sam2" / "modeling"
                 / "sam2_base.py").read_text()
    assert "out_scale, out_bias = 20.0, -10.0" in sam2_base


@pytest.mark.skipif(_sam3_root() is None, reason="no sam3 checkout")
def test_sam3_tracker_asserts_the_simple_maskmem_backbone():
    """image=None in track_step is legal ONLY on the SimpleMaskEncoder branch — the
    tracker must refuse anything else (the sam3.1 multiplex classes consume it)."""
    from occl.finetune.tracker_sam3 import DifferentiableSAM3Tracker

    class _NotSimple:
        maskmem_backbone = object()

    with pytest.raises((AssertionError, ImportError)):
        DifferentiableSAM3Tracker(_NotSimple())


def test_surgery_map_routes_tracker_and_shared_backbone_keys():
    """First real contact with sam3.pt measured 464 missing / 0 unexpected: the
    surgery's `detector.backbone.visual.` prefix was the CONSTRUCTOR PARAMETER
    name (build_tracker passes visual=...), but SAM3VLBackbone stores the trunk
    at `.vision_backbone` (vl_combiner.py), so the prefix matched nothing. The
    mapping is now pure and model-driven; this pins its routing:
      * tracker.* -> root, unconditionally (an unexpected key must surface);
      * detector.backbone.X -> X, ONLY if the model declares it (the detector's
        text tower is deliberately absent from the tracker);
      * any other detector.* NEVER crosses — the detector has its own
        `transformer`, and a name coincidence must not graft detection weights
        into the tracker's memory attention."""
    from occl.adapters.sam3 import surgery_map

    sd = {
        "tracker.transformer.layers.0.w": 1,
        "tracker.sam_mask_decoder.w": 2,
        "tracker.never_seen.w": 3,                       # forwarded -> loud unexpected
        "detector.backbone.vision_backbone.trunk.w": 4,  # the shared trunk
        "detector.backbone.text_backbone.w": 5,          # text tower: dropped
        "detector.transformer.layers.0.w": 6,            # DETR transformer: dropped
    }
    model_keys = {
        "transformer.layers.0.w", "sam_mask_decoder.w",
        "backbone.vision_backbone.trunk.w",
    }
    assert surgery_map(sd, model_keys) == {
        "transformer.layers.0.w": 1,
        "sam_mask_decoder.w": 2,
        "never_seen.w": 3,
        "backbone.vision_backbone.trunk.w": 4,
    }


def test_adapter_attends_to_every_conditioning_frame_at_5_shot():
    """S15's pre-condition, at the level that decides it: sam3's builder caps
    conditioning frames at 4, so a 5-shot episode (supports at frames 0-4, query
    at 5) would attend to the 4 closest and drop frame 0. The adapter sets -1 —
    SAM 2's setting under SST — and the tracker's selector returns every frame.
    (A full CPU build takes >15 min on random weights; the GPU commissioning
    step, `commission_s14.py --shots 5`, proves the same on the real model.)"""
    import inspect

    from occl.adapters import sam3 as adapter_mod

    src = inspect.getsource(adapter_mod.Sam3TrackerAdapter.build)
    assert "model.max_cond_frames_in_attn = -1" in src

    import sys

    from occl import paths

    # under pytest the repo root is on sys.path, so `sam3` may already be bound to
    # the CHECKOUT DIRECTORY as a namespace package (no __file__); the real package
    # is one level down, where bootstrap_sam3_syspath points
    if getattr(sys.modules.get("sam3"), "__file__", None) is None:
        for k in [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]:
            del sys.modules[k]
    paths.bootstrap_sam3_syspath()
    from sam3.model.sam3_tracker_utils import select_closest_cond_frames

    cond = {i: object() for i in range(5)}
    sel_all, unsel_all = select_closest_cond_frames(5, cond, -1)
    assert sorted(sel_all) == [0, 1, 2, 3, 4] and not unsel_all
    sel_cap, unsel_cap = select_closest_cond_frames(5, cond, 4)
    assert len(sel_cap) == 4 and sorted(unsel_cap) == [0], \
        "the builder's cap silently drops the first support at 5-shot"
