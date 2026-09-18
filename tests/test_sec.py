"""SeC port pins — everything that can drift without weights present.

The decisive pin is the first one: SeC's native scene-change gate is BLIND to the
exact transformations this benchmark contains (180° flips, same-rig specimen pairs)
— that fact is what licenses the `forced` concept policy as the primary arm, so it
is asserted, not narrated.
"""

from __future__ import annotations

import numpy as np
import pytest

from occl.adapters.sec import (
    SCENE_CHANGE_THRESHOLD, SeCAdapter, bhattacharyya_hsv,
    resolve_grounding_config,
)


def test_scene_change_statistic_is_blind_to_rotation():
    """Histograms are spatially invariant: a 180-rotated specimen is bit-identical
    to its upright self under SeC's detector, so the native gate can never fire on
    the flip cohort — the whole reason `forced` exists."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 256, 3), np.uint8)
    assert bhattacharyya_hsv(img, img) < 1e-6
    assert bhattacharyya_hsv(img, np.rot90(img, 2).copy()) < 1e-6

    # while a genuine appearance change clears the threshold
    other = np.zeros_like(img)
    other[..., 0] = 255  # saturated red field vs noise
    assert bhattacharyya_hsv(img, other) > SCENE_CHANGE_THRESHOLD


def test_grounding_config_bug_is_rewritten():
    assert (resolve_grounding_config("longsam2.1/longsam2.1_hiera_l.yaml")
            == "sec_sam2.1_hiera_l.yaml")
    assert (resolve_grounding_config("sec_sam2.1_hiera_l.yaml")
            == "sec_sam2.1_hiera_l.yaml")


def test_concept_policy_is_validated_and_named_into_the_resume_slug():
    """`sec_off` and `sec_forced` are different measurements and must live in
    different resume slug spaces — the name carries the policy."""
    assert SeCAdapter(concept_policy="off").name == "sec_off"
    assert SeCAdapter(concept_policy="forced").name == "sec_forced"
    with pytest.raises(ValueError, match="concept_policy"):
        SeCAdapter(concept_policy="native")


def test_adapter_is_importable_torch_free():
    """The pyproject law: the base install has no torch; adapter modules must keep
    heavyweight imports function-local."""
    import sys

    assert "occl.adapters.sec" in sys.modules  # imported above without torch errors


def test_precompute_encode_mirrors_the_vendored_miss_path():
    """Cache bit-identity (the S3 predicate, re-earned per family by
    commission_s14) requires `_encode` to reproduce the vendored
    `_get_image_feature` miss path EXACTLY: the state's tracker-dtype image
    upcast `.float()`, encoded inside predict's autocast. On CPU any deviation
    is invisible (autocast disabled, everything fp32) — the first s14sec
    commissioning on A100 failed precisely because `_encode` fed a bf16 input
    with no autocast — so the context is pinned at the source level, the only
    guard a CUDA-less box can hold."""
    import inspect

    src = inspect.getsource(SeCAdapter._encode)
    assert "torch.autocast" in src, "_encode must encode under predict's autocast"
    assert ".float()" in src, "_encode must upcast the tracker-dtype image as the miss path does"


def test_off_policy_drops_lvlm_before_the_device_move():
    """Order is load-bearing and only fails on a GPU box: dropped AFTER
    `.to(device)`, the LVLM's freed VRAM stays reserved by the caching allocator —
    invisible-unusable to sibling eval workers (measured 8.42 GB per off worker;
    four residents + a fifth build exhausted the 39.5 GB card on the first
    butterfly fan-out). Pin the source order, the only local guard."""
    import inspect

    src = inspect.getsource(SeCAdapter.build)
    assert src.index("self.drop_lvlm()") < src.index("model.eval().to(device)"), (
        "off-policy drop_lvlm must precede the device move in build")
