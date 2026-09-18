"""S17 — the training-pair probe must reproduce a training step exactly, and score what
the step discards without touching what it keeps.

The fidelity claims are pinned on `FakeCycleTracker` (object-independent, fixed
arithmetic), so every expected tensor is computable in the test: forward = (y0*20-10)*0.5,
closing = forward*0.5 (the reparam prompt round-trips exactly).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from occl.finetune.cycle import oc_ccl_step
from occl.finetune.probe import (CARRIER_RECIPE, cycle_probe, cyclepairs_kwargs,
                                 fmt_per_trait, loss_parity, oc_ccl_kwargs_from_recipe,
                                 parse_per_trait, recipe_from_lora_meta, replay_checks,
                                 score_leg, write_labelmap_png, read_labelmap_png,
                                 y0_labelmap)
from tests.test_finetune import FakeCycleTracker, FakePairDS, _pair

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cycle_probe_is_a_training_step_with_the_forward_leg_kept():
    x0, xu, m1, m2 = _pair(seed=3)
    y0 = torch.cat([m1, m2])
    tr = FakeCycleTracker()
    loss, closing, parts = oc_ccl_step(tr, x0, xu, y0)
    res = cycle_probe(tr, x0, xu, y0)

    # closing mask + loss: bit-identical to the plain step at the same weights
    assert res.loss == float(loss)
    assert np.array_equal(res.closing, closing.detach().numpy()[:, 0])
    assert res.parts["bce"] == parts["bce"] and res.parts["dice"] == parts["dice"]
    assert res.parts["dice_per_channel"] == parts["dice_per_channel"]
    # the forward leg is the stub's propagate of the GT-prompted init
    expected_fwd = ((y0 * 20.0 - 10.0) * 0.5).numpy()[:, 0]
    assert np.array_equal(res.forward, expected_fwd)
    assert np.array_equal(res.closing, expected_fwd * 0.5)
    assert res.forward.shape == (2, 16, 16) and res.forward.dtype == np.float32


def test_capture_default_leaves_oc_ccl_step_untouched():
    """Nothing before S17 passes ``capture``; the default path must be the old path."""
    import inspect

    assert inspect.signature(oc_ccl_step).parameters["capture"].default is None
    x0, xu, m1, _ = _pair(seed=5)
    tr = FakeCycleTracker()
    a = oc_ccl_step(tr, x0, xu, m1)
    cap: dict = {}
    b = oc_ccl_step(tr, x0, xu, m1, capture=cap)
    assert float(a[0]) == float(b[0])
    assert torch.equal(a[1], b[1])
    assert set(cap) == {"forward"}
    assert cap["forward"].shape == m1.shape


def test_cycle_probe_honours_the_recipe_knobs():
    """`excl` changes the loss (the C=2 identity breaks by design); the probe must pass the
    knob through rather than silently replaying the plain BCE."""
    x0, xu, m1, m2 = _pair(seed=7)
    y0 = torch.cat([m1, m2])
    tr = FakeCycleTracker()
    plain = cycle_probe(tr, x0, xu, y0)
    excl = cycle_probe(tr, x0, xu, y0, excl_mode="zerocat")
    assert excl.loss != plain.loss
    assert excl.parts["excl_mode"] == "zerocat" and "ce" in excl.parts
    assert np.array_equal(excl.forward, plain.forward)      # legs are knob-free


def test_y0_labelmap_round_trips_trait_mask_stack():
    from occl.finetune.pairs import trait_mask_stack

    lm = FakePairDS().load_labelmap("a")
    y0, ids = trait_mask_stack(lm, 24)          # 24 != 32: a real NEAREST resize happens
    lab = y0_labelmap(y0, ids)
    assert lab.shape == (24, 24)
    assert sorted(int(t) for t in np.unique(lab) if t) == list(ids)
    # every channel painted back exactly
    for c, t in enumerate(ids):
        assert np.array_equal(lab == t, y0[c, 0].numpy() > 0.5)
    with pytest.raises(ValueError):
        y0_labelmap(y0, ids[:1])


def test_score_leg_separates_prompted_from_unprompted_traits():
    """A GT trait the reference never carried scores 0 under the eval rule (counted) but is
    excluded from `miou_prompted`; `found` reads the channel claim, not the IoU."""
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[2:8, 2:8] = 1
    gt[12:18, 12:18] = 5           # present in GT, absent from the support
    logits = np.full((1, 20, 20), -5.0, dtype=np.float32)
    logits[0, 2:8, 2:8] = 5.0      # a perfect trait-1 prediction
    s = score_leg(logits, (1,), gt)
    assert s["per_trait"] == {1: 1.0, 5: 0.0}
    assert s["miou"] == pytest.approx(0.5)
    assert s["miou_prompted"] == pytest.approx(1.0)
    assert s["n_gt_traits"] == 2 and s["n_prompted_present"] == 1
    assert s["found"] == [True] and s["found_count"] == 1
    assert s["claimed_frac"] == pytest.approx(36 / 400)
    assert s["labelmap"].shape == gt.shape and s["iou_space"] == "gt_native"

    # an unfound channel: claims nowhere, IoU 0, found False
    s2 = score_leg(np.full((1, 20, 20), -5.0, dtype=np.float32), (1,), gt)
    assert s2["found"] == [False] and s2["per_trait"][1] == 0.0
    # model space: GT is resized to the logits, not the other way round
    s3 = score_leg(logits, (1,), gt, iou_space="model")
    assert s3["gt"].shape == (20, 20) and s3["iou_space"] == "model"
    # prompted set empty -> None, never a NaN
    s4 = score_leg(np.full((1, 20, 20), -5.0, dtype=np.float32), (7,), gt)
    assert s4["miou_prompted"] is None


def _meta(**over):
    """Shaped like `run_finetune.py`'s save_lora meta for the S11 carrier."""
    m = {"method": "lora", "epochs": 1, "lr": 1e-5, "family": "sam2",
         "image_size": 1024, "mask_size": 1024,
         "lora": {"r": 32, "alpha": 64.0, "scopes": ["memory_encoder", "sam_mask_decoder"]},
         "occl": {"prompt_mode": "logit_reparam", "train_mode": False,
                  "normalization": "imagenet", "memory_reset": True, "cycle_len": 2,
                  "mask_mode": "per_trait", "reference": "derangement",
                  "reference_id": None, "reference_run": None, "augmentation": None,
                  "scnp": None, "excl": None, "xovl": None, "bshape": None,
                  "margin": None, "found_mask": None, "bce_w": None,
                  "objness": None, "lora_legs": None, "soft_sentinel": None,
                  "close_aug": None}}
    for k, v in over.items():
        if k in m["occl"]:
            m["occl"][k] = v
        else:
            m[k] = v
    return m


def test_recipe_from_lora_meta_is_the_carrier_for_a_carrier_checkpoint():
    r = recipe_from_lora_meta(_meta())
    for k, v in CARRIER_RECIPE.items():
        assert r[k] == v, k
    assert cyclepairs_kwargs(r) == {"size": 1024, "mask_size": 1024,
                                    "normalization": "imagenet", "mask_mode": "per_trait",
                                    "reference_id": None, "rotation_aug_deg": None,
                                    "close_aug_deg": None}
    assert oc_ccl_kwargs_from_recipe(r) == {"prompt_mode": "logit_reparam",
                                            "memory_reset": True}


def test_recipe_carries_every_knob_and_refuses_what_cannot_be_replayed():
    r = recipe_from_lora_meta(_meta(
        excl={"mode": "zerocat"}, augmentation={"type": "rotation_x0", "max_deg": 27.8},
        reference="fixed", reference_id="advena/x", family="sam3",
        image_size=1008, mask_size=1152))
    assert cyclepairs_kwargs(r) == {"size": 1008, "mask_size": 1152,
                                    "normalization": "imagenet", "mask_mode": "per_trait",
                                    "reference_id": "advena/x", "rotation_aug_deg": 27.8,
                                    "close_aug_deg": None}
    assert oc_ccl_kwargs_from_recipe(r)["excl_mode"] == "zerocat"
    xo = recipe_from_lora_meta(_meta(xovl={"weight": 1.0, "where": "both"},
                                     bshape={"weight": 1.0}))
    kw = oc_ccl_kwargs_from_recipe(xo)
    assert kw["xovl_weight"] == 1.0 and kw["xovl_where"] == "both" and kw["bshape_weight"] == 1.0
    with pytest.raises(ValueError, match="margin-only"):
        oc_ccl_kwargs_from_recipe(recipe_from_lora_meta(
            _meta(margin={"m": 1.0, "weight": None, "only": True})))
    with pytest.raises(ValueError, match="occl"):
        recipe_from_lora_meta({"lora": {"r": 32, "alpha": 64.0}, "epochs": 1})


def test_replay_checks_read_the_training_report_and_record_a_capped_smoke():
    from occl.finetune.pairs import CyclePairs

    ds = FakePairDS()
    pairs = CyclePairs(ds, "all", size=24, mask_mode="per_trait")
    assert len(pairs) == 3
    full = {"steps_per_epoch": 3, "train_images": 3, "mask_channels": {"2": 3},
            "loss_first": 0.5}
    out = replay_checks(ds, pairs, full, mask_size=24)
    assert out["n_pairs"] == 3 and out["train_images"] == 3
    assert out["checks"] == {"steps_per_epoch_matches": True, "train_images_matches": True,
                             "mask_channels_matches": True, "steps_capped": False}
    assert out["mask_channels_replayed"] == {"2": 3}

    capped = dict(full, steps_per_epoch=2, mask_channels={"2": 2})
    out = replay_checks(ds, pairs, capped, mask_size=24)
    assert out["checks"]["steps_capped"] is True
    assert out["checks"]["steps_per_epoch_matches"] is True     # capped, not mismatched
    assert out["checks"]["mask_channels_matches"] is True       # over the steps that ran

    wrong = dict(full, train_images=120, mask_channels={"5": 3})
    out = replay_checks(ds, pairs, wrong, mask_size=24)
    assert out["checks"]["train_images_matches"] is False
    assert out["checks"]["mask_channels_matches"] is False

    none = replay_checks(ds, pairs, None, mask_size=24)
    assert none["train_record"] is False
    assert all(v is None for v in none["checks"].values())


def test_loss_parity_is_a_relative_error_and_survives_a_missing_record():
    p = loss_parity(0.18008950352668762, 0.18008950352668762)
    assert p["loss_first_rel_err"] == 0.0
    assert loss_parity(0.2, 0.1)["loss_first_rel_err"] == pytest.approx(1.0)
    assert loss_parity(0.2, None)["loss_first_rel_err"] is None


def test_paired_bootstrap_is_exactly_what_compare_computes():
    """Lifted out of `compare`, so its numbers must be the ones `compare` has always
    reported — same rng, same percentiles, same significance rule."""
    from occl.registry import compare, paired_bootstrap
    from tests.test_registry import make_record

    treat = [0.71, 0.66, 0.69, 0.74, 0.70, 0.68]
    base = [0.70, 0.64, 0.70, 0.71, 0.69, 0.66]
    c = compare(make_record(treat), make_record(base))
    d = np.array(treat) - np.array(base)
    b = paired_bootstrap(d)
    assert b["mean_delta"] == c["mean_delta"]
    assert b["ci95"] == c["ci95"]
    assert b["significant"] == c["significant"]
    assert b["n"] == c["n_seeds"] and b["n_improved"] == c["n_seeds_improved"]
    assert b["paired_std"] == c["paired_std"]
    assert b["n_degraded"] == int((d < 0).sum())
    with pytest.raises(ValueError):
        paired_bootstrap([])


def test_labelmap_png_round_trip_and_per_trait_cells(tmp_path):
    lab = np.zeros((7, 9), dtype=np.int32)
    lab[1:3, 1:4] = 3
    lab[5, 5] = 10
    p = tmp_path / "maps" / "base" / "fwd" / "0000.png"
    write_labelmap_png(p, lab)
    assert np.array_equal(read_labelmap_png(p), lab)
    with pytest.raises(ValueError):
        write_labelmap_png(tmp_path / "bad.png", np.array([[300]]))
    s = fmt_per_trait({3: 0.5, 1: 0.25})
    assert s == "1:0.2500;3:0.5000"
    assert parse_per_trait(s) == {1: 0.25, 3: 0.5}
    assert parse_per_trait("") == {}


def test_loss_anatomy_splits_the_epoch_by_mechanism():
    """Unfound-channel BCE, takeovers and the worst-pair share are read off the probe's
    per-channel columns — the 'before' picture the fine-tune's first step sees."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "anat", REPO_ROOT / "scripts" / "probe_loss_anatomy.py")
    anat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anat)

    def row(step, found, bce_c, dice_c, fwd_t, cls_t):
        C = len(found)
        bce = sum(bce_c) / C
        dice = sum(dice_c) / C
        return {"train_step": str(step), "ref": f"r{step}", "query": f"q{step}", "group": "all",
                "arm": "base", "class_ids": "1 2 3", "n_channels": "3",
                "fwd_found": ";".join("1" if f else "0" for f in found),
                "bce_per_channel": ";".join(f"{b:.4f}" for b in bce_c),
                "dice_per_channel": ";".join(f"{d:.4f}" for d in dice_c),
                "fwd_per_trait": fmt_per_trait(fwd_t), "cls_per_trait": fmt_per_trait(cls_t),
                "loss": str(bce + dice), "bce": str(bce), "dice": str(dice),
                "fwd_miou": str(sum(fwd_t.values()) / 3), "cls_miou": str(sum(cls_t.values()) / 3)}

    rows = [
        # a tame pair: everything found, small loss
        row(0, [1, 1, 1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1],
            {1: .9, 2: .9, 3: .9}, {1: .8, 2: .8, 3: .8}),
        # unfound trait 3: its channel carries almost all the BCE
        row(1, [1, 1, 0], [0.1, 0.1, 9.0], [0.1, 0.1, 1.0],
            {1: .9, 2: .9, 3: 0.0}, {1: .8, 2: .8, 3: 0.0}),
        # takeover: trait 2 found (fwd .8) but lost on the closing leg (cls .05)
        row(2, [1, 1, 1], [0.1, 3.0, 0.1], [0.1, 0.9, 0.1],
            {1: .9, 2: .8, 3: .9}, {1: .8, 2: .05, 3: .8}),
        row(3, [1, 1, 1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1],
            {1: .9, 2: .9, 3: .9}, {1: .8, 2: .8, 3: .8}),
    ]
    a = anat.anatomy(rows, top=2)
    assert a["n_pairs"] == 4 and a["pairs_with_an_unfound_channel"] == 1
    assert a["bce_share"]["unfound_channels"] == pytest.approx(9.0 / (0.3 + 9.2 + 3.2 + 0.3))
    assert a["per_trait"]["3"]["unfound"] == 1 and a["per_trait"]["3"]["found_rate"] == 0.75
    assert a["per_trait"]["3"]["bce_share_when_unfound"] == pytest.approx(9.0 / 3)
    assert a["takeovers"]["n"] == 1 and a["takeovers"]["by_trait"] == {"1": 0, "2": 1, "3": 0}
    assert [r["step"] for r in a["top"]] == [1, 2]
    assert "unfound t3" in a["top"][0]["tags"] and "takeover t2" in a["top"][1]["tags"]
    assert a["loss"]["n_over_3x_median"] >= 1
    text = anat.report(a, "base")
    assert "UNFOUND channels" in text and "takeovers" in text


# ── the S17 objectives: found_mask, nobce, and the post-hoc variants ─────────


def _pair_with_an_empty_channel(seed=11):
    """C=3: two real traits and one all-zero channel. The stub's init maps a zero mask
    to −10 everywhere and propagate halves it, so the forward leg claims NO pixel of
    channel 3 — exactly the unfound case (any_logit_gt_0 is False)."""
    x0, xu, m1, m2 = _pair(seed=seed)
    empty = torch.zeros_like(m1)
    return x0, xu, torch.cat([m1, m2, empty])


def test_found_mask_is_the_plain_loss_when_every_channel_is_found():
    x0, xu, m1, m2 = _pair(seed=9)
    y0 = torch.cat([m1, m2])
    tr = FakeCycleTracker()
    plain, _, pp = oc_ccl_step(tr, x0, xu, y0)
    masked, _, mp = oc_ccl_step(tr, x0, xu, y0, found_mask=True)
    assert float(masked) == pytest.approx(float(plain), rel=1e-6)
    assert mp["found_mask"] == [True, True] and mp["n_found"] == 2
    assert mp["bce"] == pp["bce"] and mp["dice"] == pp["dice"]        # the plain readings stay


def test_found_mask_drops_the_unfound_channel_from_the_loss():
    """With channel 3 unfound the masked loss equals the plain loss of the two found
    channels alone — the C=2 cycle on the same tracker (object independence)."""
    x0, xu, y0 = _pair_with_an_empty_channel()
    tr = FakeCycleTracker()
    plain3, _, p3 = oc_ccl_step(tr, x0, xu, y0)
    masked, _, mp = oc_ccl_step(tr, x0, xu, y0, found_mask=True)
    found2, _, _ = oc_ccl_step(tr, x0, xu, y0[:2])
    assert mp["found_mask"] == [True, True, False] and mp["n_found"] == 2
    assert float(masked) == pytest.approx(float(found2), rel=1e-5)
    assert float(masked) != pytest.approx(float(plain3), rel=1e-3)
    assert mp["bce"] == p3["bce"]                       # plain BCE still reported
    # nothing found -> a zero loss that keeps the graph
    zero, _, zp = oc_ccl_step(tr, x0, xu, y0[2:], found_mask=True)
    assert float(zero) == 0.0 and zp["n_found"] == 0 and zp["found_mask"] == [False]
    with pytest.raises(ValueError, match="found_mask composes"):
        oc_ccl_step(tr, x0, xu, y0, found_mask=True, excl_mode="zerocat")
    with pytest.raises(ValueError, match="found_mask composes"):
        oc_ccl_step(tr, x0, xu, y0, found_mask=True, scnp_width=3)


def test_bce_weight_zero_is_dice_only_and_is_recorded():
    x0, xu, m1, m2 = _pair(seed=13)
    y0 = torch.cat([m1, m2])
    tr = FakeCycleTracker()
    _, _, plain = oc_ccl_step(tr, x0, xu, y0)
    loss, _, parts = oc_ccl_step(tr, x0, xu, y0, bce_w=0.0)
    assert float(loss) == pytest.approx(plain["dice"], rel=1e-6)
    assert parts["bce_w"] == 0.0 and "bce_w" not in plain


def test_loss_variants_reproduce_every_objective_the_step_trains():
    """The post-hoc columns ARE the objectives: each equals `oc_ccl_step` with that knob,
    on the same closing logits (fp32 CPU) — the per-arm replay proof rests on this."""
    from occl.finetune.probe import loss_variants

    x0, xu, y0 = _pair_with_an_empty_channel(seed=17)
    tr = FakeCycleTracker()
    res = cycle_probe(tr, x0, xu, y0)
    found = [bool((res.forward[c] > 0).any()) for c in range(3)]
    assert found == [True, True, False]
    v = loss_variants(res.closing, y0, found)
    assert set(v) == {"plain", "found_mask", "nobce", "found_mask_nobce", "excl", "objterm",
                      "bce", "dice"}
    assert v["plain"] == pytest.approx(v["bce"] + v["dice"])
    assert v["objterm"] is None                                    # no scores given
    v2 = loss_variants(res.closing, y0, found, fwd_obj=[3.0, -2.0, 1.0])
    from occl.finetune.probe import objness_term, own_loss
    ob = objness_term([3.0, -2.0, 1.0])
    assert v2["objterm"] == pytest.approx(ob)
    # an arm's own objective is composed at read time: formula + w x objterm
    row = {"loss": v2["plain"], "loss_excl": v2["excl"], "loss_objterm": v2["objterm"]}
    assert own_loss(row, "plain") == pytest.approx(v2["plain"])
    assert own_loss(row, "plain+obj1") == pytest.approx(v2["plain"] + ob)
    assert own_loss(row, "excl+obj10") == pytest.approx(v2["excl"] + 10 * ob)
    assert own_loss({"loss": "1.5", "loss_objterm": ""}, "plain+obj3") is None
    assert own_loss({"loss": 1.5}, "other:x") is None
    assert v["plain"] == pytest.approx(float(oc_ccl_step(tr, x0, xu, y0)[0]), rel=1e-6)
    assert v["found_mask"] == pytest.approx(
        float(oc_ccl_step(tr, x0, xu, y0, found_mask=True)[0]), rel=1e-6)
    assert v["nobce"] == pytest.approx(float(oc_ccl_step(tr, x0, xu, y0, bce_w=0.0)[0]), rel=1e-6)
    assert v["found_mask_nobce"] == pytest.approx(
        float(oc_ccl_step(tr, x0, xu, y0, found_mask=True, bce_w=0.0)[0]), rel=1e-6)
    assert v["excl"] == pytest.approx(
        float(oc_ccl_step(tr, x0, xu, y0, excl_mode="zerocat")[0]), rel=1e-6)
    assert v["found_mask"] < v["plain"] and v["nobce"] < v["plain"]


def test_objective_naming_and_the_leg_loss_split():
    from occl.finetune.probe import (LEG_KEYS, LOSS_KEYS, objective_of, replay_subset,
                                     strip_objective)

    carrier = recipe_from_lora_meta(_meta())
    assert objective_of(carrier) == "plain"
    assert objective_of(recipe_from_lora_meta(_meta(found_mask={"rule": "any_logit_gt_0"}))) == "found_mask"
    assert objective_of(recipe_from_lora_meta(_meta(bce_w=0.0))) == "nobce"
    assert objective_of(recipe_from_lora_meta(
        _meta(found_mask={"rule": "any_logit_gt_0"}, bce_w=0.0))) == "found_mask_nobce"
    assert objective_of(recipe_from_lora_meta(_meta(excl={"mode": "zerocat"}))) == "excl"
    obj = {"weight": 1.0, "target": "present_for_prompted"}
    from occl.finetune.probe import split_objective
    assert objective_of(recipe_from_lora_meta(_meta(objness=obj))) == "plain+obj1"
    assert objective_of(recipe_from_lora_meta(_meta(objness=obj, excl={"mode": "zerocat"}))) == "excl+obj1"
    assert objective_of(recipe_from_lora_meta(_meta(objness=obj, lora_legs="forward"))) == "plain+obj1"
    assert objective_of(recipe_from_lora_meta(_meta(objness={"weight": 10.0}, excl={"mode": "zerocat"}))) == "excl+obj10"
    assert objective_of(recipe_from_lora_meta(_meta(objness={"weight": 0.0}))).startswith("other:")
    assert split_objective("excl+obj10") == ("excl", 10.0) and split_objective("plain") == ("plain", 0.0)
    fo = recipe_from_lora_meta(_meta(lora_legs="forward", objness=obj))
    kw = oc_ccl_kwargs_from_recipe(fo)
    assert kw["lora_legs"] == "forward" and kw["obj_weight"] == 1.0
    assert oc_ccl_kwargs_from_recipe(recipe_from_lora_meta(_meta(lora_legs="forward_link")))["lora_legs"] == "forward_link"
    assert "lora_legs" not in replay_subset(fo) and objective_of(strip_objective(fo)) == "plain"
    assert oc_ccl_kwargs_from_recipe(strip_objective(fo)).get("lora_legs") == "forward"
    assert objective_of(recipe_from_lora_meta(_meta(excl={"mode": "aggregate"}))).startswith("other:")
    assert objective_of(recipe_from_lora_meta(_meta(bshape={"weight": 1.0}))).startswith("other:")
    # arms of different objectives share one replay
    fm = recipe_from_lora_meta(_meta(found_mask={"rule": "any_logit_gt_0"}))
    assert replay_subset(fm) == replay_subset(carrier)
    assert set(replay_subset(fm)) == set(LEG_KEYS) and not set(LEG_KEYS) & set(LOSS_KEYS)
    plain = strip_objective(fm)
    assert objective_of(plain) == "plain" and replay_subset(plain) == replay_subset(fm)
    kw = oc_ccl_kwargs_from_recipe(fm)
    assert kw["found_mask"] is True and "bce_w" not in kw
    assert oc_ccl_kwargs_from_recipe(recipe_from_lora_meta(_meta(bce_w=0.0)))["bce_w"] == 0.0
    assert "found_mask" not in oc_ccl_kwargs_from_recipe(plain)


def test_sam2_arch_scopes_reach_memory_attention_and_nothing_else_new():
    """The S17 `arch` arm: a strict superset of the faithful scopes that adds exactly
    SAM 2's memory attention (the module the sentinel finding points at), with the
    image encoder untouched."""
    from occl.finetune.lora import (FAITHFUL_SCOPES, SAM2_ARCH_SCOPES, apply_lora,
                                    lora_delta_norm)
    from tests.test_finetune import FakeSam2

    assert set(FAITHFUL_SCOPES) < set(SAM2_ARCH_SCOPES)
    assert set(SAM2_ARCH_SCOPES) - set(FAITHFUL_SCOPES) == {"memory_attention"}
    m = FakeSam2()
    paths = apply_lora(m, r=4, alpha=8.0, scopes=SAM2_ARCH_SCOPES)
    assert any(p.startswith("memory_attention.") for p in paths)
    assert not any(p.startswith("image_encoder") for p in paths)
    base = apply_lora(FakeSam2(), r=4, alpha=8.0, scopes=FAITHFUL_SCOPES)
    assert set(base) < set(paths)
    assert lora_delta_norm(m) == 0.0


# ── S17b: the forward-only LoRA gate and the objectness term ─────────────────


class _GatedTracker(FakeCycleTracker):
    """The stub, plus a `.model` holding one LoRALinear whose output scales every
    propagated mask — so a leg run under `lora_disabled` is visibly the base model's —
    and an objectness score on propagated frames (mean logit per channel)."""

    def __init__(self):
        from occl.finetune.lora import LoRALinear

        self.model = torch.nn.Module()
        self.model.lin = LoRALinear(torch.nn.Linear(1, 1, bias=False), r=1, alpha=1.0, seed=0)
        with torch.no_grad():
            self.model.lin.orig.weight.fill_(1.0)
            self.model.lin.lora_A.fill_(1.0)
            self.model.lin.lora_B.fill_(0.5)          # enabled: x -> 1.5 x; disabled: x -> x

    def track_with_mask(self, backbone, mask, is_init_frame=True, output_dict=None,
                        n_objects=1):
        out, output_dict = super().track_with_mask(backbone, mask, is_init_frame,
                                                   output_dict, n_objects)
        if not is_init_frame:
            gain = self.model.lin(torch.ones(1, 1))[0, 0]
            out["pred_masks_high_res"] = out["pred_masks_high_res"] * gain
            out["object_score_logits"] = out["pred_masks_high_res"].flatten(1).mean(dim=1, keepdim=True)
        return out, output_dict


def test_lora_disabled_is_the_base_model_inside_the_block_and_restores_after():
    from occl.finetune.lora import lora_disabled, set_lora_enabled
    from tests.test_finetune import FakeSam2
    from occl.finetune.lora import FAITHFUL_SCOPES, apply_lora

    m = FakeSam2()
    apply_lora(m, r=2, alpha=4.0, scopes=FAITHFUL_SCOPES)
    lin = next(mod for _, mod in m.named_modules() if type(mod).__name__ == "LoRALinear")
    with torch.no_grad():
        lin.lora_B.fill_(0.3)
    x = torch.randn(2, lin.orig.in_features)
    with_lora = lin(x)
    assert not torch.equal(with_lora, lin.orig(x))
    with lora_disabled(m):
        assert torch.equal(lin(x), lin.orig(x))
        with lora_disabled(m):                    # re-entrant
            assert not lin.enabled
        assert not lin.enabled                    # inner exit restores the OUTER state
    assert lin.enabled and torch.equal(lin(x), with_lora)
    assert set_lora_enabled(m, False) > 0 and not lin.enabled
    with lora_disabled(m):
        pass
    assert not lin.enabled                        # restores whatever it found
    set_lora_enabled(m, True)
    with lora_disabled(None):                     # a tracker without a model: no-op
        pass
    # state_dict untouched: `enabled` is a plain attribute, never a buffer
    assert not any(k.endswith("enabled") for k in m.state_dict())


def test_forward_only_legs_leave_the_forward_leg_and_change_the_closing_leg():
    x0, xu, m1, m2 = _pair(seed=21)
    y0 = torch.cat([m1, m2])
    tr = _GatedTracker()
    all_, all_cls, pa = oc_ccl_step(tr, x0, xu, y0)
    cap: dict = {}
    fwd_, fwd_cls, pf = oc_ccl_step(tr, x0, xu, y0, lora_legs="forward", capture=cap)
    # forward leg identical (LoRA on), closing leg differs (LoRA off): base scale 1 vs 1.5
    expected_fwd = ((y0 * 20.0 - 10.0) * 0.5) * 1.5
    assert torch.allclose(cap["forward"], expected_fwd)
    assert torch.allclose(fwd_cls, expected_fwd * 0.5 * 1.0)      # closing on the base model
    assert torch.allclose(all_cls, expected_fwd * 0.5 * 1.5)
    assert float(fwd_) != float(all_) and pf["lora_legs"] == "forward" and "lora_legs" not in pa
    assert tr.model.lin.enabled                                    # restored after the step
    with pytest.raises(ValueError, match="lora_legs"):
        oc_ccl_step(tr, x0, xu, y0, lora_legs="closing")


def test_objectness_term_reaches_the_forward_scores_and_is_recorded():
    import torch.nn.functional as F

    x0, xu, m1, m2 = _pair(seed=23)
    y0 = torch.cat([m1, m2])
    tr = _GatedTracker()
    plain, _, pp = oc_ccl_step(tr, x0, xu, y0)
    assert "fwd_obj_score" in pp and len(pp["fwd_obj_score"]) == 2 and "obj" not in pp
    assert "cls_obj_score" in pp and len(pp["cls_obj_score"]) == 2
    loss, _, parts = oc_ccl_step(tr, x0, xu, y0, obj_weight=1.0)
    scores = torch.tensor(parts["fwd_obj_score"])
    expected = float(F.binary_cross_entropy_with_logits(scores, torch.ones(2)))
    assert parts["obj"] == pytest.approx(expected, rel=1e-3)
    assert float(loss) == pytest.approx(float(plain) + expected, rel=1e-3)
    assert parts["obj_weight"] == 1.0
    # a tracker without scores cannot carry the term
    with pytest.raises(ValueError, match="object_score_logits"):
        oc_ccl_step(FakeCycleTracker(), x0, xu, y0, obj_weight=1.0)
    # the plain stub records nothing and is byte-stable
    _, _, p0 = oc_ccl_step(FakeCycleTracker(), x0, xu, y0)
    assert "fwd_obj_score" not in p0


def test_sam2_tracker_wrap_captures_the_head_score_once_per_model():
    """The wrapper is instance-level, applied once, reports to the current owner, and
    surfaces the score `track_step` discards."""
    from occl.finetune.tracker import DifferentiableSAM2Tracker

    class _Head(torch.nn.Module):
        def _forward_sam_heads(self, *a, **k):
            return ("m", "M", "iou", "low", "high", "ptr", torch.tensor([[3.0], [-2.0]]))

        def _use_mask_as_output(self, *a, **k):
            return ("m", "M", "iou", "low", "high", "ptr", torch.tensor([[10.0]]))

    m = _Head()
    t1 = DifferentiableSAM2Tracker(m)
    t2 = DifferentiableSAM2Tracker(m)
    assert m._occl_obj_capture and m._occl_tracker is t2
    m._forward_sam_heads()
    assert t2.last_obj_score.tolist() == [[3.0], [-2.0]] and t1.last_obj_score is None
    m._use_mask_as_output()
    assert t2.last_obj_score.tolist() == [[10.0]]
    # no double wrapping: a third tracker sees one wrapper layer (the closure chain is 1)
    inner = m._forward_sam_heads
    DifferentiableSAM2Tracker(m)
    assert m._forward_sam_heads is inner


# ── S17d: the soft sentinel ────────────────────────────────────────────────


class _SentinelHead(torch.nn.Module):
    """A `_forward_sam_heads` with SAM 2's exact gate semantics: the decoder's raw masks
    and score, then the hard −1024 `torch.where` and the hard pointer mix when
    `pred_obj_scores` is on. `_use_mask_as_output` calls the heads inside (the pointer),
    as sam2_base does."""

    NO_OBJ = -1024.0

    def __init__(self):
        super().__init__()
        self.pred_obj_scores = True
        self.fixed_no_obj_ptr = True
        self.no_obj_ptr = torch.nn.Parameter(torch.full((1, 4), 7.0))
        self.calls = []
        self.raw = torch.tensor([[[[2.0, -1.0], [0.5, 3.0]]],
                                 [[[1.0, 1.5], [-2.0, 0.0]]]])            # (B=2, 1, 2, 2)
        self.score = torch.tensor([[3.0], [-0.75]])                       # (B, 1)
        self.ptr = torch.ones(2, 4)

    def _forward_sam_heads(self, *a, **k):
        self.calls.append(("heads", self.pred_obj_scores))
        low = self.raw.clone()
        ptr = self.ptr.clone()
        if self.pred_obj_scores:
            appearing = self.score > 0
            low = torch.where(appearing[:, None, None], low, self.NO_OBJ)
            lam = appearing.float()
            if self.fixed_no_obj_ptr:
                ptr = lam * ptr
            ptr = ptr + (1 - lam) * self.no_obj_ptr
        high = torch.nn.functional.interpolate(low, scale_factor=2, mode="bilinear",
                                               align_corners=False)
        return (low, high, torch.ones(2, 1), low, high, ptr, self.score.clone())

    def _use_mask_as_output(self, *a, **k):
        self.calls.append(("init", self.pred_obj_scores))
        _, _, _, _, _, ptr, _ = self._forward_sam_heads()       # the wrapped one
        return ("m", "M", "iou", "low", "high", ptr, torch.tensor([[10.0], [-10.0]]))


def test_soft_sentinel_is_the_ungated_head_shifted_by_the_score_and_exact_where_hard():
    """Straight-through: the forward value is the hard gate with −K for −1024 on every
    mask output and SAM 2's hard pointer rule; the backward pass is the soft gate's
    (K·σ'(s) into the score); the same 7-tuple; K=None is byte-for-byte the hard gate;
    the flag is restored."""
    from occl.finetune.tracker import DifferentiableSAM2Tracker

    m = _SentinelHead()
    tr = DifferentiableSAM2Tracker(m)                       # hard by default
    low_h, high_h, _, _, _, ptr_h, s_h = m._forward_sam_heads()
    assert (low_h[1] == -1024.0).all() and (low_h[0] == m.raw[0]).all()
    assert (ptr_h[1] == 7.0).all() and (ptr_h[0] == 1.0).all()
    assert tr.last_obj_score.tolist() == [[3.0], [-0.75]]

    tr.soft_sentinel = 20.0
    low, high, ious, lo, hi, ptr, s = m._forward_sam_heads()
    assert m.pred_obj_scores is True                        # restored after the call
    assert m.calls[-1] == ("heads", False)                  # the head ran UNGATED
    # FORWARD VALUE = the hard gate with -K for -1024: the present channel is untouched,
    # the absent one is the decoder's real logits shifted by exactly K
    expect = m.raw.clone()
    expect[1] = m.raw[1] - 20.0
    assert torch.equal(low[0], m.raw[0]) and torch.allclose(low, expect) and torch.allclose(lo, expect)
    assert torch.allclose(high, torch.nn.functional.interpolate(
        expect, scale_factor=2, mode="bilinear", align_corners=False))
    assert (low[1] > -1024.0).all()
    # the pointer's forward value is SAM 2's hard rule too
    assert torch.allclose(ptr, ptr_h)
    assert torch.equal(s, m.score) and tr.last_obj_score.tolist() == [[3.0], [-0.75]]
    # BACKWARD = the soft gate's: the loss on either channel's masks reaches its score
    # with the surrogate's derivative K·σ'(score), and the pointer's with σ'(score)
    m.score.requires_grad_(True)
    low, *_, ptr, _ = m._forward_sam_heads()
    (low.sum() + ptr.sum()).backward()
    sig = torch.sigmoid(m.score.detach())
    d_mask = 20.0 * sig * (1 - sig) * 4                    # 4 pixels, d(-shift)/ds = K·σ'(s)
    d_ptr = (sig * (1 - sig) * (m.ptr - m.no_obj_ptr.detach())).sum(1, keepdim=True)
    assert torch.allclose(m.score.grad, d_mask + d_ptr, atol=1e-5), m.score.grad
    assert float(m.score.grad[1]) > 0
    m.score.requires_grad_(False)
    m.score.grad = None


def test_soft_sentinel_leaves_init_frames_on_sam2s_own_rule_and_survives_an_error():
    from occl.finetune.tracker import DifferentiableSAM2Tracker

    m = _SentinelHead()
    tr = DifferentiableSAM2Tracker(m, soft_sentinel=20.0)
    out = m._use_mask_as_output()
    # the inner heads call ran GATED (pred_obj_scores stayed True): the init frame's
    # pointer is SAM 2's hard rule, and the init score is the hard ±10
    assert m.calls == [("init", True), ("heads", True)]
    assert (out[5][1] == 7.0).all() and tr.last_obj_score.tolist() == [[10.0], [-10.0]]
    assert tr._in_mask_as_output is False
    # the gate flag survives an exception inside the head
    m._forward_sam_heads_orig = m._forward_sam_heads

    class _Boom(_SentinelHead):
        def _forward_sam_heads(self, *a, **k):
            raise RuntimeError("boom")

    b = _Boom()
    DifferentiableSAM2Tracker(b, soft_sentinel=20.0)
    with pytest.raises(RuntimeError):
        b._forward_sam_heads()
    assert b.pred_obj_scores is True
    # a model without the score head is left alone under K
    c = _SentinelHead(); c.pred_obj_scores = False
    DifferentiableSAM2Tracker(c, soft_sentinel=20.0)
    low, *_ = c._forward_sam_heads()
    assert torch.equal(low, c.raw) and c.calls[-1] == ("heads", False)


def test_soft_sentinel_rides_the_recipe_as_a_per_arm_gate():
    from occl.finetune.probe import (PER_ARM_KEYS, base_arm_label, replay_subset,
                                     tracker_kwargs_from_recipe)

    r = recipe_from_lora_meta(_meta(soft_sentinel=20))
    assert r["soft_sentinel"] == 20.0 and isinstance(r["soft_sentinel"], float)
    assert recipe_from_lora_meta(_meta())["soft_sentinel"] is None
    assert CARRIER_RECIPE["soft_sentinel"] is None
    assert "soft_sentinel" in PER_ARM_KEYS and "soft_sentinel" not in replay_subset(r)
    assert tracker_kwargs_from_recipe(r) == {"soft_sentinel": 20.0}
    assert "soft_sentinel" not in oc_ccl_kwargs_from_recipe(r)      # a tracker knob
    assert base_arm_label("base", None) == "base"
    assert base_arm_label("base", 20.0) == "base_soft20"
    assert base_arm_label("base", 12.5) == "base_soft12.5"


def test_make_tracker_threads_the_gate_and_sam3_refuses_it():
    from occl.finetune.train import _make_tracker

    m = _SentinelHead()
    assert _make_tracker(m, "sam2", soft_sentinel=20.0).soft_sentinel == 20.0
    assert _make_tracker(m, "sam2").soft_sentinel is None
    with pytest.raises(ValueError, match="SAM 2 knob"):
        _make_tracker(m, "sam3", soft_sentinel=20.0)


# ── S18: the cycle's information bottleneck ─────────────────────────────────


def test_ste_prompt_is_the_binary_value_with_the_reparam_gradient():
    """`ste_threshold` = the release's VALUE (bit-identical to `hard_threshold`) with
    `logit_reparam`'s GRADIENT (1/20 into every logit); `hard_threshold` has no graph."""
    from occl.finetune.cycle import logits_to_prompt

    z = torch.randn(2, 1, 8, 8, requires_grad=True) * 15
    z.retain_grad()
    ste = logits_to_prompt(z, "ste_threshold")
    assert torch.equal(ste.detach(), logits_to_prompt(z.detach(), "hard_threshold"))
    assert ste.requires_grad and logits_to_prompt(z.detach(), "hard_threshold").grad_fn is None
    (ste * torch.arange(64.0).view(1, 1, 8, 8)).sum().backward()
    ref = torch.zeros_like(z)
    ref.copy_(torch.arange(64.0).view(1, 1, 8, 8) / 20.0)
    assert torch.allclose(z.grad, ref)


def test_ste_cycle_closes_on_the_argmax_mask_and_still_trains_the_forward_leg():
    """Under the STE the closing leg sees the binary forward mask (== the hard replay,
    == what the evaluator scores) and the forward probe still receives gradient."""
    from occl.finetune.cycle import gradient_report

    x0, xu, m1, m2 = _pair(seed=31)
    y0 = torch.cat([m1, m2])
    tr = FakeCycleTracker()
    cap: dict = {"hard_replay": True}
    loss_ste, cls_ste, _ = oc_ccl_step(tr, x0, xu, y0, prompt_mode="ste_threshold", capture=cap)
    loss_hard, cls_hard, _ = oc_ccl_step(tr, x0, xu, y0, prompt_mode="hard_threshold")
    assert torch.equal(cls_ste.detach(), cls_hard.detach()) and float(loss_ste) == float(loss_hard)
    assert torch.equal(cap["closing_hard"], cls_hard.detach())
    # (the stub's hard-threshold closing has no graph at all, so `gradient_report` under
    # it needs a real model — the s17ft gate pins its exact 0.0 there)
    rep = {m: gradient_report(tr, x0, xu, y0, m) for m in ("logit_reparam", "ste_threshold")}
    assert rep["ste_threshold"]["forward_to_reverse"] > 0.0
    assert rep["ste_threshold"]["prompt_mode"] == "ste_threshold"
    assert rep["ste_threshold"]["per_channel_query_found"] == rep["logit_reparam"]["per_channel_query_found"]


def test_hard_replay_is_the_closing_leg_on_the_binarised_forward_logits_and_off_by_default():
    """`capture["hard_replay"]` reruns legs 3-4 with the forward logits thresholded, on the
    same weights; the continuous closing and the loss are untouched, and without the
    flag the tracker is called exactly as before."""
    from tests.test_finetune import _CountingTracker

    x0, xu, m1, m2 = _pair(seed=33)
    y0 = torch.cat([m1, m2])
    tr = FakeCycleTracker()
    plain = oc_ccl_step(tr, x0, xu, y0)
    cap: dict = {"hard_replay": True}
    loss, closing, parts = oc_ccl_step(tr, x0, xu, y0, capture=cap)
    assert float(loss) == float(plain[0]) and torch.equal(closing, plain[1])
    fwd = cap["forward"]
    expected_hard = ((fwd > 0).float() * 20.0 - 10.0) * 0.5      # stub: init then propagate
    assert torch.equal(cap["closing_hard"], expected_hard)
    assert not cap["closing_hard"].requires_grad
    assert "cls_obj_hard" in cap
    res = cycle_probe(tr, x0, xu, y0, hard_replay=True)
    assert np.array_equal(res.closing_hard, expected_hard.numpy()[:, 0])
    assert cycle_probe(tr, x0, xu, y0).closing_hard is None
    ta, tb = _CountingTracker(), _CountingTracker()
    oc_ccl_step(ta, x0, xu, y0)
    oc_ccl_step(tb, x0, xu, y0, capture={})
    assert (ta.encodes, ta.tracks) == (tb.encodes, tb.tracks) == (2, 4)
    tc = _CountingTracker()
    oc_ccl_step(tc, x0, xu, y0, capture={"hard_replay": True})
    assert (tc.encodes, tc.tracks) == (2, 6)                       # legs 3-4 once more
    td = _CountingTracker()
    oc_ccl_step(td, x0, xu, y0, x0_close=torch.flip(x0, dims=(-1,)),
                y0_close=torch.flip(y0, dims=(-1,)))
    assert (td.encodes, td.tracks) == (3, 4)                       # one more encode


def test_perturbed_closing_end_encodes_x0_close_and_scores_against_y0_close():
    x0, xu, m1, m2 = _pair(seed=35)
    y0 = torch.cat([m1, m2])
    x0c = torch.flip(x0, dims=(-2, -1))
    y0c = torch.flip(y0, dims=(-2, -1))
    tr = FakeCycleTracker()
    l0, c0, p0 = oc_ccl_step(tr, x0, xu, y0)
    l1, c1, p1 = oc_ccl_step(tr, x0, xu, y0, x0_close=x0c, y0_close=y0c)
    # the stub's closing leg does not look at the frame, so the closing logits are the
    # same tensor — but the target is the flipped one, so the loss differs
    assert torch.equal(c0, c1) and float(l0) != float(l1) and p1["close_aug"] is True
    assert "close_aug" not in p0
    with pytest.raises(ValueError, match="come together"):
        oc_ccl_step(tr, x0, xu, y0, x0_close=x0c)
    with pytest.raises(ValueError, match="channels"):
        oc_ccl_step(tr, x0, xu, y0, x0_close=x0c, y0_close=y0c[:1])


def test_cyclepairs_closing_is_an_independent_rotation_of_the_reference():
    from occl.finetune.pairs import CyclePairs

    ds = FakePairDS()
    plain = CyclePairs(ds, "all", size=32, mask_mode="per_trait", require_cached=False)
    both = CyclePairs(ds, "all", size=32, mask_mode="per_trait", require_cached=False,
                      rotation_aug_deg=15.0, close_aug_deg=10.0)
    only_close = CyclePairs(ds, "all", size=32, mask_mode="per_trait", require_cached=False,
                            close_aug_deg=10.0)
    assert plain.pairs == both.pairs == only_close.pairs
    with pytest.raises(ValueError, match="close_aug_deg"):
        plain.closing(0)
    with pytest.raises(ValueError, match="close_aug_deg"):
        CyclePairs(ds, "all", size=32, require_cached=False, close_aug_deg=0.0)
    x0, xu, y0, _ = only_close[0]
    x0c, y0c, rec = only_close.closing(0)
    assert x0c.shape == x0.shape and y0c.shape == y0.shape
    assert set(rec) == {"close_drawn", "close_angle", "close_fallback"}
    assert abs(rec["close_drawn"]) <= 10.0
    # the opening end is untouched by the closing rotation; the closing end differs
    # from it whenever the drawn angle was applied
    assert torch.equal(x0, plain[0][0])
    if rec["close_angle"] != 0.0:
        assert not torch.equal(x0c, x0)
    # independent seeds: the closing angle is not the opening augmentation's angle
    _ = both[0]
    assert both.step_log[-1]["aug_drawn"] != both.closing(0)[2]["close_drawn"]
    # deterministic per (pair, epoch)
    assert only_close.closing(0)[2] == rec
    assert only_close.closing(0)[2]["close_drawn"] == \
        CyclePairs(ds, "all", size=32, mask_mode="per_trait", require_cached=False,
                   close_aug_deg=10.0).closing(0)[2]["close_drawn"]
    assert CyclePairs(ds, "all", size=32, mask_mode="per_trait", require_cached=False,
                      close_aug_deg=10.0, epoch=1).closing(0)[2]["close_drawn"] != rec["close_drawn"]


def test_s18_axes_ride_the_recipe_as_per_arm_keys():
    from occl.finetune.probe import (LEG_KEYS, PER_ARM_KEYS, REPLAY_PROMPT_MODES,
                                     base_arm_label, base_variant_of, close_aug_deg,
                                     own_loss, own_prompt_is_hard, replay_subset)

    ste = recipe_from_lora_meta(_meta(prompt_mode="ste_threshold"))
    per = recipe_from_lora_meta(_meta(close_aug={"type": "rotation_x0_close", "max_deg": 10}))
    crown = recipe_from_lora_meta(_meta())
    assert "prompt_mode" not in LEG_KEYS and "close_aug" not in LEG_KEYS
    assert {"prompt_mode", "close_aug", "soft_sentinel", "lora_legs"} <= set(PER_ARM_KEYS)
    assert replay_subset(ste) == replay_subset(crown) == replay_subset(per)
    assert own_prompt_is_hard(ste) and not own_prompt_is_hard(crown)
    assert own_prompt_is_hard(recipe_from_lora_meta(_meta(prompt_mode="hard_threshold")))
    assert close_aug_deg(per) == 10.0 and close_aug_deg(crown) is None
    assert cyclepairs_kwargs(per)["close_aug_deg"] == 10.0
    assert base_variant_of(per) == (None, 10.0) and base_variant_of(crown) == (None, None)
    assert base_arm_label("base", None, 10.0) == "base_close10"
    assert base_arm_label("base", 20.0, 10.0) == "base_soft20_close10"
    assert base_arm_label("base", 20.0) == "base_soft20" and base_arm_label("base", None) == "base"
    assert REPLAY_PROMPT_MODES == {"continuous": "logit_reparam", "hard": "hard_threshold"}
    row = {"loss": 2.0, "loss_excl": 3.0, "loss_hard": 1.0, "loss_excl_hard": 1.5,
           "loss_objterm": 0.1}
    assert own_loss(row, "excl+obj10") == pytest.approx(4.0)
    assert own_loss(row, "excl+obj10", hard=True) == pytest.approx(2.5)
    assert own_loss(row, "plain", hard=True) == 1.0
    assert own_loss({"loss": 2.0}, "plain", hard=True) is None


def test_memory_ste_keeps_the_binary_value_and_takes_the_continuous_gradient():
    """The tracker's `memory_ste` wraps the memory encoder's mask input on the model
    instance: value = whatever the binary path computed, gradient = 20·σ′(l) (the
    reparam's memory-route gradient); inert outside the context; `ste_memory` in the
    step opens it around the reverse init and refuses a tracker without it."""
    from occl.finetune.cycle import PROMPT_MODES, logits_to_prompt
    from occl.finetune.tracker import DifferentiableSAM2Tracker

    class _Mem(torch.nn.Module):
        def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
            return {"vision_features": masks * 3.0, "vision_pos_enc": None}

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_encoder = _Mem()
            self.sigmoid_scale_for_mem_enc = 20.0
            self.pred_obj_scores = True

        def _forward_sam_heads(self, *a, **k):
            return (None,) * 7

        def _use_mask_as_output(self, *a, **k):
            return (None,) * 7

    m = _M()
    tr = DifferentiableSAM2Tracker(m)
    lg = torch.randn(2, 1, 4, 4, requires_grad=True) * 6
    lg.retain_grad()
    binary_in = (torch.sigmoid((lg.detach() > 0).float() * 20.0 - 10.0) * 20.0 - 10.0)   # SAM 2's path
    out = m.memory_encoder(None, binary_in, skip_mask_sigmoid=True)["vision_features"]
    assert not out.requires_grad                                       # inert outside
    with tr.memory_ste(lg):
        out = m.memory_encoder(None, binary_in, skip_mask_sigmoid=True)["vision_features"]
    assert torch.equal(out.detach(), binary_in * 3.0)                  # the binary value
    out.sum().backward()
    assert torch.allclose(lg.grad, 3.0 * 20.0 * torch.sigmoid(lg) * (1 - torch.sigmoid(lg)))
    assert tr._memory_ste_logits is None
    with pytest.raises(ValueError, match="memory mask input"):
        with tr.memory_ste(lg[:1]):
            m.memory_encoder(None, binary_in, skip_mask_sigmoid=True)
    # a second tracker on the same model does not stack wrappers
    DifferentiableSAM2Tracker(m)
    assert m.memory_encoder.forward.__name__ == "wrapped"
    # the step: same prompt tensor as ste_threshold, a tracker without the hook refuses
    assert "ste_memory" in PROMPT_MODES
    z = torch.randn(1, 1, 8, 8)
    assert torch.equal(logits_to_prompt(z, "ste_memory"), logits_to_prompt(z, "ste_threshold"))
    x0, xu, m1, m2 = _pair(seed=41)
    with pytest.raises(ValueError, match="memory_ste"):
        oc_ccl_step(FakeCycleTracker(), x0, xu, torch.cat([m1, m2]), prompt_mode="ste_memory")

    class _MemTracker(FakeCycleTracker):
        opened = []

        def memory_ste(self, logits):
            from contextlib import contextmanager

            @contextmanager
            def _c():
                self.opened.append(logits)
                yield
            return _c()

    t = _MemTracker()
    l_mem, c_mem, _ = oc_ccl_step(t, x0, xu, torch.cat([m1, m2]), prompt_mode="ste_memory")
    l_ste, c_ste, _ = oc_ccl_step(t, x0, xu, torch.cat([m1, m2]), prompt_mode="ste_threshold")
    assert torch.equal(c_mem.detach(), c_ste.detach()) and float(l_mem) == float(l_ste)
    assert len(t.opened) == 1 and t.opened[0].shape == c_mem.shape     # once, the reverse init
