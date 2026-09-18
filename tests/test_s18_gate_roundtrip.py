"""S18 gate <-> combine schema on synthetic bundles (the S17 roundtrip's pattern).

The S18 additions: every probe row carries the closing leg under the BINARISED prompt
(`*_hard` columns), an arm trained with `ste_threshold` reads its own closing / replay
proof / disqualifier off those columns, an arm trained with `close_aug` pairs against
`base_close10`, and the combine's `stego` block is what the s18 gate requires. The s18ft
gate accepts the established recipe moved along the S18 axes only.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from occl.finetune.probe import FORMULAS
from tests.test_s17_gate_roundtrip import (ARCH_SCOPES, N_PAIRS, _row, _run, _zip,
                                           eval_cell, ft_record, probe_record)

OBJ10 = {"weight": 10.0, "target": "present_for_prompted"}
CLOSE10 = {"type": "rotation_x0_close", "max_deg": 10.0}
#: arm -> (occl overrides, training overrides, finetune overrides)
S18_ARMS = {
    "crown": ({}, {}, {}),
    "crown_ste": ({"prompt_mode": "ste_threshold"}, {"prompt_mode": "ste_threshold"}, {}),
    "crown_stemem": ({"prompt_mode": "ste_memory"}, {"prompt_mode": "ste_memory"}, {}),
    "crown_perturb": ({"close_aug": CLOSE10},
                      {"close_aug": {**CLOSE10, "angles": [3.0] * 12,
                                     "empty_channel_fallbacks": 0}}, {}),
    "crown_all": ({"lora_legs": None}, {"lora_legs": None}, {}),
    "crown_hot": ({}, {"lr": 3e-5, "epochs": 3}, {"lr": 3e-5, "epochs": 3}),
}


def crown_ft(ds, arm, sha, delta):
    """A fine-tune record on the established recipe, moved along one S18 axis."""
    occl_over, train_over, ft_over = S18_ARMS[arm]
    ft = ft_record(ds, lora_sha=sha, delta_norm=delta)
    ft["finetune"]["lora_path"] = f"/x/adapters/beetle_{arm}.pt"
    ft["finetune"]["lora"]["scopes"] = ARCH_SCOPES
    ft["training"]["lora"]["paths"] = ["memory_encoder.a", "memory_attention.q",
                                       "sam_mask_decoder.b"]
    ft["training"]["lora"]["modules"] = 94
    ft["occl"] |= {"lora_legs": "forward", "objness": OBJ10, "excl": {"mode": "zerocat"},
                   "soft_sentinel": None, "close_aug": None, "recipe": "sam2_crown"}
    ft["training"] |= {"lora_legs": "forward", "objness": {"weight": 10.0, "steps": 12},
                       "excl": {"mode": "zerocat"}, "soft_sentinel": None, "close_aug": None}
    ft["diagnostics"] |= {
        "rel_grad_logit_reparam": 25.0, "rel_grad_hard_threshold": 0.0,
        "rel_grad_ste_threshold": 0.12, "rel_grad_ste_memory": 22.0,
        "forward_leg_rel_grad": 25.0,
        "grad_ste_threshold": {"per_channel_forward_to_reverse": [0.1, 0.1, 0.0],
                               "per_channel_query_found": [True, True, False]},
        "grad_ste_memory": {"per_channel_forward_to_reverse": [18.0, 27.0, 0.0],
                            "per_channel_query_found": [True, True, False]}}
    ft["occl"].update(occl_over)
    ft["training"].update(train_over)
    ft["finetune"].update(ft_over)
    if ft_over.get("epochs"):
        n = ft["training"]["epochs"] * ft["training"]["steps_per_epoch"]
        ft["training"]["steps"] = n
        ft["training"]["losses"] = (ft["training"]["losses"] * 3)[:n]
        ft["training"]["mask_channels"] = {"5": n}
    return ft


def _hard_cols(r, gap, loss_scale=0.9):
    """Add the S18 columns to a probe row: the closing leg under the binary prompt sits
    `gap` below the continuous one, its losses `loss_scale` of the continuous ones."""
    r["cls_miou_hard"] = max(float(r["cls_miou"]) - gap, 0.0)
    r["cls_iou_binary_hard"] = 0.88
    r["cls_per_trait_hard"] = r["cls_per_trait"]
    r["cls_claimed_frac_hard"] = 0.3
    r["cls_obj_hard"] = ""
    r["loss_hard"] = float(r["loss"]) * loss_scale
    r["bce_hard"] = r["loss_hard"] * 0.7
    r["dice_hard"] = r["loss_hard"] * 0.3
    for f in FORMULAS:
        if f != "plain":
            r[f"loss_{f}_hard"] = float(r[f"loss_{f}"]) * loss_scale
    r.setdefault("prompt_mode", "logit_reparam")
    r.setdefault("close_aug", "")
    r.setdefault("close_angle", "")
    r["cls_map_hard"] = r["cls_map"].replace("/cls/", "/cls_hard/")
    return r


def s18_probe(ds, fts, *, stego_base=0.01, stego=None):
    """base + base_close10 + the S18 arms. `stego[arm]` is the arm's continuous − hard
    closing gap (the channel); the base's is `stego_base`."""
    stego = stego or {"crown": 0.04, "crown_ste": 0.01, "crown_stemem": 0.01,
                      "crown_perturb": 0.04, "crown_all": 0.05, "crown_hot": 0.08}
    n = N_PAIRS[ds]
    rng = np.random.default_rng(8)
    rows, lv = [], []
    for i in range(n):
        fb = float(np.clip(rng.normal(0.65, 0.1), 0.05, 0.95))
        cb = float(np.clip(rng.normal(0.70, 0.1), 0.05, 0.95))
        lb = float(np.clip(rng.normal(0.9, 0.4), 0.05, None))
        lv.append((fb, cb, lb))
        rows.append(_hard_cols(_row(ds, "base", i, fb, cb, lb, found_all=(i % 4 != 0)), stego_base))
        r = _hard_cols(_row(ds, "base_close10", i, fb, cb - 0.01, lb * 1.05, found_all=(i % 4 != 0)),
                       stego_base)
        r["close_aug"], r["close_angle"] = 10.0, 3.0
        rows.append(r)
    arms = {"base": {"path": None, "delta_norm": 0.0, "lora_sha": None, "soft_sentinel": None,
                     "close_aug": None, "prompt_mode": "logit_reparam"},
            "base_close10": {"path": None, "delta_norm": 0.0, "lora_sha": None,
                             "soft_sentinel": None, "close_aug": 10.0,
                             "prompt_mode": "logit_reparam"}}
    parity, base_of = {}, {}
    for arm, ft in fts.items():
        pm = ft["occl"]["prompt_mode"]
        cdeg = (ft["occl"].get("close_aug") or {}).get("max_deg")
        base_arm = "base_close10" if cdeg else "base"
        base_of[arm] = base_arm
        arms[arm] = {"path": ft["finetune"]["lora_path"], "lora_sha": ft["finetune"]["lora_sha"],
                     "delta_norm": ft["training"]["delta_norm"], "mismatches": 0, "modules": 94,
                     "r": 32, "alpha": 64.0, "objective": "excl+obj10",
                     "lora_legs": ft["occl"].get("lora_legs") or "all",
                     "soft_sentinel": None, "close_aug": cdeg, "prompt_mode": pm,
                     "scopes": ARCH_SCOPES}
        for i, (fb, cb, lb) in enumerate(lv):
            r = _row(ds, arm, i, min(fb + 0.02 + rng.normal(0, 0.01), 1.0),
                     min(cb + 0.04 + rng.normal(0, 0.01), 1.0),
                     max(lb * 0.7 + rng.normal(0, 0.02), 0.01), found_all=(i % 4 != 0))
            _hard_cols(r, stego[arm])
            r["lora_legs"] = arms[arm]["lora_legs"]
            r["prompt_mode"] = pm
            if cdeg:
                r["close_aug"], r["close_angle"] = cdeg, 3.0
            rows.append(r)
        hard = pm in ("ste_threshold", "ste_memory", "hard_threshold")
        b0 = next(r for r in rows if r["arm"] == base_arm and r["train_step"] == 0)
        first = float(b0["loss_excl_hard" if hard else "loss_excl"]) + 10.0 * float(b0["loss_objterm"])
        ft["training"]["loss_first"] = first
        ft["training"]["losses"][0] = first
        parity[arm] = {"objective": "excl+obj10", "loss_first_train": first,
                       "loss_first_probe_base": first, "loss_first_rel_err": 0.0,
                       "base_arm": base_arm, "column": "hard" if hard else "continuous",
                       "lora_sha": ft["finetune"]["lora_sha"],
                       "delta_norm": ft["training"]["delta_norm"]}
    rec = probe_record(ds, training=fts["crown"]["training"])
    rec["pairs"] = rows
    rec["arms"] = arms
    rec["arm_order"] = ["base", "base_close10", *fts]
    rec["base_arm"], rec["base_arms"] = "base", ["base", "base_close10"]
    rec["base_of"] = base_of
    rec["arm_prompt_mode"] = {a: v["prompt_mode"] for a, v in arms.items()}
    rec["recipe"]["lora"]["scopes"] = ARCH_SCOPES
    rec["recipe"] |= {"excl": {"mode": "zerocat"}, "objness": OBJ10, "lora_legs": "forward",
                      "soft_sentinel": None, "close_aug": None}
    rec["replay"]["parity"] = parity
    rec["replay"]["loss_first_rel_err"] = 0.0
    rec["replay"]["prompt_modes"] = {"continuous": "logit_reparam", "hard": "hard_threshold"}
    return rec


def _specs(tmp_path, fts, probe, cells=True):
    specs = [f"probe={_zip(tmp_path / 'probe.zip', probe, maps=True)}"]
    specs += [f"ft:{arm}={_zip(tmp_path / f'ft_{arm}.zip', ft)}" for arm, ft in fts.items()]
    if cells:
        levels = {"base": 0.632, "crown": 0.637, "crown_ste": 0.636, "crown_stemem": 0.640,
                  "crown_perturb": 0.637, "crown_all": 0.635, "crown_hot": 0.634}
        for name, lvl in levels.items():
            for tta in (False, True):
                cell = name + ("_tta" if tta else "")
                lp = None if name == "base" else fts[name]["finetune"]["lora_path"]
                dl = None if name == "base" else fts[name]["training"]["delta_norm"]
                specs.append(f"eval:{cell}={_zip(tmp_path / f'eval_{cell}.zip', eval_cell('beetle', cell, lvl + (0.02 if tta else 0), tta, lp, dl))}")
    return specs


def test_s18_combine_reads_each_arm_under_its_own_prompt_and_the_gate_passes(tmp_path):
    ds = "beetle"
    fts = {arm: crown_ft(ds, arm, sha=f"sha_{arm}", delta=0.7 + 0.01 * k)
           for k, arm in enumerate(S18_ARMS)}
    probe = s18_probe(ds, fts)
    specs = _specs(tmp_path, fts, probe)
    out = tmp_path / "s18_beetle.zip"
    p = _run(["scripts/combine_s17.py", "--stage", "s18", "--out", str(out),
              "--control", "crown", *specs])
    assert "stego gap (closing continuous − hard, own prompt hard)" in p.stdout, p.stdout
    assert "forward on the good prompt bin" in p.stdout
    assert "gate effect at base weights, base_close10 (K=None, close_aug=10.0) vs base" in p.stdout
    with zipfile.ZipFile(out) as z:
        rec = json.loads(z.read("record.json"))
        lines = z.read("deltas.csv").decode().splitlines()
    assert rec["adapter_arms"] == list(S18_ARMS) and rec["base_arms"] == ["base", "base_close10"]
    assert rec["base_of"]["crown_perturb"] == "base_close10" and rec["base_of"]["crown_ste"] == "base"
    r_ste, r_crown = rec["results"]["crown_ste"], rec["results"]["crown"]
    assert r_ste["own_prompt"] == "hard" and r_ste["closing"]["prompt"] == "hard"
    assert r_crown["own_prompt"] == "continuous" and r_crown["closing"]["prompt"] == "continuous"
    assert rec["provenance"]["crown_ste"]["own_prompt"] == "hard"
    # the ste arm's own closing IS its hard column (base hard vs ft hard)
    assert r_ste["closing"]["base_mean"] == pytest.approx(r_ste["closing_hard"]["base_mean"])
    assert r_ste["closing"]["ft_mean"] == pytest.approx(r_ste["closing_hard"]["ft_mean"])
    assert r_crown["closing"]["base_mean"] == pytest.approx(r_crown["closing_continuous"]["base_mean"])
    # the steganography read: the crown opened a channel (+0.03 over the base's 0.01),
    # the ste arm did not, the hot corner more
    for arm, gap in (("crown", 0.03), ("crown_ste", 0.0), ("crown_stemem", 0.0), ("crown_hot", 0.07)):
        st = rec["results"][arm]["stego"]
        assert st["n"] == N_PAIRS[ds]
        assert st["base_gap_mean"] == pytest.approx(0.01, abs=1e-6)
        assert st["delta"]["mean_delta"] == pytest.approx(gap, abs=1e-6)
        assert st["delta"]["significant"] == (gap > 0)
    # its own objective is read off the hard columns for the ste arm (0.9x the continuous)
    assert r_ste["loss"]["base_mean"] == pytest.approx(0.9 * r_crown["loss"]["base_mean"]
                                                        + 0.1 * 10.0 * 0.07, rel=1e-6)
    assert r_ste["loss_dice"]["delta"]["significant"] and r_ste["loss_dice"]["delta"]["mean_delta"] < 0
    assert r_crown["forward"]["good_prompt_bin"]["threshold"] == 0.75
    assert rec["vs_control"]["crown_perturb"]["same_gate"] is False
    assert rec["vs_control"]["crown_ste"]["same_gate"] is True
    assert rec["vs_control"]["crown_ste"]["same_prompt"] is False
    assert rec["results"]["crown_stemem"]["own_prompt"] == "hard"
    assert rec["provenance"]["crown_stemem"]["prompt_mode"] == "ste_memory"
    assert rec["gate_effect"]["base_close10"]["close_aug"] == 10.0
    assert lines[0].startswith("arm,objective,base_arm,train_step")
    assert lines[0].endswith("base_stego_gap,ft_stego_gap,d_stego_gap")
    assert any(l.startswith("crown_perturb,excl+obj10,base_close10,") for l in lines[1:])
    ev = rec["eval"]
    assert {"crown_ste_vs_crown", "crown_ste_tta_vs_crown_tta", "crown_hot_vs_base",
            "crown_perturb_tta_vs_crown_tta"} <= set(ev["pairings"])
    merged = tmp_path / "s18.zip"
    _run(["scripts/combine_s17.py", "--stage", "s18", "--merge", "--out", str(merged), f"beetle={out}"])
    with zipfile.ZipFile(merged) as z:
        assert "beetle/maps/base/fwd/0000.png" in z.namelist()
    res = _run(["scripts/check_gate.py", "s18", "--bundle", str(merged), "--no-file"])
    assert "GATE S18: PASS" in res.stdout, res.stdout
    assert "THE STEGANOGRAPHY READ" in res.stdout
    assert "under its OWN prompt value" in res.stdout
    assert "crown_ste/excl+obj10: probe" in res.stdout
    assert "STEGO GAP" in res.stdout                         # the bands line
    assert "pointer route only — recorded" in res.stdout

    # the plain-STE arm's objective NOT going down is recorded, not gated; the crown's is
    rec_ste = json.loads(json.dumps(probe))
    for r in rec_ste["pairs"]:
        if r["arm"] == "crown_ste":
            r["loss_nobce_hard"] = float(r["loss_nobce_hard"]) * 3.0
    specs_ste = [f"probe={_zip(tmp_path / 'probe_ste.zip', rec_ste, maps=True)}"] + specs[1:]
    out_ste = tmp_path / "s18_ste.zip"
    _run(["scripts/combine_s17.py", "--stage", "s18", "--out", str(out_ste), "--control", "crown", *specs_ste])
    res = _run(["scripts/check_gate.py", "s18", "--bundle", str(out_ste), "--no-file"])
    assert "GATE S18: PASS" in res.stdout, res.stdout
    rec_crown = json.loads(json.dumps(probe))
    for r in rec_crown["pairs"]:
        if r["arm"] == "crown":
            r["loss_nobce"] = float(r["loss_nobce"]) * 3.0
    specs_c = [f"probe={_zip(tmp_path / 'probe_c.zip', rec_crown, maps=True)}"] + specs[1:]
    out_c = tmp_path / "s18_c.zip"
    _run(["scripts/combine_s17.py", "--stage", "s18", "--out", str(out_c), "--control", "crown", *specs_c])
    res = _run(["scripts/check_gate.py", "s18", "--bundle", str(out_c), "--no-file"], ok=False)
    assert "[FAIL]  DISQUALIFIER" in res.stdout, res.stdout

    # an ste arm whose replay proof was read off the continuous column: refused
    bad = json.loads(json.dumps(probe))
    bad["replay"]["parity"]["crown_ste"]["column"] = "continuous"
    specs_bad = [f"probe={_zip(tmp_path / 'probe_bad.zip', bad, maps=True)}"] + specs[1:]
    out_bad = tmp_path / "s18_bad.zip"
    _run(["scripts/combine_s17.py", "--stage", "s18", "--out", str(out_bad), "--control", "crown", *specs_bad])
    res = _run(["scripts/check_gate.py", "s18", "--bundle", str(out_bad), "--no-file"], ok=False)
    assert "[FAIL]  every arm's closing leg, replay proof and disqualifier read under its OWN" in res.stdout, res.stdout
    # a perturb arm paired against the plain base: refused by the combine
    bad2 = json.loads(json.dumps(probe))
    bad2["base_of"]["crown_perturb"] = "base"
    bp = _run(["scripts/combine_s17.py", "--stage", "s18", "--out", str(tmp_path / "bad2.zip"),
               "--control", "crown", f"probe={_zip(tmp_path / 'probe_bad2.zip', bad2, maps=True)}",
               *specs[1:]], ok=False)
    assert "different gates or closing ends" in bp.stdout + bp.stderr
    # rows without the hard columns (a pre-S18 probe): the s18 gate refuses, s17's passes
    old = json.loads(json.dumps(probe))
    for r in old["pairs"]:
        for k in list(r):
            if k.endswith("_hard"):
                del r[k]
    old["arm_order"] = ["base", "crown"]
    old["arms"] = {a: old["arms"][a] for a in ("base", "crown")}
    old["base_arms"], old["base_of"] = ["base"], {"crown": "base"}
    old["pairs"] = [r for r in old["pairs"] if r["arm"] in ("base", "crown")]
    old["replay"]["parity"] = {"crown": old["replay"]["parity"]["crown"]}
    old_specs = [f"probe={_zip(tmp_path / 'probe_old.zip', old, maps=True)}",
                 f"ft:crown={_zip(tmp_path / 'ft_crown_old.zip', fts['crown'])}"]
    out_old = tmp_path / "s18_old.zip"
    _run(["scripts/combine_s17.py", "--stage", "s18", "--out", str(out_old), *old_specs])
    res = _run(["scripts/check_gate.py", "s18", "--bundle", str(out_old), "--no-file"], ok=False)
    assert "[FAIL]  THE STEGANOGRAPHY READ" in res.stdout, res.stdout


def test_s18ft_accepts_the_crown_moved_along_s18_axes_only(tmp_path):
    ds = "beetle"
    for arm in S18_ARMS:
        ft = crown_ft(ds, arm, sha=f"sha_{arm}", delta=0.8)
        res = _run(["scripts/check_gate.py", "s18ft", "--bundle",
                    _zip(tmp_path / f"s18ft_{arm}.zip", ft), "--no-file"])
        assert "GATE S18FT: PASS" in res.stdout, (arm, res.stdout)
        assert ("the crown exactly" if arm == "crown" else "crown + ") in res.stdout
        assert "carry gradient to the forward leg" in res.stdout
    # off-axis values, an S17-era knob, and a declared-not-trained prompt: FAIL
    for name, occl_over, train_over, ft_over in (
            ("close27", {"close_aug": {"type": "rotation_x0_close", "max_deg": 27.8}},
             {"close_aug": {"type": "rotation_x0_close", "max_deg": 27.8, "angles": [], "empty_channel_fallbacks": 0}}, {}),
            ("hardprompt", {"prompt_mode": "hard_threshold"}, {"prompt_mode": "hard_threshold"}, {}),
            ("w3", {"objness": {"weight": 3.0, "target": "present_for_prompted"}}, {"objness": {"weight": 3.0, "steps": 12}}, {}),
            ("soft", {"soft_sentinel": 20.0}, {"soft_sentinel": 20.0}, {}),
            ("lying", {"prompt_mode": "ste_threshold"}, {}, {})):
        ft = crown_ft(ds, "crown", sha="s", delta=0.8)
        ft["occl"].update(occl_over)
        ft["training"].update(train_over)
        ft["finetune"].update(ft_over)
        res = _run(["scripts/check_gate.py", "s18ft", "--bundle",
                    _zip(tmp_path / f"s18ft_{name}.zip", ft), "--no-file"], ok=False)
        assert "GATE S18FT: FAIL" in res.stdout, (name, res.stdout)
        assert "[FAIL]  the established SAM 2 recipe" in res.stdout, (name, res.stdout)
    # the S11 scopes are not the crown either
    ft = crown_ft(ds, "crown", sha="s", delta=0.8)
    ft["finetune"]["lora"]["scopes"] = ["memory_encoder", "sam_mask_decoder"]
    ft["training"]["lora"]["paths"] = ["memory_encoder.a", "sam_mask_decoder.b"]
    res = _run(["scripts/check_gate.py", "s18ft", "--bundle", _zip(tmp_path / "s11scopes.zip", ft), "--no-file"], ok=False)
    assert "[FAIL]  the established SAM 2 recipe" in res.stdout
    # no STE diagnostic recorded (an older run_finetune), or a memory STE whose gradient
    # collapsed like the plain one's: FAIL on the gradient predicate
    ft = crown_ft(ds, "crown_ste", sha="s", delta=0.8)
    del ft["diagnostics"]["grad_ste_memory"]
    res = _run(["scripts/check_gate.py", "s18ft", "--bundle", _zip(tmp_path / "noste.zip", ft), "--no-file"], ok=False)
    assert "[FAIL]  the straight-through prompts carry gradient" in res.stdout
    ft = crown_ft(ds, "crown_stemem", sha="s", delta=0.8)
    ft["diagnostics"]["grad_ste_memory"]["per_channel_forward_to_reverse"] = [0.1, 0.1, 0.0]
    res = _run(["scripts/check_gate.py", "s18ft", "--bundle", _zip(tmp_path / "weakmem.zip", ft), "--no-file"], ok=False)
    assert "[FAIL]  the straight-through prompts carry gradient" in res.stdout
    res = _run(["scripts/check_gate.py", "--list"])
    assert "s18ft" in res.stdout and "s18 " in res.stdout
