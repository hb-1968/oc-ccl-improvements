"""S17 gate <-> combine schema, exercised end to end on synthetic bundles.

The s14sec lesson (README): a gate and a combine that were never run against each other
failed every check on KeyErrors the first time a real bundle landed. So the probe record
(`scripts/probe_train_pairs.py`'s shape), the fine-tune record (`run_finetune.py`'s) and
the combine are driven here through `combine_s17.py` -> `--merge` -> `check_gate.py s17`
before any GPU-hour, including the refusals and the pre-registered disqualifier.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.finetune.probe import fmt_per_trait  # noqa: E402

N_PAIRS = {"beetle": 12, "butterfly_major": 20}
TRAITS = {"beetle": (1, 2, 3, 4, 5), "butterfly_major": (1, 2, 3, 4, 5, 6)}
LORA_SHA = "0123456789abcdef"
LORA_NAME = "beetle_sam2_lora_lr1e-05_ep1_per_trait_reset.pt"
DELTA_NORM = 0.7563


def _occl():
    return {"prompt_mode": "logit_reparam", "train_mode": False,
            "normalization": "imagenet", "memory_reset": True, "cycle_len": 2,
            "mask_mode": "per_trait", "reference": "derangement", "reference_id": None,
            "reference_run": None, "augmentation": None, "scnp": None, "excl": None,
            "xovl": None, "bshape": None, "margin": None, "found_mask": None, "bce_w": None,
            "objness": None, "lora_legs": None}


def _recipe():
    return {"mask_mode": "per_trait", "normalization": "imagenet",
            "prompt_mode": "logit_reparam", "memory_reset": True, "excl": None,
            "scnp": None, "xovl": None, "bshape": None, "margin": None,
            "augmentation": None, "reference": "derangement", "reference_id": None,
            "family": "sam2", "image_size": 1024, "mask_size": 1024,
            "epochs": 1, "lr": 1e-5, "method": "lora",
            "lora": {"r": 32, "alpha": 64.0, "scopes": ["memory_encoder", "sam_mask_decoder"]}}


def _losses(ds, seed=0):
    rng = np.random.default_rng(seed)
    n = N_PAIRS[ds]
    return [float(x) for x in np.clip(rng.normal(0.9, 0.5, n), 0.05, None)]


def ft_record(ds, *, losses=None, steps=None, lora_sha=LORA_SHA, delta_norm=DELTA_NORM,
              mask_channels=None):
    losses = losses if losses is not None else _losses(ds)
    n = N_PAIRS[ds]
    steps = steps if steps is not None else n
    hist = mask_channels or {str(len(TRAITS[ds])): steps}
    return {
        "stage": "s8b", "dataset": ds,
        "finetune": {"method": "lora", "epochs": 1, "lr": 1e-5, "family": "sam2",
                     "image_size": 1024, "mask_size": 1024,
                     "lora": {"r": 32, "alpha": 64.0,
                              "scopes": ["memory_encoder", "sam_mask_decoder"]},
                     "lora_path": f"/x/adapters/{LORA_NAME}", "lora_sha": lora_sha},
        "occl": _occl(),
        "model": {"family": "sam2", "image_size": 1024},
        "training": {"lr": 1e-5, "epochs": 1, "steps": steps, "group": None,
                     "steps_per_epoch": steps, "family": "sam2", "image_size": 1024,
                     "mask_size": 1024, "mask_mode": "per_trait", "scnp": None,
                     "excl": None, "xovl": None, "bshape": None, "margin": None,
                     "found_mask": None, "bce_w": None, "objness": None, "lora_legs": None,
                     "pairs": {"reference": "derangement", "reference_id": None},
                     "aug": None, "mask_channels": hist,
                     "train_images": n, "train_images_total": n,
                     "train_cache_fraction": 1.0,
                     "lora": {"r": 32, "alpha": 64.0, "modules": 54,
                              "paths": ["memory_encoder.a", "sam_mask_decoder.b"],
                              "trainable_params": 1_000_000},
                     "prompt_mode": "logit_reparam", "memory_reset": True,
                     "loss_first": losses[0], "loss_last": losses[-1],
                     "loss_mean": float(np.mean(losses)), "loss_per_epoch": [float(np.mean(losses))],
                     "losses": losses[:steps], "checkpoints": [], "method": "lora",
                     "train_mode": False, "normalization": "imagenet",
                     "delta_norm_before": 0.0, "delta_norm": delta_norm,
                     "peak_vram_gb": 3.0, "elapsed_s": 100.0},
        "diagnostics": {"delta_norm": delta_norm, "delta_norm_before": 0.0,
                        "roundtrip_mismatches": 0, "delta_norm_reloaded": delta_norm,
                        "grad_logit_reparam": {"per_channel_forward_to_reverse": [20.0, 30.0, 0.0],
                                               "per_channel_query_found": [True, True, False]},
                        "grad_hard_threshold": {"per_channel_forward_to_reverse": [0.0, 0.0, 0.0]}},
    }


def _row(ds, arm, i, fwd, cls, loss, found_all=True):
    traits = TRAITS[ds]
    per = {t: float(np.clip(fwd + 0.05 * ((t % 3) - 1), 0, 1)) for t in traits}
    found = [True] * len(traits) if found_all else [True] * (len(traits) - 1) + [False]
    return {"dataset": ds, "group": "all" if ds == "beetle" else "cyrbia",
            "train_step": i, "ref": f"r{i}", "query": f"q{i}",
            "class_ids": " ".join(str(t) for t in traits), "n_channels": len(traits),
            "arm": arm, "aug_angle": "",
            "fwd_miou": fwd, "fwd_miou_prompted": fwd, "fwd_iou_binary": 0.9,
            "fwd_n_gt_traits": len(traits), "fwd_n_prompted_present": len(traits),
            "fwd_per_trait": fmt_per_trait(per),
            "fwd_found": ";".join("1" if f else "0" for f in found),
            "fwd_found_count": int(sum(found)), "fwd_claimed_frac": 0.3,
            "cls_miou": cls, "cls_iou_binary": 0.9, "cls_per_trait": fmt_per_trait(per),
            "cls_claimed_frac": 0.3, "loss": loss, "bce": loss * 0.7, "dice": loss * 0.3,
            "ce": "", "dice_per_channel": "", "bce_per_channel": "",
            "loss_found_mask": loss * 0.4, "loss_nobce": loss * 0.3,
            "loss_found_mask_nobce": loss * 0.2, "loss_excl": loss * 0.6,
            "loss_objterm": 0.07, "fwd_fp_frac": 0.0, "fwd_n_absent_prompted": 0,
            "fwd_obj": "", "cls_obj": "", "lora_legs": "all",
            "fwd_map": f"maps/{arm}/fwd/{i:04d}.png",
            "cls_map": f"maps/{arm}/cls/{i:04d}.png"}


def probe_record(ds, *, fwd_gain=0.02, cls_gain=0.04, loss_gain=-0.3, n_probed=None,
                 loss_first=None, lora_sha=LORA_SHA, delta_norm=DELTA_NORM, nan_at=None,
                 training=None):
    training = training or ft_record(ds)["training"]
    n = N_PAIRS[ds]
    n_probed = n_probed or n
    losses = training["losses"]
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n_probed):
        fb = float(np.clip(rng.normal(0.65, 0.1), 0.05, 0.95))
        cb = float(np.clip(rng.normal(0.70, 0.1), 0.05, 0.95))
        lb = losses[i] if i == 0 else float(np.clip(rng.normal(0.9, 0.4), 0.05, None))
        rows.append(_row(ds, "base", i, fb, cb, lb, found_all=(i % 4 != 0)))
        rows.append(_row(ds, "ft", i, min(fb + fwd_gain + rng.normal(0, 0.01), 1.0),
                         min(cb + cls_gain + rng.normal(0, 0.01), 1.0),
                         max(lb + loss_gain * lb + rng.normal(0, 0.02), 0.01),
                         found_all=(i % 4 != 0)))
    if nan_at is not None:
        rows[nan_at]["fwd_miou"] = float("nan")
    base0 = rows[0]["loss"] if loss_first is None else loss_first
    return {
        "stage": "s17probe", "dataset": ds, "family": "sam2",
        "model": {"family": "sam2", "image_size": 1024, "checkpoint": "/x/ckpts/sam2.pt"},
        "recipe": _recipe(),
        "arms": {"base": {"path": None, "delta_norm": 0.0, "lora_sha": None},
                 "ft": {"path": f"/x/adapters/{LORA_NAME}", "lora_sha": lora_sha,
                        "delta_norm": delta_norm, "mismatches": 0, "modules": 54,
                        "r": 32, "alpha": 64.0, "objective": "plain",
                        "scopes": ["memory_encoder", "sam_mask_decoder"]}},
        "arm_order": ["base", "ft"],
        "replay": {"n_pairs": n, "train_images": n, "train_images_total": n,
                   "coverage": {"all": [n, n]}, "reference": "derangement",
                   "train_record": True, "steps_per_epoch": training["steps_per_epoch"],
                   "train_images_recorded": n,
                   "mask_channels_recorded": training["mask_channels"],
                   "mask_channels_replayed": training["mask_channels"],
                   "loss_first_train": training["loss_first"],
                   "checks": {"steps_per_epoch_matches": True, "train_images_matches": True,
                              "mask_channels_matches": True,
                              "steps_capped": training["steps_per_epoch"] < n},
                   "epoch": 0, "recipe_source": f"/x/adapters/{LORA_NAME}",
                   "n_probed": n_probed, "train_steps": list(range(n_probed)),
                   "groups": ["all"], "max_pairs": None if n_probed == n else n_probed,
                   "loss_first_probe_base": base0,
                   "loss_first_rel_err": abs(base0 - training["loss_first"]) / training["loss_first"],
                   "parity": {"ft": {"objective": "plain",
                                     "loss_first_train": training["loss_first"],
                                     "loss_first_probe_base": base0,
                                     "loss_first_rel_err": abs(base0 - training["loss_first"]) / training["loss_first"],
                                     "lora_sha": lora_sha, "delta_norm": delta_norm}}},
        "summary": {},
        "eval_parity": [],
        "pairs": rows,
        "probe": {"complete": True, "elapsed_s": 60.0, "peak_vram_gb": 5.0,
                  "dtype": "torch.bfloat16", "device": "cuda", "save_maps": "all"},
        "data": {"cache_format": "png", "source_preference": "faithful"},
    }


def _zip(path: Path, record: dict, maps: bool = False) -> str:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("record.json", json.dumps(record))
        z.writestr("env.json", json.dumps({"dtype": "synthetic"}))
        if maps:
            z.writestr("pairs.csv", "dataset,arm\n")
            z.writestr("maps/base/fwd/0000.png", b"\x89PNG")
            z.writestr("maps/ft/cls/0000.png", b"\x89PNG")
    return str(path)


def _run(args, ok=True):
    p = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, f"{args[0]} failed:\n{p.stdout}\n{p.stderr}"
    return p


def _combine(tmp_path, ds, probe=None, ft=None, ok=True):
    ftr = ft or ft_record(ds)
    pr = probe or probe_record(ds, training=ftr["training"])
    out = tmp_path / f"s17_{ds}.zip"
    p = _run(["scripts/combine_s17.py", "--out", str(out),
              f"probe={_zip(tmp_path / f'probe_{ds}.zip', pr, maps=True)}",
              f"ft={_zip(tmp_path / f'ft_{ds}.zip', ftr)}"], ok=ok)
    return out, p


def test_s17_combine_then_merge_then_gate_passes(tmp_path):
    per = {}
    for ds in N_PAIRS:
        out, p = _combine(tmp_path, ds)
        per[ds] = out
        assert "coverage_ok=True" in p.stdout and "recipe_matches=True" in p.stdout
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            rec = json.loads(z.read("record.json"))
        assert {"deltas.csv", "pairs.csv", "maps/base/fwd/0000.png"} <= set(names)
        r = rec["results"]["ft"]
        assert r["n_pairs"] == N_PAIRS[ds]
        assert r["loss"]["delta"]["mean_delta"] < 0 and r["loss"]["delta"]["significant"]
        assert r["forward"]["delta"]["n"] == N_PAIRS[ds]
        assert len(r["forward"]["per_trait"]) == len(TRAITS[ds])
        assert rec["provenance"]["ft"]["adapter_matches"] is True
        assert rec["replay"]["loss_first_rel_err"] == 0.0
        # the training loss rides along on every delta row
        with zipfile.ZipFile(out) as z2:
            lines = z2.read("deltas.csv").decode().splitlines()
        assert lines[0].startswith("arm,objective,base_arm,train_step,group,ref,query,class_ids,n_channels,train_loss")
        assert len(lines) == N_PAIRS[ds] + 1
    merged = tmp_path / "s17.zip"
    _run(["scripts/combine_s17.py", "--merge", "--out", str(merged),
          *[f"{ds}={p}" for ds, p in per.items()]])
    with zipfile.ZipFile(merged) as z:
        names = z.namelist()
        rec = json.loads(z.read("record.json"))
    assert set(rec["datasets"]) == set(N_PAIRS)
    assert "beetle/maps/base/fwd/0000.png" in names and "beetle/deltas.csv" in names
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(merged), "--no-file"])
    assert "GATE S17: PASS" in res.stdout, res.stdout
    assert "[beetle]" in res.stdout and "[butterfly_major]" in res.stdout
    assert "THE REPLAY PROOF" in res.stdout


def test_s17_gate_fails_when_the_adapter_did_not_lower_its_own_objective(tmp_path):
    """The pre-registered disqualifier: a closing loss that does not fall on the very
    pairs the adapter trained on means nothing downstream is interpretable."""
    ft = ft_record("beetle")
    out, _ = _combine(tmp_path, "beetle",
                      probe=probe_record("beetle", loss_gain=0.0, training=ft["training"]),
                      ft=ft)
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(out), "--no-file"], ok=False)
    assert "GATE S17: FAIL" in res.stdout, res.stdout
    assert "[FAIL]  DISQUALIFIER" in res.stdout, res.stdout


def test_s17_gate_fails_on_a_max_pairs_smoke_and_on_a_broken_replay_proof(tmp_path):
    ft = ft_record("beetle")
    out, p = _combine(tmp_path, "beetle",
                      probe=probe_record("beetle", n_probed=4, training=ft["training"]),
                      ft=ft)
    assert "coverage_ok=False" in p.stdout
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(out), "--no-file"], ok=False)
    assert "[FAIL]  every training pair was probed" in res.stdout, res.stdout

    out2, _ = _combine(tmp_path, "beetle",
                       probe=probe_record("beetle", loss_first=ft["training"]["loss_first"] * 1.5,
                                          training=ft["training"]),
                       ft=ft)
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(out2), "--no-file"], ok=False)
    assert "[FAIL]  THE REPLAY PROOF" in res.stdout, res.stdout


def test_s17_gate_fails_on_a_foreign_adapter_and_on_a_nan(tmp_path):
    ft = ft_record("beetle")
    out, p = _combine(tmp_path, "beetle",
                      probe=probe_record("beetle", lora_sha="ffff", training=ft["training"]),
                      ft=ft)
    assert "adapter_matches=False" in p.stdout
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(out), "--no-file"], ok=False)
    assert "[FAIL]  every probed adapter IS its fine-tune bundle's adapter" in res.stdout

    # no sha on either side -> the S15 pin (name + ||dW|| rel 1e-4) decides
    ft2 = ft_record("beetle", lora_sha=None)
    out2, p2 = _combine(tmp_path, "beetle",
                        probe=probe_record("beetle", lora_sha=None, delta_norm=DELTA_NORM * 1.01,
                                           training=ft2["training"]),
                        ft=ft2)
    assert "adapter_matches=False" in p2.stdout
    ok_out, p3 = _combine(tmp_path, "beetle",
                          probe=probe_record("beetle", lora_sha=None,
                                             delta_norm=DELTA_NORM * (1 + 5e-5),
                                             training=ft2["training"]),
                          ft=ft2)
    assert "adapter_matches=True" in p3.stdout

    out3, _ = _combine(tmp_path, "beetle",
                       probe=probe_record("beetle", nan_at=3, training=ft["training"]),
                       ft=ft)
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(out3), "--no-file"], ok=False)
    assert "[FAIL]  every scored value is a number" in res.stdout, res.stdout


def test_s17_combine_refuses_a_recipe_the_bundle_did_not_train(tmp_path):
    """A LEG key (what decides the pairs and the legs) must match; a loss knob is an
    objective, not a replay mismatch (arms of different objectives share one probe)."""
    ft = ft_record("beetle")
    ft["occl"]["memory_reset"] = False
    ft["training"]["memory_reset"] = False
    out, p = _combine(tmp_path, "beetle", ft=ft)
    assert "recipe_matches=False" in p.stdout
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(out), "--no-file"], ok=False)
    assert "[FAIL]  the replay recipe is the adapter's own" in res.stdout, res.stdout


def test_s17ft_gate_passes_on_the_carrier_and_fails_on_a_knob(tmp_path):
    ft = ft_record("beetle")
    path = _zip(tmp_path / "s17ft_sam2_beetle.zip", ft)
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", path, "--no-file"])
    assert "GATE S17FT: PASS" in res.stdout, res.stdout

    # excl zerocat IS an S17 arm; found_mask and bce_w 0 too — each alone
    for knob, tval in (("excl", {"mode": "zerocat"}), ("found_mask", {"rule": "any_logit_gt_0"}),
                       ("bce_w", 0.0)):
        arm = ft_record("beetle")
        arm["occl"][knob] = tval
        arm["training"][knob] = (tval if knob != "found_mask" else
                                 {"rule": "any_logit_gt_0", "steps": 12, "channels_total": 60,
                                  "channels_masked": 3, "steps_no_found": 0})
        res = _run(["scripts/check_gate.py", "s17ft", "--bundle",
                    _zip(tmp_path / f"s17ft_{knob}.zip", arm), "--no-file"])
        assert "GATE S17FT: PASS" in res.stdout, res.stdout
    arch = ft_arm("beetle", "arch", sha="s", delta=0.9)
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle",
                _zip(tmp_path / "s17ft_arch.zip", arch), "--no-file"])
    assert "GATE S17FT: PASS" in res.stdout and "the ARCH scopes" in res.stdout, res.stdout
    arch_knob = ft_arm("beetle", "arch", sha="s", delta=0.9)           # S17c comp_arch
    arch_knob["occl"]["excl"] = {"mode": "zerocat"}; arch_knob["training"]["excl"] = {"mode": "zerocat"}
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle",
                _zip(tmp_path / "s17ft_arch_knob.zip", arch_knob), "--no-file"])
    assert "GATE S17FT: PASS" in res.stdout, res.stdout
    # two knobs, or an S13 fold, or a declared knob the training block does not mirror: FAIL
    two = ft_record("beetle")
    two["occl"]["found_mask"] = {"rule": "any_logit_gt_0"}; two["occl"]["bce_w"] = 0.0
    two["training"]["found_mask"] = {"rule": "any_logit_gt_0"}; two["training"]["bce_w"] = 0.0
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle",
                _zip(tmp_path / "s17ft_two.zip", two), "--no-file"], ok=False)
    assert "[FAIL]  the carrier, or exactly ONE S17 change" in res.stdout, res.stdout
    fold = ft_record("beetle")
    fold["occl"]["xovl"] = {"weight": 1.0, "where": "query"}
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle",
                _zip(tmp_path / "s17ft_fold.zip", fold), "--no-file"], ok=False)
    assert "[FAIL]  the carrier, or exactly ONE S17 change" in res.stdout, res.stdout
    lying = ft_record("beetle")
    lying["occl"]["found_mask"] = {"rule": "any_logit_gt_0"}       # declared, not trained
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle",
                _zip(tmp_path / "s17ft_lying.zip", lying), "--no-file"], ok=False)
    assert "[FAIL]  the carrier, or exactly ONE S17 change" in res.stdout, res.stdout

    capped = ft_record("beetle", steps=2)
    capped_p = _zip(tmp_path / "s17ft_capped.zip", capped)
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", capped_p, "--no-file"], ok=False)
    assert "[FAIL]  the full derangement ran" in res.stdout, res.stdout


# ── the improvement arms: several adapters, a control, and the test-split cells ──

ARMS = {"carrier": {}, "found_mask": {"found_mask": {"rule": "any_logit_gt_0"}},
        "nobce": {"bce_w": 0.0}, "excl": {"excl": {"mode": "zerocat"}}, "arch": {}}
OBJ = {"carrier": "plain", "found_mask": "found_mask", "nobce": "nobce", "excl": "excl",
       "arch": "plain"}
OWN = {"plain": "loss", "found_mask": "loss_found_mask", "nobce": "loss_nobce",
       "excl": "loss_excl"}
ARCH_SCOPES = ["memory_encoder", "memory_attention", "sam_mask_decoder"]


def ft_arm(ds, arm, sha, delta):
    ft = ft_record(ds, lora_sha=sha, delta_norm=delta)
    ft["finetune"]["lora_path"] = f"/x/adapters/beetle_{arm}.pt"
    for k, v in ARMS[arm].items():
        ft["occl"][k] = v
        ft["training"][k] = (v if k != "found_mask" else
                             {"rule": "any_logit_gt_0", "steps": 12, "channels_total": 60,
                              "channels_masked": 3, "steps_no_found": 0})
    if arm == "arch":
        ft["finetune"]["lora"]["scopes"] = ARCH_SCOPES
        ft["training"]["lora"]["paths"] = ["memory_encoder.a", "memory_attention.q",
                                           "sam_mask_decoder.b"]
        ft["training"]["lora"]["modules"] = 78
    return ft


def multi_probe(ds, fts, loss_gain=-0.3, base_only_rows=False):
    """base + four adapters; every row carries every objective; per-arm parity."""
    n = N_PAIRS[ds]
    rng = np.random.default_rng(2)
    rows, parity, arms = [], {}, {"base": {"path": None, "delta_norm": 0.0, "lora_sha": None}}
    base_rows = []
    for i in range(n):
        fb = float(np.clip(rng.normal(0.65, 0.1), 0.05, 0.95))
        cb = float(np.clip(rng.normal(0.70, 0.1), 0.05, 0.95))
        lb = float(np.clip(rng.normal(0.9, 0.4), 0.05, None))
        base_rows.append((fb, cb, lb))
        rows.append(_row(ds, "base", i, fb, cb, lb, found_all=(i % 4 != 0)))
    for arm, ft in fts.items():
        arms[arm] = {"path": ft["finetune"]["lora_path"], "lora_sha": ft["finetune"]["lora_sha"],
                     "delta_norm": ft["training"]["delta_norm"], "mismatches": 0,
                     "modules": 54, "r": 32, "alpha": 64.0, "objective": OBJ[arm],
                     "scopes": ["memory_encoder", "sam_mask_decoder"]}
        for i, (fb, cb, lb) in enumerate(base_rows):
            rows.append(_row(ds, arm, i, min(fb + 0.02 + rng.normal(0, 0.01), 1.0),
                             min(cb + 0.04 + rng.normal(0, 0.01), 1.0),
                             max(lb + loss_gain * lb + rng.normal(0, 0.02), 0.01),
                             found_all=(i % 4 != 0)))
        # the arm's first training loss is the base row's column for its objective
        own = OWN[OBJ[arm]]
        ft["training"]["loss_first"] = rows[0][own]
        ft["training"]["losses"][0] = rows[0][own]
        parity[arm] = {"objective": OBJ[arm], "loss_first_train": rows[0][own],
                       "loss_first_probe_base": rows[0][own], "loss_first_rel_err": 0.0,
                       "lora_sha": ft["finetune"]["lora_sha"],
                       "delta_norm": ft["training"]["delta_norm"]}
    rec = probe_record(ds, training=next(iter(fts.values()))["training"])
    rec["pairs"] = rows
    rec["arms"] = arms
    rec["arm_order"] = ["base", *fts]
    rec["replay"]["parity"] = parity
    rec["replay"]["loss_first_train"] = parity["carrier"]["loss_first_train"]
    rec["replay"]["loss_first_probe_base"] = parity["carrier"]["loss_first_probe_base"]
    rec["replay"]["loss_first_rel_err"] = 0.0
    return rec


def eval_cell(ds, name, mean, tta, lora_path=None, delta=None, shot=1):
    runs = []
    for i in range(20):
        m = mean + 0.004 * ((i % 5) - 2)
        runs.append({"group": "all", "shot": shot, "run_idx": i, "miou": m,
                     "support_ids": [f"all-{i}-{c}" for c in "abcde"][:shot],
                     "miou_variants": {"argmax": m, "independent": m, "oracle": m + 0.02,
                                       "argmax_no_background": m - 0.2},
                     "miou_boundary": {"argmax_best3": m + 0.01, "independent_best3": m + 0.015},
                     "tta": {"episodes": 60, "rotated_chosen": 18} if tta else None})
    model = {"family": "sam2", "image_size": 1024, "normalization": "imagenet", "num_maskmem": 7}
    if tta:
        model["tta"] = "rot180_by_confidence"
    return {"stage": f"s17_{ds}_{name}", "dataset": ds,
            "adapter": "sst_sam2" + ("+tta_rot180" if tta else ""),
            "protocol": {"shots": shot, "n_runs": 20, "groups": ["all"],
                         "support_selection": "canonical_balanced",
                         "support_set_hash": f"hash_{ds}_{shot}"},
            "model": model,
            "metric": {"name": "miou", "assignment": "argmax_label",
                       "reduction": "mean_over_runs", "iou_space": "gt_native"},
            "data": {"source_preference": "faithful", "interp": "linear",
                     "cache_format": "png", "normalization": "imagenet"},
            "finetune": ({"method": "lora", "epochs": 1, "lora_path": lora_path, "lora_sha": None,
                          "lora": {"r": 32, "alpha": 64.0, "delta_norm": delta}}
                         if lora_path else None),
            "occl": {"mask_mode": "per_trait"} if lora_path else None,
            "results": {"miou": mean, "std": 0.01, "iou_binary": mean + 0.05},
            "runs": runs}


def test_s17_arms_with_a_control_and_the_test_split_pass_the_gate(tmp_path):
    ds = "beetle"
    fts = {arm: ft_arm(ds, arm, sha=f"sha_{arm}", delta=0.7 + 0.01 * k)
           for k, arm in enumerate(ARMS)}
    probe = multi_probe(ds, fts)
    levels = {"base": 0.632, "carrier": 0.633, "found_mask": 0.640, "nobce": 0.636,
              "excl": 0.635, "arch": 0.645}
    specs = [f"probe={_zip(tmp_path / 'probe.zip', probe, maps=True)}"]
    specs += [f"ft:{arm}={_zip(tmp_path / f'ft_{arm}.zip', ft)}" for arm, ft in fts.items()]
    for name, lvl in levels.items():
        for tta in (False, True):
            cell = name + ("_tta" if tta else "")
            lp = None if name == "base" else fts[name]["finetune"]["lora_path"]
            dl = None if name == "base" else fts[name]["training"]["delta_norm"]
            specs.append(f"eval:{cell}={_zip(tmp_path / f'eval_{cell}.zip', eval_cell(ds, cell, lvl + (0.02 if tta else 0), tta, lp, dl))}")
    # the 5-shot transfer fan (S17c): base and one arm at 5 shots, plain + TTA
    for name, lvl in (("base", 0.7316), ("arch", 0.7380)):
        for tta in (False, True):
            cell = name + "_5" + ("_tta" if tta else "")
            lp = None if name == "base" else fts[name]["finetune"]["lora_path"]
            dl = None if name == "base" else fts[name]["training"]["delta_norm"]
            specs.append(f"eval:{cell}={_zip(tmp_path / f'eval_{cell}.zip', eval_cell(ds, cell, lvl + (0.02 if tta else 0), tta, lp, dl, shot=5))}")
    out = tmp_path / "s17_beetle.zip"
    p = _run(["scripts/combine_s17.py", "--out", str(out), "--control", "carrier", *specs])
    assert "found_mask vs carrier per pair" in p.stdout and "test split (S11 protocol)" in p.stdout
    with zipfile.ZipFile(out) as z:
        rec = json.loads(z.read("record.json"))
    assert rec["adapter_arms"] == list(ARMS) and rec["control_arm"] == "carrier"
    assert rec["objectives"] == OBJ
    for arm in ARMS:
        assert rec["results"][arm]["loss"]["objective"] == OBJ[arm]
        assert rec["results"][arm]["loss"]["delta"]["significant"]
        assert rec["provenance"][arm]["adapter_matches"] is True
    assert set(rec["vs_control"]) == {"found_mask", "nobce", "excl", "arch"}
    assert rec["finetune"]["arch"]["finetune"]["lora"]["scopes"] == ARCH_SCOPES
    ev = rec["eval"]
    assert ev["base_pin"]["within"] is True
    assert {"found_mask_vs_base", "found_mask_vs_carrier", "found_mask_tta_vs_base_tta",
            "found_mask_tta_vs_carrier_tta", "carrier_vs_base", "base_tta_vs_plain",
            "excl_tta_vs_plain"} <= set(ev["pairings"])
    assert "carrier_vs_carrier" not in ev["pairings"]
    assert ev["shots"] == [1, 5] and ev["base_pin_5"]["within"] is True
    assert {"arch_5_vs_base_5", "arch_5_tta_vs_base_5_tta", "arch_5_tta_vs_plain",
            "base_5_tta_vs_plain"} <= set(ev["pairings"])
    # no pairing crosses shot counts
    for k in ev["pairings"]:
        if "_vs_plain" in k:
            continue
        a, b = k.split("_vs_")
        assert a.endswith("_5") == b.endswith("_5") or a.endswith("_5_tta") == b.endswith("_5_tta"), k
    assert ev["pairings"]["arch_5_vs_base_5"]["delta"] == pytest.approx(0.0064, abs=1e-6)
    assert ev["pairings"]["found_mask_vs_carrier"]["delta"] == pytest.approx(0.007, abs=1e-6)
    merged = tmp_path / "s17.zip"
    _run(["scripts/combine_s17.py", "--merge", "--out", str(merged), f"beetle={out}"])
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(merged), "--no-file"])
    assert "GATE S17: PASS" in res.stdout, res.stdout
    assert "every probed adapter IS its fine-tune bundle's adapter" in res.stdout
    assert "the plain base cell re-lands on the S11 ledger" in res.stdout
    # the same bundle combined under the s17b stage name gates as s17b, with its own bands
    out_b = tmp_path / "s17b_beetle.zip"
    _run(["scripts/combine_s17.py", "--stage", "s17b", "--out", str(out_b), "--control", "carrier", *specs])
    merged_b = tmp_path / "s17b.zip"
    _run(["scripts/combine_s17.py", "--stage", "s17b", "--merge", "--out", str(merged_b), f"beetle={out_b}"])
    with zipfile.ZipFile(merged_b) as z:
        assert json.loads(z.read("record.json"))["datasets"]["beetle"]["stage"] == "s17b"
    res = _run(["scripts/check_gate.py", "s17b", "--bundle", str(merged_b), "--no-file"])
    assert "GATE S17B: PASS" in res.stdout and "the deployment read" in res.stdout, res.stdout

    # one arm failing its OWN objective fails the whole stage; a base off the ledger too
    bad = multi_probe(ds, fts)
    for r in bad["pairs"]:
        if r["arm"] == "nobce":
            r["loss_nobce"] = float(r["loss_nobce"]) * 3          # nobce's objective went UP
    specs2 = [f"probe={_zip(tmp_path / 'probe_bad.zip', bad, maps=True)}"] + specs[1:]
    out2 = tmp_path / "s17_beetle_bad.zip"
    _run(["scripts/combine_s17.py", "--out", str(out2), "--control", "carrier", *specs2])
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(out2), "--no-file"], ok=False)
    assert "[FAIL]  DISQUALIFIER" in res.stdout, res.stdout
    specs3 = [s for s in specs if not s.startswith("eval:base=")]
    specs3.append(f"eval:base={_zip(tmp_path / 'eval_base_off.zip', eval_cell(ds, 'base', 0.600, False))}")
    out3 = tmp_path / "s17_beetle_off.zip"
    _run(["scripts/combine_s17.py", "--out", str(out3), "--control", "carrier", *specs3])
    res = _run(["scripts/check_gate.py", "s17", "--bundle", str(out3), "--no-file"], ok=False)
    assert "[FAIL]  test split: the plain base cell re-lands" in res.stdout, res.stdout


def test_s17ft_accepts_the_s17b_compositions_and_the_s17b_gate_exists(tmp_path):
    obj = {"weight": 1.0, "target": "present_for_prompted"}
    for name, occl_over in (("fwdonly", {"lora_legs": "forward"}),
                            ("objness", {"objness": obj}),
                            ("fwdonly_objness", {"lora_legs": "forward", "objness": obj}),
                            ("fwdonly_objness_excl", {"lora_legs": "forward", "objness": obj,
                                                      "excl": {"mode": "zerocat"}})):
        ft = ft_record("beetle")
        for k, v in occl_over.items():
            ft["occl"][k] = v
            ft["training"][k] = ({"weight": 1.0, "steps": 12} if k == "objness" else v)
        res = _run(["scripts/check_gate.py", "s17ft", "--bundle",
                    _zip(tmp_path / f"s17ft_{name}.zip", ft), "--no-file"])
        assert "GATE S17FT: PASS" in res.stdout, (name, res.stdout)
    # a retuned objectness weight, or objness on the arch scopes, or an untrained legs flag: FAIL
    bad = ft_record("beetle"); bad["occl"]["objness"] = {"weight": 0.5, "target": "present_for_prompted"}
    bad["training"]["objness"] = {"weight": 0.5}
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", _zip(tmp_path / "b1.zip", bad), "--no-file"], ok=False)
    assert "[FAIL]  the carrier, or exactly ONE S17 change" in res.stdout
    # S17c: the weight grid, the 2x2 and forward_link are sanctioned; anything off-grid is not
    for w in (3.0, 10.0, 30.0):
        ok_w = ft_record("beetle"); ok_w["occl"]["objness"] = {"weight": w, "target": "present_for_prompted"}
        ok_w["training"]["objness"] = {"weight": w}
        res = _run(["scripts/check_gate.py", "s17ft", "--bundle", _zip(tmp_path / f"w{w:g}.zip", ok_w), "--no-file"])
        assert "GATE S17FT: PASS" in res.stdout, res.stdout
    grid = ft_record("beetle"); grid["finetune"]["lr"] = 3e-5; grid["training"]["lr"] = 3e-5
    grid["finetune"]["epochs"] = 3; grid["training"]["epochs"] = 3; grid["training"]["steps"] = 36
    grid["training"]["losses"] = grid["training"]["losses"] * 3; grid["training"]["mask_channels"] = {"5": 36}
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", _zip(tmp_path / "grid.zip", grid), "--no-file"])
    assert "GATE S17FT: PASS" in res.stdout, res.stdout
    link = ft_record("beetle"); link["occl"]["lora_legs"] = "forward_link"; link["training"]["lora_legs"] = "forward_link"
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", _zip(tmp_path / "link.zip", link), "--no-file"])
    assert "GATE S17FT: PASS" in res.stdout, res.stdout
    off = ft_record("beetle"); off["finetune"]["lr"] = 1e-4; off["training"]["lr"] = 1e-4
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", _zip(tmp_path / "off.zip", off), "--no-file"], ok=False)
    assert "[FAIL]  one epoch at lr 1e-5" in res.stdout
    bad3 = ft_record("beetle"); bad3["occl"]["lora_legs"] = "forward"          # declared, not trained
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", _zip(tmp_path / "b3.zip", bad3), "--no-file"], ok=False)
    assert "[FAIL]  the carrier, or exactly ONE S17 change" in res.stdout
    res = _run(["scripts/check_gate.py", "--list"])
    assert "s17b" in res.stdout


# ── S17d: the soft sentinel — a per-arm gate with its own base variant ──────


def test_s17d_soft_arm_pairs_against_its_own_base_variant_and_the_gate_passes(tmp_path):
    ds = "beetle"
    obj = {"weight": 1.0, "target": "present_for_prompted"}
    # comp (hard gate, the control) and comp_soft (the same knobs + soft_sentinel 20)
    fts = {}
    for arm, soft in (("comp", None), ("comp_soft", 20.0)):
        ft = ft_record(ds, lora_sha=f"sha_{arm}", delta_norm=0.75 if soft is None else 0.81)
        ft["finetune"]["lora_path"] = f"/x/adapters/beetle_{arm}.pt"
        for k, v in (("lora_legs", "forward"), ("objness", obj), ("excl", {"mode": "zerocat"}),
                     ("soft_sentinel", soft)):
            ft["occl"][k] = v
            ft["training"][k] = ({"weight": 1.0, "steps": 12} if k == "objness" else v)
        fts[arm] = ft
    # the probe: base, base_soft20 (the base weights under the soft gate: the plain loss
    # value is lower — the sentinel constant is gone — and the forward leg is close),
    # comp vs base, comp_soft vs base_soft20
    n = N_PAIRS[ds]
    rng = np.random.default_rng(5)
    rows = []
    lv = []
    for i in range(n):
        fb = float(np.clip(rng.normal(0.65, 0.1), 0.05, 0.95))
        cb = float(np.clip(rng.normal(0.70, 0.1), 0.05, 0.95))
        lb = float(np.clip(rng.normal(0.9, 0.4), 0.05, None))
        lv.append((fb, cb, lb))
        rows.append(_row(ds, "base", i, fb, cb, lb, found_all=(i % 4 != 0)))
        r = _row(ds, "base_soft20", i, fb + 0.002, cb - 0.001, lb * 0.4, found_all=True)
        r["soft_sentinel"] = 20.0
        rows.append(r)
    for arm, base_arm, scale in (("comp", "base", 1.0), ("comp_soft", "base_soft20", 0.4)):
        for i, (fb, cb, lb) in enumerate(lv):
            r = _row(ds, arm, i, min(fb + 0.02 + rng.normal(0, 0.01), 1.0),
                     min(cb + 0.04 + rng.normal(0, 0.01), 1.0),
                     max(scale * lb * 0.7 + rng.normal(0, 0.02), 0.01),
                     found_all=(i % 4 != 0) or arm == "comp_soft")
            r["lora_legs"] = "forward"
            r["soft_sentinel"] = "" if arm == "comp" else 20.0
            rows.append(r)
    own = "loss_excl"
    parity = {}
    for arm, base_arm in (("comp", "base"), ("comp_soft", "base_soft20")):
        b0 = next(r for r in rows if r["arm"] == base_arm and r["train_step"] == 0)
        first = float(b0[own]) + 1.0 * float(b0["loss_objterm"])
        fts[arm]["training"]["loss_first"] = first
        fts[arm]["training"]["losses"][0] = first
        parity[arm] = {"objective": "excl+obj1", "loss_first_train": first,
                       "loss_first_probe_base": first, "loss_first_rel_err": 0.0,
                       "base_arm": base_arm, "lora_sha": fts[arm]["finetune"]["lora_sha"],
                       "delta_norm": fts[arm]["training"]["delta_norm"]}
    rec = probe_record(ds, training=fts["comp"]["training"])
    rec["pairs"] = rows
    rec["arms"] = {
        "base": {"path": None, "delta_norm": 0.0, "lora_sha": None, "soft_sentinel": None},
        "base_soft20": {"path": None, "delta_norm": 0.0, "lora_sha": None, "soft_sentinel": 20.0},
        **{arm: {"path": ft["finetune"]["lora_path"], "lora_sha": ft["finetune"]["lora_sha"],
                 "delta_norm": ft["training"]["delta_norm"], "mismatches": 0, "modules": 54,
                 "r": 32, "alpha": 64.0, "objective": "excl+obj1", "lora_legs": "forward",
                 "soft_sentinel": ft["occl"]["soft_sentinel"],
                 "scopes": ["memory_encoder", "sam_mask_decoder"]} for arm, ft in fts.items()}}
    rec["arm_order"] = ["base", "base_soft20", "comp", "comp_soft"]
    rec["base_arm"], rec["base_arms"] = "base", ["base", "base_soft20"]
    rec["base_of"] = {"comp": "base", "comp_soft": "base_soft20"}
    rec["replay"]["parity"] = parity
    rec["replay"]["loss_first_rel_err"] = 0.0
    specs = [f"probe={_zip(tmp_path / 'probe.zip', rec, maps=True)}"]
    specs += [f"ft:{arm}={_zip(tmp_path / f'ft_{arm}.zip', ft)}" for arm, ft in fts.items()]
    for name, lvl in (("base", 0.632), ("comp", 0.635), ("comp_soft", 0.638)):
        for tta in (False, True):
            cell = name + ("_tta" if tta else "")
            lp = None if name == "base" else fts[name]["finetune"]["lora_path"]
            dl = None if name == "base" else fts[name]["training"]["delta_norm"]
            specs.append(f"eval:{cell}={_zip(tmp_path / f'eval_{cell}.zip', eval_cell(ds, cell, lvl + (0.02 if tta else 0), tta, lp, dl))}")
    out = tmp_path / "s17d_beetle.zip"
    p = _run(["scripts/combine_s17.py", "--stage", "s17d", "--out", str(out),
              "--control", "comp", *specs])
    assert "gate effect at base weights, base_soft20 (K=20.0, close_aug=None) vs base" in p.stdout, p.stdout
    assert "comp_soft vs comp per pair" in p.stdout and "(gates differ)" in p.stdout
    with zipfile.ZipFile(out) as z:
        crec = json.loads(z.read("record.json"))
        lines = z.read("deltas.csv").decode().splitlines()
    assert crec["adapter_arms"] == ["comp", "comp_soft"]
    assert crec["base_arms"] == ["base", "base_soft20"]
    assert crec["base_of"] == {"comp": "base", "comp_soft": "base_soft20"}
    assert crec["results"]["comp_soft"]["base_arm"] == "base_soft20"
    assert crec["results"]["comp_soft"]["soft_sentinel"] == 20.0
    # the soft arm's own-loss levels are the SOFT base's (0.4x), not the hard base's
    assert crec["results"]["comp_soft"]["loss"]["base_mean"] < 0.6 * crec["results"]["comp"]["loss"]["base_mean"]
    ge = crec["gate_effect"]["base_soft20"]
    assert ge["soft_sentinel"] == 20.0 and ge["n_pairs"] == n
    assert ge["loss_plain"]["mean_delta"] < 0 and ge["loss_plain"]["significant"]
    assert ge["found_count"]["mean_delta"] > 0                # the fence rows opened
    assert crec["vs_control"]["comp_soft"]["same_gate"] is False
    assert lines[0].startswith("arm,objective,base_arm,train_step")
    assert any(l.startswith("comp_soft,excl+obj1,base_soft20,") for l in lines[1:])
    merged = tmp_path / "s17d.zip"
    _run(["scripts/combine_s17.py", "--stage", "s17d", "--merge", "--out", str(merged), f"beetle={out}"])
    res = _run(["scripts/check_gate.py", "s17d", "--bundle", str(merged), "--no-file"])
    assert "GATE S17D: PASS" in res.stdout, res.stdout
    assert "each adapter pairs against the base under its own gate" in res.stdout
    assert "gate at base weights: base_soft20" in res.stdout
    assert "comp_soft/excl+obj1: probe" in res.stdout          # the proof under ITS gate
    # a soft arm whose base variant is missing, or whose gate differs from it: refused
    bad = json.loads(json.dumps(rec))
    bad["base_of"]["comp_soft"] = "base"
    bp = _run(["scripts/combine_s17.py", "--stage", "s17d", "--out", str(tmp_path / "bad.zip"),
               "--control", "comp", f"probe={_zip(tmp_path / 'probe_bad.zip', bad, maps=True)}",
               *specs[1:]], ok=False)
    assert "ran different gates" in bp.stdout + bp.stderr
    # s17ft: the soft sentinel at K=20 composes with the S17b/c knobs; off-K, or declared
    # but not trained, fails
    for arm, ft in fts.items():
        res = _run(["scripts/check_gate.py", "s17ft", "--bundle",
                    _zip(tmp_path / f"s17ft_{arm}.zip", ft), "--no-file"])
        assert "GATE S17FT: PASS" in res.stdout, (arm, res.stdout)
        if arm == "comp_soft":
            assert "soft_sentinel=20.0" in res.stdout
    offk = json.loads(json.dumps(fts["comp_soft"]))
    offk["occl"]["soft_sentinel"] = 10.0; offk["training"]["soft_sentinel"] = 10.0
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", _zip(tmp_path / "offk.zip", offk), "--no-file"], ok=False)
    assert "[FAIL]  the carrier, or exactly ONE S17 change" in res.stdout
    lying = json.loads(json.dumps(fts["comp_soft"]))
    lying["training"]["soft_sentinel"] = None
    res = _run(["scripts/check_gate.py", "s17ft", "--bundle", _zip(tmp_path / "lying.zip", lying), "--no-file"], ok=False)
    assert "[FAIL]  the carrier, or exactly ONE S17 change" in res.stdout
