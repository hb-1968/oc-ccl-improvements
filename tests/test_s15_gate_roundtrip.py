"""The s15 gate <-> combine schema round-trip, exercised locally before any GPU run.

Same lesson as s16's: run the REAL pipeline over synthetic, run_eval-shaped cells

    9 cell bundles per dataset -> combine_s15 -> combine_s15 --merge
        -> check_gate s15 --bundle --no-file

and require PASS on both datasets. The synthetic draws are prefix-nested (run i's
1-shot support is the first of its 5-shot draw). Three refusals are proved: a 5-shot
sam3 cell whose model still carries the builder's cap of 4; a champion cell whose
adapter differs from its 1-shot anchor; and beetle's 5-shot base landing below the
72 floor (the dropped-frame disqualifier).
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SAM3 = {"family": "sam3", "normalization": "half", "image_size": 1008,
        "mask_input_size": 1152, "num_maskmem": 7, "checkpoint": "sam3.pt",
        "dtype": "torch.bfloat16", "global_autocast_exited": True,
        "max_cond_frames_in_attn": -1, "load_missing_keys": 0, "load_unexpected_keys": 0}
SAM2 = {"family": "sam2", "normalization": "imagenet", "image_size": 1024,
        "num_maskmem": 7, "checkpoint": "sam2_hiera_large.pt", "dtype": "torch.bfloat16"}
SEL = {"butterfly_major": "canonical_placement", "beetle": "canonical_balanced"}
GROUPS = {"butterfly_major": ["cyrbia", "malleti"], "beetle": ["all"]}
#: label -> (family, shots, tta, adapter?), mirrored from combine_s15.CELLS
CELLS = {
    "base": ("sam3", 5, False, False), "base_tta": ("sam3", 5, True, False),
    "arch_tta": ("sam3", 5, True, True),
    "base_1": ("sam3", 1, False, False), "base_tta_1": ("sam3", 1, True, False),
    "arch_tta_1": ("sam3", 1, True, True),
    "sam2_base": ("sam2", 5, False, False), "sam2_base_tta": ("sam2", 5, True, False),
    "sam2_crown": ("sam2", 5, True, True),
}
#: plausible levels — the S15 brief's central projections for the 5-shot cells, the
#: S14sam3 SCC cells for the 1-shot anchors, S6/S8d for the SAM 2 ledger
LEVELS = {
    "beetle": {"base": 0.750, "base_tta": 0.765, "arch_tta": 0.787,
               "base_1": 0.6526, "base_tta_1": 0.6630, "arch_tta_1": 0.7021,
               "sam2_base": 0.7316, "sam2_base_tta": 0.7520, "sam2_crown": 0.7480},
    "butterfly_major": {"base": 0.735, "base_tta": 0.780, "arch_tta": 0.787,
                        "base_1": 0.7109, "base_tta_1": 0.7650, "arch_tta_1": 0.7666,
                        "sam2_base": 0.7178, "sam2_base_tta": 0.7722, "sam2_crown": 0.7807},
}
ARCH_SHA = "arch-sha-0001"


def _draw(group, i, shots):
    return [f"{group}-{i}-{c}" for c in "abcde"][:shots]


def _cell(dataset, label, mean, cap=-1, lora_sha=None, delta_norm=6.26):
    fam, shots, tta, adapted = CELLS[label]
    model = dict(SAM3 if fam == "sam3" else SAM2)
    if fam == "sam3":
        model["max_cond_frames_in_attn"] = cap
    if tta:
        model["tta"] = "rot180_by_confidence"
    runs = []
    for g in GROUPS[dataset]:
        for i in range(20):
            m = mean + 0.002 * ((i % 5) - 2) + (0.001 if g.startswith("m") else 0.0)
            runs.append({
                "group": g, "shot": shots, "run_idx": i, "miou": m,
                "support_ids": _draw(g, i, shots),
                "miou_variants": {"argmax": m, "independent": m,
                                  "oracle": m + 0.02, "argmax_no_background": m - 0.25},
                "miou_boundary": {"argmax_best3": m + 0.01, "independent_best3": m + 0.015},
                "tta": {"episodes": 100, "rotated_chosen": 22} if tta else None,
            })
    adapter = ("sam3" if fam == "sam3" else "sst_sam2") + ("+tta_rot180" if tta else "")
    return {
        "stage": "s15_cell", "dataset": dataset, "adapter": adapter,
        "protocol": {"shots": shots, "n_runs": 20, "groups": GROUPS[dataset],
                     "support_selection": SEL[dataset],
                     "support_set_hash": f"hash_{dataset}_{shots}"},
        "model": model,
        "metric": {"name": "miou", "assignment": "argmax_label",
                   "reduction": "mean_over_runs", "iou_space": "gt_native"},
        "data": {"source_preference": "faithful", "interp": "linear",
                 "cache_format": "png", "normalization": model["normalization"]},
        # lora_sha is null in real records (save_lora does not stamp it); the
        # adapter is identified by its file name and the loaded delta_norm
        "finetune": ({"method": "lora", "epochs": 1,
                      "lora_path": f"/somewhere/{dataset}_arch.pt" if fam == "sam3" else "/x/occl.pt",
                      "lora_sha": lora_sha,
                      "lora": {"r": 32, "alpha": 64.0, "scopes": ["maskmem_backbone",
                               "transformer", "sam_mask_decoder"], "modules": 94,
                               "delta_norm": delta_norm}} if adapted else None),
        "occl": {"mask_mode": "per_trait"} if adapted else None,
        "results": {"miou": mean, "std": 0.01, "iou_binary": mean + 0.05,
                    "tta": {"episodes": 2000, "rotated_chosen": 440} if tta else None},
        "runs": runs,
    }


def _zip(path: Path, record: dict) -> str:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("record.json", json.dumps(record))
        z.writestr("env.json", json.dumps({"dtype": "synthetic"}))
    return str(path)


def _run(args, ok=True):
    p = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, f"{args[0]} failed:\n{p.stdout}\n{p.stderr}"
    return p


def _specs(tmp_path, ds, levels=None, overrides=None, without=()):
    levels = levels or LEVELS[ds]
    overrides = overrides or {}
    return [f"{k}={_zip(tmp_path / f'{ds}_{k}.zip', _cell(ds, k, levels[k], **overrides.get(k, {})))}"
            for k in CELLS if k not in without]


def _s8d_summary(tmp_path, ds):
    """The shape S8d actually shipped: a summary with cells.tta means, no per-run rows."""
    return _zip(tmp_path / f"s8d_{ds}.zip", {
        "stage": "s8d", "dataset": ds,
        "protocol": {"shots": 5, "n_runs": 20, "support_selection": SEL[ds]},
        "cells": {"tta": {"base_miou": LEVELS[ds]["sam2_base_tta"], "base_std": 0.01,
                          "occl_miou": LEVELS[ds]["sam2_crown"], "occl_std": 0.01}},
        "results": {}})


def test_s15_combine_then_merge_then_gate_passes(tmp_path):
    per_ds = {}
    for ds in ("butterfly_major", "beetle"):
        out = tmp_path / f"s15_{ds}.zip"
        p = _run(["scripts/combine_s15.py", "--out", str(out), *_specs(tmp_path, ds)])
        assert "HEADLINE" in p.stdout and "nested:" in p.stdout, p.stdout
        per_ds[ds] = out
        with zipfile.ZipFile(out) as z:
            rec = json.loads(z.read("record.json"))
        for row in ("shot_lift_base", "shot_lift_arch"):
            ev = rec["pairings"][row]["nested_shots"]
            assert ev["runs_checked"] == 20 * len(GROUPS[ds])
            assert ev["treatment_shots"] == 5 and ev["baseline_shots"] == 1
        assert rec["pairings"]["arch_vs_sam2_crown"]["cross_family"] is True
        assert rec["cells"]["arch_tta"]["adapter_id"]["delta_norm"] == 6.26
        assert rec["cells"]["arch_tta"]["adapter_id"]["lora_path"] == f"{ds}_arch.pt"
        assert rec["results"]["best_cell"] == "arch_tta"

    merged = tmp_path / "s15.zip"
    _run(["scripts/combine_s15.py", "--merge", "--out", str(merged),
          *[f"{ds}={p}" for ds, p in per_ds.items()]])
    out = _run(["scripts/check_gate.py", "s15", "--bundle", str(merged), "--no-file"]).stdout
    assert "GATE S15: PASS" in out, out
    assert "[butterfly_major]" in out and "[beetle]" in out, out
    out1 = _run(["scripts/check_gate.py", "s15", "--bundle", str(per_ds["beetle"]),
                 "--no-file"]).stdout
    assert "GATE S15: PASS" in out1, out1


def test_s15_combine_refuses_a_5shot_cell_still_on_cap_4(tmp_path):
    out = tmp_path / "s15_beetle.zip"
    p = _run(["scripts/combine_s15.py", "--out", str(out),
              *_specs(tmp_path, "beetle", overrides={"base_tta": {"cap": 4}})], ok=False)
    assert p.returncode != 0 and "max_cond_frames_in_attn is 4" in (p.stdout + p.stderr)
    assert not out.exists()


def test_s15_combine_refuses_a_shot_lift_across_two_adapters(tmp_path):
    # GPU-type noise in ||dW|| (6th decimal) must NOT refuse; a retraining must
    out = tmp_path / "s15_beetle_noise.zip"
    _run(["scripts/combine_s15.py", "--out", str(out),
          *_specs(tmp_path, "beetle", overrides={"arch_tta_1": {"delta_norm": 6.2600023}})])
    assert out.exists()
    out = tmp_path / "s15_beetle.zip"
    p = _run(["scripts/combine_s15.py", "--out", str(out),
              *_specs(tmp_path, "beetle", overrides={"arch_tta_1": {"delta_norm": 6.27}})],
             ok=False)
    assert p.returncode != 0 and "different adapters" in (p.stdout + p.stderr)
    assert not out.exists()


def test_s15_gate_fails_beetle_base_below_the_floor(tmp_path):
    """The pre-registered disqualifier: beetle 5-shot base below 72 with the cap
    raised means supports are still being dropped — plumbing, not science."""
    out = tmp_path / "s15_beetle.zip"
    levels = dict(LEVELS["beetle"], base=0.700)
    _run(["scripts/combine_s15.py", "--out", str(out),
          *_specs(tmp_path, "beetle", levels=levels)])
    res = _run(["scripts/check_gate.py", "s15", "--bundle", str(out), "--no-file"], ok=False)
    assert "GATE S15: FAIL" in res.stdout, res.stdout
    assert "[FAIL]  5-shot base clears the 1-shot level" in res.stdout, res.stdout


def test_s15_without_the_s8d_per_run_cells_ships_the_shot_lift_headline(tmp_path):
    """The S8d TTA cells were never banked per-run (Drive holds only the summary), so
    the combine must run without them: the two crown rows become UNPAIRED summary
    reads, the paired headline falls back to the champion's own shot lift, and the
    gate PASSES with both facts declared."""
    per_ds = {}
    for ds in ("butterfly_major", "beetle"):
        out = tmp_path / f"s15_{ds}.zip"
        p = _run(["scripts/combine_s15.py", "--out", str(out),
                  "--sam2-summary", _s8d_summary(tmp_path, ds),
                  *_specs(tmp_path, ds, without=("sam2_base_tta", "sam2_crown"))])
        assert "fallback" in p.stdout and "UNPAIRED" in p.stdout, p.stdout
        per_ds[ds] = out
        with zipfile.ZipFile(out) as z:
            rec = json.loads(z.read("record.json"))
        assert rec["cells_absent"] == ["sam2_base_tta", "sam2_crown"]
        assert rec["pairings"]["arch_vs_sam2_crown"] is None
        assert rec["pairings"]["base_tta_vs_sam2_base_tta"] is None
        assert rec["results"]["headline_kind"] == "shot_lift_arch"
        assert abs(rec["unpaired"]["arch_vs_sam2_crown"]
                   - (LEVELS[ds]["arch_tta"] - LEVELS[ds]["sam2_crown"])) < 1e-9
        assert rec["sam2_summary"]["source"] == f"s8d_{ds}.zip"
    merged = tmp_path / "s15.zip"
    _run(["scripts/combine_s15.py", "--merge", "--out", str(merged),
          *[f"{ds}={p}" for ds, p in per_ds.items()]])
    out = _run(["scripts/check_gate.py", "s15", "--bundle", str(merged), "--no-file"]).stdout
    assert "GATE S15: PASS" in out, out
    assert "UNPAIRED" in out and "headline = shot_lift_arch" in out, out


def test_s15_combine_refuses_absent_crown_cells_without_a_summary(tmp_path):
    out = tmp_path / "s15_beetle.zip"
    p = _run(["scripts/combine_s15.py", "--out", str(out),
              *_specs(tmp_path, "beetle", without=("sam2_crown",))], ok=False)
    assert p.returncode != 0 and "no --sam2-summary" in (p.stdout + p.stderr)
    assert not out.exists()


def test_backfill_licenses_only_an_all_lossless_dataset(tmp_path):
    """s6_beetle.zip predates data.cache_format; beetle was PNG in every era, so
    the field may be inferred — WITH the evidence in the record. A dataset with
    any lossy entry (butterfly's JPEG era) must be refused: its pixels differed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("bf", REPO / "scripts" / "backfill_ledger.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(
        json.dumps({"dataset": "beetle", "fmt": "png", "cache_path": f"b/{i}.png"}) + "\n"
        for i in range(3)) + "".join(
        json.dumps({"dataset": "butterfly_major", "fmt": f, "cache_path": f"m/{i}.{f}"}) + "\n"
        for i, f in enumerate(("png", "jpg"))))
    legacy = {"dataset": "beetle", "data": {"source_preference": "faithful", "interp": "linear"},
              "protocol": {"shots": 5}, "results": {"miou": 0.73}, "runs": []}
    out = bf.backfill(legacy, manifest)
    assert out["data"]["cache_format"] == "png"
    assert out["data"]["cache_format_backfilled"]["entries"] == 3
    assert legacy["data"].get("cache_format") is None          # input untouched
    import pytest
    with pytest.raises(ValueError, match="lossy"):
        bf.backfill(dict(legacy, dataset="butterfly_major"), manifest)
    with pytest.raises(ValueError, match="already"):
        bf.backfill(out, manifest)
    # the back-filled record now shares a comparability key with a modern PNG record
    from occl import registry
    modern = json.loads(json.dumps(out)); modern["data"].pop("cache_format_backfilled")
    assert registry.diff_fields(out, modern) == {}
