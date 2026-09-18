"""The s14sam3 gate <-> combine schema round-trip, exercised locally.

The landed s14sec bundle failed ALL EIGHT gate checks on KeyErrors: combine's
``--merge`` nests per-dataset records under ``record["datasets"]`` while the gate
read a per-dataset root — reader and writer had never been run against each other,
because no real bundle existed before the GPU session paid for one. This test runs
the REAL pipeline over synthetic cells:

    run_eval-shaped cell bundles -> combine_s14 (per dataset)
        -> combine_s14 --merge -> check_gate s14sam3 --bundle --no-file

and requires PASS. Field-name drift in any of the three schemas (cell record,
combine's lift, the gate's dig paths) fails here, on this machine, in seconds.

The model block below mirrors ``Sam3TrackerAdapter.config_report()``; the chain to
the REAL adapter is closed by ``commission_s14.py``, which asserts the same key
names (image_size / mask_input_size / num_maskmem / load_* / global_autocast_exited)
against the live report before the Colab session spends anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SAM3_MODEL = {
    "family": "sam3", "normalization": "half", "image_size": 1008,
    "mask_input_size": 1152, "num_maskmem": 7, "checkpoint": "sam3.pt",
    "dtype": "torch.bfloat16", "resize_interp": "linear",
    "global_autocast_exited": True,
    "load_missing_keys": 0, "load_unexpected_keys": 0,
}
SAM2_MODEL = {
    "family": "sam2", "normalization": "imagenet", "image_size": 1024,
    "num_maskmem": 7, "checkpoint": "sam2.1_hiera_large.pt",
    "dtype": "torch.bfloat16",
}

# the S14 protocol constants the gate pins per dataset
SEL = {"butterfly_major": "canonical_placement", "beetle": "canonical_balanced"}
GROUPS = {"butterfly_major": ["cyrbia"], "beetle": ["all"]}
# the banked-S11 ledger means the gate pins (tolerance 0.02)
PINS = {"butterfly_major": (0.7030, 0.7518, 0.7652),
        "beetle": (0.6320, 0.6386, 0.6412)}


def _cell(dataset, adapter, model, mean, tta=False, finetune=None):
    model = dict(model)
    if tta:
        model["tta"] = "rot180_by_confidence"
    runs = []
    for i in range(20):
        m = mean + 0.001 * ((i % 5) - 2)
        runs.append({
            "group": GROUPS[dataset][0], "shot": 1, "run_idx": i, "miou": m,
            "miou_variants": {"argmax": m, "independent": m,
                              "oracle": m + 0.02,
                              "argmax_no_background": m - 0.25},
            "miou_boundary": {"argmax_best3": m + 0.01,
                              "independent_best3": m + 0.015},
            "tta": {"episodes": 100, "rotated_chosen": 8} if tta else None,
        })
    return {
        "stage": "cell", "dataset": dataset, "adapter": adapter,
        "protocol": {"shots": 1, "n_runs": 20, "groups": GROUPS[dataset],
                     "support_selection": SEL[dataset],
                     "support_set_hash": f"hash_{dataset}"},
        "model": model,
        "metric": {"name": "miou", "assignment": "argmax_label",
                   "reduction": "mean_over_runs", "iou_space": "gt_native"},
        "data": {"source_preference": "faithful", "interp": "linear",
                 "cache_format": "png",
                 "normalization": model.get("normalization")},
        "finetune": finetune,
        "occl": ({"mask_mode": "per_trait", "excl": {"mode": "zerocat"}}
                 if finetune else None),
        "results": {"miou": mean, "std": 0.01, "iou_binary": mean + 0.05},
        "runs": runs,
    }


def _zip(path: Path, record: dict) -> str:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("record.json", json.dumps(record))
        z.writestr("env.json", json.dumps({"dtype": "synthetic"}))
    return str(path)


def _run(args):
    p = subprocess.run([sys.executable, *args], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 0, f"{args[0]} failed:\n{p.stdout}\n{p.stderr}"
    return p.stdout


def test_s14sam3_combine_then_merge_then_gate_passes(tmp_path):
    ft = {"method": "lora", "epochs": 1,
          "lora": {"r": 32, "alpha": 64.0,
                   "scopes": ["maskmem_backbone", "sam_mask_decoder"],
                   "modules": 54, "delta_norm": 1.0}}
    ft_arch = {"method": "lora", "epochs": 1,
               "lora": {"r": 32, "alpha": 64.0,
                        "scopes": ["maskmem_backbone", "transformer",
                                   "sam_mask_decoder"],
                        "modules": 150, "delta_norm": 1.0}}
    per_ds = {}
    for ds in ("butterfly_major", "beetle"):
        b, bt, pt = PINS[ds]
        cells = {
            "base": _cell(ds, "sam3", SAM3_MODEL, b + 0.01),
            "base_tta": _cell(ds, "sam3+tta_rot180", SAM3_MODEL, bt + 0.01,
                              tta=True),
            "stack_tta": _cell(ds, "sam3+tta_rot180", SAM3_MODEL, pt + 0.01,
                               tta=True, finetune=ft),
            "arch_tta": _cell(ds, "sam3+tta_rot180", SAM3_MODEL, pt + 0.02,
                              tta=True, finetune=ft_arch),
            "sam2_base": _cell(ds, "sst_sam2", SAM2_MODEL, b),
            "sam2_base_tta": _cell(ds, "sst_sam2+tta_rot180", SAM2_MODEL, bt,
                                   tta=True),
            "sam2_pertrait_tta": _cell(ds, "sst_sam2+tta_rot180", SAM2_MODEL, pt,
                                       tta=True, finetune=ft),
        }
        specs = [f"{k}={_zip(tmp_path / f'{ds}_{k}.zip', r)}"
                 for k, r in cells.items()]
        out = tmp_path / f"s14sam3_{ds}.zip"
        _run(["scripts/combine_s14.py", "--family", "sam3",
              "--out", str(out), *specs])
        per_ds[ds] = out

    merged = tmp_path / "s14sam3.zip"
    _run(["scripts/combine_s14.py", "--family", "sam3", "--merge",
          "--out", str(merged),
          *[f"{ds}={p}" for ds, p in per_ds.items()]])

    out = _run(["scripts/check_gate.py", "s14sam3",
                "--bundle", str(merged), "--no-file"])
    assert "GATE S14SAM3: PASS" in out, out
    # both datasets were actually scored, not just the first
    assert "[butterfly_major]" in out and "[beetle]" in out, out
