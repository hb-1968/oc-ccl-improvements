"""The s16 gate <-> combine schema round-trip, exercised locally before any GPU run.

The lesson from s14sec (all eight gate checks failed on KeyErrors because the
combiner's --merge shape and the gate's reader had never met a real bundle) applies
to every new stage: run the REAL pipeline over synthetic, run_eval-shaped cells

    2 cell bundles per dataset -> combine_s16 -> combine_s16 --merge
        -> check_gate s16 --bundle --no-file

and require PASS on both datasets. The synthetic draws are prefix-nested the way
`seeding.sample_supports` / `curation.stratified_supports` produce them (run i's
1-shot support is the first of its 2-shot pair); one test breaks that nesting to
prove the combine REFUSES rather than shipping an unlicensed pairing, and one lands
beetle's 2-shot below the banked 1-shot level to prove the dropped-frame check FAILS.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MODEL = {"family": "sam2", "normalization": "imagenet", "image_size": 1024,
         "num_maskmem": 7, "checkpoint": "sam2_hiera_large.pt", "dtype": "torch.bfloat16"}
SEL = {"butterfly_major": "canonical_placement", "beetle": "canonical_balanced"}
GROUPS = {"butterfly_major": ["cyrbia", "malleti"], "beetle": ["all"]}
#: the S11 ledger the gate pins (s1_tta) and plausible 2-shot levels — chosen so the
#: headline lands with the pre-registered SIGN on each dataset (TTA wins butterfly,
#: the explicit shot wins beetle).
LEVELS = {"butterfly_major": {"s1_tta": 0.7518, "s2": 0.7090},
          "beetle": {"s1_tta": 0.6386, "s2": 0.6750}}
CELLS = {"s1_tta": (1, True), "s2": (2, False)}


def _draw(group, i, shots, nested=True):
    """Run i's support ids: prefix-nested unless asked to break it."""
    ids = [f"{group}-{i}-a", f"{group}-{i}-b"]
    if not nested and shots == 2:
        ids = ids[::-1]                      # the 1-shot support is no longer first
    return ids[:shots]


def _cell(dataset, shots, tta, mean, nested=True):
    model = dict(MODEL)
    if tta:
        model["tta"] = "rot180_by_confidence"
    runs = []
    for g in GROUPS[dataset]:
        for i in range(20):
            m = mean + 0.002 * ((i % 5) - 2) + (0.001 if g.startswith("m") else 0.0)
            runs.append({
                "group": g, "shot": shots, "run_idx": i, "miou": m,
                "support_ids": _draw(g, i, shots, nested),
                "miou_variants": {"argmax": m, "independent": m,
                                  "oracle": m + 0.02, "argmax_no_background": m - 0.25},
                "miou_boundary": {"argmax_best3": m + 0.01, "independent_best3": m + 0.015},
                "tta": {"episodes": 100, "rotated_chosen": 8} if tta else None,
            })
    return {
        "stage": "s16_cell", "dataset": dataset,
        "adapter": "sst_sam2" + ("+tta_rot180" if tta else ""),
        "protocol": {"shots": shots, "n_runs": 20, "groups": GROUPS[dataset],
                     "support_selection": SEL[dataset],
                     "support_set_hash": f"hash_{dataset}_{shots}"},
        "model": model,
        "metric": {"name": "miou", "assignment": "argmax_label",
                   "reduction": "mean_over_runs", "iou_space": "gt_native"},
        "data": {"source_preference": "faithful", "interp": "linear",
                 "cache_format": "png", "normalization": "imagenet"},
        "finetune": None, "occl": None,
        "results": {"miou": mean, "std": 0.01, "iou_binary": mean + 0.05,
                    "tta": {"episodes": 2000, "rotated_chosen": 160} if tta else None},
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


def _pair_specs(tmp_path, ds, levels=None, nested=True):
    levels = levels or LEVELS[ds]
    return [f"{k}={_zip(tmp_path / f'{ds}_{k}.zip', _cell(ds, sh, tta, levels[k], nested))}"
            for k, (sh, tta) in CELLS.items()]


def test_s16_combine_then_merge_then_gate_passes(tmp_path):
    per_ds = {}
    for ds in ("butterfly_major", "beetle"):
        out = tmp_path / f"s16_{ds}.zip"
        p = _run(["scripts/combine_s16.py", "--out", str(out), *_pair_specs(tmp_path, ds)])
        assert "nested: " in p.stdout and "HEADLINE" in p.stdout, p.stdout
        per_ds[ds] = out
        with zipfile.ZipFile(out) as z:
            rec = json.loads(z.read("record.json"))
        # the proof travels: the headline verified all (group, run) pairs, and the
        # model.tta mask it crosses is declared
        h = rec["pairings"]["second_shot_vs_tta"]
        assert h["nested_shots"]["runs_checked"] == 20 * len(GROUPS[ds])
        assert "model.tta" in h["ignored_fields"]
        # the sign follows the levels: TTA wins butterfly, the explicit shot wins beetle
        assert (rec["results"]["headline_plain"] < 0) == (ds == "butterfly_major")

    merged = tmp_path / "s16.zip"
    _run(["scripts/combine_s16.py", "--merge", "--out", str(merged),
          *[f"{ds}={p}" for ds, p in per_ds.items()]])

    out = _run(["scripts/check_gate.py", "s16", "--bundle", str(merged), "--no-file"]).stdout
    assert "GATE S16: PASS" in out, out
    assert "[butterfly_major]" in out and "[beetle]" in out, out
    # and the per-dataset bundle is accepted on its own (a split session)
    out1 = _run(["scripts/check_gate.py", "s16", "--bundle", str(per_ds["beetle"]),
                 "--no-file"]).stdout
    assert "GATE S16: PASS" in out1, out1


def test_s16_combine_refuses_when_the_draws_are_not_nested(tmp_path):
    out = tmp_path / "s16_beetle.zip"
    p = _run(["scripts/combine_s16.py", "--out", str(out),
              *_pair_specs(tmp_path, "beetle", nested=False)], ok=False)
    assert p.returncode != 0
    assert "prefix-nested" in (p.stdout + p.stderr)
    assert not out.exists(), "no bundle may be written for an unlicensed pairing"


def test_s16_gate_fails_a_2shot_below_the_banked_1shot_level(tmp_path):
    """The one sign-gated check: beetle 2-shot at/below the banked 63.20 means a
    conditioning frame was dropped, not a scientific result."""
    out = tmp_path / "s16_beetle.zip"
    _run(["scripts/combine_s16.py", "--out", str(out),
          *_pair_specs(tmp_path, "beetle", levels={"s1_tta": 0.6386, "s2": 0.6200})])
    res = _run(["scripts/check_gate.py", "s16", "--bundle", str(out), "--no-file"], ok=False)
    assert "GATE S16: FAIL" in res.stdout, res.stdout
    assert "[FAIL]  2-shot clears the banked 1-shot level" in res.stdout, res.stdout
