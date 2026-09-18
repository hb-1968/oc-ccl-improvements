#!/usr/bin/env python3
"""S15: the champion at 5-shot — one dataset's cells, seven pairings.

    python scripts/combine_s15.py --out s15_beetle.zip \
        base=... base_tta=... arch_tta=...                        (sam3, 5-shot)
        base_1=... base_tta_1=... arch_tta_1=...                  (sam3, 1-shot: s14sam3_<ds>_*.zip)
        sam2_base=...                                             (SAM 2, 5-shot plain: s6_<ds>.zip)
        [sam2_base_tta=... sam2_crown=...]                        (SAM 2, 5-shot per-run cells, IF they exist)
        [--sam2-summary s8d_<ds>.zip]                             (else: the S8d summary, for UNPAIRED level reads)

    python scripts/combine_s15.py --merge --out s15.zip \
        butterfly_major=s15_butterfly_major.zip beetle=s15_beetle.zip

Every S14 arm was 1-shot; the README's S15 brief projected, from measured components
only, what the champion (SAM 3 + per-trait/excl + arch LoRA + TTA) scores at 5-shot,
and named the one pre-condition: sam3's tracker is built with
``max_cond_frames_in_attn=4``, so a 5-shot episode would silently attend to 4 of 5
supports. The adapter now sets it to -1 (all, SAM 2's own setting); every cell here
must carry that in ``model`` or the combine refuses.

Pairings, and the route each takes through `registry.compare`:

  shot_lift_base        base(5)     vs base_1        nested_shots  — the raw shot lift
  shot_lift_arch        arch_tta(5) vs arch_tta_1    nested_shots  — the champion's lift
  tta_vs_base           base_tta    vs base          {model.tta}   — TTA at 5-shot
  arch_vs_base_tta      arch_tta    vs base_tta      plain         — stack+arch at 5-shot
  base_vs_sam2_base     base(5)     vs sam2_base     cross-family  — the backbone edge at 5
  base_tta_vs_sam2_base_tta                          cross-family
  arch_vs_sam2_crown    arch_tta(5) vs sam2_crown    cross-family  — THE HEADLINE

The shot-count rows go through `compare(nested_shots=True)`, which verifies from the
per-run ``support_ids`` that run i's 1-shot support is the first of its 5-shot draw
before masking anything (S16's licence, re-earned at 5). The arch row additionally
requires the SAME adapter on both sides (``finetune.lora_sha``): a shot lift measured
across two trainings would be a training delta in disguise.

The two SAM 2 TTA cells (S8d ``base_tta`` / ``occl_tta``) were never banked as
per-run bundles — S8d shipped a summary — so they are OPTIONAL here. When absent, the
two rows are recorded as ``None``, the S8d summary means stand in as UNPAIRED level
reads (``unpaired``: a difference of means, no CI, no seed pairing — labelled so), and
the paired HEADLINE falls back to the champion's own shot lift (``shot_lift_arch``,
proven on the draws). ``results.headline_kind`` says which one shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl import registry  # noqa: E402
from occl.bundle import export_bundle  # noqa: E402

CROSS = ("model.family", "model.image_size")

#: label -> (family, shots, tta, adapter?)
CELLS = {
    "base":          ("sam3", 5, False, False),
    "base_tta":      ("sam3", 5, True,  False),
    "arch_tta":      ("sam3", 5, True,  True),
    "base_1":        ("sam3", 1, False, False),
    "base_tta_1":    ("sam3", 1, True,  False),
    "arch_tta_1":    ("sam3", 1, True,  True),
    "sam2_base":     ("sam2", 5, False, False),
    "sam2_base_tta": ("sam2", 5, True,  False),
    "sam2_crown":    ("sam2", 5, True,  True),
}
OPTIONAL = ("sam2_base_tta", "sam2_crown")
HEADLINE = "arch_vs_sam2_crown"          # when the per-run crown exists
HEADLINE_FALLBACK = "shot_lift_arch"     # else: the champion's own shot lift, paired


def _lift(rec, block, key):
    vals = [r[block][key] for r in rec.get("runs", [])
            if (r.get(block) or {}).get(key) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _tta_frac(rec):
    vals = [r["tta"]["rotated_chosen"] / r["tta"]["episodes"]
            for r in rec.get("runs", []) if (r.get("tta") or {}).get("episodes")]
    return float(sum(vals) / len(vals)) if vals else None


def _cell(label, rec):
    fam, shots, tta, _ = CELLS[label]
    return {
        "family": fam, "shots": shots, "tta": tta,
        "miou": rec["results"]["miou"], "std": rec["results"]["std"],
        "n_seeds": (rec.get("protocol") or {}).get("n_runs"),
        "support_set_hash": (rec.get("protocol") or {}).get("support_set_hash"),
        "adapter_id": _adapter_id(rec),
        "max_cond_frames_in_attn": (rec.get("model") or {}).get("max_cond_frames_in_attn"),
        "miou_independent": _lift(rec, "miou_variants", "independent"),
        "miou_best3": _lift(rec, "miou_boundary", "independent_best3"),
        "miou_oracle": _lift(rec, "miou_variants", "oracle"),
        "miou_argmax_no_background": _lift(rec, "miou_variants",
                                           "argmax_no_background"),
        "tta_rotated_frac": _tta_frac(rec),
        "iou_binary": rec["results"].get("iou_binary"),
    }


def _pair(t, b, name, cross=False, ignore=(), nested=False):
    kw = {}
    if cross:
        kw["ignore_fields"] = CROSS
    elif ignore:
        kw["ignore_fields"] = tuple(ignore)
    if nested:
        kw["nested_shots"] = True
    try:
        cmp = registry.compare(t, b, **kw)
    except ValueError as e:
        raise SystemExit(f"{name}: cannot pair: {e}") from e
    try:
        cmp3 = registry.compare(t, b, metric="miou_boundary.independent_best3", **kw)
    except ValueError:
        cmp3 = None
    row = {
        "delta": cmp["mean_delta"], "ci95": cmp["ci95"],
        "significant": cmp["significant"], "n_seeds": cmp["n_seeds"],
        "n_seeds_improved": cmp["n_seeds_improved"],
        "delta_best3": cmp3["mean_delta"] if cmp3 else None,
        "ci95_best3": cmp3["ci95"] if cmp3 else None,
        "significant_best3": cmp3["significant"] if cmp3 else None,
        "cross_family": cross,
        "ignored_fields": cmp.get("ignored_fields"),
        "nested_shots": cmp.get("nested_shots"),
    }
    star = "*" if cmp["significant"] else ""
    b3 = (f" | best3 {100 * cmp3['mean_delta']:+.2f}"
          f" [{100 * cmp3['ci95'][0]:+.2f},{100 * cmp3['ci95'][1]:+.2f}]"
          f"{'*' if cmp3['significant'] else ''}" if cmp3 else "")
    tag = (" (CROSS-FAMILY)" if cross else "") + (
        f" (nested: {row['nested_shots']['runs_checked']} runs verified)" if nested else "")
    print(f"{name:>28s}: {100 * cmp['mean_delta']:+.2f} "
          f"[{100 * cmp['ci95'][0]:+.2f},{100 * cmp['ci95'][1]:+.2f}]{star}"
          f"  {cmp['n_seeds_improved']}/{cmp['n_seeds']} seeds{b3}{tag}")
    return row


def _adapter_id(rec):
    """What identifies the loaded adapter in a run_eval record.

    `finetune.lora_sha` is null in practice (save_lora does not stamp it), so the
    pin rests on what IS written at load time: the adapter file's name and
    `finetune.lora.delta_norm` — ||ΔW|| of the tensors actually loaded, a float
    that two different trainings cannot share. The sha is kept when present.
    """
    ft = rec.get("finetune") or {}
    lora = ft.get("lora") or {}
    path = ft.get("lora_path")
    return {"lora_sha": ft.get("lora_sha"),
            "lora_path": Path(path).name if path else None,
            "delta_norm": lora.get("delta_norm")}


def _same_adapter(a, b):
    if a["delta_norm"] is None or b["delta_norm"] is None:
        return False
    if a["lora_sha"] and b["lora_sha"] and a["lora_sha"] != b["lora_sha"]:
        return False
    # ||dW|| of the same file differs at the 6th decimal across GPU types (measured:
    # 6.3242969 on L40S vs 6.3242992 on A40) — reduction-order noise. A retraining
    # moves it by whole units, so a relative 1e-4 separates the two cleanly.
    da, db = float(a["delta_norm"]), float(b["delta_norm"])
    return a["lora_path"] == b["lora_path"] and abs(da - db) <= 1e-4 * max(1.0, abs(da))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", metavar="label=bundle.zip")
    ap.add_argument("--stage", default="s15")
    ap.add_argument("--merge", action="store_true",
                    help="labels are DATASET names of per-dataset s15 combines; "
                         "emit the stage bundle {datasets: {...}}")
    ap.add_argument("--sam2-summary", default=None,
                    help="the S8d summary bundle (s8d_<ds>.zip) — supplies the SAM 2 "
                         "5-shot base_tta / occl_tta MEANS for unpaired level reads when "
                         "the per-run cells are not available")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.merge:
        per_ds = {}
        for spec in args.cells:
            label, _, path = spec.partition("=")
            with zipfile.ZipFile(path) as z:
                per_ds[label] = json.loads(z.read("record.json"))
        for ds, r in per_ds.items():
            if r.get("dataset") != ds:
                raise SystemExit(f"{ds}: record is for {r.get('dataset')!r}")
        record = {
            "stage": args.stage, "family": "sam3", "datasets": per_ds,
            "results": {ds: {"headline": r["results"]["headline"],
                             "headline_plain": r["results"]["headline_plain"]}
                        for ds, r in per_ds.items()},
        }
        dest = export_bundle(args.stage, record, out=args.out)
        print(f"wrote {dest}")
        return 0

    recs = {}
    for spec in args.cells:
        label, _, path = spec.partition("=")
        with zipfile.ZipFile(path) as z:
            recs[label] = json.loads(z.read("record.json"))
    required = set(CELLS) - set(OPTIONAL)
    if not required <= set(recs) or not set(recs) <= set(CELLS):
        raise SystemExit(f"cells {sorted(recs)}: need {sorted(required)} "
                         f"(+ optional {sorted(OPTIONAL)})")
    absent = [k for k in OPTIONAL if k not in recs]
    summary = None
    if args.sam2_summary:
        with zipfile.ZipFile(args.sam2_summary) as z:
            sm = json.loads(z.read("record.json"))
        tta = (sm.get("cells") or {}).get("tta") or {}
        if (sm.get("protocol") or {}).get("shots") != 5 or "occl_miou" not in tta:
            raise SystemExit(f"{args.sam2_summary} is not an S8d 5-shot summary bundle")
        summary = {"source": Path(args.sam2_summary).name, "dataset": sm.get("dataset"),
                   "base_tta": tta["base_miou"], "base_tta_std": tta.get("base_std"),
                   "crown": tta["occl_miou"], "crown_std": tta.get("occl_std")}
    if absent and summary is None:
        raise SystemExit(f"per-run cells {absent} absent and no --sam2-summary given — "
                         "the crown rows would be neither paired nor read")

    # Check the RECORDS against the labels, never the labels alone.
    ds = recs["base"]["dataset"]
    sel = recs["base"]["protocol"]["support_selection"]
    for k, r in recs.items():
        fam, shots, tta, adapted = CELLS[k]
        got_fam = (r.get("model") or {}).get("family")
        if got_fam != fam:
            raise SystemExit(f"cell {k!r} has model.family {got_fam!r}, expected {fam!r}")
        got_shots = (r.get("protocol") or {}).get("shots")
        if got_shots != shots:
            raise SystemExit(f"cell {k!r} has protocol.shots {got_shots!r}, expected {shots}")
        has_tta = str(r.get("adapter", "")).endswith("+tta_rot180")
        if has_tta != tta:
            raise SystemExit(f"cell {k!r} adapter {r.get('adapter')!r} does not match "
                             f"its {'_tta' if tta else 'plain'} label")
        if (r.get("finetune") is not None) != adapted:
            raise SystemExit(f"cell {k!r} {'lacks' if adapted else 'carries'} a finetune "
                             f"block — {'the champion needs its adapter' if adapted else 'a base cell must be base weights'}")
        if r.get("dataset") != ds:
            raise SystemExit(f"cell {k!r} is {r.get('dataset')!r}, not {ds!r}")
        if summary is not None and summary["dataset"] not in (None, ds):
            raise SystemExit(f"--sam2-summary is for {summary['dataset']!r}, not {ds!r}")
        if r["protocol"].get("support_selection") != sel:
            raise SystemExit(f"cell {k!r} uses {r['protocol'].get('support_selection')!r} "
                             f"supports; every cell must share {sel!r}")
        if fam == "sam3" and shots == 5:
            cap = (r.get("model") or {}).get("max_cond_frames_in_attn")
            if not (cap == -1 or (isinstance(cap, int) and cap >= 6)):
                raise SystemExit(
                    f"cell {k!r}: model.max_cond_frames_in_attn is {cap!r} — at 5-shot "
                    "the tracker would attend to 4 of 5 supports (the S15 pre-condition)")
    _a5, _a1 = _adapter_id(recs["arch_tta"]), _adapter_id(recs["arch_tta_1"])
    if not _same_adapter(_a5, _a1):
        raise SystemExit(
            f"arch_tta (5-shot) and arch_tta_1 carry different adapters "
            f"({_a5} vs {_a1}) — the shot lift must be measured on ONE set of weights")

    print(f"\n{ds}, {sel} — the champion at 5-shot\n")
    for k in CELLS:
        if k in recs:
            r = recs[k]["results"]
            print(f"  {k:>14s}: {100 * r['miou']:6.2f} ± {100 * r['std']:5.2f}")
        else:
            print(f"  {k:>14s}: (no per-run cell; S8d summary mean "
                  f"{100 * summary['base_tta' if k == 'sam2_base_tta' else 'crown']:.2f})")
    print()

    pairings = {
        "shot_lift_base": _pair(recs["base"], recs["base_1"], "shot_lift_base",
                                nested=True),
        "shot_lift_arch": _pair(recs["arch_tta"], recs["arch_tta_1"], "shot_lift_arch",
                                nested=True),
        "tta_vs_base": _pair(recs["base_tta"], recs["base"], "tta_vs_base",
                             ignore=("model.tta",)),
        "arch_vs_base_tta": _pair(recs["arch_tta"], recs["base_tta"], "arch_vs_base_tta"),
        "base_vs_sam2_base": _pair(recs["base"], recs["sam2_base"], "base_vs_sam2_base",
                                   cross=True),
        "base_tta_vs_sam2_base_tta": (
            _pair(recs["base_tta"], recs["sam2_base_tta"], "base_tta_vs_sam2_base_tta",
                  cross=True) if "sam2_base_tta" in recs else None),
        HEADLINE: (_pair(recs["arch_tta"], recs["sam2_crown"], HEADLINE, cross=True)
                   if "sam2_crown" in recs else None),
    }
    # UNPAIRED level reads against the S8d summary: a difference of means, no CI,
    # no seed pairing. Always recorded when a summary is given, so the paired and
    # unpaired readings can be compared where both exist.
    unpaired = None
    if summary is not None:
        unpaired = {
            "base_tta_vs_sam2_base_tta": recs["base_tta"]["results"]["miou"] - summary["base_tta"],
            "arch_vs_sam2_crown": recs["arch_tta"]["results"]["miou"] - summary["crown"],
            "note": "difference of means vs the S8d summary — unpaired, no CI",
        }
        for k in ("base_tta_vs_sam2_base_tta", "arch_vs_sam2_crown"):
            print(f"{k:>28s}: {100 * unpaired[k]:+.2f}  (UNPAIRED vs {summary['source']})")

    if pairings[HEADLINE] is not None:
        h, kind = pairings[HEADLINE], "paired_vs_crown"
        print(f"\nHEADLINE arch_tta(5) vs the SAM 2 5-shot crown: {100 * h['delta']:+.2f} "
              f"[{100 * h['ci95'][0]:+.2f},{100 * h['ci95'][1]:+.2f}]"
              f"{'*' if h['significant'] else ''}")
    else:
        h, kind = pairings[HEADLINE_FALLBACK], HEADLINE_FALLBACK
        print(f"\nHEADLINE (fallback — no per-run crown) arch_tta shot lift 1 -> 5: "
              f"{100 * h['delta']:+.2f} [{100 * h['ci95'][0]:+.2f},{100 * h['ci95'][1]:+.2f}]"
              f"{'*' if h['significant'] else ''}; unpaired vs crown "
              f"{100 * unpaired['arch_vs_sam2_crown']:+.2f}")
    print(f"oracle ceiling (sam3 base, 5-shot): "
          f"{100 * (_lift(recs['base'], 'miou_variants', 'oracle') or 0):.2f} "
          f"vs argmax {100 * recs['base']['results']['miou']:.2f}")

    best = max((k for k in recs if CELLS[k][1] == 5),
               key=lambda k: recs[k]["results"]["miou"])
    record = {
        "stage": args.stage, "dataset": ds, "family": "sam3",
        # the shared 5-shot protocol; per-cell shots and hashes live in `cells`
        "protocol": {k: v for k, v in recs["base"]["protocol"].items()
                     if k not in ("shots", "support_set_hash")},
        "model": recs["arch_tta"].get("model"),
        "finetune": recs["arch_tta"].get("finetune"),
        "occl": recs["arch_tta"].get("occl"),
        "cells": {k: _cell(k, r) for k, r in recs.items()},
        "cells_absent": absent,
        "sam2_summary": summary,
        "pairings": pairings,
        "unpaired": unpaired,
        "results": {
            "headline": h["delta_best3"],
            "headline_plain": h["delta"],
            "headline_significant": h["significant"],
            "headline_kind": kind,
            "best_cell": best,
            "best_miou": recs[best]["results"]["miou"],
            "miou": {k: r["results"]["miou"] for k, r in recs.items()},
        },
    }
    dest = export_bundle(args.stage, record, out=args.out)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
