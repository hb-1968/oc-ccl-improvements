#!/usr/bin/env python3
"""One dataset's {base, OC-CCL} x {no TTA, TTA} square — S8d.

    python scripts/combine_s8d_cells.py --out s8d_beetle.zip \
        base=a.zip occl=b.zip base_tta=c.zip occl_tta=d.zip

**Each OC-CCL arm is paired against its OWN base.** `occl_tta` vs `base` would fold TTA's
effect into OC-CCL's and report their sum as the loss's contribution. So there are two
paired deltas per dataset — OC-CCL inside the plain stack and OC-CCL inside the TTA stack —
and the difference between them is the interaction.

That interaction is the reason this is a square rather than a line. The obvious argument is
that OC-CCL is training-side and TTA is inference-side, hence independent, hence measurable
apart and added afterwards. S6c refutes exactly that reasoning for a different pair: two
"independent" techniques interacted at -4.66 points and their combination scored worse than
the better one alone. Here there is a mechanism too — TTA picks between two predictions by
logit confidence, and fine-tuning changes confidence calibration, so OC-CCL can alter how
well TTA selects while leaving the no-TTA score untouched.

`registry.compare` refuses to pair across the TTA boundary (`model.tta` is a comparability
field), which is what forces the pairing to be done correctly rather than conveniently.
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

def pairs_for(cells) -> list[tuple[str, str, str]]:
    """(base_cell, occl_cell, name) for every OC-CCL arm present.

    Derived rather than listed, because S8.5 adds a THIRD arm — the same protocol trained
    under SST's hard-threshold prompt instead of our differentiable one (defect A2) — and a
    fixed `[("base","occl"),("base_tta","occl_tta")]` would silently drop it from the record
    while still writing a bundle that looks complete.

    Every arm pairs against the base **in its own stack**: an `_tta` arm against `base_tta`,
    a plain arm against `base`. Pairing across that boundary would fold TTA's effect into
    OC-CCL's, and `registry.compare` refuses it anyway — `model.tta` is a comparability
    field.
    """
    out = []
    for name in sorted(cells):
        if not name.startswith("occl"):
            continue
        tta = name.endswith("_tta")
        base = "base_tta" if tta else "base"
        if base in cells:
            tag = name[len("occl"):].strip("_") or ("tta" if tta else "plain")
            out.append((base, name, tag))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", metavar="label=bundle.zip")
    ap.add_argument("--stage", default="s8d",
                    help="the record must name the stage that produced it. Hardcoding\n"
                         "         s8d made every s8_5a bundle claim to be s8d, which is the\n"
                         "         same class of error as a record misstating its own epochs.")
    ap.add_argument("--out", default="s8d_cells.zip")
    args = ap.parse_args()

    recs = {}
    for spec in args.cells:
        label, _, path = spec.partition("=")
        with zipfile.ZipFile(path) as z:
            recs[label] = json.loads(z.read("record.json"))

    # A cell scored without TTA but filed as a TTA cell would silently become the
    # interaction. Check the record rather than trusting the label.
    for label, r in recs.items():
        wants_tta = label.endswith("_tta")
        has_tta = str(r.get("adapter", "")).endswith("+tta_rot180")
        if wants_tta != has_tta:
            raise SystemExit(
                f"cell {label!r} has adapter {r.get('adapter')!r} — "
                f"{'expected' if wants_tta else 'did not expect'} TTA")

    first = next(iter(recs.values()))
    ds = first["dataset"]
    # Read the shot count from the record — a hardcoded "5-shot" here mislabelled every
    # 1-shot stage that reused this combiner, the same class of error as the --stage fix.
    shots = (first.get("protocol") or {}).get("shots", "?")
    print(f"\n{ds}, {shots}-shot — OC-CCL arms vs base\n")
    print(f"{'stack':>8s} {'base':>15s} {'+ OC-CCL':>15s} "
          f"{'paired delta (CI95)':>26s}")
    print("-" * 70)

    out_cells, deltas = {}, {}
    for b_label, t_label, tag in pairs_for(recs):
        if b_label not in recs or t_label not in recs:
            continue
        b, t = recs[b_label], recs[t_label]
        try:
            cmp = registry.compare(t, b)
        except ValueError as e:
            raise SystemExit(f"{tag}: cannot pair {t_label} against {b_label}: {e}") from e
        # The same pairing under the paper's boundary-tolerant reading — S12's headline.
        # None (not a crash) when either side predates the recording evaluator.
        try:
            cmp3 = registry.compare(t, b, metric="miou_boundary.independent_best3")
        except ValueError:
            cmp3 = None
        lo, hi = cmp["ci95"]
        deltas[tag] = cmp["mean_delta"]

        def _lift(rec, block, key):
            """Mean of a per-run diagnostic field, None for bundles predating it."""
            vals = [r[block][key] for r in rec.get("runs", [])
                    if (r.get(block) or {}).get(key) is not None]
            return float(sum(vals) / len(vals)) if vals else None

        def _independent(rec):
            """Per-run SST-release-rule mIoU (independent binarisation)."""
            return _lift(rec, "miou_variants", "independent")

        def _tta_frac(rec):
            """Mean fraction of episodes where TTA picked the rotated support — the
            instrument for S13's pose-prior disqualifier (D1). None off-TTA."""
            vals = [r["tta"]["rotated_chosen"] / r["tta"]["episodes"]
                    for r in rec.get("runs", [])
                    if (r.get("tta") or {}).get("episodes")]
            return float(sum(vals) / len(vals)) if vals else None

        out_cells[tag] = {
            "base_miou": b["results"]["miou"], "base_std": b["results"]["std"],
            "occl_miou": t["results"]["miou"], "occl_std": t["results"]["std"],
            "paired_delta": cmp["mean_delta"], "paired_ci95": [lo, hi],
            "significant": cmp["significant"], "n_seeds": cmp["n_seeds"],
            "n_seeds_improved": cmp["n_seeds_improved"],
            "base_iou_binary": b["results"].get("iou_binary"),
            "base_headroom": b["results"].get("headroom"),
            # The overlap-rule diagnostic, lifted per cell so the combined record can say
            # whether the arm's delta is metric-rule-dependent without reopening bundles.
            "base_miou_independent": _independent(b),
            "occl_miou_independent": _independent(t),
            # The unpublished best-of-3 boundary tolerance (independent rule — the
            # closest reading of the paper's full metric), same lifting contract.
            "base_miou_best3": _lift(b, "miou_boundary", "independent_best3"),
            "occl_miou_best3": _lift(t, "miou_boundary", "independent_best3"),
            # The TRUE assignment ceiling (S13 Day-0): `oracle` has been recorded per
            # run since S10's evaluator edit but was never lifted into a combined
            # record, which is how "+15.6 assignment headroom" (actually silhouette
            # minus mIoU, a correlate) stood in for a ceiling that had never been
            # measured. `oracle - argmax` is the recoverable-by-relabelling mass;
            # `argmax_no_background` isolates what the >0 threshold discards.
            "base_miou_oracle": _lift(b, "miou_variants", "oracle"),
            "occl_miou_oracle": _lift(t, "miou_variants", "oracle"),
            "base_miou_argmax_no_background":
                _lift(b, "miou_variants", "argmax_no_background"),
            "occl_miou_argmax_no_background":
                _lift(t, "miou_variants", "argmax_no_background"),
            # How often TTA kept the rotated support, per side — a fine-tune that
            # sharpens the pick moves this toward the truth; one that learned "the
            # support looks upright" collapses it toward 0 (S13's D1 disqualifier).
            "base_tta_rotated_frac": _tta_frac(b),
            "occl_tta_rotated_frac": _tta_frac(t),
            "paired_delta_best3": cmp3["mean_delta"] if cmp3 else None,
            "paired_ci95_best3": cmp3["ci95"] if cmp3 else None,
            "significant_best3": cmp3["significant"] if cmp3 else None,
        }
        print(f"{tag:>8s} {100 * b['results']['miou']:7.2f}±{100 * b['results']['std']:5.2f} "
              f"{100 * t['results']['miou']:7.2f}±{100 * t['results']['std']:5.2f} "
              f"{100 * cmp['mean_delta']:+8.2f} [{100 * lo:+.2f},{100 * hi:+.2f}]"
              f"{'*' if cmp['significant'] else ''}")
        c = out_cells[tag]
        if c["base_miou_oracle"] is not None and c["occl_miou_oracle"] is not None:
            print(f"{'':>8s}   oracle {100 * c['base_miou_oracle']:.2f} -> "
                  f"{100 * c['occl_miou_oracle']:.2f}  (assignment ceiling; "
                  f"oracle-argmax = {100 * (c['base_miou_oracle'] - b['results']['miou']):+.2f} "
                  f"on base)")

    results = {"deltas": deltas}
    # One method's {plain, TTA} pair — under s8d's original tags ("plain"/"tta") or the
    # generic (m, m_tta) naming every later stage uses. The literal deltas["tta"] -
    # deltas["plain"] here crashed the FIRST stage with exactly two differently-named
    # cells (s12: ptscnp/ptscnp_tta) — a tag-naming assumption of the same family as
    # pairs_for's, but unlike pairs_for it was never generalised.
    _pair = None
    if len(deltas) == 2:
        tags = sorted(deltas)
        if set(tags) == {"plain", "tta"}:
            _pair = ("plain", "tta")
        elif tags[1] == f"{tags[0]}_tta":
            _pair = (tags[0], tags[1])
    if _pair:
        _p, _t = _pair
        inter = deltas[_t] - deltas[_p]
        results["interaction"] = inter
        print(f"\nOC-CCL delta, plain stack : {100 * deltas[_p]:+.2f}")
        print(f"OC-CCL delta, TTA stack   : {100 * deltas[_t]:+.2f}")
        print(f"INTERACTION               : {100 * inter:+.2f}  "
              "(does the loss help more, or less, once TTA is on top)")
        if abs(inter) < 0.005:
            print("  -> OC-CCL and TTA are separable here; measuring either apart is safe")
        else:
            print("  -> NOT separable: the loss's effect depends on the inference stack, "
                  "so it cannot be measured once and added on")

    # The number that actually ships: the best of the four cells.
    best = max(recs, key=lambda k: recs[k]["results"]["miou"])
    results["best_cell"] = best
    results["best_miou"] = recs[best]["results"]["miou"]
    print(f"\nbest cell: {best} at {100 * recs[best]['results']['miou']:.2f}")
    if recs[best]["results"].get("headroom") is not None:
        print(f"  silhouette {100 * recs[best]['results']['iou_binary']:.2f}, "
              f"assignment headroom {100 * recs[best]['results']['headroom']:+.2f}")

    # Training provenance, PER ARM. A single `finetune` block only makes sense when every
    # OC-CCL cell came from the same checkpoint — true for the s8d square, false for s8_5a2
    # where each arm IS a different training configuration and telling them apart is the
    # entire point. The old lookup asked for a cell literally named "occl" or "occl_tta",
    # so the moment arms became `occl_hard`, `occl_train`, ... it returned None and silently
    # dropped every ||dW|| from the record. A record that cannot say which weights produced
    # a delta cannot support the delta.
    arms_meta = {k: {"finetune": r.get("finetune"), "occl": r.get("occl")}
                 for k, r in recs.items() if k != "base" and not k.startswith("base")}
    missing = sorted(k for k, v in arms_meta.items() if not v["finetune"])
    if missing:
        raise SystemExit(
            f"arm(s) {missing} carry no `finetune` block. Their per-arm bundle was written "
            "without one, so there is no record of what produced those weights — refusing "
            "to combine rather than shipping a delta with no provenance behind it.")

    def _shared(field):
        """The block itself when every arm agrees, else None — with `arms` holding the rest."""
        seen = {json.dumps(v[field], sort_keys=True, default=str)
                for v in arms_meta.values()}
        return next(iter(arms_meta.values()))[field] if len(seen) == 1 else None

    record = {
        "stage": args.stage, "dataset": ds,
        "protocol": recs[best]["protocol"],
        "finetune": _shared("finetune"),
        "occl": _shared("occl"),
        "arms": arms_meta,
        "cells": out_cells,
        "miou": {k: r["results"]["miou"] for k, r in recs.items()},
        "results": results,
    }
    dest = export_bundle(args.stage, record, out=args.out)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
