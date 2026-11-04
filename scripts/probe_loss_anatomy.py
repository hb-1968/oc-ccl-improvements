#!/usr/bin/env python3
"""Summarize forward and closing loss components from a saved training-pair probe."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl.finetune.probe import parse_per_trait  # noqa: E402

TAKEOVER_FWD, TAKEOVER_CLS = 0.5, 0.2


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(";")] if s else []


def anatomy(rows: list[dict], top: int = 10) -> dict:
    """Pure: rows are one arm's pairs.csv rows."""
    pairs = []
    for r in rows:
        ids = [int(c) for c in r["class_ids"].split()]
        found = [c == "1" for c in r["fwd_found"].split(";")]
        bce_c, dice_c = _floats(r.get("bce_per_channel", "")), _floats(r["dice_per_channel"])
        if len(ids) == 1:
            # a single-trait reference: the step records no per-channel lists (C=1 is the
            # merged cycle), so the pair's own scalars ARE the channel's
            bce_c, dice_c = [float(r["bce"])], [float(r["dice"])]
        fwd_t, cls_t = parse_per_trait(r["fwd_per_trait"]), parse_per_trait(r["cls_per_trait"])
        pairs.append({
            "step": int(r["train_step"]), "ref": r["ref"], "query": r["query"],
            "group": r["group"], "ids": ids, "found": found, "bce_c": bce_c, "dice_c": dice_c,
            "fwd_t": fwd_t, "cls_t": cls_t, "loss": float(r["loss"]),
            "bce": float(r["bce"]), "dice": float(r["dice"]),
            "fwd_miou": float(r["fwd_miou"]) if r["fwd_miou"] else None,
            "cls_miou": float(r["cls_miou"]) if r["cls_miou"] else None,
        })
    n = len(pairs)
    L = np.array([p["loss"] for p in pairs])
    order = np.argsort(L)[::-1]
    have_bce_c = all(len(p["bce_c"]) == len(p["ids"]) for p in pairs)
    for p in pairs:
        if len(p["dice_c"]) != len(p["ids"]):
            raise ValueError(f"step {p['step']}: {len(p['dice_c'])} dice channels for "
                             f"{len(p['ids'])} class ids")

    # per-channel decomposition of the epoch's BCE
    unfound_bce = found_bce = 0.0
    per_trait: dict[int, dict] = {}
    takeovers = []
    for p in pairs:
        C = len(p["ids"])
        for k, t in enumerate(p["ids"]):
            d = per_trait.setdefault(t, {"n": 0, "unfound": 0, "fwd_found": [], "cls_found": [],
                                         "cls_unfound": [], "bce_found": [], "bce_unfound": [],
                                         "dice_found": [], "dice_unfound": []})
            d["n"] += 1
            f = p["found"][k]
            bce_k = p["bce_c"][k] / C if have_bce_c else None    # its share of the pair's BCE
            if f:
                d["fwd_found"].append(p["fwd_t"].get(t, 0.0))
                d["cls_found"].append(p["cls_t"].get(t, 0.0))
                d["dice_found"].append(p["dice_c"][k])
                if bce_k is not None:
                    d["bce_found"].append(bce_k); found_bce += bce_k
                if p["fwd_t"].get(t, 0.0) >= TAKEOVER_FWD and p["cls_t"].get(t, 0.0) < TAKEOVER_CLS:
                    takeovers.append((p["step"], t, p["fwd_t"][t], p["cls_t"].get(t, 0.0)))
            else:
                d["unfound"] += 1
                d["cls_unfound"].append(p["cls_t"].get(t, 0.0))
                d["dice_unfound"].append(p["dice_c"][k])
                if bce_k is not None:
                    d["bce_unfound"].append(bce_k); unfound_bce += bce_k
    total_bce = sum(p["bce"] for p in pairs)
    n_unfound_pairs = sum(1 for p in pairs if not all(p["found"]))

    def m(xs):
        return float(np.mean(xs)) if xs else None

    # correlates of the pair loss
    n_unf = np.array([sum(1 for f in p["found"] if not f) for p in pairs], dtype=float)
    fwd = np.array([p["fwd_miou"] if p["fwd_miou"] is not None else np.nan for p in pairs])

    def corr(a, b):
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
            return None
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    top_rows = []
    for i in order[:top]:
        p = pairs[i]
        tags = []
        unf = [t for t, f in zip(p["ids"], p["found"]) if not f]
        if unf:
            tags.append("unfound " + ",".join(f"t{t}" for t in unf))
        tk = [t for (s, t, _, _) in takeovers if s == p["step"]]
        if tk:
            tags.append("takeover " + ",".join(f"t{t}" for t in tk))
        top_rows.append({"step": p["step"], "ref": p["ref"], "query": p["query"],
                         "loss": p["loss"], "bce": p["bce"], "dice": p["dice"],
                         "fwd_miou": p["fwd_miou"], "cls_miou": p["cls_miou"],
                         "share_of_epoch": p["loss"] / L.sum(),
                         "tags": "; ".join(tags) or "—"})
    q = np.percentile(L, [50, 75, 90, 95])
    return {
        "n_pairs": n,
        "loss": {"mean": float(L.mean()), "median": float(q[0]), "p75": float(q[1]),
                 "p90": float(q[2]), "p95": float(q[3]), "max": float(L.max()),
                 "top5_share": float(np.sort(L)[::-1][:5].sum() / L.sum()),
                 "top10_share": float(np.sort(L)[::-1][:10].sum() / L.sum()),
                 "top20_share": float(np.sort(L)[::-1][:20].sum() / L.sum()),
                 "n_over_3x_median": int((L > 3 * q[0]).sum()),
                 "n_over_10x_median": int((L > 10 * q[0]).sum())},
        "bce_share": {"unfound_channels": (unfound_bce / total_bce) if have_bce_c and total_bce else None,
                      "found_channels": (found_bce / total_bce) if have_bce_c and total_bce else None,
                      "total_bce": total_bce, "total_dice": sum(p["dice"] for p in pairs)},
        "pairs_with_an_unfound_channel": n_unfound_pairs,
        "found_rate": float(np.mean([np.mean(p["found"]) for p in pairs])),
        "per_trait": {str(t): {"n": d["n"], "unfound": d["unfound"],
                               "found_rate": 1 - d["unfound"] / d["n"],
                               "fwd_iou_when_found": m(d["fwd_found"]),
                               "cls_iou_when_found": m(d["cls_found"]),
                               "cls_iou_when_unfound": m(d["cls_unfound"]),
                               "bce_share_when_found": m(d["bce_found"]),
                               "bce_share_when_unfound": m(d["bce_unfound"]),
                               "dice_when_found": m(d["dice_found"]),
                               "dice_when_unfound": m(d["dice_unfound"])}
                      for t, d in sorted(per_trait.items())},
        "takeovers": {"n": len(takeovers),
                      "rule": f"fwd IoU >= {TAKEOVER_FWD} and closing IoU < {TAKEOVER_CLS}",
                      "by_trait": {str(t): sum(1 for _, tt, _, _ in takeovers if tt == t)
                                   for t in sorted(per_trait)}},
        "corr": {"loss_vs_n_unfound": corr(L, n_unf), "loss_vs_fwd_miou": corr(L, fwd)},
        "top": top_rows,
    }


def report(a: dict, arm: str) -> str:
    lo = a["loss"]
    out = [f"[{arm}] {a['n_pairs']} pairs | loss mean {lo['mean']:.3f} median {lo['median']:.3f} "
           f"p90 {lo['p90']:.3f} max {lo['max']:.3f} | top 5/10/20 pairs carry "
           f"{100 * lo['top5_share']:.0f}% / {100 * lo['top10_share']:.0f}% / "
           f"{100 * lo['top20_share']:.0f}% of the epoch | {lo['n_over_3x_median']} pairs > 3x "
           f"median, {lo['n_over_10x_median']} > 10x"]
    bs = a["bce_share"]
    if bs["unfound_channels"] is not None:
        out.append(f"  BCE {bs['total_bce']:.1f} vs Dice {bs['total_dice']:.1f} over the epoch; "
                   f"{100 * bs['unfound_channels']:.0f}% of the BCE sits on UNFOUND channels "
                   f"({a['pairs_with_an_unfound_channel']}/{a['n_pairs']} pairs have one; "
                   f"found-rate {a['found_rate']:.3f})")
    out.append(f"  takeovers (found on the query, lost on the closing leg): {a['takeovers']['n']} "
               f"— by trait {a['takeovers']['by_trait']}")
    out.append(f"  corr(loss, #unfound) {a['corr']['loss_vs_n_unfound']}, "
               f"corr(loss, forward mIoU) {a['corr']['loss_vs_fwd_miou']}")
    out.append("  per trait: found-rate | fwd IoU (found) | cls IoU found / unfound | "
               "BCE share found / unfound")
    for t, d in a["per_trait"].items():
        f = lambda v: "   —" if v is None else f"{100 * v:5.1f}"          # noqa: E731
        g = lambda v: "  —" if v is None else f"{v:.3f}"                 # noqa: E731
        out.append(f"    t{t}: {d['found_rate']:.2f} | {f(d['fwd_iou_when_found'])} | "
                   f"{f(d['cls_iou_when_found'])} / {f(d['cls_iou_when_unfound'])} | "
                   f"{g(d['bce_share_when_found'])} / {g(d['bce_share_when_unfound'])}")
    out.append("  worst pairs:")
    for r in a["top"]:
        out.append(f"    step {r['step']:3d} {r['ref']} -> {r['query']}: loss {r['loss']:.3f} "
                   f"(bce {r['bce']:.3f} dice {r['dice']:.3f}, {100 * r['share_of_epoch']:.1f}% of "
                   f"epoch) fwd {100 * (r['fwd_miou'] or 0):.1f} cls {100 * (r['cls_miou'] or 0):.1f} "
                   f"| {r['tags']}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dataset", default=None, help="for a merged s17 bundle")
    ap.add_argument("--arm", default=None, help="default: every arm in the bundle")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", default=None, help="write the anatomy as JSON here")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.bundle)
    rec = json.loads(z.read("record.json"))
    prefix = ""
    if "datasets" in rec:
        if not args.dataset:
            raise SystemExit(f"--dataset needed: {sorted(rec['datasets'])}")
        prefix = f"{args.dataset}/"
    rows = list(csv.DictReader(io.StringIO(z.read(f"{prefix}pairs.csv").decode())))
    arms = [args.arm] if args.arm else sorted({r["arm"] for r in rows})
    out = {}
    for arm in arms:
        a = anatomy([r for r in rows if r["arm"] == arm], top=args.top)
        out[arm] = a
        print(report(a, arm))
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
