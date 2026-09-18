#!/usr/bin/env python3
"""S17: one dataset's training-pair probe against its fine-tune bundle — per-pair deltas.

    python scripts/combine_s17.py --out s17_beetle.zip --control carrier \
        probe=s17probe_beetle.zip \
        ft:carrier=s17ft_sam2_beetle_carrier.zip ft:found_mask=... ft:nobce=... ft:excl=... \
        [eval:base=s17_beetle_base.zip eval:carrier=... eval:base_tta=... eval:carrier_tta=... ...]

    python scripts/combine_s17.py --merge --out s17.zip \
        beetle=s17_beetle.zip butterfly_major=s17_butterfly_major.zip

`ft:<arm>=` names the fine-tune bundle of each adapter arm the probe scored (the old
`ft=` form serves a single-adapter probe). `--control` names the arm the others are
paired against besides base (the carrier — the S11 crown recipe). `eval:<cell>=` are
run_eval bundles of the test-split evaluation (S11 protocol): `<arm>` plain and
`<arm>_tta`; they are paired per seed through `registry.compare` — every adapter vs
base, every non-control adapter vs the control, TTA vs plain per arm — and the base
cell is pinned to the S11 ledger level so a protocol slip cannot pass as a result.

The probe bundle (`scripts/probe_train_pairs.py`) holds one row per (arm, training pair)
for both cycle legs; the fine-tune bundle (`scripts/run_finetune.py`) holds the adapter's
provenance and `training.losses` per step. This joins the two and pairs every adapter arm
against the base arm PER TRAINING PAIR — the unit here is a cycle pair scored under two
weights, not a support seed, so the bootstrap is `registry.paired_bootstrap` directly
(the same numbers `compare` reports, on a different unit).

Verified before any number is emitted (the gate re-reads the evidence):
  * the probed adapter IS the fine-tune bundle's adapter (`lora_sha`, else the S15 pin:
    same file name and ||dW|| within rel 1e-4);
  * the probe's replay recipe equals the bundle's recorded `occl` block;
  * the replay checks passed and the step-0 loss parity holds (probe.replay);
  * every training pair was probed (`n_pairs == steps_per_epoch`) — a `--max-pairs`
    smoke is combined with `coverage_ok: false`, and the s17 gate refuses it.

`training.losses[train_step]` rides along on every pair: the loss the optimiser saw at
that step (weights moving) beside the frozen-base and frozen-adapter losses on the same
pair — three readings of one number.

Objectives (S17 arms): the probe scores every pair under every objective post hoc, so
each arm's disqualifier ("did the adapter lower its OWN objective on its own pairs")
reads the column named by the arm's recipe (`probe.objective_of`), and the plain
carrier loss is the common yardstick reported for every arm.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl import registry  # noqa: E402
from occl.bundle import export_bundle  # noqa: E402
from occl.finetune.probe import (FORMULAS, base_arm_label, objective_of, own_loss,  # noqa: E402
                                 own_prompt_is_hard, parse_per_trait,
                                 recipe_from_lora_meta, replay_subset, split_objective)
from occl.registry import paired_bootstrap  # noqa: E402

#: The S11 1-shot plain base levels the eval block's base cell is pinned to (±0.5):
#: the same anchor S16 used — a protocol slip cannot masquerade as a result. The 5-shot
#: anchors are S6's plain base cells (S17c's transfer read runs a 5-shot fan).
LEDGER_BASE = {1: {"beetle": 0.6320, "butterfly_major": 0.7030},
               5: {"beetle": 0.7316, "butterfly_major": 0.7178}}
LEDGER_TOL = 0.005

DELTA_FIELDS = [
    "arm", "objective", "base_arm", "train_step", "group", "ref", "query", "class_ids",
    "n_channels", "train_loss",
    "base_fwd_miou", "ft_fwd_miou", "d_fwd_miou",
    "base_fwd_miou_prompted", "ft_fwd_miou_prompted", "d_fwd_miou_prompted",
    "base_cls_miou", "ft_cls_miou", "d_cls_miou",
    "base_loss", "ft_loss", "d_loss",                    # the PLAIN yardstick
    "base_loss_own", "ft_loss_own", "d_loss_own",        # the arm's own objective
    "base_found", "ft_found", "d_found",
    # S18: the closing leg under the BINARISED prompt, and the steganography gap
    # (closing continuous − closing hard) on each side
    "base_cls_miou_hard", "ft_cls_miou_hard", "d_cls_miou_hard",
    "base_stego_gap", "ft_stego_gap", "d_stego_gap",
]
TOP_N = 10
#: S17's post-hoc read, pre-registered for S18: the forward-leg delta on the pairs whose
#: BASE forward prompt was already good (≥ 0.75) — where absorption / the side channel
#: showed as a degradation (carrier −0.51 ✱, arch −1.84 ✱, the 3e-5×3 corner −2.1 ✱).
GOOD_PROMPT = 0.75
POOR_PROMPT = 0.60


def _read(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("record.json"))


def _f(v):
    """CSV/JSON scalar -> float or None."""
    if v is None or v == "":
        return None
    v = float(v)
    return None if math.isnan(v) else v


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _boot(d):
    return paired_bootstrap(np.asarray(d, dtype=float)) if len(d) else None


def same_adapter(probe_arm: dict, ft: dict, training: dict) -> dict:
    """The adapter the probe loaded vs the one the fine-tune bundle wrote.

    `lora_sha` is the pin when both sides carry it. Otherwise the S15 rule: same file
    name and ||dW|| within rel 1e-4 (the same file's norm differs at the 6th decimal
    across GPU types; a retraining moves it by whole units)."""
    sha_p, sha_f = probe_arm.get("lora_sha"), ft.get("lora_sha")
    d_p, d_f = probe_arm.get("delta_norm"), training.get("delta_norm")
    rel = (abs(float(d_p) - float(d_f)) / max(abs(float(d_f)), 1e-12)
           if d_p is not None and d_f is not None else None)
    name_ok = (Path(probe_arm.get("path") or "").name == Path(ft.get("lora_path") or "").name)
    sha_ok = (sha_p == sha_f) if (sha_p and sha_f) else None
    ok = bool(sha_ok) if sha_ok is not None else bool(name_ok and rel is not None and rel <= 1e-4)
    return {"lora_sha_probe": sha_p, "lora_sha_ft": sha_f, "lora_sha_matches": sha_ok,
            "lora_name_matches": name_ok, "delta_norm_probe": d_p, "delta_norm_ft": d_f,
            "delta_norm_rel_err": rel, "adapter_matches": ok}


def _scored(objective: str) -> bool:
    """Whether the probe's post-hoc columns can compose this objective (`own_loss`)."""
    return split_objective(objective)[0] in FORMULAS


def combine_dataset(probe: dict, fts: dict[str, dict], control: str | None = None,
                    stage: str = "s17") -> tuple[dict, list[dict]]:
    """-> (record, delta rows). Pure: no files. ``fts`` maps adapter arm -> its
    fine-tune record; every adapter arm the probe scored must have one."""
    ds = probe["dataset"]
    arms = probe["arm_order"]
    # the base weights under the hard gate, plus (S17d) one base variant per soft gate
    # the adapters trained under; each adapter pairs against the base under ITS gate
    base_arms = [a for a in arms if probe["arms"][a]["path"] is None]
    base = probe.get("base_arm") or next(a for a in base_arms
                                         if probe["arms"][a].get("soft_sentinel") is None)
    adapters = [a for a in arms if a not in base_arms]
    base_of = dict(probe.get("base_of") or {})

    def _variant(a):
        """(soft gate, closing-end rotation) of an arm — the base-variant axes."""
        return (probe["arms"][a].get("soft_sentinel"), probe["arms"][a].get("close_aug"))

    for a in adapters:
        base_of.setdefault(a, base_arm_label(base, *_variant(a)))
        if base_of[a] not in base_arms:
            raise SystemExit(f"adapter {a!r} pairs against {base_of[a]!r}, which the probe "
                             f"did not run (base arms {base_arms})")
        if _variant(base_of[a]) != _variant(a):
            raise SystemExit(f"adapter {a!r} and its base {base_of[a]!r} ran different gates "
                             f"or closing ends ({_variant(a)} vs {_variant(base_of[a])})")
    if not adapters:
        raise SystemExit("the probe has no adapter arm — nothing to pair against base")
    missing = [a for a in adapters if a not in fts]
    if missing:
        raise SystemExit(f"no fine-tune bundle for adapter arm(s) {missing}: pass ft:<arm>=zip")
    if control is not None and control not in adapters:
        raise SystemExit(f"--control {control!r} is not an adapter arm ({adapters})")
    for a, ft in fts.items():
        if not ft.get("training"):
            raise SystemExit(f"ft:{a}: the fine-tune bundle has no training block")
        if ft.get("dataset") != ds:
            raise SystemExit(f"ft:{a}: probe is {ds!r} but the bundle is {ft.get('dataset')!r}")

    # the replay (pairs + legs) must be the same for every arm; objectives may differ
    recipe_matches = all(
        replay_subset(recipe_from_lora_meta({**ft["finetune"], "occl": ft["occl"]}))
        == replay_subset(probe["recipe"]) for ft in fts.values())
    objectives = {a: objective_of(recipe_from_lora_meta({**ft["finetune"], "occl": ft["occl"]}))
                  for a, ft in fts.items()}
    for a in adapters:
        if not _scored(objectives[a]):
            raise SystemExit(f"ft:{a}: objective {objectives[a]!r} is not scored post hoc "
                             f"(formulas {FORMULAS}, optionally +obj<w>) — its disqualifier "
                             "cannot be read")

    replay = dict(probe["replay"])
    ref_training = fts[control or adapters[0]]["training"]
    spe = int(ref_training["steps_per_epoch"])
    coverage_ok = (replay.get("n_probed") == replay.get("n_pairs") == spe
                   and not replay["checks"].get("steps_capped")
                   and all(int(ft["training"]["steps_per_epoch"]) == spe for ft in fts.values()))

    rows_by = {}
    for r in probe["pairs"]:
        rows_by.setdefault(r["arm"], {})[int(r["train_step"])] = r
    n_nan = sum(1 for r in probe["pairs"] for k in ("fwd_miou", "cls_miou", "loss")
                if isinstance(r.get(k), float) and math.isnan(r[k]))

    delta_rows: list[dict] = []
    results: dict = {}
    provenance: dict = {}
    parity: dict = {}
    for arm in adapters:
        ft, training = fts[arm], fts[arm]["training"]
        losses = training.get("losses") or []
        own = objectives[arm]
        # S18: an arm trained under the binary prompt (ste/hard) reads its OWN closing
        # leg and objective off the `_hard` columns; every arm's steganography gap is
        # continuous − hard on both sides
        hard = own_prompt_is_hard(recipe_from_lora_meta({**ft["finetune"], "occl": ft["occl"]}))
        cls_col = "cls_miou_hard" if hard else "cls_miou"
        provenance[arm] = same_adapter(probe["arms"][arm], ft["finetune"], training) | {
            "objective": objectives[arm],
            "probe_objective": (probe["arms"][arm] or {}).get("objective"),
            "own_prompt": "hard" if hard else "continuous",
            "prompt_mode": ft["occl"].get("prompt_mode"),
            "probe_prompt_mode": (probe["arms"][arm] or {}).get("prompt_mode")}
        parity[arm] = (replay.get("parity") or {}).get(arm)
        b_rows, a_rows = rows_by[base_of[arm]], rows_by[arm]
        steps = sorted(set(b_rows) & set(a_rows))
        d_fwd, d_fwdp, d_cls, d_loss, d_found = [], [], [], [], []
        b_fwd, a_fwd, b_fwdp, a_fwdp, b_cls, a_cls, b_loss, a_loss = ([] for _ in range(8))
        b_own, a_own, d_own = [], [], []
        b_dice, a_dice, d_dice = [], [], []
        b_clsc, a_clsc, d_clsc, b_clsh, a_clsh, d_clsh = ([] for _ in range(6))
        b_gap, a_gap, d_gap = [], [], []
        good_bin, poor_bin = [], []
        by_obj: dict[str, list[float]] = {o: [] for o in FORMULAS}
        b_found_rate, a_found_rate, t_loss = [], [], []
        per_trait: dict[int, list[float]] = {}
        per_trait_lvl: dict[int, list[tuple[float, float]]] = {}
        per_group: dict[str, list[float]] = {}
        for s in steps:
            b, a = b_rows[s], a_rows[s]
            bf, af = _f(b["fwd_miou"]), _f(a["fwd_miou"])
            bp, ap_ = _f(b["fwd_miou_prompted"]), _f(a["fwd_miou_prompted"])
            bc, ac = _f(b.get(cls_col)), _f(a.get(cls_col))          # the arm's OWN closing
            bcc, acc = _f(b.get("cls_miou")), _f(a.get("cls_miou"))
            bch, ach = _f(b.get("cls_miou_hard")), _f(a.get("cls_miou_hard"))
            bl, al = _f(b["loss"]), _f(a["loss"])
            nch = max(int(b["n_channels"]), 1)
            bfo, afo = int(b["fwd_found_count"]) / nch, int(a["fwd_found_count"]) / nch
            tl = losses[s] if s < len(losses) else None
            bo, ao = own_loss(b, own, hard=hard), own_loss(a, own, hard=hard)
            bg = (bcc - bch) if None not in (bcc, bch) else None
            ag = (acc - ach) if None not in (acc, ach) else None
            row = {"arm": arm, "objective": objectives[arm], "base_arm": base_of[arm],
                   "train_step": s, "group": b["group"], "ref": b["ref"],
                   "query": b["query"], "class_ids": b["class_ids"],
                   "n_channels": b["n_channels"], "train_loss": tl,
                   "base_loss_own": bo, "ft_loss_own": ao,
                   "d_loss_own": (ao - bo) if None not in (ao, bo) else None,
                   "base_fwd_miou": bf, "ft_fwd_miou": af,
                   "d_fwd_miou": (af - bf) if None not in (af, bf) else None,
                   "base_fwd_miou_prompted": bp, "ft_fwd_miou_prompted": ap_,
                   "d_fwd_miou_prompted": (ap_ - bp) if None not in (ap_, bp) else None,
                   "base_cls_miou": bc, "ft_cls_miou": ac,
                   "d_cls_miou": (ac - bc) if None not in (ac, bc) else None,
                   "base_loss": bl, "ft_loss": al,
                   "d_loss": (al - bl) if None not in (al, bl) else None,
                   "base_found": bfo, "ft_found": afo, "d_found": afo - bfo,
                   "base_cls_miou_hard": bch, "ft_cls_miou_hard": ach,
                   "d_cls_miou_hard": (ach - bch) if None not in (ach, bch) else None,
                   "base_stego_gap": bg, "ft_stego_gap": ag,
                   "d_stego_gap": (ag - bg) if None not in (ag, bg) else None}
            delta_rows.append(row)
            if row["d_fwd_miou"] is not None:
                d_fwd.append(row["d_fwd_miou"]); b_fwd.append(bf); a_fwd.append(af)
                per_group.setdefault(b["group"], []).append(row["d_fwd_miou"])
                if bf >= GOOD_PROMPT:
                    good_bin.append(row["d_fwd_miou"])
                elif bf < POOR_PROMPT:
                    poor_bin.append(row["d_fwd_miou"])
            if None not in (bcc, acc):
                d_clsc.append(acc - bcc); b_clsc.append(bcc); a_clsc.append(acc)
            if row["d_cls_miou_hard"] is not None:
                d_clsh.append(row["d_cls_miou_hard"]); b_clsh.append(bch); a_clsh.append(ach)
            if row["d_stego_gap"] is not None:
                d_gap.append(row["d_stego_gap"]); b_gap.append(bg); a_gap.append(ag)
            if row["d_fwd_miou_prompted"] is not None:
                d_fwdp.append(row["d_fwd_miou_prompted"]); b_fwdp.append(bp); a_fwdp.append(ap_)
            if row["d_cls_miou"] is not None:
                d_cls.append(row["d_cls_miou"]); b_cls.append(bc); a_cls.append(ac)
            if row["d_loss"] is not None:
                d_loss.append(row["d_loss"]); b_loss.append(bl); a_loss.append(al)
                if tl is not None:
                    t_loss.append(tl)
            if row["d_loss_own"] is not None:
                d_own.append(row["d_loss_own"]); b_own.append(bo); a_own.append(ao)
            _dc = "loss_nobce_hard" if hard else "loss_nobce"
            bdc, adc = _f(b.get(_dc)), _f(a.get(_dc))
            if None not in (bdc, adc):
                d_dice.append(adc - bdc); b_dice.append(bdc); a_dice.append(adc)
            for o in FORMULAS:
                bv, av = own_loss(b, o, hard=hard), own_loss(a, o, hard=hard)
                if None not in (bv, av):
                    by_obj[o].append(av - bv)
            d_found.append(row["d_found"]); b_found_rate.append(bfo); a_found_rate.append(afo)
            pb, pa = parse_per_trait(b["fwd_per_trait"]), parse_per_trait(a["fwd_per_trait"])
            for t in set(pb) & set(pa):
                per_trait.setdefault(t, []).append(pa[t] - pb[t])
                per_trait_lvl.setdefault(t, []).append((pb[t], pa[t]))
        # joint deltas for the correlation read: pairs with BOTH legs scored
        joint = [(r["d_fwd_miou"], r["d_cls_miou"], r["base_fwd_miou"])
                 for r in delta_rows if r["arm"] == arm
                 and r["d_fwd_miou"] is not None and r["d_cls_miou"] is not None]
        ranked = sorted((r for r in delta_rows if r["arm"] == arm
                         and r["d_fwd_miou"] is not None),
                        key=lambda r: r["d_fwd_miou"])
        results[arm] = {
            "n_pairs": len(steps),
            "base_arm": base_of[arm],
            "soft_sentinel": probe["arms"][arm].get("soft_sentinel"),
            "close_aug": probe["arms"][arm].get("close_aug"),
            "prompt_mode": ft["occl"].get("prompt_mode"),
            "own_prompt": "hard" if hard else "continuous",
            "forward": {
                "base_mean": _mean(b_fwd), "ft_mean": _mean(a_fwd), "delta": _boot(d_fwd),
                "prompted": {"base_mean": _mean(b_fwdp), "ft_mean": _mean(a_fwdp),
                             "delta": _boot(d_fwdp)},
                # S17's post-hoc bins, pre-registered for S18 (absorption / the side
                # channel show as a degradation on the good-prompt bin)
                "good_prompt_bin": {"threshold": GOOD_PROMPT, "n": len(good_bin),
                                    "delta": _boot(good_bin)},
                "poor_prompt_bin": {"threshold": POOR_PROMPT, "n": len(poor_bin),
                                    "delta": _boot(poor_bin)},
                "per_trait": {str(t): {"n": len(v),
                                       "base": _mean([x for x, _ in per_trait_lvl[t]]),
                                       "ft": _mean([y for _, y in per_trait_lvl[t]]),
                                       "delta": _boot(v)}
                              for t, v in sorted(per_trait.items())},
                "per_group": {g: {"n": len(v), "delta": _boot(v)}
                              for g, v in sorted(per_group.items())},
            },
            # the closing leg under the arm's OWN prompt value (continuous for every
            # S8–S17 arm; the binarised replay for an ste/hard-trained one)
            "closing": {"base_mean": _mean(b_cls), "ft_mean": _mean(a_cls),
                        "delta": _boot(d_cls), "prompt": "hard" if hard else "continuous"},
            "closing_continuous": {"base_mean": _mean(b_clsc), "ft_mean": _mean(a_clsc),
                                   "delta": _boot(d_clsc)},
            "closing_hard": {"base_mean": _mean(b_clsh), "ft_mean": _mean(a_clsh),
                             "delta": _boot(d_clsh)},
            # S18 THE STEGANOGRAPHY READ: closing(continuous prompt) − closing(binary
            # prompt) per pair — what the reconstruction takes from sub-threshold content
            # of the forward logits — at base weights, at the arm's, and its change
            "stego": {"base_gap_mean": _mean(b_gap), "ft_gap_mean": _mean(a_gap),
                      "base_gap": _boot(b_gap), "ft_gap": _boot(a_gap),
                      "delta": _boot(d_gap), "n": len(d_gap)},
            # the arm's OWN objective — what its disqualifier reads
            "loss": {"objective": objectives[arm],
                     "base_mean": _mean(b_own), "ft_mean": _mean(a_own),
                     "delta": _boot(d_own), "train_mean": _mean(t_loss),
                     "train_loss_first": training.get("loss_first"),
                     "train_loss_last": training.get("loss_last")},
            # the plain carrier objective — the yardstick every arm is read on
            "loss_plain": {"base_mean": _mean(b_loss), "ft_mean": _mean(a_loss),
                           "delta": _boot(d_loss)},
            # the BOUNDED reading of the objective: its Dice term alone (per channel in
            # [0, 1], exactly 1 on a sentinel channel) — immune to the ±1024 constants
            # that swing the plain loss by whole tens when the closing head changes what
            # it declares present. The disqualifier reads this (README S17 result).
            "loss_dice": {"base_mean": _mean(b_dice), "ft_mean": _mean(a_dice),
                          "delta": _boot(d_dice)},
            "loss_total": {"base": (sum(b_own) if b_own else None),
                           "ft": (sum(a_own) if a_own else None), "objective": objectives[arm]},
            "loss_by_objective": {o: (_boot(v) or {}).get("mean_delta") for o, v in by_obj.items()},
            "found_rate": {"base": _mean(b_found_rate), "ft": _mean(a_found_rate),
                           "delta": _boot(d_found)},
            "corr_fwd_cls_delta": _pearson([j[0] for j in joint], [j[1] for j in joint]),
            "corr_fwd_delta_vs_base_fwd": _pearson([j[0] for j in joint],
                                                   [j[2] for j in joint]),
            "most_improved": [r["train_step"] for r in ranked[::-1][:TOP_N]],
            "most_degraded": [r["train_step"] for r in ranked[:TOP_N]],
        }

    # arm vs control, per training pair (the improvement arms against the carrier)
    vs_control: dict = {}
    if control is not None:
        c_rows = rows_by[control]
        for arm in adapters:
            if arm == control:
                continue
            a_rows = rows_by[arm]
            steps = sorted(set(c_rows) & set(a_rows))
            def _d(key, cr=c_rows, ar=a_rows, st=steps):
                d = [_f(ar[s].get(key)) - _f(cr[s].get(key)) for s in st
                     if _f(ar[s].get(key)) is not None and _f(cr[s].get(key)) is not None]
                return _boot(d)
            vs_control[arm] = {"control": control, "n_pairs": len(steps),
                               "forward": _d("fwd_miou"), "prompted": _d("fwd_miou_prompted"),
                               "closing": _d("cls_miou"), "loss_plain": _d("loss"),
                               "found_count": _d("fwd_found_count"),
                               "closing_hard": _d("cls_miou_hard")}

    # the gate's own effect at BASE weights (S17d): each soft base variant vs the hard
    # base, per pair — what the soft sentinel changes before any training
    def _pair_rows(x_rows, y_rows, key):
        st = sorted(set(x_rows) & set(y_rows))
        d = [_f(y_rows[s][key]) - _f(x_rows[s][key]) for s in st
             if _f(y_rows[s][key]) is not None and _f(x_rows[s][key]) is not None]
        return _boot(d)
    gate_effect: dict = {}
    for bv in base_arms:
        if bv == base:
            continue
        gate_effect[bv] = {"soft_sentinel": probe["arms"][bv].get("soft_sentinel"),
                           "close_aug": probe["arms"][bv].get("close_aug"),
                           "n_pairs": len(set(rows_by[base]) & set(rows_by[bv])),
                           "forward": _pair_rows(rows_by[base], rows_by[bv], "fwd_miou"),
                           "closing": _pair_rows(rows_by[base], rows_by[bv], "cls_miou"),
                           "loss_plain": _pair_rows(rows_by[base], rows_by[bv], "loss"),
                           "loss_dice": _pair_rows(rows_by[base], rows_by[bv], "loss_nobce"),
                           "found_count": _pair_rows(rows_by[base], rows_by[bv],
                                                     "fwd_found_count"),
                           "loss_plain_mean": {"base": _mean([_f(r["loss"]) for r in rows_by[base].values()]),
                                               "soft": _mean([_f(r["loss"]) for r in rows_by[bv].values()])}}
    for arm, vc in vs_control.items():
        vc["same_gate"] = _variant(arm) == _variant(control)
        vc["same_prompt"] = (probe["arms"][arm].get("prompt_mode")
                             == probe["arms"][control].get("prompt_mode"))

    primary = control or adapters[0]
    record = {
        "stage": stage, "dataset": ds, "family": probe["family"],
        "model": probe.get("model"),
        "finetune": {a: {"finetune": ft.get("finetune"), "occl": ft.get("occl"),
                         "training": ft["training"], "diagnostics": ft.get("diagnostics"),
                         "objective": objectives[a]} for a, ft in fts.items()},
        "probe": {k: v for k, v in probe.items() if k != "pairs"},
        "recipe": probe["recipe"], "objectives": objectives,
        "replay": replay | {"recipe_matches": recipe_matches, "coverage_ok": coverage_ok,
                            "steps_per_epoch": spe, "parity": parity},
        "arms": probe["arms"], "base_arm": base, "base_arms": base_arms,
        "base_of": base_of, "adapter_arms": adapters,
        "control_arm": control,
        "provenance": provenance,
        "vs_control": vs_control,
        "gate_effect": gate_effect,
        "n_nan": n_nan,
        "results": {**results,
                    "primary": primary,
                    "n_pairs": results[primary]["n_pairs"],
                    "forward_delta": (results[primary]["forward"]["delta"] or {}).get("mean_delta"),
                    "forward_significant": (results[primary]["forward"]["delta"] or {}).get("significant"),
                    "closing_delta": (results[primary]["closing"]["delta"] or {}).get("mean_delta"),
                    "closing_significant": (results[primary]["closing"]["delta"] or {}).get("significant"),
                    "loss_delta": (results[primary]["loss"]["delta"] or {}).get("mean_delta"),
                    "loss_significant": (results[primary]["loss"]["delta"] or {}).get("significant"),
                    "found_rate_delta": (results[primary]["found_rate"]["delta"] or {}).get("mean_delta")},
    }
    return record, delta_rows


def _print_summary(rec: dict) -> None:
    ds = rec["dataset"]
    rp = rec["replay"]
    print(f"[{ds}] replay: {rp.get('n_probed')}/{rp.get('n_pairs')} pairs probed, "
          f"steps_per_epoch {rp.get('steps_per_epoch')}, coverage_ok={rp['coverage_ok']}, "
          f"recipe_matches={rp['recipe_matches']}, loss parity rel err (max over arms) "
          f"{rp.get('loss_first_rel_err')}")
    for arm in rec["adapter_arms"]:
        r, pv = rec["results"][arm], rec["provenance"][arm]
        print(f"[{ds}] {arm} (objective {pv['objective']}): adapter_matches="
              f"{pv['adapter_matches']} (sha {pv['lora_sha_matches']}, ||dW|| rel "
              f"{pv['delta_norm_rel_err']})")
        for leg, blk in (("forward", r["forward"]), ("closing", r["closing"]),
                         ("own loss", r["loss"]), ("plain loss", r["loss_plain"])):
            d = blk["delta"]
            if d is None:
                print(f"    {leg:>10s}: no pairs")
                continue
            scale = 1.0 if "loss" in leg else 100.0
            print(f"    {leg:>10s}: base {scale * blk['base_mean']:.3f} -> ft "
                  f"{scale * blk['ft_mean']:.3f} | delta {scale * d['mean_delta']:+.3f} "
                  f"[{scale * d['ci95'][0]:+.3f},{scale * d['ci95'][1]:+.3f}]"
                  f"{'*' if d['significant'] else ''} ({d['n_improved']}/{d['n']} up)")
        p = r["forward"]["prompted"]["delta"]
        if p:
            print(f"    prompted: delta {100 * p['mean_delta']:+.3f} "
                  f"[{100 * p['ci95'][0]:+.3f},{100 * p['ci95'][1]:+.3f}]"
                  f"{'*' if p['significant'] else ''}")
        fr = r["found_rate"]
        print(f"    found-rate: base {fr['base']:.3f} -> ft {fr['ft']:.3f}; "
              f"corr(fwd d, cls d) = {r['corr_fwd_cls_delta']}")
        st = r.get("stego") or {}
        if st.get("delta"):
            print(f"    stego gap (closing continuous − hard, own prompt {r['own_prompt']}): "
                  f"base {100 * st['base_gap_mean']:+.2f} -> ft {100 * st['ft_gap_mean']:+.2f} | "
                  f"delta {100 * st['delta']['mean_delta']:+.2f} "
                  f"[{100 * st['delta']['ci95'][0]:+.2f},{100 * st['delta']['ci95'][1]:+.2f}]"
                  f"{'*' if st['delta']['significant'] else ''}")
        for name in ("good_prompt_bin", "poor_prompt_bin"):
            gb = r["forward"].get(name) or {}
            if gb.get("delta"):
                d = gb["delta"]
                print(f"    forward on the {name.replace('_', ' ')} (n={gb['n']}): "
                      f"{100 * d['mean_delta']:+.2f} [{100 * d['ci95'][0]:+.2f},"
                      f"{100 * d['ci95'][1]:+.2f}]{'*' if d['significant'] else ''}")
        for t, v in r["forward"]["per_trait"].items():
            d = v["delta"]
            print(f"    trait {t}: n={v['n']} {100 * v['base']:.2f} -> {100 * v['ft']:.2f} "
                  f"({100 * d['mean_delta']:+.2f}{'*' if d['significant'] else ''})")
    for arm, vc in (rec.get("vs_control") or {}).items():
        f, c = vc["forward"], vc["closing"]
        print(f"[{ds}] {arm} vs {vc['control']} per pair: forward "
              f"{100 * f['mean_delta']:+.3f} [{100 * f['ci95'][0]:+.3f},{100 * f['ci95'][1]:+.3f}]"
              f"{'*' if f['significant'] else ''} | closing {100 * c['mean_delta']:+.3f}"
              f"{'*' if c['significant'] else ''}"
              + ("" if vc.get("same_gate", True) else " (gates differ)"))
    for bv, ge in (rec.get("gate_effect") or {}).items():
        f, c, lp = ge["forward"], ge["closing"], ge["loss_plain"]
        print(f"[{ds}] gate effect at base weights, {bv} (K={ge['soft_sentinel']}, "
              f"close_aug={ge.get('close_aug')}) vs "
              f"{rec['base_arm']}: forward {100 * f['mean_delta']:+.3f}"
              f"{'*' if f['significant'] else ''} | closing {100 * c['mean_delta']:+.3f}"
              f"{'*' if c['significant'] else ''} | plain loss "
              f"{ge['loss_plain_mean']['base']:.3f} -> {ge['loss_plain_mean']['soft']:.3f} "
              f"({lp['mean_delta']:+.3f}{'*' if lp['significant'] else ''}) | found-count "
              f"{ge['found_count']['mean_delta']:+.3f}")
    ev = rec.get("eval")
    if ev:
        print(f"[{ds}] test split (S11 protocol): base {100 * ev['base_pin']['level']:.2f} vs "
              f"ledger {100 * ev['base_pin']['ledger']:.2f} "
              f"({'within' if ev['base_pin']['within'] else 'OUTSIDE'} ±0.5)")
        if ev.get("base_pin_5"):
            p5 = ev["base_pin_5"]
            print(f"[{ds}] 5-shot base {100 * p5['level']:.2f} vs S6 ledger {100 * p5['ledger']:.2f} "
                  f"({'within' if p5['within'] else 'OUTSIDE'} ±0.5)")
        for k, v in ev["cells"].items():
            print(f"    {k:>16s}: {100 * v['miou']:.2f} ± {100 * v['std']:.2f}"
                  + (f" | best3 {100 * v['miou_best3']:.2f}" if v.get("miou_best3") else ""))
        for k, v in ev["pairings"].items():
            print(f"    {k:>28s}: {100 * v['delta']:+.2f} [{100 * v['ci95'][0]:+.2f},"
                  f"{100 * v['ci95'][1]:+.2f}]{'*' if v['significant'] else ''} "
                  f"{v['n_seeds_improved']}/{v['n_seeds']} seeds"
                  + (f" | best3 {100 * v['delta_best3']:+.2f}"
                     f"{'*' if v['significant_best3'] else ''}" if v.get("delta_best3") is not None else ""))


def _eval_pair(t, b, name, ignore=()):
    kw = {"ignore_fields": tuple(ignore)} if ignore else {}
    try:
        cmp = registry.compare(t, b, **kw)
    except ValueError as e:
        raise SystemExit(f"{name}: cannot pair: {e}") from e
    try:
        cmp3 = registry.compare(t, b, metric="miou_boundary.independent_best3", **kw)
    except ValueError:
        cmp3 = None
    return {"delta": cmp["mean_delta"], "ci95": cmp["ci95"], "significant": cmp["significant"],
            "n_seeds": cmp["n_seeds"], "n_seeds_improved": cmp["n_seeds_improved"],
            "delta_best3": cmp3["mean_delta"] if cmp3 else None,
            "ci95_best3": cmp3["ci95"] if cmp3 else None,
            "significant_best3": cmp3["significant"] if cmp3 else None,
            "ignored_fields": cmp.get("ignored_fields")}


def _lift(rec, block, key):
    vals = [r[block][key] for r in rec.get("runs", []) if (r.get(block) or {}).get(key) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _parse_cell(name: str) -> tuple[str, int, bool]:
    """``"<arm>[_5][_tta]"`` -> (arm, shot, tta). The shot suffix is S17c's 5-shot fan."""
    tta = name.endswith("_tta")
    if tta:
        name = name[:-4]
    shot = 1
    if name.endswith("_5"):
        shot, name = 5, name[:-2]
    return name, shot, tta


def combine_eval(cells: dict[str, dict], ds: str, adapters: list[str],
                 control: str | None) -> dict:
    """The test-split block: run_eval cells ``<arm>[_5][_tta]``, paired per seed WITHIN a
    shot count (compare refuses cross-shot pairs; S16's nested-shot licence is not used).

    `base` (1-shot plain) is required and pinned to the S11 ledger; `base_5` is pinned to
    S6's when present; every other cell is optional — pairings form for what is there.
    """
    if "base" not in cells:
        raise SystemExit("eval: the plain `base` cell is required (the ledger pin)")
    out_cells = {}
    for name, rec in cells.items():
        arm, shot, tta = _parse_cell(name)
        got = (rec.get("protocol") or {}).get("shots")
        if got != shot:
            raise SystemExit(f"eval:{name}: the record ran {got} shots, the name says {shot}")
        ft = rec.get("finetune") or {}
        out_cells[name] = {
            "arm": arm, "shot": shot, "tta": tta,
            "miou": rec["results"]["miou"], "std": rec["results"]["std"],
            "n_seeds": (rec.get("protocol") or {}).get("n_runs"),
            "support_set_hash": (rec.get("protocol") or {}).get("support_set_hash"),
            "lora_path": Path(ft.get("lora_path")).name if ft.get("lora_path") else None,
            "delta_norm": (ft.get("lora") or {}).get("delta_norm"),
            "miou_best3": _lift(rec, "miou_boundary", "independent_best3"),
            "miou_oracle": _lift(rec, "miou_variants", "oracle"),
        }

    def _pin(shot):
        key = "base" if shot == 1 else f"base_{shot}"
        if key not in cells:
            return None
        level = cells[key]["results"]["miou"]
        ledger = LEDGER_BASE.get(shot, {}).get(ds)
        return {"level": level, "ledger": ledger,
                "within": (abs(level - ledger) <= LEDGER_TOL) if ledger is not None else None}

    pairings = {}
    shots = sorted({_parse_cell(n)[1] for n in cells})
    for shot in shots:
        sfx = "" if shot == 1 else f"_{shot}"
        for tta in ("", "_tta"):
            b = cells.get(f"base{sfx}{tta}")
            for arm in adapters:
                a = cells.get(f"{arm}{sfx}{tta}")
                if a is None or b is None:
                    continue
                pairings[f"{arm}{sfx}{tta}_vs_base{sfx}{tta}"] = _eval_pair(
                    a, b, f"{arm}{sfx}{tta} vs base{sfx}")
                if control and arm != control and (c := cells.get(f"{control}{sfx}{tta}")) is not None:
                    pairings[f"{arm}{sfx}{tta}_vs_{control}{sfx}{tta}"] = _eval_pair(
                        a, c, f"{arm}{sfx}{tta} vs {control}{sfx}")
        for arm in ["base", *adapters]:
            if f"{arm}{sfx}" in cells and f"{arm}{sfx}_tta" in cells:
                pairings[f"{arm}{sfx}_tta_vs_plain"] = _eval_pair(
                    cells[f"{arm}{sfx}_tta"], cells[f"{arm}{sfx}"], f"{arm}{sfx} tta vs plain",
                    ignore=("model.tta",))
    return {"cells": out_cells, "base_pin": _pin(1), "base_pin_5": _pin(5),
            "shots": shots, "pairings": pairings,
            "protocol": {k: v for k, v in (cells["base"].get("protocol") or {}).items()
                         if k not in ("support_set_hash", "shots")}}


def _carry(zip_path, prefix: str, tmp: Path, extra: dict) -> None:
    """Copy a bundle's pairs.csv and maps/ into `extra` under `prefix`."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name in ("pairs.csv", "deltas.csv") or name.startswith("maps/"):
                dest = tmp / prefix / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(name))
                extra[f"{prefix}{name}" if prefix else name] = dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", metavar="label=bundle.zip")
    ap.add_argument("--stage", default="s17")
    ap.add_argument("--merge", action="store_true",
                    help="labels are DATASET names of per-dataset s17 combines")
    ap.add_argument("--control", default=None,
                    help="the adapter arm the improvement arms are paired against "
                         "(the carrier); optional")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = {}
    for s in args.inputs:
        label, _, path = s.partition("=")
        spec[label] = path
    tmp = Path(tempfile.mkdtemp(prefix="s17combine_"))
    extra: dict = {}

    if args.merge:
        per_ds = {}
        for ds, path in spec.items():
            r = _read(path)
            if r.get("dataset") != ds:
                raise SystemExit(f"{ds}: record is for {r.get('dataset')!r}")
            per_ds[ds] = r
            _carry(path, f"{ds}/", tmp, extra)
        record = {
            "stage": args.stage, "family": "sam2", "datasets": per_ds,
            "results": {ds: {**{k: r["results"][k] for k in (
                "primary", "n_pairs", "forward_delta", "forward_significant",
                "closing_delta", "closing_significant", "loss_delta",
                "loss_significant", "found_rate_delta")},
                "eval_deltas": r["results"].get("eval_deltas")}
                for ds, r in per_ds.items()},
        }
        dest = export_bundle(args.stage, record, out=args.out, extra_files=extra)
        print(f"wrote {dest}")
        return 0

    if "probe" not in spec:
        raise SystemExit("need probe=<s17probe zip>")
    probe = _read(spec["probe"])
    adapters = [a for a in probe["arm_order"] if probe["arms"][a]["path"] is not None]
    fts = {k.split(":", 1)[1]: _read(v) for k, v in spec.items() if k.startswith("ft:")}
    if "ft" in spec:                       # the single-adapter form
        if len(adapters) != 1:
            raise SystemExit(f"`ft=` needs a single-adapter probe; this one has {adapters} "
                             "— pass ft:<arm>=zip per arm")
        fts[adapters[0]] = _read(spec["ft"])
    unknown = sorted(k for k in spec if not (k in ("probe", "ft") or k.startswith(("ft:", "eval:"))))
    if unknown:
        raise SystemExit(f"unknown inputs {unknown}: use probe=, ft:<arm>=, eval:<cell>=")
    record, delta_rows = combine_dataset(probe, fts, control=args.control, stage=args.stage)
    ev_cells = {k.split(":", 1)[1]: _read(v) for k, v in spec.items() if k.startswith("eval:")}
    if ev_cells:
        record["eval"] = combine_eval(ev_cells, record["dataset"], record["adapter_arms"],
                                      args.control)
        pr = record["results"]
        for k, v in record["eval"]["pairings"].items():
            pr.setdefault("eval_deltas", {})[k] = v["delta"]
    _print_summary(record)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=DELTA_FIELDS)
    w.writeheader()
    w.writerows(delta_rows)
    (tmp / "deltas.csv").write_text(buf.getvalue())
    extra["deltas.csv"] = tmp / "deltas.csv"
    _carry(spec["probe"], "", tmp, extra)
    dest = export_bundle(args.stage, record, out=args.out, extra_files=extra)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
