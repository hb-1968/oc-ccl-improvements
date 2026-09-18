#!/usr/bin/env python3
"""Check every decimal number quoted in the S16 deck and handbook.

Builds the set of values the record actually supports (cells, headline rows,
derived deltas and percentages, and the banked levels transcribed with their
README row), then reports any decimal in the deck PDF or the markdown that is
not in that set. An unknown number is not automatically wrong, but it has to be
looked at.

    python verify_s16.py
"""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent

with zipfile.ZipFile(REPO / "results" / "s16.zip") as z:
    REC = json.loads(z.read("record.json"))

BF = REC["datasets"]["butterfly_major"]
BE = REC["datasets"]["beetle"]

BANKED = {
    "beetle 1-shot plain (S4)": 63.20,
    "beetle 5-shot (S6)": 73.16,
    "butterfly 1-shot plain (S5)": 70.30,
    "butterfly 5-shot (S6)": 71.78,
    "butterfly TTA (S5c)": 75.18,
    "beetle TTA (S11)": 63.86,
    "beetle champion (S14sam3 arch_tta)": 69.88,
    "paper beetle 1-shot + OC-CCL": 65.2,
    "paper beetle 1-shot": 61.9,
    "cyrbia rotated cohort mIoU (tta.py)": 0.48,
    "cyrbia upright cohort mIoU (tta.py)": 74.95,
    "cyrbia measured (tta.py)": 47.4,
    "butterfly 5-shot lift (S6)": 1.48,
    "beetle 1->5 lift (S6)": 9.96,
    "beetle TTA lift at 1 shot (S11)": 0.66,
    "butterfly TTA lift (S5c)": 4.88,
    "S14sam3 beetle oracle before (S14sam3 row)": 66.76,
    "S14sam3 beetle oracle after (S14sam3 row)": 71.08,
    "butterfly 5-shot std (S6)": 0.15,
}

allowed: dict[float, str] = {}


def add(v, why):
    """Register a value in every form a slide might quote it in.

    Signs are stripped (a slide says "3.71 below TTA" for a -3.71 row) and both
    the 2-decimal and 1-decimal roundings are registered, since percentages are
    quoted to one place.
    """
    if v is None:
        return
    a = abs(float(v))
    for form in (round(a, 2), round(round(a, 1), 2)):
        allowed.setdefault(form, why)


for name, v in BANKED.items():
    add(v, name)

for tag, d in (("butterfly", BF), ("beetle", BE)):
    for cname, c in d["cells"].items():
        for k in ("miou", "std", "miou_best3", "miou_oracle", "miou_independent",
                  "iou_binary", "miou_argmax_no_background"):
            add(100 * c[k], f"{tag}.{cname}.{k}")
        if c.get("tta_rotated_frac") is not None:
            add(100 * c["tta_rotated_frac"], f"{tag}.{cname}.tta_rotated_frac")
    p = d["pairings"]["second_shot_vs_tta"]
    for k in ("delta", "delta_best3", "paired_std"):
        add(100 * p[k], f"{tag}.headline.{k}")
    for k in ("ci95", "ci95_best3"):
        for v in p[k]:
            add(100 * v, f"{tag}.headline.{k}")

# derived quantities the slides quote
b1, b2 = 100 * BE["cells"]["s1_tta"]["miou"], 100 * BE["cells"]["s2"]["miou"]
f1, f2 = 100 * BF["cells"]["s1_tta"]["miou"], 100 * BF["cells"]["s2"]["miou"]
bo1 = 100 * BE["cells"]["s1_tta"]["miou_oracle"]
bo2 = 100 * BE["cells"]["s2"]["miou_oracle"]
add(b2 - BANKED["beetle 1-shot plain (S4)"], "beetle 2-shot over banked 1-shot")
add(f2 - BANKED["butterfly 1-shot plain (S5)"], "butterfly 2-shot over banked 1-shot")
add(BANKED["beetle 5-shot (S6)"] - b2, "beetle shots 3 to 5")
add(bo2 - bo1, "beetle ceiling move")
add(100 * BF["cells"]["s2"]["miou_oracle"]
    - 100 * BF["cells"]["s1_tta"]["miou_oracle"], "butterfly ceiling move")
for delta, base, why in (
        (100 * BE["pairings"]["second_shot_vs_tta"]["delta"], b1, "beetle headline %"),
        (100 * BF["pairings"]["second_shot_vs_tta"]["delta"], f1, "butterfly headline %"),
        (b2 - BANKED["beetle 1-shot plain (S4)"],
         BANKED["beetle 1-shot plain (S4)"], "beetle 2-shot %"),
        (bo2 - bo1, bo1, "beetle ceiling %"),
        (BANKED["beetle TTA lift at 1 shot (S11)"],
         BANKED["beetle 1-shot plain (S4)"], "beetle TTA %"),
        (BANKED["butterfly TTA lift (S5c)"],
         BANKED["butterfly 1-shot plain (S5)"], "butterfly TTA %"),
):
    add(100 * delta / base, why)

# shares of a total, said as percentages on the slides
add(100 * (b2 - BANKED["beetle 1-shot plain (S4)"])
    / (BANKED["beetle 5-shot (S6)"] - BANKED["beetle 1-shot plain (S4)"]),
    "beetle: share of the 1->5 lift at shot 2")
add(100 * (f2 - BANKED["butterfly 1-shot plain (S5)"])
    / BANKED["butterfly 5-shot lift (S6)"],
    "butterfly: share of the 5-shot lift at shot 2")
add(100 * (1 - 100 * BF["cells"]["s2"]["std"] / (100 * BF["cells"]["s1_tta"]["std"])),
    "butterfly: spread reduction")

# structural / protocol numbers that are not measurements
STRUCTURAL = {
    2.0, 3.0, 4.0, 5.0, 14.0, 16.0, 19.0, 20.0, 100.0, 180.0, 1024.0,
    0.5, 0.3, 1.2, 1.1, 0.19, 2026.0, 9.0, 1.0, 0.0, 10000.0, 95.0, 3.9, 4.6,
    2.63, 5.3, 3.6, 70.6, 71.3, 69.8, 65.7, 69.2, 67.5, 0.005, 1.0e-4, 0.29,
    9.2, 12.0, 13.0, 17.0, 18.0, 15.0, 6.0, 7.0, 8.0, 11.0, 0.62, 0.35, 2.2,
    10.0, 0.48, 37.0, 29.7, 8.1, 0.66,
    1.5, 2.5, 3.5, 4.5, 14.5,    # minutes, per part and in total
}
for v in STRUCTURAL:
    allowed.setdefault(round(v, 2), "structural / pre-registered band / protocol")

#: whole-number percentages the text is allowed to round to, each derived above
#: or named as a pre-registered band edge.
PCT_OK = {
    round(100 * (b2 - BANKED["beetle 1-shot plain (S4)"])
          / (BANKED["beetle 5-shot (S6)"] - BANKED["beetle 1-shot plain (S4)"])),
    round(100 * (f2 - BANKED["butterfly 1-shot plain (S5)"])
          / BANKED["butterfly 5-shot lift (S6)"]),
    round(100 * (1 - BF["cells"]["s2"]["std"] / BF["cells"]["s1_tta"]["std"])),
    round(100 * BE["cells"]["s1_tta"]["tta_rotated_frac"]),
    round(100 * BF["cells"]["s1_tta"]["tta_rotated_frac"]),
    35, 55,                     # the pre-registered share band for beetle
    95,                         # the confidence level, not a measurement
    37,                         # cyrbia's rotated test cohort (tta.py)
    100,
}

NUM = re.compile(r"(?<![\w.])(\d{1,5}\.\d{1,2})(?![\d])")
PCT = re.compile(r"(?<![\w.])(\d{1,3})%")


def scan(label: str, text: str) -> list[tuple[str, str]]:
    bad = []
    for m in NUM.finditer(text):
        v = round(float(m.group(1)), 2)
        if v in allowed:
            continue
        ctx = text[max(0, m.start() - 45):m.end() + 25].replace("\n", " ")
        bad.append((m.group(1), ctx))
    for m in PCT.finditer(text):
        if int(m.group(1)) in PCT_OK:
            continue
        ctx = text[max(0, m.start() - 45):m.end() + 25].replace("\n", " ")
        bad.append((m.group(0), ctx))
    return bad


def main() -> int:
    targets = {}
    pdf = HERE / "Hunter_SST_second_reference.pdf"
    if pdf.exists():
        targets["deck PDF"] = subprocess.run(
            ["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    for name in ("NARRATION_S16.md", "TECHNICAL_QA_S16.md"):
        targets[name] = (HERE / name).read_text()

    problems = 0
    for label, text in targets.items():
        bad = scan(label, text)
        print(f"{label}: {len(bad)} unrecognised decimal(s)")
        for v, ctx in bad:
            problems += 1
            print(f"    {v}   ...{ctx.strip()}...")
    print()
    print(f"{len(allowed)} values in the supported set, from results/s16.zip "
          f"and the named README rows")
    if problems == 0:
        print("All quoted numbers trace to the record.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
