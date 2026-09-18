#!/usr/bin/env python3
"""Generate the Beetle 120/60 split ONCE, then commit it and never run this again.

The paper reports 10 train / 5 test per species across 12 species (120/60), but the
assignment is unpublished and cannot be recovered from the repo: every URL in
`metadata.json` points into the same `.../splits_80_20/test/...` directory, which is an
unrelated split of the whole NEON corpus.

So this is *our* split. Regenerating it would silently shift every historical Beetle
number, so the script refuses to overwrite an existing file without --force.

Usage:
    python scripts/make_beetle_split.py            # writes if absent
    python scripts/make_beetle_split.py --force    # DANGER: invalidates prior results
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from occl.datasets.beetle import SPLIT_FILE, read_raw_metadata
from occl.seeding import derive_seed

N_TRAIN, N_TEST = 10, 5


def build() -> dict:
    by_species: dict[str, list[str]] = collections.defaultdict(list)
    for image_id, _, _ in read_raw_metadata():
        by_species[image_id.split("/")[0]].append(image_id)

    rng = np.random.default_rng(derive_seed("beetle", "__split__", 0, 0))
    train, test, trimmed = [], [], []

    for species in sorted(by_species):
        ids = sorted(by_species[species])
        if len(ids) > N_TRAIN + N_TEST:
            # restrictus has 16; drop the extras so the shape matches the paper's 12x15.
            keep = [ids[int(i)] for i in rng.permutation(len(ids))[: N_TRAIN + N_TEST]]
            trimmed += sorted(set(ids) - set(keep))
            ids = sorted(keep)
        if len(ids) != N_TRAIN + N_TEST:
            raise ValueError(f"{species}: expected {N_TRAIN + N_TEST} masks, got {len(ids)}")
        perm = [ids[int(i)] for i in rng.permutation(len(ids))]
        train += sorted(perm[:N_TRAIN])
        test += sorted(perm[N_TRAIN:])

    return {
        "_comment": (
            "SYNTHESIZED SPLIT — not the paper's. The authors' 120/60 assignment is "
            "unpublished and unrecoverable from this repo. Generated once by "
            "scripts/make_beetle_split.py; do not regenerate."
        ),
        "seed_key": "occl/v1|beetle|__split__|shot=0|run=0",
        "seed": derive_seed("beetle", "__split__", 0, 0),
        "n_train": len(train),
        "n_test": len(test),
        "trimmed": trimmed,
        "train": sorted(train),
        "test": sorted(test),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = ap.parse_args()

    if SPLIT_FILE.exists() and not args.force:
        cur = json.load(open(SPLIT_FILE))
        print(f"{SPLIT_FILE} already exists ({cur['n_train']}/{cur['n_test']}). "
              "Refusing to regenerate — that would invalidate every prior Beetle number. "
              "Pass --force only if you mean it.")
        return

    split = build()
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=2) + "\n")
    print(f"wrote {SPLIT_FILE}: {split['n_train']} train / {split['n_test']} test, "
          f"trimmed {len(split['trimmed'])} ({', '.join(split['trimmed']) or 'none'})")


if __name__ == "__main__":
    main()
