"""Deterministic, model-independent support sampling.

Two properties carry the whole comparison story:

1. **Model-independent.** The seed depends only on (dataset, group, run_idx) -- never on
   the adapter, dtype or device. Run 7 of Butterfly Major therefore draws the same
   support image for pretrained SAM 2, every OC-CCL variant, and later SAM 3. That makes
   differences *paired*: support-set difficulty is the dominant variance component and
   cancels exactly when you difference per-seed, which is the only way an effect as
   small as the paper's +0.2 on Major is detectable at n=20.

2. **Prefix-nested.** The seed key pins ``shot=0`` and the caller slices ``[:shot]`` from
   a permutation, so the 20 one-shot supports are exactly the first elements of the 20
   five-shot support sets. The 1-shot -> 5-shot comparison is then paired too, and the
   number of distinct train images to download does not grow.

``sorted()`` before permuting is load-bearing: filesystem ordering is not stable across
machines, and without it "reproducible" would be a lie.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence

import numpy as np

#: Bump this if the sampling scheme ever changes, so old and new runs cannot silently
#: be compared as though they shared support sets.
SEED_NAMESPACE = "occl/v1"


def derive_seed(dataset: str, group: str, shot: int, run_idx: int) -> int:
    """Stable 31-bit seed from a human-readable key. No global RNG state involved."""
    key = f"{SEED_NAMESPACE}|{dataset}|{group}|shot={shot}|run={run_idx}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def sample_supports(
    train_ids: Iterable[str], dataset: str, group: str, shot: int, run_idx: int
) -> list[str]:
    """Draw ``shot`` support IDs for one run.

    Note the literal ``0`` passed as the seed's ``shot`` field -- that is what makes
    1-shot a strict prefix of 5-shot. Do not "fix" it to ``shot``.
    """
    ids = sorted(str(x) for x in train_ids)
    if shot > len(ids):
        raise ValueError(f"asked for {shot} supports but only {len(ids)} train ids exist")
    rng = np.random.default_rng(derive_seed(dataset, group, 0, run_idx))
    perm = rng.permutation(len(ids))
    return [ids[int(i)] for i in perm[:shot]]


def support_set_hash(support_ids_by_run: dict[int, Sequence[str]] | Sequence[Sequence[str]]) -> str:
    """Hash the *actually chosen* support IDs across all runs.

    Deliberately not a hash of the seed. Two runs can share seeds yet sample different
    supports if the image manifest differs (one image failed to download, so it was
    filtered out of ``train_ids``). Only hashing the resolved IDs catches that, and this
    value gates whether two results may appear in the same table.
    """
    if isinstance(support_ids_by_run, dict):
        payload = {str(k): [str(x) for x in v] for k, v in sorted(support_ids_by_run.items())}
    else:
        payload = {str(i): [str(x) for x in v] for i, v in enumerate(support_ids_by_run)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
