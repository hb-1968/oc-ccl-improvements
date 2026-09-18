"""Episode construction — the paper's one-shot and five-shot protocols.

**One-shot.** An independent 2-frame pseudo-video ``{x0, xn}`` per query, propagated once.
The paper is explicit that the headline numbers use short videos, not one long one:

    "To eliminate the uncertainty in creating pseudo-videos, we implement SST by
    constructing multiple short, two-frame videos, unless stated otherwise."

This also keeps the setting *inductive*: each test image is segmented independently, as
conventional few-shot segmentation methods do, with no leakage between queries. Note that
`src/sst/sam_utils.py::load_masks` in the SST repo does the opposite -- it builds one long
video, so query *n* sees memory from queries 1..n-1 and drifts. Only the repo's ablation
scripts implement the paper's construction.

**Five-shot.** One **6-frame** sequence ``{x0..x4, xn}`` per query, with all five supports
prompted at frames 0-4 -- not five separate passes merged:

    "given labeled reference pairs (x0,y0)...(x4,y4) and unlabeled target images
    {x5,...,xN}, we create sequences {x0,...,x4,x5},...,{x0,...,x4,xN}"

The paper calls this design "naive" and leaves better multi-shot mechanisms to future work.
"""

from __future__ import annotations

from typing import Iterator

from .curation import draw_supports, support_pool
from .datasets.base import Episode, union_class_ids


def build_run(ds, group: str, shot: int, run_idx: int,
              support_selection: str = "uniform") -> tuple[list[str], tuple, tuple, tuple, dict]:
    """Resolve one run's support set and load it **once**.

    Loading supports per-query instead would repeat the decode 100-500 times per run for
    no benefit; at 20 runs x 5 subspecies that is the difference between minutes and hours.

    ``support_selection`` only changes the *pool* that `sample_supports` draws from --
    the seeding, the 20-run structure and the paired-comparison machinery are untouched.
    See `occl.curation` for why the pool is filtered and why the filter is train-only.
    """
    pool, pool_stats = support_pool(ds, group, support_selection)
    support_ids = draw_supports(ds, group, pool, support_selection, shot, run_idx)
    images = tuple(ds.load_image(i) for i in support_ids)
    labelmaps = tuple(ds.load_labelmap(i) for i in support_ids)
    return support_ids, images, labelmaps, union_class_ids(labelmaps), pool_stats


def build_episodes(ds, group: str, shot: int, run_idx: int,
                   support_selection: str = "uniform") -> Iterator[Episode]:
    """Yield one Episode per test image for a single (group, shot, run)."""
    support_ids, images, labelmaps, class_ids, _ = build_run(
        ds, group, shot, run_idx, support_selection)
    if not class_ids:
        raise RuntimeError(
            f"{ds.name}/{group} run {run_idx}: supports {support_ids} carry no non-background "
            "labels — nothing could be prompted"
        )
    for query_id in ds.test_ids(group):
        yield Episode(
            query_id=query_id,
            support_ids=tuple(support_ids),
            support_images=images,
            support_labelmaps=labelmaps,
            query_image=ds.load_image(query_id),
            class_ids=class_ids,
        )
