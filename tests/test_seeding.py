"""Guards on the properties that make cross-variant comparison legitimate."""

from __future__ import annotations

from occl.seeding import derive_seed, sample_supports, support_set_hash

TRAIN = [f"img{i:03d}" for i in range(200)]


def test_seed_is_deterministic():
    assert derive_seed("beetle", "all", 0, 7) == derive_seed("beetle", "all", 0, 7)


def test_different_runs_draw_different_supports():
    a = sample_supports(TRAIN, "beetle", "all", 1, 0)
    b = sample_supports(TRAIN, "beetle", "all", 1, 1)
    assert a != b


def test_one_shot_is_a_strict_prefix_of_five_shot():
    """Makes the 1-shot vs 5-shot comparison paired, and bounds the download set."""
    for run in range(20):
        one = sample_supports(TRAIN, "major", "cyrbia", 1, run)
        five = sample_supports(TRAIN, "major", "cyrbia", 5, run)
        assert five[:1] == one


def test_sampling_is_independent_of_input_order():
    """Filesystem ordering is not stable across machines; sorting must absorb that."""
    a = sample_supports(TRAIN, "beetle", "all", 5, 3)
    b = sample_supports(list(reversed(TRAIN)), "beetle", "all", 5, 3)
    assert a == b


def test_supports_within_a_run_are_distinct():
    assert len(set(sample_supports(TRAIN, "beetle", "all", 5, 11))) == 5


def test_support_set_hash_detects_a_filtered_manifest():
    """Same seeds, one image missing -> different supports -> different hash.

    This is the failure the seed alone cannot catch, and why comparability keys on the
    resolved IDs rather than on the seed.
    """
    full = {r: sample_supports(TRAIN, "major", "malleti", 1, r) for r in range(20)}
    filtered_pool = [t for t in TRAIN if t != full[0][0]]
    filtered = {r: sample_supports(filtered_pool, "major", "malleti", 1, r) for r in range(20)}
    assert full != filtered
    assert support_set_hash(full) != support_set_hash(filtered)


def test_support_set_hash_is_stable():
    runs = {r: sample_supports(TRAIN, "beetle", "all", 5, r) for r in range(20)}
    assert support_set_hash(runs) == support_set_hash(dict(reversed(list(runs.items()))))
