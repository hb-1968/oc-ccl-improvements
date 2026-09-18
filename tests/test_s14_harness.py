"""S14 harness cross-cuts — the seams that let a second model family in safely.

Two properties are load-bearing enough to pin: the TTA wrapper must MIRROR the inner
adapter's precompute presence (a class-level method defeated the evaluator's hasattr
guard and crashed at call time on a precompute-less inner), and the resume fingerprint
must stay byte-identical for SAM 2 runs after the adapter-source parameterization —
otherwise landing S14 would orphan every banked per-run JSON in the project.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from occl.adapters.tta import OrientationTTAAdapter


class _NoPrecompute:
    name = "stub"

    def predict(self, ep):  # pragma: no cover - never called
        raise NotImplementedError


class _WithPrecompute:
    name = "stub"

    def __init__(self):
        self.seen = []

    def precompute(self, pairs):
        self.seen = list(pairs)
        return {"n_cached": len(self.seen), "total_gb": 0.0}


def test_tta_wrapper_mirrors_inner_precompute_presence():
    assert not hasattr(OrientationTTAAdapter(_NoPrecompute()), "precompute")

    inner = _WithPrecompute()
    wrapper = OrientationTTAAdapter(inner)
    assert hasattr(wrapper, "precompute")
    rep = wrapper.precompute([("k1", np.zeros((4, 4, 3), np.uint8))])
    assert rep["n_cached"] == 2
    assert [k for k, _ in inner.seen] == ["k1", "k1@r180"]


def test_run_fingerprint_for_sst_sam2_is_byte_identical_to_the_hardcoded_era():
    """For SSTSam2Adapter — wrapped in TTA or not — the fingerprint must equal what
    the old hardcoded `adapters/sst_sam2.py` list entry produced, byte for byte."""
    from occl import evaluator
    from occl.adapters.sst_sam2 import SSTSam2Adapter

    class _DS:
        name = "beetle"
        cache = None

    adapter = SSTSam2Adapter.__new__(SSTSam2Adapter)  # no ctor side effects wanted
    adapter.name = "sst_sam2"
    adapter.dtype = None
    adapter.weights_id = "w"

    src = Path(evaluator.__file__).parent
    parts = [None, None, None, "None", "w"]
    for rel in ("metrics.py", "protocol.py", "curation.py", "seeding.py",
                "evaluator.py", "adapters/sst_sam2.py"):
        parts.append(hashlib.sha1((src / rel).read_bytes()).hexdigest())
    want = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8]

    assert evaluator.run_fingerprint(_DS(), adapter) == want
    # the TTA wrapper resolves to the INNER's source — same fingerprint space
    assert evaluator.run_fingerprint(_DS(), OrientationTTAAdapter(adapter)) == want


def test_run_fingerprint_separates_adapter_families():
    """Two adapters with identical dtype/weights but different defining files must
    not share a resume slug space — the stale-resume failure the fingerprint exists
    to prevent, now per family."""
    from occl import evaluator

    class _DS:
        name = "beetle"
        cache = None

    a, b = _NoPrecompute(), _WithPrecompute()
    a.dtype = b.dtype = None
    a.weights_id = b.weights_id = "w"
    # both defined in THIS file -> same source -> equal
    assert (evaluator.run_fingerprint(_DS(), a)
            == evaluator.run_fingerprint(_DS(), b))

    from occl.adapters.sst_sam2 import SSTSam2Adapter

    real = SSTSam2Adapter.__new__(SSTSam2Adapter)
    real.name, real.dtype, real.weights_id = "sst_sam2", None, "w"
    assert (evaluator.run_fingerprint(_DS(), real)
            != evaluator.run_fingerprint(_DS(), a))
