"""Locate the SST checkout and make its vendored SAM 2 importable.

SST is consumed **read-only**: we use its committed masks, its split manifests, and its
vendored `sam2` package. We never edit it. Resolution order:

1. ``$OCCL_SST_ROOT``
2. ``../SST`` next to this repo (the local dev layout)
3. ``./third_party/SST`` (the Colab layout, populated by a shallow clone)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The checkouts live INSIDE this repo since 2026-09 (`./SST`, `./sam3`, untracked); the
# sibling and third_party layouts are kept for older machines. Every local script used
# to need OCCL_SST_ROOT — the S17b pull line died on the render for exactly that.
_CANDIDATES = (REPO_ROOT / "SST", REPO_ROOT.parent / "SST", REPO_ROOT / "third_party" / "SST")


def sst_root() -> Path:
    """Return the SST checkout root, or raise with actionable guidance."""
    env = os.environ.get("OCCL_SST_ROOT")
    candidates = ([Path(env).expanduser()] if env else []) + list(_CANDIDATES)
    for c in candidates:
        if (c / "data" / "cambridge_butterfly").is_dir():
            return c.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate an SST checkout. Tried:\n  " + tried + "\n"
        "Set OCCL_SST_ROOT, or clone it:\n"
        "  git clone --depth 1 https://github.com/Imageomics/SST third_party/SST"
    )


def bootstrap_sst_syspath() -> Path:
    """Put SST's vendored SAM 2 on ``sys.path``. Call before importing sam2.

    Both entries are required, and for different reasons:

    * ``SST/src``     -- makes ``import sst.segment_anything_2...`` resolve.
    * ``SST/src/sst`` -- the vendored ``sam2/__init__.py`` runs
      ``initialize_config_module("sam2_configs")`` at import time, which needs
      ``sam2_configs`` importable as a *top-level* module. It only ever worked upstream
      because scripts were run as ``python src/sst/segment.py``, which puts that
      directory on ``sys.path[0]``. Omitting it raises MissingConfigException on import.

    Callers should still ``GlobalHydra.instance().clear()`` before their own
    ``initialize_config_dir``, since notebook cell re-runs leave dirty global state.
    """
    root = sst_root()
    for p in (root / "src", root / "src" / "sst"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def _model_repo_root(env_var: str, name: str, sentinel: str, clone_url: str) -> Path:
    """Shared resolver for the S14 model checkouts — the `sst_root` pattern.

    All third-party model repos are consumed READ-ONLY as checkouts, never
    pip-installed: sam3 pins ``numpy<2`` (which would downgrade a numpy-2-ABI venv)
    and SeC's requirements drag GUI/training stacks — the checkout + syspath route
    takes none of that on.
    """
    env = os.environ.get(env_var)
    candidates = (([Path(env).expanduser()] if env else [])
                  + [REPO_ROOT / name, REPO_ROOT.parent / name, REPO_ROOT / "third_party" / name])
    for c in candidates:
        if (c / sentinel).exists():
            return c.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate a {name} checkout. Tried:\n  {tried}\n"
        f"Set {env_var}, or clone it:\n"
        f"  git clone --depth 1 {clone_url} third_party/{name}"
    )


def sam3_root() -> Path:
    return _model_repo_root("OCCL_SAM3_ROOT", "sam3",
                            "sam3/model/sam3_tracking_predictor.py",
                            "https://github.com/facebookresearch/sam3")


def bootstrap_sam3_syspath() -> Path:
    """Make ``import sam3`` resolve from the checkout. Call before any sam3 import.

    Unlike SST there is no config-module hack — sam3's builder takes explicit
    kwargs. Callers must still ``GlobalHydra.instance().clear()`` before building a
    DIFFERENT family in the same process (the SST/SeC vendored sam2 packages
    initialize Hydra config modules at import; families cannot co-build without
    re-clearing between them — recorded in each adapter's build).
    """
    root = sam3_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def sam3_checkpoint() -> Path:
    """`sam3.pt` — resolution: $OCCL_SAM3_CKPT, else <sam3_root>/checkpoints/sam3.pt.

    The plain-tracker weights live ONLY in `sam3.pt` (`sam3.1_multiplex.pt` keys
    belong to the multiplex classes). HF `facebook/sam3` is gated (manual approval);
    Colab resolves via HF_TOKEN and caches the 3.45 GB file to Drive.
    """
    env = os.environ.get("OCCL_SAM3_CKPT")
    return Path(env).expanduser() if env else sam3_root() / "checkpoints" / "sam3.pt"


def sec_root() -> Path:
    return _model_repo_root("OCCL_SEC_ROOT", "SeC",
                            "inference/modeling_sec.py",
                            "https://github.com/OpenIXCLab/SeC")


def bootstrap_sec_syspath() -> Path:
    """Make SeC's ``inference`` package importable from the checkout root.

    ``inference/sam2/__init__.py`` runs ``initialize_config_module`` at import —
    the same Hydra global-state seam as SST's vendored sam2 (see
    `bootstrap_sst_syspath`); callers clear GlobalHydra before building.
    """
    root = sec_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def sec_checkpoint_dir() -> Path:
    """The SeC-4B snapshot dir — $OCCL_SEC_CKPT, else <sec_root>/saved_models/SeC-4B.

    Ungated: `huggingface_hub.snapshot_download("OpenIXCLab/SeC-4B")` (15.18 GB fp32
    shards) into this directory when absent — the caller decides whether to download.
    """
    env = os.environ.get("OCCL_SEC_CKPT")
    return (Path(env).expanduser() if env
            else sec_root() / "saved_models" / "SeC-4B")


def butterfly_root() -> Path:
    return sst_root() / "data" / "cambridge_butterfly"


def beetle_root() -> Path:
    return sst_root() / "data" / "neon_beetles"


#: The public source of `default_checkpoint()` and its digest (the local file, S17). The
#: SCC data layer deliberately ships no checkpoints; a SAM 2 stage there downloads this
#: once into ckpts/ and verifies it — a silently different file would score a different
#: model under the paper's name.
SAM2_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_CKPT_SHA256 = "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"


def default_checkpoint() -> Path:
    """SAM 2.0 hiera-large (072824 release) -- the checkpoint the paper used.

    Not SAM 2.1: the 2.1 checkpoints add keys (``no_obj_embed_spatial`` and friends)
    that will not load into the vendored 2.0 config.
    """
    return sst_root() / "checkpoints" / "sam2_hiera_large.pt"


def results_root() -> Path:
    return REPO_ROOT / "results"


def inbox_root() -> Path:
    """Where Colab result bundles land before `scripts/check_gate.py` validates them."""
    return REPO_ROOT / "results" / "inbox"
