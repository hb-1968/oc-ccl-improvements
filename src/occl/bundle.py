"""Result bundles — how a Colab stage returns its numbers to the local machine.

Every GPU stage ends by writing ``<stage>.zip`` containing:

* ``record.json`` — the stage's result record, which `scripts/check_gate.py` validates.
* ``env.json``    — GPU, compute capability, resolved dtype, library versions, and both
  repo commits. A number must never be orphaned from the conditions that produced it;
  "81.2 on Major" means nothing without knowing which checkpoint and dtype produced it.
* any extra artifacts the stage attaches (``per_query.csv``, overlay PNGs, adapters).

Download it from Colab, drop it in ``results/inbox/``, and run the gate.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def _git_commit(repo: Path) -> str:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return p.stdout.strip() if p.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def env_info(dtype: str | None = None) -> dict:
    """Capture everything needed to interpret or reproduce a number."""
    from . import paths

    info: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dtype": dtype,
        "occl_commit": _git_commit(paths.REPO_ROOT),
    }
    try:
        info["sst_commit"] = _git_commit(paths.sst_root())
    except FileNotFoundError:
        info["sst_commit"] = "unknown"

    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            info |= {
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_capability": list(cap),
                "cuda": torch.version.cuda,
                "vram_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                ),
                # Never trust torch.cuda.is_bf16_supported(): since torch 2.6 it returns
                # True on sm_75 via emulation, which is correct-but-glacial and not what
                # the caller is asking.
                "bf16_native": cap[0] >= 8,
            }
        else:
            info["gpu_name"] = "cpu"
    except ImportError:
        info["torch"] = None
        info["gpu_name"] = "cpu (no torch)"
    return info


def export_bundle(
    stage: str,
    record: dict,
    out: str | Path | None = None,
    extra_files: dict[str, str | Path] | None = None,
    dtype: str | None = None,
) -> Path:
    """Write ``<stage>.zip``. Returns its path.

    ``extra_files`` maps archive name -> local path, e.g.
    ``{"per_query.csv": "/content/per_query.csv"}``.
    """
    out = Path(out) if out else Path(f"{stage}.zip")
    record = dict(record)
    record.setdefault("stage", stage)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("record.json", json.dumps(record, indent=2, default=str))
        z.writestr("env.json", json.dumps(env_info(dtype), indent=2, default=str))
        for name, src in (extra_files or {}).items():
            src = Path(src)
            if src.exists():
                z.write(src, arcname=name)
    return out


def download(path: str | Path) -> None:
    """Push the bundle to the browser when running in Colab; no-op elsewhere."""
    try:
        from google.colab import files  # type: ignore
    except ImportError:
        print(f"not running in Colab — bundle is at {path}")
        return
    files.download(str(path))
