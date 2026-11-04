#!/usr/bin/env python3
"""Package code for Colab without local environments, data or model weights.

Use the explicit archive allowlist to avoid recursive Drive copies. The
notebook checks the extracted source before launching workers.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: What the Colab runtime actually needs. `docs/` is figures for the README and stays
#: out; `results/` and data never belong in a code bundle.
INCLUDE = ("src", "scripts", "notebooks", "tests", "pyproject.toml", "README.md")
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def main() -> int:
    # The ipynb is generated FROM the tree being zipped, in the same invocation — the
    # pairing is the point of this script existing.
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_notebook.py")],
                   check=True)

    out = REPO / "occl-repo.zip"
    files: list[Path] = []
    for top in INCLUDE:
        p = REPO / top
        if p.is_file():
            files.append(p)
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and not (set(f.parts) & EXCLUDE_PARTS) \
                    and f.suffix != ".pyc":
                files.append(f)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=str(f.relative_to(REPO)))

    kb = out.stat().st_size / 1024
    print(f"wrote {out} ({kb:.0f} KB, {len(files)} files)")
    print("upload it to Drive as one of the paths the notebook probes "
          "(first match wins):")
    print("  MyDrive/occl-repo.zip")
    print("  MyDrive/occl/occl-repo.zip          <- the usual spot ($DRIVE)")
    print("  MyDrive/oc-ccl-data/occl-repo.zip")
    print("  MyDrive/occl/oc-ccl-improvements/occl-repo.zip")
    print("if notebooks/run_stage.py changed, re-open the regenerated "
          "notebooks/run_stage.ipynb in Colab as well.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
