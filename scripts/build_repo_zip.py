#!/usr/bin/env python3
"""Build ``occl-repo.zip`` — the code bundle the Colab notebook extracts.

    python scripts/build_repo_zip.py

The notebook and the code travel separately: the ipynb is opened in Colab by hand, while
`REPO_SOURCE = "drive_zip"` pulls everything else from this one zip on Drive. Nothing
else pairs the two — the first S10 session ran a current notebook against a stale zip and
died at argparse, with every fine-tune reporting `unrecognized arguments` for flags the
notebook had just been taught to pass. (The notebook used to say ``make occl-repo.zip``;
no Makefile ever existed — the zip was built ad hoc, which is exactly how it went stale.)

So this script does two things, always together:

1. regenerates ``notebooks/run_stage.ipynb`` from the percent-format source
   (`scripts/make_notebook.py`), so the ipynb you upload and the zip you upload were
   produced from the same tree;
2. zips the runtime tree with repo-relative paths (the notebook does ``unzip -d {REPO}``,
   so there must be no top-level wrapper directory).

The extraction cell also asserts the extracted `STAGES` tuple contains the stage being
run, so a zip that predates a new stage fails at extraction with a message naming this
script, rather than 400 lines later inside a worker log.

After building: replace ``occl-repo.zip`` on Drive (the notebook probes
``MyDrive/occl/occl-repo.zip`` and its siblings) and, when the notebook itself changed,
re-open the fresh ipynb too.
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
