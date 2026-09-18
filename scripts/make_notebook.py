#!/usr/bin/env python3
"""Build the Colab notebooks from the percent-format source run_stage.py.

The .py is the maintainable, diffable source of truth; the .ipynb files are the
artifacts you open in Colab. Regenerate after editing the .py:

    python scripts/make_notebook.py

Two notebooks, two sources:

* ``run_stage.py -> run_stage.ipynb`` — the parameterized multi-stage notebook.
* ``run_s14sam3.py -> run_s14sam3.ipynb`` — the STANDALONE sam3 pipeline (user
  decision after the s14sec sessions: conflicting dependency stacks get a
  dedicated notebook that can never run the wrong stage). Its shared helper
  functions are verbatim copies of run_stage.py's, pinned by AST equality in
  tests/test_s14_standalone.py — edit run_stage.py, re-copy, or the suite fails.
* ``run_s14sam3_scc.py -> run_s14sam3_scc.ipynb`` — the same stage re-hosted on
  the BU SCC's OnDemand Jupyter (multi-GPU, project space instead of Drive, no
  google.colab). Its copied helpers are pinned the same way in tests/test_scc.py;
  the multi-GPU packing lives in occl.scc rather than in the notebook.
* ``run_s15_scc.py -> run_s15_scc.ipynb`` — the champion at 5-shot on the SCC. Its
  session/cache/GPU/checkpoint cells are verbatim copies of run_s14sam3_scc.py's,
  pinned cell-by-cell in tests/test_scc.py.
* ``run_s17_scc.py -> run_s17_scc.ipynb`` — the training-pair probe on the SCC (SAM 2:
  the carrier retrained, then every cycle pair replayed under both arms). Its five
  family-neutral cells are verbatim copies of run_s15_scc.py's, pinned in
  tests/test_scc.py; the checkpoint cell is its own (SAM 2 is public, sha-pinned).
* ``run_s17b_scc.py -> run_s17b_scc.ipynb`` — S17b: the forward-only and objectness arms
  on the same probe + test-split contract; its cells are pinned to run_s17_scc.py's.
* ``run_s17c_scc.py -> run_s17c_scc.ipynb`` — S17c: the objectness weight grid, the lr x
  epochs 2x2, the arch and link compositions and a 5-shot transfer fan, in one session;
  pinned to run_s17b_scc.py's cells.
* ``run_s17d_scc.py -> run_s17d_scc.ipynb`` — S17d: comp_arch x w10 and the soft
  sentinel arms (+ their base variants in the probe); pinned to run_s17c_scc.py's cells
  except commissioning (it also commissions the soft gate).
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "notebooks" / "run_stage.py"
DST = SRC.with_suffix(".ipynb")

SAM3_SRC = SRC.parent / "run_s14sam3.py"
SAM3_DST = SAM3_SRC.with_suffix(".ipynb")

SCC_SRC = SRC.parent / "run_s14sam3_scc.py"
SCC_DST = SCC_SRC.with_suffix(".ipynb")

S15_SRC = SRC.parent / "run_s15_scc.py"
S15_DST = S15_SRC.with_suffix(".ipynb")

S17_SRC = SRC.parent / "run_s17_scc.py"
S17_DST = S17_SRC.with_suffix(".ipynb")

S17B_SRC = SRC.parent / "run_s17b_scc.py"
S17B_DST = S17B_SRC.with_suffix(".ipynb")

S17C_SRC = SRC.parent / "run_s17c_scc.py"
S17C_DST = S17C_SRC.with_suffix(".ipynb")
S17D_SRC = SRC.parent / "run_s17d_scc.py"
S17D_DST = S17D_SRC.with_suffix(".ipynb")
S18_SRC = SRC.parent / "run_s18_scc.py"
S18_DST = S18_SRC.with_suffix(".ipynb")


def split_cells(text: str) -> list[tuple[str, str]]:
    """Split a `# %%` / `# %% [markdown]` percent-format file into (kind, source)."""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if body:
                cells.append((kind, body))
            kind = "markdown" if "[markdown]" in line else "code"
            body = []
            continue
        body.append(line)
    if body:
        cells.append((kind, body))

    out = []
    for k, lines in cells:
        if k == "markdown":
            # strip the leading "# " that makes markdown valid Python
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        src = "\n".join(lines).strip("\n")
        if src.strip():
            out.append((k, src))
    return out


def build(dst: Path, text: str, banner: str | None = None,
          kernel: tuple[str, str] = ("python3", "Python 3"),
          colab: bool = True) -> None:
    cells = []
    if banner is not None:
        stripped = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                             for ln in banner.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": stripped.splitlines(keepends=True)})
    for kind, src in split_cells(text):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)

    metadata = {
        "kernelspec": {"display_name": kernel[1], "name": kernel[0]},
        "language_info": {"name": "python"},
    }
    if colab:
        metadata |= {"accelerator": "GPU",
                     "colab": {"provenance": [], "toc_visible": True}}
    nb = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    dst.write_text(json.dumps(nb, indent=1) + "\n")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"wrote {dst}  ({len(cells)} cells: {len(cells) - n_md} code, {n_md} markdown)")


def main() -> None:
    build(DST, SRC.read_text())
    build(SAM3_DST, SAM3_SRC.read_text())
    # The SCC notebook opens in the venv kernel scc_setup.sh registers, so its
    # kernelspec names that kernel — Jupyter then selects it on open instead of
    # the module Python, which has no torch.
    build(SCC_DST, SCC_SRC.read_text(), kernel=("occl-sam3", "occl (sam3)"),
          colab=False)
    build(S15_DST, S15_SRC.read_text(), kernel=("occl-sam3", "occl (sam3)"),
          colab=False)
    build(S17_DST, S17_SRC.read_text(), kernel=("occl-sam3", "occl (sam3)"),
          colab=False)
    build(S17B_DST, S17B_SRC.read_text(), kernel=("occl-sam3", "occl (sam3)"),
          colab=False)
    build(S17C_DST, S17C_SRC.read_text(), kernel=("occl-sam3", "occl (sam3)"),
          colab=False)
    build(S17D_DST, S17D_SRC.read_text(), kernel=("occl-sam3", "occl (sam3)"),
          colab=False)
    build(S18_DST, S18_SRC.read_text(), kernel=("occl-sam3", "occl (sam3)"),
          colab=False)


if __name__ == "__main__":
    raise SystemExit(main())
