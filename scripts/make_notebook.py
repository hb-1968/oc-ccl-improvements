#!/usr/bin/env python3
"""Generate .ipynb files from percent-format notebook sources.

Edit notebooks/*.py, then run this script. Shared helpers and SCC cells are
checked for consistency by the existing notebook tests.
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
