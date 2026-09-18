#!/usr/bin/env python3
"""Combine the speaking guide and the technical Q&A into one document.

Writes `Hunter_SST_talk_handbook.md`, which `docs_to_pdf.sh` renders. Headings
are demoted one level so both documents sit under a part heading and the table
of contents nests correctly.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "Hunter_SST_talk_handbook.md"

PREAMBLE = """---
title: "Talk handbook"
subtitle: "Speaking guide and technical Q&A for Hunter_SST_techniques.pptx"
author: "Hunter Bridges"
---

The deck is 34 slides: 31 presented, 3 backup. Part I is the speaking guide, one
entry per slide, giving the time to spend and what the slide has to establish.
Part II answers the implementation questions at the level of functions, formulas
and thresholds, and is what backup slides 32 to 34 point at.

\\newpage

"""

PARTS = [
    ("Part I. Speaking guide", "NARRATION.md"),
    ("Part II. Technical Q&A", "TECHNICAL_QA.md"),
]


def demote(md: str) -> str:
    """Push every ATX heading down one level, and drop the document's own title."""
    lines = md.splitlines()
    out = []
    for ln in lines:
        m = re.match(r"^(#{1,5})\s+(.*)$", ln)
        if not m:
            out.append(ln)
            continue
        hashes, text = m.groups()
        if len(hashes) == 1 and not out:          # the file's own H1
            continue
        out.append("#" * (len(hashes) + 1) + " " + text)
    return "\n".join(out).lstrip("\n")


def main() -> None:
    chunks = [PREAMBLE]
    for i, (title, name) in enumerate(PARTS):
        body = demote((HERE / name).read_text())
        chunks.append(f"# {title}\n\n{body}\n")
        if i < len(PARTS) - 1:
            chunks.append("\n\\newpage\n\n")
    OUT.write_text("\n".join(chunks))
    print(f"wrote {OUT.name} ({len(OUT.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
