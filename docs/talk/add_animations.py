#!/usr/bin/env python3
"""Inject click-triggered fade-in builds into the deck.

python-pptx has no animation API, so this writes the <p:timing> tree directly.
Input: anim_spec.json = {slide_index: [[shape_id, ...], ...]} — one inner list
per click. Shapes not named in the spec are static.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = f'xmlns:p="{P}" xmlns:a="{A}"'


def effect(sid: str, tid: int, first: bool) -> tuple[str, int]:
    """One shape's fade entrance. Returns (xml, next free timing id)."""
    node = (
        f'<p:par><p:cTn id="{tid}" presetID="10" presetClass="entr" '
        f'presetSubtype="0" fill="hold" grpId="0" '
        f'nodeType="{"clickEffect" if first else "withEffect"}">'
        f'<p:stCondLst><p:cond delay="0"/></p:stCondLst>'
        f'<p:childTnLst>'
        f'<p:set><p:cBhvr><p:cTn id="{tid + 1}" dur="1" fill="hold">'
        f'<p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
        f'<p:tgtEl><p:spTgt spid="{sid}"/></p:tgtEl>'
        f'<p:attrNameLst><p:attrName>style.visibility</p:attrName>'
        f'</p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to>'
        f'</p:set>'
        f'<p:animEffect transition="in" filter="fade">'
        f'<p:cBhvr><p:cTn id="{tid + 2}" dur="400"/>'
        f'<p:tgtEl><p:spTgt spid="{sid}"/></p:tgtEl></p:cBhvr>'
        f'</p:animEffect>'
        f'</p:childTnLst></p:cTn></p:par>'
    )
    return node, tid + 3


def timing_xml(groups: list[list[int]]) -> str:
    tid = 4                       # 1 tmRoot, 2 mainSeq, 3 reserved
    clicks = []
    for g in groups:
        inner, first = [], True
        outer_a, outer_b = tid, tid + 1
        tid += 2
        for sid in g:
            node, tid = effect(str(sid), tid, first)
            inner.append(node)
            first = False
        clicks.append(
            f'<p:par><p:cTn id="{outer_a}" fill="hold">'
            f'<p:stCondLst><p:cond delay="indefinite"/></p:stCondLst>'
            f'<p:childTnLst><p:par><p:cTn id="{outer_b}" fill="hold">'
            f'<p:stCondLst><p:cond delay="0"/></p:stCondLst>'
            f'<p:childTnLst>{"".join(inner)}</p:childTnLst>'
            f'</p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
        )
    return (
        f'<p:timing {NS}><p:tnLst>'
        f'<p:par><p:cTn id="1" dur="indefinite" restart="never" '
        f'nodeType="tmRoot"><p:childTnLst>'
        f'<p:seq concurrent="1" nextAc="seek">'
        f'<p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
        f'{"".join(clicks)}'
        f'</p:childTnLst></p:cTn>'
        f'<p:prevCondLst><p:cond evt="onPrev" delay="0">'
        f'<p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
        f'<p:nextCondLst><p:cond evt="onNext" delay="0">'
        f'<p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
        f'</p:seq></p:childTnLst></p:cTn></p:par>'
        f'</p:tnLst></p:timing>'
    )


def main(deck: str, spec: str, out: str) -> None:
    prs = Presentation(deck)
    plan = {int(k): v for k, v in json.loads(Path(spec).read_text()).items()}
    for idx, sld in enumerate(prs.slides):
        groups = plan.get(idx)
        if not groups:
            continue
        present = {sh.shape_id for sh in sld.shapes}
        groups = [[s for s in g if s in present] for g in groups]
        groups = [g for g in groups if g]
        el = sld.element
        for old in el.findall(qn("p:timing")):
            el.remove(old)
        el.append(parse_xml(timing_xml(groups)))
        print(f"  slide {idx + 1}: {len(groups)} clicks, "
              f"{sum(len(g) for g in groups)} shapes")
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
