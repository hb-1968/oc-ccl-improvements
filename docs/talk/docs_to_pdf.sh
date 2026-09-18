#!/usr/bin/env bash
# Render the talk's markdown to PDF, in the deck's own typography.
#
#   ./docs_to_pdf.sh            # all three, into the same directory
#
# xelatex is used rather than pdflatex because the documents contain characters
# pdflatex cannot set: the significance mark, arrows, degree signs, the plus-minus
# and the true minus sign.
set -euo pipefail
cd "$(dirname "$0")"

python3 make_handbook.py          # the combined guide + Q&A document

DOCS=("Hunter_SST_talk_handbook.md" "NARRATION.md" "TECHNICAL_QA.md" "README.md")

for md in "${DOCS[@]}"; do
  [ -f "$md" ] || { echo "skip $md (not here)"; continue; }
  out="${md%.md}.pdf"
  pandoc "$md" -o "$out" \
    --pdf-engine=xelatex \
    --toc --toc-depth=3 \
    -V geometry:"a4paper,margin=2.2cm" \
    -V mainfont="DejaVu Sans" \
    -V monofont="DejaVu Sans Mono" \
    -V fontsize=10pt \
    -V linkcolor=black \
    -V colorlinks=true \
    -V toccolor=black
  echo "wrote $out"
done
