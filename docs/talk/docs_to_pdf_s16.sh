#!/usr/bin/env bash
# Render the S16 talk's markdown to PDF, in the deck's own typography.
#
#   ./docs_to_pdf_s16.sh
#
# xelatex rather than pdflatex: these documents contain characters pdflatex
# cannot set (arrows, the plus-minus, the true minus sign).
set -euo pipefail
cd "$(dirname "$0")"

python3 make_handbook_s16.py

DOCS=("Hunter_SST_s16_handbook.md" "NARRATION_S16.md" "TECHNICAL_QA_S16.md")

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
