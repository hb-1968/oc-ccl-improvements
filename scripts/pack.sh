#!/usr/bin/env bash
# Build the ~88 KB repo zip that Colab extracts (REPO_SOURCE="drive_zip" or "upload").
#
# Regenerates the notebook first, so the .ipynb in the zip always matches the
# percent-format source rather than silently shipping a stale one.
#
#   bash scripts/pack.sh              -> ../occl-repo.zip
#   bash scripts/pack.sh /path/out.zip
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-../occl-repo.zip}"

python scripts/make_notebook.py

rm -f "$OUT"
zip -qr "$OUT" src scripts notebooks tests pyproject.toml README.md -x '*__pycache__*'

# The notebook must sit OUTSIDE the zip as well: Colab has to open it before anything
# can extract the zip. The copy inside is only so the extracted repo is complete.
NB_OUT="$(dirname "$OUT")/run_stage.ipynb"
cp notebooks/run_stage.ipynb "$NB_OUT"

echo
echo "Copy BOTH of these to Drive:"
echo "  $(readlink -f "$NB_OUT")   $(du -h "$NB_OUT" | cut -f1)  <- open THIS in Colab"
echo "  $(readlink -f "$OUT")   $(du -h "$OUT" | cut -f1)  <- cell 3 extracts this ($(unzip -l "$OUT" | tail -1 | awk '{print $2}') files)"
echo
echo "The zip may go in MyDrive/, MyDrive/occl/ or MyDrive/oc-ccl-data/ — the notebook probes all three."
