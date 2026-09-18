#!/usr/bin/env bash
# One-time environment for notebooks/run_s14sam3_scc.ipynb on the BU SCC.
#
#   cd /projectnb/<project> && tar xzf occl-scc-data.tar.gz && tar xzf occl-scc-code.tar.gz && bash occl/setup.sh
#   bash scripts/scc_setup.sh <scc-project>            # from a LOGIN node, any checkout location
#   PYMOD=python3/3.12.4 bash scripts/scc_setup.sh <scc-project>
#
# Builds everything under /projectnb/<project>/occl — never under $HOME, which
# is quota'd at 10 GB and would not hold the venv (~6 GB with torch), the HF
# cache or the 3.45 GB checkpoint:
#
#   venv/     python venv with torch (CUDA wheels; no cuda module needed — the
#             wheel ships its runtime, the node only needs a driver), the occl
#             package (editable, from THIS checkout) and the sam3 import deps
#   SST/      Imageomics/SST checkout (masks, split manifests)     read-only
#   sam3/     facebookresearch/sam3 checkout (tracker code)         read-only
#   hf/       HF_HOME — hub cache and `token` for the gated checkpoint
#   ckpts/    sam3.pt lands here (downloaded by the notebook, once)
#   data/     the image cache: cache/ (from the DATA layer) or occl-cache-png.zip
#   results/  per-run JSON, adapters, logs, bundles — and the S11 ledger zips
#
# and registers a Jupyter kernel "occl (sam3)" that OnDemand's Jupyter app lists.
#
# sam3 is deliberately NOT pip-installed: its numpy<2 pin would downgrade the
# venv; occl consumes it as a checkout on sys.path (paths.sam3_root).
set -euo pipefail

# The project can be given, or inferred from where occl-scc.tar.gz was unpacked:
# this script lives at /projectnb/<project>/occl/oc-ccl-improvements/scripts/, so
# the occl root is two levels up and the project is the path segment after /projectnb/.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${OCCL_PROJECT_ROOT:-}"
PROJ="${1:-${OCCL_PROJECT:-}}"
if [[ -z "$ROOT" ]]; then
  if [[ -n "$PROJ" ]]; then
    ROOT="/projectnb/$PROJ/occl"
  elif [[ "$REPO" == /projectnb/*/occl/oc-ccl-improvements ]]; then
    ROOT="$(dirname "$REPO")"
    PROJ="$(echo "$REPO" | cut -d/ -f3)"
    echo "inferred project '$PROJ' from $REPO"
  else
    echo "usage: bash scripts/scc_setup.sh <scc-project>   (or unpack the occl-scc-*.tar.gz layers under /projectnb/<project>/)" >&2
    exit 1
  fi
fi
[[ -n "$PROJ" ]] || PROJ="$(echo "$ROOT" | cut -d/ -f3)"
# python3/3.12.4 is the pinned default: the cluster's bare `python3` module moved to
# 3.13.8, whose CPython pairing breaks `import torch` (IndentationError in
# torch/_sources.py — the first SCC session, and again on 2026-09-16 when the bare
# default was taken as "fine"). Override with PYMOD=... if 3.12.4 is ever retired.
PYMOD="${PYMOD:-python3/3.12.4}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

[[ -d "$(dirname "$ROOT")" ]] || { echo "no $(dirname "$ROOT") — is '$PROJ' your SCC project name?" >&2; exit 1; }
mkdir -p "$ROOT"/{results,ckpts,hf,data,tmp,results/runs,results/adapters,results/logs}

# Keep every cache off $HOME.
export HF_HOME="$ROOT/hf"
export PIP_CACHE_DIR="$ROOT/tmp/pip"
export TMPDIR="$ROOT/tmp"

# shellcheck disable=SC1091
source /etc/profile.d/modules.sh 2>/dev/null || true
if ! module load "$PYMOD" 2>/dev/null; then
  echo "module $PYMOD is not available — falling back to the bare python3 module;" >&2
  echo "if the import smoke below fails, pick a 3.12.x from \`module avail python3\` via PYMOD=" >&2
  module load python3
fi
echo "python: $(command -v python3) ($(python3 --version))"

# A venv made by one python3 module keeps serving that interpreter after `module
# load` of another. Rebuild it ONLY when it is actually broken: a venv that still
# imports torch and occl is kept whatever module is loaded now — the alternative
# (2026-09-16) was the bare 3.13.8 default deleting a working 3.12.4 venv and
# rebuilding it on the interpreter that cannot import torch.
_want="$(python3 -c 'import sys; print(sys.version.split()[0])')"
if [[ -x "$ROOT/venv/bin/python" ]]; then
  _have="$("$ROOT/venv/bin/python" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo none)"
  if "$ROOT/venv/bin/python" -c 'import torch, occl' >/dev/null 2>&1; then
    if [[ "$_have" != "$_want" ]]; then
      echo "venv (python $_have) imports torch and occl — keeping it; the loaded module's $_want is ignored"
    fi
  else
    echo "venv (python $_have) cannot import torch/occl — rebuilding with python $_want"
    rm -rf "$ROOT/venv"
  fi
fi
if [[ ! -x "$ROOT/venv/bin/python" ]]; then
  python3 -m venv "$ROOT/venv"
fi
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
pip install -q -U pip
# torch first, from the CUDA index — a bare `pip install torch` may resolve a CPU wheel.
pip install -q torch torchvision --index-url "$TORCH_INDEX"
# ONE dependency line, mirroring the Colab notebook's empirical list: hydra-core /
# omegaconf / iopath are on the tracker's import path; timm/ftfy/regex/einops/
# pycocotools are sam3's import-time deps. ipykernel is what makes the venv a kernel.
pip install -q -e "$REPO[gpu]" timm ftfy regex einops pycocotools ipykernel
python -m ipykernel install --user --name occl-sam3 --display-name "occl (sam3)"

[[ -f "$ROOT/SST/README.md" ]] || git clone -q --depth 1 https://github.com/Imageomics/SST "$ROOT/SST"
[[ -f "$ROOT/sam3/sam3/model/sam3_tracking_predictor.py" ]] \
  || git clone -q --depth 1 https://github.com/facebookresearch/sam3 "$ROOT/sam3"

# Import smoke. torch's JIT overload check re-parses function sources through
# `inspect`, and some CPython point releases return truncated source for it — the
# import then dies in torch/_sources.py with an IndentationError (seen on the SCC
# python3/3.13.8 module; 3.12 and 3.14 both work). That is a Python/torch pairing
# problem, so the remedy is another module, not a code change.
if ! python - <<'PY'
import numpy, torch, timm, hydra  # noqa: F401
import occl, occl.scc  # noqa: F401
assert not numpy.__version__.startswith("1."), numpy.__version__
print(f"torch {torch.__version__} (cuda {torch.version.cuda}) | numpy {numpy.__version__} | occl build {occl.BUILD}")
PY
then
  cat >&2 <<EOF

IMPORT SMOKE FAILED under $(python3 --version) (module $PYMOD), torch $(pip show torch 2>/dev/null | awk '/^Version/{print $2}').
If the traceback ends in torch/_sources.py with an IndentationError, this python3
module and torch disagree. Pick another module and re-run; a venv that cannot
import torch is rebuilt on the next run (a working one is never touched):
  module avail python3
  PYMOD=python3/3.12.4 bash $ROOT/setup.sh      # any 3.12.x; 3.14.x also works
EOF
  exit 1
fi

echo
echo "ready: $ROOT"
missing=0
_need() { if [[ -e "$1" ]]; then echo "  ok       $1"; else echo "  MISSING  $1   <- $2"; missing=1; fi; }
echo "prerequisites for the first session:"
if [[ -f "$ROOT/data/cache/manifest.jsonl" ]]; then echo "  ok       $ROOT/data/cache/"; else
  _need "$ROOT/data/occl-cache-png.zip" "scp the lossless cache zip (or a cache/ folder) from the machine that ran S2"; fi
for ds in butterfly_major beetle; do for cell in base base_tta occl_pertrait_tta; do
  _need "$ROOT/results/s11_${ds}_${cell}.zip" "banked SAM 2 ledger (Drive: occl/results/)"
done; done
if [[ -f "$ROOT/ckpts/sam3.pt" ]]; then echo "  ok       $ROOT/ckpts/sam3.pt"; else
  _need "$ROOT/hf/token" "echo <hf-token> > $ROOT/hf/token  (facebook/sam3 is gated; the notebook downloads sam3.pt)"; fi

cat <<EOF

OnDemand (scc-ondemand.bu.edu) -> Interactive Apps -> Jupyter Notebook:
  Project              $PROJ
  Pre-Launch Command   export OCCL_PROJECT_ROOT=$ROOT
  Working Directory    $REPO
  Number of cores      4 x GPUs (cores cap live workers at NSLOTS-1)
  Number of GPUs       2-4 (more cards = more of the graph in flight)
  GPU compute cap.     8.0 or higher (bf16)
  Number of hours      8+ (the stage resumes across sessions; nothing is lost at wall-time)
then open notebooks/run_s14sam3_scc.ipynb, kernel "occl (sam3)", Run All.
EOF
[[ $missing -eq 0 ]] || echo "(some prerequisites are missing — see above; the notebook asserts each before spending GPU time)"
