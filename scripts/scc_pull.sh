#!/usr/bin/env bash
# Pull bundles from the cluster's results dir in ONE ssh session (one password).
#
#   bash scripts/scc_pull.sh [--dest DIR] FILE [FILE ...]
#
# FILEs are names (or globs, quoted) under $SCC_RESULTS on the cluster; they land in
# --dest (default results/inbox/). Streams a tar over stdout instead of scp, so any number
# of files costs one connection. Override SCC_HOST / SCC_RESULTS in the environment.
set -euo pipefail
HOST="${SCC_HOST:-hbridges@scc1.bu.edu}"
REMOTE="${SCC_RESULTS:-/projectnb/bu-mlb/hbridges/sst/occl/results}"
DEST="results/inbox"
if [[ "${1:-}" == "--dest" ]]; then DEST="$2"; shift 2; fi
[[ $# -ge 1 ]] || { echo "usage: bash scripts/scc_pull.sh [--dest DIR] FILE [FILE ...]" >&2; exit 2; }
mkdir -p "$DEST"
# the names are passed on the remote command line unquoted so a glob expands there
ssh "$HOST" "cd '$REMOTE' && tar cf - $*" | tar xf - -C "$DEST"
ls -la "$DEST"/$(echo "$1" | sed 's/\*.*//')* 2>/dev/null | tail -n +1
