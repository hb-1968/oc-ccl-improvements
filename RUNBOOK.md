# Runbook

Setup and execution for the published S18 implementation. See the
[overview](README.md) for scope and the [ledger](EXPERIMENTS.md) for stage details.
Model checkouts (`SST/`, `sam3/`, `SeC/`), weights, the pixel cache and raw inbox
bundles are external inputs excluded from version control.

## 1. Laptop bootstrap (once)

```bash
git clone git@github.com:hb-1968/oc-ccl-improvements.git && cd oc-ccl-improvements
python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e '.[gpu,dev]' matplotlib
git clone --depth 1 https://github.com/Imageomics/SST SST          # masks + split manifests (~1 GB; local runs/render need them)
git clone --depth 1 https://github.com/facebookresearch/sam3 sam3  # only for the full test suite / sam3 stages
```
`paths.sst_root()` / `paths.sam3_root()` find `./SST` and `./sam3` first; no env vars needed
when the checkouts sit inside the repo. Tests: `.venv/bin/python -m pytest tests/ -q`
(the BUILD 18 record reports 343 passing tests).

The pixel cache is needed only to RENDER sheets locally or to run a CPU smoke. Copy it
from the desktop or the cluster (one password each):
```bash
rsync -a <desktop>:Projects/oc-ccl-improvements/oc-ccl-data/cache-png/ oc-ccl-data/cache-png/
# or from the SCC project space (the notebooks stage it there as data/cache):
rsync -a <user>@scc1.bu.edu:/projectnb/<project>/occl/data/cache/ oc-ccl-data/cache-png/
```
A CPU smoke of the real model additionally needs `SST/checkpoints/sam2_hiera_large.pt`
(`bash SST/checkpoints/download_ckpts.sh`, 898 MB, sha256 pinned in `occl.paths`).

## 2. Cluster execution (S18 implementation)

SCC access requires your password and Duo authentication. Each command below uses
one SSH session. The code-layer replacement command wipes the old cluster layer;
do not use it while a stage is active.

```bash
# a) push the code layer (rebuilds the tarball from THIS checkout, wipes the old layer on the cluster, re-runs setup)
python scripts/build_scc_tarball.py --layer code && ssh <user>@scc1.bu.edu 'cd /projectnb/<project> && (bash occl/wipe.sh </dev/null || true) && tar xzf - && PYMOD=python3/3.12.4 bash occl/setup.sh </dev/null' < ../occl-scc-code.tar.gz
# first time on a fresh project space: ship the data layer too (once, ~3 GB)
python scripts/build_scc_tarball.py --layer data && cat ../occl-scc-data.tar.gz ../occl-scc-code.tar.gz | ssh <user>@scc1.bu.edu 'mkdir -p /projectnb/<project> && cd /projectnb/<project> && tar xzif - && PYMOD=python3/3.12.4 bash occl/setup.sh </dev/null'
```
b) OnDemand (scc-ondemand.bu.edu) → Jupyter → 4 GPUs (A100/L40S), the longest wall you
can → Pre-Launch Command: the `export OCCL_PROJECT_ROOT=…` line setup.sh printed → open
`oc-ccl-improvements/notebooks/run_s18_scc.ipynb` → kernel `occl (sam3)` → Run All.
The graph is resumable: a wall-time kill loses nothing finished (adapters, probe bundles,
banked runs); Run All again next session.

c) The notebook's export cell prints ONE line for the laptop: pull every bundle in one ssh
session, run the gates in order, render the sheets. Paste it at the repo root. If the
session died before the export cell, the equivalent by hand:
```bash
ssh <user>@scc1.bu.edu 'cd /projectnb/<project>/occl/results && tar cf - s18.zip s18ft_sam2_*.zip s18_beetle_*.zip s18_butterfly_major_*.zip' | tar xf - -C results/inbox/ && mkdir -p results/inbox/cells && mv results/inbox/s18_*_*.zip results/inbox/cells/ \
 && .venv/bin/python scripts/check_gate.py s18 \
 && for f in results/inbox/s18ft_sam2_*.zip; do .venv/bin/python scripts/check_gate.py s18ft --bundle $f || break; done \
 && .venv/bin/python scripts/render_probe_sheets.py --bundle results/s18.zip --dataset beetle --select most_degraded --figures \
 && .venv/bin/python scripts/render_probe_sheets.py --bundle results/s18.zip --dataset butterfly_major --select most_degraded --figures
```
`check_gate.py s18` files a passing merged bundle to `results/s18.zip`; the `s18ft` gates
file `results/s18ft_sam2_<ds>_<arm>.zip`. Deployment reads between two cells, paired per
seed: `.venv/bin/python scripts/pair_cells.py results/inbox/cells/s18_beetle_crown_stemem_tta.zip results/inbox/cells/s18_beetle_crown_tta.zip`.

d) Commit the landed evidence: `git add results/s18*.zip docs/figures/s18 README.md && git commit -m "S18: landed"` and push.

## 3. Local work and authentication

Build archives, check bundles, render figures and run tests from the repository
root with `.venv/bin/python`. SCC login requires password/Duo authentication;
OnDemand sessions require a scheduler allocation. Git publication requires
credentials for the destination repository. Preserve filed results and run banks.
