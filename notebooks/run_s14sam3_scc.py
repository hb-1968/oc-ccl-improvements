# %% [markdown]
# # s14sam3 on the BU SCC — the crowned stack on SAM 3, multi-GPU
#
# The same stage as `run_s14sam3.ipynb` (per-trait OC-CCL + `excl` zerocat + TTA
# rot180 on `facebook/sam3`'s plain tracker, plus the **arch arm** with LoRA on
# `transformer`), re-hosted on the BU Shared Computing Cluster's OnDemand Jupyter
# instead of Colab. Same worker scripts, same bundle, same local gate
# (`python scripts/check_gate.py s14sam3`). What changes is *how the work is
# laid out*: an SCC session holds several GPUs and a chosen core count, so the
# stage is one dependency graph — four fine-tunes, sixty eval slices, eight
# merges — packed onto every card at once by `occl.scc.Scheduler`, instead of
# Colab's one-group-at-a-time fan-out on a single A100.
#
# | needs | why |
# | :-- | :-- |
# | `occl-scc-data.tar.gz` + `occl-scc-code.tar.gz` unpacked under `/projectnb/<project>/` (`scripts/build_scc_tarball.py`), then `bash occl/setup.sh` on a login node | the data layer (PNG cache, SST masks, S11 ledger) is uploaded once; the code layer (`src/occl`, worker scripts, this notebook, the sam3 package) is wiped and re-uploaded per OC-CCL iteration (`occl/wipe.sh`); setup builds the venv (home is 10 GB) and the Jupyter kernel `occl-sam3` |
# | OnDemand → Jupyter: **GPUs ≥ 2, compute capability ≥ 8.0, cores ≥ 4 per GPU, ≥ 8 h**, working dir = this repo | bf16 needs sm_80; each worker needs a core; the scheduler caps live processes at `NSLOTS - 1` |
# | kernel **occl (sam3)** selected | the module Python has no torch/timm; the venv does |
# | `sam3.pt` under `$PROJECT/ckpts/`, or an HF token in `$HF_HOME/token` / `HF_TOKEN` | HF-gated, manual approval — the checkpoint cell fails fast with instructions |
# | `occl-cache-png.zip` (or an unpacked `cache/`) under `$PROJECT/data/` | pixels; staged to node-local `$TMPDIR` |
# | the S11 SAM 2 ledger zips under `$PROJECT/results/` | S14 pairs against banked SAM 2 cells and never recomputes them |
#
# Cells run top to bottom (`Run → Run All Cells`). Everything is banked under
# `$PROJECT/results/runs`, so a session that hits its wall-time resumes on the
# next Run-all: finished slices are skipped, adapters are reused after a config
# check. The stage ends by writing `s14sam3.zip` into `$PROJECT/results/`;
# download it through OnDemand → Files (or `scp`), drop it in `results/inbox/`
# locally and run the gate. Predictions P1/P2/P5/P6/P7/P8 are frozen in the gate.

# %%
# ── configuration ────────────────────────────────────────────────────────────
import os

STAGE = "s14sam3"        # fixed — this notebook runs exactly one stage

# SCC project space. setup.sh prints the exact line for the OnDemand "Pre-Launch
# Command": `export OCCL_PROJECT_ROOT=<where occl/ was unpacked>`. OCCL_PROJECT
# (just the project name) is the shorthand for the default /projectnb/<name>/occl
# layout; a nested unpack (e.g. /projectnb/<name>/<user>/sst/occl) needs the root.
# Home directories are quota'd at 10 GB, so NOTHING large lives there: venv,
# HF cache, checkpoint, image cache and results all sit under this root.
SCC_PROJECT = os.environ.get("OCCL_PROJECT", "")
PROJECT = os.environ.get("OCCL_PROJECT_ROOT") or f"/projectnb/{SCC_PROJECT}/occl"
DATA_ROOT = f"{PROJECT}/data"          # occl-cache-png.zip and/or cache/ (manifest.jsonl)
RESULTS_ROOT = f"{PROJECT}/results"    # per-run JSON, adapters, bundles, logs — persistent
CKPT_ROOT = f"{PROJECT}/ckpts"         # sam3.pt (3.45 GB)
HF_HOME = f"{PROJECT}/hf"              # HF hub cache + token, kept out of $HOME

CACHE_ZIP = "occl-cache-png.zip"
CACHE_DELTA_ZIP = "occl-cache-png-delta.zip"   # optional; merged idempotently if present
STAGE_CACHE_LOCALLY = True    # copy the cache to node-local $TMPDIR (recommended)

# ── the S14 recipe (identical to run_s14sam3.py) ─────────────────────────────
S14_LR, S14_EPOCHS = 1e-5, 1
S14_REUSE = True
S14_DATASETS = ["butterfly_major", "beetle"]
S14_SEL = {"butterfly_major": "canonical_placement",   # each dataset's best 1-shot
           "beetle": "canonical_balanced"}

# ── layout knobs (the only SCC-specific tuning) ──────────────────────────────
#: Per-job VRAM in GB, from the Colab A100-40GB session that ran this stage:
#: plain eval workers measured 5-7 GB, TTA workers 9.8-10.0 GB (dual-orientation
#: fp32 cache + sam3's decoder workspace across three arms). Merges build the
#: model and replay banked runs with --no-precompute. Fine-tunes are `None` =
#: EXCLUSIVE (a whole card): their residency was never measured — the Colab
#: session ran each alone on 40 GB — and an under-priced fine-tune sharing a
#: card would OOM the eval workers next to it. Lower it once a run has measured
#: it (`nvidia-smi` on the node during the finetune phase).
S14_JOB_GB = {"plain": 7.5, "tta": 10.5, "merge": 8.0, "ft": None}
JOB_GB = S14_JOB_GB          # the stage-neutral name the shared GPU-probe cell reads
#: How many run-slices each (dataset, group) is cut into, per fan. Narrower
#: slices mean more independent jobs to spread over the cards; the price is
#: that every slice re-encodes its group's test images (~100 images, seconds).
#: 20 runs / 5 slices = 4 runs a job: butterfly 5 groups x 5 x 2 fans = 50 jobs,
#: beetle 10, plus 4 fine-tunes and 8 merges.
S14_SLICES = {"plain": 5, "tta": 5}
VRAM_HEADROOM = 0.92
STAGGER_S = 20               # one launch per interval: N workers must not load sam3.pt at once
RESERVED_CORES = 1           # for the kernel itself

# PRE-REGISTERED PREDICTIONS (frozen before any sam3 GPU run; the gate prints,
# never sign-gates) — verbatim from run_s14sam3.py:
#   P1 base vs sam2_base (cross-family, paired on identical draws): optimist
#      beetle +1..+8, butterfly +1..+6; prior ±2.
#   P2 THE MECHANISM READ: beetle oracle−argmax on base. > ~+5: proposals
#      improved, assignment lags -> excl's pot is real, P5 optimist activates.
#      ≈ +2.65: same proposal boundary — the upgrade is cosmetic here.
#   P5 stack_tta vs base_tta: prior ±0.3 (the S13 law); conditional optimist
#      +0.5..+2 iff the P2 read ≥ ~3 pts.
#   P6 TTA's value (base_tta vs base): recorded, not banded.
#   P7 HEADLINE: stack_tta vs sam2_pertrait_tta (76.52 / 64.12) — any CI > 0
#      means the model upgrade pays under the deployment stack.
#   P8 arch vs stack: conditional optimist beetle +0.5..+3 iff P2 shows the
#      proposal boundary persisting; prior ±0.5; disqualifiers: divergent loss
#      or butterfly < −1.

# %%
# ── SCC session + repo ───────────────────────────────────────────────────────
# No Drive, no zip: on SCC the notebook is opened FROM the repo checkout in
# project space, so the code and the notebook are the same tree by construction
# (the pairing check the Colab notebooks need does not apply — there is only
# one copy). What is checked instead: that this is a GPU job with a scheduler
# grant, and that the kernel is the project venv rather than the module Python.
import sys

assert SCC_PROJECT or os.environ.get("OCCL_PROJECT_ROOT"), (
    "neither OCCL_PROJECT_ROOT nor OCCL_PROJECT is set. Put the line setup.sh printed "
    "— `export OCCL_PROJECT_ROOT=/projectnb/.../occl` — in the OnDemand form's "
    "Pre-Launch Command, then restart the kernel.")
assert os.path.isdir(PROJECT), (
    f"{PROJECT} does not exist — run `bash scripts/scc_setup.sh {SCC_PROJECT or '<project>'}` "
    "on a login node first.")

# The kernel must be the venv scc_setup.sh built: the module python3 has no
# torch, and a `pip install --user` from inside a notebook lands in the 10 GB
# home quota and shadows the venv on the next session.
_VENV = f"{PROJECT}/venv"
assert sys.prefix.startswith(_VENV) or os.environ.get("OCCL_ANY_KERNEL"), (
    f"kernel is {sys.executable}, not the project venv under {_VENV}.\n"
    "  Kernel -> Change Kernel -> 'occl (sam3)'. If that kernel is missing, run\n"
    f"  `bash scripts/scc_setup.sh {SCC_PROJECT or '<project>'}` on a login node.")

# Every worker is launched as a `python` subprocess, and that name resolves through
# PATH — which a Jupyter kernelspec never touches: it execs the venv's interpreter
# WITHOUT activating the venv, so PATH still leads with the module python3, which
# has no torch. The first SCC session died in commissioning on exactly that
# (ModuleNotFoundError: torch, from a kernel that had just imported it). On Colab
# the kernel and `python` were always the same interpreter. Do what
# `source venv/bin/activate` does, then prove it took before any subprocess runs.
import shutil

_BIN = os.path.dirname(sys.executable)
if os.environ.get("PATH", "").split(os.pathsep)[:1] != [_BIN]:
    os.environ["PATH"] = _BIN + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VIRTUAL_ENV", sys.prefix)
_which = shutil.which("python") or ""
assert _which.startswith(sys.prefix), (
    f"`python` resolves to {_which!r}, not the kernel's venv {sys.prefix} — "
    "workers would run without torch")

# SGE facts. JOB_ID names this session's logs; NSLOTS is the core grant the
# scheduler enforces (os.cpu_count() would report the whole node).
JOB_ID = os.environ.get("JOB_ID", "nojob")
assert JOB_ID != "nojob" or os.environ.get("OCCL_ALLOW_NO_SGE"), (
    "not inside an SGE job (JOB_ID unset). Run this from an OnDemand Jupyter "
    "session, not a login node — login nodes have no GPUs and kill long processes.")

# Repo root: Jupyter's cwd is the notebook's directory when opened from Files.
REPO = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) \
    if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
assert os.path.isdir(f"{REPO}/src/occl"), (
    f"{REPO} is not the oc-ccl-improvements checkout (no src/occl). Open this "
    "notebook from <repo>/notebooks/ in the OnDemand file browser, or set the "
    "session's working directory to the repo.")
os.chdir(REPO)
_SRC = f"{REPO}/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# HF cache/token out of $HOME, before anything imports huggingface_hub.
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)
for _d in (DATA_ROOT, RESULTS_ROOT, CKPT_ROOT, f"{RESULTS_ROOT}/runs",
           f"{RESULTS_ROOT}/adapters", f"{RESULTS_ROOT}/logs/{STAGE}_{JOB_ID}"):
    os.makedirs(_d, exist_ok=True)
LOG_DIR = f"{RESULTS_ROOT}/logs/{STAGE}_{JOB_ID}"
RUNS_DIR = f"{RESULTS_ROOT}/runs"
ADAPTER_DIR = f"{RESULTS_ROOT}/adapters"

import occl  # noqa: E402  — fail here, next to the fix, rather than three cells later
from occl import scc  # noqa: E402  — the multi-GPU scheduler; build >= 11 carries it

REPO_MIN_BUILD = 18
_build = getattr(occl, "BUILD", 0)
assert _build >= REPO_MIN_BUILD, (
    f"this checkout is occl build {_build}; the SCC notebooks need >= {REPO_MIN_BUILD} "
    "(occl.scc + the sam3 adapter attending all conditioning frames). Re-upload the "
    "code layer (scripts/build_scc_tarball.py --layer code).")

import numpy as _np  # noqa: E402

assert not _np.__version__.startswith("1."), (
    f"numpy {_np.__version__}: something pip-installed the sam3 repo (its numpy<2 pin) "
    "into this venv. Rebuild the venv with scripts/scc_setup.sh — sam3 is consumed as "
    "a checkout on sys.path, never installed.")

print(f"occl {occl.__version__} (build {_build}) from {_SRC}")
print(f"project {PROJECT} | job {JOB_ID} | cores granted {scc.granted_cores()} | "
      f"scratch {scc.local_scratch()}")

# %%
# ── SST + sam3 checkouts (masks, split manifests; the sam3 tracker code) ─────
# Both are consumed READ-ONLY from project space. scc_setup.sh clones them; the
# cell only clones if they are missing (compute nodes have outbound access).
import subprocess

SST = f"{PROJECT}/SST"
SAM3 = f"{PROJECT}/sam3"
for _root, _url, _sentinel in ((SST, "https://github.com/Imageomics/SST", "README.md"),
                               (SAM3, "https://github.com/facebookresearch/sam3",
                                "sam3/model/sam3_tracking_predictor.py")):
    if not os.path.exists(os.path.join(_root, _sentinel)):
        print(f"cloning {_url} -> {_root}", flush=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", _url, _root], check=True)
os.environ["OCCL_SST_ROOT"] = SST
os.environ["OCCL_SAM3_ROOT"] = SAM3

from occl import paths  # noqa: E402

print("SST at", paths.sst_root(), "| sam3 at", paths.sam3_root())
print("butterfly masks:", paths.butterfly_root().exists(),
      "| beetle:", paths.beetle_root().exists())

# %%
# ── stage the image cache onto node-local scratch ────────────────────────────
# GPFS serves small-file reads far better than Drive's FUSE layer did, but a
# 20-run pass still reads each of ~600 images thousands of times across 30+
# concurrent workers, and $TMPDIR is a local NVMe wiped at job end. So: stage
# once per session, from the zip (one sequential read) or the unpacked folder.
#
# The cache is LOSSLESS PNG (~2.5 GB) — a pure memoisation of SST's own pixel
# path, verified bit-exact. A lossy cache is flagged below and must not be used.
import glob as _glob
import json as _json
import shutil
import time as _t


def _cache_shortfall(root):
    """(entries, missing_paths, formats) for a staged cache — or None if unusable.

    Checked against the manifest, not a file count: the manifest is the authority
    on what should be present, and `ImageCache.has` consults it.
    """
    m = os.path.join(root, "manifest.jsonl")
    if not os.path.exists(m):
        return None
    entries = [_json.loads(line) for line in open(m)]
    missing = [e["cache_path"] for e in entries
               if not os.path.exists(os.path.join(root, e["cache_path"]))]
    return entries, missing, sorted({(e.get("dataset"), e.get("fmt")) for e in entries})


_CACHE_DIR = next((d for d in (f"{DATA_ROOT}/cache", f"{DATA_ROOT}/oc-ccl-data/cache",
                               f"{PROJECT}/cache")
                   if os.path.exists(os.path.join(d, "manifest.jsonl"))), None)
_CACHE_ZIP = next((z for z in (f"{DATA_ROOT}/{CACHE_ZIP}", f"{PROJECT}/{CACHE_ZIP}")
                   if os.path.exists(z)), None)
assert _CACHE_DIR or _CACHE_ZIP, (
    f"no image cache: looked for cache/manifest.jsonl and {CACHE_ZIP} under {DATA_ROOT} "
    f"and {PROJECT}. Copy occl-cache-png.zip there (scp from the machine that ran S2).")

if STAGE_CACHE_LOCALLY:
    LOCAL_CACHE = os.path.join(scc.local_scratch(), "occl-cache")
    _have = _cache_shortfall(LOCAL_CACHE) if os.path.isdir(LOCAL_CACHE) else None
    if _have is not None and _have[1]:
        print(f"local cache incomplete ({len(_have[1])} of {len(_have[0])} absent) — "
              "re-staging")
        shutil.rmtree(LOCAL_CACHE, ignore_errors=True)
        _have = None
    if _have is None:
        _t0 = _t.time()
        _tmp = LOCAL_CACHE + ".staging"
        shutil.rmtree(_tmp, ignore_errors=True)
        if _CACHE_ZIP:
            # `unzip` returns non-zero on a bad archive but still extracts what it
            # can; test first so a half-copied zip stops the cell.
            _test = subprocess.run(["unzip", "-tqq", _CACHE_ZIP], capture_output=True,
                                   text=True)
            assert _test.returncode == 0, (
                f"{_CACHE_ZIP} is not a complete archive (still copying?).\n"
                f"{_test.stdout[-800:]}{_test.stderr[-800:]}")
            os.makedirs(_tmp)
            subprocess.run(["unzip", "-q", "-o", _CACHE_ZIP, "-d", _tmp], check=True)
            _found = [r for r, _, f in os.walk(_tmp) if "manifest.jsonl" in f]
            assert _found, f"no manifest.jsonl inside {_CACHE_ZIP}"
            os.rename(_found[0], LOCAL_CACHE)
        else:
            shutil.copytree(_CACHE_DIR, LOCAL_CACHE)
        shutil.rmtree(_tmp, ignore_errors=True)
        print(f"staged cache to {LOCAL_CACHE} in {_t.time() - _t0:.0f}s")
        _have = _cache_shortfall(LOCAL_CACHE)
else:
    assert _CACHE_DIR, "STAGE_CACHE_LOCALLY=False needs an unpacked cache/ folder"
    LOCAL_CACHE = _CACHE_DIR
    _have = _cache_shortfall(LOCAL_CACHE)

# Optional delta on top of the base — idempotent by cache_path, manifest merged
# (copying files without their manifest entries reports every one as a miss).
_dz = next((z for z in (f"{DATA_ROOT}/{CACHE_DELTA_ZIP}", f"{PROJECT}/{CACHE_DELTA_ZIP}")
            if CACHE_DELTA_ZIP and os.path.exists(z)), None)
if _dz and _have is not None and STAGE_CACHE_LOCALLY:
    _dt = subprocess.run(["unzip", "-tqq", _dz], capture_output=True, text=True)
    assert _dt.returncode == 0, f"{_dz} is not a complete archive"
    _dtmp = os.path.join(scc.local_scratch(), "_delta")
    shutil.rmtree(_dtmp, ignore_errors=True)
    os.makedirs(_dtmp)
    subprocess.run(["unzip", "-q", "-o", _dz, "-d", _dtmp], check=True)
    _droot = next((r for r, _, f in os.walk(_dtmp) if "manifest.jsonl" in f), None)
    assert _droot, f"no manifest.jsonl inside {_dz}"
    _mpath = os.path.join(LOCAL_CACHE, "manifest.jsonl")
    _seen = {_json.loads(l)["cache_path"] for l in open(_mpath)}
    _new = [l for l in open(os.path.join(_droot, "manifest.jsonl"))
            if _json.loads(l)["cache_path"] not in _seen]
    for _l in _new:
        _rel = _json.loads(_l)["cache_path"]
        os.makedirs(os.path.dirname(os.path.join(LOCAL_CACHE, _rel)), exist_ok=True)
        shutil.copy(os.path.join(_droot, _rel), os.path.join(LOCAL_CACHE, _rel))
    with open(_mpath, "a") as _mf:
        _mf.writelines(_new)
    shutil.rmtree(_dtmp, ignore_errors=True)
    print(f"merged delta: +{len(_new)} manifest entries ({len(_seen)} already present)")
    _have = _cache_shortfall(LOCAL_CACHE)

assert _have is not None, f"no manifest.jsonl in {LOCAL_CACHE}"
_entries, _absent, _fmts = _have
print(f"cache: {len(_entries)} manifest entries at {LOCAL_CACHE}, {len(_absent)} absent | "
      f"formats {_fmts}")
_lossy = [d for d, f in _fmts if f not in ("png", "webp")]
assert not _lossy, (f"{_lossy} stored lossily — this is the superseded cache; rebuild "
                    "with the lossless policy")
assert not _absent, f"{len(_absent)} cached files missing after staging, e.g. {_absent[:3]}"

# %%
# ── GPU probe: every card the job holds, and the process ceiling ─────────────
import torch

assert torch.cuda.is_available(), "no GPU — request GPUs in the OnDemand form"
GPUS = scc.visible_gpus()
for _g in GPUS:
    print(f"  GPU {_g.id}: {_g.name}, {_g.capacity_gb:.1f} GB")
_cap = torch.cuda.get_device_capability(0)
# Select on compute capability, NOT torch.cuda.is_bf16_supported(): since torch 2.6
# that returns True on sm_75 via emulation, which is correct but glacial.
DTYPE = torch.bfloat16 if _cap[0] >= 8 else torch.float16
assert _cap[0] >= 8, (f"sm_{_cap[0]}{_cap[1]}: no native bf16 — request compute "
                      "capability >= 8.0 (A100/L40S/H100) in the OnDemand form")

MAX_PROCS = max(1, scc.granted_cores() - RESERVED_CORES)
_capacity = sum(int(g.capacity_gb * VRAM_HEADROOM // JOB_GB["tta"]) for g in GPUS)
print(f"{len(GPUS)} GPU(s), dtype={DTYPE}, cores {scc.granted_cores()} -> "
      f"max {MAX_PROCS} live worker(s); VRAM would take {_capacity} TTA workers")

# SCC's GPU nodes run in Exclusive_Process compute mode: ONE CUDA context per
# card, a second process fails at creation with cudaErrorDevicesUnavailable.
# The first SCC session found out by losing every second-tenant slice. So the
# VRAM packing above is moot here — concurrency is the card count — unless an
# MPS server is started inside the job (the lever to recover 3-4 workers per
# card; not attempted yet). Detected from nvidia-smi; override by hand if needed.
_modes = scc.compute_modes()
ONE_JOB_PER_GPU = scc.gpus_are_exclusive(_modes)
print(f"compute modes {_modes or '(nvidia-smi unavailable)'} -> "
      f"{'ONE job per GPU' if ONE_JOB_PER_GPU else 'VRAM-packed workers per GPU'}")
if ONE_JOB_PER_GPU:
    print(f"  effective concurrency: {min(len(GPUS), MAX_PROCS)} job(s) "
          f"(cards), not {min(_capacity, MAX_PROCS)}")
elif MAX_PROCS < min(_capacity, 4 * len(GPUS)):
    print(f"  NOTE: cores are the binding constraint ({MAX_PROCS} < {_capacity}) — "
          "request ~4 cores per GPU next session to fill the cards")
assert len(GPUS) >= 1

# %%
# ── shared machinery, copied VERBATIM from run_stage.py ──────────────────────
# tests/test_scc.py pins every one of these to the run_stage.py original by AST
# equality — edit THERE, re-copy here, or the suite fails. The VRAM planner
# (PEAK_GB, plan_group_workers, wait_all) is NOT copied: on SCC the packing is
# occl.scc.Scheduler's job, priced by S14_JOB_GB.
import subprocess
import time as _t


def worker_env(**extra):
    """The environment EVERY worker subprocess must get. One place, because it drifted.

    `fanout` set the thread caps; the three bespoke launchers written later (s8c, s8c_tune,
    s8d) each set only `PYTORCH_CUDA_ALLOC_CONF` and inherited the default thread pools.
    That is the measured regression between S6b and S8d: identical work — butterfly_major,
    5-shot, 100 episodes a run — ran at **2.95 s/image through `fanout`** and **~13 s/image
    through the s8d launcher**. Nothing about the protocol changed; only who set the
    environment.

    Why it bites butterfly at 5-shot and nothing before it. Each episode pushes six
    1024x1024 frames through SST's `load_video_frames`, which is `torch` tensor arithmetic
    on the CPU: a float64 divide, a float32 down-cast and a 75 MB normalise. Measured
    single-threaded that is **145 ms/episode at 6 frames against 48 ms at 2**. Unpinned,
    each of 8 workers opens a ~12-thread pool for it on a ~12-vCPU box — 96 threads over 12
    cores, all contending, all of it in front of the GPU. Beetle 1-shot (2 frames, 387x241,
    ~9 ms) never had enough CPU work for the thrash to show, which is why s8c passed through
    the same defect unnoticed.

    `expandable_segments` matters for a different reason: the allocator otherwise carves
    fixed segments per process and strands memory it cannot hand back, and with N processes
    that slack compounds — the OOM we hit reported ~300 MB reserved-but-unallocated in EACH
    of 10 workers.
    """
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env.update({k: str(v) for k, v in extra.items()})   # update, not dict(**) — a caller
    return env                                          # overriding a cap must not TypeError


GROUPS_OF = {
    "beetle": ["all"],
    "butterfly_major": ["cyrbia", "lativitta", "malleti",
                        "(malleti x plesseni) x malleti", "notabilis x lativitta"],
}


def run_slices(n_runs, n_workers):
    """Split ONE group's run range into `n_workers` near-equal slices."""
    edges = [round(i * n_runs / n_workers) for i in range(n_workers + 1)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def run(cmd: list[str]) -> None:
    # Tee the child through the kernel's stdout: a subprocess writing to its
    # inherited fds is invisible in the cell output, so a failing step surfaced as
    # a bare CalledProcessError with its traceback lost to the runtime log (the
    # first s14sec commissioning attempt died blind exactly this way).
    for _i, _c in enumerate(cmd):
        assert isinstance(_c, str), (
            f"cmd[{_i}] is {_c!r} (after {cmd[max(0, _i - 1)]!r}) — a config "
            f"constant this stage needs is unset; ' '.join would only say "
            f"'expected str instance'")
    print("$", " ".join(cmd), "\n", flush=True)
    p = subprocess.Popen(cmd, cwd=REPO,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)


def banked_runs(resume_dir=None) -> int:
    """How many per-run JSONs exist — a cheap, honest progress signal.

    Workers write to their own log files, so the notebook sees nothing between launch and
    exit. On a five-hour butterfly pass that is indistinguishable from a hang, and the only
    way to tell was to open a log by hand.
    """
    import glob as _g

    d = resume_dir or f"{RESULTS_ROOT}/runs"
    try:
        return len(_g.glob(os.path.join(d, "*.json")))
    except Exception:  # noqa: BLE001 — instrumentation must never fail a stage
        return -1


# ── SCC-specific: how a scheduled job becomes a pinned subprocess ────────────
_open_logs = []


def spawn(job, gpu_id):
    """Launch one scheduler job pinned to ONE card via CUDA_VISIBLE_DEVICES.

    The id is the string the parent saw in its own CUDA_VISIBLE_DEVICES (SGE may
    hand out indices or UUIDs — either passes through). Workers get worker_env()
    — the BLAS caps and the allocator setting every launcher must set — with the
    pin layered on top. Logs go to project space, so a session that dies at its
    wall-time still leaves every worker's output readable.
    """
    log = open(job.log, "w")
    _open_logs.append(log)
    log.write("$ " + " ".join(job.cmd) + f"\n[GPU {gpu_id}]\n\n")
    log.flush()
    return subprocess.Popen(job.cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            env=worker_env(CUDA_VISIBLE_DEVICES=gpu_id))


def _close_logs():
    for _f in _open_logs:
        try:
            _f.close()
        except Exception:  # noqa: BLE001
            pass
    _open_logs.clear()

# %%
# ── SAM 3: gated checkpoint ──────────────────────────────────────────────────
def _stop(msg):
    print("\n" + "=" * 72 + f"\nSTOP: {msg}\n" + "=" * 72, flush=True)
    raise RuntimeError("see the STOP message above") from None


_CKPT = f"{CKPT_ROOT}/sam3.pt"
if not os.path.exists(_CKPT):
    _tok = os.environ.get("HF_TOKEN")
    _tok_file = os.path.join(HF_HOME, "token")
    if not _tok and os.path.exists(_tok_file):
        _tok = open(_tok_file).read().strip()
    if not _tok:
        _stop("no HF token and no sam3.pt. sam3.pt is HF-GATED:\n"
              "  1. request access at huggingface.co/facebook/sam3 (manual approval — "
              "can take days);\n"
              "  2. create a token at huggingface.co/settings/tokens;\n"
              f"  3. on a login node: `echo <token> > {_tok_file}` (or export HF_TOKEN "
              "in the Pre-Launch Command);\n"
              "  4. re-run this cell.")
    from huggingface_hub import hf_hub_download

    try:
        _tmp = hf_hub_download("facebook/sam3", "sam3.pt", token=_tok)
    except Exception as _e:  # noqa: BLE001 — surface the real fix
        _stop(f"sam3.pt download failed ({type(_e).__name__}: {_e}).\n"
              "If this is a 403/gated error, the facebook/sam3 access request on this "
              "HF account is not approved yet. Nothing in s14sam3 can run without it.")
    # copy-then-rename so the existence guard can never pass on a half-copied file
    shutil.copy(_tmp, _CKPT + ".part")
    os.replace(_CKPT + ".part", _CKPT)
os.environ["OCCL_SAM3_CKPT"] = _CKPT
print("sam3 checkpoint ready:", _CKPT, f"({os.path.getsize(_CKPT) / 1e9:.2f} GB)")

# %%
# ── commissioning: the gate-shaped asserts, in seconds, on ONE card ──────────
# Strict 0/0 weight surgery, geometry pins 1008/1152/7, the global bf16 autocast
# exited, and cache bit-identity (cold vs precomputed) ON THIS GPU — the exact
# predicate that caught sec's autocast-context divergence, which no CPU run can
# see. Runs BEFORE anything spends GPU-hours.
_ph: dict[str, float] = {}
_stage_t0 = _t.time()


def _timed(_name):
    class _C:
        def __enter__(self):
            self.t0 = _t.time()

        def __exit__(self, *exc):
            _ph[_name] = round(_t.time() - self.t0, 1)
            print(f"  [timing] {_name}: {_ph[_name]:.0f}s", flush=True)

    return _C()


with _timed("commission"):
    run(["python", "scripts/commission_s14.py", "--family", "sam3",
         "--checkpoint", _CKPT, "--device", "cuda"])

# %%
# ── banked SAM 2 ledger cells (fail before training, not at the combine) ─────
_sam2 = {}
for _ds_name in S14_DATASETS:
    _sam2[_ds_name] = {
        "sam2_base": os.path.join(RESULTS_ROOT, f"s11_{_ds_name}_base.zip"),
        "sam2_base_tta": os.path.join(RESULTS_ROOT, f"s11_{_ds_name}_base_tta.zip"),
        "sam2_pertrait_tta": os.path.join(RESULTS_ROOT,
                                          f"s11_{_ds_name}_occl_pertrait_tta.zip"),
    }
    for _p in _sam2[_ds_name].values():
        assert os.path.exists(_p), (
            f"{_p} not in project space — S14 pairs against S11's banked SAM 2 cells "
            "and never recomputes them. Copy the three s11_<dataset>_*.zip files per "
            f"dataset into {RESULTS_ROOT}/ (they are on Drive under occl/results/).")
print("banked SAM 2 ledger present for", list(_sam2))

# %%
# ── the stage as ONE dependency graph over every card ────────────────────────
# Colab ran this serially: ft_stack, ft_arch, plain fan, TTA fan, merges — per
# dataset, one group at a time. Here every job is declared up front with its
# VRAM cost and dependencies, and the scheduler keeps every GPU busy:
#
#   ft_stack_<ds>, ft_arch_<ds>   exclusive card each (4 in flight on >= 4 GPUs)
#   plain_<ds>_g<i>_<k>            base arm only — NO adapter dependency, so the
#                                  plain fans fill the cards left over during the
#                                  fine-tunes (or wait for them on a 4-GPU grant)
#   tta_<ds>_g<i>_<k>              base + stack + arch in one worker; needs both
#                                  of its dataset's adapters
#   merge_<ds>_<cell>              inference-free replay of the banked runs into a
#                                  cell bundle; needs its fan (and adapter)
#
# A failed job skips everything downstream of it and the cell raises AFTER the
# rest of the graph has drained, so one OOM'd slice costs one re-run of this
# cell (every finished run is banked), not the session.
#
# Wall-time on SCC is a SIGKILL to every process in the job: no except path runs,
# the timing summary below never prints. What survives is on disk — banked runs,
# finished adapters, worker logs, and {LOG_DIR}/schedule.json (per-job timings).
# Run All in the next session resumes: finished adapters are reused after the
# config check, banked runs are skipped, a fine-tune killed mid-epoch restarts
# (it checkpoints only per epoch, and this recipe is one epoch).
BUNDLE = os.path.join(RESULTS_ROOT, f"{STAGE}.zip")

# the two LoRA scope sets, verbatim from occl.finetune.lora — the reuse check
# pins a cached checkpoint to the arm it claims to be
_SCOPES_STACK = ("maskmem_backbone", "sam_mask_decoder")
_SCOPES_ARCH = ("maskmem_backbone", "transformer", "sam_mask_decoder")


def _adapter_path(_ds_name, _suffix):
    return (f"{ADAPTER_DIR}/{_ds_name}_sam3_lora_lr{S14_LR:g}_ep{S14_EPOCHS}"
            f"_per_trait_excl-zerocat_noaug_derange_logit_reparam_eval_reset{_suffix}.pt")


def _purge_corrupt_runs(_dir):
    """Drop per-run JSONs a wall-time SIGKILL left half-written.

    The evaluator banks each run with a plain write and reads it back with a bare
    json.loads, so a truncated file does not get recomputed — it crashes the worker
    that meets it, and the scheduler then skips everything downstream. The evaluator
    is a fingerprinted module (editing it orphans every banked run on Drive), so
    the guard lives here: unparsable files are set aside as .corrupt and the run
    is simply recomputed.
    """
    _bad = 0
    for _f in _glob.glob(os.path.join(_dir, "*.json")):
        try:
            with open(_f) as _fh:
                _json.load(_fh)
        except (ValueError, OSError):
            os.replace(_f, _f + ".corrupt")
            _bad += 1
    if _bad:
        print(f"  set aside {_bad} half-written per-run file(s) in {_dir} "
              "(a previous session was killed mid-write) — they will be recomputed")


def _adapter_reusable(_pt, _scopes_expect):
    """A banked adapter is reused only if its meta says it IS this arm."""
    if not (S14_REUSE and os.path.exists(_pt)):
        return False
    try:
        _m = (torch.load(_pt, map_location="cpu", weights_only=False).get("meta") or {})
    except Exception as _e:  # noqa: BLE001 — a save cut off by a wall-time kill
        print(f"  {os.path.basename(_pt)} unreadable ({type(_e).__name__}) — "
              "setting aside and retraining", flush=True)
        os.replace(_pt, _pt + ".corrupt")
        return False
    _o = _m.get("occl") or {}
    # scopes live NESTED under meta["lora"] (save_lora packs record["finetune"]
    # wholesale) — reading the top level here would never match and silently
    # retrain every session
    _got_scopes = tuple((_m.get("lora") or {}).get("scopes") or ())
    ok = (_m.get("family") == "sam3"
          and _got_scopes == _scopes_expect
          and _o.get("mask_mode") == "per_trait"
          and (_o.get("excl") or {}).get("mode") == "zerocat"
          and not any(_o.get(k) is not None for k in ("xovl", "bshape", "margin", "scnp"))
          and _o.get("augmentation") is None)
    if not ok:
        print(f"  {os.path.basename(_pt)} config mismatch — retraining", flush=True)
    return ok


_purge_corrupt_runs(RUNS_DIR)

_jobs: list[scc.Job] = []
_adapters: dict[str, dict[str, str]] = {}
_ft_tags: dict[str, list[str]] = {}
_fan_tags: dict[str, dict[str, list[str]]] = {}
_cells: dict[str, dict[str, str]] = {}

for _ds_name in S14_DATASETS:
    _sel = S14_SEL[_ds_name]
    _adapters[_ds_name] = {"stack": _adapter_path(_ds_name, ""),
                           "arch": _adapter_path(_ds_name, "_arch")}
    _ft_tags[_ds_name] = []
    for _arm, _preset, _scopes in (("stack", "faithful", _SCOPES_STACK),
                                   ("arch", "sam3_arch", _SCOPES_ARCH)):
        _pt = _adapters[_ds_name][_arm]
        if _adapter_reusable(_pt, _scopes):
            print(f"  reusing banked adapter {os.path.basename(_pt)}")
            continue
        _tag = f"ft_{_arm}_{_ds_name}"
        _ft_tags[_ds_name].append(_tag)
        _jobs.append(scc.Job(
            tag=_tag, phase="finetune", gb=S14_JOB_GB["ft"],
            log=f"{LOG_DIR}/{_tag}.log",
            cmd=["python", "scripts/run_finetune.py",
                 "--family", "sam3", "--scope-preset", _preset,
                 "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                 "--checkpoint", _CKPT, "--device", "cuda",
                 "--method", "lora",
                 "--lr", repr(S14_LR), "--epochs", str(S14_EPOCHS),
                 "--mask-mode", "per_trait", "--excl-mode", "zerocat",
                 "--lora-out", _pt,
                 "--out", os.path.join(RESULTS_ROOT,
                                       f"s14ft_sam3_{_ds_name}{'_arch' if _arm == 'arch' else ''}.zip"),
                 "--no-download"]))

    _fan_tags[_ds_name] = {"plain": [], "tta": []}
    for _fan, _tta in (("plain", False), ("tta", True)):
        _armspec = (["--arms", "base="] if not _tta else
                    ["--arms", "base=",
                     "--arms", f"stack={_adapters[_ds_name]['stack']}",
                     "--arms", f"arch={_adapters[_ds_name]['arch']}"])
        for _gi, _group in enumerate(GROUPS_OF[_ds_name]):
            for _k, (_a, _b) in enumerate(run_slices(20, S14_SLICES[_fan])):
                _tag = f"{_fan}_{_ds_name}_g{_gi}_{_k}"
                _fan_tags[_ds_name][_fan].append(_tag)
                _jobs.append(scc.Job(
                    tag=_tag, phase=f"fan_{_fan}", gb=S14_JOB_GB[_fan],
                    deps=tuple(_ft_tags[_ds_name]) if _tta else (),
                    log=f"{LOG_DIR}/{_tag}.log",
                    cmd=["python", "scripts/run_arms.py",
                         "--adapter", "sam3",
                         "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                         "--checkpoint", _CKPT,
                         "--shot", "1", "--n-runs", "20",
                         "--support-selection", _sel,
                         "--resume-dir", RUNS_DIR,
                         "--groups", _group,
                         "--run-start", str(_a), "--run-end", str(_b)]
                        + _armspec + (["--tta"] if _tta else [])))

    _cells[_ds_name] = {}
    for _cell, _tta, _lora in (("base", False, None),
                               ("base_tta", True, None),
                               ("stack_tta", True, _adapters[_ds_name]["stack"]),
                               ("arch_tta", True, _adapters[_ds_name]["arch"])):
        _out = os.path.join(RESULTS_ROOT, f"{STAGE}_{_ds_name}_{_cell}.zip")
        _cells[_ds_name][_cell] = _out
        _tag = f"merge_{_ds_name}_{_cell}"
        _jobs.append(scc.Job(
            tag=_tag, phase="merge", gb=S14_JOB_GB["merge"],
            deps=tuple(_fan_tags[_ds_name]["tta" if _tta else "plain"])
            + tuple(_ft_tags[_ds_name] if _lora else ()),
            log=f"{LOG_DIR}/{_tag}.log",
            cmd=["python", "scripts/run_eval.py",
                 "--stage", f"{STAGE}_{_ds_name}_{_cell}",
                 "--adapter", "sam3", "--dataset", _ds_name,
                 "--cache-root", LOCAL_CACHE, "--checkpoint", _CKPT,
                 "--shot", "1", "--n-runs", "20",
                 "--support-selection", _sel,
                 "--resume-dir", RUNS_DIR, "--no-precompute",
                 "--out", _out, "--no-download"]
                + (["--tta"] if _tta else [])
                + (["--lora", _lora, "--lora-epochs", str(S14_EPOCHS)] if _lora else [])))

print(f"{len(_jobs)} jobs: "
      + ", ".join(f"{_p}={sum(1 for j in _jobs if j.phase == _p)}"
                  for _p in ("finetune", "fan_plain", "fan_tta", "merge")))

# The journal survives a wall-time kill (which is a SIGKILL — no cell output, no
# timing summary): per-job start/end/GPU, rewritten atomically on every event.
_sched = scc.Scheduler(gpus=GPUS, max_procs=MAX_PROCS, headroom=VRAM_HEADROOM,
                       stagger_s=STAGGER_S, journal=f"{LOG_DIR}/schedule.json",
                       exclusive_gpus=ONE_JOB_PER_GPU,
                       progress=lambda: f"{banked_runs()} runs banked")
with _timed("graph"):
    try:
        _results = _sched.run(_jobs, spawn, label="job")
    finally:
        _close_logs()

_bad = scc.failed(_results)
for _p, _s in _sched.phase_summary().items():
    print(f"  {_p:>10s}: {_s['jobs']:3d} jobs, wall {_s['wall_min']:6.1f} min, "
          f"GPU-min {_s['gpu_min']:7.1f}")
assert not _bad, (f"{len(_bad)} job(s) failed or were skipped: {_bad}\n"
                  f"  logs: {LOG_DIR}/<tag>.log — fix the cause and re-run THIS cell; "
                  "banked runs and adapters are reused, only the failed work recomputes")
for _ds_name in S14_DATASETS:
    for _pt in _adapters[_ds_name].values():
        assert os.path.exists(_pt), f"{_pt} was not written"

# ── the seven-pairing combine, per dataset, then the merged bundle (CPU) ─────
_per_ds = {}
with _timed("combine"):
    for _ds_name in S14_DATASETS:
        _dsb = os.path.join(RESULTS_ROOT, f"{STAGE}_{_ds_name}.zip")
        run(["python", "scripts/combine_s14.py", "--family", "sam3", "--out", _dsb,
             *[f"{k}={v}" for k, v in _cells[_ds_name].items()],
             *[f"{k}={v}" for k, v in _sam2[_ds_name].items()]])
        _per_ds[_ds_name] = _dsb
    run(["python", "scripts/combine_s14.py", "--family", "sam3", "--merge",
         "--out", BUNDLE, *[f"{k}={v}" for k, v in _per_ds.items()]])

print(f"\n=== {STAGE} timing summary (total {(_t.time() - _stage_t0) / 60:.1f} min, "
      f"{len(GPUS)} GPUs) ===")
for _k, _v in _ph.items():
    print(f"  {_k:>12s}: {_v / 60:6.1f} min")

# %%
# ── export ───────────────────────────────────────────────────────────────────
# The bundle already sits in project space (no browser download step on SCC).
# Fetch it to the machine that holds results/inbox/ and run the gate there.
assert os.path.exists(BUNDLE), (
    f"{BUNDLE} was not written — the stage cell did not complete; see {LOG_DIR}/")
_user = os.environ.get("USER", "<bu-login>")
print(f"bundle: {BUNDLE} ({os.path.getsize(BUNDLE) / 1e6:.1f} MB)\n")
print("fetch it, then gate it locally:")
print(f"  scp {_user}@scc1.bu.edu:{BUNDLE} results/inbox/")
print("  (or OnDemand -> Files -> navigate to the path above -> Download)")
print(f"  python scripts/check_gate.py {STAGE}")
print(f"\nfine-tune bundles for the s14ft gates: "
      f"{sorted(os.path.basename(p) for p in _glob.glob(f'{RESULTS_ROOT}/s14ft_sam3_*.zip'))}")
