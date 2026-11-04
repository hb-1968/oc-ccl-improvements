# %% [markdown]
# # s18 on the BU SCC — the cycle's information bottleneck (beetle + butterfly, single-shot)
#
# S17 measured the fine-tune from the inside and found the cycle satisfied without the
# forward mask improving: with the closing head frozen (`fwdonly`, in the established
# recipe) the closing leg still gains +1.8..+9.7 ✱ at forward ±0, and +2.4 ✱ of it sits
# on channels whose forward mask did not move. Under `fwdonly` + memory reset the only
# route from the forward leg to the closing frame is the PROMPT — and `logit_reparam`
# hands the reverse init the full continuous logit map (1024² floats per trait), a side
# channel for "what y₀ looked like" that the evaluator never scores (CycleGAN's
# steganography failure, in tracking form). The mask is NOT carried at the pixel level
# (forward-vs-y₀ similarity sits at chance, 0.127 vs 0.124); the sub-threshold content is
# the suspect. This session measures the channel on every arm and tests two treatments:
#
# | arm | what it is | what it asks |
# | :-- | :-- | :-- |
# | `crown` | the established recipe (`--recipe sam2_crown`), retrained | the control; its steganography gap vs base's is the premise's test |
# | `crown_ste` | + `--prompt-mode ste_threshold`: the release's BINARY prompt value with the reparam's gradient | the plain straight-through — measured on the real model it leaves the forward leg the POINTER route only (the memory reads sigmoid(±10), saturated): what does the prior alone teach? |
# | `crown_stemem` | + `--prompt-mode ste_memory`: the binary value everywhere, the memory encoder's input straight-through to the continuous logits | closing the channel WITH the paper's gradient — does what the cycle teaches change? |
# | `crown_perturb` | + `--close-aug-deg 10`: the closing end is the reference under its own rotation | the user's proposal 2; predicted null under fwdonly (memory attention transfers whatever sits on x's pixels) |
# | `crown_all` | `--lora-legs all`: the closing head trains again | the absorption route re-opened, for the perturb read |
# | `crown_all_perturb` | both | does perturbing the closing end curb absorption where it exists? |
# | `crown_hot` | `--lr 3e-5 --epochs 3` | S17c's overfit corner (closing +9.7 ✱, good-prompt bin −2.1 ✱) — where the channel showed most |
# | `crown_stemem_hot` | the corner under the memory STE | the sharp read: does the corner stop being one? |
#
# **The steganography read.** The probe replays every arm's closing legs under BOTH
# prompt values in one call — the continuous prompt and the forward logits binarised
# before the reverse init — so `cls_miou − cls_miou_hard` per pair is how much of the
# reconstruction rides on sub-threshold content. An `ste` arm reads its own closing off
# the `_hard` columns; a `perturb` arm pairs against `base_close10` (the base weights
# closing on the rotated reference).
#
# | needs | why |
# | :-- | :-- |
# | the code layer carrying `occl.BUILD >= 18` | `ste_threshold`, `--close-aug-deg`, the hard replay in the probe, the stego block in the combine and the s18 gates |
# | `ckpts/sam2_hiera_large.pt` | downloaded once by S17's notebook (sha256-pinned); re-fetched if absent |
# | the image cache (`data/cache`) | the pairs are the CACHED train images |
# | 4 GPUs, ~4–6 h (resumable) | per dataset: eight fine-tunes (beetle 30 s–3 min, butterfly 16–48 min each), one probe (8 arms + 2 bases, two closings per pair; butterfly ~1.5 h), 1-shot plain + TTA fans with 8 arms (butterfly's five groups dominate), 18 merges |
#
# Cells run top to bottom (`Run → Run All Cells`); a killed session reuses finished
# adapters, a complete probe bundle and banked per-run JSON. The stage ends by
# writing `s18.zip` into `$PROJECT/results/`; the export cell prints ONE line for the
# local machine (pull + gates + sheets, one password).
#
# PRE-REGISTERED (README "S18 brief"; the gate prints the bands, never sign-gates
# them): stego gap on the training pairs — base's level negative (the continuous prompt
# is out of distribution for the prompt encoder; the CPU smoke: −10 pp on 3 pairs),
# **crown's gap rises vs base's by +2..+5 ✱** (the channel exists; the premise's
# DISQUALIFIER if not above base), crown_hot more, crown_stemem ≈ base's, crown_perturb
# ≈ crown's; **crown_ste** (pointer route only) may fail the Dice disqualifier —
# recorded, not gated — and lands near the objness-alone arm on test (+0.4 TTA vs base);
# **crown_stemem vs crown** 1-shot TTA paired −0.3..+0.8; **crown_stemem_hot vs
# crown_hot**: good-prompt-bin forward ≥ −0.5 (was −2.1 ✱), forward Δ ≥ 0, test ≥
# crown_hot; **crown_perturb vs crown** ±0.3 (null under fwdonly — the mechanism
# argument's falsifier); **crown_all_perturb vs crown_all**: closing gain reduced, test
# ±0.5; base 63.20 / base_tta 63.86 on the ledger. Butterfly (the dataset where the
# forward leg DOES move under the cycle — S17 carrier +1.39 ✱ on its 1107 pairs): the
# same reads; crown's gap rises vs base +1..+4 ✱; crown_stemem vs crown TTA −0.5..+0.5,
# its forward gain on the training pairs within ±0.5 of crown's (the channel is not
# what butterfly learns from); base 70.30 / base_tta 75.18 on the ledger.

# %%
# ── configuration ────────────────────────────────────────────────────────────
import os

STAGE = "s18"            # fixed — this notebook runs exactly one stage

# Use OCCL_PROJECT_ROOT for the unpacked tree. Keep environments, caches, weights and
# results in project space, not the home quota.
SCC_PROJECT = os.environ.get("OCCL_PROJECT", "")
PROJECT = os.environ.get("OCCL_PROJECT_ROOT") or f"/projectnb/{SCC_PROJECT}/occl"
DATA_ROOT = f"{PROJECT}/data"          # occl-cache-png.zip and/or cache/ (manifest.jsonl)
RESULTS_ROOT = f"{PROJECT}/results"    # per-run JSON, adapters, bundles, logs — persistent
CKPT_ROOT = f"{PROJECT}/ckpts"         # sam2_hiera_large.pt lands here (898 MB, public)
HF_HOME = f"{PROJECT}/hf"              # HF hub cache + token, kept out of $HOME

CACHE_ZIP = "occl-cache-png.zip"
CACHE_DELTA_ZIP = "occl-cache-png-delta.zip"   # optional; merged idempotently if present
STAGE_CACHE_LOCALLY = True    # copy the cache to node-local $TMPDIR (recommended)

# ── the established SAM 2 recipe, moved along the S18 axes only ──────────────
S17_LR, S17_EPOCHS = 1e-5, 1
S17_DATASETS = ["beetle", "butterfly_major"]       # both, single-shot (user, 2026-09-18)
S17_REUSE = True              # a banked adapter / complete probe bundle is reused
S17_EVAL_PARITY = 3           # pairs per group also read through adapter.predict (recorded)
#: The arms, per dataset: label -> the run_finetune.py flags that make it. Every arm
#: starts from `--recipe sam2_crown` (registry.SAM2_CROWN_FLAGS, spliced in front by
#: run_finetune.py so a later flag overrides) and moves along ONE or TWO S18 axes at
#: their a-priori values — never retuned. `crown` is the control.
_CROWN = ["--recipe", "sam2_crown"]
_STE = ["--prompt-mode", "ste_threshold"]
_STEMEM = ["--prompt-mode", "ste_memory"]
_PERTURB = ["--close-aug-deg", "10"]
_HOT = ["--lr", "3e-05", "--epochs", "3"]
_S18_ARMS = {"crown": _CROWN,                                            # the control
             "crown_ste": [*_CROWN, *_STE],                              # pointer route only
             "crown_stemem": [*_CROWN, *_STEMEM],                        # the treatment
             "crown_perturb": [*_CROWN, *_PERTURB],
             "crown_all": [*_CROWN, "--lora-legs", "all"],
             "crown_all_perturb": [*_CROWN, "--lora-legs", "all", *_PERTURB],
             "crown_hot": [*_CROWN, *_HOT],
             "crown_stemem_hot": [*_CROWN, *_STEMEM, *_HOT]}
#: the same eight arms on both datasets (butterfly: 1107 pairs — ~16 min per epoch and
#: ~10 min of probe per arm on an L40S; the stage is resumable across sessions)
S17_ARMS = {"beetle": dict(_S18_ARMS), "butterfly_major": dict(_S18_ARMS)}
#: no 5-shot fan in S18 (three SAM 2 adapters measured null there)
S17_SHOT5_ARMS: list[str] = []
S17_CONTROL = "crown"
#: The test-split evaluation (S11 protocol): datasets that get the plain + TTA fans
S17_EVAL_DATASETS = ["beetle", "butterfly_major"]
S17_SEL = {"butterfly_major": "canonical_placement", "beetle": "canonical_balanced"}
S17_SLICES = {"plain": 5, "tta": 5}      # 20 runs / 5 slices = 4 runs a job

# ── layout knobs ─────────────────────────────────────────────────────────────
#: Per-job VRAM in GB. Both jobs are exclusive on a card (None): the fine-tune's
#: residency was never measured on SAM 2 at per-trait C<=10, and the probe is a
#: single process per dataset anyway. "tta" is only the sizing name the shared
#: GPU-probe cell prints a capacity line from; moot under SCC's exclusive mode.
S17_JOB_GB = {"ft": None, "probe": None, "plain": 7.5, "tta": 10.5, "merge": 8.0,
              "plain5": 9.5, "tta5": 12.5}
JOB_GB = S17_JOB_GB          # the stage-neutral name the shared GPU-probe cell reads
VRAM_HEADROOM = 0.92
STAGGER_S = 20               # one launch per interval: N workers must not load the ckpt at once
RESERVED_CORES = 1           # for the kernel itself

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
    # Stream child output into the cell so failures retain their traceback.
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
# ── SAM 2: public checkpoint ─────────────────────────────────────────────────
# The data layer deliberately ships no checkpoints (the sam3 stages never open
# the SAM 2 weights). This stage does: fetch the paper's SAM 2.0 hiera-large
# (072824) once into ckpts/ from Meta's public URL and verify its sha256 against
# the digest recorded in occl.paths — a silently different file would score a
# different model under the paper's name. copy-then-rename, so the existence
# guard can never pass on a half-downloaded file. wipe.sh keeps ckpts/.
import hashlib
import urllib.request


def _stop(msg):
    print("\n" + "=" * 72 + f"\nSTOP: {msg}\n" + "=" * 72, flush=True)
    raise RuntimeError("see the STOP message above") from None


def _sha256(_p):
    h = hashlib.sha256()
    with open(_p, "rb") as _f:
        while _chunk := _f.read(1 << 24):
            h.update(_chunk)
    return h.hexdigest()


_CKPT = f"{CKPT_ROOT}/sam2_hiera_large.pt"
if not os.path.exists(_CKPT):
    print(f"downloading {paths.SAM2_CKPT_URL} -> {_CKPT}", flush=True)
    try:
        urllib.request.urlretrieve(paths.SAM2_CKPT_URL, _CKPT + ".part")
    except Exception as _e:  # noqa: BLE001 — surface the real fix
        _stop(f"download failed ({type(_e).__name__}: {_e}). Compute nodes normally "
              "have outbound access; otherwise scp SST/checkpoints/sam2_hiera_large.pt "
              f"from the local machine to {_CKPT} and re-run this cell.")
    _got = _sha256(_CKPT + ".part")
    if _got != paths.SAM2_CKPT_SHA256:
        os.remove(_CKPT + ".part")
        _stop(f"sha256 mismatch: got {_got}, expected {paths.SAM2_CKPT_SHA256} — "
              "not the 072824 release; nothing downstream would be the paper's model")
    os.replace(_CKPT + ".part", _CKPT)
else:
    assert _sha256(_CKPT) == paths.SAM2_CKPT_SHA256, (
        f"{_CKPT} is not the 072824 sam2_hiera_large.pt (sha256 mismatch) — delete it "
        "and re-run this cell")
print("sam2 checkpoint ready:", _CKPT, f"({os.path.getsize(_CKPT) / 1e9:.2f} GB)")

# %%
# ── commissioning: one real pair through the probe, on ONE card ─────────────
# commission_s14.py is sam3/sec-only. Here the probe worker itself is the smoke:
# base arm, the carrier recipe from flags (no adapter exists yet), ONE beetle
# pair — SAM 2 builds strict 0/0 on this GPU, CyclePairs rebuilds the full
# derangement from the staged cache (120 pairs on beetle, or the cache is not
# the one the fine-tune will train on), both legs score, the bundle writes.
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


_COMMISSION = f"{LOG_DIR}/commission_probe.zip"
with _timed("commission"):
    run(["python", "scripts/probe_train_pairs.py",
         "--dataset", "beetle", "--cache-root", LOCAL_CACHE,
         "--checkpoint", _CKPT, "--device", "cuda",
         "--arms", "base=", "--mask-mode", "per_trait",
         "--max-pairs", "1", "--eval-parity", "1", "--save-maps", "none",
         "--out", _COMMISSION])
import json as _json  # noqa: E402
import zipfile as _zipfile  # noqa: E402

with _zipfile.ZipFile(_COMMISSION) as _z:
    _rec = _json.loads(_z.read("record.json"))
assert _rec["replay"]["n_pairs"] == 120 and _rec["replay"]["train_images"] == 120, (
    f"beetle rebuilt {_rec['replay']['n_pairs']} pairs from {_rec['replay']['train_images']} "
    "cached train images, not 120/120 — the staged cache is not the full beetle cache")
assert _rec["probe"]["complete"] and _rec["summary"]["base"]["n"] == 1
print(f"commissioned: pair 0 fwd {100 * _rec['summary']['base']['fwd_miou']:.1f} "
      f"cls {100 * _rec['summary']['base']['cls_miou']:.1f} "
      f"loss {_rec['summary']['base']['loss']:.4f}; eval-path parity "
      f"{_rec['eval_parity'][0]['labelmap_agreement']:.3f} label agreement")

# The S18 knobs are new code paths — the straight-through prompts (the memory one hooks
# the memory encoder's input on the model instance), the rotated closing end, the
# probe's hard replay: two training steps under the memory STE + the rotated closing
# end on ONE card WITH the gradient diagnostics (all four prompt modes on the trained
# weights: the plain STE must leave the pointer route only, the memory STE must sit
# within 10x of the reparam's gradient — the s18ft gate's predicate, asserted here
# first), then the probe on one pair with that adapter beside the base — the worker
# must derive `base_close10`, replay the arm's closing on the rotated reference, prove
# its step-0 loss off the HARD column (the STE's value is the binary prompt) against
# the record, and write the steganography column for every arm.
_S18_PT = f"{LOG_DIR}/commission_s18_lora.pt"
_S18_FT = f"{LOG_DIR}/commission_s18_ft.zip"
_S18_PROBE = f"{LOG_DIR}/commission_s18_probe.zip"
with _timed("commission_s18"):
    run(["python", "scripts/run_finetune.py",
         "--recipe", "sam2_crown", "--prompt-mode", "ste_memory", "--close-aug-deg", "10",
         "--dataset", "beetle", "--cache-root", LOCAL_CACHE,
         "--checkpoint", _CKPT, "--device", "cuda",
         "--max-steps", "2",
         "--lora-out", _S18_PT, "--out", _S18_FT, "--no-download"])
    with _zipfile.ZipFile(_S18_FT) as _z:
        _d = _json.loads(_z.read("record.json"))["diagnostics"]
    _pr = _d["grad_logit_reparam"]["per_channel_forward_to_reverse"]
    _pm = _d["grad_ste_memory"]["per_channel_forward_to_reverse"]
    _fd = _d["grad_ste_memory"]["per_channel_query_found"]
    assert _d["rel_grad_hard_threshold"] == 0.0 and _d["rel_grad_ste_threshold"] > 0.0, _d
    assert all(m > 0.1 * c for m, c, f in zip(_pm, _pr, _fd) if f), (_pm, _pr, _fd)
    print(f"  forward/reverse per channel: reparam {[round(v, 1) for v in _pr]} | plain STE "
          f"{[round(v, 3) for v in _d['grad_ste_threshold']['per_channel_forward_to_reverse']]} "
          f"| memory STE {[round(v, 1) for v in _pm]}")
    run(["python", "scripts/probe_train_pairs.py",
         "--dataset", "beetle", "--cache-root", LOCAL_CACHE,
         "--checkpoint", _CKPT, "--device", "cuda",
         "--arms", "base=", "--arms", f"ste_close={_S18_PT}", "--recipe-from", _S18_PT,
         "--train-record", f"ste_close={_S18_FT}",
         "--max-pairs", "1", "--eval-parity", "0", "--save-maps", "none",
         "--out", _S18_PROBE])
with _zipfile.ZipFile(_S18_PROBE) as _z:
    _rs = _json.loads(_z.read("record.json"))
assert _rs["arm_order"] == ["base", "base_close10", "ste_close"], _rs["arm_order"]
assert _rs["base_of"] == {"ste_close": "base_close10"} and _rs["arms"]["base_close10"]["close_aug"] == 10.0
assert _rs["arm_prompt_mode"]["ste_close"] == "ste_memory"
_pv = _rs["replay"]["parity"]["ste_close"]
assert _pv["base_arm"] == "base_close10" and _pv["column"] == "hard" and _pv["loss_first_rel_err"] < 1e-2, _pv
_b0 = next(r for r in _rs["pairs"] if r["arm"] == "base" and r["train_step"] == 0)
_c0 = next(r for r in _rs["pairs"] if r["arm"] == "base_close10" and r["train_step"] == 0)
assert _b0["cls_miou_hard"] is not None and _b0["loss_hard"] is not None, "no hard replay column"
assert _c0["close_aug"] == 10.0 and _c0["close_angle"] != "", _c0
print(f"commissioned S18: base closing continuous {100 * _b0['cls_miou']:.1f} vs hard "
      f"{100 * _b0['cls_miou_hard']:.1f} (stego gap {100 * (_b0['cls_miou'] - _b0['cls_miou_hard']):+.2f}); "
      f"closing end rotated by {_c0['close_angle']:+.2f}° -> continuous {100 * _c0['cls_miou']:.1f}; "
      f"replay proof off the hard column rel err {_pv['loss_first_rel_err']:.2e}")

# %%
# ── the stage as ONE dependency graph over every card ────────────────────────
# Per dataset: `ft_<ds>_<arm>` (each arm retrained, exclusive card) -> `probe_<ds>`
# (every arm, every training pair, one process). Beetle additionally runs the
# test-split evaluation: a plain fan and a TTA fan of `run_arms.py` carrying base
# + every adapter (one shared encode), sliced over runs, then one `run_eval.py`
# merge per cell — the s14sam3 shape. The scheduler places one job per card under
# SCC's exclusive compute mode; a job that fails within 90 s is re-queued once.
#
# Wall-time on SCC is a SIGKILL to every process in the job: no except path runs,
# the timing summary below never prints. What survives is on disk — finished
# adapters + their fine-tune bundles, complete probe bundles (written .part then
# renamed), banked per-run JSON, worker logs, and {LOG_DIR}/schedule.json. Run All
# in the next session reuses what finished.
BUNDLE = os.path.join(RESULTS_ROOT, f"{STAGE}.zip")


def _purge_corrupt_runs(_dir):
    """Drop per-run JSONs a wall-time SIGKILL left half-written (s15's guard: the
    evaluator reads banked runs back with a bare json.loads and a truncated file
    crashes the worker that meets it; the module is fingerprinted, so the guard
    lives here)."""
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


def _adapter_reusable(_pt, _bundle, _flags):
    """A banked adapter is reused only if its meta says it IS this arm AND its
    fine-tune bundle exists (the gate and the probe's --train-record need it)."""
    if not (S17_REUSE and os.path.exists(_pt) and os.path.exists(_bundle)):
        return False
    try:
        _m = (torch.load(_pt, map_location="cpu", weights_only=False).get("meta") or {})
    except Exception as _e:  # noqa: BLE001 — a save cut off by a wall-time kill
        print(f"  {os.path.basename(_pt)} unreadable ({type(_e).__name__}) — "
              "setting aside and retraining", flush=True)
        os.replace(_pt, _pt + ".corrupt")
        return False
    _o = _m.get("occl") or {}
    _flags = expand_recipe(list(_flags))       # `--recipe` -> its flags, in front
    # scopes live NESTED under meta["lora"] (save_lora packs record["finetune"]
    # wholesale) — reading the top level here would never match and silently
    # retrain every session
    _got_scopes = tuple((_m.get("lora") or {}).get("scopes") or ())
    _want = {"found_mask": ("--found-mask" in _flags) or None,
             "bce_w": (0.0 if "--bce-weight" in _flags else None),
             "excl": ({"mode": "zerocat"} if "--excl-mode" in _flags else None),
             "objness": ({"weight": _arm_weight(_flags), "target": "present_for_prompted"}
                         if "--objness" in _flags else None),
             "lora_legs": _arm_legs(_flags),
             "soft_sentinel": _arm_soft(_flags),
             "prompt_mode": _arm_after(_flags, "--prompt-mode", "logit_reparam"),
             "close_aug": _arm_close(_flags),
             "scopes": (tuple(SAM2_ARCH_SCOPES) if "sam2_arch" in _flags
                        else tuple(FAITHFUL_SCOPES_BY_FAMILY["sam2"]))}
    ok = (_m.get("family") == "sam2"
          and _got_scopes == _want["scopes"]
          and _m.get("epochs") == _arm_epochs(_flags)
          and float(_m.get("lr", 0)) == _arm_lr(_flags)
          and _o.get("mask_mode") == "per_trait"
          and _o.get("reference", "derangement") == "derangement"
          and not any(_o.get(k) is not None for k in ("xovl", "bshape", "margin", "scnp"))
          and _o.get("augmentation") is None
          and (_o.get("found_mask") is not None) == bool(_want["found_mask"])
          and _o.get("bce_w") == _want["bce_w"]
          and _o.get("excl") == _want["excl"]
          and _o.get("objness") == _want["objness"]
          and _o.get("lora_legs") == _want["lora_legs"]
          and _o.get("soft_sentinel") == _want["soft_sentinel"]
          and _o.get("prompt_mode", "logit_reparam") == _want["prompt_mode"]
          and (_o.get("close_aug") or {}).get("max_deg") == _want["close_aug"])
    if not ok:
        print(f"  {os.path.basename(_pt)} config mismatch — retraining", flush=True)
    return ok


def _arm_after(_flags, _flag, _default):
    """The value following the LAST `_flag` in an arm's flag list, else the default
    (argparse keeps the last occurrence: a flag after `--recipe` overrides the recipe's)."""
    if _flag not in _flags:
        return _default
    return _flags[len(_flags) - 1 - _flags[::-1].index(_flag) + 1]


def _arm_lr(_flags):
    return float(_arm_after(expand_recipe(list(_flags)), "--lr", S17_LR))


def _arm_epochs(_flags):
    return int(_arm_after(expand_recipe(list(_flags)), "--epochs", S17_EPOCHS))


def _arm_weight(_flags):
    return float(_arm_after(_flags, "--objness-weight", 1.0))


def _arm_legs(_flags):
    _v = _arm_after(_flags, "--lora-legs", None)
    return None if _v == "all" else _v          # the record writes None for "all"


def _arm_close(_flags):
    _v = _arm_after(_flags, "--close-aug-deg", None)
    return None if _v is None else float(_v)


def _arm_soft(_flags):
    _v = _arm_after(_flags, "--soft-sentinel", None)
    return None if _v is None else float(_v)


def _arm_knobs(_flags):
    """The arm's flags with --lr/--epochs removed (those are passed explicitly)."""
    out, skip = [], False
    for f in _flags:
        if skip:
            skip = False
            continue
        if f in ("--lr", "--epochs"):
            skip = True
            continue
        out.append(f)
    return out


def _probe_reusable(_zip, _pts):
    """A probe bundle is reused only if it is complete, covers every pair, and was
    scored on exactly THESE adapter files (lora_sha per arm)."""
    if not (S17_REUSE and os.path.exists(_zip) and all(os.path.exists(p) for p in _pts.values())):
        return False
    try:
        with _zipfile.ZipFile(_zip) as _z:
            _r = _json.loads(_z.read("record.json"))
    except Exception as _e:  # noqa: BLE001
        print(f"  {os.path.basename(_zip)} unreadable ({type(_e).__name__}) — "
              "setting aside and re-probing", flush=True)
        os.replace(_zip, _zip + ".corrupt")
        return False
    _rp = _r.get("replay") or {}
    _arms = _r.get("arms") or {}
    ok = (bool((_r.get("probe") or {}).get("complete"))
          and _rp.get("n_probed") == _rp.get("n_pairs")
          and not (_rp.get("checks") or {}).get("steps_capped")
          and set(_pts) == {a for a in _arms if _arms[a].get("path")}
          and all(_arms[a].get("lora_sha") == checkpoint_hash(p) for a, p in _pts.items()))
    if not ok:
        print(f"  {os.path.basename(_zip)} incomplete or from other adapters — "
              "re-probing", flush=True)
    return ok


from occl.finetune.lora import FAITHFUL_SCOPES_BY_FAMILY, SAM2_ARCH_SCOPES  # noqa: E402
from occl.finetune.train import checkpoint_hash  # noqa: E402
from occl.registry import expand_recipe  # noqa: E402

_purge_corrupt_runs(RUNS_DIR)

_jobs: list[scc.Job] = []
_ft_bundle: dict[str, dict[str, str]] = {}
_probe_bundle: dict[str, str] = {}
_adapters: dict[str, dict[str, str]] = {}
_ft_tags: dict[str, list[str]] = {}
_cells: dict[str, dict[str, str]] = {}
_fan_tags: dict[str, dict[str, list[str]]] = {}
for _ds_name in S17_DATASETS:
    _adapters[_ds_name], _ft_bundle[_ds_name], _ft_tags[_ds_name] = {}, {}, []
    for _arm, _flags in S17_ARMS[_ds_name].items():
        _pt = (f"{ADAPTER_DIR}/{_ds_name}_sam2_lora_lr{_arm_lr(_flags):g}_ep{_arm_epochs(_flags)}"
               f"_per_trait_noaug_derange_logit_reparam_eval_reset_{STAGE}_{_arm}.pt")
        _ftb = os.path.join(RESULTS_ROOT, f"{STAGE}ft_sam2_{_ds_name}_{_arm}.zip")
        _adapters[_ds_name][_arm], _ft_bundle[_ds_name][_arm] = _pt, _ftb
        if _adapter_reusable(_pt, _ftb, _flags):
            print(f"  {_ds_name}/{_arm}: reusing {os.path.basename(_pt)}")
            continue
        for _stale in (_pt, _ftb):
            if os.path.exists(_stale):
                os.replace(_stale, _stale + ".stale")
        _ft_tag = f"ft_{_ds_name}_{_arm}"
        _ft_tags[_ds_name].append(_ft_tag)
        _jobs.append(scc.Job(
            tag=_ft_tag, phase="finetune", gb=S17_JOB_GB["ft"],
            log=f"{LOG_DIR}/{_ft_tag}.log",
            # the arm's knobs (with `--recipe sam2_crown` first) BEFORE the explicit
            # lr/epochs: run_finetune.py splices the recipe's flags in front of argv and
            # argparse keeps the last value, so the notebook's lr/epochs are what runs
            cmd=["python", "scripts/run_finetune.py",
                 *_arm_knobs(_flags),
                 "--family", "sam2",
                 "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                 "--checkpoint", _CKPT, "--device", "cuda",
                 "--method", "lora", "--lr", repr(_arm_lr(_flags)),
                 "--epochs", str(_arm_epochs(_flags)),
                 "--mask-mode", "per_trait",
                 "--lora-out", _pt, "--out", _ftb, "--no-download"]))

    _prb = os.path.join(RESULTS_ROOT, f"{STAGE}probe_{_ds_name}.zip")
    _probe_bundle[_ds_name] = _prb
    if not _ft_tags[_ds_name] and _probe_reusable(_prb, _adapters[_ds_name]):
        print(f"  {_ds_name}: reusing {os.path.basename(_prb)}")
    else:
        if os.path.exists(_prb):
            os.replace(_prb, _prb + ".stale")
        _pr_tag = f"probe_{_ds_name}"
        _armspec = ["--arms", "base="]
        for _arm, _pt in _adapters[_ds_name].items():
            _armspec += ["--arms", f"{_arm}={_pt}"]
        _recspec = []
        for _arm, _ftb in _ft_bundle[_ds_name].items():
            _recspec += ["--train-record", f"{_arm}={_ftb}"]
        _jobs.append(scc.Job(
            tag=_pr_tag, phase="probe", gb=S17_JOB_GB["probe"],
            deps=tuple(_ft_tags[_ds_name]),
            log=f"{LOG_DIR}/{_pr_tag}.log",
            cmd=["python", "scripts/probe_train_pairs.py",
                 "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                 "--checkpoint", _CKPT, "--device", "cuda",
                 *_armspec, "--recipe-from", _adapters[_ds_name][S17_CONTROL], *_recspec,
                 "--eval-parity", str(S17_EVAL_PARITY),
                 "--stage", f"{STAGE}probe", "--out", _prb]))

    # the test split (S11 protocol): plain + TTA fans over base + every adapter
    if _ds_name not in S17_EVAL_DATASETS:
        continue
    _sel = S17_SEL[_ds_name]
    _fan_tags[_ds_name] = {"plain": [], "tta": []}
    _armspec = ["--arms", "base="]
    for _arm, _pt in _adapters[_ds_name].items():
        _armspec += ["--arms", f"{_arm}={_pt}"]
    for _fan, _tta in (("plain", False), ("tta", True)):
        for _gi, _group in enumerate(GROUPS_OF[_ds_name]):
            for _k, (_a, _b) in enumerate(run_slices(20, S17_SLICES[_fan])):
                _tag = f"{_fan}_{_ds_name}_g{_gi}_{_k}"
                _fan_tags[_ds_name][_fan].append(_tag)
                _jobs.append(scc.Job(
                    tag=_tag, phase=f"fan_{_fan}", gb=S17_JOB_GB[_fan],
                    deps=tuple(_ft_tags[_ds_name]),
                    log=f"{LOG_DIR}/{_tag}.log",
                    cmd=["python", "scripts/run_arms.py",
                         "--adapter", "sst_sam2",
                         "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                         "--checkpoint", _CKPT,
                         "--shot", "1", "--n-runs", "20",
                         "--support-selection", _sel,
                         "--resume-dir", RUNS_DIR,
                         "--groups", _group,
                         "--run-start", str(_a), "--run-end", str(_b)]
                        + _armspec + (["--tta"] if _tta else [])))
    _cells[_ds_name] = {}
    for _arm in ["base", *_adapters[_ds_name]]:
        for _tta in (False, True):
            _cell = _arm + ("_tta" if _tta else "")
            _out = os.path.join(RESULTS_ROOT, f"{STAGE}_{_ds_name}_{_cell}.zip")
            _cells[_ds_name][_cell] = _out
            _lora = _adapters[_ds_name].get(_arm)
            _tag = f"merge_{_ds_name}_{_cell}"
            _jobs.append(scc.Job(
                tag=_tag, phase="merge", gb=S17_JOB_GB["merge"],
                deps=tuple(_fan_tags[_ds_name]["tta" if _tta else "plain"])
                + tuple(_ft_tags[_ds_name] if _lora else ()),
                log=f"{LOG_DIR}/{_tag}.log",
                cmd=["python", "scripts/run_eval.py",
                     "--stage", f"{STAGE}_{_ds_name}_{_cell}",
                     "--adapter", "sst_sam2", "--dataset", _ds_name,
                     "--cache-root", LOCAL_CACHE, "--checkpoint", _CKPT,
                     "--shot", "1", "--n-runs", "20",
                     "--support-selection", _sel,
                     "--resume-dir", RUNS_DIR, "--no-precompute",
                     "--out", _out, "--no-download"]
                    + (["--tta"] if _tta else [])
                    + (["--lora", _lora, "--lora-epochs",
                        str(_arm_epochs(S17_ARMS[_ds_name][_arm]))] if _lora else [])))

    # the 5-shot transfer fan (S6's protocol): base + S17_SHOT5_ARMS, plain + TTA
    if not S17_SHOT5_ARMS:
        continue                                   # S18: no 5-shot fan
    _armspec5 = ["--arms", "base="]
    for _arm in S17_SHOT5_ARMS:
        _armspec5 += ["--arms", f"{_arm}={_adapters[_ds_name][_arm]}"]
    _fan_tags[_ds_name]["plain5"], _fan_tags[_ds_name]["tta5"] = [], []
    for _fan, _tta in (("plain5", False), ("tta5", True)):
        for _gi, _group in enumerate(GROUPS_OF[_ds_name]):
            for _k, (_a, _b) in enumerate(run_slices(20, S17_SLICES["tta" if _tta else "plain"])):
                _tag = f"{_fan}_{_ds_name}_g{_gi}_{_k}"
                _fan_tags[_ds_name][_fan].append(_tag)
                _jobs.append(scc.Job(
                    tag=_tag, phase=f"fan_{_fan}", gb=S17_JOB_GB[_fan],
                    deps=tuple(_ft_tags[_ds_name]),
                    log=f"{LOG_DIR}/{_tag}.log",
                    cmd=["python", "scripts/run_arms.py",
                         "--adapter", "sst_sam2",
                         "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                         "--checkpoint", _CKPT,
                         "--shot", "5", "--n-runs", "20",
                         "--support-selection", _sel,
                         "--resume-dir", RUNS_DIR,
                         "--groups", _group,
                         "--run-start", str(_a), "--run-end", str(_b)]
                        + _armspec5 + (["--tta"] if _tta else [])))
    for _arm in ["base", *S17_SHOT5_ARMS]:
        for _tta in (False, True):
            _cell = _arm + "_5" + ("_tta" if _tta else "")
            _out = os.path.join(RESULTS_ROOT, f"{STAGE}_{_ds_name}_{_cell}.zip")
            _cells[_ds_name][_cell] = _out
            _lora = _adapters[_ds_name].get(_arm)
            _tag = f"merge_{_ds_name}_{_cell}"
            _jobs.append(scc.Job(
                tag=_tag, phase="merge", gb=S17_JOB_GB["merge"],
                deps=tuple(_fan_tags[_ds_name]["tta5" if _tta else "plain5"])
                + tuple(_ft_tags[_ds_name] if _lora else ()),
                log=f"{LOG_DIR}/{_tag}.log",
                cmd=["python", "scripts/run_eval.py",
                     "--stage", f"{STAGE}_{_ds_name}_{_cell}",
                     "--adapter", "sst_sam2", "--dataset", _ds_name,
                     "--cache-root", LOCAL_CACHE, "--checkpoint", _CKPT,
                     "--shot", "5", "--n-runs", "20",
                     "--support-selection", _sel,
                     "--resume-dir", RUNS_DIR, "--no-precompute",
                     "--out", _out, "--no-download"]
                    + (["--tta"] if _tta else [])
                    + (["--lora", _lora, "--lora-epochs",
                        str(_arm_epochs(S17_ARMS[_ds_name][_arm]))] if _lora else [])))

print(f"{len(_jobs)} jobs: "
      + ", ".join(f"{_p}={sum(1 for j in _jobs if j.phase == _p)}"
                  for _p in ("finetune", "probe", "fan_plain", "fan_tta", "fan_plain5",
                             "fan_tta5", "merge")))

# The journal survives a wall-time kill (which is a SIGKILL — no cell output, no
# timing summary): per-job start/end/GPU, rewritten atomically on every event.
if _jobs:
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
                      "finished adapters, probe bundles and banked runs are reused, only "
                      "the failed work recomputes")
for _ds_name in S17_DATASETS:
    for _arm in S17_ARMS[_ds_name]:
        assert os.path.exists(_adapters[_ds_name][_arm]) \
            and os.path.exists(_ft_bundle[_ds_name][_arm]), \
            f"{_ds_name}/{_arm}: fine-tune left no adapter/bundle"
    assert os.path.exists(_probe_bundle[_ds_name]), f"{_ds_name}: no probe bundle"

# ── the per-pair pairing (+ the test-split cells), per dataset, then the merge (CPU) ──
_per_ds = {}
with _timed("combine"):
    for _ds_name in S17_DATASETS:
        _dsb = os.path.join(RESULTS_ROOT, f"{STAGE}_{_ds_name}.zip")
        _spec = [f"probe={_probe_bundle[_ds_name]}"]
        _spec += [f"ft:{_arm}={_ftb}" for _arm, _ftb in _ft_bundle[_ds_name].items()]
        _spec += [f"eval:{_cell}={_out}" for _cell, _out in _cells.get(_ds_name, {}).items()]
        run(["python", "scripts/combine_s17.py", "--stage", STAGE, "--out", _dsb,
             "--control", S17_CONTROL, *_spec])
        _per_ds[_ds_name] = _dsb
    run(["python", "scripts/combine_s17.py", "--stage", STAGE, "--merge", "--out", BUNDLE,
         *[f"{k}={v}" for k, v in _per_ds.items()]])

print(f"\n=== {STAGE} timing summary (total {(_t.time() - _stage_t0) / 60:.1f} min, "
      f"{len(GPUS)} GPUs) ===")
for _k, _v in _ph.items():
    print(f"  {_k:>12s}: {_v / 60:6.1f} min")

# %%
# ── export ───────────────────────────────────────────────────────────────────
# The bundle already sits in project space (no browser download step on SCC).
# Fetch it to the machine that holds results/inbox/ and run the gates there;
# the sheets are rendered locally from the image cache.
assert os.path.exists(BUNDLE), (
    f"{BUNDLE} was not written — the stage cell did not complete; see {LOG_DIR}/")
_user = os.environ.get("USER", "<bu-login>")
print(f"bundle: {BUNDLE} ({os.path.getsize(BUNDLE) / 1e6:.1f} MB)\n")
# ONE command for the local machine, from the repo root: a single ssh session streams
# every bundle as a tar (one password), then the gates and the sheets run in order and
# the chain stops at the first gate that fails (nothing is filed or rendered past it).
# the per-cell bundles ({STAGE}_<ds>_<cell>.zip) ride along into results/inbox/cells/:
# the deployment reads (an arm vs another arm, paired per seed) run on them locally with
# scripts/pair_cells.py, and the merged bundle carries only each cell's summary.
_cells = " ".join(f"{STAGE}_{_ds_name}_*.zip" for _ds_name in S17_DATASETS)
_pull = (f"ssh {_user}@scc1.bu.edu 'cd {RESULTS_ROOT} && tar cf - "
         f"{os.path.basename(BUNDLE)} {STAGE}ft_sam2_*.zip {_cells}' | tar xf - -C results/inbox/ "
         f"&& mkdir -p results/inbox/cells && mv results/inbox/{STAGE}_*_*.zip results/inbox/cells/")
# the local steps name the repo venv explicitly: the system python has no matplotlib
# (the render died on it once), and an activated venv is not something a pasted line
# can assume.
_PY = ".venv/bin/python"
_gates = [f"{_PY} scripts/check_gate.py {STAGE}"]
for _ds_name in S17_DATASETS:
    for _arm in S17_ARMS[_ds_name]:
        _gates.append(f"{_PY} scripts/check_gate.py {STAGE}ft --bundle "
                      f"results/inbox/{STAGE}ft_sam2_{_ds_name}_{_arm}.zip")
_sheets = [f"{_PY} scripts/render_probe_sheets.py --bundle results/{STAGE}.zip "
           f"--dataset {_ds_name} --select most_degraded --figures"
           for _ds_name in S17_DATASETS]
print("paste this ONE line into a terminal at the repo root (one password prompt):\n")
print(" && ".join([_pull, *_gates, *_sheets]))
print("\n(or OnDemand -> Files -> Download each bundle into results/inbox/ and run the "
      "gates by hand)")
