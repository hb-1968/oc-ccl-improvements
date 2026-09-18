# %% [markdown]
# # s14sam3 — the crowned stack on SAM 3, standalone
#
# One stage, nothing else. S13 closed the loss program and located beetle's
# residual (~33 pts) in regions SAM 2 never proposes; S14sec measured the first
# model swap (SeC-4B: **does not pay**, −2.8✱/−3.0✱ cross-family). This notebook
# is the second and final swap: the crowned stack — per-trait OC-CCL +
# `excl` zerocat + TTA rot180 — on `facebook/sam3`'s plain tracker, **plus the
# architecture-adapted arm**: LoRA additionally on `transformer` (sam3's memory
# attention, the proposal-forming module no arm on any family has ever trained;
# a strict scope superset, so `arch − stack` is attributable to that one
# addition).
#
# | needs | why |
# | :-- | :-- |
# | fresh runtime, A100 | never one that ran s14sec (its pip downgraded numpy/hub — a guard cell stops early) |
# | `HF_TOKEN` Colab secret, notebook access ON | `sam3.pt` is HF-gated, manual approval — the env cell fails fast with instructions |
# | `occl-repo.zip` on Drive (current) | the repo cell asserts it carries this notebook |
# | `occl-cache-png.zip` + S11 cell bundles on Drive | pixels + the banked SAM 2 ledger |
#
# Cells run top to bottom (`Runtime -> Run all`). Every phase is timed; the
# stage ends by writing `s14sam3.zip`, copying it to Drive, and downloading it —
# drop it into `results/inbox/` locally and run `python scripts/check_gate.py
# s14sam3`. Predictions P1/P2/P5/P6/P7/P8 are frozen in the gate and below.

# %%
# ── configuration ────────────────────────────────────────────────────────────
STAGE = "s14sam3"        # fixed — this notebook runs exactly one stage

DRIVE = "/content/drive/MyDrive/occl"
CACHE_ROOT = f"{DRIVE}/cache"          # derived 1024^2 images (~290 MB)
ORIGINALS_ROOT = f"{DRIVE}/originals"  # untouched source bytes; unused here
RESULTS_ROOT = f"{DRIVE}/results"      # per-run JSON, so a session boundary is survivable

REPO_SOURCE = "drive_zip"
REPO_ZIP = "occl-repo.zip"          # filename to look for on Drive
REPO_DRIVE_PATH = f"{DRIVE}/oc-ccl-improvements"
REPO_GIT_URL = ""

CACHE_ZIP = "occl-cache-png.zip"
CACHE_DELTA_ZIP = "occl-cache-png-delta.zip"

WORKERS = "auto"
VRAM_HEADROOM = 0.92
STAGGER_S = 20

# ── the S14 recipe (identical to s14sec's sam2-side constants) ───────────────
S14_LR, S14_EPOCHS = 1e-5, 1
S14_REUSE = True
S14_SAM3_CKPT_DRIVE = "occl_ckpts/sam3.pt"   # 3.45 GB, HF-gated — downloaded once
#: eval-worker widths, MEASURED on the first butterfly session (A100 40 GB), not
#: projected — the projection said ~6.6 GB for a TTA worker and the card said
#: ~9.8-10.0 GB (dual-orientation fp32 cache + sam3's decoder workspace across
#: three arms): 4 wide = 39.4 of 39.49 GB, two groups squeaked through and the
#: third OOM'd. Plain workers finished 4-wide cleanly. Passing these as
#: `requested` OVERRIDES the planner's per-worker fit deliberately; vCPU and
#: run-count ceilings still apply inside plan_group_workers.
S14_EVAL_WORKERS = {"plain": 4, "tta": 3}
S14_DATASETS = ["butterfly_major", "beetle"]
S14_SEL = {"butterfly_major": "canonical_placement",   # each dataset's best 1-shot
           "beetle": "canonical_balanced"}

# PRE-REGISTERED PREDICTIONS (frozen before any sam3 GPU run; the gate prints,
# never sign-gates):
#   P1 base vs sam2_base (cross-family, paired on identical draws): optimist
#      beetle +1..+8, butterfly +1..+6; prior ±2.
#   P2 THE MECHANISM READ: beetle oracle−argmax on base. > ~+5: proposals
#      improved, assignment lags -> excl's pot is real, P5 optimist activates.
#      ≈ +2.65: same proposal boundary — the upgrade is cosmetic here.
#   P5 stack_tta vs base_tta: prior ±0.3 (the S13 law); conditional optimist
#      +0.5..+2 iff the P2 read ≥ ~3 pts (S14sec measured +0.60✱/+0.58✱ on sec).
#   P6 TTA's value (base_tta vs base): recorded, not banded — S14sec measured
#      +4.49✱ butterfly / −1.80✱ beetle on sec; a beetle-negative here too would
#      make the confidence rule's miscalibration a cross-family fact.
#   P7 HEADLINE: stack_tta vs sam2_pertrait_tta (76.52 / 64.12) — any CI > 0
#      means the model upgrade pays under the deployment stack.
#   P8 arch vs stack (the architecture-adapted arm): conditional optimist beetle
#      +0.5..+3 iff P2 shows the proposal boundary persisting (the transformer is
#      the only trainable module that can change what memory proposes); prior
#      ±0.5; disqualifiers: divergent loss (this module was never cycle-trained
#      anywhere) or butterfly < −1 (S13's margin-style calibration damage).

# %%
# ── mount Drive ──────────────────────────────────────────────────────────────
import os

from google.colab import drive

drive.mount("/content/drive")

# Resolve the cache wherever it actually landed. The cache root is whichever directory
# *directly* contains `manifest.jsonl` alongside the dataset folders — so uploading
# `oc-ccl-data/` wholesale puts it at `.../oc-ccl-data/cache`, not at `occl/cache`.
# Probing a handful of candidates beats walking Drive, which can be very slow.
_MD = "/content/drive/MyDrive"
_CANDIDATES = [
    CACHE_ROOT,
    f"{_MD}/oc-ccl-data/cache",
    f"{DRIVE}/oc-ccl-data/cache",
    f"{_MD}/cache",
]
for _c in _CANDIDATES:
    if os.path.exists(os.path.join(_c, "manifest.jsonl")):
        CACHE_ROOT = _c
        break
else:
    # A cache *zip* on Drive is sufficient on its own — the staging cell extracts it —
    # so do not demand an unpacked folder as well. Uploading only the zip is the
    # recommended shape: 2.5 GB as one file rather than ~1,500 FUSE writes.
    _zc = [f"{_MD}/{CACHE_ZIP}", f"{DRIVE}/{CACHE_ZIP}", f"{_MD}/oc-ccl-data/{CACHE_ZIP}"]
    if not any(os.path.exists(z) for z in _zc):
        raise FileNotFoundError(
            "Could not find the image cache. Looked for manifest.jsonl in:\n  "
            + "\n  ".join(_CANDIDATES)
            + f"\nand for {CACHE_ZIP} in:\n  " + "\n  ".join(_zc)
        )
    CACHE_ROOT = None
    print(f"no unpacked cache folder; will extract {CACHE_ZIP}")

os.makedirs(RESULTS_ROOT, exist_ok=True)
print("drive ready | cache:", CACHE_ROOT or f"(from {CACHE_ZIP})")
if CACHE_ROOT:
    print("  datasets:", sorted(d for d in os.listdir(CACHE_ROOT)
                                if not d.endswith(".jsonl")))

# %%
# ── get this repo onto the runtime ───────────────────────────────────────────
# Installed from /content rather than in place on Drive: an editable install through
# Drive's FUSE layer is slow and occasionally serves stale bytes.
REPO = "/content/occl-repo"

# Copy an explicit allowlist, never `cp -r` the whole folder. A local `.venv` is ~2,800
# files, and Drive's FUSE layer manages only a handful of small files per second, so a
# blind recursive copy stalls for 10-30 minutes while appearing to do nothing — and then
# yields a venv that is useless anyway (wrong Python, absolute local paths). An allowlist
# means FUSE never descends into it. The real payload is ~37 files / 292 KB.
REPO_CONTENTS = ("src", "scripts", "notebooks", "tests", "pyproject.toml", "README.md")

if REPO_SOURCE == "drive_zip":
    # Probe the usual spots rather than pinning one path.
    _zips = [f"{_MD}/{REPO_ZIP}", f"{DRIVE}/{REPO_ZIP}",
             f"{_MD}/oc-ccl-data/{REPO_ZIP}", f"{REPO_DRIVE_PATH}/{REPO_ZIP}"]
    _zip = next((z for z in _zips if os.path.exists(z)), None)
    assert _zip, ("could not find " + REPO_ZIP + " on Drive. Looked in:\n  "
                  + "\n  ".join(_zips))
    print(f"extracting {_zip} ({os.path.getsize(_zip) / 1024:.0f} KB)")
    !rm -rf {REPO} && mkdir -p {REPO} && unzip -q -o "{_zip}" -d {REPO}
    # The zip and this notebook travel separately, and nothing else pairs them: the
    # first S10 session ran a current notebook against a stale zip, and the failure
    # surfaced 400 lines later as `unrecognized arguments` in six worker logs. Cheapest
    # honest check: the extracted copy of this very file must know the stage this
    # session is about to run — a zip predating a new stage fails HERE, naming the fix.
    # The standalone pairing check: the zip must carry this very notebook source —
    # a zip predating the standalone fails HERE, naming the fix, instead of 400
    # lines later as `unrecognized arguments` in six worker logs.
    assert os.path.exists(f"{REPO}/notebooks/run_s14sam3.py"), (
        f"{os.path.basename(_zip)} predates the standalone s14sam3 notebook — "
        "rebuild with scripts/build_repo_zip.py and replace the zip on Drive.")
elif REPO_SOURCE == "drive":
    assert os.path.isdir(REPO_DRIVE_PATH), (
        f"{REPO_DRIVE_PATH} not found. Copy the oc-ccl-improvements folder to Drive once "
        "(it is only a few hundred KB), or switch REPO_SOURCE."
    )
    !rm -rf {REPO} && mkdir -p {REPO}
    for _item in REPO_CONTENTS:
        _src = os.path.join(REPO_DRIVE_PATH, _item)
        if os.path.exists(_src):
            !cp -r "{_src}" {REPO}/
        else:
            print(f"  missing on Drive (skipped): {_item}")
elif REPO_SOURCE == "git":
    assert REPO_GIT_URL, "set REPO_GIT_URL"
    !rm -rf {REPO} && git clone -q {REPO_GIT_URL} {REPO}
else:
    from google.colab import files
    up = files.upload()
    !rm -rf {REPO} && mkdir -p {REPO} && unzip -q -o "{list(up)[0]}" -d {REPO}

%cd {REPO}
_n = sum(len(f) for _, _, f in os.walk(REPO))
print(f"{_n} files copied")
assert os.path.isdir(f"{REPO}/src/occl"), "src/occl missing — check REPO_DRIVE_PATH"
if _n > 500:
    print(f"WARNING: {_n} files is far more than the expected ~37. A .venv or the image "
          "cache has probably been copied into the repo folder on Drive.")
!ls src/occl

# %%
# ── dependencies ─────────────────────────────────────────────────────────────
# ONE pip line, EMPIRICAL: a local build+predict on random weights was traced
# through sys.modules and diffed against Colab's preinstalls (the method that
# would have caught sec's missing iopath before its first session). hydra-core /
# omegaconf are on the tracker's import path and Colab ships neither. NEVER
# `pip install -e` the sam3 repo itself — its numpy<2 pin would downgrade the
# runtime this stage was validated on.
!pip install -q -e . requests pillow tqdm timm iopath ftfy regex einops \
    pycocotools hydra-core omegaconf

# Fresh-runtime guard: an s14sec session's pip downgraded numpy to 1.26 and
# huggingface-hub below 1.0 for transformers 4.42.3. sam3 was never validated on
# that mixture — a fresh runtime is one click, so stop here rather than debug an
# untested hybrid forty minutes in.
import numpy as _np

assert not _np.__version__.startswith("1."), (
    f"numpy {_np.__version__}: this runtime already ran an s14sec session (its "
    "pip line downgrades numpy). Runtime -> Disconnect and delete runtime, then "
    "Run-all on a fresh one.")

# Put src/ on sys.path explicitly — an editable install's .pth is only read at
# interpreter startup, and this kernel is already running.
import sys

_SRC = f"{REPO}/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import occl  # noqa: F401  — fail here, next to the fix, rather than three cells later

# The notebook and the repo zip travel as a pair; only the notebook is open in
# front of you. Both directions are asserted (see run_stage.py's deps cell for
# the two sessions this pairing check has already paid for).
REPO_MIN_BUILD = 10
NOTEBOOK_BUILD = 10
_build = getattr(occl, "BUILD", 0)
assert _build >= REPO_MIN_BUILD, (
    f"occl-repo.zip on Drive is build {_build}; this notebook needs >= "
    f"{REPO_MIN_BUILD}. Re-upload BOTH occl-repo.zip and run_s14sam3.ipynb, "
    "then restart the runtime.")
_want_nb = getattr(occl, "NOTEBOOK_MIN_BUILD", 0)
assert NOTEBOOK_BUILD >= _want_nb, (
    f"this notebook is build {NOTEBOOK_BUILD}; occl-repo.zip (build {_build}) "
    f"needs a notebook >= {_want_nb}. Re-download run_s14sam3.ipynb from the "
    "same build.")
print(f"occl {occl.__version__} (build {_build}) importable from {_SRC}")

# %%
# ── SST checkout (masks, split manifests, vendored SAM 2) ────────────────────
SST = "/content/SST"
if not os.path.isdir(SST):
    !git clone -q --depth 1 https://github.com/Imageomics/SST {SST}
os.environ["OCCL_SST_ROOT"] = SST

from occl import paths

print("SST at", paths.sst_root())
print("butterfly masks:", paths.butterfly_root().exists(), "| beetle:", paths.beetle_root().exists())

# %%
# ── stage the image cache onto local disk ────────────────────────────────────
# Every read during evaluation then hits local NVMe instead of Drive's FUSE layer. S5
# alone reads 598 images across 10,000 episodes, and FUSE small-file latency would
# dominate. Colab gives ~236 GB of local disk, so this is free. Skipped for s2, which
# writes the cache rather than reading it.
#
# The cache is LOSSLESS PNG (~2.5 GB). It was JPEG q95 until measurement showed that
# inserted an encode SST does not have — its read_image goes decode -> cv2.resize ->
# model with no intermediate file. Lossless makes our cache a pure memoisation of that
# path, verified bit-exact (0 deviation) for both the PIL and rawpy decode routes.
LOCAL_CACHE = CACHE_ROOT
if STAGE != "s2":
    import json as _json
    import subprocess as _sub
    import time as _t

    LOCAL_CACHE = "/content/occl-cache"

    def _cache_shortfall(root):
        """(n_entries, missing_paths, formats) for a staged cache — or None if unusable.

        Checked against the manifest rather than a file count: the manifest is the
        authority on what should be present, and `ImageCache.has` consults it. A count
        threshold cannot tell a complete small cache from a truncated large one.
        """
        m = os.path.join(root, "manifest.jsonl")
        if not os.path.exists(m):
            return None
        entries = [_json.loads(line) for line in open(m)]
        missing = [e["cache_path"] for e in entries
                   if not os.path.exists(os.path.join(root, e["cache_path"]))]
        return entries, missing, sorted({(e.get("dataset"), e.get("fmt")) for e in entries})

    # Re-stage if the local copy is absent OR INCOMPLETE. Guarding only on "does the
    # directory exist" means a partial staging is never repaired: the next run sees the
    # directory, skips the copy, and re-validates the same broken files forever.
    _have = _cache_shortfall(LOCAL_CACHE) if os.path.isdir(LOCAL_CACHE) else None
    if _have is not None and _have[1]:
        print(f"local cache is incomplete ({len(_have[1])} of {len(_have[0])} files "
              "absent) — discarding and re-staging")
        !rm -rf {LOCAL_CACHE}
        _have = None

    if _have is None:
        !rm -rf {LOCAL_CACHE} /content/_cache
        _t0 = _t.time()
        # Prefer ONE zip over a folder copy: 2.5 GB as a single FUSE read beats ~1,500
        # small-file reads, which Drive services at only a handful per second.
        _cz = next((z for z in (f"{_MD}/{CACHE_ZIP}", f"{DRIVE}/{CACHE_ZIP}",
                                f"{_MD}/oc-ccl-data/{CACHE_ZIP}")
                    if os.path.exists(z)), None)
        if _cz:
            # `!unzip` does NOT raise on a bad archive — it extracts what it can and
            # returns non-zero, which the shell magic discards. Test first, and use
            # subprocess so a failure actually stops the cell.
            print(f"verifying {os.path.basename(_cz)} "
                  f"({os.path.getsize(_cz) / 1e9:.2f} GB)...")
            _test = _sub.run(["unzip", "-tqq", _cz], capture_output=True, text=True)
            assert _test.returncode == 0, (
                f"{CACHE_ZIP} is not a complete archive — Drive may still be syncing the "
                f"upload. Wait for it to finish, then re-run.\n"
                f"{_test.stdout[-800:]}{_test.stderr[-800:]}"
            )
            os.makedirs("/content/_cache", exist_ok=True)
            _sub.run(["unzip", "-q", "-o", _cz, "-d", "/content/_cache"], check=True)
            _found = [r for r, _, f in os.walk("/content/_cache") if "manifest.jsonl" in f]
            assert _found, "no manifest.jsonl inside " + CACHE_ZIP
            os.rename(_found[0], LOCAL_CACHE)
        else:
            # The folder fallback is the DANGEROUS path: Drive's FUSE layer silently
            # drops files during large recursive copies (why this repo ships as a zip),
            # and an old cache folder may hold the superseded JPEG images, so a
            # "successful" copy can quietly feed the model the wrong pixels.
            print(f"WARNING: {CACHE_ZIP} not found on Drive. Copying {CACHE_ROOT} "
                  "file-by-file — check the reported formats below before trusting any "
                  "number from this run.")
            !mkdir -p {LOCAL_CACHE} && cp -r "{CACHE_ROOT}"/* {LOCAL_CACHE}/
        print(f"staged cache locally in {_t.time() - _t0:.0f}s")
        _have = _cache_shortfall(LOCAL_CACHE)

    # Delta goes on top of whichever base landed above — zip OR folder — and runs on
    # EVERY execution of this cell, not just a fresh stage. Both halves matter:
    #
    #   * folder path too, or the fallback stage is silently short of the new arm's images;
    #   * every execution, or a session that staged the base BEFORE the delta was uploaded
    #     can never repair itself — the base is complete by its own manifest, so the
    #     re-stage guard above is satisfied and re-running the cell changes nothing. That
    #     is exactly the "a partial staging is never repaired" bug, one level up.
    #
    # Idempotent by cache_path, so repeated runs merge only what is genuinely new. The
    # manifest merge is the load-bearing half: `ImageCache.has` consults the manifest, so
    # copying files without appending their entries puts the images on disk and still
    # reports every one of them as a cache miss.
    # Resolution accepts an absolute path OR a bare filename, probes the usual three
    # directories, then falls back to a shallow glob. Probing a fixed list and printing
    # only "not found" is how a 0.49 GB upload sitting one directory over, or under a
    # browser-renamed "... (1).zip", turns into 265 cache misses per worker with no
    # indication of why — so a miss here reports what IS on Drive.
    import glob as _glob

    _dcand = []
    if CACHE_DELTA_ZIP:
        _dcand = ([CACHE_DELTA_ZIP] if os.path.isabs(CACHE_DELTA_ZIP) else [])
        _dcand += [f"{_MD}/{CACHE_DELTA_ZIP}", f"{DRIVE}/{CACHE_DELTA_ZIP}",
                   f"{_MD}/oc-ccl-data/{CACHE_DELTA_ZIP}"]
        _dcand += sorted(_glob.glob(f"{_MD}/*/{os.path.basename(CACHE_DELTA_ZIP)}"))
    _dz = next((z for z in _dcand if os.path.exists(z)), None)
    if CACHE_DELTA_ZIP and not _dz:
        _near = sorted(set(_glob.glob(f"{_MD}/*.zip") + _glob.glob(f"{DRIVE}/*.zip")
                           + _glob.glob(f"{_MD}/*/*.zip")))
        print(f"{CACHE_DELTA_ZIP} NOT FOUND. Looked for:\n  " + "\n  ".join(_dcand)
              + ("\nzips actually visible on Drive:\n  "
                 + "\n  ".join(f"{z}  ({os.path.getsize(z) / 1e9:.2f} GB)" for z in _near)
                 if _near else "\nno .zip files visible on Drive at all — still syncing?")
              + "\n  -> set CACHE_DELTA_ZIP to one of these paths (an absolute path is "
                "accepted) and re-run this cell; the merge is idempotent.")
    if _dz and _have is not None:
        print(f"verifying delta {os.path.basename(_dz)} "
              f"({os.path.getsize(_dz) / 1e9:.2f} GB)...")
        _dt = _sub.run(["unzip", "-tqq", _dz], capture_output=True, text=True)
        assert _dt.returncode == 0, (
            f"{CACHE_DELTA_ZIP} is not a complete archive — Drive may still be syncing."
            f"\n{_dt.stdout[-800:]}{_dt.stderr[-800:]}")
        !rm -rf /content/_delta && mkdir -p /content/_delta
        _sub.run(["unzip", "-q", "-o", _dz, "-d", "/content/_delta"], check=True)
        _droot = next((r for r, _, f in os.walk("/content/_delta")
                       if "manifest.jsonl" in f), None)
        assert _droot, "no manifest.jsonl inside " + CACHE_DELTA_ZIP
        _sub.run(f'cp -r "{_droot}"/*/ "{LOCAL_CACHE}"/', shell=True, check=True)
        _mpath = os.path.join(LOCAL_CACHE, "manifest.jsonl")
        _seen = {_json.loads(l)["cache_path"] for l in open(_mpath)}
        _new = [l for l in open(os.path.join(_droot, "manifest.jsonl"))
                if _json.loads(l)["cache_path"] not in _seen]
        with open(_mpath, "a") as _mf:
            _mf.writelines(_new)
        print(f"merged delta: +{len(_new)} manifest entries "
              f"({len(_seen)} already present)")
        !rm -rf /content/_delta
        _have = _cache_shortfall(LOCAL_CACHE)

    assert _have is not None, f"no manifest.jsonl in {LOCAL_CACHE}"
    _entries, _absent, _fmts = _have
    print(f"local cache: {len(_entries)} manifest entries at {LOCAL_CACHE}, "
          f"{len(_absent)} files absent | formats {_fmts}")
    _lossy = [d for d, f in _fmts if f not in ("png", "webp")]
    if _lossy:
        print(f"  WARNING: {_lossy} stored lossily — this is the superseded cache. "
              "SST has no intermediate encode; rebuild with the lossless policy.")
    assert not _absent, (
        f"{len(_absent)} of {len(_entries)} cached files are missing after re-staging, "
        f"e.g. {_absent[:3]}. The source archive or folder is itself incomplete."
    )
    # run_eval preflights the exact per-arm requirement; this only catches a stale mount.

# %%
# ── GPU probe + dtype decision (skipped for s2, which is network-bound) ──────
DTYPE = None
if STAGE != "s2":
    import torch

    assert torch.cuda.is_available(), "no GPU — set Runtime > Change runtime type"
    cap = torch.cuda.get_device_capability(0)
    # Select on compute capability, NOT torch.cuda.is_bf16_supported(): since torch 2.6
    # that returns True on sm_75 (T4) via emulation, which is correct but glacial.
    DTYPE = torch.bfloat16 if cap[0] >= 8 else torch.float16
    print(f"{torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]} | dtype={DTYPE}")
    if cap[0] < 8:
        print("WARNING: no native bf16 on this GPU; running fp16. Prefer L4 or A100.")

# %%
# ── shared machinery, copied VERBATIM from run_stage.py ──────────────────────
# tests/test_s14_standalone.py pins every one of these to the run_stage.py
# original by AST equality — edit THERE, re-copy here, or the suite fails.
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


#: Per-worker VRAM, {fixed, cache} in GB. `cache` is one group's embedding cache, which
#: TTA doubles (it stores both orientations); `fixed` is weights plus activations.
#:
#: These were badly overstated until s5c: they came from *peak totals reported during
#: precompute*, where the image encoder is resident, and were then used as if they were
#: steady state. Measured, s5c ran 6 workers in 13.5 GB — **2.25 GB each against the 5.62
#: projected**, leaving two thirds of the card idle. Every run now records its own
#: `vram.peak_reserved_gb`, so these numbers can be replaced with measurements rather than
#: re-estimated; see `scripts/calibrate_workers.py`.
#:
#: Consequence of the correction: **vCPU is the binding constraint, not VRAM.** At ~2.3 GB
#: a worker, 40 GB would allow ~15, but Colab's A100 runtime has ~12 cores and each worker
#: needs about one to feed the GPU.
#: Defaults only — `vram_calibration.json` on Drive overrides these once a run has
#: measured itself. Every figure in this table has been wrong at least once, always
#: because it was taken from an unrepresentative moment:
#:   * 4.6 GB read off an OOM traceback, where surviving processes were already starved
#:     (true value 5.64) -> ceiling set too high -> another OOM;
#:   * 2.25 GB read from a run that resumed everything from cache and did ZERO inference
#:     (0.96 GB peak reserved) -> ceiling set far too low, then far too high once TTA
#:     actually ran -> another OOM.
#: Measured under real inference: butterfly 1-shot + TTA = 4.12-4.45 GB per worker.
#:
#: The `cache` column is now DERIVED (see `cache_gb`) and kept here only as the value the
#: old flat model used, because that model is what over-committed butterfly at 5-shot: it
#: treats the embedding cache as a constant when it is a function of how many distinct
#: images the worker's own run-slice touches.
#: `fixed` MUST NOT INCREASE AS SHOT COUNT FALLS. A 1-shot episode does strictly less work
#: than a 5-shot one — fewer prompt calls, fewer frames in the bank, the same object count —
#: so its peak activations cannot be larger. The table violated that in both datasets
#: (butterfly 3.45 at 1-shot against a measured 2.95 at 5-shot; beetle 2.36 against 2.06),
#: which cost workers on exactly the stages that are cheapest to run. Corrected by taking
#: the SMALLER FIGURE ALREADY IN THE TABLE rather than inventing a new one — this removes an
#: ordering violation, it does not add a measurement. `tests/test_fanout` now enforces it.
PEAK_GB = {
    ("butterfly_major", 1): {"fixed": 2.95, "cache": 0.50},   # was 3.45; see above
    # 4.07 -> 2.95, from TWO independent measurements that agree, which is the bar this
    # table has never previously cleared:
    #   * `profile.zip`: peak_reserved minus that run's own 1.10 GB cache = 2.95 GB.
    #   * the s8d butterfly plain pass: 6 workers observed at ~27 GiB (29.0 GB) holding
    #     11.54 GB of embeddings between them -> (29.0 - 11.54)/6 = 2.91 GB each.
    # One is a 12-episode single process, the other six processes under full load for 100
    # minutes; they land 1.4% apart. The old 4.07 was a peak captured DURING precompute and
    # then used as steady state, which is the same mistake that produced 4.6 and 2.25.
    ("butterfly_major", 5): {"fixed": 2.95, "cache": 0.79},
    ("beetle", 1):          {"fixed": 2.06, "cache": 0.32},   # was 2.36; see above
    ("beetle", 5):          {"fixed": 2.06, "cache": 0.47},
    # s16 (2-shot) rows, mirrored verbatim from run_stage.py — this dict is AST-pinned
    # to the multi-stage notebook's copy (tests/test_s14_standalone.py).
    ("butterfly_major", 2): {"fixed": 2.95, "cache": 0.57},
    ("beetle", 2):          {"fixed": 2.06, "cache": 0.36},
}


#: Bytes one image's cached `backbone_fpn` occupies, measured from the bundles rather than
#: estimated: S5 recorded 119 images at 1.248 GB (butterfly) and S4 79 at 0.828 GB (beetle)
#: — **10.49 MB either way**, because every image is encoded at 1024x1024 regardless of its
#: source resolution, so the three feature levels have a fixed shape.
EMBED_MB_PER_IMAGE = 10.49


#: Test images per group. Asserted by gate S1; here they set the cache floor, which is the
#: part of a worker's residency that run-splitting CANNOT reduce — every run scores every
#: test image, so all workers on a group hold all of it.
TEST_PER_GROUP = {"butterfly_major": 100, "beetle": 60}


def cache_gb(dataset, shot, runs_in_slice, n_groups=1, tta=False):
    """Upper bound on one worker's embedding cache, in GB.

    ``|test| + shot * runs_in_slice`` distinct images per group: the test set is fixed and
    duplicated across every worker on that group, while the supports are the only part a
    run-split actually divides. `evaluate` frees between groups, so a worker holding several
    groups peaks at one — except under `run_arms.py`, which passes `keep_embeddings=True`
    and frees at the arm loop instead, so the bound is still one group.

    The flat table said 0.79 GB for butterfly at 5-shot. A worker owning a whole group's 20
    runs holds 100 + 100 = 200 images = **2.10 GB**, and a half-slice 1.57 GB. Planning 8
    workers at 4.86 GB projects 38.9 of 42.4 GB and looks comfortable; the real figure is
    4.07 + 2.10 = 6.17, i.e. **49.4 GB on a 42.4 GB card**. With
    `expandable_segments:True` that does not fail cleanly — the allocator keeps retrying
    near the ceiling, which is the "VRAM pegged, nothing progressing, no traceback" the
    butterfly s8d pass sat in for five hours.
    """
    per_group = TEST_PER_GROUP.get(dataset, 100) + shot * max(1, int(runs_in_slice))
    gb = per_group * EMBED_MB_PER_IMAGE / 1000.0
    return gb * (2 if tta else 1)      # TTA caches both orientations


CALIBRATION = "vram_calibration.json"


def _cal_key(dataset, shot, tta):
    """Calibration key, with the schema in it.

    The `|fixed` suffix is a version marker, not decoration. Entries written before this
    change hold peak = fixed + cache under the key `<ds>|<shot>|<tta>`; read as fixed they
    would overstate it. They happen to overstate CONSERVATIVELY, which is safe, but "safe by
    luck" is how the last two VRAM mistakes survived review. A changed key means old entries
    are simply not found and the table is used until a run re-measures under the new
    meaning.
    """
    return f"{dataset}|{shot}|{int(bool(tta))}|fixed"


def _calibration_path():
    return os.path.join(RESULTS_ROOT, CALIBRATION)


def load_calibration():
    """Measured per-worker peaks from previous runs, keyed 'dataset|shot|tta'."""
    import json as _js

    try:
        with open(_calibration_path()) as fh:
            return _js.load(fh)
    except Exception:  # noqa: BLE001
        return {}


#: What TTA multiplies the WHOLE per-worker figure by, not just the embedding cache.
#:
#: The old model added one extra cache term (`fixed + 2*cache`), reasoning that TTA only
#: stores a second orientation. Measured on beetle 5-shot that predicts 3.00 GB and the
#: real figure is 3.67 — TTA also runs a second prediction per episode, so the transient
#: activations rise too. 11 workers were planned at 2.68 GB, 9 fit at 3.67, and workers 9
#: and 10 died: exactly the two the arithmetic did not leave room for.
#:
#: 1.5 because the ratio has to be taken against whatever base the planner will actually
#: use. Against the calibrated plain figure it is 3.67/2.68 = 1.37; against this table's
#: 2.53 it is 1.45. At 1.4 the planner allows 10 workers, and 10 x 3.67 = 36.7 GB exceeds
#: the 36.3 GB headroom — fitting only because the hard limit is 39.5. 1.5 yields 9 either
#: way, which is exactly the number that survived.
#:
#: Used only until a real TTA run calibrates the pair — measured always wins — and the
#: first TTA run on a new (dataset, shot) is precisely when nothing is measured yet.
TTA_VRAM_FACTOR = 1.5


def per_worker_gb(dataset, shot, tta=False, runs_in_slice=None, n_runs=20):
    """Per-worker VRAM: measured fixed cost plus a *derived* embedding cache.

    A calibrated number is still preferred for the FIXED part, but it can no longer stand
    in for the whole figure: what a previous run measured was that run's slice width, and
    a wider slice holds a bigger cache. Recording 4.9 GB from a 2-run slice and then
    planning a 20-run slice with it is how butterfly was over-committed.

    ``runs_in_slice=None`` means "a whole group" — the worst case, and the right default
    for a caller that has not partitioned yet.
    """
    p = PEAK_GB.get((dataset, shot), {"fixed": 4.07, "cache": 0.79})
    fixed = p["fixed"]
    if tta:
        fixed *= TTA_VRAM_FACTOR
    # `vram_calibration.json` now holds the FIXED cost alone — peak minus that worker's own
    # embedding cache — so it is directly comparable to this table and is not re-scaled for
    # TTA (the recorded value already is a TTA measurement when tta is set). `max`, because
    # under-sizing costs an OOM and over-sizing costs one worker.
    measured = load_calibration().get(_cal_key(dataset, shot, tta))
    if measured:
        fixed = max(fixed, float(measured))
    slice_runs = n_runs if runs_in_slice is None else runs_in_slice
    return fixed + cache_gb(dataset, shot, slice_runs, tta=tta)


def run_slices(n_runs, n_workers):
    """Split ONE group's run range into `n_workers` near-equal slices."""
    edges = [round(i * n_runs / n_workers) for i in range(n_workers + 1)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def plan_group_workers(dataset, shot, tta=False, n_runs=20, headroom=None, requested=None):
    """How many workers to put on ONE group, splitting its runs — the BALANCED layout.

    `_assign` partitions across groups first and only splits runs to reach the requested
    count, which cannot balance: 5 groups over 6 workers gives `[10, 10, 20, 20, 20, 20]`,
    so two workers own half as much as the other four and go idle early. Measured on the
    s8d butterfly plain pass, those two sat idle from minute 26 to the end — **133 idle
    worker-minutes of 551 provisioned, 24% of the card**. The same partition at 3 workers
    for TTA is worse: groups split 2/1/2, so one worker owns half of what its neighbours do.

    Doing groups one at a time and splitting only *runs* is balanced by construction — every
    group has 20 runs and every slice is the same width — and it needs no change to the
    worker CLI, which already takes one `--groups` and one run range.

    Two things pay for the extra pass structure, and neither is significant:

    * every worker on a group re-encodes that group's 100 test images, so total encoder work
      rises with the worker count. Measured, a group's precompute is ~8 s (105 images), and
      the workers do it concurrently — seconds per group against hours per pass.
    * one launch/teardown per group rather than per pass. `STAGGER_S` between launches costs
      `n_workers * STAGGER_S` per group.

    Narrower slices also draw fewer distinct supports, so each worker's cache SHRINKS — which
    is what buys the extra workers back. Butterfly 5-shot + TTA: `_assign` fits 3, this fits
    4, and the critical path drops from 80 run-units to 50.

    Returns (n_workers, per_worker_gb, capacity_gb, note).
    """
    import torch

    headroom = VRAM_HEADROOM if headroom is None else headroom
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9
    budget = cap * headroom
    by_cpu = max(1, (os.cpu_count() or 4) - 1)

    def price(n):
        pers = [per_worker_gb(dataset, shot, tta, runs_in_slice=(b - a), n_runs=n_runs)
                for a, b in run_slices(n_runs, n)]
        return max(pers), sum(pers)

    # Never more workers than runs — an empty slice is a worker that loads 898 MB to do
    # nothing, and `_assign`'s equivalent bug (returning fewer slices than requested) is
    # what put 56 GB onto a 39.5 GB card.
    fits = [n for n in range(1, min(by_cpu, n_runs) + 1) if price(n)[1] <= budget]
    n = max(fits) if fits else 1
    forced = ""
    if requested:
        # An explicit override still respects the vCPU ceiling and the run count — those are
        # hard, unlike the VRAM table, which is an estimate that has been wrong in both
        # directions. Overriding UPWARD is the caller taking responsibility for a measurement
        # the table does not have; an OOM is recoverable because every finished run is banked.
        n = max(1, min(int(requested), by_cpu, n_runs))
        forced = f" [forced from {requested}]" if n != int(requested) else " [forced]"
    per, total = price(n)
    note = (f"{n} workers per group{forced} ({per:.2f} GB projected each -> {total:.1f} of "
            f"{cap:.1f} GB, {100 * total / cap:.0f}%); "
            f"slices {[b - a for a, b in run_slices(n_runs, n)]}")
    return n, per, cap, note


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


def wait_all(procs, label="worker", heartbeat=180):
    """Wait on (tag, logfile, Popen) triples, and KILL SURVIVORS on interrupt.

    Polls rather than blocking on `wait()`, so it can report progress: every `heartbeat`
    seconds it prints how many workers are still up and how many runs have been banked
    since launch. Without that, a long pass produces no output at all until the first
    worker exits.

    `Popen` children do not die with a KeyboardInterrupt in the parent — the notebook cell
    stops, the workers keep running, and they keep holding their GPU memory. That is not a
    tidiness issue: the next launch then plans against a card that already has ~35 GB of
    orphans on it, its workers OOM or hang, and the failure looks like a fresh bug rather
    than the previous run still being alive. Measured once: five butterfly workers survived
    an interrupt for hours and silently blocked the retry.

    Returns the tags that exited non-zero.
    """
    bad = []
    t0, start_banked = _t.time(), banked_runs()
    last = t0
    try:
        pending = list(procs)
        while pending:
            still = []
            for tag, log, proc in pending:
                rc = proc.poll()
                if rc is None:
                    still.append((tag, log, proc))
                    continue
                log.close()
                print(f"  {label} {tag} exit {rc}", flush=True)
                if rc != 0:
                    bad.append(tag)
                    print(open(log.name).read()[-2500:])
            pending = still
            if pending:
                if heartbeat and _t.time() - last >= heartbeat:
                    done = banked_runs() - start_banked
                    mins = (_t.time() - t0) / 60
                    print(f"  ... {len(pending)}/{len(procs)} {label}(s) running, "
                          f"{done} runs banked in {mins:.0f} min", flush=True)
                    last = _t.time()
                _t.sleep(5)
    except BaseException:                       # KeyboardInterrupt included, deliberately
        alive = [t for t, _, pr in procs if pr.poll() is None]
        print(f"\ninterrupted — killing {len(alive)} surviving {label}(s): {alive}",
              flush=True)
        for _tag, log, pr in procs:
            if pr.poll() is None:
                pr.kill()
            try:
                log.close()
            except Exception:  # noqa: BLE001
                pass
        for _tag, _log, pr in procs:
            try:
                pr.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
        raise
    return bad

# %%
# ── SAM 3: checkout + gated checkpoint ───────────────────────────────────────
import shutil

# Fail on missing credentials BEFORE any cloning/installing: the checkpoint is
# HF-gated with manual approval, and nothing in this stage can run without it.
# PRINT the guidance, then raise a plain, unchained exception: `raise
# SystemExit(...) from e` in a Colab cell trips an IPython ultratb bug
# ("'tuple' object has no attribute 'f_lineno'") that buries the message under
# an Internal-Python-error wall — measured on this notebook's first run.
def _stop(msg):
    print("\n" + "=" * 72 + f"\nSTOP: {msg}\n" + "=" * 72, flush=True)
    raise RuntimeError("see the STOP message above") from None


_CKPT = f"{DRIVE}/{S14_SAM3_CKPT_DRIVE}"
_tok = None
if not os.path.exists(_CKPT):
    try:
        from google.colab import userdata

        _tok = userdata.get("HF_TOKEN")
    except Exception:  # noqa: BLE001 — any miss means the same fix
        _stop("HF_TOKEN Colab secret missing (or this notebook was not granted "
              "access to it). sam3.pt is HF-GATED:\n"
              "  1. request access at huggingface.co/facebook/sam3 (manual "
              "approval — can take days);\n"
              "  2. create a token at huggingface.co/settings/tokens;\n"
              "  3. Colab: key icon in the left sidebar -> add secret HF_TOKEN "
              "-> toggle notebook access ON;\n"
              "  4. re-run this cell.")

if not os.path.exists("/content/sam3"):
    !git clone -q --depth 1 https://github.com/facebookresearch/sam3 /content/sam3
os.environ["OCCL_SAM3_ROOT"] = "/content/sam3"

if not os.path.exists(_CKPT):
    from huggingface_hub import hf_hub_download

    try:
        _tmp = hf_hub_download("facebook/sam3", "sam3.pt", token=_tok)
    except Exception as _e:  # noqa: BLE001 — surface the real fix
        _stop(f"sam3.pt download failed ({type(_e).__name__}: {_e}).\n"
              "If this is a 403/gated error, the facebook/sam3 access request "
              "on this HF account is not approved yet — approval is manual and "
              "can take days. Nothing in s14sam3 can run without the "
              "checkpoint.")
    os.makedirs(os.path.dirname(_CKPT), exist_ok=True)
    # copy-then-rename so the existence guard can never pass on a half-copied
    # checkpoint (the sec snapshot lost a 5 GB shard exactly this way)
    shutil.copy(_tmp, _CKPT + ".part")
    os.replace(_CKPT + ".part", _CKPT)
    print(f"  cached sam3.pt to Drive ({_CKPT})", flush=True)
print("sam3 checkout + checkpoint ready:", os.path.getsize(_CKPT) / 1e9, "GB")

# %%
# ── commissioning: the gate-shaped asserts, in seconds ───────────────────────
# Strict 0/0 weight surgery, geometry pins 1008/1152/7, the global bf16 autocast
# exited, and cache bit-identity (cold vs precomputed) ON THIS GPU — the exact
# predicate that caught sec's autocast-context divergence, which no CPU run can
# see. Runs BEFORE anything spends GPU-hours.
import time as _t

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
        "sam2_base_tta": os.path.join(RESULTS_ROOT,
                                      f"s11_{_ds_name}_base_tta.zip"),
        "sam2_pertrait_tta": os.path.join(
            RESULTS_ROOT, f"s11_{_ds_name}_occl_pertrait_tta.zip"),
    }
    for _p in _sam2[_ds_name].values():
        assert os.path.exists(_p), (
            f"{_p} not on Drive — S14 pairs against S11's banked SAM 2 cells "
            "and never recomputes them")
print("banked SAM 2 ledger present for", list(_sam2))

# %%
# ── the stage: two fine-tunes, two fan-outs, four merges, one combine — per dataset ──
import json as _json
import subprocess

BUNDLE = os.path.join(REPO, "s14sam3.zip")
_ft_dir = "/content/s14sam3_adapters"
os.makedirs(_ft_dir, exist_ok=True)

# the two LoRA scope sets, verbatim from occl.finetune.lora — the reuse check
# pins a cached checkpoint to the arm it claims to be
_SCOPES_STACK = ("maskmem_backbone", "sam_mask_decoder")
_SCOPES_ARCH = ("maskmem_backbone", "transformer", "sam_mask_decoder")

_per_ds = {}
for _ds_name in S14_DATASETS:
    _sel = S14_SEL[_ds_name]
    print(f"\n=== s14sam3 {_ds_name}: crowned stack + arch arm, 1-shot, {_sel} "
          f"===\n", flush=True)

    # ── fine-tunes: the ported stack, then the architecture-adapted arm ──
    def _ft(_preset, _scopes_expect, _suffix):
        _pt = (f"{_ft_dir}/{_ds_name}_sam3_lora_lr{S14_LR:g}_ep{S14_EPOCHS}"
               f"_per_trait_excl-zerocat"
               f"_noaug_derange_logit_reparam_eval_reset{_suffix}.pt")
        _tb = os.path.join(REPO, f"s14ft_sam3_{_ds_name}{_suffix}.zip")
        if S14_REUSE and not os.path.exists(_pt):
            _dpt = os.path.join(RESULTS_ROOT, os.path.basename(_pt))
            if os.path.exists(_dpt):
                shutil.copy(_dpt, _pt)
        _reuse = S14_REUSE and os.path.exists(_pt)
        if _reuse:
            import torch as _torch
            _m = (_torch.load(_pt, map_location="cpu",
                              weights_only=False).get("meta") or {})
            _o = _m.get("occl") or {}
            # scopes live NESTED under meta["lora"] (save_lora packs
            # record["finetune"] wholesale) — reading the top level here would
            # never match and silently retrain every session
            _got_scopes = tuple((_m.get("lora") or {}).get("scopes") or ())
            if (_m.get("family") != "sam3"
                    or _got_scopes != _scopes_expect
                    or _o.get("mask_mode") != "per_trait"
                    or (_o.get("excl") or {}).get("mode") != "zerocat"
                    or any(_o.get(k) is not None
                           for k in ("xovl", "bshape", "margin", "scnp"))
                    or _o.get("augmentation") is not None):
                print(f"  {os.path.basename(_pt)} config mismatch — retraining",
                      flush=True)
                _reuse = False
        if not _reuse:
            with _timed(f"ft{_suffix or '_stack'}_{_ds_name}"):
                run(["python", "scripts/run_finetune.py",
                     "--family", "sam3", "--scope-preset", _preset,
                     "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                     "--checkpoint", _CKPT, "--device", "cuda",
                     "--method", "lora",
                     "--lr", repr(S14_LR), "--epochs", str(S14_EPOCHS),
                     "--mask-mode", "per_trait", "--excl-mode", "zerocat",
                     "--lora-out", _pt, "--out", _tb, "--no-download"])
            if os.path.exists(_tb):
                shutil.copy(_tb, os.path.join(RESULTS_ROOT,
                                              os.path.basename(_tb)))
        assert os.path.exists(_pt), f"{_pt} was not written"
        shutil.copy(_pt, os.path.join(RESULTS_ROOT, os.path.basename(_pt)))
        return _pt

    _pt = _ft("faithful", _SCOPES_STACK, "")
    _apt = _ft("sam3_arch", _SCOPES_ARCH, "_arch")

    # ── eval fan-outs: base(plain); then ONE TTA fan carrying all three arms ──
    _grp = GROUPS_OF[_ds_name]

    def _fan(_tta, _armspec, _log_tag):
        # Width from the S14 table — `requested` deliberately OVERRIDES the
        # sam2-calibrated per-worker fit, which cannot price this family's
        # residency; vCPU and run-count ceilings still apply inside the planner.
        _w, _per, _cap, _note = plan_group_workers(
            _ds_name, 1, tta=_tta, n_runs=20,
            requested=S14_EVAL_WORKERS["tta" if _tta else "plain"])
        print(f"\n  === {_ds_name} sam3 {'TTA' if _tta else 'plain'} "
              f"[{_log_tag}]: {_note} ===", flush=True)
        for _gi, _group in enumerate(_grp):
            _procs = []
            for _i, (_a, _b) in enumerate(run_slices(20, _w)):
                _log = open(f"/content/s14sam3_{_ds_name}_{_log_tag}"
                            f"_g{_gi}_{_i}.log", "w")
                _wcmd = (["python", "scripts/run_arms.py",
                          "--adapter", "sam3",
                          "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                          "--checkpoint", _CKPT,
                          "--shot", "1", "--n-runs", "20",
                          "--support-selection", _sel,
                          "--resume-dir", f"{RESULTS_ROOT}/runs",
                          "--groups", _group,
                          "--run-start", str(_a), "--run-end", str(_b)]
                         + _armspec + (["--tta"] if _tta else []))
                print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}",
                      flush=True)
                _procs.append((_i, _log, subprocess.Popen(
                    _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
                    env=worker_env())))
                _t.sleep(STAGGER_S)
            _bad = wait_all(_procs, f"s14sam3 {_ds_name} {_group} worker")
            assert not _bad, (f"{_ds_name}/{_group} workers {_bad} failed — "
                              f"see /content/s14sam3_{_ds_name}_"
                              f"{_log_tag}_g{_gi}_*.log")

    with _timed(f"fan_plain_{_ds_name}"):
        _fan(False, ["--arms", "base="], "plain")
    with _timed(f"fan_tta_{_ds_name}"):
        _fan(True, ["--arms", "base=", "--arms", f"stack={_pt}",
                    "--arms", f"arch={_apt}"], "tta")

    # ── merges (inference-free) + the seven-pairing combine ──────────────
    def _merge(_cell, _tta, _lora=None):
        _out = os.path.join(REPO, f"s14sam3_{_ds_name}_{_cell}.zip")
        run(["python", "scripts/run_eval.py",
             "--stage", f"s14sam3_{_ds_name}_{_cell}",
             "--adapter", "sam3", "--dataset", _ds_name,
             "--cache-root", LOCAL_CACHE, "--checkpoint", _CKPT,
             "--shot", "1", "--n-runs", "20",
             "--support-selection", _sel,
             "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
             "--out", _out, "--no-download"]
            + (["--tta"] if _tta else [])
            + (["--lora", _lora, "--lora-epochs", str(S14_EPOCHS)]
               if _lora else []))
        shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
        return _out

    with _timed(f"merge_combine_{_ds_name}"):
        _cells = {"base": _merge("base", False),
                  "base_tta": _merge("base_tta", True),
                  "stack_tta": _merge("stack_tta", True, _lora=_pt),
                  "arch_tta": _merge("arch_tta", True, _lora=_apt)}
        _dsb = os.path.join(REPO, f"s14sam3_{_ds_name}.zip")
        run(["python", "scripts/combine_s14.py", "--family", "sam3",
             "--out", _dsb,
             *[f"{k}={v}" for k, v in _cells.items()],
             *[f"{k}={v}" for k, v in _sam2[_ds_name].items()]])
        shutil.copy(_dsb, os.path.join(RESULTS_ROOT, os.path.basename(_dsb)))
    _per_ds[_ds_name] = _dsb

for _ds_name in S14_DATASETS:
    if _ds_name not in _per_ds:
        _cand = os.path.join(RESULTS_ROOT, f"s14sam3_{_ds_name}.zip")
        if os.path.exists(_cand):
            print(f"  pulling {_ds_name} from an earlier session: {_cand}",
                  flush=True)
            _per_ds[_ds_name] = _cand
run(["python", "scripts/combine_s14.py", "--family", "sam3", "--merge",
     "--out", BUNDLE, *[f"{k}={v}" for k, v in _per_ds.items()]])

# The measured wall-clock ledger — what next session's widths get tuned from.
print(f"\n=== s14sam3 timing summary "
      f"(total {(_t.time() - _stage_t0) / 60:.1f} min) ===")
for _k, _v in _ph.items():
    print(f"  {_k:>28s}: {_v / 60:6.1f} min")

# %%
# ── download the bundle ──────────────────────────────────────────────────────
# Copy to Drive FIRST — the browser download is the fragile step. Then locally:
# drop s14sam3.zip into results/inbox/ and run `python scripts/check_gate.py
# s14sam3` (it gates both datasets from this one merged bundle).
assert os.path.exists(BUNDLE), (
    f"{BUNDLE} was not written — the stage cell did not complete; scroll up "
    "for the worker logs (/content/s14sam3_*.log)")
shutil.copy(BUNDLE, os.path.join(RESULTS_ROOT, os.path.basename(BUNDLE)))
print(f"banked to Drive: {RESULTS_ROOT}/{os.path.basename(BUNDLE)}")
from google.colab import files

files.download(BUNDLE)
