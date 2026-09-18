# %% [markdown]
# # SST / SAM-2 reproduction — staged runner
#
# One notebook for every Colab stage. Set `STAGE` in the next cell, run top to bottom,
# download the bundle it produces, drop it into `results/inbox/` on your machine, and run:
#
# ```
# python scripts/check_gate.py <stage>
# ```
#
# **A stage is not done until its gate prints PASS. Do not start the next stage before then.**
#
# | stage | GPU | what it does |
# |---|---|---|
# | `s2` | none | build the image cache (beetle: seconds; butterfly: hours, resumable) |
# | `s3` | A100 | adapter smoke — run ONCE PER DATASET (`S3_DATASET`). Config asserts, |
# |      |      | embedding-cache exactness, and an overlay you must eyeball. Butterfly |
# |      |      | had never been checked, and it is the dataset that disagrees. |
# | `s4a` | A100 | beetle 1-shot × 20, uniform supports (ran: 55.96 ± 7.90) |
# | `s4b` | A100 | beetle 1-shot × 20, canonical supports (ran: 58.54 ± 5.77) |
# | `s4c` | A100 | beetle 1-shot × 20, **species-balanced** canonical → 61.9 ± 3.7 |
# | `s5` | A100 | butterfly Major 1-shot × 20 runs → 81.0 ± 1.0 (ran: 68.54 ± 3.15) |
# | `s5diag` | A100 | ~2 min. Why S5 missed: scores identical logits under every overlap rule |
# | `s5b` | A100 | Major 1-shot × 20, **placement-canonical** supports (ran: 70.44 ± 1.25) |
# | `s5c` | A100 | Major 1-shot × 20 + **orientation TTA** — predicts with the support as |
# |      |      | given AND rotated 180°, keeps the more confident. Targets the 37% of |
# |      |      | cyrbia queries that are upside down and score ~0. `modified` track. |
# | `s6` | A100 | 5-shot, one dataset per session (`DATASET_5SHOT`). ran: beetle 73.16 ± |
# |      |      | 2.31, butterfly 71.78 ± 0.15. Combine with scripts/combine_s6.py |
# | `s6b` | A100 | 5-shot with **orientation-diverse** supports — the 5-set spans upright |
# |      |      | and rotated, drawn from real train specimens (cyrbia train is 34% |
# |      |      | rotated). Tests whether the memory bank selects or blends. `modified`. |
# |      |      | On butterfly this needs CACHE_DELTA_ZIP: it draws 265 rot180 supports |
# |      |      | the older arms never picked, so the main cache does not hold them. |
# | `s8b` | A100 | OC-CCL fine-tune: ONE epoch at ONE learning rate (`FT_LR`). Run five |
# |      |      | times to sweep 3e-6 / 1e-5 / 3e-5 / 1e-4 / 3e-4 — each writes its own |
# |      |      | bundle. Serial by construction; WORKERS is ignored. VRAM is far above |
# |      |      | eval's ~4.86 GB — the cycle keeps 4 tracker passes' activations. |
# | `s8c_tune` | A100 | choose the EPOCH count by k-fold CV inside the TRAIN split. The |
# |      |      | test images are unreachable by construction, and the criterion is |
# |      |      | val mIoU, not the cycle loss. Sweeps lr x epochs jointly; folds run |
# |      |      | concurrently. |
# | `s8c` | A100 | **train + score in one shot.** Fine-tunes one arm per `S8C_LRS`, then |
# |      |      | evaluates base + each arm over the S4c protocol (1-shot x 20, |
# |      |      | canonical_balanced) and pairs them with registry.compare. The adapters |
# |      |      | never touch Drive, so nothing can be stale. ~45 s/epoch training, then |
# |      |      | ~6 min/arm of fanned-out inference. |
# | `s6c` | A100 | completes the rotation 2x2 on Major at 5-shot: {placement, |
# |      |      | orientation-diverse} x {no TTA, TTA}. Answers whether the two |
# |      |      | rotation fixes STACK or are interchangeable. Two cells, TTA doubles |
# |      |      | inference, so this is the most expensive stage in the project. |
# | `s8d` | A100 | base vs OC-CCL at **5-shot, both datasets** — the coherent baseline |
# |      |      | that must exist before the loss is modified. One arm (1e-5, 1 epoch), |
# |      |      | chosen by s8c rather than swept again. |
# | `s8` | A100 | OC-CCL paired evaluation (A100: FA2 backward at head_dim=256) |
# | `s9` | A100 | loss modification experiments
# | `s8_5a` | A100 | OC-CCL with SST's OWN method: **full fine-tune, no LoRA** (freeze
# |      |      | the image encoder, train the rest), at its shipped `lr 1e-5 / 10 epochs`.
# |      |      | 1-shot, plain, both datasets. Every S8 number used adapters that appear
# |      |      | nowhere in SST, so the paper's +3.3 was never actually tested.
# | `s8_5a2` | A100 | the REMAINING defect axes, **Beetle 1-shot only**: A5 `model.train()`,
# |      |      | A1 `/255` normalisation, both-plus-A2 as SST's literal config, and an
# |      |      | exploratory A7 no-reset arm. Five arms share one inference pass. See
# |      |      | `S8_5A2_ARMS`.
# | `s8_5b` | A100 | as s8_5a, 1-shot **+ TTA**.
# | `s8_5c` | A100 | as s8_5a, **5-shot + TTA** — the stack S8d measured with LoRA.
# | `s10` | A100 | the PAPER's OC-CCL (A2.2), which the release contradicts on five axes:
# |      |      | per-trait cycle, LoRA r32/α64, ONE epoch, one fixed labeled x₀, beetle
# |      |      | rotation aug. Three fixed-x₀ composites + three single-axis reverts +
# |      |      | a 20-adapter per-run anchored arm, all paired vs base. See `S10_ARMS`.
# | `s11` | A100 | OC-CCL method SELECTION, both datasets, 1-shot: {base, merged,
# |      |      | per-trait, merged+SCNP} × {plain, TTA} on each dataset's best
# |      |      | protocol, paired per seed. Which method helps the current stack, each
# |      |      | method's TTA interaction (the S12 carrier question), and beetle's
# |      |      | first-ever 1-shot TTA base. Splittable per dataset (`S11_DATASETS`). |
# | `s12` | A100 | SCNP as a TOGGLE on the per-trait cycle, both datasets, 1-shot:
# |      |      | trains and evaluates ONLY `occl_ptscnp` (per-trait × SCNP w=3) and
# |      |      | pairs it against S11's banked per-trait bundles (pulled from Drive —
# |      |      | the toggle-off arm is never recomputed). Headline = the paired
# |      |      | best-of-3 delta. ~1.5-2 h: evaluation is one arm, not a slate. |
# | `s13` | A100 | FIVE loss arms on the per-trait carrier {excl, xovl, bshape,
# |      |      | margin, ptmargin}, both datasets, 1-shot, TTA-ONLY cells; base
# |      |      | cells = S11's banked bundles (pertrait_tta for every arm; the
# |      |      | margin isolation also vs base_tta). Headline = paired best-of-3.
# |      |      | ~2 sessions — butterfly training dominates; fully resumable. |
# | `s14sam3` | A100 | MOVED to its own standalone notebook, run_s14sam3.ipynb
# |      |      | (user decision after s14sec: conflicting dependency stacks get a
# |      |      | dedicated pipeline; it also carries the architecture-adapted
# |      |      | `arch` arm — LoRA additionally on sam3's memory attention). |
# | `s14sec` | A100 | the same on SeC-4B: {off, forced, off_tta, stack_tta} — the
# |      |      | concept module FORCED on (its native gate is blind here; fire
# |      |      | stats recorded on every off episode). Ungated, 15.18 GB direct
# |      |      | HF download. forced-butterfly ~2-2.6 h is the whale. ~1-2 sessions. |
# | `s16` | A100 | **TTA as an implicit 2nd shot** — 4 runs on base SAM 2: {butterfly,
# |      |      | beetle} × {1-shot + TTA, 2-shot plain}, canonical protocols, 20
# |      |      | seeds. One paired row per dataset — 2-shot plain vs 1-shot TTA,
# |      |      | via the prefix-nested draws — asking whether TTA's implicit second
# |      |      | reference beats an explicit second support in the memory bank. The
# |      |      | TTA cell is its own S11-pinned anchor. ~1.5 h (`S16_DATASETS`). |
# | `profile` | A100 | **measure, before committing to anything long.** Times one episode's
# |      |      | frame prep, `_get_image_feature` calls and host->device bytes, prompting
# |      |      | and propagation, then A/Bs the two `init_state` placement flags and
# |      |      | asserts the logits are bit-identical across them. ~2 minutes, 12
# |      |      | episodes, one group, no bundle. Set `PROFILE_DATASET` / `PROFILE_SHOT`. |

# %%
# ── configuration ────────────────────────────────────────────────────────────
# The shipped default is the stage of the CURRENT campaign, not the oldest one:
# every re-upload reverts in-Colab edits, so a "Run all" on a fresh copy must land
# in the stage actually being run — a stale "s2" default sent one session into the
# curation branch, which died on a constant that only exists on curation setups.
STAGE = "s16"

#: Every value STAGE may take. Checked immediately, because an unrecognised one is SILENT:
#: the dispatch is a chain of `if STAGE == "..."`, so a typo matches nothing, runs nothing,
#: and surfaces ~400 lines later as "<typo>.zip was not written — the stage cell did not
#: complete", which is exactly wrong. The cell completed; it had nothing to do. A typo of
#: `"stage"` (the notebook is called run_stage) cost one session to that message.
#:
#: `tests/test_fanout` derives the dispatchable set from the AST and asserts it equals this
#: tuple, so a stage added to the dispatch and forgotten here fails locally rather than in
#: Colab.
STAGES = ("s2", "s3", "s4a", "s4b", "s4c", "s5", "s5b", "s5c", "s5diag",
          "s6", "s6b", "s6c", "s8b", "s8c", "s8c_tune", "s8d",
          "s8_5a", "s8_5a2", "s8_5b", "s8_5c", "s10", "s11", "s12", "s13",
          "s14sec", "s16", "profile")

if STAGE not in STAGES:
    import difflib as _dl

    # cutoff 0.6: at 0.4 the typo "stage" suggested s5diag, s4a and s8c_tune, which is
    # worse than no suggestion — it invites trying one. Real typos ("s8dd", "profiel",
    # "s8ctune") still resolve to exactly one candidate.
    _near = _dl.get_close_matches(str(STAGE), STAGES, n=3, cutoff=0.6)
    raise SystemExit(
        f"STAGE = {STAGE!r} is not a stage.\n"
        + (f"  did you mean: {', '.join(_near)}?\n" if _near else "")
        + f"  valid: {', '.join(STAGES)}\n"
        "  See the table at the top of this notebook for what each one does."
    )

DRIVE = "/content/drive/MyDrive/occl"
CACHE_ROOT = f"{DRIVE}/cache"          # derived 1024² images (~290 MB)
ORIGINALS_ROOT = f"{DRIVE}/originals"  # untouched source bytes (~11.9 GB); "" to discard
RESULTS_ROOT = f"{DRIVE}/results"      # per-run JSON, so a session boundary is survivable

# How this repo reaches Colab.
#   "drive_zip" — read ONE zip from Drive and extract it locally. Best default:
#                 a single file is one FUSE read, versus dozens for a folder, and there
#                 is no upload prompt each session. Rebuild it after ANY code change
#                 with `python scripts/build_repo_zip.py` (which also regenerates the
#                 ipynb, so the pair cannot skew) and replace it on Drive. The
#                 extraction cell asserts the zip knows the stage being run.
#   "upload"    — files.upload() the same zip by hand; no Drive involvement at all
#   "git"       — clone from REPO_GIT_URL; always current, and stamps a real commit hash
#                 into every result bundle (see the provenance note in the README)
#   "drive"     — copy the project folder file-by-file. Slowest; kept as a fallback.
REPO_SOURCE = "drive_zip"
REPO_ZIP = "occl-repo.zip"          # filename to look for on Drive
REPO_DRIVE_PATH = f"{DRIVE}/oc-ccl-improvements"
REPO_GIT_URL = ""

# S2 only: "faithful" decodes .CR2 with rawpy exactly as SST does (~11.9 GB).
# "jpeg" substitutes Zenodo JPEG siblings (~6.8 GB) but is NOT paper-faithful and will
# fail the S2 gate.
SOURCE_PREFERENCE = "faithful"
DATASETS = ["beetle", "butterfly_major"]

# S6 only: 5-shot is run once per dataset. Set this to "beetle" then "butterfly_major".
DATASET_5SHOT = "beetle"

# S3 (adapter smoke) — run once per dataset. Beetle reproduces the paper at both shot
# counts (63.20 vs 61.9, 73.16 vs 74.2) while butterfly is ~11 points low at BOTH, so the
# butterfly overlay is the unexamined artifact that matters most.
#
# S3_GROUP: butterfly defaults to cyrbia, which is the KNOWN OUTLIER and therefore the
# wrong thing to generalise from — set a mid-scoring subspecies such as "malleti"
# (80.71 at 5-shot) to see what the typical case looks like. None = first group.
S3_DATASET = "beetle"
S3_GROUP = None
S3_OVERLAY_EPISODES = 3
# Parallel fan-out width, for EVERY stage that runs the model (s4*, s5*, s6, and the
# evaluation half of s8/s9). 1 = a single process, which is the old behaviour.
#
# Runs are independent by construction — `derive_seed` is a pure hash of
# (dataset, group, shot, run_idx) and every episode does reset_state + del state — and a
# split is verified bit-exact against a sequential control in tests/test_registry.py.
# Two axes compose: groups (butterfly's 5 subspecies, free — no duplicated precompute)
# and run ranges (the ONLY axis for beetle, a single universal group by design).
#
# Measured on the A100-SXM4-40GB (39.5 GB usable), ~3 GB per worker, flat:
#     beetle  5 workers -> 15 GB      butterfly 10 workers -> 35 GB
# MEASURED peaks, not estimates: butterfly hit 4.0-4.6 GB per worker and OOM'd at 10
# (39.40 of 39.49 GB). Beetle ran 5 workers in 15 GB (~3.0 GB each) in ~6 min against
# 27.4 min of summed compute — ~4.6x, near-linear.
#   butterfly: 7 is the safe ceiling (7 x 4.6 = 32 GB)
#   beetle:    11 (11 x 3.0 = 33 GB)
# Above ~10-12 you run out of vCPUs anyway, so there is little left to win.
# "auto" fills the card; an integer requests that many and is still capped if it would
# not fit. Auto is the default because a fixed number cannot know which GPU it landed on,
# and because TTA changes per-worker cost enough to move the answer by 2-3 workers.
WORKERS = "auto"

# Fraction of VRAM the fan-out may occupy. High on purpose: peak residency is a
# *precompute* transient that STAGGER_S already offsets, and an OOM is recoverable
# because the shared resume dir keeps every completed run. Lower it to 0.85 if a run
# dies; raise toward 0.95 to squeeze the last worker in.
VRAM_HEADROOM = 0.92

# Seconds between worker launches. Peak VRAM is during `precompute` (the image encoder),
# so simultaneous starts stack their transients; offsetting them flattens the peak. Costs
# (N-1) x STAGGER_S of wall clock, which is noise against a 6-40 minute stage.
STAGGER_S = 20


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

# s5diag only — which resampling arm. "linear" re-runs the S5a condition from the main
# cache; "area" swaps in a small INTER_AREA re-derive of the SAME 220 images, holding
# everything else fixed, to test whether aliasing on the ~5.3x downsample is costing
# silhouette quality. Build the slice locally and upload the zip beside occl-repo.zip:
#
#   python scripts/build_cache.py --cache-root ~/oc-ccl-data/cache-area \
#       --originals-root ~/oc-ccl-data/originals --datasets butterfly_major \
#       --groups cyrbia malleti --n-runs 2 --interp area --local
#   (cd ~/oc-ccl-data && zip -qr ~/occl-cache-area.zip cache-area)
DIAG_RESAMPLE = "linear"
CACHE_AREA_ZIP = "occl-cache-area.zip"

# ── s8b only: OC-CCL fine-tuning ─────────────────────────────────────────────
# One epoch (FT_EPOCHS=1) is the spec's value; SST defaults to 10, which is defect A4.
# Run the sweep by setting FT_LR to each value in turn — each writes its own bundle, so
# they do not overwrite one another:
#
#   3e-6, 1e-5 (SST's default), 3e-5, 1e-4, 3e-4
#
# Beetle is the instrument: OC-CCL moves it +3.3 against a ±3.7 spread, where it moves
# Major +0.2 against ±1.0 — unfalsifiable at n=20.
FT_DATASET = "beetle"
FT_LR = 1e-5
# 10, matching SST's released defaults ("Defaults: --lr 1e-5, --batch_size 1, --epochs 10")
# and the paper's methods as understood here.
#
# NOTE this classifies as `modified` on `finetune.epochs`, which is currently pinned to 1 —
# see the comment in `registry.FAITHFUL_FINETUNE`. Two tests record that ten is the repo's
# default and NOT the paper's, so the spec is left alone until the paper settles it. The
# run is unaffected; only the faithful/modified label is.
FT_EPOCHS = 10
FT_LORA_R = 32
FT_LORA_ALPHA = 64.0
# "hard_threshold" reproduces SST's released defect A2 — the cycle prompt `(pred > 0)`,
# which severs the autograd graph so the forward leg receives nothing. Measured, the
# forward leg carries 26x the reverse leg's gradient, so that arm trains on ~4% of the
# signal. Kept reachable as an ablation, not as a default.
FT_PROMPT_MODE = "logit_reparam"
FT_NO_RESET = False          # A7: reuse the forward leg's memory instead of resetting it
FT_MAX_STEPS = None          # cap steps for a smoke test; None = full epoch (120 pairs)

# ── s8c only: train + paired evaluation, in one session ──────────────────────
# Three arms scored against ONE baseline: base weights, and one per learning rate here.
# The comparison is `registry.compare` — S7's machinery — over the S4c protocol exactly
# (beetle, 1-shot, 20 runs, canonical_balanced), because the treatment must differ from
# the baseline in weights and nothing else.
#
# The baseline is re-run rather than read from results/s4c.zip: that bundle predates the
# `metric` and `data` blocks, so 8 of 16 comparability fields are None and `compare()`
# refuses it. Re-running is ~6 min and makes the pairing valid.
# Arms as explicit (learning rate, epochs) pairs, so a tuned point and the released
# default can be compared side by side — they differ in BOTH, and a single global epoch
# count could not express that.
#
#   "from_tune"  -> read s8c_tune_<dataset>.zip and score two arms: SST's released default
#                   (1e-5, 10 epochs) and the cross-validated winner. That pairing answers
#                   both questions at once — does OC-CCL help, and did tuning matter.
#   [(lr, ep)..] -> exactly these arms.
#
# Run s8c_tune FIRST if using "from_tune": the selection must be made on held-out train
# folds before the test split is ever scored.
S8C_ARMS = "from_tune"

# ── s8c_tune only: choose the epoch count by cross-validation ────────────────
# Epochs is a hyperparameter, so it is selected on held-out data — folds carved out of the
# TRAIN split, never the test split. `datasets.folds.CVFold` makes the test images
# unreachable by construction rather than by discipline, and `run_tune.py` asserts it.
#
# The criterion is per-trait mIoU on the fold's validation images, NOT the cycle loss:
# S8b showed the cycle loss falling monotonically with learning rate with no evidence any
# of it transfers, so tuning against it would optimise the proxy.
#
# One training run per fold yields the whole epochs curve — adapters are checkpointed each
# epoch and every checkpoint shares the same image embeddings.
S8C_TUNE_K = 5              # 120 train -> 96 fit / 24 val per fold, all 12 species in each
S8C_TUNE_EPOCHS = 10        # the largest epoch count to consider
S8C_TUNE_RUNS = 10          # inner-loop runs; this SELECTS, it does not report

# The lr grid, log-spaced across the interval S8b bracketed. That sweep's paired deltas vs
# 3e-6 were -0.0044 / -0.0126 / -0.0248 / -0.0308 at 1e-5 / 3e-5 / 1e-4 / 3e-4: the gain
# per step peaked between 3e-5 and 1e-4 and then halved while ||dW|| kept doubling, so the
# optimum is INSIDE [1e-5, 1e-4] and refining outside it would spend compute on the tails.
# Five points at 10^(1/4) spacing resolve that interval to within a factor of 1.8.
S8C_TUNE_LRS = [1e-5, 1.78e-5, 3.16e-5, 5.62e-5, 1e-4]

# Grid size, so the cost is visible before the run: (lrs x epochs + 1) points per fold.
# Training dominates — lrs x epochs-worth of steps per fold — while scoring is nearly free
# because every point shares one embedding cache.
# Reuse adapters already trained in this session instead of retraining. Off by default:
# a stale adapter silently scored as a fresh one is the failure this whole stage is
# built to avoid.
S8C_REUSE_ADAPTERS = False

# ── s8d only: base vs OC-CCL at FIVE-SHOT, both datasets ─────────────────────
# The coherent baseline that has to exist before the loss is modified at all. S8c settled
# 1-shot on Beetle (-0.05); 5-shot has never been tested on either dataset, and it is the
# regime where curation stopped mattering (Beetle 5-shot lost nothing to a much worse
# support pool), so it is where a loss that substitutes for curation would have to show up.
#
# ONE arm, not a sweep. S8c scored three configurations and (1e-5, 1 epoch) was both the
# strongest (-0.05 vs -0.24 and -0.35) and the cheapest — a tenth of the training of SST's
# default. Re-sweeping here would spend compute to re-learn that.
S8D_ARM = (1e-5, 1)
# Set to one dataset to resume after a partial failure — e.g. ["butterfly_major"] when
# beetle already landed. The per-dataset bundle is copied to Drive as each dataset
# finishes, and the final combine below then stacks whatever is present, so a one-dataset
# run produces a valid (incomplete) bundle that the gate will correctly refuse for missing
# the other half. Combine the two per-dataset bundles locally to get the full record.
#
# Re-running a dataset that already finished is CHEAP, not wasteful: per-run JSON lives in
# the shared resume dir keyed by weights and module source, not by stage name, so a
# completed arm does zero inference and only merges. `scripts/resume_status.py <dir>` says
# exactly what is banked before you decide.
S8D_DATASETS = ["beetle", "butterfly_major"]
# One adapter per dataset serves every protocol: `CyclePairs` draws from `train_ids` and
# never sees the support selection or the shot count, so the fine-tune is protocol-blind.
S8D_SELECTION = {"beetle": "canonical_balanced", "butterfly_major": "canonical_placement"}

# {base, OC-CCL} x {no TTA, TTA} — four cells per dataset, not two.
#
# The tempting simplification is "OC-CCL is training-side, TTA is inference-side, therefore
# independent, therefore measure OC-CCL without TTA and add them afterwards". S6c is the
# counter-example: two techniques that looked independent had a -4.66 interaction, and the
# combination scored WORSE than the better one alone. There is also a concrete mechanism
# here — TTA chooses between two predictions by logit confidence, and fine-tuning changes
# confidence calibration, so OC-CCL can change how well TTA selects without changing the
# no-TTA score at all.
#
# It is also the number that matters: `placement + TTA` is the best Major arm (77.22), so
# the final OC-CCL result has to be measured on top of the stack we would actually ship.
S8D_TTA_CELLS = [False, True]

# Which passes actually COMPUTE. Merging is separate and free (`--no-precompute` does no
# encoding and no inference), so all four cells are always merged from whatever is banked.
#
# Set to [True] to resume after a failure that only lost the TTA pass: the plain runs are
# already on disk, so recomputing them would pay a full re-encode to then skip every run —
# `evaluate` precomputes before it checks the resume cache, and that ordering cannot be
# fixed without editing `evaluator.py`, which is fingerprinted and would orphan every
# banked run in the project.
S8D_COMPUTE_CELLS = S8D_TTA_CELLS

# ── s8_5 only: OC-CCL as SST's RELEASE trains it (full fine-tune, no adapters) ──
# Every S8 number was produced with LoRA r=32/alpha=64 on memory_encoder +
# sam_mask_decoder — a configuration that appears nowhere in SST's *release*. Its
# `train()` freezes the image encoder and trains the remaining ~11.7M parameters; LoRA
# lives only in `experiments/oc_ccl_ablation.py` with `--lora_rank` defaulting to 0.
# (Re-reading the paper later showed that config IS the paper's A2.2 — the release is
# the thing that deviates; see `registry.FAITHFUL_FINETUNE` and stage `s10`. s8_5 stays
# what it always was: the reproduction of the RELEASED method.)
#
#   s8_5a  1-shot, plain, both datasets
#   s8_5b  1-shot + TTA, both datasets
#   s8_5c  5-shot + TTA, both datasets
#
# SST's own defaults, because the point is to test what produced the paper's +3.3:
# `--lr 1e-5 --batch_size 1 --epochs 10` (its README). Ten epochs still classifies as
# `modified` on `finetune.epochs` — that field remains the one genuinely unresolved entry
# in the spec — and the record carries the true value either way.
S8_5_LR = 1e-5
S8_5_EPOCHS = 10
#: The A2 axis, and the reason s8.5 runs more than one OC-CCL arm.
#:
#: `logit_reparam` is our fix: `m = (logits + 10) / 20` is the exact inverse of SAM 2's own
#: mask->logit map, so the cycle stays differentiable and BOTH legs train. `hard_threshold`
#: is what SST actually ships — `(pred > 0).float()`, zero derivative almost everywhere, so
#: only the REVERSE leg trains.
#:
#: The first full fine-tune measured **-3.95 on Beetle, 0/20 seeds improved**, against the
#: paper's +3.3, with harm scaling in how much the method may change the model (LoRA 1ep
#: -0.05, LoRA 10ep -0.35, full 10ep -3.95). If that pattern holds, the hard threshold —
#: which trains strictly less — should be LESS damaging. A falsifiable prediction, recorded
#: before the run.
S8_5_PROMPT_MODES = ["logit_reparam", "hard_threshold"]

#: s8_5a2 — the REMAINING defect axes, one arm each, Beetle 1-shot only.
#:
#: s8_5a settles A2 (prompt mode). These are what is left between our fine-tune and SST's
#: literal code, plus one arm that goes beyond it:
#:
#:   occl_hard     A2   SST's `(pred > 0).float()` prompt — zero derivative, so only the
#:                      REVERSE leg trains. The single-axis version of the arm s8_5a was
#:                      meant to carry; the run that landed used the pre-axis notebook and
#:                      tested `logit_reparam` only, leaving A2 open.
#:   occl_train    A5   `model.train()` as SST calls it. Inert under LoRA — adapters have no
#:                      dropout or norm state — but under a FULL fine-tune it activates
#:                      dropout AND lets normalisation layers update running statistics that
#:                      then ship inside the checkpoint. Plausibly a regulariser at 10
#:                      epochs. The axis that changed meaning when the adapters went away.
#:   occl_unit     A1   SST's `/255` alone: trains on [0,1], scores on ~[-2.1,+2.6]. Expected
#:                      to HURT, since it reintroduces a train/test mismatch — so it is a
#:                      CONTROL. If it helps, our model of what the loss does is wrong.
#:   occl_sst    A1+2+5 all three restored at once: their literal training configuration, and
#:                      the only arm that can carry the claim "we ran their method as written".
#:   occl_noreset  A7   reuse the forward leg's memory on the reverse pass. NOT toward SST —
#:                      they hardcode the reset, so our default already matches. Exploratory:
#:                      it lets the reverse leg attend to the frame it is reconstructing, a
#:                      shortcut the loss cannot see, so it should make the cycle EASIER to
#:                      satisfy and, if the dose-response holds, more damaging.
S8_5A2_ARMS = {
    "occl_hard":    {"prompt_mode": "hard_threshold"},
    "occl_train":   {"train_mode": True},
    "occl_unit":    {"normalization": "unit"},
    "occl_sst":     {"prompt_mode": "hard_threshold", "train_mode": True,
                     "normalization": "unit"},
    "occl_noreset": {"no_reset": True},
}
S8_5_DATASETS = ["beetle", "butterfly_major"]
S8_5_SELECTION = {"beetle": "canonical_balanced",
                  "butterfly_major": "canonical_placement"}
#: Reuse a trained checkpoint if it is already on disk. Full fine-tunes are ~47 MB rather
#: than LoRA's ~4, so they are worth not recomputing across the three sub-stages — a and b
#: and c all score THE SAME trained model, differing only in the inference protocol.
S8_5_REUSE = True

# ── s10 only: the PAPER's OC-CCL, which the release contradicts on five axes ──
# Every prior OC-CCL arm — including `occl_sst`, the "literal" one — reproduced the
# RELEASED code, and the release (committed ~16 months post-paper) is not the paper's
# method: §3 supervises per-trait y ∈ {0,1}^{W×H×C} (release merges to one blob), A2.2
# fine-tunes LoRA r=32/α=64 on memory encoder + mask decoder for ONE epoch (release:
# full FT, 10 epochs), on ONE fixed labeled x₀ (release: every train image), with
# beetle-only rotation aug on the labeled example (release: none), and §3.2 says the
# cycle prompt "remains differentiable" (release: hard threshold). S8c already ran the
# right LoRA recipe; the three axes it lacked are exactly the S10 reverts below.
#
# PRE-REGISTERED PREDICTIONS — recorded here before the first GPU session; the s10 gate
# prints deltas against them and does not gate on sign:
#   occl_paper_a/b/c  project prior −0.6..+0.6; the paper's claim +1.0..+4.5. Replicate
#                     spread < the paired per-seed std, else x₀ identity dominates.
#   occl_merged       ≈ S8c's −0.05 — the per-trait axis should carry most of any gain,
#                     because Beetle's room is assignment headroom (+15.6).
#   occl_noaug        forfeits ≥ half of any occl_paper gain if orientation is the
#                     mechanism; the head trait falls back first.
#   occl_derange      ≈ occl_paper (120 labels vs 1 shouldn't matter at 1 epoch); a
#                     significant gap makes the label-budget distinction mechanistic.
#   occl_anchored     ≥ occl_paper if OC-CCL is "adapt to your reference" (the paper's
#                     OOD section fine-tunes per test image, so per-run is in character).
#
# Arm axes: defaults are the paper composite (per_trait, aug, fixed x₀ = the seeded
# canonical_balanced support of `reference_run`); each entry overrides ONE axis.
S10_ARMS = {
    "occl_paper_a": {"reference_run": 0},
    "occl_paper_b": {"reference_run": 1},
    "occl_paper_c": {"reference_run": 2},
    "occl_merged":  {"reference_run": 0, "mask_mode": "merged"},
    "occl_noaug":   {"reference_run": 0, "aug": False},
    "occl_derange": {},                      # no reference_run -> derangement sampler
}
#: A2.2 says only "a small degree" — the magnitude is OUR a-priori stand-in, derived the
#: same way the s4b orientation threshold was: from the train distribution itself, not
#: from what scores well. Half the beetle train pool's circular std (55.6°/2). Pinned by
#: a test and by the s10ft gate; if S10 misses, report the gap — do not move the angle.
S10_AUG_DEG = 27.8
S10_AUG_RULE = "half the beetle train pool circular std (55.6/2); s4b's a-priori constant"
#: A2.2: "the default LoRA initialization and learning rate scheduling ... one epoch".
#: The paper names no numeric lr; 1e-5 is carried from S8c's CV choice and recorded as a
#: choice, not a derivation.
S10_LR = 1e-5
S10_EPOCHS = 1
#: Per-trait training runs C≈5 objects through every cycle leg — estimated ~10 GB per
#: process (hard bound 5 × the measured 2.75 GB single-object peak), so cap concurrency
#: at 3 (~30 GB) where the C=1 stages ran 4.
S10_TRAIN_CONCURRENCY = 3
S10_REUSE = True

# ── s11 only: which OC-CCL method helps the CURRENT stack — both datasets, 1-shot ──
# S10 answered the reproduction question (the paper's method is null on Beetle, like the
# release's). S11 changes the question to SELECTION: of the three cycle methods on the
# table — merged silhouette (the release's; the project's only real positive, Major
# 5-shot +0.85..+0.91), per-trait (the paper's), and merged+SCNP (S9a, built and tested
# offline, never trained) — which helps each dataset's best 1-shot protocol, and how
# does each interact with TTA? The square shape is S8d's discipline: on Beetle the
# fine-tune DEGRADED TTA's confidence pick (interaction −0.29), so a plain-only
# measurement can misstate the deployed effect by 4×. The per-objective interaction on
# BOTH datasets is the S12 carrier question — a TTA-folded cycle rides the objective
# that least disturbs (or best sharpens) the confidence criterion TTA selects with.
#
# Simplifications licensed by measurement: every arm uses the derangement sampler (S10:
# reference axis null, derange ≈ fixed ≈ anchored within ±0.1) and NO augmentation on
# either dataset (S10: occl_noaug ≈ occl_paper on beetle; A2.2 prescribes none for
# butterfly anyway). All arms are `modified`-track by construction — selection, not
# reproduction.
#
# PRE-REGISTERED PREDICTIONS (before the run; the s11 gate prints, never gates on sign):
#   BUTTERFLY base_tta        must reproduce S5c 75.18 (gate-pinned; plain base 70.30).
#   BEETLE base_tta           NEW — never measured at 1-shot. Predicted 65..67: TTA was
#                             +2.04 at 5-shot and support variance is larger at 1-shot.
#                             If it lands ≈ the paper's OC-CCL row (65.2), the paper's
#                             +3.3 is numerically indistinguishable from "base + query-
#                             side orientation handling" — the reconciliation reading.
#   merged, butterfly         +0.2..+1.2 plain (silhouette room; 5-shot measured +0.91),
#                             ≈ additive under TTA (S8d Major: +0.91 → +0.85).
#   merged, beetle            −0.3..0.0 plain (4 consistent small negatives), MORE
#                             negative under TTA (S8d: −0.11 → −0.40).
#   per-trait, both           −0.2..+0.8 plain (S10 beetle ~+0.1); TTA sign unknown and
#                             THE measurement: if per-trait's TTA interaction is more
#                             negative than merged's, calibration sensitivity scales
#                             with per-channel supervision — S12's direct motivation.
#   scnp (merged+SCNP w=3)    butterfly plain +0.2..+1.5 (sharpens a silhouette that has
#                             room; the term's own paper gains most on boundary/topology
#                             errors); beetle plain −0.5..0.0 — scnp.py's docstring
#                             pre-registers beetle as the near-analogue of the paper's
#                             DETRIMENTAL case (small structures), and the S9a demotion
#                             predicted "more of what already hurts". TTA: unknown;
#                             sharpened logits plausibly help the confidence pick —
#                             the one arm that could enhance TTA without a fold.
S11_ARMS = {
    "occl_merged":   {"mask_mode": "merged"},
    "occl_pertrait": {"mask_mode": "per_trait"},
    "occl_scnp":     {"mask_mode": "merged", "scnp_width": 3},   # S9a; width 3 = the
                                                                 # tested/paper value
}
S11_DATASETS = ["butterfly_major", "beetle"]
S11_SEL = {"butterfly_major": "canonical_placement",   # S5b/S5c protocol
           "beetle": "canonical_balanced"}             # S4c protocol
S11_LR = 1e-5
S11_EPOCHS = 1
#: Butterfly per-trait training runs C≤10 objects per cycle leg (hard bound ~27 GB from
#: the measured 2.75 GB single-object peak) — train sequentially, never concurrently.
S11_TRAIN_CONCURRENCY = 1
S11_REUSE = True

# ── s12 only: SCNP as the toggle on the per-trait cycle, best-of-3 headline ──
# S11 crowned per-trait + TTA on both datasets and closed SCNP-on-merged (harmful on
# beetle, nothing over its carrier on butterfly, gains boundary-scale). What it could
# not test is the composition `scnp.py`'s docstring was written for: per-trait channels
# give "same-class neighbour" a trait identity, so cross-trait bleed — the failure mode
# behind Beetle's assignment headroom — finally collects SCNP's multiplied gradient.
# The guard in cycle.py is lifted; per-channel independence is pinned by test.
#
# Design economy, per the project's own data: the toggle-OFF arm (plain per-trait) is
# NOT recomputed — S11's banked per-trait bundles are the base cells, pulled from
# Drive. Only `occl_ptscnp` trains and evaluates. The HEADLINE is the paired
# best-of-3 delta (`registry.compare(..., metric="miou_boundary.independent_best3")`):
# the boundary-tolerant reading is the paper's own scoring and it is the metric that
# already absorbed SCNP's boundary-scale gains once — a composition that only moves the
# plain metric is trading in annotation jitter.
#
# PRE-REGISTERED PREDICTIONS (before the run; printed by the gate, never sign-gated):
#   prior (every SCNP measurement to date): best-of-3 paired Δ vs per-trait within
#     ±0.3 on both datasets → SCNP closes entirely, composition included.
#   the docstring-optimist alternative: beetle +0.5..+2 under best-of-3 with CI > 0
#     (trait bleed finally bitten) → the composition is real and earns extension.
#   the disqualifier, regardless of plain gains: if adding SCNP flips per-trait's TTA
#     interaction negative (per-trait is the carrier BECAUSE that interaction is
#     positive on both datasets under both metrics), the term damages the property
#     that selected its carrier and is closed.
S12_LR = 1e-5
S12_EPOCHS = 1
S12_SCNP_WIDTH = 3
S12_TRAIN_CONCURRENCY = 1          # butterfly per-trait: C≤10 per leg
S12_REUSE = True

# ── s13 only: five loss arms on the per-trait carrier, TTA-only, best-of-3 ───
# The loss-modification program's theory+research restart: (A) trait separation and
# (B) shape highlighting, on the crowned per-trait + TTA stack, every arm a pure
# per-object-logit-stack operation so it transfers to SAM 3 (which keeps SAM 2's
# tracker wholesale) and to Cutie (whose native objective arm A1 literally is).
#
# The evidence base that reshaped the targeting (S13 exploration, banked data only):
# the project's "+15.6 assignment headroom" was silhouette−mIoU — a CORRELATE, not a
# ceiling; the true oracle−argmax ceiling has never been computed on beetle and gets
# lifted into this stage's own combine (combine_s8d_cells now lifts miou_variants.
# oracle — Day-0, free). Beetle's 36.8-pt deficit decomposes into ~9.5 pts of IoU=0
# episodes (head 24%/pronotum 12% — the 180°-flip signature) and ~27.3 pts of
# shape-on-found (antenna/legs at ~50 even when found); 1-px jitter is +0.51. The
# ordinary cycle never constructs a 180° support↔query relation — the coherent theory
# for why every cycle arm measured ~null on beetle — and the margin fold is the one
# arm that constructs it.
#
# TTA-ONLY cells (user decision): the deployment stack is TTA and every comparator is
# banked from S11, so the plain cells are dropped — halves eval; forfeits the per-arm
# TTA-interaction diagnostic and the margin fold's plain-~null check, on the record.
#
# PRE-REGISTERED PREDICTIONS (frozen before any GPU run; the s13 gate prints, never
# sign-gates):
#   P1 margin (vsbase, beetle): +0.2..+1.5 with CI > 0 — every cycle loss measured
#      <= 0 there, so any gain is selection-sharpening proven in isolation.
#      P1-null path (pre-registered after the CPU smoke, before any GPU run): the
#      smoke's beetle pairs came back hinge-PRE-SATISFIED (gaps 1.37/3.00 > m=1.0);
#      if violations == 0 over the full epoch the arm is structurally a no-op
#      (||dW|| exactly 0, delta exactly 0.00) — the s13ft gate records that rather
#      than failing, and the RECORDED fallback (softplus form, which never turns
#      off) becomes the follow-up decision. Per "let the run decide", the hinge is
#      NOT swapped preemptively on two steps of smoke evidence.
#   P2 ptmargin vs pertrait_tta: beetle +0.2..+1.0; butterfly +0.0..+0.8 with the
#      crowned 76.52/77.81 stack unharmed.
#   P3 cyrbia per-run TTA std collapses 10.89 -> < 6 under ptmargin (arm bundles).
#   P4 beetle IoU=0 cohort (~9.5 pts of mass) shrinks >= 2 pts under ptmargin+TTA.
#   P5 excl: judged against the Day-0 oracle ceiling. If beetle oracle−argmax >= ~3
#      pts: optimist +0.5..+2.0, prior (every beetle cycle-loss arm to date) ±0.3.
#      If the ceiling is < ~2 pts the assignment pot never existed and a null is
#      PREDICTED, not a failure.
#   P6 xovl: excl's direction, smaller magnitude (softer coupling); prior ±0.3.
#   P7 bshape: beetle optimist +0.5..+2.0 (thin-structure recall — antenna/legs are
#      the deficit's majority), butterfly +0.3..+1.0 (its ~22-pt post-TTA shape
#      mass); prior ±0.3. Must show in best3: the 1-px dead zone trades only in
#      error the metric cannot forgive.
#   D1 disqualifier (pose-prior shortcut): the fold learns "the support LOOKS
#      upright" instead of correspondence — cyrbia's rotated-pick fraction
#      (cells' *_tta_rotated_frac; truth ~37%) collapses toward 0 with zero-cohort
#      regrowth -> the fold is closed.
#   D2 disqualifier: margin (vsbase, beetle) CI < 0 — the trained statistic degrades
#      the pick it trains (S8d reproduced by the fix) -> fold closed; the
#      calibration guard (README deferred brief, option 2) becomes the only route.
S13_ARMS = {
    # (A) trait separation
    "occl_excl":     ["--excl-mode", "zerocat"],       # the eval rule as the loss;
                                                       # STM/Cutie's native objective
    "occl_xovl":     ["--xovl-weight", "1.0"],         # label-free repulsion on the
                                                       # forward leg (SAM-MT)
    # (B) shape highlighting
    "occl_bshape":   ["--bshape-weight", "1.0"],       # Kervadec signed-DT, 1-px
                                                       # dead zone, chan-max norm
    # (A) the fold — margin ALONE (mechanism isolation) and on the cycle
    "occl_margin":   ["--margin-only", "--margin-m", "1.0"],
    "occl_ptmargin": ["--margin-weight", "1.0", "--margin-m", "1.0"],
}
#: every axis in the checkpoint filename — the s8_5a reuse trap.
S13_ARM_TAGS = {
    "occl_excl":     "excl-zerocat",
    "occl_xovl":     "xovl-w1-query",
    "occl_bshape":   "bshape-w1-const-dz1-cmax",
    "occl_margin":   "marginM1-only",
    "occl_ptmargin": "marginM1-w1",
}
S13_LR = 1e-5
S13_EPOCHS = 1
S13_MARGIN_M = 1.0                 # a-priori: |SAM2_OUT_BIAS| / 10 — see the rule
S13_MARGIN_RULE = ("one tenth of SAM 2's own mask-prompt logit amplitude "
                   "|out_bias|=10 (cycle.py SAM2_OUT_BIAS)")
S13_TRAIN_CONCURRENCY = 1          # butterfly per-trait C≤10 per leg; ptmargin is
                                   # 1.5x a cycle step (3 encodes + 6 track legs)
S13_REUSE = True

# ── s16 only: TTA as an implicit second shot — 4 runs, SAM 2, both datasets ──
# The premise (user, 2026-09-01): orientation TTA is analogous to an IMPLICIT 2-shot —
# the rotated support is a second reference (same specimen, second orientation) carried
# OUTSIDE the memory bank and selected by a confidence rule over the logits. Explicit
# 2-shot puts a second real specimen INSIDE the bank (one 3-frame video {x0, x1, xq},
# the paper's own multi-shot construction) and lets memory attention do the selecting.
# Same tracker (base SAM 2, no adapters), same canonical protocol, the same 20 seeds —
# FOUR runs: {butterfly, beetle} x {1-shot + TTA, 2-shot plain}. One paired row per
# dataset, the HEADLINE: 2-shot plain vs 1-shot TTA — which second reference pays, the
# implicit or the explicit one. In SST's own currency the implicit one is free and the
# explicit one costs a labelled mask. (The {1,2}-shot x {plain, TTA} square was cut to
# these 4 runs by user decision; the 1-shot TTA cell doubles as its own anchor, pinned
# to the S11 ledger, and the cut single-lever deltas remain reconstructible against
# S11's banked plain cells if the headline demands them.)
#
# Pairing across shot counts is licensed by the draws, not assumed: `sample_supports`
# and `stratified_supports` are prefix-nested (run i's 1-shot support is the first of
# its 2-shot pair — verified on the real canonical pools, 100/100 butterfly and 20/20
# beetle (group, run) pairs, before any GPU was spent), and
# `registry.compare(nested_shots=True)` re-verifies it on every per-run record before
# masking `protocol.shots` — S14's cross-family discipline, with the proof embedded in
# the row. SAM 2's `max_cond_frames_in_attn` is −1 (all), so both supports are
# attended; the sam3 trap (cap 4) does not arise at 2.
#
# PRE-REGISTERED PREDICTIONS (measured components only; the gate prints them, never
# sign-gates the headline — README "S16 brief" carries the derivations):
#   butterfly: 2-shot plain 70.6..71.3 (banked 1-shot 70.30; 5-shot gave only +1.48);
#     the TTA cell re-lands 75.18 ± 0.5 (pinned). HEADLINE ≈ −3.9..−4.6 ✱ — the
#     IMPLICIT shot wins: Major's defect is QUERY orientation, and a second upright
#     specimen in the bank cannot reach it.
#   beetle: 2-shot plain 65.7..69.2, central ≈ 67.5 (5-shot gave +9.96 by species
#     diversity — a 2-shot draw spans 2 strata; the shot curve is concave); the TTA
#     cell re-lands 63.86 ± 0.5. HEADLINE ≈ +2..+5.3 ✱, central +3.6 — the EXPLICIT
#     shot wins: a rotated copy of the same specimen adds no species.
#   The structural claim: the datasets are MIRROR IMAGES on which second reference
#   pays, each by the mechanism its residual named.
#   DISQUALIFIERS (gate FAILS — plumbing, not science): beetle 2-shot ≤ the BANKED
#     1-shot level 63.20 (a conditioning frame dropped — the S6 check one shot down;
#     unpaired against the ledger, since the plain 1-shot cell is not re-run);
#     butterfly 2-shot < 69.80 (ledger 70.30 − 0.5, same reading). Recorded, not
#     gated: beetle 2-shot above 73.16 (exceeds the measured 5-shot LEVEL — suspect
#     the draw, e.g. two strata collapsing to one).
S16_DATASETS = ["butterfly_major", "beetle"]
S16_SEL = {"butterfly_major": "canonical_placement",   # S5b/S5c protocol
           "beetle": "canonical_balanced"}             # S4c protocol
#: label -> (shots, tta): the implicit second reference and the explicit one.
S16_CELLS = {"s1_tta": (1, True), "s2": (2, False)}

# ── s14 only: the crowned stack on SAM 3 and SeC-4B ──────────────────────────
# S13 closed the loss program and located beetle's residual (~33 pts) in regions
# SAM 2 never proposes (true oracle ceiling +2.65) — a model-capacity boundary. S14
# is the swap the program was built for: per-trait OC-CCL + excl(zerocat) + TTA on
# two newer models, every port operating on per-object logit stacks only.
#
# PRE-REGISTERED PREDICTIONS (frozen before any GPU run; gates print, never
# sign-gate):
#   P1 sam3 base vs sam2_base (cross-family, paired on identical draws): optimist
#      beetle +1..+8 (the ViT backbone proposes the thin structures SAM 2 never
#      did), butterfly +1..+6; prior ±2 (museum stills are off-distribution for
#      SAM 3's video training too).
#   P2 THE MECHANISM READ: beetle oracle−argmax on the new family's base. > ~+5:
#      proposals improved but assignment lags -> excl's pot is real and P5's
#      optimist band activates. ≈ +2.65: same proposal boundary — the upgrade is
#      cosmetic on this benchmark. Either reading is a finding.
#   P3 sec off vs sam2_base: prior ±2 — a stage-1 SA-V-full-fine-tuned SAM 2.1-L;
#      drift in either direction measures SA-V fine-tuning vs specimen stills.
#   P4 sec forced vs off: optimist +0.5..+2 beetle (concept-level identity reaches
#      the 180° flip cohort that no pixel memory does); prior ±0.5. The NATIVE gate
#      is pre-registered blind (HSV histograms are rotation-invariant): every off
#      episode records the would-fire statistic, expected fire-rate ≈ 0.
#   P5 stack_tta vs own base_tta, per family: prior ±0.3 (the S13 law — bounded by
#      that family's own assignment pot); conditional optimist +0.5..+2 iff P2 read
#      ≥ ~3 pts on beetle.
#   P6 TTA's value on the new families (base_tta vs base): recorded, not banded —
#      SAM 2's +4.7 butterfly may compress if orientation handling improved
#      natively; a collapse to ~0 with base ≈ sam2_base+5 would mean the flip
#      cohort was eaten by the backbone.
#   P7 HEADLINE: stack_tta vs sam2_pertrait_tta (76.52 / 64.12, cross-family) —
#      any CI > 0 means the model upgrade pays under the deployment stack.
#      Beetle band if P2-optimist: +2..+8.
S14_LR, S14_EPOCHS = 1e-5, 1
S14_SAM3_CKPT_DRIVE = "occl_ckpts/sam3.pt"   # 3.45 GB, HF-gated — downloaded once
S14_SEC_HF = "OpenIXCLab/SeC-4B"             # ungated, 15.18 GB, per-session direct
#: Per-phase eval-worker widths, from the VRAM arithmetic the SAM2-calibrated
#: `plan_group_workers` cannot know (A100 40 GB):
#:   sam3:       ~1.9 GB fp32 weights + ~2.1 GB dual-orientation group cache
#:               (10.6 MB/img) + ~3 GB workspace ≈ 7 GB/worker  -> 4 wide.
#:   sec off/tta: LVLM dropped BEFORE the device move (adapters/sec.py), so it
#:               never reaches the GPU; ~0.45 GB bf16 tracker + slice cache +
#:               workspace ≈ 2.5-4 GB/worker                    -> 5 wide.
#:               (Measured hazard: with the drop AFTER the move, each worker held
#:               8.42 GB — the allocator keeps freed VRAM reserved — and 4
#:               residents + a fifth build exhausted the 39.5 GB card.)
#:   sec forced: the 7.6 GB LVLM is resident and INVOKED per episode x class;
#:               + tracker + shared-pos-enc cache + workspace
#:               ≈ 10-11 GB/worker                              -> 3 wide, hard cap.
#:               (Measured hazard: per-image pos-enc caching added ~4.6 GB of
#:               identical tensors per worker — 15 GB each, OOM at 3 wide; the
#:               cache now keeps ONE shared copy, the sst_sam2/sam3 discipline.)
#: The forced fan-out is the stage's whale (~2-2.6 h single-stream on butterfly);
#: 3-wide cuts it to ~0.8 h — the single biggest wall-clock lever in S14. A
#: fine-tune/forced overlap was considered and REJECTED: FT (~14 GB) would force
#: the whale down to 2 workers, a net loss.
S14_EVAL_WORKERS = {"sam3": 4, "sec_off": 5, "sec_forced": 3}
S14_REUSE = True

# ── profile only ─────────────────────────────────────────────────────────────
# The `profile` stage. Butterfly at 5-shot is the configuration that went wrong, so it is
# the default: 6 frames of 1024x1024 through SST's CPU-side `load_video_frames`, and
# `1 + shot*classes + shot + 1` = 57 `_get_image_feature` calls per episode against 13 at
# 1-shot. Point it at beetle 1-shot to see the configuration that was always fine.
# Force the per-group worker count for s8d instead of letting `plan_group_workers` size it
# from `PEAK_GB`. That table is MEASURED-conservative for TTA: it projects 8.73 GB a worker
# where the running pass shows ~5. Set an integer once `nvidia-smi --query-compute-apps` has
# told you the real figure; leave None to let the planner choose.
S8D_WORKERS_PER_GROUP = None

PROFILE_DATASET = "butterfly_major"
PROFILE_SHOT = 5
PROFILE_GROUP = "malleti"       # mid-scoring; cyrbia is the known outlier
PROFILE_EPISODES = 12

# Keep video frames and tracking state on the GPU rather than offloading to host.
#
# The embedding cache skips the image ENCODER, but the patch that installs it still
# re-uploads the raw 12.6 MB frame on every `_get_image_feature` call — and all three
# callers in SAM 2 discard it (`_, _, current_vision_feats, ...`). On-device frames make
# that upload a no-op and move the ImageNet normalisation off the CPU.
#
# Placement only: it decides where tensors live, not what is computed from them. Reached
# through a CONSTRUCTOR ARGUMENT rather than an edit to `adapters/sst_sam2.py`, which is
# fingerprinted — a line changed there orphans every per-run JSON ever banked.
#
# **ON, because it was measured** (results/inbox/profile.zip, A100-SXM4-40GB, butterfly
# malleti 5-shot, 12 episodes):
#
#   offload both (was the default)   1.803 s/episode   717 MB host->device
#   frames on GPU                    1.606             0 MB
#   frames + state on GPU            0.848             0 MB      <- 2.13x
#
# and both alternatives returned logits **bit-identical** to the default, max|diff| = 0.0.
# That equivalence is the licence: the resume fingerprint does not cover these flags, so a
# placement that changed the arithmetic would silently mix old and new runs in one cache.
#
# The attribution is not what was predicted. The redundant frame upload — 57 calls x 12.6 MB
# that every SAM 2 caller discards — is real and measured, but worth only **21%** of the
# saving (170.5 -> 5.3 ms). **79% is `offload_state_to_cpu`**: the memory bank round-tripping
# to host between frames, visible almost entirely in `add_new_mask`, 1569 -> 598 ms. Six
# frames x ten traits means 50 prompt calls an episode, each shuttling its maskmem to the
# host and back, and that was the dominant cost all along.
GPU_FRAMES = True

# ── s6c only: does query-side TTA stack with orientation-diverse supports? ───
# Rotation has been solved TWICE on Major and never together — TTA at 1-shot (+4.88,
# query-side: predict both ways, keep the confident one) and orientation-diverse supports
# at 5-shot (+4.01, support-side: the bank holds both orientations). Both land ~+4.5 on
# the same defect. This completes the 2x2 at 5-shot; the no-TTA cells are S6 (71.78) and
# S6b (75.80).
#
# PREDICTION, recorded before the run: `tta.rotate_episode` rotates ALL supports jointly,
# so against an already-diverse 5-set (3 upright + 2 rotated) it produces 3 rotated + 2
# upright — a set that still spans both orientations. The two candidates are then near
# equivalent and the confidence pick is close to arbitrary, so `orientation_diverse + TTA`
# should be roughly a no-op. `canonical_placement + TTA` is the informative cell: it is the
# clean upright-vs-rotated choice, and it is what tells us whether the two mechanisms are
# interchangeable rather than additive.
S6C_SELECTIONS = ["canonical_placement", "orientation_diverse"]

# The whole cache as ONE zip — lossless PNG, both datasets, every support arm.
# A single FUSE read is far faster than copying ~1,500 files out of Drive.
CACHE_ZIP = "occl-cache-png.zip"

# Optional second zip, extracted ON TOP of CACHE_ZIP and merged into its manifest.
#
# A new support arm draws images the old arms never did: `orientation_diverse`
# deliberately selects rot180 specimens, which the uniform and placement draws almost
# never picked, so 265 of its 445 supports are absent from a cache built before it
# existed. Re-zipping the whole 2.4 GB cache to add ~0.5 GB is the wrong trade over a
# home upload, so the delta ships separately. Set to None when there is no delta.
#
# A bare filename is probed in MyDrive/, MyDrive/occl/, MyDrive/oc-ccl-data/ and one
# level below MyDrive. An ABSOLUTE PATH is also accepted and skips probing entirely —
# use that if the file landed somewhere else, e.g.
#   CACHE_DELTA_ZIP = "/content/drive/MyDrive/occl-cache-png-delta.zip"
# A miss prints every path tried and every zip it can actually see on Drive.
CACHE_DELTA_ZIP = "occl-cache-png-delta.zip"

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
    _extracted_stages = open(f"{REPO}/notebooks/run_stage.py").read() \
        .split("STAGES = (", 1)[1].split(")", 1)[0]
    assert f'"{STAGE}"' in _extracted_stages, (
        f"{os.path.basename(_zip)} predates stage {STAGE!r} (its STAGES tuple lists: "
        f"{_extracted_stages.strip()}). Rebuild with scripts/build_repo_zip.py and "
        "replace the zip on Drive.")
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
# NEVER `pip install -r` SST's requirements.txt: it pins transformers==4.34.0, has no
# torch line at all, and carries conflicting opencv pins.
!pip install -q -e . requests pillow
if STAGE == "s2":
    !pip install -q rawpy                      # .CR2 decoding, faithful mode
else:
    !pip install -q hydra-core omegaconf iopath tqdm

# Put src/ on sys.path explicitly. `pip install -e .` alone is NOT enough here: an
# editable install works through a .pth file (or a PEP 660 finder referenced from one),
# and .pth files are only read by site.py at interpreter startup. This kernel is already
# running, so it never sees the new entry and `import occl` raises ModuleNotFoundError
# until the runtime is restarted. Inserting the path directly avoids that entirely.
import sys

_SRC = f"{REPO}/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import occl  # noqa: F401  — fail here, next to the fix, rather than three cells later

# The notebook and the repo zip travel as a PAIR and are packed together, but only the
# notebook is open in front of you — so it is entirely possible to re-run with a months-old
# zip and never notice. Twice now that has produced a bundle that looked fine and was
# missing the thing the stage existed to produce. This is the cheap end of that: the
# notebook knows the build it was packed against and refuses to run against an older one.
REPO_MIN_BUILD = 9
NOTEBOOK_BUILD = 9          # bumped with REPO_MIN_BUILD; pack.sh ships them together
_build = getattr(occl, "BUILD", 0)
assert _build >= REPO_MIN_BUILD, (
    f"occl-repo.zip on Drive is build {_build}; this notebook needs >= {REPO_MIN_BUILD}.\n"
    f"  The two are packed together by scripts/pack.sh — re-upload BOTH\n"
    f"  occl-repo.zip and run_stage.ipynb, then restart the runtime.\n"
    f"  Running anyway produces a bundle that looks correct and is not."
)
# The other direction, which was missing. Stage dispatch lives in THIS file, so a stale
# notebook with a fresh zip runs the previous experiment against new library code and
# reports it under the current stage name — no error, wrong experiment.
_want_nb = getattr(occl, "NOTEBOOK_MIN_BUILD", 0)
assert NOTEBOOK_BUILD >= _want_nb, (
    f"this notebook is build {NOTEBOOK_BUILD}; occl-repo.zip (build {_build}) needs a "
    f"notebook >= {_want_nb}.\n"
    f"  You updated the zip but not run_stage.ipynb. The stage dispatch lives in the\n"
    f"  notebook, so this would run the PREVIOUS experiment against the new library.\n"
    f"  Re-download run_stage.ipynb from the same pack.sh output and re-open it."
)
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

    # Stages that CANNOT run without the delta stop here rather than at inference. Without
    # this the shortfall surfaces 8 times over, once per worker, as a preflight failure
    # several cells later — same information, buried, and after the GPU is already warm.
    if STAGE == "s6b" and DATASET_5SHOT == "butterfly_major" and not _dz:
        raise FileNotFoundError(
            f"s6b on butterfly_major needs {CACHE_DELTA_ZIP}: the orientation-diverse arm "
            "draws 265 rot180 supports that no earlier arm ever selected, so they are "
            "absent from the main cache. Upload it (or point CACHE_DELTA_ZIP at wherever "
            "it landed — an absolute path works) and re-run this cell."
        )

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
# ── Hydra import smoke — fails fast, BEFORE the 898 MB checkpoint download ───
# The vendored sam2/__init__.py calls initialize_config_module("sam2_configs") at import
# time, which only resolves with SST/src/sst on sys.path. Catching that here costs
# seconds; catching it after the weights download costs minutes.
if STAGE not in ("s2",):
    paths.bootstrap_sst_syspath()
    import sst.segment_anything_2.sam2  # noqa: F401

    print("vendored SAM 2 imports cleanly")

# %%
# ── checkpoint (SAM 2.0 hiera-large, 072824 — the paper's checkpoint) ────────
# NOT SAM 2.1: its checkpoints add keys that will not load into the vendored 2.0 config.
if STAGE not in ("s2",):
    CKPT = f"{SST}/checkpoints/sam2_hiera_large.pt"
    if not os.path.exists(CKPT):
        !mkdir -p {SST}/checkpoints
        !wget -q --show-progress -O {CKPT} \
            https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    print("checkpoint:", os.path.getsize(CKPT) / 1e6, "MB")

# %%
# ── run the stage ────────────────────────────────────────────────────────────
# ── parallel fan-out, shared by every model-running stage ────────────────────
import subprocess as _sp
import time as _t


def _assign(groups, n_runs, n_workers):
    """Partition (group, run) work into EXACTLY n_workers slices, each pair owned once.

    The previous version computed `per_group = ceil(n_workers / n_groups)` and gave every
    group that many splits, so the total was `n_groups * ceil(...)` — NOT what was asked
    for. On butterfly's 5 groups, requesting 6, 7, 8, 9 or 10 workers all produced **10**,
    and 11 produced 15. Asking for 7 at 5.64 GB each therefore launched 56 GB onto a
    39.5 GB card. That is what OOM'd S6.

    It went unnoticed because the coverage test checked that every (group, run) was owned
    exactly once — which the old version did satisfy — and only ever exercised 1, 5 and
    10 workers, the three values where the arithmetic happens to be right.

    Group-splitting first (it duplicates no encoder work), run-splitting only to reach the
    requested count. With fewer workers than groups, a worker takes several groups.
    """
    g = list(groups)
    n_workers = max(1, int(n_workers))

    if n_workers <= len(g):
        # Bundle groups: worker i takes a contiguous slice of the group list.
        edges = [round(i * len(g) / n_workers) for i in range(n_workers + 1)]
        return [(g[a:b], 0, n_runs) for a, b in zip(edges, edges[1:]) if b > a]

    # More workers than groups: give each group a share of the run range. Groups differ
    # by at most one split, so the load stays even.
    base, rem = divmod(n_workers, len(g))
    out = []
    for k, grp in enumerate(g):
        splits = base + (1 if k < rem else 0)
        edges = [round(i * n_runs / splits) for i in range(splits + 1)]
        out += [([grp], a, b) for a, b in zip(edges, edges[1:]) if b > a]
    return out


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
    # s16 (2-shot): `fixed` is weights + activations and the 1- and 5-shot rows agree on
    # it, so it is copied rather than left to the 4.07 default (which would halve the
    # fan-out for no reason); the cache term is derived by `cache_gb` from the shot.
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


def save_calibration(dataset, shot, tta, fixed_gb):
    """Record a worker's FIXED cost, so the next run sizes itself from measurement.

    Hand-maintained estimates have been wrong in both directions and cost two OOMs and
    one run at a third of the card. A number the machine measured beats a number I
    inferred from a screenshot.

    **Fixed, not peak.** Residency is fixed + the worker's own slice's embedding cache, and
    the cache scales with how many runs that worker owns. Storing the total meant a peak
    measured on a narrow slice was later used to price a wide one — the same conflation that
    made `PEAK_GB["cache"]` a constant. `fanout` subtracts each worker's recorded
    `embedding_cache` before calling this, so what lands here is comparable across slices.
    """
    import json as _js

    cal = load_calibration()
    cal[_cal_key(dataset, shot, tta)] = round(float(fixed_gb), 3)
    try:
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        with open(_calibration_path(), "w") as fh:
            _js.dump(cal, fh, indent=1, sort_keys=True)
    except Exception:  # noqa: BLE001 — never fail a run over bookkeeping
        pass


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


def plan_workers(dataset, shot, requested, tta=False, headroom=None,
                 n_runs=20, groups=None):
    """Choose the worker count from the card and the CPU actually present.

    Scales **up** as well as down. The previous version only ever capped a request, so the
    default WORKERS=5 stayed 5 even where 7 fit — 23 GB of a 39.5 GB card, with the
    remaining 16 GB simply unused. Pass `WORKERS = "auto"` (the default) to fill to
    `VRAM_HEADROOM`, or an explicit integer to request that many and still be capped if it
    would not fit.

    Two ceilings, and the lower wins:

    * **VRAM** — measured per-worker cost against real capacity. Headroom can sit high
      because peak residency is a *precompute* transient and STAGGER_S already offsets
      those; steady-state is well below peak. An OOM is also recoverable: the shared
      resume dir keeps every completed run, so a restart at a lower count resumes rather
      than starting over.
    * **vCPU** — ~12 on Colab's A100 runtime. With OMP_NUM_THREADS=1 each worker needs
      about one core to feed the GPU, and past that they thrash and get *slower*. Beetle
      at 3.00 GB/worker would allow 12 on VRAM alone, which the CPU cannot feed.

    The VRAM ceiling is now solved rather than divided. Per-worker cost depends on the
    *slice* a worker gets and the slice depends on the worker count, so `cap / per` was
    answering a question whose input it did not have: it priced every candidate as if each
    worker cached a whole group. Each candidate is therefore partitioned with the real
    `_assign` and priced from that.

    Total residency *is* monotone in the worker count — an extra worker adds a whole model
    plus that group's 100 test embeddings, which run-splitting cannot divide, and only
    subdivides the supports — so the largest fitting count is well defined. The scan runs to
    the vCPU cap anyway rather than breaking early, because that monotonicity is a property
    of the current cost model rather than something the loop should assume.

    Returns (n_workers, per_worker_gb, capacity_gb, note).
    """
    import torch

    headroom = VRAM_HEADROOM if headroom is None else headroom
    groups = list(groups or GROUPS_OF.get(dataset, ["all"]))
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9
    budget = cap * headroom
    by_cpu = max(1, (os.cpu_count() or 4) - 1)     # leave one core for the parent

    def price(n):
        """(worst per-worker GB, total GB) for an n-way partition of this stage."""
        work = _assign(groups, n_runs, n)
        pers = [per_worker_gb(dataset, shot, tta, runs_in_slice=(b - a), n_runs=n_runs)
                for _g, a, b in work]
        return max(pers), sum(pers)

    # Every candidate up to the vCPU cap. See the docstring: the total is monotone under
    # this cost model, so `max(fits)` is the answer either way — the scan just does not
    # depend on that staying true.
    fits = [n for n in range(1, by_cpu + 1) if price(n)[1] <= budget]
    by_vram = max(fits) if fits else 1
    ceiling = min(by_vram, by_cpu)

    per, total = price(ceiling)
    if str(requested).lower() == "auto":
        n = ceiling
        note = (f"auto: {n} workers (worst {per:.2f} GB -> {total:.1f} GB of {cap:.1f}, "
                f"{100 * total / cap:.0f}%); limited by "
                f"{'VRAM' if by_vram <= by_cpu else f'vCPU ({by_cpu + 1} cores)'}")
    else:
        n = min(int(requested), ceiling)
        per, total = price(n)
        note = "" if n == int(requested) else (
            f"capped {requested} -> {n}: {requested} workers would need "
            f"{price(int(requested))[1]:.1f} GB against {cap:.1f} GB "
            f"and {by_cpu + 1} vCPUs")
    return n, per, cap, note


def train_parallel(jobs, label="train", stagger=None):
    """Run several fine-tunes concurrently. ``jobs`` = [(name, argv, logpath), ...].

    Training was serial everywhere except s8c_tune, which has run its folds concurrently
    since it was written. Nothing justified the difference: a training process is **2.33 GB
    peak** (measured, `s8b_beetle_lr1e-05.zip`) against the ~4.9 GB an inference worker
    holds, and a full fine-tune adds only gradients plus Adam state for 11.7M parameters —
    47 MB and 94 MB — so ~2.5 GB each. Four concurrent is ~10 GB on a 42 GB card.

    The saving is real because training is not short: 43 s/epoch on beetle, so s8_5a2's four
    arms at 10 epochs are ~29 minutes serially and roughly a third of that in parallel.

    Independence is by construction, not by hope — each job is its own process with its own
    model, and pair order comes from `seeding.derive_seed`, which is a pure hash. They share
    only the GPU.

    Uses `worker_env()` for the same reason every other launcher does, and `wait_all` so an
    interrupt kills the survivors instead of orphaning them.
    """
    stagger = STAGGER_S if stagger is None else stagger
    procs = []
    for i, (name, cmd, logpath) in enumerate(jobs):
        if i:
            _t.sleep(stagger)
        log = open(logpath, "w")
        print(f"  [{name}] $ {' '.join(cmd)}  -> {logpath}", flush=True)
        procs.append((name, log, subprocess.Popen(
            cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=worker_env())))
    bad = wait_all(procs, label)
    assert not bad, f"{label} failed for {bad} — see the logs above"


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


def fanout(*, stage, dataset, shot, n_runs, selection, workers, bundle, extra=()):
    """Run a stage across `workers` processes, then aggregate. workers=1 is serial.

    Every worker writes per-run JSON into the SAME --resume-dir; the slug carries group
    and run index, so disjoint assignments never collide. The merge is then just another
    pass with --no-precompute: every run is already cached, so it does zero inference and
    only aggregates. The resume machinery IS the merge machinery.
    """
    base = ["python", "scripts/run_eval.py", "--stage", stage,
            "--adapter", "sst_sam2", "--dataset", dataset,
            "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
            "--shot", str(shot), "--n-runs", str(n_runs),
            *(["--gpu-frames"] if GPU_FRAMES else []),
            "--resume-dir", f"{RESULTS_ROOT}/runs"]
    if selection:
        base += ["--support-selection", selection]

    if str(workers).lower() != "auto" and int(workers) <= 1:
        run(base + list(extra) + ["--out", bundle, "--no-download"])
        return

    _tta = "--tta" in list(extra)
    workers, _per, _cap, _note = plan_workers(dataset, shot, workers, tta=_tta,
                                              n_runs=n_runs, groups=GROUPS_OF[dataset])
    if _note:
        print(f"  {_note}")
    work = _assign(GROUPS_OF[dataset], n_runs, workers)
    assert len(work) == workers, (
        f"_assign returned {len(work)} slices for {workers} workers — the partition must "
        "match the request exactly, or the VRAM projection is meaningless")
    print(f"{dataset} {shot}-shot{' +TTA' if _tta else ''}: {len(work)} workers x "
          f"{_per:.2f} GB = {len(work) * _per:.1f} GB of {_cap:.1f} GB")
    for g, a, b in work:
        print(f"   {g[0]:32s} runs [{a}:{b})")

    # See `worker_env` for why the thread caps are load-bearing, and for the measurement
    # that caught three later launchers omitting them. Called inline at the Popen below,
    # not bound to a local: `tests/test_fanout` reads the call site, and an alias is one
    # indirection away from being an alias for something else.

    # Stagger the launches. Peak VRAM is NOT steady state: it happens inside
    # `precompute`, where the 212M-param image encoder runs. Launched together, all N
    # workers hit that encoder phase simultaneously and their transients stack — which is
    # exactly how butterfly OOM'd at 10 workers (39.40 of 39.49 GB, the failing
    # allocation being 72 MB inside hieradet's GELU) while the GPU tracker never showed a
    # sustained spike. Once past precompute a worker drops to cached-decode residency, so
    # offsetting the starts flattens the peak without costing wall clock.
    procs = []
    for i, (g, a, b) in enumerate(work):
        if i:
            _t.sleep(STAGGER_S)
        log = open(f"/content/w{i}.log", "w")
        wcmd = base + list(extra) + ["--groups", *g, "--run-start", str(a),
                                     "--run-end", str(b),
                                     "--out", f"/content/w{i}.zip", "--no-download"]
        print(f"  [{i}] $ {' '.join(wcmd)}  -> {log.name}", flush=True)
        procs.append((i, g[0], a, b, log, _sp.Popen(
            wcmd, stdout=log, stderr=_sp.STDOUT, env=worker_env())))

    # (tag, log, proc) for wait_all, which kills survivors if this cell is interrupted.
    failed = wait_all([(f"{i} ({g} [{a}:{b}))", log, proc)
                       for i, g, a, b, log, proc in procs])
    failed = [f.split()[0] for f in failed]
    assert not failed, f"workers {failed} failed — see /content/w<i>.log above"

    # Report what the workers ACTUALLY used against what was projected. The table above
    # was overstated 2.5x for months because nothing ever compared the two.
    import json as _js
    import glob as _gl

    # Record the FIXED part, not the whole peak. A worker's residency is fixed + its own
    # slice's embedding cache, and the cache depends on how many runs that worker owns — so
    # a peak harvested from a 2-run slice cannot price a 20-run one. Storing the total is
    # what let the planner treat a slice-dependent quantity as a constant. Each worker
    # records its own cache size in `embedding_cache`, so the subtraction is exact rather
    # than modelled.
    peaks, fixeds = [], []
    for f in sorted(_gl.glob("/content/w*.zip")):
        try:
            with __import__("zipfile").ZipFile(f) as _z:
                _rec = _js.loads(_z.read("record.json"))
            v = _rec.get("vram") or {}
            if not v.get("peak_reserved_gb"):
                continue
            peaks.append(v["peak_reserved_gb"])
            _cached = sum((r or {}).get("total_gb", 0.0)
                          for r in (_rec.get("embedding_cache") or {}).values()
                          if isinstance(r, dict))
            fixeds.append(v["peak_reserved_gb"] - _cached)
        except Exception:  # noqa: BLE001
            pass
    if peaks:
        print(f"  measured per-worker VRAM: {min(peaks):.2f}-{max(peaks):.2f} GB "
              f"(projected {_per:.2f}); {len(work)} workers used ~{sum(peaks):.1f} GB "
              f"of {_cap:.1f}")
        # MAX, not mean: the planner must fit the hungriest worker, and a worker still
        # ramping drags the mean down (which is how 2.25 GB happened). Guard against a
        # merge pass, which does zero inference and caches nothing — its "fixed" is just
        # the idle model and would under-size everything after it.
        _fixed = max(fixeds) if fixeds else 0.0
        if _fixed > 1.5:
            save_calibration(dataset, shot, _tta, _fixed)
            print(f"  -> fixed cost {_fixed:.2f} GB (peak minus each worker's own cache) "
                  f"recorded to {CALIBRATION}; the next run sizes itself from this")
        else:
            print(f"  -> fixed cost {_fixed:.2f} GB looks like a no-inference pass; "
                  f"NOT recorded")

    # `extra` on BOTH the workers and the merge. Omitting it here is what made s5c run
    # twice as plain s5b: --tta never reached the subprocesses, so the adapter stayed
    # `sst_sam2`, the resume slug matched s5b's, and every run was served from cache —
    # 0.96 GB peak VRAM, zero inference, and results bit-identical to the arm it was
    # supposed to differ from.
    run(base + list(extra) + ["--no-precompute", "--out", bundle, "--no-download"])

    # Verify the bundle describes the arm that was asked for. Three separate silent
    # misconfigurations have now produced plausible-looking numbers for the wrong
    # experiment; a stage must not be able to report success for a different arm.
    import json as _json

    with __import__("zipfile").ZipFile(bundle) as _z:
        _rec = _json.loads(_z.read("record.json"))
    _got_sel = _rec.get("protocol", {}).get("support_selection")
    assert _got_sel == (selection or "uniform"), (
        f"bundle says support_selection={_got_sel!r}, stage asked for {selection!r}")
    if _tta:
        assert str(_rec.get("adapter", "")).endswith("+tta_rot180"), (
            f"--tta was requested but the bundle's adapter is {_rec.get('adapter')!r} — "
            "the flag did not reach the workers")
        # NOT model.tta_episodes: that is the adapter's live counter, and the merge pass
        # resumes every run from cache and calls predict() zero times, so it is correctly
        # 0 there. The per-run records carry the real counts through the merge.
        # NOT named `_t`: that is the module-level `import time as _t` this function
        # calls for STAGGER_S, and assigning it anywhere in the body makes it local for
        # the WHOLE body — so `_t.sleep` above raises UnboundLocalError before this line
        # is ever reached. Fires only with >1 worker, which is why serial runs missed it.
        _counts = _rec.get("results", {}).get("tta") or {}
        assert _counts.get("episodes"), (
            "TTA adapter is set but no episodes were recorded in any run — the workers "
            "resumed everything from cache, so nothing was actually inferred")
        print(f"  TTA active: rotated support chosen on {_counts['rotated_chosen']}/"
              f"{_counts['episodes']} episodes "
              f"({100 * _counts['rotated_chosen'] / _counts['episodes']:.1f}%)")


BUNDLE = os.path.join(REPO, f"{STAGE}.zip")
if STAGE == "s6c":
    BUNDLE = os.path.join(REPO, "s6c_butterfly_major.zip")
if STAGE == "s8d":
    BUNDLE = os.path.join(REPO, "s8d.zip")
if STAGE == "s8c_tune":
    BUNDLE = os.path.join(REPO, f"s8c_tune_{FT_DATASET}.zip")
if STAGE == "s8c":
    BUNDLE = os.path.join(REPO, f"s8c_{FT_DATASET}.zip")
if STAGE == "s8b":
    # One bundle per learning rate, or the sweep's five runs overwrite each other and only
    # the last survives — with nothing in the surviving record to say the others existed.
    BUNDLE = os.path.join(REPO, f"s8b_{FT_DATASET}_lr{FT_LR:g}.zip")
if STAGE in ("s6", "s6b"):
    # S6 runs one dataset per session and BOTH would otherwise write s6.zip, so the
    # second run would destroy the first. Combine them locally afterwards with
    # scripts/combine_s6.py, which is what the gate reads.
    BUNDLE = os.path.join(REPO, f"{STAGE}_{DATASET_5SHOT}.zip")
if STAGE == "s3":
    # Same reasoning: S3 is now run once per dataset, and a beetle smoke must not
    # overwrite a butterfly one (or read as clearance for it).
    BUNDLE = os.path.join(REPO, f"s3_{S3_DATASET}.zip")
if STAGE in ("s8_5a", "s8_5a2", "s8_5b", "s8_5c"):
    # One bundle per sub-stage per dataset; combined locally, like s6.
    BUNDLE = os.path.join(REPO, f"{STAGE}.zip")
if STAGE == "s5diag":
    # Both resampling arms are run, and they must not overwrite each other on Drive.
    BUNDLE = os.path.join(REPO, f"s5diag_{DIAG_RESAMPLE}.zip")

# subprocess.run(check=True), not `!python`: a `!` command that exits non-zero only prints
# its error and lets the notebook carry on to the export cell, which then ships a stale or
# missing bundle. A failed stage has to stop here.
import subprocess


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


if STAGE == "s6c":
    # Both cells run as ordinary 5-shot fan-outs with --tta; nothing new is needed except
    # the pairing of selection with TTA. Cost is real: TTA doubles inference AND caches
    # both orientations of every image, so `plan_workers(..., tta=True)` sizes it down.
    import shutil

    _cells = {}
    for _sel in S6C_SELECTIONS:
        _out = os.path.join(REPO, f"s6c_butterfly_major_{_sel}.zip")
        print(f"\n=== s6c: 5-shot, {_sel}, + orientation TTA ===\n")
        fanout(stage=f"s6c_{_sel}", dataset="butterfly_major", shot=5, n_runs=20,
               selection=_sel, workers=WORKERS, bundle=_out, extra=["--tta"])
        shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
        _cells[_sel] = _out

    run(["python", "scripts/combine_s6c.py", "--out", BUNDLE,
         *[f"{k}={v}" for k, v in _cells.items()]])

elif STAGE == "s8d":
    # base vs OC-CCL at 5-shot, one dataset at a time, three phases like s8c: train the
    # single arm, score both arms sharing embeddings, then aggregate per arm and pair.
    import json as _json
    import shutil

    _lr, _ep = S8D_ARM
    _lora_dir = "/content/s8d_adapters"
    os.makedirs(_lora_dir, exist_ok=True)
    _per_ds = {}

    for _ds_name in S8D_DATASETS:
        _sel = S8D_SELECTION[_ds_name]
        _pt = f"{_lora_dir}/{_ds_name}_lr{_lr:g}_ep{_ep}.pt"
        _tb = os.path.join(REPO, f"s8b_{_ds_name}_lr{_lr:g}_ep{_ep}.zip")
        print(f"\n=== s8d {_ds_name}: 5-shot, {_sel}, arm (lr={_lr:g}, {_ep} epoch) ===\n")

        if not (S8C_REUSE_ADAPTERS and os.path.exists(_pt)):
            run(["python", "scripts/run_finetune.py",
                 "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                 "--checkpoint", CKPT, "--device", "cuda",
                 "--lr", repr(_lr), "--epochs", str(_ep),
                 "--lora-r", str(FT_LORA_R), "--lora-alpha", str(FT_LORA_ALPHA),
                 "--prompt-mode", FT_PROMPT_MODE,
                 "--lora-out", _pt, "--out", _tb, "--no-download"]
                + (["--no-reset"] if FT_NO_RESET else []))
            if os.path.exists(_tb):
                shutil.copy(_tb, os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
        assert os.path.exists(_pt), f"{_pt} was not written"

        _grp = {"beetle": ["all"],
                "butterfly_major": list(_json.loads(_json.dumps(
                    ["cyrbia", "lativitta", "malleti",
                     "(malleti x plesseni) x malleti", "notabilis x lativitta"])))}[_ds_name]
        _armspec = ["--arms", "base=", "--arms", f"occl={_pt}"]
        # One pass per TTA setting. Within a pass the two weight arms share the encode;
        # across passes they cannot, because the TTA pass caches both orientations and
        # scores under a different `adapter.name`. Two passes, not four processes.
        #
        # GROUPS ONE AT A TIME, splitting each group's RUNS across the workers — not
        # `_assign`, which hands whole groups out and cannot balance. Every group has 20 runs
        # and 100 test images, so equal run-slices are equal work by construction. Measured
        # on this stage's own plain pass: `_assign`'s [10,10,20,20,20,20] left two workers
        # idle from minute 26 onward, **24% of the provisioned card**. At 3 workers for TTA
        # it splits the five groups 2/1/2 and is worse still.
        #
        # Throughput is linear in worker count here — losing two workers mid-pass left the
        # survivors' rate unchanged at 0.341 vs 0.340 runs/min — so idle workers are a
        # straight loss, not slack that the others absorb.
        for _use_tta in S8D_COMPUTE_CELLS:
            _tag = "tta" if _use_tta else "plain"
            # Plan PER PASS: the TTA pass holds both orientations and runs two predictions
            # per episode, so it fits fewer workers. Planning once for both (and with
            # tta=False) is what put 11 workers at 3.67 GB onto a 39.5 GB card.
            _w, _per, _cap, _note = plan_group_workers(
                _ds_name, 5, tta=_use_tta, n_runs=20, requested=S8D_WORKERS_PER_GROUP)
            print(f"\n  === {_ds_name}: {_tag} pass, {len(_grp)} group(s) in sequence, "
                  f"{_note} ===", flush=True)
            for _gi, _group in enumerate(_grp):
                _slices = run_slices(20, _w)
                print(f"\n  --- {_group}: {len(_slices)} workers x 2 arms, runs "
                      f"{[f'[{a}:{b})' for a, b in _slices]} ---", flush=True)
                _procs = []
                for _i, (_a, _b) in enumerate(_slices):
                    _log = open(f"/content/s8d_{_ds_name}_{_tag}_g{_gi}_{_i}.log", "w")
                    _wcmd = ["python", "scripts/run_arms.py", "--dataset", _ds_name,
                         "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                         "--shot", "5", "--n-runs", "20", "--support-selection", _sel,
                         "--lora-r", str(FT_LORA_R), "--lora-alpha", str(FT_LORA_ALPHA),
                         "--resume-dir", f"{RESULTS_ROOT}/runs",
                         "--groups", _group, "--run-start", str(_a), "--run-end", str(_b)] \
                        + (["--gpu-frames"] if GPU_FRAMES else []) \
                        + _armspec + (["--tta"] if _use_tta else [])
                    print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}", flush=True)
                    _procs.append((_i, _log, subprocess.Popen(
                        _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
                        env=worker_env())))
                    _t.sleep(STAGGER_S)
                _bad = wait_all(_procs, f"{_ds_name} {_group} {_tag} worker")
                assert not _bad, (f"{_ds_name}/{_group} {_tag} workers {_bad} failed — see "
                                  f"/content/s8d_{_ds_name}_{_tag}_g{_gi}_*.log")

        # Merge every cell, not just the computed ones: a merge is inference-free, so a
        # pass that was skipped because it is already banked still produces its bundle.
        _bundles = {}
        for _use_tta in S8D_TTA_CELLS:
            _tag = "tta" if _use_tta else "plain"
            for _label, _p2 in (("base", None), ("occl", _pt)):
                _cell = _label if not _use_tta else f"{_label}_tta"
                _out = os.path.join(REPO, f"s8d_{_ds_name}_{_cell}.zip")
                run(["python", "scripts/run_eval.py", "--stage", f"s8d_{_ds_name}_{_cell}",
                     "--adapter", "sst_sam2", "--dataset", _ds_name,
                     "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                     "--shot", "5", "--n-runs", "20", "--support-selection", _sel,
                     "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
                     "--out", _out, "--no-download"]
                    + (["--gpu-frames"] if GPU_FRAMES else [])
                    + (["--tta"] if _use_tta else [])
                    + (["--lora", _p2, "--lora-epochs", str(_ep),
                        "--lora-prompt-mode", FT_PROMPT_MODE]
                       + (["--lora-no-reset"] if FT_NO_RESET else []) if _p2 else []))
                _bundles[_cell] = _out

        # Two paired comparisons per dataset: OC-CCL within the plain stack and within the
        # TTA stack. Each pairs against ITS OWN base — comparing the OC-CCL+TTA arm against
        # the plain base would fold TTA's effect into OC-CCL's.
        _dsb = os.path.join(REPO, f"s8d_{_ds_name}.zip")
        run(["python", "scripts/combine_s8d_cells.py", "--out", _dsb,
             *[f"{k}={v}" for k, v in _bundles.items()]])
        shutil.copy(_dsb, os.path.join(RESULTS_ROOT, os.path.basename(_dsb)))
        _per_ds[_ds_name] = _dsb

    if len(_per_ds) < 2:
        print(f"\nNOTE: only {sorted(_per_ds)} ran. The combined bundle will be "
              "incomplete and\n  the s8d gate will refuse it for missing a dataset — "
              "which is correct. Combine\n  this with the other dataset's "
              "s8d_<name>.zip once both exist:\n"
              "    python scripts/combine_s8d.py --out s8d.zip "
              "beetle=<a>.zip butterfly_major=<b>.zip")
    run(["python", "scripts/combine_s8d.py", "--out", BUNDLE,
         *[f"{k}={v}" for k, v in _per_ds.items()]])

elif STAGE == "s8c_tune":
    # One process per fold, all concurrent. Each holds a training run (2.33 GB) plus its
    # own embedding cache for a 24-image validation fold — small, so k=5 fits easily.
    # Folds are independent by construction: disjoint validation sets, and the split is
    # seeded from (dataset, species, k) rather than from anything the run computes.
    import shutil

    _procs = []
    for _f in range(S8C_TUNE_K):
        _log = open(f"/content/tune{_f}.log", "w")
        _cmd = ["python", "scripts/run_tune.py",
                "--dataset", FT_DATASET, "--cache-root", LOCAL_CACHE,
                "--checkpoint", CKPT, "--device", "cuda",
                "--k", str(S8C_TUNE_K), "--fold", str(_f),
                "--epochs", str(S8C_TUNE_EPOCHS),
                "--lrs", *[repr(x) for x in S8C_TUNE_LRS],
                "--lora-r", str(FT_LORA_R), "--lora-alpha", str(FT_LORA_ALPHA),
                "--prompt-mode", FT_PROMPT_MODE,
                "--n-runs", str(S8C_TUNE_RUNS),
                "--support-selection", "canonical_balanced",
                "--out", os.path.join(REPO, f"tune_f{_f}.zip")] \
               + (["--no-reset"] if FT_NO_RESET else [])
        print("$", " ".join(_cmd))
        _procs.append((_f, _log, subprocess.Popen(
            _cmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
            env=worker_env())))
        _t.sleep(STAGGER_S)

    _bad = wait_all(_procs, "fold")
    # Every fold or none: a mean over the folds that happened to finish is biased toward
    # them, and nothing in the combined record would show it.
    assert not _bad, f"folds {_bad} failed — see /content/tune*.log"

    run(["python", "scripts/combine_tune.py", "--out", BUNDLE,
         *[os.path.join(REPO, f"tune_f{f}.zip") for f in range(S8C_TUNE_K)]])

elif STAGE == "s8c":
    # Train, then score everything, in one session. Two reasons it is one stage rather
    # than two: the adapters never touch Drive, so nothing can be stale or missing; and
    # the baseline is produced by the same code, on the same GPU, in the same hour as the
    # treatments, which is what makes `compare()` willing to pair them at all.
    #
    # Both phases are parallel, for different reasons. Training is tiny (2.33 GB measured)
    # so every arm trains at once. Evaluation is the expensive half, and its saving is
    # structural rather than just wider: LoRA never touches the image encoder, so all arms
    # share one set of embeddings and one worker scores all three for barely more than the
    # cost of one.
    # Imported here, not relied on from another cell: `shutil` is bound by the EXPORT
    # cell (which runs after this one) and `_json` only inside the cache-staging block.
    # A NameError from either would land after the training has already been paid for.
    import json as _json
    import shutil

    _lora_dir = "/content/s8c_adapters"
    os.makedirs(_lora_dir, exist_ok=True)
    _arms = []          # (label, lora_path_or_None)

    # ── phase 1: train every arm CONCURRENTLY ────────────────────────────────
    # Training measured 2.33 GB peak, so N arms cost 2.33N GB — five would still fit in a
    # quarter of the A100. They are independent processes writing to different files, so
    # this is pure wall-clock saving: 2 arms x 10 epochs goes from ~14 min to ~7.
    # Resolve the arms before anything trains, and print what was resolved: a run that
    # silently scored different hyperparameters than intended would be indistinguishable
    # from one that worked.
    if S8C_ARMS == "from_tune":
        _tz = os.path.join(RESULTS_ROOT, f"s8c_tune_{FT_DATASET}.zip")
        if not os.path.exists(_tz):
            _tz = os.path.join(REPO, f"s8c_tune_{FT_DATASET}.zip")
        assert os.path.exists(_tz), (
            f"S8C_ARMS='from_tune' but no tune bundle at {_tz}.\n"
            "  Run STAGE='s8c_tune' first — the hyperparameters must be chosen on held-out\n"
            "  train folds BEFORE the test split is scored. Or set S8C_ARMS explicitly.")
        import zipfile as _zf2
        with _zf2.ZipFile(_tz) as _z2:
            _tr = _json.loads(_z2.read("record.json"))
        _clr, _cep = _tr["results"]["chosen_lr"], _tr["results"]["chosen_epoch"]
        assert _cep > 0, (
            "the tuning chose the BASE model — no (lr, epoch) beat it on held-out data.\n"
            "  That is a result, not a configuration: report it rather than scoring a\n"
            "  fine-tune the cross-validation rejected.")
        _alr, _aep = _tr["results"]["argmax_lr"], _tr["results"]["argmax_epoch"]
        # Three arms: the released default, the tie-break's cheapest point, and the raw
        # argmax. The last is what tests the plateau rather than assuming it — the
        # tie-break asserts those two points are equivalent, and the only way to know
        # whether that holds on the real split is to score both.
        _specs = [(1e-5, 10), (_clr, _cep)]
        if (_alr, _aep) != (_clr, _cep):
            _specs.append((_alr, _aep))
        print(f"[s8c] arms from {os.path.basename(_tz)}:\n"
              f"    SST default   (1e-5, 10ep)\n"
              f"    tuned/cheapest ({_clr:g}, {_cep}ep)  <- the tie-break's pick\n"
              + (f"    cv argmax      ({_alr:g}, {_aep}ep)  <- tests the plateau claim\n"
                 if (_alr, _aep) != (_clr, _cep) else "")
              + f"  {_tr['results']['n_within_one_sem']}/"
                f"{_tr['results']['n_grid_points']} grid points were within one sem, so "
                f"the cv could not separate them")
    else:
        _specs = [tuple(a) for a in S8C_ARMS]
        print(f"[s8c] arms: {[(f'{lr:g}', ep) for lr, ep in _specs]}")

    _todo = [(lr, ep, f"{_lora_dir}/lora_lr{lr:g}_ep{ep}.pt",
              os.path.join(REPO, f"s8b_{FT_DATASET}_lr{lr:g}_ep{ep}.zip"))
             for lr, ep in _specs]
    _pending = [t for t in _todo if not (S8C_REUSE_ADAPTERS and os.path.exists(t[2]))]
    if _pending:
        print(f"\n[s8c] training {len(_pending)} arm(s) concurrently, "
              f"{FT_EPOCHS} epoch(s) each (~2.33 GB apiece)\n")
        _procs = []
        for _i, (_lr, _ep, _pt, _tb) in enumerate(_pending):
            _log = open(f"/content/ft{_i}.log", "w")
            _cmd = ["python", "scripts/run_finetune.py",
                    "--dataset", FT_DATASET, "--cache-root", LOCAL_CACHE,
                    "--checkpoint", CKPT, "--device", "cuda",
                    "--lr", repr(_lr), "--epochs", str(_ep),
                    "--lora-r", str(FT_LORA_R), "--lora-alpha", str(FT_LORA_ALPHA),
                    "--prompt-mode", FT_PROMPT_MODE,
                    "--lora-out", _pt, "--out", _tb, "--no-download"] \
                   + (["--no-reset"] if FT_NO_RESET else [])
            print("$", " ".join(_cmd))
            _procs.append(((_lr, _ep), _log, subprocess.Popen(
                _cmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
                env=worker_env())))
            _t.sleep(STAGGER_S)
        _bad = wait_all(_procs, "train")
        assert not _bad, f"training failed for {_bad} — see /content/ft*.log"
    for _lr, _ep, _pt, _tb in _todo:
        assert os.path.exists(_pt), f"{_pt} was not written by run_finetune"
        if os.path.exists(_tb):
            # Keep the training bundles: they carry the loss curves, and a paired delta
            # with no training record behind it cannot be interpreted later.
            shutil.copy(_tb, os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
        _arms.append((f"lora_lr{_lr:g}_ep{_ep}", _pt, _ep))
    _arms.insert(0, ("base", None, 0))

    # ── phase 2: score every arm, sharing the image embeddings ───────────────
    # LoRA never touches the image encoder, so all arms have IDENTICAL backbone features.
    # Encoding is ~95% of a run's compute, so one worker doing all three arms costs barely
    # more than one arm — where three separate fan-outs would pay that 95% three times.
    # Each worker owns a run slice and writes per-run JSON for every arm into the shared
    # resume dir; the weights fingerprint keeps the arms' caches distinct.
    _n_runs = 20
    _w, _per, _cap, _note = plan_workers(FT_DATASET, 1, WORKERS, tta=False,
                                         n_runs=_n_runs, groups=GROUPS_OF[FT_DATASET])
    if _note:
        print(_note)
    _edges = [round(i * _n_runs / _w) for i in range(_w + 1)]
    _slices = [(a, b) for a, b in zip(_edges, _edges[1:]) if b > a]
    print(f"\n[s8c] {len(_arms)} arms x {_n_runs} runs across {len(_slices)} workers "
          f"({_per:.2f} GB each -> {len(_slices) * _per:.1f} of {_cap:.1f} GB)\n")

    _armspec = []
    for _label, _pt, _ep in _arms:
        _armspec += ["--arms", f"{_label}={_pt or ''}"]

    _procs = []
    for _i, (_a, _b) in enumerate(_slices):
        _log = open(f"/content/arm{_i}.log", "w")
        _procs.append((_i, _a, _b, _log, subprocess.Popen(
            ["python", "scripts/run_arms.py", "--dataset", FT_DATASET,
             "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
             "--shot", "1", "--n-runs", str(_n_runs),
             "--support-selection", "canonical_balanced",
             "--lora-r", str(FT_LORA_R), "--lora-alpha", str(FT_LORA_ALPHA),
             "--resume-dir", f"{RESULTS_ROOT}/runs",
             "--run-start", str(_a), "--run-end", str(_b)]
            + (["--gpu-frames"] if GPU_FRAMES else []) + _armspec,
            cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
            env=worker_env())))
        _t.sleep(STAGGER_S)
    _bad = []
    for _i, _a, _b, _log, _p in _procs:
        _rc = _p.wait()
        _log.close()
        print(f"  worker {_i} (runs [{_a}:{_b})) exit {_rc}")
        if _rc != 0:
            _bad.append(_i)
            print(open(_log.name).read()[-2500:])
    assert not _bad, f"arm workers {_bad} failed — see /content/arm*.log"

    # ── phase 3: aggregate, per arm. Zero inference — every run is on disk. ──
    _bundles = {}
    for _label, _pt, _ep in _arms:
        _out = os.path.join(REPO, f"s8c_{FT_DATASET}_{_label}.zip")
        run(["python", "scripts/run_eval.py", "--stage", f"s8c_{_label}",
             "--adapter", "sst_sam2", "--dataset", FT_DATASET,
             "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
             "--shot", "1", "--n-runs", str(_n_runs),
             "--support-selection", "canonical_balanced",
             "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
             "--out", _out, "--no-download"]
            + (["--lora", _pt, "--lora-epochs", str(_ep),
                "--lora-prompt-mode", FT_PROMPT_MODE]
               + (["--lora-no-reset"] if FT_NO_RESET else []) if _pt else []))
        _bundles[_label] = _out

    run(["python", "scripts/combine_s8c.py", "--out", BUNDLE,
         *[f"{k}={v}" for k, v in _bundles.items()]])

elif STAGE == "s8b":
    # OC-CCL fine-tune, one epoch at one learning rate. Serial by construction: this is a
    # single training loop, not 20 independent runs, so none of the fan-out machinery
    # applies and `WORKERS` is ignored.
    #
    # VRAM is NOT the evaluation stages' ~4.86 GB/worker. The cycle runs 2 image encodes
    # and 4 tracker passes *with gradient*, so activations are retained across all of it;
    # expect several times the inference figure. The first run of the sweep is the probe —
    # read `training.peak_vram_gb` off its record before assuming anything about the rest.
    cmd = ["python", "scripts/run_finetune.py",
           "--dataset", FT_DATASET, "--cache-root", LOCAL_CACHE,
           "--checkpoint", CKPT, "--device", "cuda",
           "--lr", repr(FT_LR), "--epochs", str(FT_EPOCHS),
           "--lora-r", str(FT_LORA_R), "--lora-alpha", str(FT_LORA_ALPHA),
           "--prompt-mode", FT_PROMPT_MODE,
           "--out", BUNDLE, "--no-download"]
    if FT_NO_RESET:
        cmd += ["--no-reset"]
    if FT_MAX_STEPS:
        cmd += ["--max-steps", str(FT_MAX_STEPS)]
    run(cmd)

elif STAGE in ("s8_5a", "s8_5b", "s8_5c"):
    # Train once per dataset with SST's OWN method, then score base vs occl under this
    # sub-stage's protocol. Same three-phase shape as s8d; the only differences are
    # `--method full` and the (shot, tta) pair.
    import json as _json
    import shutil

    _shot, _tta = {"s8_5a": (1, False), "s8_5b": (1, True), "s8_5c": (5, True)}[STAGE]
    _ft_dir = "/content/s8_5_adapters"
    os.makedirs(_ft_dir, exist_ok=True)
    _per_ds = {}

    for _ds_name in S8_5_DATASETS:
        _sel = S8_5_SELECTION[_ds_name]
        # ONE checkpoint per dataset, shared by a/b/c: they score the same trained model
        # and differ only in the inference protocol. Naming it after the training config
        # rather than the stage is what makes that sharing safe and visible.
        # The prompt mode MUST be in the filename. Without it, flipping S8_5_PROMPT_MODES
        # and re-running would find the old file, skip training under S8_5_REUSE, and score
        # the logit_reparam weights while the record claimed hard_threshold — a silent swap
        # of the very variable this stage exists to measure.
        _pts = {_m: f"{_ft_dir}/{_ds_name}_full_lr{S8_5_LR:g}_ep{S8_5_EPOCHS}_{_m}.pt"
                for _m in S8_5_PROMPT_MODES}
        print(f"\n=== {STAGE} {_ds_name}: {_shot}-shot"
              f"{' + TTA' if _tta else ''}, {_sel}, FULL fine-tune "
              f"(lr={S8_5_LR:g}, {S8_5_EPOCHS} epochs) ===\n", flush=True)

        for _m, _pt in _pts.items():
            _tb = os.path.join(REPO, f"s8_5ft_{_ds_name}_{_m}.zip")
            _reuse = S8_5_REUSE and os.path.exists(_pt)
            if _reuse:
                # Trust the checkpoint's own metadata, not its name. `run_finetune` writes
                # the `occl` block into it precisely so this check is possible.
                import torch as _torch
                _meta = (_torch.load(_pt, map_location="cpu",
                                     weights_only=False).get("meta") or {})
                _got = (_meta.get("occl") or {}).get("prompt_mode")
                if _got != _m or _meta.get("epochs") != S8_5_EPOCHS:
                    print(f"  {os.path.basename(_pt)} records prompt_mode={_got!r} "
                          f"epochs={_meta.get('epochs')!r}, wanted {_m!r}/{S8_5_EPOCHS} "
                          f"— retraining", flush=True)
                    _reuse = False
            if not _reuse:
                run(["python", "scripts/run_finetune.py",
                     "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                     "--checkpoint", CKPT, "--device", "cuda",
                     "--method", "full",                  # <- the whole point of s8.5
                     "--lr", repr(S8_5_LR), "--epochs", str(S8_5_EPOCHS),
                     "--prompt-mode", _m,
                     "--lora-out", _pt, "--out", _tb, "--no-download"]
                    + (["--no-reset"] if FT_NO_RESET else []))
                if os.path.exists(_tb):
                    shutil.copy(_tb, os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
                # ~47 MB against LoRA's ~4, and the VM dies with the session.
                shutil.copy(_pt, os.path.join(RESULTS_ROOT, os.path.basename(_pt)))
            assert os.path.exists(_pt), f"{_pt} was not written"

        _grp = GROUPS_OF[_ds_name]
        # All arms in ONE pass: they share an image encoder, so scoring them separately
        # would pay the encode once per arm for byte-identical features.
        _armspec = ["--arms", "base="]
        for _m, _p in _pts.items():
            _armspec += ["--arms", f"occl_{_m.split('_')[0]}={_p}"]
        _w, _per, _cap, _note = plan_group_workers(
            _ds_name, _shot, tta=_tta, n_runs=20, requested=S8D_WORKERS_PER_GROUP)
        print(f"\n  === {_ds_name}: {len(_grp)} group(s) in sequence, {_note} ===",
              flush=True)
        for _gi, _group in enumerate(_grp):
            _slices = run_slices(20, _w)
            print(f"\n  --- {_group}: {len(_slices)} workers x 2 arms, runs "
                  f"{[f'[{a}:{b})' for a, b in _slices]} ---", flush=True)
            _procs = []
            for _i, (_a, _b) in enumerate(_slices):
                _log = open(f"/content/{STAGE}_{_ds_name}_g{_gi}_{_i}.log", "w")
                _wcmd = ["python", "scripts/run_arms.py", "--dataset", _ds_name,
                         "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                         "--shot", str(_shot), "--n-runs", "20",
                         "--support-selection", _sel,
                         "--resume-dir", f"{RESULTS_ROOT}/runs",
                         "--groups", _group,
                         "--run-start", str(_a), "--run-end", str(_b)] \
                    + (["--gpu-frames"] if GPU_FRAMES else []) \
                    + _armspec + (["--tta"] if _tta else [])
                print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}", flush=True)
                _procs.append((_i, _log, subprocess.Popen(
                    _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
                    env=worker_env())))
                _t.sleep(STAGGER_S)
            _bad = wait_all(_procs, f"{STAGE} {_ds_name} {_group} worker")
            assert not _bad, (f"{_ds_name}/{_group} workers {_bad} failed — see "
                              f"/content/{STAGE}_{_ds_name}_g{_gi}_*.log")

        # Merge each arm (inference-free) and pair them.
        _cells = {}
        _arm_paths = [("base", None)] + [(f"occl_{_m.split('_')[0]}", _p)
                                         for _m, _p in _pts.items()]
        for _label, _p2 in _arm_paths:
            _out = os.path.join(REPO, f"{STAGE}_{_ds_name}_{_label}.zip")
            run(["python", "scripts/run_eval.py", "--stage", f"{STAGE}_{_ds_name}_{_label}",
                 "--adapter", "sst_sam2", "--dataset", _ds_name,
                 "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                 "--shot", str(_shot), "--n-runs", "20", "--support-selection", _sel,
                 "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
                 "--out", _out, "--no-download"]
                + (["--gpu-frames"] if GPU_FRAMES else [])
                + (["--tta"] if _tta else [])
                + (["--lora", _p2, "--lora-epochs", str(S8_5_EPOCHS),
                    "--lora-prompt-mode", FT_PROMPT_MODE]
                   + (["--lora-no-reset"] if FT_NO_RESET else []) if _p2 else []))
            _cells[_label if not _tta else f"{_label}_tta"] = _out
        _dsb = os.path.join(REPO, f"{STAGE}_{_ds_name}.zip")
        run(["python", "scripts/combine_s8d_cells.py", "--out", _dsb, "--stage", STAGE,
             *[f"{k}={v}" for k, v in _cells.items()]])
        shutil.copy(_dsb, os.path.join(RESULTS_ROOT, os.path.basename(_dsb)))
        _per_ds[_ds_name] = _dsb

    run(["python", "scripts/combine_s8d.py", "--out", BUNDLE, "--stage", STAGE,
         *[f"{k}={v}" for k, v in _per_ds.items()]])

elif STAGE == "s8_5a2":
    # Beetle, 1-shot, plain, FULL fine-tune — every remaining defect axis as its own arm,
    # all scored in one pass so they share the image encoder. Beetle only and 1-shot only:
    # it is the cheapest protocol we have (critical path ~4 run-units for two arms) and it
    # is the dataset the paper's +3.3 was claimed on.
    import shutil

    _ds_name, _sel, _shot = "beetle", "canonical_balanced", 1
    _ft_dir = "/content/s8_5_adapters"
    os.makedirs(_ft_dir, exist_ok=True)

    def _cfg(over):
        """Full training config for one arm: our defaults, then this arm's overrides."""
        return {"prompt_mode": FT_PROMPT_MODE, "train_mode": False,
                "normalization": "imagenet", "no_reset": FT_NO_RESET, **over}

    _pts, _todo = {}, []
    for _label, _over in S8_5A2_ARMS.items():
        c = _cfg(_over)
        # EVERY axis in the filename. The s8_5a reuse trap was exactly this: a name that
        # omits an axis lets `S8_5_REUSE` hand one arm another arm's weights.
        _tag = (f"{c['prompt_mode']}_{c['normalization']}"
                f"_{'train' if c['train_mode'] else 'eval'}"
                f"_{'noreset' if c['no_reset'] else 'reset'}")
        _pt = f"{_ft_dir}/{_ds_name}_full_lr{S8_5_LR:g}_ep{S8_5_EPOCHS}_{_tag}.pt"
        _pts[_label] = _pt
        _tb = os.path.join(REPO, f"s8_5ft_{_ds_name}_{_label}.zip")

        _reuse = S8_5_REUSE and os.path.exists(_pt)
        if _reuse:
            import torch as _torch
            _meta = (_torch.load(_pt, map_location="cpu",
                                 weights_only=False).get("meta") or {})
            _o = _meta.get("occl") or {}
            _want = {"prompt_mode": c["prompt_mode"], "train_mode": c["train_mode"],
                     "normalization": c["normalization"],
                     "memory_reset": not c["no_reset"]}
            _got = {k: _o.get(k) for k in _want}
            if _got != _want or _meta.get("epochs") != S8_5_EPOCHS:
                print(f"  {os.path.basename(_pt)} records {_got} / epochs="
                      f"{_meta.get('epochs')!r}, wanted {_want} / {S8_5_EPOCHS} "
                      f"— retraining", flush=True)
                _reuse = False
        if not _reuse:
            print(f"  [{_label}] queued: {c}", flush=True)
            _todo.append((_label, [
                "python", "scripts/run_finetune.py",
                "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                "--checkpoint", CKPT, "--device", "cuda",
                "--method", "full",
                "--lr", repr(S8_5_LR), "--epochs", str(S8_5_EPOCHS),
                "--prompt-mode", c["prompt_mode"],
                "--normalization", c["normalization"],
                "--lora-out", _pt, "--out", _tb, "--no-download"]
                + (["--train-mode"] if c["train_mode"] else [])
                + (["--no-reset"] if c["no_reset"] else []),
                f"/content/s8_5a2_train_{_label}.log", _tb))

    # All four fine-tunes at once. ~2.5 GB each against a 42 GB card, and they share nothing
    # but the GPU — separate processes, separate models, pair order from a pure hash. Serial
    # this is ~29 min on beetle (43 s/epoch x 10 x 4); concurrent it is roughly a third.
    if _todo:
        print(f"\n=== s8_5a2: training {len(_todo)} arm(s) concurrently ===\n", flush=True)
        train_parallel([(lab, cmd, log) for lab, cmd, log, _tb in _todo], "s8_5a2 train")
        for _label, _cmd, _log, _tb in _todo:
            if os.path.exists(_tb):
                shutil.copy(_tb, os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
    for _label, _pt in _pts.items():
        assert os.path.exists(_pt), f"{_pt} was not written"
        shutil.copy(_pt, os.path.join(RESULTS_ROOT, os.path.basename(_pt)))

    _armspec = ["--arms", "base="]
    for _label, _pt in _pts.items():
        _armspec += ["--arms", f"{_label}={_pt}"]

    _w, _per, _cap, _note = plan_group_workers(_ds_name, _shot, tta=False, n_runs=20,
                                               requested=S8D_WORKERS_PER_GROUP)
    print(f"\n  === {_ds_name}: {len(_pts) + 1} arms, {_note} ===", flush=True)
    _slices = run_slices(20, _w)
    _procs = []
    for _i, (_a, _b) in enumerate(_slices):
        _log = open(f"/content/s8_5a2_{_i}.log", "w")
        _wcmd = ["python", "scripts/run_arms.py", "--dataset", _ds_name,
                 "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                 "--shot", str(_shot), "--n-runs", "20", "--support-selection", _sel,
                 "--resume-dir", f"{RESULTS_ROOT}/runs",
                 "--groups", "all",
                 "--run-start", str(_a), "--run-end", str(_b)] \
            + (["--gpu-frames"] if GPU_FRAMES else []) + _armspec
        print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}", flush=True)
        _procs.append((_i, _log, subprocess.Popen(
            _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT, env=worker_env())))
        _t.sleep(STAGGER_S)
    _bad = wait_all(_procs, "s8_5a2 worker")
    assert not _bad, f"workers {_bad} failed — see /content/s8_5a2_*.log"

    _cells = {}
    for _label, _pt in [("base", None)] + list(_pts.items()):
        _out = os.path.join(REPO, f"s8_5a2_{_label}.zip")
        _c = _cfg(S8_5A2_ARMS.get(_label, {}))
        run(["python", "scripts/run_eval.py", "--stage", f"s8_5a2_{_label}",
             "--adapter", "sst_sam2", "--dataset", _ds_name,
             "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
             "--shot", str(_shot), "--n-runs", "20", "--support-selection", _sel,
             "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
             "--out", _out, "--no-download"]
            + (["--gpu-frames"] if GPU_FRAMES else [])
            + (["--lora", _pt, "--lora-epochs", str(S8_5_EPOCHS),
                "--lora-prompt-mode", _c["prompt_mode"],
                "--lora-normalization", _c["normalization"]]
               + (["--lora-no-reset"] if _c["no_reset"] else [])
               + (["--lora-train-mode"] if _c["train_mode"] else []) if _pt else []))
        # To DRIVE, not just /content. The per-arm bundles are the combine's only inputs,
        # and s8_5a2 kept them on the VM alone — so when the runtime was lost, re-running
        # the combine meant rebuilding all six merges from the banked per-run JSON with a
        # fresh GPU session. They are a few hundred KB each; the VM is not a storage tier.
        shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
        # To DRIVE, not just /content. These are the combine's only inputs, and s8_5a2 kept
        # them on the VM alone — so a lost runtime turned a re-combine into six GPU merges.
        shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
        _cells[_label] = _out
    run(["python", "scripts/combine_s8d_cells.py", "--out", BUNDLE, "--stage", STAGE,
         *[f"{k}={v}" for k, v in _cells.items()]])

elif STAGE == "s10":
    # The paper's OC-CCL (see the S10_ARMS block for the five release-vs-paper
    # contradictions and the pre-registered predictions). Beetle, 1-shot, plain,
    # canonical_balanced — the exact protocol the paper's +3.3 was claimed on and the
    # one where S4c reproduces the base to 63.20.
    import shutil

    _ds_name, _sel, _shot = "beetle", "canonical_balanced", 1
    _ft_dir = "/content/s10_adapters"
    os.makedirs(_ft_dir, exist_ok=True)

    def _s10_cfg(over):
        """One arm's full axis set: paper composite defaults, then the arm's override."""
        return {"mask_mode": "per_trait", "aug": True, "reference_run": None,
                "prompt_mode": "logit_reparam", "normalization": "imagenet",
                "train_mode": False, "no_reset": False, **over}

    _pts, _todo = {}, []
    for _label, _over in S10_ARMS.items():
        c = _s10_cfg(_over)
        # EVERY axis in the filename — the s8_5a reuse trap. A name that omits an axis
        # lets reuse hand one arm another arm's weights.
        _tag = (f"{c['mask_mode']}"
                f"_{f'aug{S10_AUG_DEG:g}' if c['aug'] else 'noaug'}"
                f"_{('ref%d' % c['reference_run']) if c['reference_run'] is not None else 'derange'}"
                f"_{c['prompt_mode']}_{c['normalization']}"
                f"_{'train' if c['train_mode'] else 'eval'}"
                f"_{'noreset' if c['no_reset'] else 'reset'}")
        _pt = f"{_ft_dir}/{_ds_name}_lora_lr{S10_LR:g}_ep{S10_EPOCHS}_{_tag}.pt"
        _pts[_label] = _pt
        _tb = os.path.join(REPO, f"s10ft_{_label}.zip")

        _reuse = S10_REUSE and os.path.exists(_pt)
        if _reuse:
            import torch as _torch
            _meta = (_torch.load(_pt, map_location="cpu",
                                 weights_only=False).get("meta") or {})
            _o = _meta.get("occl") or {}
            _aug = _o.get("augmentation") or {}
            _want = {"mask_mode": c["mask_mode"],
                     "reference_run": c["reference_run"],
                     "aug_deg": S10_AUG_DEG if c["aug"] else None,
                     "prompt_mode": c["prompt_mode"],
                     "normalization": c["normalization"],
                     "train_mode": c["train_mode"],
                     "memory_reset": not c["no_reset"]}
            _got = {"mask_mode": _o.get("mask_mode"),
                    "reference_run": _o.get("reference_run"),
                    "aug_deg": _aug.get("max_deg"),
                    "prompt_mode": _o.get("prompt_mode"),
                    "normalization": _o.get("normalization"),
                    "train_mode": _o.get("train_mode"),
                    "memory_reset": _o.get("memory_reset")}
            if _got != _want or _meta.get("epochs") != S10_EPOCHS:
                print(f"  {os.path.basename(_pt)} records {_got} / epochs="
                      f"{_meta.get('epochs')!r}, wanted {_want} / {S10_EPOCHS} "
                      f"— retraining", flush=True)
                _reuse = False
        if not _reuse:
            print(f"  [{_label}] queued: {c}", flush=True)
            _todo.append((_label, [
                "python", "scripts/run_finetune.py",
                "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                "--checkpoint", CKPT, "--device", "cuda",
                "--method", "lora",                      # <- the paper's A2.2 method
                "--lr", repr(S10_LR), "--epochs", str(S10_EPOCHS),
                "--mask-mode", c["mask_mode"],
                "--prompt-mode", c["prompt_mode"],
                "--normalization", c["normalization"],
                "--lora-out", _pt, "--out", _tb, "--no-download"]
                + (["--reference-run", str(c["reference_run"]),
                    "--reference-selection", _sel]
                   if c["reference_run"] is not None else [])
                + (["--rotation-aug-deg", repr(S10_AUG_DEG),
                    "--rotation-aug-rule", S10_AUG_RULE] if c["aug"] else [])
                + (["--train-mode"] if c["train_mode"] else [])
                + (["--no-reset"] if c["no_reset"] else []),
                f"/content/s10_train_{_label}.log", _tb))

    # Chunked, not all-at-once: a per-trait cycle runs C≈5 objects through every leg,
    # so each training process is ~10 GB estimated (bound 13.7) where C=1 measured 2.75.
    if _todo:
        print(f"\n=== s10: training {len(_todo)} arm(s), "
              f"{S10_TRAIN_CONCURRENCY} at a time ===\n", flush=True)
        for _k in range(0, len(_todo), S10_TRAIN_CONCURRENCY):
            train_parallel([(lab, cmd, log) for lab, cmd, log, _tb in
                            _todo[_k:_k + S10_TRAIN_CONCURRENCY]], "s10 train")
        for _label, _cmd, _log, _tb in _todo:
            if os.path.exists(_tb):
                shutil.copy(_tb, os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
    for _label, _pt in _pts.items():
        assert os.path.exists(_pt), f"{_pt} was not written"
        shutil.copy(_pt, os.path.join(RESULTS_ROOT, os.path.basename(_pt)))

    # The anchored arm: 20 per-run fine-tunes, each on that run's own support, scored
    # run-for-run and merged — one process, one script, its own bundle (which is also
    # its arm cell for the combine). Runs after the fixed trainings so the card is free.
    _anch = os.path.join(REPO, "s10_occl_anchored.zip")
    run(["python", "scripts/run_anchored.py",
         "--dataset", _ds_name, "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
         "--n-runs", "20", "--support-selection", _sel,
         "--resume-dir", f"{RESULTS_ROOT}/runs",
         "--adapter-dir", f"{_ft_dir}/anchored",
         "--lr", repr(S10_LR), "--epochs", str(S10_EPOCHS),
         "--mask-mode", "per_trait",
         "--rotation-aug-deg", repr(S10_AUG_DEG), "--rotation-aug-rule", S10_AUG_RULE,
         "--stage", "s10_occl_anchored", "--out", _anch, "--no-download"]
        + (["--gpu-frames"] if GPU_FRAMES else []))
    shutil.copy(_anch, os.path.join(RESULTS_ROOT, os.path.basename(_anch)))

    # base + the six fixed arms in one fan-out, sharing each worker's encode.
    _armspec = ["--arms", "base="]
    for _label, _pt in _pts.items():
        _armspec += ["--arms", f"{_label}={_pt}"]
    _w, _per, _cap, _note = plan_group_workers(_ds_name, _shot, tta=False, n_runs=20,
                                               requested=S8D_WORKERS_PER_GROUP)
    print(f"\n  === {_ds_name}: {len(_pts) + 1} arms, {_note} ===", flush=True)
    _slices = run_slices(20, _w)
    _procs = []
    for _i, (_a, _b) in enumerate(_slices):
        _log = open(f"/content/s10_{_i}.log", "w")
        _wcmd = ["python", "scripts/run_arms.py", "--dataset", _ds_name,
                 "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                 "--shot", str(_shot), "--n-runs", "20", "--support-selection", _sel,
                 "--resume-dir", f"{RESULTS_ROOT}/runs",
                 "--groups", "all",
                 "--run-start", str(_a), "--run-end", str(_b)] \
            + (["--gpu-frames"] if GPU_FRAMES else []) + _armspec
        print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}", flush=True)
        _procs.append((_i, _log, subprocess.Popen(
            _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT, env=worker_env())))
        _t.sleep(STAGGER_S)
    _bad = wait_all(_procs, "s10 worker")
    assert not _bad, f"workers {_bad} failed — see /content/s10_*.log"

    # Merge each fixed arm (inference-free; the record populates itself from the
    # checkpoint's own meta — flags are pre-meta fallbacks only), then combine all
    # cells including the anchored bundle.
    _cells = {}
    for _label, _pt in [("base", None)] + list(_pts.items()):
        _out = os.path.join(REPO, f"s10_{_label}.zip")
        run(["python", "scripts/run_eval.py", "--stage", f"s10_{_label}",
             "--adapter", "sst_sam2", "--dataset", _ds_name,
             "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
             "--shot", str(_shot), "--n-runs", "20", "--support-selection", _sel,
             "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
             "--out", _out, "--no-download"]
            + (["--gpu-frames"] if GPU_FRAMES else [])
            + (["--lora", _pt, "--lora-epochs", str(S10_EPOCHS)] if _pt else []))
        # To DRIVE — per-arm bundles are the combine's only inputs; the VM is not a
        # storage tier (the s8_5a2 lesson).
        shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
        _cells[_label] = _out
    _cells["occl_anchored"] = _anch
    run(["python", "scripts/combine_s8d_cells.py", "--out", BUNDLE, "--stage", STAGE,
         *[f"{k}={v}" for k, v in _cells.items()]])

elif STAGE == "s11":
    # OC-CCL method selection, BOTH datasets — see the S11_ARMS block for the question,
    # the measurement-licensed simplifications and the pre-registered predictions.
    # Per dataset: 1-shot × 20 on its best selection arm, {base, merged, per-trait,
    # merged+SCNP} × {plain, TTA}, every cell paired per support seed, per-dataset
    # combine, then one cross-dataset stack via combine_s8d.
    #
    # Budget: butterfly ~3.5 h (10,000 episodes per cell pass, 4 arms × 2 stacks) +
    # beetle ~1.2 h. To split across sessions, set S11_DATASETS to one dataset per
    # session — per-dataset bundles land on Drive and the final combine pulls a missing
    # dataset's bundle from there.
    import shutil

    _shot = 1
    _ft_dir = "/content/s11_adapters"
    os.makedirs(_ft_dir, exist_ok=True)
    _per_ds = {}

    for _ds_name in S11_DATASETS:
        _sel = S11_SEL[_ds_name]
        print(f"\n=== s11 {_ds_name}: 1-shot, {_sel}, "
              f"{len(S11_ARMS) + 1} arms x {{plain, TTA}} ===\n", flush=True)

        _pts, _todo = {}, []
        for _label, _over in S11_ARMS.items():
            _mm = _over["mask_mode"]
            _sw = _over.get("scnp_width", 0)
            # Every axis in the filename (the s8_5a reuse trap): mask mode, scnp,
            # sampler, no-aug, prompt, norm, mode, reset — pinned even where s11 does
            # not vary them.
            _pt = (f"{_ft_dir}/{_ds_name}_lora_lr{S11_LR:g}_ep{S11_EPOCHS}_{_mm}"
                   + (f"_scnp{_sw}" if _sw else "")
                   + "_noaug_derange_logit_reparam_imagenet_eval_reset.pt")
            _pts[_label] = _pt
            _tb = os.path.join(REPO, f"s11ft_{_ds_name}_{_label}.zip")

            # A previous session's adapter survives on Drive; the VM copy dies with the
            # runtime. Pull it back so a rescore session (e.g. after an evaluator
            # metric addition) is evaluation-weighted rather than retraining ~1 h.
            if S11_REUSE and not os.path.exists(_pt):
                _drive_pt = os.path.join(RESULTS_ROOT, os.path.basename(_pt))
                if os.path.exists(_drive_pt):
                    print(f"  pulling {os.path.basename(_pt)} from Drive", flush=True)
                    shutil.copy(_drive_pt, _pt)

            _reuse = S11_REUSE and os.path.exists(_pt)
            if _reuse:
                import torch as _torch
                _meta = (_torch.load(_pt, map_location="cpu",
                                     weights_only=False).get("meta") or {})
                _o = _meta.get("occl") or {}
                _got_sw = (_o.get("scnp") or {}).get("width", 0)
                if (_o.get("mask_mode") != _mm or _got_sw != _sw
                        or _o.get("augmentation") is not None
                        or _o.get("reference", "derangement") != "derangement"
                        or _meta.get("epochs") != S11_EPOCHS):
                    print(f"  {os.path.basename(_pt)} config mismatch — retraining",
                          flush=True)
                    _reuse = False
            if not _reuse:
                _todo.append((_label, [
                    "python", "scripts/run_finetune.py",
                    "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                    "--checkpoint", CKPT, "--device", "cuda",
                    "--method", "lora",
                    "--lr", repr(S11_LR), "--epochs", str(S11_EPOCHS),
                    "--mask-mode", _mm]
                    + (["--scnp-width", str(_sw)] if _sw else [])
                    + ["--lora-out", _pt, "--out", _tb, "--no-download"],
                    f"/content/s11_train_{_ds_name}_{_label}.log", _tb))

        # SEQUENTIAL: butterfly per-trait runs up to C=10 objects per leg.
        if _todo:
            print(f"  training {len(_todo)} arm(s), {S11_TRAIN_CONCURRENCY} at a time",
                  flush=True)
            for _k in range(0, len(_todo), S11_TRAIN_CONCURRENCY):
                train_parallel([(lab, cmd, log) for lab, cmd, log, _tb in
                                _todo[_k:_k + S11_TRAIN_CONCURRENCY]],
                               f"s11 {_ds_name} train")
            for _label, _cmd, _log, _tb in _todo:
                if os.path.exists(_tb):
                    shutil.copy(_tb, os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
        for _label, _pt in _pts.items():
            assert os.path.exists(_pt), f"{_pt} was not written"
            shutil.copy(_pt, os.path.join(RESULTS_ROOT, os.path.basename(_pt)))

        _armspec = ["--arms", "base="]
        for _label, _pt in _pts.items():
            _armspec += ["--arms", f"{_label}={_pt}"]
        _grp = GROUPS_OF[_ds_name]

        # The square: one fan-out per TTA setting, all arms sharing every worker's
        # encode (the TTA wrapper caches both orientations, so the sharing survives it).
        for _tta in (False, True):
            _w, _per, _cap, _note = plan_group_workers(
                _ds_name, _shot, tta=_tta, n_runs=20, requested=S8D_WORKERS_PER_GROUP)
            print(f"\n  === {_ds_name} {'TTA' if _tta else 'plain'}: {len(_grp)} "
                  f"group(s), {_note} ===", flush=True)
            for _gi, _group in enumerate(_grp):
                _slices = run_slices(20, _w)
                _procs = []
                for _i, (_a, _b) in enumerate(_slices):
                    _log = open(f"/content/s11_{_ds_name}_"
                                f"{'tta' if _tta else 'plain'}_g{_gi}_{_i}.log", "w")
                    _wcmd = ["python", "scripts/run_arms.py", "--dataset", _ds_name,
                             "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                             "--shot", str(_shot), "--n-runs", "20",
                             "--support-selection", _sel,
                             "--resume-dir", f"{RESULTS_ROOT}/runs",
                             "--groups", _group,
                             "--run-start", str(_a), "--run-end", str(_b)] \
                        + (["--gpu-frames"] if GPU_FRAMES else []) \
                        + _armspec + (["--tta"] if _tta else [])
                    print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}", flush=True)
                    _procs.append((_i, _log, subprocess.Popen(
                        _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
                        env=worker_env())))
                    _t.sleep(STAGGER_S)
                _bad = wait_all(_procs, f"s11 {_ds_name} {_group} worker")
                assert not _bad, (f"{_ds_name}/{_group} workers {_bad} failed — see "
                                  f"/content/s11_{_ds_name}_*_g{_gi}_*.log")

        # Eight inference-free merges, then the per-dataset combine. Records populate
        # themselves from the checkpoints' own meta.
        _cells = {}
        for _tta in (False, True):
            for _label, _pt in [("base", None)] + list(_pts.items()):
                _cell = _label if not _tta else f"{_label}_tta"
                _out = os.path.join(REPO, f"s11_{_ds_name}_{_cell}.zip")
                run(["python", "scripts/run_eval.py",
                     "--stage", f"s11_{_ds_name}_{_cell}",
                     "--adapter", "sst_sam2", "--dataset", _ds_name,
                     "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                     "--shot", str(_shot), "--n-runs", "20",
                     "--support-selection", _sel,
                     "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
                     "--out", _out, "--no-download"]
                    + (["--gpu-frames"] if GPU_FRAMES else [])
                    + (["--tta"] if _tta else [])
                    + (["--lora", _pt, "--lora-epochs", str(S11_EPOCHS)] if _pt else []))
                shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
                _cells[_cell] = _out
        _dsb = os.path.join(REPO, f"s11_{_ds_name}.zip")
        run(["python", "scripts/combine_s8d_cells.py", "--out", _dsb, "--stage", STAGE,
             *[f"{k}={v}" for k, v in _cells.items()]])
        shutil.copy(_dsb, os.path.join(RESULTS_ROOT, os.path.basename(_dsb)))
        _per_ds[_ds_name] = _dsb

    # Cross-dataset stack. A dataset run in an EARLIER session contributes via its
    # Drive-banked per-dataset bundle, so per-dataset sessions still end in one record.
    for _ds_name in ("butterfly_major", "beetle"):
        if _ds_name not in _per_ds:
            _cand = os.path.join(RESULTS_ROOT, f"s11_{_ds_name}.zip")
            if os.path.exists(_cand):
                print(f"  pulling {_ds_name} from an earlier session: {_cand}",
                      flush=True)
                _per_ds[_ds_name] = _cand
    run(["python", "scripts/combine_s8d.py", "--out", BUNDLE, "--stage", STAGE,
         *[f"{k}={v}" for k, v in _per_ds.items()]])

elif STAGE == "s12":
    # SCNP as the toggle on the per-trait cycle — see the S12 block for the question
    # and predictions. Trains and evaluates ONE arm per dataset; the toggle-off base
    # cells are S11's banked per-trait bundles, pulled from Drive, never recomputed.
    import shutil

    _shot = 1
    _ft_dir = "/content/s12_adapters"
    os.makedirs(_ft_dir, exist_ok=True)
    _per_ds = {}

    for _ds_name in S11_DATASETS:
        _sel = S11_SEL[_ds_name]
        print(f"\n=== s12 {_ds_name}: per-trait × SCNP w={S12_SCNP_WIDTH}, 1-shot, "
              f"{_sel} ===\n", flush=True)

        # The base cells this stage pairs against MUST exist before anything trains —
        # failing here costs seconds; failing at the combine costs the session.
        _b_plain = os.path.join(RESULTS_ROOT, f"s11_{_ds_name}_occl_pertrait.zip")
        _b_tta = os.path.join(RESULTS_ROOT, f"s11_{_ds_name}_occl_pertrait_tta.zip")
        for _p in (_b_plain, _b_tta):
            assert os.path.exists(_p), (
                f"{_p} not on Drive — S12 pairs against S11's banked per-trait arm "
                "and never recomputes it; re-run STAGE='s11' for this dataset first")

        _pt = (f"{_ft_dir}/{_ds_name}_lora_lr{S12_LR:g}_ep{S12_EPOCHS}_per_trait"
               f"_scnp{S12_SCNP_WIDTH}"
               f"_noaug_derange_logit_reparam_imagenet_eval_reset.pt")
        _tb = os.path.join(REPO, f"s12ft_{_ds_name}.zip")
        if S12_REUSE and not os.path.exists(_pt):
            _drive_pt = os.path.join(RESULTS_ROOT, os.path.basename(_pt))
            if os.path.exists(_drive_pt):
                print(f"  pulling {os.path.basename(_pt)} from Drive", flush=True)
                shutil.copy(_drive_pt, _pt)
        _reuse = S12_REUSE and os.path.exists(_pt)
        if _reuse:
            import torch as _torch
            _meta = (_torch.load(_pt, map_location="cpu",
                                 weights_only=False).get("meta") or {})
            _o = _meta.get("occl") or {}
            if (_o.get("mask_mode") != "per_trait"
                    or (_o.get("scnp") or {}).get("width") != S12_SCNP_WIDTH
                    or _o.get("augmentation") is not None
                    or _o.get("reference", "derangement") != "derangement"
                    or _meta.get("epochs") != S12_EPOCHS):
                print(f"  {os.path.basename(_pt)} config mismatch — retraining",
                      flush=True)
                _reuse = False
        if not _reuse:
            run(["python", "scripts/run_finetune.py",
                 "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                 "--checkpoint", CKPT, "--device", "cuda",
                 "--method", "lora",
                 "--lr", repr(S12_LR), "--epochs", str(S12_EPOCHS),
                 "--mask-mode", "per_trait",
                 "--scnp-width", str(S12_SCNP_WIDTH),
                 "--lora-out", _pt, "--out", _tb, "--no-download"])
            if os.path.exists(_tb):
                shutil.copy(_tb, os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
        assert os.path.exists(_pt), f"{_pt} was not written"
        shutil.copy(_pt, os.path.join(RESULTS_ROOT, os.path.basename(_pt)))

        # One arm, both stacks, fanned out over groups × run slices per stack.
        _grp = GROUPS_OF[_ds_name]
        _armspec = ["--arms", f"occl_ptscnp={_pt}"]
        for _tta in (False, True):
            _w, _per, _cap, _note = plan_group_workers(
                _ds_name, _shot, tta=_tta, n_runs=20, requested=S8D_WORKERS_PER_GROUP)
            print(f"\n  === {_ds_name} {'TTA' if _tta else 'plain'}: {len(_grp)} "
                  f"group(s), {_note} ===", flush=True)
            for _gi, _group in enumerate(_grp):
                _slices = run_slices(20, _w)
                _procs = []
                for _i, (_a, _b) in enumerate(_slices):
                    _log = open(f"/content/s12_{_ds_name}_"
                                f"{'tta' if _tta else 'plain'}_g{_gi}_{_i}.log", "w")
                    _wcmd = ["python", "scripts/run_arms.py", "--dataset", _ds_name,
                             "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                             "--shot", str(_shot), "--n-runs", "20",
                             "--support-selection", _sel,
                             "--resume-dir", f"{RESULTS_ROOT}/runs",
                             "--groups", _group,
                             "--run-start", str(_a), "--run-end", str(_b)] \
                        + (["--gpu-frames"] if GPU_FRAMES else []) \
                        + _armspec + (["--tta"] if _tta else [])
                    print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}", flush=True)
                    _procs.append((_i, _log, subprocess.Popen(
                        _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
                        env=worker_env())))
                    _t.sleep(STAGGER_S)
                _bad = wait_all(_procs, f"s12 {_ds_name} {_group} worker")
                assert not _bad, (f"{_ds_name}/{_group} workers {_bad} failed — see "
                                  f"/content/s12_{_ds_name}_*_g{_gi}_*.log")

        # Two inference-free merges for the new arm, then the combine against S11's
        # banked per-trait cells.
        _cells = {"base": _b_plain, "base_tta": _b_tta}
        for _tta in (False, True):
            _cell = "occl_ptscnp" if not _tta else "occl_ptscnp_tta"
            _out = os.path.join(REPO, f"s12_{_ds_name}_{_cell}.zip")
            run(["python", "scripts/run_eval.py",
                 "--stage", f"s12_{_ds_name}_{_cell}",
                 "--adapter", "sst_sam2", "--dataset", _ds_name,
                 "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                 "--shot", str(_shot), "--n-runs", "20",
                 "--support-selection", _sel,
                 "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
                 "--out", _out, "--no-download"]
                + (["--gpu-frames"] if GPU_FRAMES else [])
                + (["--tta"] if _tta else [])
                + ["--lora", _pt, "--lora-epochs", str(S12_EPOCHS)])
            shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
            _cells[_cell] = _out
        _dsb = os.path.join(REPO, f"s12_{_ds_name}.zip")
        run(["python", "scripts/combine_s8d_cells.py", "--out", _dsb, "--stage", STAGE,
             *[f"{k}={v}" for k, v in _cells.items()]])
        shutil.copy(_dsb, os.path.join(RESULTS_ROOT, os.path.basename(_dsb)))
        _per_ds[_ds_name] = _dsb

    for _ds_name in ("butterfly_major", "beetle"):
        if _ds_name not in _per_ds:
            _cand = os.path.join(RESULTS_ROOT, f"s12_{_ds_name}.zip")
            if os.path.exists(_cand):
                print(f"  pulling {_ds_name} from an earlier session: {_cand}",
                      flush=True)
                _per_ds[_ds_name] = _cand
    run(["python", "scripts/combine_s8d.py", "--out", BUNDLE, "--stage", STAGE,
         *[f"{k}={v}" for k, v in _per_ds.items()]])

elif STAGE == "s13":
    # Five loss arms on the crowned carrier — see the S13 block for the evidence base
    # and the frozen predictions. TTA-ONLY evaluation; base cells are S11's banked
    # bundles (every arm vs occl_pertrait_tta = the `vspt` combine; the margin
    # isolation additionally vs base_tta = `vsbase`, its cleanest test). Trains five
    # checkpoints per dataset sequentially, then ONE TTA fan-out carries all five arms
    # over one shared encode.
    import shutil

    _shot = 1
    _ft_dir = "/content/s13_adapters"
    os.makedirs(_ft_dir, exist_ok=True)
    _per_ds = {}

    def _s13_knobs_ok(_o, _arm):
        """The checkpoint's own meta declares EXACTLY this arm's knob at its pinned
        value, the other three off — a stale knob combination retrains."""
        _on = {k: _o.get(k) for k in ("excl", "xovl", "bshape", "margin")
               if _o.get(k) is not None}
        if _arm == "occl_excl":
            return _on == {"excl": {"mode": "zerocat"}}
        if _arm == "occl_xovl":
            return _on == {"xovl": {"weight": 1.0, "where": "query"}}
        if _arm == "occl_bshape":
            return (set(_on) == {"bshape"}
                    and _on["bshape"].get("weight") == 1.0
                    and _on["bshape"].get("schedule") == "const")
        _m = _on.get("margin") or {}
        return (set(_on) == {"margin"} and _m.get("m") == S13_MARGIN_M
                and _m.get("only") == (_arm == "occl_margin")
                and _m.get("weight") == (None if _arm == "occl_margin" else 1.0))

    for _ds_name in S11_DATASETS:
        _sel = S11_SEL[_ds_name]
        print(f"\n=== s13 {_ds_name}: 5 loss arms on the per-trait carrier, 1-shot, "
              f"{_sel}, TTA-only ===\n", flush=True)

        # The base cells this stage pairs against MUST exist before anything trains —
        # failing here costs seconds; failing at the combine costs the session.
        _b_tta = os.path.join(RESULTS_ROOT, f"s11_{_ds_name}_base_tta.zip")
        _b_pt_tta = os.path.join(RESULTS_ROOT,
                                 f"s11_{_ds_name}_occl_pertrait_tta.zip")
        for _p in (_b_tta, _b_pt_tta):
            assert os.path.exists(_p), (
                f"{_p} not on Drive — S13 pairs against S11's banked bundles and "
                "never recomputes them; re-run STAGE='s11' for this dataset first")

        _pts = {}
        for _arm, _flags in S13_ARMS.items():
            _pt = (f"{_ft_dir}/{_ds_name}_lora_lr{S13_LR:g}_ep{S13_EPOCHS}_per_trait"
                   f"_{S13_ARM_TAGS[_arm]}"
                   f"_noaug_derange_logit_reparam_imagenet_eval_reset.pt")
            _tb = os.path.join(REPO, f"s13ft_{_ds_name}_{_arm}.zip")
            if S13_REUSE and not os.path.exists(_pt):
                _drive_pt = os.path.join(RESULTS_ROOT, os.path.basename(_pt))
                if os.path.exists(_drive_pt):
                    print(f"  pulling {os.path.basename(_pt)} from Drive", flush=True)
                    shutil.copy(_drive_pt, _pt)
            _reuse = S13_REUSE and os.path.exists(_pt)
            if _reuse:
                import torch as _torch
                _meta = (_torch.load(_pt, map_location="cpu",
                                     weights_only=False).get("meta") or {})
                _o = _meta.get("occl") or {}
                if (_o.get("mask_mode") != "per_trait"
                        or _o.get("scnp") is not None
                        or _o.get("augmentation") is not None
                        or _o.get("reference", "derangement") != "derangement"
                        or _meta.get("epochs") != S13_EPOCHS
                        or not _s13_knobs_ok(_o, _arm)):
                    print(f"  {os.path.basename(_pt)} config mismatch — retraining",
                          flush=True)
                    _reuse = False
            if not _reuse:
                _cmd = ["python", "scripts/run_finetune.py",
                        "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                        "--checkpoint", CKPT, "--device", "cuda",
                        "--method", "lora",
                        "--lr", repr(S13_LR), "--epochs", str(S13_EPOCHS),
                        "--mask-mode", "per_trait", *_flags,
                        "--lora-out", _pt, "--out", _tb, "--no-download"]
                if "--margin-m" in _flags:
                    _cmd += ["--margin-rule", S13_MARGIN_RULE]
                run(_cmd)
                if os.path.exists(_tb):
                    shutil.copy(_tb,
                                os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
            assert os.path.exists(_pt), f"{_pt} was not written"
            shutil.copy(_pt, os.path.join(RESULTS_ROOT, os.path.basename(_pt)))
            _pts[_arm] = _pt

        # ONE TTA fan-out, all five arms over one shared encode per group slice.
        _grp = GROUPS_OF[_ds_name]
        _armspec = []
        for _arm, _pt in _pts.items():
            _armspec += ["--arms", f"{_arm}={_pt}"]
        _w, _per, _cap, _note = plan_group_workers(
            _ds_name, _shot, tta=True, n_runs=20, requested=S8D_WORKERS_PER_GROUP)
        print(f"\n  === {_ds_name} TTA (the only stack): {len(_grp)} group(s), "
              f"{_note} ===", flush=True)
        for _gi, _group in enumerate(_grp):
            _slices = run_slices(20, _w)
            _procs = []
            for _i, (_a, _b) in enumerate(_slices):
                _log = open(f"/content/s13_{_ds_name}_tta_g{_gi}_{_i}.log", "w")
                _wcmd = ["python", "scripts/run_arms.py", "--dataset", _ds_name,
                         "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                         "--shot", str(_shot), "--n-runs", "20",
                         "--support-selection", _sel,
                         "--resume-dir", f"{RESULTS_ROOT}/runs",
                         "--groups", _group,
                         "--run-start", str(_a), "--run-end", str(_b)] \
                    + (["--gpu-frames"] if GPU_FRAMES else []) \
                    + _armspec + ["--tta"]
                print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}", flush=True)
                _procs.append((_i, _log, subprocess.Popen(
                    _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
                    env=worker_env())))
                _t.sleep(STAGGER_S)
            _bad = wait_all(_procs, f"s13 {_ds_name} {_group} worker")
            assert not _bad, (f"{_ds_name}/{_group} workers {_bad} failed — see "
                              f"/content/s13_{_ds_name}_tta_g{_gi}_*.log")

        # Five inference-free merges, then the two combines: every arm against the
        # banked per-trait TTA cells (vspt); the margin isolation also against the
        # banked base TTA cells (vsbase).
        _cells_vspt = {"base_tta": _b_pt_tta}
        _cells_vsbase = {"base_tta": _b_tta}
        for _arm, _pt in _pts.items():
            _cell = f"{_arm}_tta"
            _out = os.path.join(REPO, f"s13_{_ds_name}_{_cell}.zip")
            run(["python", "scripts/run_eval.py",
                 "--stage", f"s13_{_ds_name}_{_cell}",
                 "--adapter", "sst_sam2", "--dataset", _ds_name,
                 "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
                 "--shot", str(_shot), "--n-runs", "20",
                 "--support-selection", _sel,
                 "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
                 "--out", _out, "--no-download"]
                + (["--gpu-frames"] if GPU_FRAMES else [])
                + ["--tta", "--lora", _pt, "--lora-epochs", str(S13_EPOCHS)])
            shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
            _cells_vspt[_cell] = _out
            if _arm == "occl_margin":
                _cells_vsbase[_cell] = _out
        _dsb = os.path.join(REPO, f"s13vspt_{_ds_name}.zip")
        run(["python", "scripts/combine_s8d_cells.py", "--out", _dsb,
             "--stage", STAGE, *[f"{k}={v}" for k, v in _cells_vspt.items()]])
        shutil.copy(_dsb, os.path.join(RESULTS_ROOT, os.path.basename(_dsb)))
        _vsb = os.path.join(REPO, f"s13vsbase_{_ds_name}.zip")
        run(["python", "scripts/combine_s8d_cells.py", "--out", _vsb,
             "--stage", STAGE, *[f"{k}={v}" for k, v in _cells_vsbase.items()]])
        shutil.copy(_vsb, os.path.join(RESULTS_ROOT, os.path.basename(_vsb)))
        _per_ds[_ds_name] = _dsb

    for _ds_name in ("butterfly_major", "beetle"):
        if _ds_name not in _per_ds:
            _cand = os.path.join(RESULTS_ROOT, f"s13vspt_{_ds_name}.zip")
            if os.path.exists(_cand):
                print(f"  pulling {_ds_name} from an earlier session: {_cand}",
                      flush=True)
                _per_ds[_ds_name] = _cand
    run(["python", "scripts/combine_s8d.py", "--out", BUNDLE, "--stage", STAGE,
         *[f"{k}={v}" for k, v in _per_ds.items()]])

elif STAGE == "s14sec":
    # The crowned stack on a NEW model family — see the S14 block for predictions.
    # One family per session by design: the two dependency stacks conflict (sam3's
    # repo pins numpy<2 — never pip-installed, checkout-only; sec pins
    # transformers==4.42.3). Commissioning runs BEFORE anything spends GPU-hours.
    import shutil

    _fam = STAGE[len("s14"):]
    _shot = 1
    _ft_dir = f"/content/{STAGE}_adapters"
    os.makedirs(_ft_dir, exist_ok=True)

    # Every phase is timed and summarised at the end — measured seconds are what the
    # next session's worker widths get tuned from, not estimates.
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

    # ── family environment ────────────────────────────────────────────────
    if not os.path.exists("/content/SeC"):
        run(["git", "clone", "-q", "--depth", "1",
             "https://github.com/OpenIXCLab/SeC", "/content/SeC"])
    os.environ["OCCL_SEC_ROOT"] = "/content/SeC"
    # Hand-picked runtime set — the repo's requirements drag GUI/training
    # stacks (cchardet kills modern Python). numpy is left as the runtime has
    # it; if the commission step fails on a numpy-2 incompatibility inside
    # transformers 4.42.3, add "numpy<2" here and RESTART the runtime.
    # iopath: imported unconditionally by the vendored Hiera backbone
    # (inference/sam2/modeling/backbones/hieradet.py) — not preinstalled on
    # Colab; its absence killed the first s14sec commissioning attempt.
    run(["pip", "install", "-q", "transformers==4.42.3", "peft==0.11.1",
         "timm==1.0.9", "einops", "hydra-core", "omegaconf", "iopath"])
    _fam_ckpt = "/content/SeC-4B"
    # UNCONDITIONAL, not guarded on config.json: snapshot_download verifies
    # every file's presence and size against the repo and fills only the gaps,
    # so a complete snapshot costs seconds and a broken one costs exactly the
    # missing bytes. The existence guard it replaces certified a snapshot that
    # had LOST a 5 GB shard between kernel restarts (FileNotFoundError at
    # model-00003, second s14sec session).
    from huggingface_hub import snapshot_download

    snapshot_download(S14_SEC_HF, local_dir=_fam_ckpt)   # 15.18 GB, ungated

    # ── commissioning: the gate-shaped asserts, in seconds ────────────────
    with _timed("commission"):
        run(["python", "scripts/commission_s14.py", "--family", _fam,
             "--checkpoint", _fam_ckpt, "--device", "cuda"])

    # ── banked SAM 2 ledger cells (fail before training, not at the combine) ──
    _sam2 = {}
    for _ds_name in S11_DATASETS:
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

    _per_ds = {}
    for _ds_name in S11_DATASETS:
        _sel = S11_SEL[_ds_name]
        print(f"\n=== {STAGE} {_ds_name}: crowned stack on {_fam}, 1-shot, {_sel} "
              f"===\n", flush=True)

        # ── the stack fine-tune (per-trait + excl zerocat on the family) ──
        _pt = (f"{_ft_dir}/{_ds_name}_{_fam}_lora_lr{S14_LR:g}_ep{S14_EPOCHS}"
               f"_per_trait_excl-zerocat"
               f"_noaug_derange_logit_reparam_eval_reset.pt")
        _tb = os.path.join(REPO, f"s14ft_{_fam}_{_ds_name}.zip")
        if S14_REUSE and not os.path.exists(_pt):
            _drive_pt = os.path.join(RESULTS_ROOT, os.path.basename(_pt))
            if os.path.exists(_drive_pt):
                shutil.copy(_drive_pt, _pt)
        _reuse = S14_REUSE and os.path.exists(_pt)
        if _reuse:
            import torch as _torch
            _meta = (_torch.load(_pt, map_location="cpu",
                                 weights_only=False).get("meta") or {})
            _o = _meta.get("occl") or {}
            if (_meta.get("family") != _fam
                    or _o.get("mask_mode") != "per_trait"
                    or (_o.get("excl") or {}).get("mode") != "zerocat"
                    or any(_o.get(k) is not None
                           for k in ("xovl", "bshape", "margin", "scnp"))
                    or _o.get("augmentation") is not None):
                print(f"  {os.path.basename(_pt)} config mismatch — retraining",
                      flush=True)
                _reuse = False
        if not _reuse:
            with _timed(f"ft_{_ds_name}"):
                run(["python", "scripts/run_finetune.py",
                     "--family", _fam,
                     "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                     "--checkpoint", _fam_ckpt, "--device", "cuda",
                     "--method", "lora",
                     "--lr", repr(S14_LR), "--epochs", str(S14_EPOCHS),
                     "--mask-mode", "per_trait", "--excl-mode", "zerocat",
                     "--lora-out", _pt, "--out", _tb, "--no-download"])
            if os.path.exists(_tb):
                shutil.copy(_tb, os.path.join(RESULTS_ROOT, os.path.basename(_tb)))
        assert os.path.exists(_pt), f"{_pt} was not written"
        shutil.copy(_pt, os.path.join(RESULTS_ROOT, os.path.basename(_pt)))

        # ── eval fan-outs ─────────────────────────────────────────────────
        # sam3 cells: base(plain), base_tta, stack_tta.
        # sec  cells: off(plain), forced(plain), off_tta, stack_tta — `forced`
        # is its own fan-out because the concept policy is adapter-level.
        _grp = GROUPS_OF[_ds_name]

        def _fan(_tta, _armspec, _policy="off", _log_tag="a"):
            # Width from the S14 VRAM table. Passing it as `requested` OVERRIDES
            # the planner's sam2-calibrated per-worker fit deliberately — that fit
            # cannot price this family's residency; the table's arithmetic (in the
            # S14 constants block) is the measurement. vCPU and run-count ceilings
            # still apply inside plan_group_workers.
            _req = S14_EVAL_WORKERS[
                _fam if _fam == "sam3"
                else ("sec_forced" if _policy == "forced" else "sec_off")]
            # S8D_WORKERS_PER_GROUP is None-by-design ("let the planner choose") —
            # here that means "no manual override"; a manual integer only tightens.
            # min() on the raw constant is the TypeError that ended the first
            # butterfly fan-out.
            _manual = S8D_WORKERS_PER_GROUP or _req
            _w, _per, _cap, _note = plan_group_workers(
                _ds_name, _shot, tta=_tta, n_runs=20,
                requested=min(_manual, _req))
            print(f"\n  === {_ds_name} {_fam} {'TTA' if _tta else 'plain'} "
                  f"[{_log_tag}]: {_note} ===", flush=True)
            for _gi, _group in enumerate(_grp):
                _procs = []
                for _i, (_a, _b) in enumerate(run_slices(20, _w)):
                    _log = open(f"/content/{STAGE}_{_ds_name}_{_log_tag}"
                                f"_g{_gi}_{_i}.log", "w")
                    _wcmd = (["python", "scripts/run_arms.py",
                              "--adapter", _fam,
                              "--dataset", _ds_name, "--cache-root", LOCAL_CACHE,
                              "--checkpoint", _fam_ckpt,
                              "--shot", str(_shot), "--n-runs", "20",
                              "--support-selection", _sel,
                              "--resume-dir", f"{RESULTS_ROOT}/runs",
                              "--groups", _group,
                              "--run-start", str(_a), "--run-end", str(_b)]
                             + (["--concept-policy", _policy]
                                if _fam == "sec" else [])
                             + _armspec + (["--tta"] if _tta else []))
                    print(f"  [{_i}] $ {' '.join(_wcmd)}  -> {_log.name}",
                          flush=True)
                    _procs.append((_i, _log, subprocess.Popen(
                        _wcmd, cwd=REPO, stdout=_log, stderr=subprocess.STDOUT,
                        env=worker_env())))
                    _t.sleep(STAGGER_S)
                _bad = wait_all(_procs, f"{STAGE} {_ds_name} {_group} worker")
                assert not _bad, (f"{_ds_name}/{_group} workers {_bad} failed — "
                                  f"see /content/{STAGE}_{_ds_name}_"
                                  f"{_log_tag}_g{_gi}_*.log")

        with _timed(f"fan_plain_{_ds_name}"):
            _fan(False, ["--arms", "base="], _policy="off", _log_tag="plain")
        if _fam == "sec":
            with _timed(f"fan_forced_{_ds_name}"):
                _fan(False, ["--arms", "base="], _policy="forced",
                     _log_tag="forced")
        with _timed(f"fan_tta_{_ds_name}"):
            _fan(True, ["--arms", "base=", "--arms", f"stack={_pt}"],
                 _policy="off", _log_tag="tta")

        # ── merges (inference-free) ───────────────────────────────────────
        def _merge(_cell, _tta, _policy="off", _lora=None):
            _out = os.path.join(REPO, f"{STAGE}_{_ds_name}_{_cell}.zip")
            run(["python", "scripts/run_eval.py",
                 "--stage", f"{STAGE}_{_ds_name}_{_cell}",
                 "--adapter", _fam, "--dataset", _ds_name,
                 "--cache-root", LOCAL_CACHE, "--checkpoint", _fam_ckpt,
                 "--shot", str(_shot), "--n-runs", "20",
                 "--support-selection", _sel,
                 "--resume-dir", f"{RESULTS_ROOT}/runs", "--no-precompute",
                 "--out", _out, "--no-download"]
                + (["--concept-policy", _policy] if _fam == "sec" else [])
                + (["--tta"] if _tta else [])
                + (["--lora", _lora, "--lora-epochs", str(S14_EPOCHS)]
                   if _lora else []))
            shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
            return _out

        with _timed(f"merge_combine_{_ds_name}"):
            _cells = {"off": _merge("off", False),
                      "forced": _merge("forced", False, _policy="forced"),
                      "off_tta": _merge("off_tta", True),
                      "stack_tta": _merge("stack_tta", True, _lora=_pt)}

            # ── the five-pairing combine ──────────────────────────────────
            _dsb = os.path.join(REPO, f"{STAGE}_{_ds_name}.zip")
            run(["python", "scripts/combine_s14.py", "--family", _fam,
                 "--out", _dsb,
                 *[f"{k}={v}" for k, v in _cells.items()],
                 *[f"{k}={v}" for k, v in _sam2[_ds_name].items()]])
            shutil.copy(_dsb, os.path.join(RESULTS_ROOT, os.path.basename(_dsb)))
        _per_ds[_ds_name] = _dsb

    for _ds_name in ("butterfly_major", "beetle"):
        if _ds_name not in _per_ds:
            _cand = os.path.join(RESULTS_ROOT, f"{STAGE}_{_ds_name}.zip")
            if os.path.exists(_cand):
                print(f"  pulling {_ds_name} from an earlier session: {_cand}",
                      flush=True)
                _per_ds[_ds_name] = _cand
    run(["python", "scripts/combine_s14.py", "--family", _fam, "--merge",
         "--out", BUNDLE, *[f"{k}={v}" for k, v in _per_ds.items()]])

    # The measured wall-clock ledger — what next session's widths get tuned from.
    print(f"\n=== {STAGE} timing summary "
          f"(total {(_t.time() - _stage_t0) / 60:.1f} min) ===")
    for _k, _v in _ph.items():
        print(f"  {_k:>28s}: {_v / 60:6.1f} min")

elif STAGE == "s16":
    # TTA as an implicit second shot — 4 runs on SAM 2. See the S16 config block for
    # the premise and the pre-registered predictions.
    #
    # Per dataset: base weights, the dataset's canonical 1-shot protocol, 20 seeds, TWO
    # fanout passes — 1-shot + TTA (the implicit second reference, and the pinned
    # anchor) and 2-shot plain (the explicit one). The resume slug carries shot,
    # selection and adapter, so the cells never collide and a TTA cell banked by S11 is
    # served from Drive with zero inference. Then the per-dataset combine (the ONE
    # paired row, carrying its prefix-nesting proof and the declared model.tta mask)
    # and one cross-dataset merge. Budget ≈ 1.1 h butterfly (the TTA cell dominates) +
    # 0.3 h beetle; set S16_DATASETS to one dataset per session to split — the final
    # merge pulls a missing dataset's combine from Drive.
    import shutil

    _per_ds = {}
    for _ds_name in S16_DATASETS:
        _sel = S16_SEL[_ds_name]
        _cells = {}
        for _cell, (_shot, _tta) in S16_CELLS.items():
            _out = os.path.join(REPO, f"s16_{_ds_name}_{_cell}.zip")
            print(f"\n=== s16 {_ds_name}: {_shot}-shot "
                  f"{'TTA' if _tta else 'plain'}, {_sel} ===\n", flush=True)
            fanout(stage=f"s16_{_ds_name}_{_cell}", dataset=_ds_name, shot=_shot,
                   n_runs=20, selection=_sel, workers=WORKERS, bundle=_out,
                   extra=["--tta"] if _tta else [])
            shutil.copy(_out, os.path.join(RESULTS_ROOT, os.path.basename(_out)))
            _cells[_cell] = _out
        _dsb = os.path.join(REPO, f"s16_{_ds_name}.zip")
        run(["python", "scripts/combine_s16.py", "--out", _dsb, "--stage", STAGE,
             *[f"{k}={v}" for k, v in _cells.items()]])
        shutil.copy(_dsb, os.path.join(RESULTS_ROOT, os.path.basename(_dsb)))
        _per_ds[_ds_name] = _dsb

    # Cross-dataset merge. A dataset run in an EARLIER session contributes via its
    # Drive-banked per-dataset combine, so split sessions still end in one record.
    for _ds_name in ("butterfly_major", "beetle"):
        if _ds_name not in _per_ds:
            _cand = os.path.join(RESULTS_ROOT, f"s16_{_ds_name}.zip")
            if os.path.exists(_cand):
                print(f"  pulling {_ds_name} from an earlier session: {_cand}",
                      flush=True)
                _per_ds[_ds_name] = _cand
    run(["python", "scripts/combine_s16.py", "--merge", "--out", BUNDLE,
         "--stage", STAGE, *[f"{k}={v}" for k, v in _per_ds.items()]])

elif STAGE == "profile":
    # No gate and no resume dir — this measures rather than producing a result — but it DOES
    # write a bundle, like every other stage, and the export cell below is unconditional.
    #
    # The first version of this branch said "no bundle" in a comment and left the export cell
    # alone, so the stage ran, printed its numbers, and then died on `profile.zip was not
    # written`. An exemption stated in prose is not an exemption. It also had the timings
    # surviving only as scrollback, which is how a number gets retyped from memory a week
    # later — the failure mode this whole investigation exists to correct.
    run(["python", "scripts/profile_episode.py",
         "--dataset", PROFILE_DATASET, "--cache-root", LOCAL_CACHE, "--checkpoint", CKPT,
         "--group", PROFILE_GROUP, "--shot", str(PROFILE_SHOT),
         "--support-selection", S8D_SELECTION.get(PROFILE_DATASET, "canonical_placement"),
         "--episodes", str(PROFILE_EPISODES),
         "--out", BUNDLE, "--no-download"])

elif STAGE == "s2":
    # Prefer running S2 on your own machine instead: it is network-bound (Zenodo throttles
    # at ~14 s/image), so it costs Colab compute units for nothing, and it has no 12-hour
    # session cap. See the README's "Running S2 locally" section.
    cmd = ["python", "scripts/build_cache.py", "--cache-root", CACHE_ROOT,
           "--datasets", *DATASETS, "--source-preference", SOURCE_PREFERENCE,
           "--net-workers", "6", "--out", BUNDLE, "--no-download"]
    if ORIGINALS_ROOT:
        cmd += ["--originals-root", ORIGINALS_ROOT]
    run(cmd)

elif STAGE in ("s3", "s4a", "s4b", "s4c", "s5", "s5b", "s5c", "s6", "s6b"):
    # One table, one execution path. `workers` is the only thing that differs between a
    # single-process run and a fan-out, so there is no separate parallel code path to
    # drift out of sync with the serial one.
    cfg = {
        # Smoke — must stay serial (renders an overlay and verifies the embedding cache).
        # Run it ONCE PER DATASET. It was beetle-only for a long time, which is exactly
        # backwards: beetle reproduces the paper at both shot counts while butterfly sits
        # ~11 points low at both, and butterfly is the one nobody had looked at.
        "s3":  dict(dataset=S3_DATASET, shot=1, n_runs=2, sel=None, workers=1,
                    extra=["--overlay", "--verify-cache",
                           "--overlay-episodes", str(S3_OVERLAY_EPISODES)]
                          + (["--overlay-group", S3_GROUP] if S3_GROUP else [])),
        "s4a": dict(dataset="beetle", shot=1, n_runs=20, sel=None),
        "s4b": dict(dataset="beetle", shot=1, n_runs=20, sel="canonical"),
        "s4c": dict(dataset="beetle", shot=1, n_runs=20, sel="canonical_balanced"),
        "s5":  dict(dataset="butterfly_major", shot=1, n_runs=20, sel=None),
        "s5b": dict(dataset="butterfly_major", shot=1, n_runs=20, sel="canonical_placement"),
        # s5c — 1-shot + orientation TTA. The rotated-query defect is query-side, so no
        # support choice can fix it; TTA is the first arm that can. Modified track.
        "s5c": dict(dataset="butterfly_major", shot=1, n_runs=20,
                    sel="canonical_placement", extra=["--tta"]),
        # s6b — 5-shot with supports SPANNING orientations, drawn from real train
        # specimens (cyrbia's train pool is 34% rotated). Tests whether the memory bank
        # SELECTS the matching orientation or just blends all five — blending is the most
        # likely reason plain 5-shot gave Major only +1.48.
        "s6b": dict(dataset=DATASET_5SHOT, shot=5, n_runs=20, sel="orientation_diverse"),
        "s6":  dict(dataset=DATASET_5SHOT, shot=5, n_runs=20,
                    sel={"beetle": "canonical_balanced",
                         "butterfly_major": "canonical_placement"}[DATASET_5SHOT]),
    }[STAGE]
    fanout(stage=STAGE, dataset=cfg["dataset"], shot=cfg["shot"], n_runs=cfg["n_runs"],
           selection=cfg.get("sel"), workers=cfg.get("workers", WORKERS),
           bundle=BUNDLE, extra=cfg.get("extra", []))

    if STAGE == "s3":
        from IPython.display import Image, display

        _ov = os.path.join(REPO, "overlay_s3.png")
        assert os.path.exists(_ov), f"{_ov} was not produced — the stage did not complete"
        display(Image(filename=_ov))
        print(f"Rendered: {S3_DATASET} / {S3_GROUP or '(first group)'}, "
              f"{S3_OVERLAY_EPISODES} episodes.\n"
              "Check, in order:\n"
              "  1. Do the trait colours mean the same thing in the support mask, the\n"
              "     prediction and the GT? A permutation or mirror produces perfectly\n"
              "     plausible IoU numbers and no assertion catches it.\n"
              "  2. Is the prediction on the SPECIMEN at all, or on background//ruler/\n"
              "     colour-chart? Butterfly frames contain all three.\n"
              "  3. Does the per-trait IoU under each row match what the picture shows?\n"
              "     A high number over a visibly wrong mask means the metric and the\n"
              "     image disagree, which is the most serious thing this can find.\n"
              "If it looks right, set overlay_reviewed=true in record.json before running "
              "the gate — nothing but a human can certify this.")

# %%
# ── download the bundle ──────────────────────────────────────────────────────
# Then: drop it into results/inbox/ locally and run
#   python scripts/check_gate.py {STAGE}
import glob as _glob
import shutil

if not os.path.exists(BUNDLE):
    # Be specific about what went wrong. "missing" is ambiguous between "the stage
    # failed" and "it wrote somewhere else", and the second is easy to recover from.
    _found = sorted(_glob.glob(os.path.join(REPO, "*.zip")))
    raise FileNotFoundError(
        f"{BUNDLE} was not written.\n"
        f"  zips present in {REPO}: {[os.path.basename(f) for f in _found] or 'none'}\n"
        "  If one of those is the run you just did, the stage's BUNDLE name and the\n"
        "  export path disagree. If there are none, the stage cell did not complete —\n"
        "  scroll up for the worker logs (/content/w*.log)."
    )

# s8b's product is the ADAPTER WEIGHTS, not the numbers. Assert they are in the bundle
# before anything else, because the failure is otherwise silent and only discovered a
# session later: the zip looks normal, the gate passes on it, and S8c then has nothing to
# score. This exact miss cost two Colab runs — the first wrote the weights to /content
# (destroyed with the VM), and the second re-ran against a STALE occl-repo.zip that still
# had the old code, producing a byte-identical bundle with the same absence.
if STAGE == "s8b":
    import zipfile as _zf

    with _zf.ZipFile(BUNDLE) as _z:
        _names = _z.namelist()
        _lora_mb = (_z.getinfo("lora.pt").file_size / 1e6) if "lora.pt" in _names else 0.0
    assert "lora.pt" in _names, (
        f"{os.path.basename(BUNDLE)} has no lora.pt (contains {_names}).\n"
        "  The adapters are this stage's only durable output and S8c cannot run without\n"
        "  them. This means occl-repo.zip on Drive PREDATES the fix that embeds them —\n"
        "  re-upload it and re-run. Training is bit-reproducible (same lr gives the same\n"
        "  lora_sha), so the re-run costs ~45 s and loses nothing."
    )
    print(f"  lora.pt in bundle: {_lora_mb:.2f} MB")

# Copy to Drive FIRST. The browser download is the fragile step — it can be blocked by
# the browser, by a disconnected frontend, or by a session that has been idle — and it
# must never be the only route to a bundle that took an hour of A100 to produce.
os.makedirs(RESULTS_ROOT, exist_ok=True)
_drive_copy = os.path.join(RESULTS_ROOT, os.path.basename(BUNDLE))
shutil.copy(BUNDLE, _drive_copy)
print(f"bundle: {BUNDLE} ({os.path.getsize(BUNDLE) / 1024:.0f} KB)")
print(f"SAVED ON DRIVE: {_drive_copy}")
print("  ^ this is the copy that matters; the download below is a convenience.")

from occl.bundle import download

try:
    download(BUNDLE)
except Exception as _e:  # noqa: BLE001
    print(f"\nbrowser download failed ({type(_e).__name__}: {_e}) — this is harmless.\n"
          f"Fetch it from Drive instead: {_drive_copy}")
