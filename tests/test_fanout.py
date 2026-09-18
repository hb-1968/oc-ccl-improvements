"""Guards on the fan-out: worker environment, VRAM sizing, and the base arm's weights.

Three defects are covered here, all of the same shape — something that must hold in every
launcher held in only some of them, with no test able to see the difference:

* `fanout` capped the BLAS thread pools; the three launchers written later did not.
* `plan_workers` priced a worker's embedding cache as a constant when it scales with the
  worker's own run-slice, so it over-committed the card in five of eight configurations.
* `run_arms.set_weights` relabelled the base arm without detaching the previous arm's LoRA.

None of these could fail a run. They made it slow, or wrong, and quietly.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "run_stage.py"


def notebook_ast() -> ast.Module:
    """Parse the notebook driver, neutralising IPython `!shell` / `%magic` lines.

    Same treatment as `test_curation.test_notebook_functions_do_not_shadow_module_imports`
    — including backslash continuations, because a bare wget URL on its own line is not
    valid Python and would fail the parse before any assertion ran.
    """
    out, in_magic = [], False
    for ln in NOTEBOOK.read_text().splitlines():
        cont = ln.rstrip().endswith("\\")
        if re.match(r"^\s*(!(?!=)|%)", ln):  # not `!=`-led continuation lines
            out.append(re.sub(r"^(\s*).*$", r"\1pass", ln))
        elif in_magic:
            out.append("")
        else:
            out.append(ln)
            cont = False
        in_magic = cont
    return ast.parse("\n".join(out))


def notebook_namespace(*names):
    """Exec ONLY the named top-level definitions from the notebook, in dependency order.

    The notebook cannot be imported — it mounts Drive, downloads a checkpoint and dispatches
    a stage on import. Lifting the specific nodes keeps the test against the real source
    rather than a re-implementation that can drift, which is how `_assign` ended up with a
    copy in `test_registry` that could silently disagree with the original.
    """
    tree = notebook_ast()
    want, keep = set(names), []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in want:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & want:
                keep.append(node)
    ns: dict = {"os": __import__("os")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<notebook>", "exec"), ns)
    missing = want - set(ns)
    assert not missing, f"notebook lost {sorted(missing)}"
    return ns


# ── worker environment ───────────────────────────────────────────────────────

def test_every_worker_launcher_uses_worker_env():
    """Discover the subprocess launches; do not enumerate them.

    Enumerating is exactly how this defect survived: `fanout` set OMP_NUM_THREADS=1 and was
    the only launcher anyone looked at, while s8c, s8c_tune and s8d each grew their own
    `Popen` with only PYTORCH_CUDA_ALLOC_CONF. Butterfly at 5-shot then ran ~4.5x slower
    through the s8d launcher than the identical work ran through `fanout`.

    Any launcher added later is caught by construction, because this walks the AST for
    `Popen` calls rather than checking a list.

    The one exemption is `run()` — the SEQUENTIAL fail-fast runner (git/pip/
    commissioning/fine-tune/merges), whose Popen exists only to tee the child's
    output into the cell. It waits inline, so it cannot become the N-parallel-pools
    defect this guards; and worker_env's thread caps would be actively wrong there
    (a sequential fine-tune should use the full BLAS pool). A companion assert below
    pins that its env still inherits os.environ rather than replacing it.
    """
    tree = notebook_ast()
    launches = []
    run_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_lines.update(range(node.lineno, node.end_lineno + 1))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute)
                else fn.id if isinstance(fn, ast.Name) else "")
        if name != "Popen":
            continue
        env = next((kw.value for kw in node.keywords if kw.arg == "env"), None)
        if node.lineno in run_lines:
            # the tee-runner: env must EXTEND os.environ ({**os.environ, ...}),
            # never replace it — a bare dict would drop OCCL_SEC_ROOT et al.
            assert (isinstance(env, ast.Dict)
                    and any(k is None for k in env.keys)), \
                "run()'s Popen must spread os.environ into its env"
            continue
        launches.append((node.lineno, env))

    assert len(launches) >= 4, (
        f"expected the fan-out plus the s8c/s8c_tune/s8d launchers, found {len(launches)}")

    bad = []
    for lineno, env in launches:
        ok = (isinstance(env, ast.Call) and isinstance(env.func, ast.Name)
              and env.func.id == "worker_env")
        if not ok:
            bad.append(lineno)
    assert not bad, (
        f"Popen at line(s) {bad} does not pass env=worker_env(). Every worker needs the "
        "BLAS thread caps: N workers each opening a ~12-thread pool on a ~12-vCPU box "
        "thrash in front of the GPU, which is the measured s8d regression.")


def test_worker_env_pins_every_thread_pool():
    """The caps that matter, by name — torch, MKL, OpenBLAS and numexpr all read their own.

    Setting only OMP_NUM_THREADS leaves MKL free to open its own pool on an Intel host.
    """
    ns = notebook_namespace("worker_env")
    env = ns["worker_env"]()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        assert env[var] == "1", f"{var} is {env.get(var)!r}, not '1'"
    assert "expandable_segments" in env["PYTORCH_CUDA_ALLOC_CONF"]
    assert env is not None and "PATH" in env, "worker_env must inherit os.environ"


def test_worker_env_extra_overrides():
    ns = notebook_namespace("worker_env")
    assert ns["worker_env"](OMP_NUM_THREADS="4")["OMP_NUM_THREADS"] == "4"


# ── every stage must produce the bundle the export cell demands ──────────────

def _stage_branches(tree):
    """(stage_name, source-of-that-branch's BODY) for every `STAGE == "x"` dispatch test.

    Reads the dispatch out of the source rather than taking a list, for the same reason
    `test_every_worker_launcher_uses_worker_env` does: the defect being guarded is a stage
    that exists in one place and is forgotten in another.

    **`node.body`, never the node.** An `elif` is a nested `If` living in its parent's
    `orelse`, so unparsing the whole node drags in every branch BELOW it. The first version
    of this helper did that and the guard passed on the very source that had just failed in
    Colab — `profile` inherited the `BUNDLE` from the `s2` branch underneath it. A guard
    that cannot fail on the defect it was written for is worse than none, because it reads
    as coverage.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for cmp_ in [n for n in ast.walk(node.test) if isinstance(n, ast.Compare)]:
            left = cmp_.left
            if not (isinstance(left, ast.Name) and left.id == "STAGE"):
                continue
            body = "\n".join(ast.unparse(s) for s in node.body)
            for comparator in cmp_.comparators:
                consts = ([comparator] if isinstance(comparator, ast.Constant)
                          else getattr(comparator, "elts", []))
                for c in consts:
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        out.append((c.value, body))
    return out


def test_every_dispatchable_stage_writes_the_bundle_the_export_cell_expects():
    """The export cell is unconditional; a stage that writes nothing dies there.

    `profile` did exactly this. Its branch carried the comment "No bundle, no gate" while
    the export cell still ran `if not os.path.exists(BUNDLE): raise FileNotFoundError`, so
    the stage completed, printed everything it had measured, and then failed on
    `profile.zip was not written`. An exemption written in prose is not an exemption.

    A branch qualifies if it hands `BUNDLE` to a subprocess in any form — `--out BUNDLE`,
    `--out`, `bundle=BUNDLE` — which is how every stage actually produces its zip.
    """
    tree = notebook_ast()
    # Branches whose body only CONFIGURES (setting BUNDLE, choosing a selection) rather than
    # running the stage; the dispatch chain has both kinds and only the runners matter.
    runners = {}
    for stage, body in _stage_branches(tree):
        if "run(" not in body and "Popen" not in body and "fanout(" not in body:
            continue
        runners.setdefault(stage, []).append(body)

    assert len(runners) >= 6, f"expected the dispatch chain, found {sorted(runners)}"

    bad = []
    for stage, bodies in sorted(runners.items()):
        if not any("BUNDLE" in b for b in bodies):
            bad.append(stage)
    assert not bad, (
        f"stage(s) {bad} run but never pass BUNDLE to what they launch. The export cell "
        "raises FileNotFoundError on a missing bundle, so the stage dies AFTER doing all "
        "of its work.")


def test_stages_tuple_matches_the_dispatch():
    """`STAGES` guards STAGE at config time; it is only useful if it is complete.

    A stage added to the dispatch but missing from `STAGES` would be rejected before it
    could run — the guard turning into the bug. A stage in `STAGES` with no dispatch branch
    would pass the guard and then do nothing, which is the failure the guard exists to
    prevent. Both directions, so neither can drift.
    """
    ns = notebook_namespace("STAGES")
    declared = set(ns["STAGES"])
    dispatched = {s for s, _ in _stage_branches(notebook_ast())}
    assert declared == dispatched, (
        f"declared but not dispatched: {sorted(declared - dispatched)}; "
        f"dispatched but not declared: {sorted(dispatched - declared)}")


def test_unknown_stage_is_rejected_at_config_time():
    """A typo must fail immediately, not 400 lines later at the export cell.

    `STAGE = "stage"` matched no branch, ran nothing, and produced
    `stage.zip was not written — the stage cell did not complete`. The cell had completed.
    """
    src = NOTEBOOK.read_text()
    i = src.index("STAGES = (")
    guard = src[i:i + 900]
    assert "if STAGE not in STAGES" in guard, "the STAGE guard is gone"
    assert "raise SystemExit" in guard, "the guard must stop the notebook, not warn"
    # And it must sit ABOVE the dispatch, or it guards nothing.
    assert i < src.index("BUNDLE = os.path.join"), "the guard runs after BUNDLE is chosen"


def test_profile_script_accepts_the_bundle_flags_the_notebook_passes():
    """The notebook and the script must not drift — the notebook cannot be run here."""
    src = (REPO / "scripts" / "profile_episode.py").read_text()
    for flag in ("--out", "--no-download"):
        assert f'"{flag}"' in src, f"profile_episode.py lost {flag}"
    assert "export_bundle" in src, (
        "profile_episode.py must emit a bundle; a timing that lives only in notebook "
        "scrollback is one that gets retyped from memory later")


# ── VRAM sizing ──────────────────────────────────────────────────────────────

CARD_GB = 42.41          # Colab A100-40GB, as reported by every run's `vram.capacity_gb`
HEADROOM = 0.92


def _price(ns, dataset, shot, n, tta, n_runs=20):
    """Total residency of an n-way partition, using the notebook's own helpers."""
    groups = ns["GROUPS_OF"][dataset]
    fixed = ns["PEAK_GB"][(dataset, shot)]["fixed"] * (1.5 if tta else 1)
    return sum(fixed + ns["cache_gb"](dataset, shot, b - a, tta=tta)
               for _g, a, b in ns["_assign"](groups, n_runs, n))


def test_cache_gb_matches_the_sizes_runs_actually_recorded():
    """Calibrate against the bundles, not against the estimate that was wrong twice.

    S5 recorded 119 butterfly images at 1.248 GB and S4 79 beetle images at 0.828 GB —
    10.49 MB each, identical across datasets because every image is encoded at 1024x1024
    whatever its source resolution. A 1-shot butterfly worker owning all 20 runs holds
    100 test + 20 supports = 120 images, which must land on the number S5 measured.
    """
    ns = notebook_namespace("cache_gb", "EMBED_MB_PER_IMAGE", "TEST_PER_GROUP")
    assert ns["cache_gb"]("butterfly_major", 1, 20) == pytest.approx(1.259, abs=0.02)
    assert ns["cache_gb"]("beetle", 1, 20) == pytest.approx(0.839, abs=0.02)
    # TTA holds both orientations.
    assert ns["cache_gb"]("beetle", 1, 20, tta=True) == pytest.approx(
        2 * ns["cache_gb"]("beetle", 1, 20))
    # And it must GROW with the slice — the property the flat constant did not have.
    assert ns["cache_gb"]("butterfly_major", 5, 20) > ns["cache_gb"]("butterfly_major", 5, 10)


def test_the_old_flat_model_over_committed_the_card():
    """Prove the fix, rather than restate it: the previous sizing really did not fit.

    The old rule was `n = cap * headroom / (fixed + cache)` with a constant cache. For
    butterfly at 5-shot that yields 8 workers projecting 38.9 GB, and those 8 workers would
    have held 46.2 GB on a 42.4 GB card. That is the state the s8d butterfly pass sat in
    for five hours: VRAM pegged, no progress, and no traceback, because
    `expandable_segments` retries near the ceiling instead of failing.
    """
    ns = notebook_namespace("cache_gb", "GROUPS_OF", "_assign", "TEST_PER_GROUP",
                            "EMBED_MB_PER_IMAGE")
    # The HISTORICAL constants, pinned here on purpose. Reading them from the live PEAK_GB
    # made this test evaporate the moment the table was corrected to the measured 2.95 — it
    # started "reproducing" a rule that had never run. A regression test about a past defect
    # has to carry the past's numbers.
    OLD_FIXED, OLD_CACHE = 4.07, 0.79
    old_n = int(CARD_GB * HEADROOM / (OLD_FIXED + OLD_CACHE))
    assert old_n == 8, f"the old rule chose {old_n}, not the 8 that ran"
    # ...and price those 8 workers with the cache model that replaced it, which is what
    # revealed the over-commit. `_price` uses the live `fixed`, so add back the difference:
    # the defect was in the CACHE term, and the test must isolate it from later corrections.
    live = ns["cache_gb"]
    true_gb = sum(OLD_FIXED + live("butterfly_major", 5, b - a)
                  for _g, a, b in ns["_assign"](ns["GROUPS_OF"]["butterfly_major"], 20, old_n))
    assert true_gb > CARD_GB, (
        f"{old_n} workers really held {true_gb:.1f} GB — if this ever stops exceeding "
        f"{CARD_GB} the regression test has lost its subject")


def test_planner_fits_every_configuration_we_actually_run():
    """No (dataset, shot, tta) combination may be sized above the card.

    Five of eight were, under the flat model — including beetle 5-shot + TTA at 45.6 GB,
    which is the OOM that killed the last two workers of the s8d beetle TTA pass.
    """
    ns = notebook_namespace("cache_gb", "PEAK_GB", "GROUPS_OF", "_assign", "TEST_PER_GROUP",
                            "EMBED_MB_PER_IMAGE")
    budget = CARD_GB * HEADROOM
    for dataset in ("butterfly_major", "beetle"):
        for shot in (1, 5):
            for tta in (False, True):
                fits = [n for n in range(1, 12)
                        if _price(ns, dataset, shot, n, tta) <= budget]
                assert fits, f"{dataset} {shot}-shot tta={tta} does not fit even serially"
                chosen = max(fits)
                total = _price(ns, dataset, shot, chosen, tta)
                assert total <= budget, (
                    f"{dataset} {shot}-shot tta={tta}: {chosen} workers -> {total:.1f} GB "
                    f"against a {budget:.1f} GB budget")


def test_run_splitting_cannot_divide_the_test_set():
    """Why a worker's cache does not fall as 1/n — and why the old flat model looked fine.

    Every run scores every test image, so all workers on a group hold all 100 of its test
    embeddings; only the supports divide. Total residency is therefore monotone *increasing*
    in the worker count: an extra worker adds a whole model plus that 100-image floor and
    subdivides at most `shot * n_runs` supports.

    That is the shape the constant `cache` term got wrong in both directions at once — too
    small for a wide slice, and insensitive to the slice entirely.
    """
    ns = notebook_namespace("cache_gb", "PEAK_GB", "GROUPS_OF", "_assign", "TEST_PER_GROUP",
                            "EMBED_MB_PER_IMAGE")
    totals = [_price(ns, "butterfly_major", 5, n, tta=False) for n in range(1, 12)]
    assert all(b > a for a, b in zip(totals, totals[1:])), (
        f"expected monotone growth in total residency, got {[round(t, 1) for t in totals]}")

    # The floor: halving a worker's runs does NOT halve its cache.
    full = ns["cache_gb"]("butterfly_major", 5, 20)
    half = ns["cache_gb"]("butterfly_major", 5, 10)
    assert half > full / 2, "the 100-image test floor must survive a run-split"
    assert half < full, "supports must still divide"


# ── the balanced (group-serial, run-parallel) layout ─────────────────────────

def test_run_slices_are_balanced_and_cover_every_run():
    """Equal work by construction — the property `_assign` cannot provide."""
    ns = notebook_namespace("run_slices")
    for n_runs in (20, 20, 7):
        for n in range(1, n_runs + 1):
            sl = ns["run_slices"](n_runs, n)
            assert len(sl) == n, f"{n} workers -> {len(sl)} slices"
            covered = [r for a, b in sl for r in range(a, b)]
            assert covered == list(range(n_runs)), "every run owned exactly once, in order"
            widths = [b - a for a, b in sl]
            assert max(widths) - min(widths) <= 1, f"unbalanced: {widths}"


def test_balanced_layout_beats_assign_on_the_configurations_we_run():
    """Prove the change, rather than restate it: the critical path must actually shrink.

    Wall clock is set by the slowest worker. `_assign` gives butterfly's five groups to six
    workers as [10,10,20,20,20,20] run-slices — 40 run-units for four of them and 20 for the
    other two, so two workers finish early and idle. Splitting each group's runs instead
    makes every slice the same width.
    """
    ns = notebook_namespace("run_slices", "_assign", "GROUPS_OF")
    G = ns["GROUPS_OF"]["butterfly_major"]
    for n in (3, 4, 6):
        assign_units = max((b - a) * 2 for _g, a, b in ns["_assign"](G, 20, n))
        # groups run in sequence; each group's critical path is its widest run-slice
        balanced_units = max(b - a for a, b in ns["run_slices"](20, n)) * 2 * len(G)
        assign_total = assign_units          # groups run concurrently in the old layout
        assert balanced_units <= assign_total * len(G), "sanity"
        # per-group comparison is the honest one: same total work, better balance
        assert max(b - a for a, b in ns["run_slices"](20, n)) * 2 <= assign_units, (
            f"{n} workers: balanced slice {max(b - a for a, b in ns['run_slices'](20, n)) * 2}"
            f" is not shorter than _assign's {assign_units}")


def test_plan_group_workers_never_makes_an_empty_worker():
    """More workers than runs would load 898 MB of checkpoint to do nothing.

    `_assign` had the mirror-image bug — returning a different number of slices than asked
    for — and that is what put 56 GB onto a 39.5 GB card.
    """
    ns = notebook_namespace("run_slices")
    assert len(ns["run_slices"](20, 25)) == 20, "cannot exceed one worker per run"
    assert all(b > a for a, b in ns["run_slices"](20, 25)), "no empty slices"


def test_narrower_slices_shrink_the_cache_which_is_what_buys_the_workers_back():
    """The reason this layout fits MORE workers, not merely balances the ones it has."""
    ns = notebook_namespace("cache_gb", "TEST_PER_GROUP", "EMBED_MB_PER_IMAGE")
    wide = ns["cache_gb"]("butterfly_major", 5, 20, tta=True)     # _assign: whole group
    narrow = ns["cache_gb"]("butterfly_major", 5, 5, tta=True)    # balanced: quarter of it
    assert narrow < wide, "a narrower run-slice must draw fewer distinct supports"
    # ...but the 100-image test floor keeps it from falling proportionally.
    assert narrow > wide / 4


# ── the base arm ─────────────────────────────────────────────────────────────

def test_base_arm_resets_adapters_before_scoring():
    """`run_arms` scores group-outer/arm-inner; base must undo the previous arm.

    Without the reset, every group after the first scores "base" with the OC-CCL adapters
    still attached and files it under `:base` — the reference the delta is measured against.
    Latent so far only because `_assign` gave one group per worker at the 5 and 8 worker
    counts butterfly ran at.
    """
    torch = pytest.importorskip("torch")
    from torch import nn

    from occl.finetune.lora import apply_lora, lora_delta_norm, reset_lora

    model = nn.Module()
    model.memory_encoder = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    model.sam_mask_decoder = nn.Sequential(nn.Linear(8, 8))
    apply_lora(model, r=2, alpha=4.0, scopes=("memory_encoder", "sam_mask_decoder"))
    assert lora_delta_norm(model) == 0.0            # zero-init B

    # Stand in for a trained arm being loaded.
    for m in model.modules():
        if hasattr(m, "lora_B"):
            with torch.no_grad():
                m.lora_B.normal_()
    assert lora_delta_norm(model) > 0.0

    n = reset_lora(model)
    assert n == 3
    assert lora_delta_norm(model) == 0.0, (
        "the base arm would be scored with the previous arm's weights")


def test_run_arms_base_path_calls_reset():
    """The source really routes the base arm through the reset — not just the helper.

    A unit test on `reset_lora` alone would pass while `set_weights` still early-returned,
    which is precisely the state this fixes.
    """
    src = (REPO / "scripts" / "run_arms.py").read_text()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "set_weights"), None)
    assert fn is not None, "run_arms lost set_weights"
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "reset_lora" in called, (
        "set_weights must reset the adapters on the base arm, not merely relabel it")
    assert "lora_delta_norm" in called, (
        "and must assert ||dW|| == 0 afterwards rather than trusting the reset")


# ── s8.5: OC-CCL as SST actually trains it ───────────────────────────────────

def test_s8_5_stages_pass_method_full():
    """Without `--method full`, s8.5 is an expensive duplicate of s8d.

    The whole reason the stage exists is that S8 measured a LoRA configuration which
    appears nowhere in SST. `--method` defaults to `lora` (so old S8 commands still
    reproduce S8), which means the default is exactly wrong for this stage — omitting the
    flag would produce plausible numbers for the experiment we already ran.
    """
    branches = {s: b for s, b in _stage_branches(notebook_ast())}
    for stage in ("s8_5a", "s8_5b", "s8_5c"):
        assert stage in branches, f"{stage} has no dispatch branch"
    body = branches["s8_5a"]
    assert "'--method', 'full'" in body or '"--method", "full"' in body, \
        "the s8.5 branch does not pass --method full"


def test_s8_5_covers_the_three_protocols_asked_for():
    """a = 1-shot plain, b = 1-shot + TTA, c = 5-shot + TTA, both datasets."""
    body = "\n".join(b for s, b in _stage_branches(notebook_ast()) if s.startswith("s8_5"))
    assert "'s8_5a': (1, False)" in body
    assert "'s8_5b': (1, True)" in body
    assert "'s8_5c': (5, True)" in body


def test_run_finetune_default_method_is_lora_so_s8_reproduces():
    """Flipping this default would make every archived S8 command mean a new experiment.

    `FAITHFUL_FINETUNE` has flipped twice on which method is faithful (release's `full`,
    then the paper's `lora` — see test_registry for the history). Neither flip is a
    licence to change what an old command does: classification and CLI defaults move
    independently, and this default is pinned to the commands already archived.
    """
    src = (REPO / "scripts" / "run_finetune.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and getattr(node.args[0], "value", "") == "--method"):
            default = next((k.value.value for k in node.keywords if k.arg == "default"), None)
            assert default == "lora", f"--method default is {default!r}, not 'lora'"
            return
    raise AssertionError("run_finetune.py has no --method argument")


def test_scoring_record_can_describe_every_training_axis():
    """`run_eval`'s `occl` block must cover every field `run_finetune`'s does.

    The training record carried `normalization` and `train_mode`; the scoring record did
    not. So `occl_unit` and `occl_train` — arms that differ in exactly those fields — landed
    in the combined bundle with IDENTICAL `occl` blocks, distinguishable only by `||dW||`
    and a filename. The numbers were right and the record could not say what produced them.

    Discovered by AST from both sources rather than enumerated, because the failure is a
    field added on one side and forgotten on the other.
    """
    def occl_keys(path):
        tree = ast.parse((REPO / "scripts" / path).read_text())
        best = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if {"prompt_mode", "cycle_len"} <= keys:      # the occl block, either side
                best |= keys
        assert best, f"{path}: no occl block found"
        return best

    train, score = occl_keys("run_finetune.py"), occl_keys("run_eval.py")
    missing = sorted(train - score)
    assert not missing, (
        f"run_eval.py's occl block cannot record {missing}, which run_finetune.py varies. "
        "Arms differing only in those fields would be indistinguishable in the record.")


def test_combine_cells_computes_the_interaction_for_any_single_method_square():
    """The combiner's interaction block indexed deltas as ["tta"] - ["plain"] — s8d's
    tag names — and crashed the first stage whose two cells were named anything else
    (s12: ptscnp/ptscnp_tta). Pin the generic (m, m_tta) pairing with real zips through
    the real script."""
    import json
    import subprocess
    import sys
    import tempfile
    import zipfile
    from pathlib import Path

    def rec(adapter, vals, finetune=None):
        return {
            "dataset": "beetle", "adapter": adapter,
            "model": {"family": "sam2", "image_size": 1024},
            "metric": {"name": "iou_per_trait_macro", "assignment": "argmax_label",
                       "reduction": "macro_over_gt_traits", "iou_space": "gt_native"},
            "data": {"source_preference": "faithful", "interp": "linear",
                     "cache_format": "png"},
            "protocol": {"shots": 1, "n_runs": len(vals), "groups": ["all"],
                         "support_selection": "canonical_balanced",
                         "support_set_hash": "h",
                         # model.tta differs per stack via the record's model block in
                         # real bundles; here both stacks share a key because each pair
                         # is compared within its own stack only.
                         },
            "results": {"miou": sum(vals) / len(vals), "std": 0.01,
                        "per_run": list(vals),
                        "best_cell": None, "best_miou": None},
            "runs": [{"run_idx": i, "miou": v,
                      "miou_boundary": {"independent_best3": v + 0.01},
                      "miou_variants": {"independent": v}}
                     for i, v in enumerate(vals)],
            **({"finetune": finetune, "occl": {"mask_mode": "per_trait"}}
               if finetune else {}),
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cells = {
            "base": rec("sst_sam2", [0.60, 0.62, 0.64]),
            "base_tta": rec("sst_sam2+tta_rot180", [0.62, 0.64, 0.66]),
            "occl_ptscnp": rec("sst_sam2", [0.61, 0.63, 0.64],
                               finetune={"method": "lora"}),
            "occl_ptscnp_tta": rec("sst_sam2+tta_rot180", [0.64, 0.66, 0.67],
                                   finetune={"method": "lora"}),
        }
        for name, r in cells.items():
            with zipfile.ZipFile(tmp / f"{name}.zip", "w") as z:
                z.writestr("record.json", json.dumps(r))
        out = tmp / "combined.zip"
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "combine_s8d_cells.py"),
             "--out", str(out), "--stage", "s12test",
             *[f"{k}={tmp / (k + '.zip')}" for k in cells]],
            capture_output=True, text=True)
        assert p.returncode == 0, f"combiner failed:\n{p.stdout}\n{p.stderr}"
        combined = json.loads(zipfile.ZipFile(out).read("record.json"))
        assert set(combined["cells"]) == {"ptscnp", "ptscnp_tta"}
        # plain deltas: +0.00666..; tta: +0.01666.. -> interaction +0.01
        assert combined["results"]["interaction"] == pytest.approx(0.01, abs=1e-6)
        c = combined["cells"]["ptscnp"]
        assert c["paired_delta_best3"] is not None
        assert c["paired_delta_best3"] == pytest.approx(c["paired_delta"], abs=1e-9)


def test_the_none_default_worker_override_is_never_fed_to_arithmetic():
    """S8D_WORKERS_PER_GROUP defaults to None ("let the planner choose"), so any
    arithmetic on the raw constant assumes an int and detonates at stage runtime —
    min() on it ended the first s14sec butterfly fan-out ten minutes in, AFTER the
    fine-tune had already been paid for. The constant may be passed through or
    None-coalesced (`or`), never handed bare to min()/max()."""
    tree = notebook_ast()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("min", "max")):
            names = [a.id for a in node.args if isinstance(a, ast.Name)]
            assert "S8D_WORKERS_PER_GROUP" not in names, (
                f"line {node.lineno}: {node.func.id}() on the None-default "
                f"S8D_WORKERS_PER_GROUP")
