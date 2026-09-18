"""Guards on the SCC path: `occl.scc` (the multi-GPU scheduler) and the SCC
notebook (notebooks/run_s14sam3_scc.py).

The scheduler is exercised with a fake `spawn` and injected clock/sleep, so
the packing, dependency and failure rules are tested here on a machine with
no GPU. The notebook guards mirror tests/test_s14_standalone.py: copied
helpers are pinned to run_stage.py by AST equality, the stage constant is
pinned, every fan-out Popen goes through worker_env(), no google.colab import
survives, and every `--flag` the notebook passes exists in the target script's
argparse (the sweep that would have caught `unrecognized arguments` before it
cost a session).
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re

import pytest

from occl import scc

REPO = pathlib.Path(__file__).resolve().parent.parent
MULTI = REPO / "notebooks" / "run_stage.py"
SCC_NB = REPO / "notebooks" / "run_s14sam3_scc.py"
SCRIPTS = REPO / "scripts"


# ── a fake process world ─────────────────────────────────────────────────────

class FakeHandle:
    def __init__(self, ticks: int, rc: int = 0):
        self.ticks, self.rc, self.killed = ticks, rc, False

    def poll(self):
        if self.killed:
            return -9
        self.ticks -= 1
        return self.rc if self.ticks < 0 else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.rc


class World:
    """Records launches; `ticks[tag]` / `rc[tag]` shape each fake process."""

    def __init__(self, ticks=None, rc=None):
        self.ticks, self.rc = ticks or {}, rc or {}
        self.launches: list[tuple[str, str, int]] = []   # (tag, gpu, launch_step)
        self.step = 0
        self.live: dict[str, FakeHandle] = {}

    def spawn(self, job, gpu):
        h = FakeHandle(self.ticks.get(job.tag, 2), self.rc.get(job.tag, 0))
        self.launches.append((job.tag, gpu, self.step))
        self.live[job.tag] = h
        return h

    def sleep(self, s):
        self.step += 1

    def clock(self):
        return float(self.step)


def _sched(world, gpus, max_procs=99, **kw):
    kw.setdefault("stagger_s", 0)
    kw.setdefault("heartbeat_s", 0)
    return scc.Scheduler(gpus=gpus, max_procs=max_procs, sleep=world.sleep,
                         clock=world.clock, log=lambda *_: None, **kw)


G40 = [scc.GPU("0", 40.0), scc.GPU("1", 40.0)]


def test_exclusive_job_owns_a_card_and_others_pack_by_free_vram():
    w = World(ticks={"ft": 10})
    jobs = [scc.Job("ft", ["x"], gb=None),
            scc.Job("a", ["x"], gb=10.0), scc.Job("b", ["x"], gb=10.0),
            scc.Job("c", ["x"], gb=10.0), scc.Job("d", ["x"], gb=10.0)]
    res = _sched(w, G40).run(jobs, w.spawn)
    assert all(rc == 0 for rc in res.values())
    by_tag = {t: g for t, g, _ in w.launches}
    # the exclusive fine-tune launched first, and NOTHING shared its card while
    # it was alive: every eval job that launched before ft exited sits on the
    # other GPU (40 * 0.92 = 36.8 GB holds three 10 GB workers, the fourth waits)
    assert w.launches[0][0] == "ft"
    others_during = [g for t, g, s in w.launches if t != "ft" and s <= 10]
    assert others_during and all(g != by_tag["ft"] for g in others_during)


def test_dependencies_order_launches_and_a_failure_skips_dependents():
    w = World(rc={"ft": 1})
    jobs = [scc.Job("plain", ["x"], gb=5.0),
            scc.Job("ft", ["x"], gb=None),
            scc.Job("tta", ["x"], gb=5.0, deps=("ft",)),
            scc.Job("merge", ["x"], gb=5.0, deps=("tta",))]
    res = _sched(w, G40, retry_quick_failures=0).run(jobs, w.spawn)
    assert res["plain"] == 0 and res["ft"] == 1
    assert res["tta"] == scc.SKIPPED and res["merge"] == scc.SKIPPED
    assert [t for t, _, _ in w.launches] in (["ft", "plain"], ["plain", "ft"])
    assert scc.failed(res) == ["ft", "tta", "merge"] or set(scc.failed(res)) == {
        "ft", "tta", "merge"}


def test_dependents_wait_for_success_then_run():
    w = World(ticks={"ft": 3})
    jobs = [scc.Job("ft", ["x"], gb=None), scc.Job("tta", ["x"], gb=5.0, deps=("ft",))]
    res = _sched(w, G40).run(jobs, w.spawn)
    assert res == {"ft": 0, "tta": 0}
    launch = {t: s for t, _, s in w.launches}
    assert launch["tta"] > launch["ft"] + 3


def test_core_ceiling_caps_live_processes():
    w = World(ticks={f"j{i}": 4 for i in range(6)})
    jobs = [scc.Job(f"j{i}", ["x"], gb=1.0) for i in range(6)]
    s = _sched(w, G40, max_procs=2)
    res = s.run(jobs, w.spawn)
    assert all(rc == 0 for rc in res.values())
    # never more than two alive: reconstruct occupancy from launch steps + ticks
    for t, _, step in w.launches:
        alive = sum(1 for t2, _, s2 in w.launches if s2 <= step < s2 + 5)
        assert alive <= 2, (t, alive)


def test_stagger_spaces_launches():
    w = World()
    jobs = [scc.Job(f"j{i}", ["x"], gb=1.0) for i in range(3)]
    _sched(w, G40, stagger_s=3).run(jobs, w.spawn)
    steps = sorted(s for _, _, s in w.launches)
    assert all(b - a >= 3 for a, b in zip(steps, steps[1:]))


def test_oversized_job_fails_in_the_plan_not_after_the_finetunes():
    w = World()
    with pytest.raises(AssertionError, match="needs 50.0 GB"):
        _sched(w, G40).run([scc.Job("big", ["x"], gb=50.0)], w.spawn)
    assert not w.launches


def test_unknown_dependency_and_duplicate_tags_are_rejected():
    w = World()
    with pytest.raises(AssertionError, match="unknown job"):
        _sched(w, G40).run([scc.Job("a", ["x"], gb=1.0, deps=("zz",))], w.spawn)
    with pytest.raises(AssertionError, match="duplicate"):
        _sched(w, G40).run([scc.Job("a", ["x"], gb=1.0), scc.Job("a", ["x"], gb=1.0)],
                           w.spawn)


def test_interrupt_kills_survivors_and_reraises():
    w = World(ticks={"long": 100})

    def sleep_then_interrupt(s):
        w.step += 1
        if w.step == 3:
            raise KeyboardInterrupt

    s = scc.Scheduler(gpus=G40, max_procs=4, stagger_s=0, heartbeat_s=0,
                      sleep=sleep_then_interrupt, clock=w.clock, log=lambda *_: None)
    with pytest.raises(KeyboardInterrupt):
        s.run([scc.Job("long", ["x"], gb=1.0)], w.spawn)
    assert w.live["long"].killed
    assert s.records["long"].rc == -9


def test_phase_summary_aggregates_per_phase():
    w = World(ticks={"a": 2, "b": 4})
    jobs = [scc.Job("a", ["x"], gb=1.0, phase="fan"), scc.Job("b", ["x"], gb=1.0, phase="fan"),
            scc.Job("m", ["x"], gb=1.0, phase="merge", deps=("a", "b"))]
    s = _sched(w, G40)
    s.run(jobs, w.spawn)
    summ = s.phase_summary()
    assert set(summ) == {"fan", "merge"}
    assert summ["fan"]["jobs"] == 2 and summ["merge"]["jobs"] == 1
    assert summ["fan"]["gpu_min"] >= summ["fan"]["wall_min"]


# ── the environment helpers ──────────────────────────────────────────────────

def test_visible_gpu_ids_passes_sge_strings_through():
    assert scc.visible_gpu_ids({}) is None
    assert scc.visible_gpu_ids({"CUDA_VISIBLE_DEVICES": ""}) == []
    assert scc.visible_gpu_ids({"CUDA_VISIBLE_DEVICES": "2, 5"}) == ["2", "5"]
    uu = "GPU-aaaa,GPU-bbbb"
    assert scc.visible_gpu_ids({"CUDA_VISIBLE_DEVICES": uu}) == uu.split(",")


def test_granted_cores_prefers_nslots():
    assert scc.granted_cores({"NSLOTS": "12"}) == 12
    assert scc.granted_cores({"NSLOTS": "junk"}) >= 1
    assert scc.granted_cores({}) >= 1
    assert scc.local_scratch({"TMPDIR": "/scratch/123.1.gpu"}) == "/scratch/123.1.gpu"


# ── the notebook ─────────────────────────────────────────────────────────────

#: every top-level definition the SCC notebook copies from run_stage.py
COPIED = ["worker_env", "GROUPS_OF", "run_slices", "run", "banked_runs"]


def _ast_of(path):
    out, in_magic = [], False
    for ln in path.read_text().splitlines():
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


def _top_level(tree):
    defs = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            defs.setdefault(node.targets[0].id, node)
    return defs


def test_copied_helpers_do_not_drift_from_run_stage():
    multi, alone = _top_level(_ast_of(MULTI)), _top_level(_ast_of(SCC_NB))
    missing = [n for n in COPIED if n not in alone]
    assert not missing, f"SCC notebook lost {missing}"
    drifted = [n for n in COPIED if ast.dump(multi[n]) != ast.dump(alone[n])]
    assert not drifted, (f"{drifted} differ between run_stage.py and "
                         "run_s14sam3_scc.py — edit run_stage.py, re-copy verbatim")


def test_scc_notebook_has_no_colab_and_no_shell_magics():
    """The SCC notebook must run under a plain Jupyter kernel: nothing from
    google.colab, and no `!`/`%` lines (they are IPython-only and were the
    reason the Colab notebooks need the magic-neutralising parser)."""
    src = SCC_NB.read_text()
    assert "google.colab" not in src
    assert "files.download" not in src
    assert not [ln for ln in src.splitlines() if re.match(r"^\s*(!(?!=)|%)", ln)]
    ast.parse(src)   # plain-Python-valid, without neutralising anything


def test_stage_is_pinned_and_the_arch_arm_is_carried():
    from occl.finetune.lora import FAITHFUL_SCOPES_BY_FAMILY, SAM3_ARCH_SCOPES

    tree = _ast_of(SCC_NB)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "STAGE" for t in n.targets)]
    assert len(assigns) == 1 and assigns[0].value.value == "s14sam3"
    defs = _top_level(tree)
    assert ast.literal_eval(defs["_SCOPES_STACK"].value) == FAITHFUL_SCOPES_BY_FAMILY["sam3"]
    assert ast.literal_eval(defs["_SCOPES_ARCH"].value) == SAM3_ARCH_SCOPES
    src = SCC_NB.read_text()
    assert '"--scope-preset", _preset' in src
    assert "\"--arms\", f\"arch={_adapters[_ds_name]['arch']}\"" in src
    # the reuse check reads scopes where save_lora writes them (nested)
    assert '(_m.get("lora") or {}).get("scopes")' in src


def test_fanout_popen_pins_one_gpu_through_worker_env():
    """The scheduler's spawn must pass env=worker_env(CUDA_VISIBLE_DEVICES=...)
    — the BLAS caps plus the per-card pin — and run()'s tee-Popen must spread
    os.environ (the test_fanout discipline, applied here)."""
    tree = _ast_of(SCC_NB)
    run_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_lines.update(range(node.lineno, node.end_lineno + 1))
    launches = []
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
            assert isinstance(env, ast.Dict) and any(k is None for k in env.keys)
            continue
        launches.append((node.lineno, env))
    assert len(launches) == 1, "exactly one fan-out Popen (inside spawn) expected"
    _, env = launches[0]
    assert isinstance(env, ast.Call) and env.func.id == "worker_env"
    assert [kw.arg for kw in env.keywords] == ["CUDA_VISIBLE_DEVICES"]


def _argparse_flags(script: pathlib.Path) -> set[str]:
    tree = ast.parse(script.read_text())
    flags = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            for a in node.args:
                if isinstance(a, ast.Constant) and str(a.value).startswith("--"):
                    flags.add(a.value)
    return flags


def _commands_in(tree):
    """Every list literal ["python", "scripts/<x>.py", ...] with the `--flags`
    found in its enclosing STATEMENT (so `+ [...]` continuations count)."""
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.List) and len(node.elts) >= 2
                and isinstance(node.elts[0], ast.Constant) and node.elts[0].value == "python"
                and isinstance(node.elts[1], ast.Constant)
                and str(node.elts[1].value).startswith("scripts/")):
            continue
        stmt = node
        while not isinstance(stmt, ast.stmt):
            stmt = parents[stmt]
        flags = {n.value for n in ast.walk(stmt)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and n.value.startswith("--")}
        out.append((node.elts[1].value, flags))
    return out


def test_every_flag_the_notebook_passes_exists_in_its_target_script():
    cmds = _commands_in(_ast_of(SCC_NB))
    scripts = {s for s, _ in cmds}
    assert scripts == {"scripts/commission_s14.py", "scripts/run_finetune.py",
                       "scripts/run_arms.py", "scripts/run_eval.py",
                       "scripts/combine_s14.py"}
    for script, flags in cmds:
        known = _argparse_flags(REPO / script)
        unknown = sorted(flags - known)
        assert not unknown, f"{script} does not accept {unknown}"


def test_make_notebook_emits_the_scc_ipynb_with_the_venv_kernel(tmp_path):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("mk", SCRIPTS / "make_notebook.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    dst = tmp_path / "scc.ipynb"
    mk.build(dst, SCC_NB.read_text(), kernel=("occl-sam3", "occl (sam3)"), colab=False)
    nb = json.loads(dst.read_text())
    assert nb["metadata"]["kernelspec"]["name"] == "occl-sam3"
    assert "colab" not in nb["metadata"]
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and code[0]["source"][0].startswith("# ── configuration")


# ── the SCC archives: a data layer and a code layer ─────────────────────────

def _load_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bt", SCRIPTS / "build_scc_tarball.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_world(tmp_path):
    """A miniature repo / SST / sam3 / cache / ledger tree."""
    repo = tmp_path / "repo"
    for f in ("src/occl/__init__.py", "src/occl/scc.py", "src/occl/adapters/sam3.py",
              "src/occl/finetune/train.py", "src/occl/__pycache__/x.pyc",
              "scripts/scc_setup.sh", "scripts/commission_s14.py", "scripts/run_finetune.py",
              "scripts/run_arms.py", "scripts/run_eval.py", "scripts/combine_s14.py",
              "scripts/combine_s15.py", "scripts/probe_train_pairs.py",
              "scripts/combine_s17.py", "scripts/render_probe_sheets.py",
              "scripts/check_gate.py", "scripts/make_notebook.py",
              "notebooks/run_s14sam3_scc.ipynb", "notebooks/run_s14sam3_scc.py",
              "notebooks/run_s15_scc.ipynb", "notebooks/run_s15_scc.py",
              "notebooks/run_s17_scc.ipynb", "notebooks/run_s17_scc.py",
              "notebooks/run_s17b_scc.ipynb", "notebooks/run_s17b_scc.py",
              "notebooks/run_s17c_scc.ipynb", "notebooks/run_s17c_scc.py",
              "notebooks/run_s17d_scc.ipynb", "notebooks/run_s17d_scc.py",
              "notebooks/run_s18_scc.ipynb", "notebooks/run_s18_scc.py",
              "notebooks/run_stage.ipynb", "tests/t.py", "pyproject.toml", "README.md",
              ".gitignore", ".venv/lib/junk.py"):
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text("x")
    sst = tmp_path / "SST"
    for f in ("README.md", "data/cambridge_butterfly/m.png", "data/neon_beetles/b.png",
              "src/sst/segment.py", "src/sst/__pycache__/a.pyc", "assets/big.gif",
              "gui/app.py", "experiments/bench/x.py", "checkpoints/sam2_hiera_large.pt",
              "checkpoints/download_ckpts.sh", ".git/HEAD", ".git/objects/aa/bb"):
        (sst / f).parent.mkdir(parents=True, exist_ok=True)
        (sst / f).write_text("x")
    sam3 = tmp_path / "sam3"
    for f in ("sam3/model/sam3_tracking_predictor.py", "sam3/__init__.py", "LICENSE",
              "README.md", "assets/demo.mp4", "examples/e.ipynb", ".git/HEAD",
              ".git/objects/cc/dd"):
        (sam3 / f).parent.mkdir(parents=True, exist_ok=True)
        (sam3 / f).write_text("x")
    cache = tmp_path / "cache-png"
    cache.mkdir()
    (cache / "manifest.jsonl").write_text(
        '{"dataset": "beetle", "fmt": "png", "cache_path": "beetle/a.png"}\n')
    (cache / "beetle").mkdir()
    (cache / "beetle" / "a.png").write_bytes(b"\x89PNG")
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    return repo, sst, sam3, cache, ledger


def test_code_scripts_constant_matches_what_the_notebooks_launch():
    """The code layer ships exactly the scripts the SCC notebooks' command literals
    name (plus scc_setup.sh); a notebook that starts launching a new script must
    fail here, not on the cluster as `No such file`."""
    bt = _load_builder()
    launched = set()
    for nb in (SCC_NB, REPO / "notebooks" / "run_s15_scc.py",
               REPO / "notebooks" / "run_s17_scc.py", REPO / "notebooks" / "run_s17b_scc.py",
               REPO / "notebooks" / "run_s17c_scc.py", REPO / "notebooks" / "run_s17d_scc.py",
               REPO / "notebooks" / "run_s18_scc.py"):
        launched |= {s.split("/", 1)[1] for s, _ in _commands_in(_ast_of(nb))}
    assert launched | {"scc_setup.sh"} == set(bt.CODE_SCRIPTS)


def test_data_layer_keeps_labels_pixels_and_provenance_and_drops_the_rest(tmp_path):
    bt = _load_builder()
    repo, sst, sam3, cache, ledger = _stub_world(tmp_path)
    (ledger / "s11_beetle_base.zip").write_bytes(b"PK")
    members, generated, missing = bt.plan_data(sst, cache, ledger)
    names = set(members)
    assert {"occl/SST/data/cambridge_butterfly/m.png", "occl/SST/data/neon_beetles/b.png",
            "occl/SST/src/sst/segment.py", "occl/SST/README.md",
            "occl/data/cache/manifest.jsonl", "occl/data/cache/beetle/a.png",
            "occl/results/s11_beetle_base.zip"} <= names
    assert not any(n.startswith(("occl/SST/assets", "occl/SST/gui", "occl/SST/experiments",
                                 "occl/SST/checkpoints", "occl/SST/.git")) for n in names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    # provenance without the object store: a stub HEAD, generated not copied
    assert "occl/SST/.git/HEAD" in generated
    assert "occl/results/LEDGER-NEEDED.md" in generated
    assert len(missing) == len(bt.ledger_names()) - 1 and "s11_beetle_base.zip" not in missing
    assert "s6_beetle.zip" in missing and "s8d_beetle.zip" in missing
    assert not any(n.startswith("occl/oc-ccl-improvements") for n in names)


def test_code_layer_is_the_iteration_unit_and_nothing_more(tmp_path):
    bt = _load_builder()
    repo, sst, sam3, cache, ledger = _stub_world(tmp_path)
    members, generated = bt.plan_code(repo, sam3)
    names = set(members)
    r = "occl/oc-ccl-improvements"
    assert {f"{r}/src/occl/scc.py", f"{r}/src/occl/finetune/train.py", f"{r}/pyproject.toml",
            f"{r}/scripts/run_finetune.py", f"{r}/scripts/scc_setup.sh",
            f"{r}/notebooks/run_s14sam3_scc.ipynb", f"{r}/notebooks/run_s15_scc.ipynb",
            f"{r}/notebooks/run_s17_scc.ipynb", f"{r}/notebooks/run_s17b_scc.ipynb",
            f"{r}/notebooks/run_s17c_scc.ipynb", f"{r}/notebooks/run_s17d_scc.ipynb",
            f"{r}/notebooks/run_s18_scc.ipynb",
            f"{r}/scripts/combine_s15.py",
            f"{r}/scripts/probe_train_pairs.py", f"{r}/scripts/combine_s17.py",
            "occl/sam3/sam3/model/sam3_tracking_predictor.py", "occl/sam3/LICENSE"} <= names
    shaved = [f"{r}/README.md", f"{r}/tests/t.py", f"{r}/scripts/check_gate.py",
              f"{r}/scripts/make_notebook.py", f"{r}/notebooks/run_stage.ipynb",
              f"{r}/notebooks/run_s14sam3_scc.py", f"{r}/notebooks/run_s15_scc.py",
              f"{r}/notebooks/run_s17_scc.py", f"{r}/notebooks/run_s17b_scc.py",
              f"{r}/notebooks/run_s17c_scc.py", f"{r}/notebooks/run_s17d_scc.py",
              f"{r}/notebooks/run_s18_scc.py",
              f"{r}/scripts/render_probe_sheets.py",
              f"{r}/.gitignore",
              "occl/sam3/assets/demo.mp4", "occl/sam3/examples/e.ipynb", "occl/sam3/README.md",
              "occl/sam3/.git/HEAD"]
    assert not [n for n in shaved if n in names]
    assert not any(".venv" in n or "__pycache__" in n for n in names)
    assert "occl/sam3/COMMIT" in generated
    assert not any(n.startswith(("occl/data", "occl/SST")) for n in names)


def test_plan_data_refuses_a_lossy_cache(tmp_path):
    bt = _load_builder()
    repo, sst, sam3, cache, ledger = _stub_world(tmp_path)
    (cache / "manifest.jsonl").write_text(
        '{"dataset": "butterfly_major", "fmt": "jpg", "cache_path": "b/x.jpg"}\n')
    with pytest.raises(AssertionError, match="lossily"):
        bt.plan_data(sst, cache, ledger)


def _build_layer(bt, layer, world, out):
    repo, sst, sam3, cache, ledger = world
    # the stub checkouts have no real git: make the recorded shas deterministic
    bt.git_head = lambda path: {"SST": "b" * 40, "sam3": "c" * 40}.get(path.name, "a" * 40)
    members, generated, missing = {}, {}, []
    if layer in ("data", "all"):
        m, g, missing = bt.plan_data(sst, cache, ledger)
        members |= m
        generated |= g
    if layer in ("code", "all"):
        m, g = bt.plan_code(repo, sam3)
        members |= m
        generated |= g
    return bt.build(out, layer, members, generated, missing, 11,
                    {"occl": "a" * 40, "sst": "b" * 40, "sam3": "c" * 40})


def test_two_layers_unpack_into_one_tree_and_wipe_keeps_the_data_layer(tmp_path):
    """The whole cycle: data + code unpacked over each other, wipe.sh drops the code
    layer and derived results and keeps everything a new iteration reuses, then the
    code layer unpacks again cleanly."""
    import subprocess
    import tarfile

    bt = _load_builder()
    world = _stub_world(tmp_path)
    data_tgz, code_tgz = tmp_path / "occl-scc-data.tar.gz", tmp_path / "occl-scc-code.tar.gz"
    md = _build_layer(bt, "data", world, data_tgz)
    mc = _build_layer(bt, "code", world, code_tgz)
    bt.verify(data_tgz, md)
    bt.verify(code_tgz, mc)
    proj = tmp_path / "projectnb" / "proj"
    proj.mkdir(parents=True)
    for t in (data_tgz, code_tgz):
        with tarfile.open(t) as tar:
            tar.extractall(proj)
    occl = proj / "occl"
    # the stub .git resolves the recorded sha, so env.json's sst_commit still works
    head = subprocess.run(["git", "-C", str(occl / "SST"), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    assert head.returncode == 0 and head.stdout.strip() == "b" * 40
    assert (occl / "sam3" / "COMMIT").read_text().strip() == "c" * 40
    assert json.loads((occl / "MANIFEST-data.json").read_text())["layer"] == "data"
    assert json.loads((occl / "MANIFEST-code.json").read_text())["layer"] == "code"
    # derived state a session would leave behind
    (occl / "results" / "runs" / "x.json").write_text("{}")
    (occl / "results" / "adapters" / "a.pt").write_text("w")
    (occl / "results" / "s14sam3.zip").write_text("z")
    (occl / "results" / "s14ft_sam3_beetle.zip").write_text("z")
    (occl / "ckpts" / "sam3.pt").write_text("ckpt")
    (occl / "hf" / "token").write_text("t")
    (occl / "venv").mkdir()
    r = subprocess.run(["bash", str(occl / "wipe.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not (occl / "oc-ccl-improvements").exists() and not (occl / "sam3").exists()
    assert not (occl / "results" / "runs").exists()
    assert not (occl / "results" / "s14sam3.zip").exists()
    assert not (occl / "results" / "s14ft_sam3_beetle.zip").exists()
    for kept in ("data/cache/manifest.jsonl", "SST/data/cambridge_butterfly/m.png",
                 "SST/.git/HEAD", "results/LEDGER-NEEDED.md", "ckpts/sam3.pt", "hf/token",
                 "venv", "MANIFEST-data.json"):
        assert (occl / kept).exists(), kept
    with tarfile.open(code_tgz) as tar:
        tar.extractall(proj)
    assert (occl / "oc-ccl-improvements" / "src" / "occl" / "scc.py").is_file()
    assert os.access(occl / "setup.sh", os.X_OK) and os.access(occl / "wipe.sh", os.X_OK)
    r = subprocess.run(["bash", str(occl / "wipe.sh"), "--all"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in occl.iterdir()) == ["wipe.sh"]


def test_verify_catches_a_missing_sentinel_per_layer(tmp_path):
    bt = _load_builder()
    world = _stub_world(tmp_path)
    repo, sst, sam3, cache, ledger = world
    members, generated = bt.plan_code(repo, sam3)
    del members["occl/oc-ccl-improvements/scripts/run_arms.py"]
    out = tmp_path / "bad.tar.gz"
    manifest = bt.build(out, "code", members, generated, [], 11, {})
    with pytest.raises(AssertionError, match="run_arms.py"):
        bt.verify(out, manifest)
    members, generated, missing = bt.plan_data(sst, cache, ledger)
    del members["occl/data/cache/manifest.jsonl"]
    manifest = bt.build(out, "data", members, generated, missing, 11, {})
    with pytest.raises(AssertionError, match="manifest.jsonl"):
        bt.verify(out, manifest)


def test_journal_is_written_on_every_event_and_survives_a_kill(tmp_path):
    """A wall-time SIGKILL leaves no cell output; the per-job timings must already
    be on disk. The journal is rewritten on each launch and exit, atomically."""
    w = World(ticks={"a": 2, "b": 3})
    jobs = [scc.Job("a", ["x"], gb=1.0, phase="fan"),
            scc.Job("b", ["x"], gb=1.0, phase="fan", deps=("a",))]
    jpath = tmp_path / "schedule.json"
    s = _sched(w, G40, journal=str(jpath))
    s.run(jobs, w.spawn)
    doc = json.loads(jpath.read_text())
    assert set(doc["jobs"]) == {"a", "b"}
    assert doc["jobs"]["a"]["rc"] == 0 and doc["jobs"]["a"]["ended"] is not None
    assert doc["jobs"]["b"]["phase"] == "fan" and doc["pending"] == 0
    assert not (tmp_path / "schedule.json.tmp").exists()
    # a journal under a running job records the launch before any exit
    w2 = World(ticks={"long": 100})
    calls = {"n": 0}

    def interrupt_after_launch(_s):
        w2.step += 1
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt

    s2 = scc.Scheduler(gpus=G40, max_procs=4, stagger_s=0, heartbeat_s=0,
                       sleep=interrupt_after_launch, clock=w2.clock,
                       log=lambda *_: None, journal=str(tmp_path / "j2.json"))
    with pytest.raises(KeyboardInterrupt):
        s2.run([scc.Job("long", ["x"], gb=1.0)], w2.spawn)
    doc2 = json.loads((tmp_path / "j2.json").read_text())
    assert doc2["jobs"]["long"]["ended"] is None and doc2["running"] == 1


def test_notebook_purges_corrupt_runs_before_building_the_graph():
    src = SCC_NB.read_text()
    assert "_purge_corrupt_runs(RUNS_DIR)" in src
    assert src.index("def _purge_corrupt_runs") < src.index("_purge_corrupt_runs(RUNS_DIR)")
    assert src.index("_purge_corrupt_runs(RUNS_DIR)") < src.index("_sched = scc.Scheduler(")
    assert 'journal=f"{LOG_DIR}/schedule.json"' in src


def test_kernel_bin_leads_path_before_any_subprocess_is_launched():
    """A Jupyter kernelspec execs the venv interpreter without activating the venv,
    so a bare `python` subprocess resolves to the module python3 (no torch). The
    notebook must prepend the kernel's bin dir to PATH — and assert it took —
    before the first run()/spawn call."""
    src = SCC_NB.read_text()
    fix = src.index('os.environ["PATH"] = _BIN + os.pathsep')
    check = src.index('assert _which.startswith(sys.prefix)')
    first_launch = min(src.index('run(["python"'), src.index("def spawn("))
    assert fix < check < first_launch


def test_exclusive_gpus_places_one_job_per_card_regardless_of_vram():
    """SCC's Exclusive_Process mode: a second CUDA context on a card fails at
    creation. With exclusive_gpus every job needs an idle card."""
    w = World(ticks={f"j{i}": 4 for i in range(6)})
    jobs = [scc.Job(f"j{i}", ["x"], gb=1.0) for i in range(6)]
    s = _sched(w, G40, exclusive_gpus=True)
    res = s.run(jobs, w.spawn)
    assert all(rc == 0 for rc in res.values())
    for t, g, step in w.launches:
        same_card_alive = [t2 for t2, g2, s2 in w.launches
                           if g2 == g and t2 != t and s2 <= step < s2 + 5]
        assert not same_card_alive, (t, g, same_card_alive)


def test_quick_failure_is_retried_once_then_skips_dependents():
    class Flaky(World):
        def spawn(self, job, gpu):
            n = sum(1 for t, _, _ in self.launches if t == job.tag)
            h = FakeHandle(1, 1 if (job.tag == "a" and n == 0) else 0)  # 1st try fails fast
            self.launches.append((job.tag, gpu, self.step))
            self.live[job.tag] = h
            return h

    w = Flaky()
    jobs = [scc.Job("a", ["x"], gb=1.0), scc.Job("m", ["x"], gb=1.0, deps=("a",))]
    s = _sched(w, G40, quick_fail_s=100)
    res = s.run(jobs, w.spawn)
    assert res == {"a": 0, "m": 0}
    assert [t for t, _, _ in w.launches] == ["a", "a", "m"]
    assert s.attempts["a"] == 2
    # a persistent failure exhausts the single retry and skips the dependent
    w2 = World(rc={"a": 1})
    s2 = _sched(w2, G40, quick_fail_s=100)
    res2 = s2.run([scc.Job("a", ["x"], gb=1.0), scc.Job("m", ["x"], gb=1.0, deps=("a",))],
                  w2.spawn)
    assert res2["a"] == 1 and res2["m"] == scc.SKIPPED
    assert [t for t, _, _ in w2.launches] == ["a", "a"]
    # a SLOW failure is a real defect: no retry
    w3 = World(ticks={"a": 200}, rc={"a": 1})
    s3 = _sched(w3, G40, quick_fail_s=100)
    res3 = s3.run([scc.Job("a", ["x"], gb=1.0)], w3.spawn)
    assert res3["a"] == 1 and [t for t, _, _ in w3.launches] == ["a"]


def test_compute_mode_parsing_and_exclusivity():
    assert scc.parse_compute_modes("Default\nDefault\n") == ["Default", "Default"]
    assert scc.gpus_are_exclusive(["Default", "Default"]) is False
    assert scc.gpus_are_exclusive(["Exclusive_Process"] * 4) is True
    assert scc.gpus_are_exclusive(["Default", "Prohibited"]) is True
    assert scc.gpus_are_exclusive([]) is False
    src = SCC_NB.read_text()
    assert "exclusive_gpus=ONE_JOB_PER_GPU" in src
    assert src.index("ONE_JOB_PER_GPU = scc.gpus_are_exclusive") < src.index("_sched = scc.Scheduler(")


# ── the S15 notebook: shared cells pinned to the s14sam3 SCC notebook ────────

S15_NB = REPO / "notebooks" / "run_s15_scc.py"
#: cells run_s15_scc.py copies VERBATIM from run_s14sam3_scc.py, by header line
SHARED_CELLS = ("# ── SCC session + repo", "# ── SST + sam3 checkouts",
                "# ── stage the image cache", "# ── GPU probe", "# ── shared machinery",
                "# ── SAM 3: gated checkpoint", "# ── export")


def _cells_by_header(path):
    out = {}
    for chunk in re.split(r"(?m)^(?=# %%)", path.read_text()):
        lines = chunk.splitlines()
        if len(lines) > 1:
            out[lines[1].strip()] = chunk.rstrip("\n")
    return out


def test_s15_shared_cells_are_byte_identical_to_the_s14_scc_notebook():
    """The SCC boilerplate (session, checkouts, cache, GPU probe, helpers,
    checkpoint, export) is copied, not shared — so a fix landing in one notebook
    and not the other fails HERE, not on the cluster."""
    a, b = _cells_by_header(SCC_NB), _cells_by_header(S15_NB)
    for h in SHARED_CELLS:
        ka = next((k for k in a if k.startswith(h)), None)
        kb = next((k for k in b if k.startswith(h)), None)
        assert ka and kb, f"cell {h!r} missing from {'s14' if not ka else 's15'}"
        assert a[ka] == b[kb], f"cell {h!r} drifted between the two SCC notebooks"


def test_s15_notebook_is_evaluation_only_on_the_reused_arch_adapter():
    src = S15_NB.read_text()
    tree = _ast_of(S15_NB)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "STAGE" for t in n.targets)]
    assert len(assigns) == 1 and assigns[0].value.value == "s15"
    assert "run_finetune.py" not in src, "s15 must never retrain — the sha pin would break"
    assert '"--shot", str(S15_SHOT)' in src and "S15_SHOT = 5" in src
    assert '"--shots", str(S15_SHOT)' in src, "commissioning must prove all 5 supports attend"
    assert "_purge_corrupt_runs(RUNS_DIR)" in src and "exclusive_gpus=ONE_JOB_PER_GPU" in src
    assert "google.colab" not in src
    assert not [ln for ln in src.splitlines() if re.match(r"^\s*(!(?!=)|%)", ln)]
    # the shot-lift anchors are the s14sam3 cells and the SAM 2 5-shot ledger is S6 + S8d
    for name in ("s14sam3_{_ds_name}_{_c}.zip", "s6_{_ds_name}.zip",
                 "s8d_{_ds_name}_base_tta.zip", "s8d_{_ds_name}_occl_tta.zip",
                 "s8d_{_ds_name}.zip", "--sam2-summary"):
        assert name in src, name


def test_s15_flags_and_scripts_match_the_code_layer():
    cmds = _commands_in(_ast_of(S15_NB))
    scripts = {s for s, _ in cmds}
    assert scripts == {"scripts/commission_s14.py", "scripts/run_arms.py",
                       "scripts/run_eval.py", "scripts/combine_s15.py"}
    for script, flags in cmds:
        unknown = sorted(flags - _argparse_flags(REPO / script))
        assert not unknown, f"{script} does not accept {unknown}"
    bt = _load_builder()
    launched = set()
    for nb in (SCC_NB, S15_NB, S17_NB, S17B_NB, S17C_NB, S17D_NB, S18_NB):
        launched |= {s.split("/", 1)[1] for s, _ in _commands_in(_ast_of(nb))}
    assert launched | {"scc_setup.sh"} == set(bt.CODE_SCRIPTS)
    assert set(bt.CODE_NOTEBOOKS) == {"run_s14sam3_scc.ipynb", "run_s15_scc.ipynb",
                                      "run_s17_scc.ipynb", "run_s17b_scc.ipynb",
                                      "run_s17c_scc.ipynb", "run_s17d_scc.ipynb",
                                      "run_s18_scc.ipynb"}


def test_wipe_keeps_the_butterfly_ledger():
    """`results/*ft*.zip` matched "butterfly" — a wipe would have deleted the S11/S6
    butterfly ledger cells. The fine-tune bundles are matched by prefix now."""
    bt = _load_builder()
    assert "*ft*" not in bt.WIPE_SH
    assert "results/s14ft_*.zip" in bt.WIPE_SH and "results/s15*.zip" in bt.WIPE_SH
    assert "results/s17*.zip" in bt.WIPE_SH       # s17.zip, s17_<ds>, s17ft_*, s17probe_*
    assert "results/s18*.zip" in bt.WIPE_SH


# ── the S17 notebook: the training-pair probe on SAM 2 ──────────────────────

S17_NB = REPO / "notebooks" / "run_s17_scc.py"
S17B_NB = REPO / "notebooks" / "run_s17b_scc.py"
S17C_NB = REPO / "notebooks" / "run_s17c_scc.py"
S17D_NB = REPO / "notebooks" / "run_s17d_scc.py"
S18_NB = REPO / "notebooks" / "run_s18_scc.py"
#: the family-neutral cells run_s17_scc.py copies VERBATIM from run_s15_scc.py. The
#: checkpoint cell is its own (SAM 2 is public, sha-pinned) and so is export (it
#: names s17ft bundles), so five of S15's seven shared cells are pinned here.
S17_SHARED_CELLS = ("# ── SCC session + repo", "# ── SST + sam3 checkouts",
                    "# ── stage the image cache", "# ── GPU probe", "# ── shared machinery")


def test_s17_shared_cells_are_byte_identical_to_the_s15_scc_notebook():
    a, b = _cells_by_header(S15_NB), _cells_by_header(S17_NB)
    for h in S17_SHARED_CELLS:
        ka = next((k for k in a if k.startswith(h)), None)
        kb = next((k for k in b if k.startswith(h)), None)
        assert ka and kb, f"cell {h!r} missing from {'s15' if not ka else 's17'}"
        assert a[ka] == b[kb], f"cell {h!r} drifted between the two SCC notebooks"
    # the pinned session cell carries REPO_MIN_BUILD: every SCC notebook says 18 now
    for nb in (SCC_NB, S15_NB, S17_NB):
        m = re.search(r"^REPO_MIN_BUILD = (\d+)", nb.read_text(), re.M)
        assert m and int(m.group(1)) == 18, nb.name
    import occl

    assert occl.BUILD >= 18


def test_s17_notebook_retrains_the_carrier_and_probes_every_pair_on_sam2():
    src = S17_NB.read_text()
    tree = _ast_of(S17_NB)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "STAGE" for t in n.targets)]
    assert len(assigns) == 1 and assigns[0].value.value == "s17"
    assert "google.colab" not in src
    assert not [ln for ln in src.splitlines() if re.match(r"^\s*(!(?!=)|%)", ln)]
    assert "_purge_corrupt_runs(RUNS_DIR)" in src and "exclusive_gpus=ONE_JOB_PER_GPU" in src
    assert src.index("ONE_JOB_PER_GPU = scc.gpus_are_exclusive") < src.index("_sched = scc.Scheduler(")
    assert src.index('os.environ["PATH"] = _BIN') < src.index("run([")
    # the SAM 2 checkpoint: public, fetched into ckpts/, pinned to the recorded digest
    assert "paths.SAM2_CKPT_URL" in src and "paths.SAM2_CKPT_SHA256" in src
    assert "sam3.pt" not in src.split("# ── SAM 2: public checkpoint")[1].split("# %%")[0]
    cmds = _commands_in(tree)
    by_script = {}
    for script, flags in cmds:
        by_script.setdefault(script, set()).update(flags)
    assert set(by_script) == {"scripts/run_finetune.py", "scripts/probe_train_pairs.py",
                              "scripts/combine_s17.py", "scripts/run_arms.py",
                              "scripts/run_eval.py"}
    # the fine-tunes: the S11 carrier on SAM 2 with its gradient diagnostics kept, plus
    # the arm flags — each ONE knob, spliced in from S17_ARMS (so they live outside the
    # command literal; checked against run_finetune's argparse below)
    ft = by_script["scripts/run_finetune.py"]
    assert {"--family", "--mask-mode", "--method", "--lr", "--epochs", "--checkpoint",
            "--lora-out", "--out"} <= ft
    assert not ({"--skip-grad-diagnostics", "--max-steps", "--scnp-width",
                 "--rotation-aug-deg", "--reference-run"} & ft)
    assert '"--family", "sam2"' in src and '"--mask-mode", "per_trait"' in src
    arms = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "S17_ARMS" for t in n.targets))
    arm_flags = {c.value for c in ast.walk(arms.value)
                 if isinstance(c, ast.Constant) and isinstance(c.value, str)
                 and c.value.startswith("--")}
    assert arm_flags == {"--found-mask", "--excl-mode", "--scope-preset"}
    assert '"arch": ["--scope-preset", "sam2_arch"]' in src
    assert arm_flags <= _argparse_flags(REPO / "scripts" / "run_finetune.py")
    assert "S17_CONTROL = \"carrier\"" in src
    # the probe replays from the CONTROL adapter's meta and joins every arm's record
    pr = by_script["scripts/probe_train_pairs.py"]
    assert {"--recipe-from", "--arms", "--checkpoint", "--eval-parity"} <= pr
    assert "--max-pairs" in pr        # the commissioning smoke only
    stage_cell = src.split("# ── the stage as ONE dependency graph")[1]
    assert '["--train-record", f"{_arm}={_ftb}"]' in stage_cell   # one record per arm
    assert "--max-pairs" not in stage_cell, "the real probe must cover every pair"
    assert "--groups\", _group" in stage_cell           # the eval fans slice by group
    assert '"--recipe-from", _adapters[_ds_name][S17_CONTROL]' in stage_cell
    # the test split: S11 protocol, 1-shot, 20 runs, both fans, per-cell merges
    ev = by_script["scripts/run_arms.py"]
    assert {"--shot", "--n-runs", "--support-selection", "--resume-dir", "--tta"} <= ev
    assert '_armspec += ["--arms", f"{_arm}={_pt}"]' in stage_cell   # every adapter rides
    assert '"--shot", "1", "--n-runs", "20"' in stage_cell
    assert '"--adapter", "sst_sam2"' in stage_cell
    assert "--lora" in by_script["scripts/run_eval.py"]
    assert "--control" in by_script["scripts/combine_s17.py"]
    # every worker gets the SCC checkpoint path — the default would look in SST/checkpoints
    for script, flags in cmds:
        if script != "scripts/combine_s17.py":
            assert "--checkpoint" in flags, script
    # reuse guards read the adapter's nested meta and the probe bundle's lora_sha
    assert '(_m.get("lora") or {}).get("scopes")' in src
    assert "checkpoint_hash(p) for a, p in _pts.items()" in src
    assert "S17_DATASETS = [\"beetle\", \"butterfly_major\"]" in src
    assert "S17_EVAL_DATASETS = [\"beetle\"]" in src


def test_s17_flags_match_their_target_scripts_and_make_notebook_emits_the_ipynb():
    for script, flags in _commands_in(_ast_of(S17_NB)):
        unknown = sorted(flags - _argparse_flags(REPO / script))
        assert not unknown, f"{script} does not accept {unknown}"
    ipynb = REPO / "notebooks" / "run_s17_scc.ipynb"
    assert ipynb.exists(), "run `python scripts/make_notebook.py`"
    nb = json.loads(ipynb.read_text())
    assert nb["metadata"]["kernelspec"]["name"] == "occl-sam3"
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert "".join(cells[0]["source"]).lstrip().startswith("# ── configuration")
    assert any("# ── SAM 2: public checkpoint" in "".join(c["source"]) for c in cells)


# ── the S17b notebook: the same contract, the forward-only / objectness arms ──

#: every cell S17b shares with S17 byte for byte — only the banner, the configuration
#: and the stage cell's names differ
S17B_SHARED_CELLS = S17_SHARED_CELLS + ("# ── SAM 2: public checkpoint", "# ── commissioning")


def test_s17b_shares_its_cells_with_s17_and_runs_the_s17b_arms():
    a, b = _cells_by_header(S17_NB), _cells_by_header(S17B_NB)
    for h in S17B_SHARED_CELLS:
        ka = next((k for k in a if k.startswith(h)), None)
        kb = next((k for k in b if k.startswith(h)), None)
        assert ka and kb, f"cell {h!r} missing from {'s17' if not ka else 's17b'}"
        assert a[ka] == b[kb], f"cell {h!r} drifted between the S17 and S17b notebooks"
    src = S17B_NB.read_text()
    tree = _ast_of(S17B_NB)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "STAGE" for t in n.targets)]
    assert len(assigns) == 1 and assigns[0].value.value == "s17b"
    m = re.search(r"^REPO_MIN_BUILD = (\d+)", src, re.M)
    assert m and int(m.group(1)) == 18
    arms = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "S17_ARMS" for t in n.targets))
    arm_flags = {c.value for c in ast.walk(arms.value)
                 if isinstance(c, ast.Constant) and isinstance(c.value, str)
                 and c.value.startswith("--")}
    assert arm_flags == {"--lora-legs", "--objness", "--excl-mode"}
    assert arm_flags <= _argparse_flags(REPO / "scripts" / "run_finetune.py")
    for arm in ("carrier", "excl", "fwdonly", "objness", "fwdonly_objness", "fwdonly_objness_excl"):
        assert f'"{arm}":' in src
    assert 'S17_DATASETS = ["beetle"]' in src
    # every bundle and adapter this stage writes is named after ITS stage, never s17's
    stage_cell = src.split("# ── the stage as ONE dependency graph")[1]
    assert "s17ft_sam2" not in stage_cell and "s17probe_" not in stage_cell
    assert 'f"{STAGE}ft_sam2_{_ds_name}_{_arm}.zip"' in stage_cell
    assert '"--stage", STAGE' in stage_cell and '_reset_{STAGE}_{_arm}.pt' in stage_cell
    for script, flags in _commands_in(tree):
        unknown = sorted(flags - _argparse_flags(REPO / script))
        assert not unknown, f"{script} does not accept {unknown}"
    ipynb = REPO / "notebooks" / "run_s17b_scc.ipynb"
    assert ipynb.exists(), "run `python scripts/make_notebook.py`"
    assert json.loads(ipynb.read_text())["metadata"]["kernelspec"]["name"] == "occl-sam3"


# ── the S17c notebook: the weight grid, the 2x2, arch/link, the 5-shot fan ──

S17C_SHARED_CELLS = S17B_SHARED_CELLS + ("# ── export",)


def test_s17c_shares_its_cells_with_s17b_and_declares_the_grid():
    a, b = _cells_by_header(S17B_NB), _cells_by_header(S17C_NB)
    for h in S17C_SHARED_CELLS:
        ka = next((k for k in a if k.startswith(h)), None)
        kb = next((k for k in b if k.startswith(h)), None)
        assert ka and kb, f"cell {h!r} missing from {'s17b' if not ka else 's17c'}"
        assert a[ka] == b[kb], f"cell {h!r} drifted between the S17b and S17c notebooks"
    src = S17C_NB.read_text()
    tree = _ast_of(S17C_NB)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "STAGE" for t in n.targets)]
    assert len(assigns) == 1 and assigns[0].value.value == "s17c"
    assert re.search(r"^REPO_MIN_BUILD = 18$", src, re.M)
    arms = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "S17_ARMS" for t in n.targets))
    arm_flags = {c.value for c in ast.walk(arms.value)
                 if isinstance(c, ast.Constant) and isinstance(c.value, str)
                 and c.value.startswith("--")}
    comp = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_COMP" for t in n.targets))
    arm_flags |= {c.value for c in ast.walk(comp.value)
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)
                  and c.value.startswith("--")}
    assert arm_flags == {"--lora-legs", "--objness", "--excl-mode", "--objness-weight",
                         "--lr", "--epochs", "--scope-preset"}
    assert arm_flags <= _argparse_flags(REPO / "scripts" / "run_finetune.py")
    for arm in ("carrier", "excl", "comp", "comp_w3", "comp_w10", "comp_w30", "comp_lr3e-5",
                "comp_ep3", "comp_lr3e-5_ep3", "comp_arch", "comp_link"):
        assert f'"{arm}":' in src, arm
    assert 'S17_SHOT5_ARMS = ["comp"]' in src
    stage_cell = src.split("# ── the stage as ONE dependency graph")[1]
    assert '"--shot", "5"' in stage_cell and "fan_plain5" in stage_cell
    assert "_arm_knobs(_flags)" in stage_cell and "_arm_epochs(S17_ARMS[_ds_name][_arm])" in stage_cell
    assert "s17bft_sam2" not in stage_cell and "s17bprobe_" not in stage_cell
    for script, flags in _commands_in(tree):
        unknown = sorted(flags - _argparse_flags(REPO / script))
        assert not unknown, f"{script} does not accept {unknown}"
    ipynb = REPO / "notebooks" / "run_s17c_scc.ipynb"
    assert ipynb.exists(), "run `python scripts/make_notebook.py`"
    assert json.loads(ipynb.read_text())["metadata"]["kernelspec"]["name"] == "occl-sam3"


def test_arm_flag_helpers_in_s17c_strip_lr_and_epochs_correctly():
    """The notebook's helpers, executed: --lr/--epochs go to the command explicitly and
    are stripped from the knob list; the rest passes through in order."""
    src = S17C_NB.read_text()
    ns = {"S17_LR": 1e-5, "S17_EPOCHS": 1}
    for name in ("_arm_after", "_arm_lr", "_arm_epochs", "_arm_weight", "_arm_legs", "_arm_knobs"):
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "nb", "exec"), ns)
    f = ["--lora-legs", "forward", "--objness", "--excl-mode", "zerocat", "--lr", "3e-05", "--epochs", "3"]
    assert ns["_arm_lr"](f) == 3e-5 and ns["_arm_epochs"](f) == 3
    assert ns["_arm_knobs"](f) == ["--lora-legs", "forward", "--objness", "--excl-mode", "zerocat"]
    assert ns["_arm_weight"](["--objness", "--objness-weight", "10"]) == 10.0
    assert ns["_arm_legs"](["--lora-legs", "forward_link"]) == "forward_link" and ns["_arm_legs"]([]) is None
    assert ns["_arm_lr"]([]) == 1e-5 and ns["_arm_epochs"]([]) == 1 and ns["_arm_knobs"]([]) == []



# ── the S17d notebook: comp_arch x w10 and the soft sentinel ─────────────────

#: S17d shares every S17c cell except commissioning (it also commissions the soft gate)
S17D_SHARED_CELLS = tuple(h for h in S17C_SHARED_CELLS if h != "# ── commissioning")


def test_s17d_shares_its_cells_with_s17c_and_declares_the_soft_arms():
    a, b = _cells_by_header(S17C_NB), _cells_by_header(S17D_NB)
    for h in S17D_SHARED_CELLS:
        ka = next((k for k in a if k.startswith(h)), None)
        kb = next((k for k in b if k.startswith(h)), None)
        assert ka and kb, f"cell {h!r} missing from {'s17c' if not ka else 's17d'}"
        assert a[ka] == b[kb], f"cell {h!r} drifted between the S17c and S17d notebooks"
    src = S17D_NB.read_text()
    tree = _ast_of(S17D_NB)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "STAGE" for t in n.targets)]
    assert len(assigns) == 1 and assigns[0].value.value == "s17d"
    assert re.search(r"^REPO_MIN_BUILD = 18$", src, re.M)
    flags = set()
    for name in ("S17_ARMS", "_COMP", "_SOFT"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        flags |= {c.value for c in ast.walk(node.value)
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)
                  and c.value.startswith("--")}
    assert flags == {"--lora-legs", "--objness", "--excl-mode", "--objness-weight",
                     "--scope-preset", "--soft-sentinel"}
    assert flags <= _argparse_flags(REPO / "scripts" / "run_finetune.py")
    for arm in ("comp", "comp_arch_w10", "fwdonly_excl_soft", "comp_soft", "comp_arch_w10_soft"):
        assert f'"{arm}":' in src, arm
    assert 'S17_SHOT5_ARMS = ["comp_arch_w10"]' in src and 'S17_CONTROL = "comp"' in src
    # the soft gate is commissioned on the real model before the graph: a two-step
    # fine-tune under it, then the probe proving the step-0 loss under the SOFT gate
    comm = src.split("# ── commissioning")[1].split("# ── the stage as ONE dependency graph")[0]
    assert '"--soft-sentinel", "20"' in comm and '"--max-steps", "2"' in comm
    assert '_rs["arm_order"] == ["base", "base_soft20", "soft"]' in comm
    assert '_pv["base_arm"] == "base_soft20"' in comm and "loss_first_rel_err" in comm
    # reuse: a banked adapter must match on the gate too
    assert '"soft_sentinel": _arm_soft(_flags)' in src
    assert '_o.get("soft_sentinel") == _want["soft_sentinel"]' in src
    stage_cell = src.split("# ── the stage as ONE dependency graph")[1]
    assert "s17cft_sam2" not in stage_cell and "s17cprobe_" not in stage_cell
    assert '"--shot", "5"' in stage_cell and "fan_plain5" in stage_cell
    for script, flags in _commands_in(tree):
        unknown = sorted(flags - _argparse_flags(REPO / script))
        assert not unknown, f"{script} does not accept {unknown}"
    ipynb = REPO / "notebooks" / "run_s17d_scc.ipynb"
    assert ipynb.exists(), "run `python scripts/make_notebook.py`"
    assert json.loads(ipynb.read_text())["metadata"]["kernelspec"]["name"] == "occl-sam3"
    # the helpers, executed: the gate value is read from the flag list as a float
    ns = {"S17_LR": 1e-5, "S17_EPOCHS": 1}
    for name in ("_arm_after", "_arm_soft", "_arm_knobs"):
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "nb", "exec"), ns)
    assert ns["_arm_soft"](["--lora-legs", "forward", "--soft-sentinel", "20"]) == 20.0
    assert ns["_arm_soft"](["--lora-legs", "forward"]) is None
    assert ns["_arm_knobs"](["--soft-sentinel", "20", "--lr", "3e-05"]) == ["--soft-sentinel", "20"]


# ── the S18 notebook: the cycle's information bottleneck ────────────────────

#: S18 shares every S17d cell except commissioning (it commissions the S18 knobs) and
#: export (its fine-tune gate is `s18ft`, not `s17ft` — pinned by text below)
S18_SHARED_CELLS = tuple(h for h in S17D_SHARED_CELLS
                         if h not in ("# ── commissioning", "# ── export"))


def test_s18_shares_its_cells_with_s17d_and_moves_the_crown_along_s18_axes_only():
    a, b = _cells_by_header(S17D_NB), _cells_by_header(S18_NB)
    for h in S18_SHARED_CELLS:
        ka = next((k for k in a if k.startswith(h)), None)
        kb = next((k for k in b if k.startswith(h)), None)
        assert ka and kb, f"cell {h!r} missing from {'s17d' if not ka else 's18'}"
        assert a[ka] == b[kb], f"cell {h!r} drifted between the S17d and S18 notebooks"
    src = S18_NB.read_text()
    tree = _ast_of(S18_NB)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "STAGE" for t in n.targets)]
    assert len(assigns) == 1 and assigns[0].value.value == "s18"
    assert re.search(r"^REPO_MIN_BUILD = 18$", src, re.M)
    flags = set()
    for name in ("S17_ARMS", "_S18_ARMS", "_CROWN", "_STE", "_STEMEM", "_PERTURB", "_HOT"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        flags |= {c.value for c in ast.walk(node.value)
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)
                  and c.value.startswith("--")}
    # every arm is `--recipe sam2_crown` moved along the S18 axes — no S17 knob is
    # spelled out in the notebook (the recipe carries them)
    assert flags == {"--recipe", "--prompt-mode", "--close-aug-deg", "--lora-legs", "--lr",
                     "--epochs"}
    assert flags <= _argparse_flags(REPO / "scripts" / "run_finetune.py")
    for arm in ("crown", "crown_ste", "crown_stemem", "crown_perturb", "crown_all",
                "crown_all_perturb", "crown_hot", "crown_stemem_hot"):
        assert f'"{arm}":' in src, arm
    assert 'S17_SHOT5_ARMS: list[str] = []' in src and 'S17_CONTROL = "crown"' in src
    assert 'S17_DATASETS = ["beetle", "butterfly_major"]' in src
    assert 'S17_EVAL_DATASETS = ["beetle", "butterfly_major"]' in src
    comm = src.split("# ── commissioning")[1].split("# ── the stage as ONE dependency graph")[0]
    assert '"--recipe", "sam2_crown", "--prompt-mode", "ste_memory", "--close-aug-deg", "10"' in comm
    assert "--skip-grad-diagnostics" not in comm.split("scripts/run_finetune.py")[1].split("])")[0]
    assert 'assert all(m > 0.1 * c for m, c, f in zip(_pm, _pr, _fd) if f)' in comm
    assert '_rs["arm_order"] == ["base", "base_close10", "ste_close"]' in comm
    assert '_pv["column"] == "hard"' in comm and "cls_miou_hard" in comm
    # reuse: a banked adapter must match on the prompt mode and the closing end too,
    # read THROUGH the recipe expansion
    assert '"prompt_mode": _arm_after(_flags, "--prompt-mode", "logit_reparam")' in src
    assert '"close_aug": _arm_close(_flags)' in src
    assert "_flags = expand_recipe(list(_flags))" in src
    stage_cell = src.split("# ── the stage as ONE dependency graph")[1]
    assert "s17dft_sam2" not in stage_cell and "s17dprobe_" not in stage_cell
    # the arm's knobs go BEFORE the explicit lr/epochs so the notebook's values win
    assert '*_arm_knobs(_flags),\n                 "--family", "sam2",' in stage_cell
    assert "if not S17_SHOT5_ARMS:" in stage_cell
    assert 'check_gate.py {STAGE}ft --bundle' in src
    for script, flags in _commands_in(tree):
        unknown = sorted(flags - _argparse_flags(REPO / script))
        assert not unknown, f"{script} does not accept {unknown}"
    ipynb = REPO / "notebooks" / "run_s18_scc.ipynb"
    assert ipynb.exists(), "run `python scripts/make_notebook.py`"
    assert json.loads(ipynb.read_text())["metadata"]["kernelspec"]["name"] == "occl-sam3"
    # the helpers, executed: the LAST occurrence wins, the recipe expands in front,
    # "all" legs read as None (what the record writes)
    from occl.registry import SAM2_CROWN_FLAGS, expand_recipe

    ns = {"S17_LR": 1e-5, "S17_EPOCHS": 1, "expand_recipe": expand_recipe}
    for name in ("_arm_after", "_arm_lr", "_arm_epochs", "_arm_weight", "_arm_legs",
                 "_arm_close", "_arm_soft", "_arm_knobs"):
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "nb", "exec"), ns)
    crown = ["--recipe", "sam2_crown"]
    assert ns["_arm_lr"](crown) == 1e-5 and ns["_arm_epochs"](crown) == 1
    assert ns["_arm_lr"](crown + ["--lr", "3e-05", "--epochs", "3"]) == 3e-5
    assert ns["_arm_epochs"](crown + ["--lr", "3e-05", "--epochs", "3"]) == 3
    ex = expand_recipe(crown + ["--lora-legs", "all"])
    assert ns["_arm_legs"](ex) is None and ns["_arm_legs"](expand_recipe(crown)) == "forward"
    assert ns["_arm_weight"](expand_recipe(crown)) == 10.0
    assert ns["_arm_close"](expand_recipe(crown + ["--close-aug-deg", "10"])) == 10.0
    assert ns["_arm_close"](expand_recipe(crown)) is None
    assert ns["_arm_after"](expand_recipe(crown + ["--prompt-mode", "ste_threshold"]),
                            "--prompt-mode", "logit_reparam") == "ste_threshold"
    assert ns["_arm_knobs"](crown + ["--lr", "3e-05", "--epochs", "3"]) == crown
    assert list(SAM2_CROWN_FLAGS[:2]) == ["--family", "sam2"]
