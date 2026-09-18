"""Guards on the standalone s14sam3 notebook (notebooks/run_s14sam3.py).

The standalone exists by user decision after the s14sec sessions: a stage whose
dependency stack conflicts with the others' gets a dedicated pipeline that can
never run the wrong stage. Its price is COPIED machinery — the helper functions
are verbatim copies of run_stage.py's — and the tests here make that price
explicit: every copied definition is pinned to the original by AST equality, so
a fix landing in one file and not the other fails HERE, not in Colab.
"""

from __future__ import annotations

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
MULTI = REPO / "notebooks" / "run_stage.py"
STANDALONE = REPO / "notebooks" / "run_s14sam3.py"

#: every top-level definition the standalone copies from run_stage.py
COPIED = ["worker_env", "GROUPS_OF", "PEAK_GB", "EMBED_MB_PER_IMAGE",
          "TEST_PER_GROUP", "cache_gb", "CALIBRATION", "_cal_key",
          "_calibration_path", "load_calibration", "TTA_VRAM_FACTOR",
          "per_worker_gb", "run_slices", "plan_group_workers", "run",
          "banked_runs", "wait_all"]


def _ast_of(path):
    """Parse a percent-format notebook source, neutralising !shell / %magic lines
    (the same treatment as test_fanout.notebook_ast, kept local so the two guard
    files stay independent)."""
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
    """AST equality, name by name. A drifted copy is worse than a missing one:
    it looks maintained and silently diverges — the exact failure the
    one-parameterized-notebook rule existed to prevent, re-earned here as a test
    because the user chose isolation for this stage."""
    multi, alone = _top_level(_ast_of(MULTI)), _top_level(_ast_of(STANDALONE))
    missing = [n for n in COPIED if n not in alone]
    assert not missing, f"standalone lost {missing}"
    drifted = [n for n in COPIED
               if ast.dump(multi[n]) != ast.dump(alone[n])]
    assert not drifted, (
        f"{drifted} differ between run_stage.py and run_s14sam3.py — edit "
        "run_stage.py, re-copy the block verbatim into the standalone")


def test_standalone_workers_use_worker_env_and_run_spreads_environ():
    """The same Popen discipline test_fanout enforces on run_stage.py, applied
    to the standalone: fan-out workers get worker_env() (BLAS caps), and run()'s
    tee-Popen extends os.environ rather than replacing it."""
    tree = _ast_of(STANDALONE)
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
            assert (isinstance(env, ast.Dict)
                    and any(k is None for k in env.keys)), \
                "run()'s Popen must spread os.environ into its env"
            continue
        launches.append((node.lineno, env))
    assert launches, "no fan-out Popen found in the standalone"
    bad = [ln for ln, env in launches
           if not (isinstance(env, ast.Call) and isinstance(env.func, ast.Name)
                   and env.func.id == "worker_env")]
    assert not bad, f"Popen at line(s) {bad} does not pass env=worker_env()"


def test_stage_ownership_is_exclusive():
    """s14sam3 lives ONLY in the standalone: run_stage.py must not dispatch it
    (a re-uploaded default once sent a session into the wrong stage), and the
    standalone must pin the stage constant and never reassign it."""
    multi_src = MULTI.read_text()
    stages = multi_src.split("STAGES = (", 1)[1].split(")", 1)[0]
    assert '"s14sam3"' not in stages, "run_stage.py still dispatches s14sam3"

    tree = _ast_of(STANDALONE)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "STAGE"
                       for t in n.targets)]
    assert len(assigns) == 1, f"STAGE assigned {len(assigns)} times"
    assert isinstance(assigns[0].value, ast.Constant) \
        and assigns[0].value.value == "s14sam3"


def test_standalone_carries_the_arch_arm_and_its_scopes():
    """The architecture-adapted arm is the standalone's reason to exist beyond
    isolation: the scope tuples must match occl.finetune.lora exactly, and the
    TTA fan must carry all three arms."""
    from occl.finetune.lora import FAITHFUL_SCOPES_BY_FAMILY, SAM3_ARCH_SCOPES

    defs = _top_level(_ast_of(STANDALONE))
    assert ast.literal_eval(defs["_SCOPES_STACK"].value) \
        == FAITHFUL_SCOPES_BY_FAMILY["sam3"]
    assert ast.literal_eval(defs["_SCOPES_ARCH"].value) == SAM3_ARCH_SCOPES
    src = STANDALONE.read_text()
    assert '"--scope-preset", _preset' in src
    assert '"--arms", f"arch={_apt}"' in src


def test_reuse_check_reads_scopes_where_save_lora_writes_them():
    """save_lora packs record["finetune"] wholesale, so scopes live NESTED at
    meta["lora"]["scopes"]. The standalone's reuse check must read that path —
    a top-level read never matches and silently retrains every session (found
    by the second debug sweep, before it cost a session)."""
    src = STANDALONE.read_text()
    assert '(_m.get("lora") or {}).get("scopes")' in src
