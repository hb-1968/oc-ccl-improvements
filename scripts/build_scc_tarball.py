#!/usr/bin/env python3
"""Build separate SCC data and code archives with provenance manifests.

Upload the data layer once and replace the code layer between experiments.
Never run wipe.sh during an active stage: it removes derived runs and adapters.
Model weights and the original image downloads are not included.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

TOP = "occl"                                    # the single directory both layers unpack to
LAYERS = ("data", "code", "all")

#: the scripts the SCC notebook launches (tests/test_scc.py pins this against the
#: notebook's own command literals) plus the one setup.sh runs
CODE_SCRIPTS = ("commission_s14.py", "run_finetune.py", "run_arms.py", "run_eval.py",
                "combine_s14.py", "combine_s15.py", "probe_train_pairs.py",
                "combine_s17.py", "scc_setup.sh")
CODE_NOTEBOOKS = ("run_s14sam3_scc.ipynb", "run_s15_scc.ipynb", "run_s17_scc.ipynb",
                  "run_s17b_scc.ipynb", "run_s17c_scc.ipynb", "run_s17d_scc.ipynb", "run_s18_scc.ipynb")
CODE_NOTEBOOK = CODE_NOTEBOOKS[0]
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv", ".venv-sec",
                ".mypy_cache", ".ruff_cache", ".git"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")

LEDGER_CELLS = ("base", "base_tta", "occl_pertrait_tta")
DATASETS = ("butterfly_major", "beetle")


def ledger_names() -> list[str]:
    """The banked SAM 2 cells the SCC stages pair against and never recompute:
    S11 (1-shot, for s14sam3) and the 5-shot ledger for s15 — S6 plain (per-run) and
    the S8d summary (its TTA cells were never banked per-run); all local, results/inbox."""
    s11 = [f"s11_{ds}_{cell}.zip" for ds in DATASETS for cell in LEDGER_CELLS]
    s15 = [f"s6_{ds}.zip" for ds in DATASETS] + [f"s8d_{ds}.zip" for ds in DATASETS]
    return s11 + s15


def git_head(repo: Path) -> str:
    try:
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return bool(set(rel.parts) & EXCLUDE_DIRS
                or any(p.endswith(".egg-info") for p in rel.parts)
                or path.suffix in EXCLUDE_SUFFIXES)


def _add_tree(members: dict[str, Path], src: Path, arc: str) -> None:
    assert src.is_dir(), f"{src} is not a directory"
    for p in sorted(src.rglob("*")):
        if p.is_file() and not _skip(p, src):
            members[f"{arc}/{p.relative_to(src).as_posix()}"] = p


# the two plans — pure; the tests run them on stub trees
def plan_data(sst: Path, cache_dir: Path | None, ledger_dir: Path | None
              ) -> tuple[dict[str, Path], dict[str, str], list[str]]:
    """(members, generated_text_files, missing_ledger) for the DATA layer."""
    members: dict[str, Path] = {}
    generated: dict[str, str] = {}

    assert (sst / "data" / "cambridge_butterfly").is_dir(), f"{sst}: not an SST checkout"
    _add_tree(members, sst / "data", f"{TOP}/SST/data")
    _add_tree(members, sst / "src", f"{TOP}/SST/src")
    members[f"{TOP}/SST/README.md"] = sst / "README.md"
    # a stub repo: HEAD + the two dirs git needs to recognise it, no objects. `git
    # rev-parse HEAD` resolves it, so every bundle's env.json still names the commit.
    generated[f"{TOP}/SST/.git/HEAD"] = git_head(sst) + "\n"

    if cache_dir is not None:
        manifest = cache_dir / "manifest.jsonl"
        assert manifest.is_file(), f"{cache_dir} has no manifest.jsonl"
        lossy = sorted({(e.get("dataset"), e.get("fmt"))
                        for e in map(json.loads, manifest.read_text().splitlines())
                        if e.get("fmt") not in ("png", "webp")})
        assert not lossy, (f"{cache_dir} stores {lossy} lossily — ship the PNG-era cache "
                           "(oc-ccl-data/cache-png), never the superseded JPEG one")
        _add_tree(members, cache_dir, f"{TOP}/data/cache")

    missing = []
    for name in ledger_names():
        src = ledger_dir / name if ledger_dir else None
        if src is not None and src.is_file():
            members[f"{TOP}/results/{name}"] = src
        else:
            missing.append(name)
    if missing:
        generated[f"{TOP}/results/LEDGER-NEEDED.md"] = LEDGER_NEEDED.format(
            names="\n".join(f"- {n}" for n in missing))
    return members, generated, missing


def plan_code(repo: Path, sam3: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """(members, generated_text_files) for the CODE layer."""
    members: dict[str, Path] = {}
    generated: dict[str, str] = {}
    r = f"{TOP}/oc-ccl-improvements"

    _add_tree(members, repo / "src", f"{r}/src")
    members[f"{r}/pyproject.toml"] = repo / "pyproject.toml"
    for s in CODE_SCRIPTS:
        p = repo / "scripts" / s
        assert p.is_file(), f"missing worker script {p}"
        members[f"{r}/scripts/{s}"] = p
    for name in CODE_NOTEBOOKS:
        nb = repo / "notebooks" / name
        assert nb.is_file(), f"{nb} missing — run scripts/make_notebook.py"
        members[f"{r}/notebooks/{name}"] = nb

    assert (sam3 / "sam3" / "model" / "sam3_tracking_predictor.py").is_file(), \
        f"{sam3}: not a sam3 checkout"
    _add_tree(members, sam3 / "sam3", f"{TOP}/sam3/sam3")
    if (sam3 / "LICENSE").is_file():
        members[f"{TOP}/sam3/LICENSE"] = sam3 / "LICENSE"
    generated[f"{TOP}/sam3/COMMIT"] = git_head(sam3) + "\n"
    return members, generated


# generated files
SETUP_SH = """#!/usr/bin/env bash
# Thin wrapper: the real work is oc-ccl-improvements/scripts/scc_setup.sh, which
# infers the SCC project from where this archive was unpacked (/projectnb/<project>/occl).
set -euo pipefail
cd "$(dirname "$0")"
exec bash oc-ccl-improvements/scripts/scc_setup.sh "$@"
"""

WIPE_SH = """#!/usr/bin/env bash
# Remove the CODE layer and everything derived from it, keeping what the next
# iteration reuses unchanged. Then unpack the new occl-scc-code.tar.gz and re-run setup.sh.
#
#   bash occl/wipe.sh            code + derived results (the per-iteration reset)
#   bash occl/wipe.sh --data     ... and the data layer (cache, SST, ledger)
#   bash occl/wipe.sh --all      ... and ckpts/, hf/, venv/ — back to an empty project dir
#
# KEPT by default: data/ SST/ results/s11_*.zip ckpts/ hf/ venv/
# REMOVED by default:
#   oc-ccl-improvements/ sam3/            the code layer
#   results/runs/ adapters/ logs/         per-run JSON, LoRA weights, worker logs
#   results/s14*.zip results/s15*.zip results/s17*.zip results/s18*.zip results/s14ft_*.zip   stage + cell + fine-tune bundles
# Per-run JSON is keyed by a source fingerprint, so a changed occl.finetune would already
# refuse to resume from it — wiping just makes the invalidation visible instead of silent.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f MANIFEST-code.json || -f MANIFEST-all.json || -d oc-ccl-improvements ]] \\
  || { echo "not an occl project dir: $(pwd)" >&2; exit 1; }
mode="${1:-}"
rm -rf oc-ccl-improvements sam3 results/runs results/adapters results/logs
rm -f results/s14*.zip results/s15*.zip results/s17*.zip results/s18*.zip results/s14ft_*.zip setup.sh README-SCC.md MANIFEST-code.json MANIFEST-all.json
echo "wiped code layer + derived results in $(pwd)"
if [[ "$mode" == "--data" || "$mode" == "--all" ]]; then
  rm -rf data SST results/s11_*.zip results/LEDGER-NEEDED.md MANIFEST-data.json
  echo "wiped data layer"
fi
if [[ "$mode" == "--all" ]]; then
  rm -rf ckpts hf venv tmp results
  echo "wiped checkpoints, HF cache, venv — $(pwd) is empty"
fi
"""

LEDGER_NEEDED = """# S11 SAM 2 ledger — REQUIRED before the first session

The SCC notebook pairs every SAM 3 cell against S11's banked SAM 2 cells and never
recomputes them. These six files were not available when the archive was built.
Copy them into THIS directory (they are on Google Drive under `occl/results/`):

{names}

The notebook's "banked SAM 2 ledger" cell asserts each one before any fine-tune starts.
"""


def readme_scc(build: int, layer: str, missing: list[str]) -> str:
    ledger = ("included at results/" if not missing
              else f"MISSING {len(missing)}/6 — see results/LEDGER-NEEDED.md")
    return f"""# occl on the BU SCC — unpacked layout (occl build {build}, from the `{layer}` archive)

You are in `/projectnb/<project>/occl` (`OCCL_PROJECT_ROOT`). Two layers unpack here:

| layer | path | what | when it changes |
| :-- | :-- | :-- | :-- |
| data | `data/cache/` | the lossless PNG image cache the workers read | never (pixels) |
| data | `SST/` | masks + split manifests + vendored SAM 2 (no weights) | never (labels) |
| data | `results/s11_*.zip` | the S11 SAM 2 ledger the combine pairs against ({ledger}) | never (the anchor) |
| code | `oc-ccl-improvements/` | `src/occl`, the worker scripts, `notebooks/run_s14sam3_scc.ipynb` + `run_s15_scc.ipynb` + `run_s17_scc.ipynb` + `run_s17b_scc.ipynb` + `run_s17c_scc.ipynb` + `run_s17d_scc.ipynb` | every OC-CCL iteration |
| code | `sam3/` | the tracker package, consumed as a checkout | rarely |
| — | `ckpts/sam3.pt` | HF-gated, downloaded once by the notebook | needs `hf/token` |
| — | `ckpts/sam2_hiera_large.pt` | public, downloaded once by the s17 notebook (sha256-pinned) | outbound access |
| — | `venv/`, `hf/`, `tmp/` | built by `setup.sh`; survive a wipe | — |

## First session

1. `bash setup.sh` — from a LOGIN node. Builds `venv/`, registers the Jupyter kernel
   **occl (sam3)**, prints exactly which prerequisites are still missing.
2. `echo <hf-token> > hf/token` (request access at huggingface.co/facebook/sam3 first).
3. OnDemand (scc-ondemand.bu.edu) → Interactive Apps → Jupyter Notebook: Pre-Launch
   Command `export OCCL_PROJECT=<project>`; Working Directory
   `/projectnb/<project>/occl/oc-ccl-improvements`; 2–4 GPUs at compute capability
   ≥ 8.0; ~4 cores per GPU; 8+ hours.
4. Open `notebooks/run_s14sam3_scc.ipynb`, kernel **occl (sam3)**, Run All. Banked
   under `results/runs/`, so a session that hits its wall-time resumes on the next Run All.
5. The bundle lands at `results/s14sam3.zip`; `scp` it to `results/inbox/` on the machine
   that holds the full repo and run `python scripts/check_gate.py s14sam3`.

## Each later iteration (a change to OC-CCL's loss / fine-tuning)

```
bash wipe.sh                      # drops the code layer + everything derived from it
tar xzf ../occl-scc-code.tar.gz   # the new code layer (built by scripts/build_scc_tarball.py --layer code)
bash setup.sh                     # re-links the venv's editable install; seconds
```

`wipe.sh --data` also drops the data layer; `wipe.sh --all` empties the project dir.
`MANIFEST-<layer>.json` records what each archive contained and the build it came from.
"""


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(out: Path, layer: str, members: dict[str, Path], generated: dict[str, str],
          missing: list[str], build_no: int, shas: dict[str, str]) -> dict:
    t0 = time.time()
    by_top: dict[str, dict[str, float]] = {}
    for arc, src in members.items():
        key = arc.split("/", 2)[1]
        d = by_top.setdefault(key, {"files": 0, "bytes": 0})
        d["files"] += 1
        d["bytes"] += src.stat().st_size
    manifest = {
        "archive": out.name, "layer": layer, "top": TOP, "occl_build": build_no,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "commits": shas,
        "components": by_top, "ledger_missing": missing if layer != "code" else None,
        "generated": sorted(generated),
    }
    generated = dict(generated)
    generated[f"{TOP}/setup.sh"] = SETUP_SH
    generated[f"{TOP}/wipe.sh"] = WIPE_SH
    generated[f"{TOP}/README-SCC.md"] = readme_scc(build_no, layer, missing)
    generated[f"{TOP}/MANIFEST-{layer}.json"] = json.dumps(manifest, indent=1) + "\n"
    dirs = ["ckpts", "hf", "tmp", "results", "data", "results/runs", "results/adapters",
            "results/logs"]
    if layer != "code":
        dirs += ["SST/.git/objects", "SST/.git/refs"]   # what git needs to see a repo

    now = int(time.time())
    # gzip level 1: the data payload is ~2.8 GB of already-compressed PNG, so a
    # higher level buys nothing and costs minutes.
    with tarfile.open(out, "w:gz", compresslevel=1) as tar:
        for d in dirs:
            info = tarfile.TarInfo(f"{TOP}/{d}")
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, now
            tar.addfile(info)
        for arc, src in members.items():
            tar.add(src, arcname=arc, recursive=False)
        for name, text in generated.items():
            info = tarfile.TarInfo(name)
            data = text.encode()
            info.size, info.mtime = len(data), now
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
    manifest["seconds"] = round(time.time() - t0, 1)
    manifest["archive_bytes"] = out.stat().st_size
    return manifest


def verify(out: Path, manifest: dict) -> None:
    """Re-list the archive and assert every path the notebook / setup.sh asserts."""
    layer = manifest["layer"]
    with tarfile.open(out, "r:gz") as tar:
        names = set(tar.getnames())
    want = [f"{TOP}/setup.sh", f"{TOP}/wipe.sh", f"{TOP}/README-SCC.md",
            f"{TOP}/MANIFEST-{layer}.json", f"{TOP}/ckpts", f"{TOP}/hf"]
    if layer in ("code", "all"):
        r = f"{TOP}/oc-ccl-improvements"
        want += [f"{r}/src/occl/scc.py", f"{r}/src/occl/adapters/sam3.py",
                 f"{r}/src/occl/finetune/train.py", f"{r}/pyproject.toml",
                 *[f"{r}/notebooks/{n}" for n in CODE_NOTEBOOKS],
                 *[f"{r}/scripts/{s}" for s in CODE_SCRIPTS],
                 f"{TOP}/sam3/sam3/model/sam3_tracking_predictor.py", f"{TOP}/sam3/COMMIT"]
    if layer in ("data", "all"):
        want += [f"{TOP}/SST/README.md", f"{TOP}/SST/.git/HEAD", f"{TOP}/SST/.git/objects",
                 f"{TOP}/data/cache/manifest.jsonl"]
        if not manifest["ledger_missing"]:
            want += [f"{TOP}/results/{n}" for n in ledger_names()]
        else:
            want.append(f"{TOP}/results/LEDGER-NEEDED.md")
        assert any(n.startswith(f"{TOP}/SST/data/cambridge_butterfly/") for n in names)
    absent = [w for w in want if w not in names]
    assert not absent, f"archive is missing {absent}"
    assert not any(n.endswith((".pt", ".pth")) for n in names), "a checkpoint slipped in"
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert not any("/.git/" in n and not n.startswith(f"{TOP}/SST/.git/") for n in names), \
        "an object store slipped in"
    stray = [n for n in names if not n.startswith(TOP + "/") and n != TOP]
    assert not stray, f"members outside {TOP}/: {stray[:5]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layer", choices=LAYERS + ("both",), default="both",
                    help="data (once), code (per iteration), all (one combined archive), "
                         "or both = data + code as two files (default)")
    ap.add_argument("--out-dir", default=str(REPO.parent),
                    help="where the archives go (default: the repo's parent, so an archive "
                         "can never be built into itself)")
    ap.add_argument("--cache-dir", default=str(REPO / "oc-ccl-data" / "cache-png"),
                    help="the lossless PNG cache folder (manifest.jsonl inside)")
    ap.add_argument("--ledger-dir", default=str(REPO / "results" / "inbox"),
                    help="where the six s11_<dataset>_<cell>.zip files are")
    ap.add_argument("--sst", default=os.environ.get("OCCL_SST_ROOT", str(REPO / "SST")))
    ap.add_argument("--sam3", default=os.environ.get("OCCL_SAM3_ROOT", str(REPO / "sam3")))
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    # The ipynb is generated FROM the tree being archived, in the same invocation.
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_notebook.py")], check=True)
    import occl

    out_dir = Path(args.out_dir).resolve()
    assert REPO not in out_dir.parents and out_dir != REPO, \
        f"--out-dir {out_dir} is inside the repo; an archive would be built into itself"
    sst, sam3 = Path(args.sst), Path(args.sam3)
    shas = {"occl": git_head(REPO), "sst": git_head(sst), "sam3": git_head(sam3)}

    layers = ("data", "code") if args.layer == "both" else (args.layer,)
    outs = []
    for layer in layers:
        members: dict[str, Path] = {}
        generated: dict[str, str] = {}
        missing: list[str] = []
        if layer in ("data", "all"):
            m, g, missing = plan_data(sst, Path(args.cache_dir),
                                      Path(args.ledger_dir) if args.ledger_dir else None)
            members |= m
            generated |= g
        if layer in ("code", "all"):
            m, g = plan_code(REPO, sam3)
            members |= m
            generated |= g
        out = out_dir / f"occl-scc-{layer}.tar.gz"
        total = sum(p.stat().st_size for p in members.values())
        print(f"\n[{layer}] {len(members)} files, {total / 1e9:.2f} GB uncompressed -> {out}")
        if missing:
            print(f"  LEDGER MISSING ({len(missing)}/6) — {args.ledger_dir} lacks: "
                  + ", ".join(missing))
            print("  the archive ships results/LEDGER-NEEDED.md; drop the files into "
                  "/projectnb/<project>/occl/results/ before the first session")
        manifest = build(out, layer, members, generated, missing, occl.BUILD, shas)
        if not args.no_verify:
            verify(out, manifest)
            print("  verified: every path the notebook and setup.sh assert is present")
        for k, v in manifest["components"].items():
            print(f"  {k:>20s}: {v['files']:5d} files, {v['bytes'] / 1e6:9.1f} MB")
        print(f"  wrote {out.name} ({manifest['archive_bytes'] / 1e6:.1f} MB in "
              f"{manifest['seconds']}s)  sha256 {sha256_of(out)[:16]}...")
        outs.append(out)

    # ONE ssh session each: the archive(s) travel on ssh's stdin and are unpacked on the
    # far side in the same session as the wipe and the setup — one password prompt,
    # instead of one for the copy and one for the shell. stdin is redirected away from
    # wipe.sh/setup.sh so neither can swallow archive bytes; `tar -i` reads the
    # concatenated first-time archives past the first end-of-archive marker.
    print("\nfirst time — paste as ONE line (replace <user>/<project>):")
    print(f"  cat {' '.join(str(o) for o in outs)} | ssh <user>@scc1.bu.edu "
          "'mkdir -p /projectnb/<project> && cd /projectnb/<project> && tar xzif - && "
          "PYMOD=python3/3.12.4 bash occl/setup.sh </dev/null'")
    print("each later iteration (code layer only) — ONE line:")
    print("  python scripts/build_scc_tarball.py --layer code && ssh <user>@scc1.bu.edu "
          "'cd /projectnb/<project> && (bash occl/wipe.sh </dev/null || true) && "
          "tar xzf - && PYMOD=python3/3.12.4 bash occl/setup.sh </dev/null' "
          "< ../occl-scc-code.tar.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
