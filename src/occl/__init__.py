"""Paper-faithful SST/SAM-2 reproduction harness and OC-CCL improvements."""

__version__ = "0.1.0"

#: Bumped whenever a change makes an OLD repo zip produce a wrong or incomplete artifact —
#: which INCLUDES a crash fix. Build 4 exists because build 3 was re-uploaded ambiguously:
#: the s8c_tune inference-tensor fix shipped without a bump, the stale zip passed the guard,
#: and the identical traceback reappeared looking like the fix had not worked.
#: The notebook compares it against `REPO_MIN_BUILD` and refuses to run if Drive is stale.
#:
#: A stale `occl-repo.zip` is the most expensive silent failure in this workflow: the run
#: succeeds, the bundle looks normal, the gate passes on it, and the defect surfaces a
#: session later. It has now happened twice — `--tta` never reaching the workers (s5c ran
#: as plain s5b), and s8b writing no `lora.pt`, where the second re-run reproduced the
#: same absence byte-for-byte because the repo on Drive had not changed.
#:
#: Bump this ONLY for changes that invalidate an old artifact, not for every edit —
#: a version that cries wolf gets ignored.
#:
#: But "invalidates an old artifact" is broader than it looks. The combiner fix that made
#: `finetune` per-arm changed no interface and broke nothing, so it was not bumped — and a
#: stale zip then re-ran the OLD combiner, produced a byte-identical record with the
#: provenance still missing, and passed the guard because both states were build 8. If a
#: change alters what a stage WRITES, an old copy of it is stale even though nothing errors.
#: Bump.
#:
#: 11: `occl.scc` (the multi-GPU scheduler) — the SCC notebook cannot run on a tree
#: without it, and asserts BUILD >= 11 so a stale sync fails at import, not mid-graph.
#: 12: S15 — the sam3 adapter attends to ALL conditioning frames (max_cond_frames_in_attn
#: -1, was the builder's 4) and commission_s14.py --shots proves it on the GPU; a
#: 5-shot run on build 11 would score 4 of 5 supports and look perfectly healthy.
#: 13: S17 — `oc_ccl_step(capture=)`, `finetune.probe`, `registry.paired_bootstrap` and
#: the two new workers (probe_train_pairs.py, combine_s17.py); an older tree has no
#: probe to run and the notebook asserts BUILD >= 13 before its first launch.
#: 14: S17 first SCC session — probe_train_pairs.py shadowed its eval-parity list with
#: the replay-parity dict, so build 13 wrote probe bundles with an EMPTY `eval_parity`
#: (the commissioning cell died on it); also the arms (found_mask / bce_w / sam2_arch),
#: the per-arm parity and the eval cells in combine_s17 — a build-13 tree runs a
#: different S17.
#: 15: S17b — `LoRALinear.enabled` / `lora_disabled`, the objectness capture in the
#: sam2 tracker, `oc_ccl_step(lora_legs=, obj_weight=)`, the probe's fwd_obj/cls_obj
#: columns and per-arm legs; a build-14 tree cannot train or probe the S17b arms.
#: 16: S17c — `lora_legs="forward_link"`, `--objness-weight`, objectives composed at read
#: time (`probe.own_loss`: formula + w x objterm), the false-positive area per pair, and
#: the shot-aware eval combine with S6's 5-shot pin; a build-15 tree cannot run S17c's
#: arms or read its bundle.
#: 17: S17d — the soft sentinel (`tracker.soft_sentinel_heads`, `--soft-sentinel`), the
#: per-arm gate in the probe with one base variant per gate (`base_of`), the gate-effect
#: block in the combine and the base-variant predicate in the gate; a build-16 tree
#: trains the soft arms under the HARD gate and probes them against the wrong base.
#: 18: S18 — `ste_threshold` (the binary-valued prompt with the reparam's gradient),
#: `--close-aug-deg` / `CyclePairs.closing` (the rotated closing end), the probe's hard
#: replay (`*_hard` columns, the steganography read) with per-arm prompt modes and
#: `base_close<D>` variants, the combine's `stego` block and the s18/s18ft gates; a
#: build-17 tree rejects `--prompt-mode ste_threshold` at argparse and its probe writes
#: no hard column, so the s18 gate fails on it — nothing silent.
BUILD = 18

#: The notebook build this package requires. The guard runs BOTH ways because the pair can
#: go stale in either direction and only one direction was covered:
#:
#:   old zip + new notebook -> caught by `occl.BUILD >= REPO_MIN_BUILD`
#:   new zip + OLD NOTEBOOK -> was NOT caught, and is the more likely mistake, because the
#:                             notebook is the file you have open and re-uploading the zip
#:                             feels like "updating the code".
#:
#: Stage dispatch lives in the notebook, so a stale one silently runs the previous
#: experiment against fresh library code — which is how S8d ran as two arms instead of the
#: {base, OC-CCL} x {no TTA, TTA} square.
NOTEBOOK_MIN_BUILD = 9
