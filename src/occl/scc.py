"""Multi-GPU fan-out for the BU Shared Computing Cluster (SCC) — OnDemand Jupyter.

Colab gives one GPU and ~12 vCPUs, so the Colab notebooks plan *one group at a
time* and split its 20 runs across workers on that single card. An SCC OnDemand
session can hold several GPUs (``-l gpus=N``) and a chosen core count
(``NSLOTS``), so the right unit of planning is no longer "how many workers fit
on the card" but "which of these independent jobs can run right now, and where".

This module is that planner. It is deliberately small and torch-free at import
(the package's ``dependencies`` stay torch-free; ``visible_gpus`` imports torch
lazily) so the packing logic can be unit-tested on a machine with no GPU by
injecting a fake ``spawn``.

Three facts about SCC drive the design:

* **GPU assignment is by environment.** SGE exposes the job's GPUs through
  ``CUDA_VISIBLE_DEVICES``; a worker is pinned to one card by re-exporting a
  single id from that list. Ids are passed through as the *strings* the parent
  saw — they may be indices or UUIDs, and only the parent's list is authoritative.
* **Cores are a hard ceiling.** ``NSLOTS`` is what the scheduler granted, not
  ``os.cpu_count()`` (which reports the whole node). Each worker pins BLAS to one
  thread and needs about one core to feed its GPU; one core is reserved for the
  kernel itself.
* **Everything is banked, so failures are cheap.** Every worker resumes from the
  shared per-run directory, so an OOM'd slice only costs its own recompute. That
  is why VRAM costs here are plain numbers a caller can overstate safely rather
  than a calibration table (see the ``PEAK_GB`` saga in ``notebooks/run_stage.py``
  for what happens when the table is trusted).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol


# ── the environment ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GPU:
    """One card the job may use. ``id`` is the string to export as
    ``CUDA_VISIBLE_DEVICES`` for a worker pinned to it."""

    id: str
    capacity_gb: float
    name: str = ""


def visible_gpu_ids(env: dict | None = None) -> list[str] | None:
    """Parse ``CUDA_VISIBLE_DEVICES`` into the id strings a child must re-export.

    ``None`` when the variable is unset (the caller falls back to torch's count);
    an empty list when it is set but empty (no GPUs granted — a real SGE state).
    """
    env = os.environ if env is None else env
    raw = env.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def visible_gpus() -> list[GPU]:
    """Every GPU this process can see, with its id string and capacity.

    ``torch.cuda.device_count()`` already honours ``CUDA_VISIBLE_DEVICES``, so
    the ordinal ``i`` torch reports maps to the ``i``-th id in that list. When
    the variable is unset the ids are plain ordinals.
    """
    import torch

    n = torch.cuda.device_count()
    ids = visible_gpu_ids()
    if ids is None:
        ids = [str(i) for i in range(n)]
    assert len(ids) == n, (
        f"CUDA_VISIBLE_DEVICES lists {len(ids)} ids but torch sees {n} devices — "
        "the kernel's view of the job's GPUs is inconsistent; restart the session")
    out = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        out.append(GPU(id=ids[i], capacity_gb=p.total_memory / 1e9, name=p.name))
    return out


def granted_cores(env: dict | None = None) -> int:
    """Cores the scheduler granted (``NSLOTS``), else the node's count."""
    env = os.environ if env is None else env
    try:
        n = int(env.get("NSLOTS", "") or 0)
    except ValueError:
        n = 0
    return n if n > 0 else (os.cpu_count() or 1)


def local_scratch(env: dict | None = None) -> str:
    """Node-local scratch for this job — ``$TMPDIR`` on SCC (wiped at job end)."""
    env = os.environ if env is None else env
    return env.get("TMPDIR") or env.get("TMP") or "/tmp"


#: nvidia-smi compute modes under which a card accepts ONE process at a time.
#: SCC's GPU nodes run Exclusive_Process: a second CUDA context on the same card
#: fails at creation with cudaErrorDevicesUnavailable ("CUDA-capable device(s)
#: is/are busy or unavailable"). The first SCC session measured this the hard
#: way — every second tenant died in 0.1 min, every first tenant ran fine.
EXCLUSIVE_MODES = ("Exclusive_Process", "Exclusive_Thread", "Prohibited")


def parse_compute_modes(text: str) -> list[str]:
    """One mode per line of ``nvidia-smi --query-gpu=compute_mode --format=csv,noheader``."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def compute_modes() -> list[str]:
    """The compute mode of each visible GPU, or [] if nvidia-smi is unavailable."""
    import subprocess

    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=compute_mode", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001 — no nvidia-smi: caller decides
        return []
    return parse_compute_modes(p.stdout) if p.returncode == 0 else []


def gpus_are_exclusive(modes: list[str] | None = None) -> bool:
    """True if ANY card refuses a second process. Unknown (no nvidia-smi) -> False,
    so the notebook prints what it decided and why."""
    modes = compute_modes() if modes is None else modes
    return any(m in EXCLUSIVE_MODES for m in modes)


# ── the jobs ─────────────────────────────────────────────────────────────────

@dataclass
class Job:
    """One subprocess the scheduler may launch.

    ``gb`` is the VRAM it needs; ``None`` means *exclusive* — the whole card,
    nothing else co-resident (fine-tunes, whose residency has never been
    measured on this family). ``deps`` are tags that must have exited 0 first.
    """

    tag: str
    cmd: list[str]
    gb: float | None = None
    deps: tuple[str, ...] = ()
    log: str | None = None
    phase: str = ""


class Handle(Protocol):
    """What ``spawn`` must return — the subset of ``subprocess.Popen`` used."""

    def poll(self) -> int | None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


@dataclass
class Record:
    tag: str
    gpu: str
    started: float
    ended: float | None = None
    rc: int | None = None
    phase: str = ""

    @property
    def seconds(self) -> float:
        return (self.ended or time.time()) - self.started


SKIPPED = -999  #: rc recorded for a job never launched because a dependency failed


@dataclass
class Scheduler:
    """Dependency- and VRAM-aware packing of jobs onto the job's GPUs.

    Policy, in order:

    1. a job is *ready* when every dependency has exited 0; if any dependency
       failed (or was itself skipped) the job is skipped, so a failure never
       silently produces a downstream merge of half the runs;
    2. ready jobs are considered exclusive-first, then by descending ``gb`` —
       fine-tunes are the critical path and the heaviest eval workers are the
       longest, so they go first;
    3. a job lands on the GPU with the most free VRAM that fits it (worst-fit
       spreads load across cards); an exclusive job needs an *idle* card and
       leaves no room for anyone else;
    4. never more than ``max_procs`` live subprocesses (the core ceiling), and
       one launch per ``stagger_s`` so N workers do not load the checkpoint in
       the same second;
    5. with ``exclusive_gpus`` (the SCC reality: Exclusive_Process compute mode)
       every job is placed as if exclusive — one process per card, VRAM prices
       ignored for packing (still checked against capacity);
    6. a job that exits non-zero within ``quick_fail_s`` is re-queued up to
       ``retry_quick_failures`` times before its dependents are skipped — the
       signature of a card refusing a context (a previous tenant still tearing
       down), never of a real defect, which takes minutes to reach.

    ``spawn(job, gpu_id)`` must return a Popen-like handle; ``sleep`` / ``clock``
    are injectable so tests run in zero time.
    """

    gpus: list[GPU]
    max_procs: int
    headroom: float = 0.92
    stagger_s: float = 20.0
    heartbeat_s: float = 180.0
    poll_s: float = 5.0
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time
    log: Callable[[str], None] = print
    progress: Callable[[], str] | None = None
    #: Path to a JSON journal rewritten (atomically) on every launch and exit.
    #: A cluster wall-time kill is a SIGKILL: the except path below never runs and
    #: nothing in memory survives — so the per-job timings the next session is
    #: tuned from must already be on disk. Optional; None disables it.
    journal: str | None = None
    #: one process per card (see policy 5); the notebook sets it from nvidia-smi
    exclusive_gpus: bool = False
    retry_quick_failures: int = 1
    quick_fail_s: float = 90.0
    records: dict[str, Record] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)

    # ── capacity ─────────────────────────────────────────────────────────
    def _budget(self, gpu: GPU) -> float:
        return gpu.capacity_gb * self.headroom

    def _free(self, gpu: GPU, running: dict[str, tuple[Job, GPU, Handle]]) -> float:
        used = 0.0
        for job, g, _ in running.values():
            if g.id != gpu.id:
                continue
            if job.gb is None:
                return 0.0                       # exclusive tenant: nothing fits
            used += job.gb
        return self._budget(gpu) - used

    def _place(self, job: Job, running) -> GPU | None:
        if job.gb is None or self.exclusive_gpus:
            idle = [g for g in self.gpus
                    if not any(r[1].id == g.id for r in running.values())]
            return idle[0] if idle else None
        fits = [(self._free(g, running), g) for g in self.gpus
                if self._free(g, running) >= job.gb]
        if not fits:
            return None
        fits.sort(key=lambda t: -t[0])
        return fits[0][1]

    def fits_anywhere(self, job: Job) -> bool:
        """Can this job run on an EMPTY card at all? Caught before launch, so a
        mis-sized job fails in the plan, not after the fine-tunes."""
        if job.gb is None:
            return bool(self.gpus)
        return any(self._budget(g) >= job.gb for g in self.gpus)

    # ── the journal ──────────────────────────────────────────────────────
    def _write_journal(self, pending: int, running: int) -> None:
        if not self.journal:
            return
        import json

        doc = {
            "written": self.clock(), "pending": pending, "running": running,
            "gpus": [{"id": g.id, "capacity_gb": g.capacity_gb, "name": g.name}
                     for g in self.gpus],
            "exclusive_gpus": self.exclusive_gpus,
            "jobs": {t: {"gpu": r.gpu, "phase": r.phase, "started": r.started,
                         "ended": r.ended, "rc": r.rc,
                         "minutes": round(r.seconds / 60, 2),
                         "attempts": self.attempts.get(t, 1)}
                     for t, r in self.records.items()},
        }
        tmp = self.journal + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(doc, fh, indent=1)
            os.replace(tmp, self.journal)           # never a half-written journal
        except OSError as e:  # instrumentation must never fail a stage
            self.log(f"  (journal write failed: {e})")

    # ── the loop ─────────────────────────────────────────────────────────
    def run(self, jobs: Iterable[Job], spawn: Callable[[Job, str], Handle],
            label: str = "job") -> dict[str, int]:
        """Run every job; return ``{tag: rc}`` (``SKIPPED`` for never-launched).

        Kills every survivor on ``KeyboardInterrupt`` (or any BaseException) —
        a Popen child does not die with its parent's cell, and orphans keep
        their VRAM, which turns the retry into a fresh-looking OOM.
        """
        jobs = list(jobs)
        tags = [j.tag for j in jobs]
        assert len(set(tags)) == len(tags), f"duplicate job tags: {tags}"
        known = set(tags)
        for j in jobs:
            unknown = [d for d in j.deps if d not in known]
            assert not unknown, f"{j.tag} depends on unknown job(s) {unknown}"
            assert self.fits_anywhere(j), (
                f"{j.tag} needs {j.gb} GB but the largest card is "
                f"{max((g.capacity_gb for g in self.gpus), default=0):.1f} GB "
                f"(x{self.headroom} headroom) — lower its cost or request bigger GPUs")
        assert self.gpus, "no GPUs visible — request them in the OnDemand form"
        assert self.max_procs >= 1

        pending: list[Job] = list(jobs)
        running: dict[str, tuple[Job, GPU, Handle]] = {}
        done: dict[str, int] = {}
        t0 = self.clock()
        last_beat = t0
        last_launch = -1e18
        self.log(f"scheduler: {len(jobs)} {label}(s) over {len(self.gpus)} GPU(s) "
                 f"{[f'{g.id}:{g.capacity_gb:.0f}GB' for g in self.gpus]}, "
                 f"max {self.max_procs} live process(es)"
                 + (", ONE JOB PER CARD (exclusive compute mode)"
                    if self.exclusive_gpus else ""))
        try:
            while pending or running:
                # ── reap ──
                for tag in list(running):
                    job, gpu, h = running[tag]
                    rc = h.poll()
                    if rc is None:
                        continue
                    del running[tag]
                    rec = self.records[tag]
                    rec.ended, rec.rc = self.clock(), rc
                    if (rc != 0 and rec.seconds < self.quick_fail_s
                            and self.attempts.get(tag, 1) <= self.retry_quick_failures):
                        # policy 6: an instant failure is a card refusing a
                        # context, not a defect — try again on the next idle card
                        self.attempts[tag] = self.attempts.get(tag, 1) + 1
                        pending.insert(0, job)
                        self.log(f"  {label} {tag} exit {rc} on GPU {gpu.id} after "
                                 f"{rec.seconds:.0f}s — re-queued (attempt "
                                 f"{self.attempts[tag]})")
                        self._write_journal(len(pending), len(running))
                        continue
                    done[tag] = rc
                    self.log(f"  {label} {tag} exit {rc} on GPU {gpu.id} "
                             f"({rec.seconds / 60:.1f} min)")
                    if rc != 0 and job.log and os.path.exists(job.log):
                        with open(job.log, errors="replace") as fh:
                            self.log(fh.read()[-2500:])
                    self._write_journal(len(pending), len(running))

                # ── skip anything whose dependency failed ──
                for job in list(pending):
                    if any(d in done and done[d] != 0 for d in job.deps):
                        pending.remove(job)
                        done[job.tag] = SKIPPED
                        bad = [d for d in job.deps if done.get(d, 0) != 0]
                        self.log(f"  {label} {job.tag} SKIPPED — dependency "
                                 f"{bad} failed")

                # ── launch (at most one per stagger interval) ──
                now = self.clock()
                if len(running) < self.max_procs and now - last_launch >= self.stagger_s:
                    ready = [j for j in pending
                             if all(d in done and done[d] == 0 for d in j.deps)]
                    ready.sort(key=lambda j: (j.gb is not None, -(j.gb or 0.0)))
                    for job in ready:
                        gpu = self._place(job, running)
                        if gpu is None:
                            continue
                        h = spawn(job, gpu.id)
                        running[job.tag] = (job, gpu, h)
                        pending.remove(job)
                        self.attempts.setdefault(job.tag, 1)
                        self.records[job.tag] = Record(job.tag, gpu.id, now,
                                                       phase=job.phase)
                        last_launch = now
                        self.log(f"  -> {job.tag} on GPU {gpu.id} "
                                 f"[{'exclusive' if job.gb is None else f'{job.gb:g} GB'}] "
                                 f"live {len(running)}/{self.max_procs}, "
                                 f"queued {len(pending)}")
                        self._write_journal(len(pending), len(running))
                        break

                if not pending and not running:
                    break
                if self.heartbeat_s and now - last_beat >= self.heartbeat_s:
                    extra = f", {self.progress()}" if self.progress else ""
                    self.log(f"  ... {len(running)} running, {len(pending)} queued, "
                             f"{len(done)} done in {(now - t0) / 60:.0f} min{extra}")
                    last_beat = now
                self.sleep(self.poll_s)
        except BaseException:                      # KeyboardInterrupt included
            alive = list(running)
            self.log(f"\ninterrupted — killing {len(alive)} surviving {label}(s): {alive}")
            for tag, (_j, _g, h) in running.items():
                try:
                    h.kill()
                except Exception:  # noqa: BLE001
                    pass
            for tag, (_j, _g, h) in running.items():
                try:
                    h.wait(timeout=30)
                except Exception:  # noqa: BLE001
                    pass
                rec = self.records.get(tag)
                if rec is not None:
                    rec.ended, rec.rc = self.clock(), -9
            raise
        return done

    # ── reporting ────────────────────────────────────────────────────────
    def phase_summary(self) -> dict[str, dict[str, float]]:
        """Per phase: wall-clock span and summed GPU-minutes, from the records."""
        out: dict[str, dict[str, float]] = {}
        for r in self.records.values():
            ph = out.setdefault(r.phase or "(none)",
                                {"start": r.started, "end": r.ended or r.started,
                                 "gpu_min": 0.0, "jobs": 0})
            ph["start"] = min(ph["start"], r.started)
            ph["end"] = max(ph["end"], r.ended or r.started)
            ph["gpu_min"] += r.seconds / 60
            ph["jobs"] += 1
        for ph in out.values():
            ph["wall_min"] = (ph["end"] - ph["start"]) / 60
        return out


def failed(results: dict[str, int]) -> list[str]:
    """Tags that did not exit 0 (skipped ones included)."""
    return [t for t, rc in results.items() if rc != 0]
