"""Append-only JSONL manifest for the image cache.

JSONL rather than one JSON blob, deliberately: a Colab session that is preempted or
disconnected mid-write cannot corrupt entries already recorded. Appends at these sizes
are effectively atomic, and re-fetching an image appends a new line rather than
requiring a rewrite (last-write-wins on load).

Each record keeps *both* the original manifest URL and the URL actually fetched, plus a
`rewritten` flag, so the CR2 -> JPG substitution is auditable per image. If a reviewer
later asks "did you evaluate on raw or JPEG?", the manifest answers per image instead of
per project.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class CacheEntry:
    dataset: str
    image_id: str
    original_url: str
    source_url: str
    rewritten: bool
    raw_fallback: bool
    src_bytes: int
    src_sha256: str
    cache_path: str  # relative to cache root
    cache_sha256: str
    size: int | None  # stored square edge, or None when kept at native resolution
    fmt: str
    decoder: str  # "rawpy" | "pil" — which path actually produced the pixels
    # Preprocessing provenance. These change the pixels the model sees, so they are
    # recorded per image and feed the registry's comparability key.
    source_preference: str = "faithful"
    interp: str = "linear"
    original_path: str | None = None  # archived source bytes, when retained
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def key(self) -> tuple[str, str]:
        return (self.dataset, self.image_id)


class CacheManifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[tuple[str, str], CacheEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = CacheEntry(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue  # tolerate a torn final line from a killed session
            self._entries[e.key()] = e  # last write wins

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._entries

    def get(self, dataset: str, image_id: str) -> CacheEntry | None:
        return self._entries.get((dataset, image_id))

    def append(self, entry: CacheEntry) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
        self._entries[entry.key()] = entry

    def entries(self, dataset: str | None = None) -> list[CacheEntry]:
        vals = list(self._entries.values())
        return [e for e in vals if e.dataset == dataset] if dataset else vals

    def verify(self, cache_root: Path, level: str = "fast", sample: float = 1.0,
               seed: int = 0) -> dict:
        """Check cached files against the manifest.

        ``fast`` — file exists and is non-empty.
        ``hash`` — re-hash the cached file. Catches Drive's truncated-write failure mode,
        which decodes to a plausible image with a corrupted lower band.
        """
        import numpy as np

        entries = self.entries()
        if sample < 1.0:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(entries))[: max(1, int(len(entries) * sample))]
            entries = [entries[int(i)] for i in idx]

        failures = []
        for e in entries:
            p = Path(cache_root) / e.cache_path
            if not p.exists() or p.stat().st_size == 0:
                failures.append((e.image_id, "missing or empty"))
                continue
            if level == "hash" and sha256_file(p) != e.cache_sha256:
                failures.append((e.image_id, "sha256 mismatch"))
        return {"checked": len(entries), "failures": failures, "n_failures": len(failures)}
