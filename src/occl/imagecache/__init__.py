"""Drive-backed image cache: download once, resize once, reuse forever.

The expensive resource is bandwidth, not storage: Zenodo throttles at roughly 14 s/image,
and every image is resized before the model sees it, so the derived cache is ~40x smaller
than the sources. Set ``originals_root`` to also archive the untouched source bytes --
that costs the full download volume on disk but makes any later re-derive free, which is
worth it precisely because re-downloading is the slow part.

Preprocessing provenance (``source_preference``, ``interp``) is recorded per image,
because both choices change the pixels the model sees and therefore the score. The
defaults reproduce SST's own loaders; see `fetch.py` and `decode.py`.

Per-dataset policy, because the two datasets are not alike:

* **butterfly_major** -- stored at 1024x1024 as **lossless PNG**. The committed masks are
  natively 1024x1024, so this keeps image and mask pixel-aligned and introduces no
  interpolation anywhere in the metric path.

  It was JPEG q95 until measurement showed that inserted a lossy step SST does not have:
  its `read_image` goes original -> decode -> `cv2.resize` -> model with no intermediate
  encode. The round-trip cost **2.017/255** mean error. That is very unlikely to have
  mattered -- it was flat across the image (boundary/interior 0.98-1.15x) and the model
  is measurably insensitive at that magnitude, since the INTER_AREA arm perturbed pixels
  *more* (2.79/255) and moved `iou_binary` by -0.12 -- but it was an unforced deviation,
  and lossless costs only ~1.8 GB against ~0.6 GB.

  With a lossless policy the cache is a pure memoisation of SST's own pixel path, and
  every link is verified rather than assumed: raw decode is `rawpy.postprocess()` in
  both; our PIL JPEG decode is **bit-identical** to SST's `cv2.imread` (measured
  0.0000/255, zero pixels differing); and `resize_square` is literally
  `cv2.resize(img, (size, size))` with the same INTER_LINEAR default.
* **beetle** -- stored at **native resolution** as lossless PNG. The crops are 33-472 px
  wide; upsampling them to 1024 would multiply the cache ~10x and add no information.
  The adapter resizes up for the model, and the metric resizes logits back down to the
  GT's own resolution.
"""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._worker import decode_and_encode, probe as _probe_worker
from .decode import FAITHFUL_INTERP, INTERPOLATIONS
from .fetch import FAITHFUL_SOURCE, SOURCE_PREFERENCES, fetch_with_fallback
from .manifest import CacheEntry, CacheManifest, sha256_bytes

__all__ = [
    "CachePolicy", "POLICIES", "ImageCache", "CacheManifest", "CacheEntry",
    "SOURCE_PREFERENCES", "FAITHFUL_SOURCE", "INTERPOLATIONS", "FAITHFUL_INTERP",
]


#: Formats that survive a round-trip bit-exactly. A cache in one of these is a pure
#: memoisation; anything else silently alters the pixels the model sees.
LOSSLESS_FORMATS = frozenset({"png", "webp"})


@dataclass(frozen=True)
class CachePolicy:
    size: int | None  # square edge, or None to keep native resolution
    fmt: str  # "png" | "webp" (lossless) | "jpg" (lossy — a deviation)
    quality: int = 95  # webp: >100 means lossless; ignored for png

    @property
    def lossless(self) -> bool:
        return self.fmt in LOSSLESS_FORMATS


POLICIES: dict[str, CachePolicy] = {
    "butterfly_major": CachePolicy(size=1024, fmt="png"),
    "beetle": CachePolicy(size=None, fmt="png"),
}


class ImageCache:
    """
    Parameters
    ----------
    source_preference
        ``"faithful"`` (default) fetches exactly the URL the manifest names, so ``.CR2``
        raws are decoded with ``rawpy.postprocess()`` exactly as SST does. ``"jpeg"``
        substitutes Zenodo's JPEG sibling for raws, cutting the pull ~43% at the cost of
        no longer matching the reference preprocessing.
    interp
        Downsample filter. ``"linear"`` matches SST's default `cv2.resize`; ``"area"`` is
        better but is a deviation.
    originals_root
        When set, the raw downloaded bytes are archived here. Costs the full ~11.9 GB but
        means a later re-derive (different resolution, different interpolation) never
        re-downloads -- which matters because Zenodo throttles at ~14 s/image.
    """

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path | None = None,
        source_preference: str = FAITHFUL_SOURCE,
        interp: str = FAITHFUL_INTERP,
        originals_root: str | Path | None = None,
    ):
        if source_preference not in SOURCE_PREFERENCES:
            raise ValueError(f"source_preference must be one of {SOURCE_PREFERENCES}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_preference = source_preference
        self.interp = interp
        self.originals_root = Path(originals_root) if originals_root else None
        if self.originals_root:
            self.originals_root.mkdir(parents=True, exist_ok=True)
        self.manifest = CacheManifest(
            Path(manifest_path) if manifest_path else self.root / "manifest.jsonl"
        )
        self._lock = threading.Lock()

    def policy(self, dataset: str) -> CachePolicy:
        if dataset not in POLICIES:
            raise KeyError(f"no cache policy for dataset {dataset!r}")
        return POLICIES[dataset]

    def rel_path(self, dataset: str, image_id: str) -> str:
        # image_id may contain "/" (beetle uses "<species>/<stem>"); subdirs are fine.
        return f"{dataset}/{image_id}.{self.policy(dataset).fmt}"

    def abs_path(self, dataset: str, image_id: str) -> Path:
        return self.root / self.rel_path(dataset, image_id)

    def has(self, dataset: str, image_id: str) -> bool:
        # The POLICY path — what `load()` will actually read — not the manifest's
        # recorded path. They diverge after a format migration: a JPEG-era cache has
        # manifest entries whose `cache_path` ends `.jpg` while the current policy says
        # `.png`, and the old semantics made `has()` say True for images `load()` then
        # raised on. That is how a `build_cache` top-up skipped all 957 butterfly images
        # as "already cached" against a cache the evaluator could not read.
        e = self.manifest.get(dataset, image_id)
        return e is not None and self.abs_path(dataset, image_id).exists()

    # ── writing ──────────────────────────────────────────────────────────────

    def ensure(self, dataset: str, image_id: str, url: str) -> CacheEntry:
        """Download-and-store if absent; otherwise return the existing entry.

        "Absent" means the POLICY-format file is absent (same rule as `has`): an entry
        from a superseded format does not satisfy the current policy, and re-deriving
        writes a fresh manifest line (last line wins on reload). The archived original
        makes that re-derive local and fast."""
        existing = self.manifest.get(dataset, image_id)
        if existing is not None and self.abs_path(dataset, image_id).exists():
            return existing

        pol = self.policy(dataset)
        payload, source_url, rewritten, raw_fallback = self._acquire(image_id, url, dataset)
        blob, decoder, _, _ = decode_and_encode(
            payload, source_url, pol.size, pol.fmt, pol.quality, self.interp
        )
        original_path = self._archive_original(dataset, image_id, source_url, payload)
        self._write_blob(dataset, image_id, blob)

        entry = CacheEntry(
            dataset=dataset,
            image_id=image_id,
            original_url=url,
            source_url=source_url,
            rewritten=rewritten,
            raw_fallback=raw_fallback,
            src_bytes=len(payload),
            src_sha256=sha256_bytes(payload),
            cache_path=self.rel_path(dataset, image_id),
            cache_sha256=sha256_bytes(blob),
            size=pol.size,
            fmt=pol.fmt,
            decoder=decoder,
            source_preference=self.source_preference,
            interp=self.interp,
            original_path=original_path,
        )
        with self._lock:
            self.manifest.append(entry)
        return entry

    @staticmethod
    def _decode_pool(workers: int) -> tuple[object, str]:
        """A process pool if one can start here, otherwise threads.

        Returns ``(executor, kind)``. Falling back is not a big loss: both
        `rawpy.postprocess()` and PIL's codecs release the GIL for their heavy work, so
        threads still overlap real CPU -- just less cleanly than separate processes.
        """
        # Cheap pre-check first: forkserver workers re-import __main__ by path, so if
        # __main__ has no real file behind it (stdin, `-c`, some REPLs) the child is
        # doomed before it starts and would spew a traceback to our stderr on the way out.
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if not (main_file and Path(main_file).is_file()):
            return ThreadPoolExecutor(max_workers=workers), "thread"

        pool = None
        try:
            pool = ProcessPoolExecutor(max_workers=workers)
            pool.submit(_probe_worker).result(timeout=60)
            return pool, "process"
        except Exception:  # noqa: BLE001 — e.g. a Jupyter kernel as __main__
            if pool is not None:
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:  # noqa: BLE001
                    pass
            return ThreadPoolExecutor(max_workers=workers), "thread"

    def _write_blob(self, dataset: str, image_id: str, blob: bytes) -> None:
        """Temp-then-replace. A plain write leaves a half-written file on a kill or a
        Drive FUSE hiccup, which decodes as a plausible image with a corrupted lower band
        -- the worst kind of failure, because nothing raises."""
        dst = self.abs_path(dataset, image_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, dst)

    def _archive_original(
        self, dataset: str, image_id: str, source_url: str, payload: bytes
    ) -> str | None:
        """Keep the untouched source bytes so a re-derive never re-downloads."""
        if self.originals_root is None:
            return None
        suffix = Path(source_url.split("?")[0]).suffix or ".bin"
        rel = f"{dataset}/{image_id}{suffix}"
        dst = self.originals_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, dst)
        return rel

    def _acquire(self, image_id: str, url: str, dataset: str) -> tuple[bytes, str, bool, bool]:
        """Get source bytes, preferring the local archive over the network.

        This is what makes an interrupted run cheap to resume and a re-derive free: once
        a source is archived, changing the resolution or interpolation never touches
        Zenodo again -- which matters because re-downloading is the slow part, not decoding.
        """
        if self.originals_root is not None:
            hit = sorted(self.originals_root.glob(f"{dataset}/{image_id}.*"))
            hit = [p for p in hit if not p.name.endswith(".tmp")]
            if hit:
                return hit[0].read_bytes(), hit[0].name, False, False
        return fetch_with_fallback(url, self.source_preference)

    def build(self, dataset: str, items: list[tuple[str, str]], net_workers: int = 6,
              cpu_workers: int | None = None, chunk: int = 32,
              progress: bool = True) -> dict:
        """Cache many ``(image_id, url)`` pairs. Resumable: cached ids are skipped.

        Two-stage pipeline, because the two costs are unlike each other:

        * **fetch** -- threads, ``net_workers``. Bounded by Zenodo's throttle (~14 s/image),
          not by us. Keep this modest; above ~8 concurrent, backoff dominates transfer
          and you risk being rate-limited outright.
        * **decode/resize/encode** -- *processes*, ``cpu_workers`` (default: cores - 2).
          `rawpy.postprocess()` is seconds of real CPU per raw, so this is the stage that
          actually uses the machine.

        Work is chunked so peak memory stays bounded: a chunk of 32 raws holds ~870 MB of
        payloads in flight, versus ~7 GB if the whole set were buffered. The manifest is
        appended per chunk, so a kill costs at most one chunk.
        """
        todo = [(i, u) for i, u in items if not self.has(dataset, i)]
        base = {"dataset": dataset, "requested": len(items),
                "skipped": len(items) - len(todo)}
        if not todo:
            return base | {"fetched": 0, "failed": [], "raw_fallbacks": 0}

        cpu_workers = cpu_workers or max(1, (os.cpu_count() or 4) - 2)
        pol = self.policy(dataset)
        done, failed, raw_fallbacks, from_archive = 0, [], 0, 0

        cpu_pool, pool_kind = self._decode_pool(cpu_workers)
        if progress:
            print(f"  {dataset}: {len(todo)} to fetch "
                  f"({net_workers} net threads, {cpu_workers} decode "
                  f"{pool_kind}{'es' if pool_kind == 'process' else 's'})", flush=True)

        with cpu_pool as cpu, ThreadPoolExecutor(max_workers=net_workers) as net:
            for start in range(0, len(todo), chunk):
                batch = todo[start : start + chunk]

                # ── stage 1: acquire bytes (threads / local archive) ──────────
                fetched: list[tuple[str, str, bytes, str, bool, bool]] = []
                futs = {net.submit(self._acquire, i, u, dataset): (i, u) for i, u in batch}
                for fut in as_completed(futs):
                    image_id, url = futs[fut]
                    try:
                        payload, src, rewritten, raw_fb = fut.result()
                        fetched.append((image_id, url, payload, src, rewritten, raw_fb))
                    except Exception as e:  # noqa: BLE001
                        failed.append((image_id, f"fetch: {type(e).__name__}: {e}"))

                # ── stage 2: decode/resize/encode (processes) ────────────────
                dfuts = {
                    cpu.submit(decode_and_encode, payload, src, pol.size, pol.fmt,
                               pol.quality, self.interp): (image_id, url, src, rw, rfb)
                    for image_id, url, payload, src, rw, rfb in fetched
                }
                for fut in as_completed(dfuts):
                    image_id, url, src, rewritten, raw_fb = dfuts[fut]
                    try:
                        blob, decoder, native_hw, stored_hw = fut.result()
                    except Exception as e:  # noqa: BLE001
                        failed.append((image_id, f"decode: {type(e).__name__}: {e}"))
                        continue

                    # ── stage 3: write + record (parent, serial, cheap) ──────
                    try:
                        self._write_blob(dataset, image_id, blob)
                        payload = next(p for i, _, p, *_ in fetched if i == image_id)
                        original_path = self._archive_original(dataset, image_id, src, payload)
                        self.manifest.append(CacheEntry(
                            dataset=dataset, image_id=image_id, original_url=url,
                            source_url=src, rewritten=rewritten, raw_fallback=raw_fb,
                            src_bytes=len(payload), src_sha256=sha256_bytes(payload),
                            cache_path=self.rel_path(dataset, image_id),
                            cache_sha256=sha256_bytes(blob), size=pol.size, fmt=pol.fmt,
                            decoder=decoder, source_preference=self.source_preference,
                            interp=self.interp, original_path=original_path,
                        ))
                        done += 1
                        raw_fallbacks += int(raw_fb)
                        from_archive += int(not src.startswith("http"))
                    except Exception as e:  # noqa: BLE001
                        failed.append((image_id, f"write: {type(e).__name__}: {e}"))

                del fetched
                if progress:
                    print(f"  {dataset}: {done + len(failed)}/{len(todo)} "
                          f"({len(failed)} failed, {from_archive} from archive)", flush=True)

        return base | {"fetched": done, "failed": failed, "raw_fallbacks": raw_fallbacks,
                       "from_archive": from_archive, "net_workers": net_workers,
                       "cpu_workers": cpu_workers, "decode_pool": pool_kind}

    # ── reading ──────────────────────────────────────────────────────────────

    def load(self, dataset: str, image_id: str) -> np.ndarray:
        """Read a cached image as (H, W, 3) uint8 RGB."""
        from PIL import Image

        p = self.abs_path(dataset, image_id)
        if not p.exists():
            raise FileNotFoundError(
                f"{dataset}/{image_id} is not cached ({p}). Run stage S2 first."
            )
        img = Image.open(p)
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)
