"""Bytes -> RGB uint8, covering JPEG/PNG/TIFF and (as a fallback) camera raw."""

from __future__ import annotations

import io

import numpy as np

from .fetch import RAW_SUFFIXES


def decode_rgb(buf: bytes, url: str) -> tuple[np.ndarray, str]:
    """Decode image bytes to an (H, W, 3) uint8 RGB array. Returns (array, decoder_name).

    Dispatch is on the URL suffix, then verified against the payload: a Zenodo JPEG
    sibling that silently 404s to an HTML error page would otherwise be cached as a
    valid-looking file.
    """
    if url.lower().endswith(RAW_SUFFIXES):
        import rawpy

        with rawpy.imread(io.BytesIO(buf)) as raw:
            return np.ascontiguousarray(raw.postprocess()), "rawpy"

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(buf))
        img.load()
    except Exception as e:  # noqa: BLE001
        head = buf[:64].decode("utf-8", "replace")
        raise ValueError(f"not a decodable image ({e}); first bytes: {head!r}") from e

    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8), "pil"


#: SST resizes with `cv2.resize(img, (size, size))` and no `interpolation=` argument,
#: i.e. INTER_LINEAR. For a 5472->1024 downsample that aliases noticeably -- INTER_AREA
#: averages the whole source footprint and is the better choice on quality grounds.
#: But INTER_LINEAR is what produced the paper's numbers, so it is the faithful default
#: and INTER_AREA is available as a `modified`-track experiment.
INTERPOLATIONS = {"linear": "INTER_LINEAR", "area": "INTER_AREA"}
FAITHFUL_INTERP = "linear"


def resize_square(img: np.ndarray, size: int, interp: str = FAITHFUL_INTERP) -> np.ndarray:
    """Resize to (size, size), matching SST's `cv2.resize(img, (size, size))`.

    Aspect ratio is deliberately *not* preserved: the committed masks were produced by
    the same non-preserving resize and must stay pixel-aligned with their images.
    """
    import cv2

    if img.shape[:2] == (size, size):
        return img
    if interp not in INTERPOLATIONS:
        raise ValueError(f"interp must be one of {sorted(INTERPOLATIONS)}, got {interp!r}")
    return cv2.resize(img, (size, size), interpolation=getattr(cv2, INTERPOLATIONS[interp]))


def encode(img: np.ndarray, fmt: str, quality: int = 95) -> bytes:
    """Encode the derived image for storage.

    `png` and `webp` (at quality > 100) round-trip **bit-exactly**, which is what makes
    the cache a pure memoisation of SST's decode->resize path rather than an extra step
    that alters the pixels. `jpg` is retained only so the deviation stays measurable:
    at q95 it cost 2.017/255 mean error against the in-memory resize.
    """
    from PIL import Image

    bio = io.BytesIO()
    pil = Image.fromarray(img)
    if fmt == "jpg":
        pil.save(bio, format="JPEG", quality=quality, subsampling=0)
    elif fmt == "png":
        pil.save(bio, format="PNG", optimize=True)
    elif fmt == "webp":
        # lossless=True ignores `quality` for fidelity; it only tunes effort/size.
        pil.save(bio, format="WEBP", lossless=True, quality=100)
    else:
        raise ValueError(f"unsupported cache format {fmt!r}")
    return bio.getvalue()
