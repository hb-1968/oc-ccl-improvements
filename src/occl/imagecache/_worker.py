"""Decode/resize/encode worker — runs in a separate *process*.

Kept in its own module with module-level functions so it pickles cleanly for
`ProcessPoolExecutor`. Splitting this out of the network stage is what lets a local run
use all cores: `rawpy.postprocess()` on a 5472x3648 raw takes seconds, and 266 of the
957 butterfly sources are raws, so serialising them behind the GIL would make CPU the
bottleneck exactly when the network is not.
"""

from __future__ import annotations

from .decode import decode_rgb, encode, resize_square


def probe() -> bool:
    """Trivial task used to confirm a process pool can actually start workers.

    Python 3.14 defaults to the `forkserver` start method, whose workers re-import
    `__main__`. That succeeds from a script with an `if __name__ == "__main__"` guard but
    fails outright when `__main__` is `<stdin>` or a Jupyter kernel. Probing costs one
    round trip and turns a late, cryptic `ConnectionResetError` into an early, handled
    fallback to threads.
    """
    return True


def decode_and_encode(
    payload: bytes,
    source_url: str,
    size: int | None,
    fmt: str,
    quality: int,
    interp: str,
) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """Returns ``(encoded_bytes, decoder_name, native_hw, stored_hw)``.

    Returning the encoded blob rather than the array keeps the parent->child pickle
    small: a 1024x1024 JPEG q95 is ~300 KB against ~3 MB for the raw ndarray, and for a
    CR2 the decoded array is ~60 MB.
    """
    img, decoder = decode_rgb(payload, source_url)
    native_hw = (int(img.shape[0]), int(img.shape[1]))
    if size is not None:
        img = resize_square(img, size, interp)
    stored_hw = (int(img.shape[0]), int(img.shape[1]))
    return encode(img, fmt, quality), decoder, native_hw, stored_hw
