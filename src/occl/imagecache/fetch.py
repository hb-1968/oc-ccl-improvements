"""Network fetching, with an optional Zenodo CR2 -> JPG substitution.

266 of the 957 required Butterfly Major images are `.CR2` raws at 26-28 MB each, against
6-9 MB for a JPEG sibling. Substituting the sibling cuts the pull from ~11.9 GB to
~6.8 GB.

**But it is not what the paper did.** SST's own loaders (`butterfly_common.read_image`,
`eval_all_ablations.load_image`) branch on the extension and decode `.CR2` with
`rawpy.postprocess()`. Demosaic, white balance and tone curve all differ from the
camera-baked JPEG, and while those differences are small after a downsample to 1024x1024,
they are not nothing. So the faithful default is to fetch the URL exactly as the manifest
gives it, and the sibling rewrite is opt-in.

Either way the choice is recorded per image in the manifest rather than assumed, and a
missing sibling degrades to the raw rather than dropping the image.
"""

from __future__ import annotations

import time

#: Zenodo moved from /record/<id>/files/<name> to /records/<id>/files/<name>.
#: The legacy form still redirects, but following it costs a round trip per image.
_LEGACY, _MODERN = "/record/", "/records/"

RAW_SUFFIXES = (".cr2", ".nef", ".arw", ".dng")


def zenodo_normalize(url: str) -> tuple[str, bool]:
    """Return (url_to_fetch, was_rewritten).

    Case matters: Zenodo filenames use an upper-case `.JPG` for these records, and the
    stem's case is preserved.
    """
    out = url.replace(_LEGACY, _MODERN)
    if out.lower().endswith(".cr2"):
        return out[:-4] + ".JPG", True
    return out, False


def is_raw(url: str) -> bool:
    return url.lower().endswith(RAW_SUFFIXES)


def fetch_bytes(url: str, retries: int = 4, backoff: float = 2.0, timeout: int = 120) -> bytes:
    """GET with exponential backoff. Raises the last error if every attempt fails.

    Zenodo throttles aggressively (~14 s/image has been observed), so keep concurrency
    modest — 4-8 workers, not 32 — or the backoff spends longer than the download.
    """
    import requests

    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001 — retry on anything transient
            last = e
            if attempt < retries - 1:
                time.sleep(backoff**attempt)
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts: {last}")


#: "faithful" — fetch exactly what the manifest names, so `.CR2` is decoded by rawpy
#: just as SST does. "jpeg" — prefer a JPEG sibling for raws, ~43% less to download.
SOURCE_PREFERENCES = ("faithful", "jpeg")
FAITHFUL_SOURCE = "faithful"


def fetch_with_fallback(
    original_url: str, preference: str = FAITHFUL_SOURCE
) -> tuple[bytes, str, bool, bool]:
    """Fetch one image.

    Returns ``(payload, source_url, rewritten, raw_fallback)``.

    ``rewritten``    — a JPEG sibling was used in place of a raw (only under "jpeg").
    ``raw_fallback`` — the sibling was requested but unavailable, so the raw was pulled
                       instead: slower, and informational, but not an error.
    """
    if preference not in SOURCE_PREFERENCES:
        raise ValueError(f"preference must be one of {SOURCE_PREFERENCES}, got {preference!r}")

    modern = original_url.replace(_LEGACY, _MODERN)
    if preference == FAITHFUL_SOURCE:
        return fetch_bytes(modern), modern, False, False

    url, rewritten = zenodo_normalize(original_url)
    if not rewritten:
        return fetch_bytes(url), url, False, False
    try:
        return fetch_bytes(url, retries=2), url, True, False
    except RuntimeError:
        return fetch_bytes(modern), modern, False, True
