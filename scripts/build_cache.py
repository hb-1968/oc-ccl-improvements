#!/usr/bin/env python3
"""Stage S2 — build the image cache and emit its gate bundle.

Resumable and idempotent: re-run after a disconnect and it picks up where it stopped.
Beetle finishes in seconds (180 small PNGs from HuggingFace, no auth); Butterfly Major is
a multi-hour one-time Zenodo pull that only happens once because the cache lives on Drive.

Defaults reproduce SST's own preprocessing: raws are fetched as raws and decoded with
`rawpy.postprocess()`, and the downsample uses `cv2.resize`'s default INTER_LINEAR.
~11.9 GB downloaded, ~290 MB of derived cache.

    # faithful (default), archiving sources so a re-derive never re-downloads
    python scripts/build_cache.py --cache-root /content/drive/MyDrive/occl-cache \
        --originals-root /content/drive/MyDrive/occl-originals

    # beetle only — seconds, unblocks the S3 smoke while butterfly downloads
    python scripts/build_cache.py --cache-root ... --datasets beetle

    # bandwidth-constrained fallback: JPEG siblings, ~6.8 GB, NOT paper-faithful
    python scripts/build_cache.py --cache-root ... --source-preference jpeg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from occl import paths  # noqa: E402
from occl.bundle import download, export_bundle  # noqa: E402
from occl.datasets.beetle import Beetle  # noqa: E402
from occl.datasets.butterfly_major import ButterflyMajor  # noqa: E402
from occl.imagecache import (  # noqa: E402
    FAITHFUL_INTERP,
    LOSSLESS_FORMATS,
    POLICIES,
    CachePolicy,
    FAITHFUL_SOURCE,
    INTERPOLATIONS,
    SOURCE_PREFERENCES,
    ImageCache,
)
from occl.curation import SELECTIONS, draw_supports, support_pool  # noqa: E402

MAX_SHOT = 5  # 1-shot supports are a strict prefix of the 5-shot sets
N_RUNS = 20


def required_butterfly(ds: ButterflyMajor, groups: list[str] | None = None,
                       n_runs: int = N_RUNS, selections=SELECTIONS) -> list[tuple[str, str]]:
    """Every test image, plus the supports every support-selection arm will draw.

    Because sampling is prefix-nested, caching the 5-shot supports also covers 1-shot,
    so this set does not grow when we move from S5 to S6.

    **The union over `selections` is load-bearing.** Curation filters the support *pool*,
    and `sample_supports` permutes whatever pool it is handed — so shrinking the pool by
    even one image reshuffles every draw and selects largely different supports. A cache
    built for `uniform` alone is missing 87 of the images the placement-curated 1-shot arm
    asks for (385 at 5-shot), and the run dies partway through with a cache miss. Ask for
    every arm up front; the overlap is high and the marginal images are cheap next to a
    failed A100 session.

    ``groups``/``n_runs`` narrow the set to one experiment's slice — used by S5-DIAG's
    INTER_AREA arm, where re-deriving all 957 images is cheap locally but ~290 MB to push
    to Drive. `sample_supports` sorts its input internally, so the ids selected here are
    identical to the ones `protocol.build_run` will ask the cache for.
    """
    need: dict[str, str] = {}
    for g in (groups or list(ds.groups())):
        for image_id in ds.test_ids(g):
            need[image_id] = ds.image_url(image_id)
        for sel in selections:
            pool, _ = support_pool(ds, g, sel)
            for run in range(n_runs):
                # Routed through curation.draw_supports, NOT branched on here: this
                # function and protocol.build_run must ask for the same images, and when
                # they each had their own branch they silently diverged on
                # `orientation_diverse`.
                for image_id in draw_supports(ds, g, pool, sel, MAX_SHOT, run):
                    need[image_id] = ds.image_url(image_id)
    return sorted(need.items())


def required_beetle(ds: Beetle) -> list[tuple[str, str]]:
    """All 180 split members — train doubles as the support pool, test as queries."""
    return sorted((i, ds.image_url(i)) for i in ds.train_ids() + ds.test_ids())


def shape_agreement(ds, cache: ImageCache, image_ids: list[str]) -> dict:
    """Confirm each cached image matches its mask's dimensions.

    Beetle crops are stored at native resolution, so a transposed or mismatched crop
    would otherwise be silently absorbed by the metric's logit resize and quietly
    depress IoU. Cheap to check, hard to debug later.
    """
    bad = []
    for i in image_ids:
        if not cache.has(ds.name, i):
            continue
        img = cache.load(ds.name, i)
        mask = ds.load_labelmap(i)
        if img.shape[:2] != mask.shape[:2]:
            bad.append({"image_id": i, "image": list(img.shape[:2]), "mask": list(mask.shape)})
    return {"checked": len(image_ids), "mismatches": bad, "n_mismatches": len(bad)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["beetle", "butterfly_major"])
    ap.add_argument("--net-workers", type=int, default=6,
                    help="concurrent downloads; keep <=8, Zenodo throttles and backoff "
                         "dominates above that")
    ap.add_argument("--cpu-workers", type=int, default=None,
                    help="decode/resize processes (default: cores-2). This is the stage "
                         "that uses the machine — rawpy is seconds of CPU per raw")
    ap.add_argument("--chunk", type=int, default=32,
                    help="items in flight; bounds peak memory (~27MB per raw payload)")
    ap.add_argument("--local", action="store_true",
                    help="write the bundle straight to results/inbox/ so the gate can be "
                         "run immediately, and skip the Colab download")
    ap.add_argument("--source-preference", choices=SOURCE_PREFERENCES, default=FAITHFUL_SOURCE,
                    help="faithful: fetch raws and decode with rawpy, as SST does. "
                         "jpeg: substitute Zenodo JPEG siblings (~43%% smaller, NOT faithful)")
    ap.add_argument("--interp", choices=sorted(INTERPOLATIONS), default=FAITHFUL_INTERP,
                    help="downsample filter; linear matches SST's cv2.resize default")
    ap.add_argument("--originals-root", default=None,
                    help="archive untouched source bytes here (~11.9GB) so a later "
                         "re-derive at another resolution never re-downloads")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="butterfly only: restrict to these subspecies. With --n-runs, "
                         "derives just one experiment's slice (see S5-DIAG's INTER_AREA arm)")
    ap.add_argument("--n-runs", type=int, default=N_RUNS,
                    help="butterfly only: how many runs' supports to cache")
    ap.add_argument("--selections", nargs="+", default=list(SELECTIONS),
                    choices=list(SELECTIONS),
                    help="butterfly only: cache the supports every listed arm draws. "
                         "Defaults to ALL — a pool change reshuffles every draw, so a "
                         "cache built for one arm is missing images another needs.")
    ap.add_argument("--cache-format", default=None, choices=["png", "webp", "jpg"],
                    help="override the stored format for ALL datasets. Default is each "
                         "dataset's policy (lossless). 'jpg' is a deviation: SST has no "
                         "intermediate encode, so a lossy cache alters the pixels the "
                         "model sees and lands the run in the `modified` track.")
    ap.add_argument("--verify-sample", type=float, default=0.10)
    ap.add_argument("--out", default="s2.zip")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    if args.cache_format:
        for _name, _pol in list(POLICIES.items()):
            POLICIES[_name] = CachePolicy(size=_pol.size, fmt=args.cache_format,
                                          quality=101 if args.cache_format == "webp"
                                          else _pol.quality)
        if args.cache_format not in LOSSLESS_FORMATS:
            print(f"  WARNING: {args.cache_format} is LOSSY. SST decodes and resizes "
                  "straight into the model with no intermediate encode, so this is a "
                  "deviation and records will classify as `modified`.")

    cache = ImageCache(
        args.cache_root,
        source_preference=args.source_preference,
        interp=args.interp,
        originals_root=args.originals_root,
    )
    print(f"cache={args.cache_root} | source={args.source_preference} | interp={args.interp} | "
          f"originals={'archived -> ' + args.originals_root if args.originals_root else 'discarded'}")
    report: dict = {
        "cache_root": str(args.cache_root),
        "preprocessing": {
            "source_preference": args.source_preference,
            "interp": args.interp,
            "keep_originals": bool(args.originals_root),
            "faithful": args.source_preference == FAITHFUL_SOURCE and args.interp == FAITHFUL_INTERP,
        },
        "datasets": {},
    }

    for name in args.datasets:
        if name == "beetle":
            ds = Beetle(cache=cache)
            items = required_beetle(ds)
        elif name == "butterfly_major":
            ds = ButterflyMajor(cache=cache)
            items = required_butterfly(ds, args.groups, args.n_runs, args.selections)
        else:
            raise SystemExit(f"unknown dataset {name!r}")

        print(f"\n[{name}] {len(items)} images required")
        t0 = time.time()
        res = cache.build(name, items, net_workers=args.net_workers,
                          cpu_workers=args.cpu_workers, chunk=args.chunk)
        res["elapsed_s"] = round(time.time() - t0, 1)
        rate = res["fetched"] / res["elapsed_s"] if res["elapsed_s"] else 0
        print(f"[{name}] fetched {res['fetched']}, skipped {res['skipped']}, "
              f"failed {len(res['failed'])}, raw fallbacks {res['raw_fallbacks']}, "
              f"from archive {res.get('from_archive', 0)} "
              f"— {res['elapsed_s']}s ({rate:.2f} img/s)")
        for image_id, err in res["failed"][:10]:
            print(f"    FAIL {image_id}: {err}")

        ids = [i for i, _ in items]
        res["n_missing"] = sum(1 for i in ids if not cache.has(name, i))
        res["shapes"] = shape_agreement(ds, cache, ids)
        res["failed"] = res["failed"][:50]  # keep the bundle small
        report["datasets"][name] = res

    ver = cache.manifest.verify(cache.root, level="hash", sample=args.verify_sample)
    print(f"\nhash-verify {args.verify_sample:.0%} sample: "
          f"{ver['checked']} checked, {ver['n_failures']} failures")

    d = report["datasets"]
    entries = cache.manifest.entries()
    report["cache"] = {
        "n_required": sum(v["requested"] for v in d.values()),
        "n_missing": sum(v["n_missing"] for v in d.values()),
        "n_shape_mismatches": sum(v["shapes"]["n_mismatches"] for v in d.values()),
        "n_raw_fallback": sum(v["raw_fallbacks"] for v in d.values()),
        # How many images actually went through rawpy — under the faithful default this
        # should equal the CR2 count (266 for butterfly), and 0 under --source-preference jpeg.
        "n_rawpy_decoded": sum(1 for e in entries if e.decoder == "rawpy"),
        "n_jpeg_substituted": sum(1 for e in entries if e.rewritten),
        # Gated by S2: a lossy format inserts an encode SST does not have, so it changes
        # the pixels the model sees. Recorded per dataset since the policies differ.
        "formats": {n: POLICIES[n].fmt for n in args.datasets if n in POLICIES},
        "src_bytes_total": sum(e.src_bytes for e in entries),
        "hash_sample_failures": ver["n_failures"],
        "hash_sample_checked": ver["checked"],
    }
    print(f"\nsummary: {report['cache']}")

    # `--local` means "file it where the gate looks", but an explicit `--out` must still
    # win: a partial build (--datasets/--groups/--selections) is NOT a gate-able S2 bundle,
    # and silently dropping it into results/inbox/ invites `check_gate s2` to file it over
    # the real one. Twice now a butterfly-only top-up has landed there.
    explicit_out = args.out != ap.get_default("out")
    if args.local and not explicit_out:
        out = paths.inbox_root()
        out.mkdir(parents=True, exist_ok=True)
        out = out / "s2.zip"
        partial = (set(args.datasets) != {"beetle", "butterfly_major"}
                   or args.groups or args.n_runs != N_RUNS)
        if partial:
            print("\n  NOTE: this is a PARTIAL build "
                  f"(datasets={args.datasets}, groups={args.groups}, n_runs={args.n_runs}).\n"
                  "  It is not a complete S2 bundle — pass --out to keep it out of "
                  "results/inbox/.")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
    out = export_bundle("s2", report, out,
                        extra_files={"manifest.jsonl": cache.manifest.path})
    print(f"\nwrote {out}")
    if args.local:
        print("next:  python scripts/check_gate.py s2")
    elif not args.no_download:
        download(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
