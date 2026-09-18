#!/usr/bin/env python3
"""Compute-unit cost model for the eval stages, A100 vs L4.

The framing that matters: Colab bills per *hour*, so a pricier GPU is cheaper only if it
is faster by more than its price ratio. At 5.4 vs 1.7 units/hr the A100 costs **3.18x**
more per hour, so it only wins on cost if it is more than 3.18x faster on *this* workload.

That threshold is hard to clear here, and the embedding cache makes it harder rather than
easier. The cache removes ~97% of image-encoder passes -- the compute-dense, tensor-core-
friendly part where the A100's 312 vs 121 bf16 TFLOPS actually tells. What remains per
episode is memory attention plus mask decoding over small tensors: latency-bound work
where neither GPU saturates and the gap narrows well below 3.18x.

So the cache does two things at once: it makes everything much cheaper, and it shifts the
cost-optimal GPU *down*.

Every throughput figure below is an ESTIMATE until S3 reports real seconds/episode.
Re-run with --enc-ms/--ep-ms from the bundle to replace the guesses with measurements.
"""

from __future__ import annotations

import argparse

# units/hour, Colab Pro (July 2026)
GPUS = {
    "A100": {"units_hr": 5.40, "bf16_tflops": 312.0, "bw_gbs": 1555.0},
    "L4": {"units_hr": 1.71, "bf16_tflops": 121.0, "bw_gbs": 300.0},
}

# (episodes, encoder passes uncached, encoder passes cached) — computed from the real
# manifests, not estimated. See the `distinct_images` derivation in occl.evaluator.
STAGES = {
    "s4 beetle 1-shot": (1200, 2400, 79),
    "s5 major 1-shot": (10000, 20000, 598),
    "s6 beetle 5-shot": (1200, 7200, 135),
    "s6 major 5-shot": (10000, 60000, 957),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-ms", type=float, default=110.0,
                    help="A100 ms per image-encoder pass (hiera-L @1024, bf16). ESTIMATE.")
    ap.add_argument("--ep-ms", type=float, default=45.0,
                    help="A100 ms per episode excluding the encoder: memory attention + "
                         "mask decoder + host transfer. ESTIMATE.")
    ap.add_argument("--enc-ratio", type=float, default=2.6,
                    help="L4/A100 time ratio for the encoder. Compute-bound, so it tracks "
                         "the 312/121 TFLOPS ratio.")
    ap.add_argument("--ep-ratio", type=float, default=1.8,
                    help="L4/A100 time ratio for the per-episode remainder. Small, "
                         "latency-bound kernels: neither GPU saturates, so the gap is "
                         "much narrower than the spec sheets suggest.")
    args = ap.parse_args()

    price_ratio = GPUS["A100"]["units_hr"] / GPUS["L4"]["units_hr"]
    print(f"A100 costs {price_ratio:.2f}x more per hour "
          f"({GPUS['A100']['units_hr']} vs {GPUS['L4']['units_hr']} units/hr)")
    print(f"=> the A100 is cheaper ONLY if it is more than {price_ratio:.2f}x faster.\n")

    hdr = (f"{'stage':20s} {'cache':6s} {'A100 h':>7s} {'L4 h':>7s} "
           f"{'speedup':>8s} {'A100 u':>7s} {'L4 u':>7s} {'cheaper':>8s}")
    print(hdr)
    print("-" * len(hdr))

    totals = {"cached": [0.0, 0.0, 0.0, 0.0], "uncached": [0.0, 0.0, 0.0, 0.0]}
    for name, (eps, enc_un, enc_ca) in STAGES.items():
        for mode, enc in (("uncached", enc_un), ("cached", enc_ca)):
            a_h = (enc * args.enc_ms + eps * args.ep_ms) / 3.6e6
            l_h = (enc * args.enc_ms * args.enc_ratio
                   + eps * args.ep_ms * args.ep_ratio) / 3.6e6
            speedup = l_h / a_h
            a_u, l_u = a_h * GPUS["A100"]["units_hr"], l_h * GPUS["L4"]["units_hr"]
            totals[mode][0] += a_u
            totals[mode][1] += l_u
            totals[mode][2] += a_h
            totals[mode][3] += l_h
            print(f"{name:20s} {mode:6s} {a_h:7.2f} {l_h:7.2f} {speedup:7.2f}x "
                  f"{a_u:7.2f} {l_u:7.2f} {('A100' if a_u < l_u else 'L4'):>8s}")

    print("-" * len(hdr))
    for mode in ("uncached", "cached"):
        a, l, ah, lh = totals[mode]
        print(f"{'TOTAL':20s} {mode:6s} {ah:7.2f} {lh:7.2f} {lh/ah:7.2f}x "
              f"{a:7.2f} {l:7.2f} {('A100' if a < l else 'L4'):>8s}")

    a_c, l_c, ah_c, lh_c = totals["cached"]
    a_u, l_u, ah_u, lh_u = totals["uncached"]
    print(f"\ncached wall clock: A100 {ah_c*60:.0f} min | L4 {lh_c*60:.0f} min")
    print(f"\ncache saves {a_u - a_c:.1f} units on A100, {l_u - l_c:.1f} on L4 "
          f"({a_u / a_c:.0f}x / {l_u / l_c:.0f}x less compute)")
    print(f"cheapest path: L4 with cache at {l_c:.2f} units "
          f"(vs {a_u:.1f} for the naive A100 run — {a_u / l_c:.0f}x)")
    print("\nCaveats:")
    print("  * S8/S9 fine-tuning is NOT modelled here. It backprops through")
    print("    memory_attention at head_dim=256, where FA2's backward has historically")
    print("    required sm_80 — that is a capability question, not a speed one, and it")
    print("    may force A100 regardless of cost.")
    print("  * enc-ratio/ep-ratio are estimates. Re-run with measured values from the")
    print("    S3 bundle; the conclusion flips only if the A100 clears "
          f"{price_ratio:.2f}x overall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
