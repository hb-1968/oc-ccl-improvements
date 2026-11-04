# Zoom / tiling at inference — local CPU probe

Status: **pre-registered 2026-10-06, before the probe ran; probe LANDED 2026-10-06: pre-registered decision GO**
(see "Result" at the end; the pre-registration above it is unedited). This is the model-side lever proposed in
[QUERY_DIAGNOSIS.md §4/§6a](QUERY_DIAGNOSIS.md). Nothing here is a banked cell, and no stage, gate or locked module
changes.

## Question

Butterfly thin traits are under-resolved: IoU is 60.6 below 2 cells of SAM 2's 256² mask grid, about 72 at 2–5
cells, and about 88 at 12 or more. Specimen size predicts thin-trait IoU within cyrbia and malleti only, which is the
resolution signature.

**Does predicting on a crop around the specimen raise IoU, thin traits above all, without hurting wide ones?**

## What the zoom can add (stated before running)

- Butterfly images are cached at **1024²**, the locked pixel path (native photo → `cv2.resize` 1024² → model), on the
  same grid as the natively 1024² GT masks.
- A crop of the cached frame is therefore **upsampled**. The zoom adds decoder cells and finer encoder features per
  trait, but **no image detail**.
- Native-resolution crops would add real detail. They need the original photos and are a separate, later question.

## Wrapper (`src/occl/adapters/zoom.py`, unit-tested in `tests/test_zoom.py`)

All constants were fixed a priori.

**Boxes** are square in the 1024² frame. The frame's own anisotropic squash is kept, so a crop is a pure scale
change. Side = (1 + 2·`PAD`) × the longer side of the foreground's bounding box, with `PAD` = 0.10. The box is
centred on the foreground, shifted inside the frame, and capped at the frame.
- **Support box:** from its GT foreground. Support labels are the prompt, so this is legal.
- **Query box:** label-free. It comes from a first full-frame prediction's foreground (argmax, top logit > 0), with
  coordinates trimmed to the 0.5 %/99.5 % quantiles (`TRIM`).
  - If the first pass predicts nothing, the full-frame logits are returned.

**Modes:**
- **specimen:** one sub-episode on the crops.
- **quadrants:** a 2×2 tiling of the box. Each tile's side is (0.5 + `OVERLAP` = 0.10) × the box side. Support tile
  *i* pairs with query tile *i* (forewings above hindwings in the dissected layout). A tile with an unlabelled
  support is skipped.

**Paste-back:** the inner adapter runs unchanged and resizes crops to 1024². Its logits are bilinearly resized to the
crop and pasted into the frame:
- inside the box, the zoomed logits are used, with a channel-wise max where tiles overlap;
- outside the box, or in a skipped tile's area, the full-frame logits stand.

**Cache keys:** crops get their own embedding-cache keys (`<id>@zoom:<y0>,<x0>,<side>`).

## Probe (`scripts/probe_zoom.py` → `results/diagnostics/zoom_probe.json`)

**Model and setting:** base SAM 2 (`sam2_hiera_large`), fp32 on CPU, 1-shot, **no TTA**, plain argmax through the
locked `occl.metrics`.

**Queries:** the first 20 **upright dorsal** test queries of cyrbia and of malleti, 40 in all, in test-split order.
- Orientation is the GT layout vs the train-median layout (`curation.orientation_of`).
- This selection uses labels, and is disclosed. It keeps cyrbia's 180° specimens, which only TTA rescues, from
  diluting a resolution read.

**Supports:** the locked S18b `base_tta` 1-shot draws. For each group, the first 2 runs whose support is upright:
- cyrbia 13148 and 12366;
- malleti 21415 and 13840.

**Four predictions per (support, query), all on the same support:**

| Variant | What it is |
| :-- | :-- |
| **F** | full frame, the locked path without TTA |
| **S** | specimen crop, first-pass box |
| **S\*** | specimen crop, GT box (oracle, diagnostic only: it separates box-finding error from the zoom) |
| **Q** | quadrant tiles of the first-pass box |

The S and S\* crop sizes give the zoom factor, 1024 / side.

**Measured cost:** 6 s for F, 14 s for each specimen variant and 44 s for Q on the first pair, before support crops
were cached. The full run is roughly 1–1.5 h on the 16-core workstation.

## Pre-registered reads and decision

- **Primary:** pooled per-query mIoU, S − F and Q − F.
  - Per query, average over its 2 supports.
  - Bootstrap over the 40 queries: 10,000 resamples, seed 0, 95 % CI; ✱ = excludes zero.
- **Thin vs wide:** per (query, trait) IoU deltas, binned by the GT trait width in 256-grid cells (2 × p90 inside
  distance / 4, as in QUERY_DIAGNOSIS §4): < 5 cells, 5–12, ≥ 12.
- **Box error:** S\* − S.
- **Go (build an SCC inference-only stage on the banked adapters, TTA composed):** the better label-free variant
  gains **≥ +0.5 ✱** pooled, **and** its thin-trait gain exceeds its wide-trait gain.
- **Stop (zoom on cached pixels closed):** both label-free variants are ≤ 0 or not significant. Native-resolution
  crops would then be a separate question, gated on S\*.
- **Expected,** from §6a's headroom (whole-specimen zoom 1.3–2×, quadrants about 2× more): +0.5 to +2 on these
  queries. Tiles risk support↔query wing mismatch.

## Limitations

- 1-shot base SAM 2 without TTA, not the tuned 5-shot TTA champion. The read is a resolution effect, not a Major
  forecast.
- Only cyrbia and malleti upright dorsal queries, and 2 supports per group.
- Cached 1024² pixels: no new detail.
- CPU fp32, not GPU bf16.

## Result (landed 2026-10-06): GO

Tables: [ZOOM_PROBE_RESULTS.md](ZOOM_PROBE_RESULTS.md), from `.venv/bin/python scripts/report_zoom_probe.py`. Data:
`results/diagnostics/zoom_probe.json`, 80 (support, query) pairs on 40 queries.

**Run note.** The first run stopped after 57 pairs when its parent session ended. `--resume` kept those rows and ran
the other 23; the plan was checked to be identical. Each row depends only on its own pair: the first pair gave the
same four scores, to 0.1, in the timing run (support crops uncached) and in the main run (cached).

**Pre-registered reads** (per query averaged over its 2 supports; bootstrap over queries):

| Contrast | pooled | traits < 5 grid cells | 5–12 | ≥ 12 |
| :-- | :-- | :-- | :-- | :-- |
| S − F (specimen crop) | **+1.24 [+0.40, +2.18] ✱** | +6.10 ✱ | −0.64 | −1.23 ✱ |
| Q − F (quadrant tiles) | **+1.55 [+0.55, +2.60] ✱** | +6.54 ✱ | −0.19 | −0.65 |
| S\* − F (GT-box oracle) | +1.84 [+0.75, +3.08] ✱ | +5.84 ✱ | +0.37 | +0.21 |
| S\* − S (box-finding error) | +0.60 [−0.02, +1.25] | −0.26 | +1.01 ✱ | +1.44 ✱ |

- **Means:** F 78.08, S 79.32, Q 79.63, S\* 79.92.
- **Zoom factor:** first-pass median 1.54 (p10 1.28, p90 1.89); GT box 1.48. No first-pass fallbacks.
- **Decision: GO.** The better label-free variant is Q, at +1.55 ✱ ≥ +0.5, and its thin-trait gain (+6.5 ✱) exceeds
  its wide-trait change (−0.65, not significant).
- **The gain is the predicted resolution signature.** It sits entirely in thin traits, while wide traits stay flat or
  slightly down.
  - The per-trait CIs treat (query, trait) units as independent. They are optimistic, and the per-query CIs are the
    primary read.

**Exploratory (not pre-registered; observations, not claims):**
- **Spread:**
  - Q improves 29 of 40 queries and degrades 11 (median +0.9; worst −4.5, best +10.1). S improves 31 and degrades 9.
  - By group: S is significant in cyrbia (+1.51 ✱) and Q in malleti (+1.96 ✱).
- **Support 13840 (malleti) carries most of the losses.** Its Q − F is +0.27 and S − F −0.54, against +1 to +3.7 for
  the other three supports.
  - Malleti trait 8 drops −24.6 under S with 13840, but only −0.6 with 21415. That trait's GT lies wholly inside the
    query box every time, so this is not cropping.
  - 13840's own trait-8 mask covers 0.32 % of the frame against 0.76 % for its mirror trait 5. The zoomed pass appears
    to follow an undersized support mask more literally than the full frame does.
  - Five-shot supports should dilute such single-support quirks; that is untested.
- **Malleti's thin traits 7 and 10 gain most:** +11 to +17 per trait.
- **The gain does not scale with the zoom factor** within the observed range 1.28–1.89 (Spearman ρ −0.06 for Q,
  +0.02 for S).
  - Either the effect saturates early, or the stride-4 decoder grid is what matters, and any crop at 1.3× or more
    clears it.
  - So a larger zoom (native-resolution photos) is not obviously the next lever.
- **Box error is real in malleti:** S\* − S +1.14 ✱, concentrated in wide traits. That makes a padded or two-pass box
  the obvious refinement. It is not done here.

**What this does NOT establish:**
- Whether the gain survives the tuned 5-shot TTA champion. Tuning already lifted thin traits; for example, it moved
  lativitta's rays +5.
- Whether it holds on the other three subspecies. Lativitta's loose ray envelopes and notabilis's hand-drawn half may
  *lose* from edge-exact predictions, as S23's SHAPE did; notabilis's tight half may gain.
- Its Major effect: the 40 queries are cyrbia and malleti upright dorsal.

**Next:** built as **S24** on 2026-10-06 ([S24_PLAN.md](S24_PLAN.md)). The user narrowed it to the best tuned model, quadrant mode, 1-shot and 5-shot. The proposal as first written: The probe answers "is there a
resolution effect", and that stage would answer "does it move Major".
- **Wrapper:** `ZoomAdapter` over `OrientationTTAAdapter`. The first pass is the TTA prediction, and the crops reuse
  its chosen support orientation.
- **Cells:** SAM 2 base and `crown_all_mref` under the label-consistent pool, at 1-shot and 5-shot, in specimen and
  quadrant modes.
- **Pairing:** each cell is paired per seed with its banked parent; draws are identical.
- **Embeddings:** support-crop embeddings are precomputed per run; query crops are encoded per episode.
