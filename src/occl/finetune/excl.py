"""Cross-channel exclusivity for the per-trait cycle — S13 arm ``excl``.

Provenance: the STM/XMem/Cutie lineage enforces multi-object exclusivity **in the
loss**, not the architecture. Per-object sigmoid probabilities are produced
independently — exactly like SAM 2's object batch — then a background is synthesized as
``p0 = prod_c(1 - p_c)``, every channel is logit-transformed, and a softmax
cross-entropy supervises the C+1 stack jointly (STM, ICCV 2019; Cutie CVPR 2024,
``cutie/utils/tensor_utils.py:aggregate``). SAM 2 produces the same independent
sigmoids and simply never aggregates, so the port is a pure loss-level change — and one
that transfers to SAM 3 (which keeps SAM 2's tracker and per-object mask logits) by
construction, and to Cutie for free (it is Cutie's native objective).

Two modes, one identity apart. Because ``logit(sigmoid(z)) = z``, the STM aggregation
reduces to ``softmax([logit(p0), z_1..z_C])`` — it differs from the plain
zero-background concatenation ``softmax([0, z_1..z_C])`` (``zerocat``) ONLY in the
background logit. And ``zerocat`` is this project's evaluation rule verbatim:
`metrics.logits_to_labelmap` documents itself as argmax over the logits with a
constant-zero background channel appended. So ``zerocat`` trains literally the decision
rule the model is scored by, and is the S13 arm setting; ``aggregate`` keeps the ported
STM form reachable (a deviation that cannot be applied cannot be measured).

Anchors, both pinned by test:
- At C=1, ``zerocat`` CE is ``softmax([0, z])`` = per-pixel BCE-with-logits —
  identically `cycle.bce_loss` (both are global means). Single-trait references degrade
  to the current objective with no special case.
- At C=1, ``aggregate`` is ``softmax([-z, z]) = sigmoid(2z)`` — BCE at temperature 2, a
  silent objective change, which is exactly why it is not the default.

bf16 hazard: ``1 - 1e-7`` is not representable in bf16 (eps ~ 7.8e-3 rounds it to 1.0),
so the aggregate branch computes its background logit in fp32 via
``log p0 = -softplus(z).sum(0)`` — stable with no clamp on the log side; one fp32 clamp
guards the ``log1p`` when p0 -> 1 (every channel confidently absent). ``zerocat`` needs
no clamps at all: `F.cross_entropy` does log-softmax internally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EXCL_MODES = ("off", "aggregate", "zerocat")


def exclusive_target(y0: torch.Tensor) -> torch.Tensor:
    """``(C, 1, H, W)`` channel stack -> ``(H, W)`` long labelmap, 0 = background.

    Channels partition the foreground (`pairs.trait_mask_stack` builds one channel per
    trait present in the reference, pinned by test), so the argmax is exact, not a
    tie-break. The 0.5 threshold guards the resize-guard soft-rebind case
    (`cycle.oc_ccl_step` step 5) the same way `scnp.py` documents for its targets.
    """
    hard = y0[:, 0] > 0.5
    tgt = torch.zeros(hard.shape[-2:], dtype=torch.long, device=y0.device)
    any_fg = hard.any(dim=0)
    tgt[any_fg] = hard.float().argmax(dim=0)[any_fg] + 1
    return tgt


def excl_ce(pred_logits: torch.Tensor, y0: torch.Tensor, mode: str) -> torch.Tensor:
    """Softmax CE over ``[background, trait channels]`` — the exclusivity term.

    Replaces the BCE slot in `cycle.oc_ccl_step` (equal to it at C=1 under ``zerocat``);
    Dice stays on the independent sigmoids so the arm varies exactly one mechanism.
    """
    if mode not in EXCL_MODES or mode == "off":
        raise ValueError(f"excl mode must be 'aggregate' or 'zerocat', got {mode!r}")
    z = pred_logits[:, 0].float()  # (C, H, W); fp32 so bf16 autocast cannot bite
    if mode == "zerocat":
        bg = torch.zeros_like(z[:1])  # THE eval rule's background channel
    else:
        # STM: bg logit = logit(prod_c(1 - sigmoid(z_c))).
        # log p0 = sum_c log sigmoid(-z_c) = -sum_c softplus(z_c), exactly.
        lp0 = -F.softplus(z).sum(dim=0, keepdim=True)
        bg = lp0 - torch.log1p(-lp0.exp().clamp(max=1.0 - 1e-6))
    cat = torch.cat([bg, z], dim=0)  # (C+1, H, W)
    return F.cross_entropy(cat[None], exclusive_target(y0)[None])


def softmax_dice(pred_logits: torch.Tensor, y0: torch.Tensor) -> float:
    """Cutie's Dice reading — Dice on the softmax foreground probabilities. DETACHED.

    Recorded as a diagnostic only: moving Dice onto the softmax simultaneously with the
    CE swap would confound two mechanism changes in one arm. This value existing in the
    record is what lets a follow-up arm be motivated without having been run.
    """
    with torch.no_grad():
        z = pred_logits[:, 0].float()
        probs = torch.softmax(
            torch.cat([torch.zeros_like(z[:1]), z], dim=0), dim=0)[1:]
        p = probs.flatten(1)
        t = (y0[:, 0] > 0.5).float().flatten(1)
        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        return float((1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean())
