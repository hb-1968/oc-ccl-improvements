"""SCNP — Same-Class Neighbor Penalization, from Valverde et al., CVPR 2026.

    "Towards High-Quality Image Segmentation: Improving Topology Accuracy by
     Penalizing Neighbor Pixels"  (arXiv 2603.18671)

The transform, verbatim from the paper's Eq. and Algorithm 1, applied to logits **before**
any loss:

    z~_ki = min_{j in W(i), y_kj = 1} z_kj      if y_ki = 1
    z~_ki = max_{j in W(i), y_kj = 0} z_kj      if y_ki = 0

    L_SCNP = L(sigma(Z~), Y)

implemented in three lines with a large masking constant:

    V1 = MinPool((Z * Y) + k(1 - Y), w)
    V2 = MaxPool((Z * (1 - Y)) - k*Y, w)
    Z~ = V1 * Y + V2 * (1 - Y)

**What it does, mechanically.** Every pixel is scored using its *worst same-class
neighbour* instead of itself. A pixel therefore cannot reduce its own loss until its
neighbours improve. The paper's observation is that small structure breakages and isolated
false positives are precisely the pixels that are the worst neighbour of many pixels, so
they collect gradient once per neighbour that selects them — an isolated false positive in a
3x3 window is penalised up to 9 times, a boundary pixel a few times, and a well-predicted
interior pixel not at all.

**Why it is a candidate here.** OC-CCL's cycle already supplies the one thing SCNP needs
that a self-supervised loss usually lacks: a genuine one-hot target. The closing mask is
supervised against `y0`, the reference specimen's own committed labelmap, so `Y` is real
ground truth and the transform applies unchanged. Cost is two pooling ops — the paper
measures +1.47 s/epoch and +340 MiB on a 2D dataset, and our closing mask is a single
1024x1024 channel.

**What it cannot do, in the form S9a uses it.** `finetune.tracker` is single-object:
`track_with_mask` takes one `mask_inputs` channel and `pred_masks_high_res` comes back as
(B, 1, H, W). So the cycle supervises a MERGED FOREGROUND silhouette and SCNP can only
sharpen that silhouette's topology. It cannot touch trait assignment, because there are no
trait channels for a "same-class neighbour" to be same-class *with*. Making the cycle
trait-aware — prompting each trait as its own object, so background pixels of trait k are
penalised by their strongest trait-k neighbour, which is exactly the bleed that costs Beetle
its 15.6 points — is S9b, and it is the change that gives this term something to bite on.

Recording that distinction here rather than in a commit message because the failure it
guards against is subtle: SCNP on a trait-blind cycle would produce a plausible loss curve,
a real (if small) silhouette gain, and no assignment effect whatsoever — and would read as
"the trait-aware loss did not work."

**Known failure mode, from the paper.** Effectiveness on non-tubular structures ranges "from
highly beneficial to detrimental"; it was detrimental on MSLesSeg, whose structures average
447 voxels, the authors hypothesising that the structural coherence and smooth borders SCNP
promotes hurt when structures are very small and contrast is low. Beetle crops are 172-431
px and its traits are small, so that is our nearest analogue and the reason Beetle is the
instrument rather than the assumption.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: The masking constant. The paper writes `k = inf` and its reference implementation uses
#: 9999. A finite value is required — `inf * 0` is NaN, and `(Z * Y) + inf * (1 - Y)`
#: produces NaN wherever Y is exactly 0 and Z is exactly 0. 9999 is far outside any logit
#: range SAM 2 produces (`out_scale=20, out_bias=-10` bounds prompt logits to [-10, 10]).
MASK_CONST = 9999.0


def scnp_logits(logits: torch.Tensor, target: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Replace each logit with its poorest same-class neighbour. Algorithm 1.

    ``logits``/``target``: (B, C, H, W). ``target`` must be one-hot or binary in {0, 1};
    a soft target would make the masks `Y` and `1 - Y` overlap and both branches would
    select from the wrong pool.

    ``width`` is the pooling kernel; the window is centred and **includes the pixel
    itself**, so a foreground pixel always has at least one foreground neighbour (itself)
    and the degenerate "no same-class neighbour in window" case cannot arise. ``width=1``
    makes this the identity, which is what `tests/test_scnp.py` anchors on.
    """
    if width < 1 or width % 2 == 0:
        raise ValueError(f"width must be a positive odd integer, got {width}")
    if logits.shape != target.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} "
                         f"vs target {tuple(target.shape)}")
    if width == 1:
        return logits

    pad = width // 2
    fg = target
    bg = 1.0 - target
    # MinPool via -MaxPool(-x). Non-members are pushed to +k (foreground pass) or -k
    # (background pass) so they can never win their respective extremum.
    v1 = -F.max_pool2d(-(logits * fg + MASK_CONST * bg), width, stride=1, padding=pad)
    v2 = F.max_pool2d(logits * bg - MASK_CONST * fg, width, stride=1, padding=pad)
    return v1 * fg + v2 * bg


def scnp_loss(base_loss, logits: torch.Tensor, target: torch.Tensor,
              width: int = 3, joint: bool = True) -> torch.Tensor:
    """``base_loss`` evaluated on the SCNP-transformed logits.

    ``joint`` adds the untransformed term, ``L(sigma(Z~), Y) + L(sigma(Z), Y)``. The paper
    offers this "to reduce convergence delay", and it is the default here for a reason
    specific to us: `s8c_tune` chose **one epoch** by cross-validation, so we are in exactly
    the regime where a transform that slows convergence is most likely to be mistaken for a
    transform that does not work. Pass ``joint=False`` for the paper's plain form; the arms
    are cheap to run both ways and the S9 gate reports both.
    """
    out = base_loss(scnp_logits(logits, target, width), target)
    return out + base_loss(logits, target) if joint else out


def neighbor_gradient_multiplicity(logits: torch.Tensor, target: torch.Tensor,
                                   width: int = 3) -> torch.Tensor:
    """How many pixels select each pixel as their poorest same-class neighbour.

    The paper's mechanism, made measurable: "poorest misclassifications are penalised as
    many times as they are propagated by SCNP". An isolated false positive should collect up
    to ``width**2``; a correct interior pixel should collect 0.

    This is a diagnostic, not part of the loss — it is what the S9 gate asserts to show the
    term is doing what it claims on OUR data rather than on the paper's. Returned as counts
    per pixel, shape (B, C, H, W).
    """
    z = logits.detach().clone().requires_grad_(True)
    # A linear read-out, so d(sum Z~)/dz_i counts exactly the selections of pixel i.
    scnp_logits(z, target, width).sum().backward()
    return z.grad.detach().abs()
