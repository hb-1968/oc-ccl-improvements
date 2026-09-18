"""(reference, query, reference-mask) sampling over a dataset's TRAIN split.

SST has no Beetle equivalent of this — `src/sst/butterfly_dataset.py` is its only OC-CCL
dataset — so the sampler is ours. That makes two things our responsibility to get right:

**A1, the normalisation.** Pixels come from `imagecache.ImageCache` and go through
`adapters.sst_sam2.to_model_tensor`, which is the same function the inference path uses.
Training therefore sees byte-identical input to evaluation, rather than SST's `/255`-only
tensors (`butterfly_dataset.py:139-140`). This is asserted directly in the tests, not
argued.

**Train-only, and seeded.** Pairs are drawn from `ds.train_ids()` exclusively — a test
image reaching the fine-tune would leak the evaluation set into the weights and make every
downstream number meaningless. Ordering comes from `seeding.derive_seed`, so a training run
is reproducible the same way an evaluation run is.

**The mask axis: merged foreground vs per-trait channels.** The *released* SST code
supervises one merged-foreground mask — `butterfly_dataset.py:121` does `(mask > 0)` and
`oc_ccl.py` carries a single `(B, 1, H, W)` channel — so a merged cycle trains a
*silhouette* objective while the metric judging it is per-trait. **The paper does not
describe that.** Its notation is per-trait throughout: §3 defines y ∈ {0,1}^{W×H×C} over
"C distinct traits", OC-CCL supervises ŷ₀† against that same y₀, and §3.3 explicitly
reserves single-channel handling for retrieval ("y₀★ — a single channel in the original
y₀"). The merge is the release's regression (the OC-CCL code landed 16 months after the
paper), not the paper's design. Both are reachable via ``mask_mode``, because a deviation
that cannot be un-applied cannot be measured — `merged` reproduces the release,
`per_trait` reproduces the paper.

**The reference axis.** The released code enumerates every train image as a labeled
reference (`all_entries[idx]`, so one epoch consumes the whole split's labels); the paper
instead states "We take one fixed labeled image as x₀, and iteratively sample x₁ from the
unlabeled training set" (A2.2) — ONE label. ``reference_id`` selects the paper's protocol;
``None`` keeps the derangement (every image once in each role), which is this repo's
release-shaped sampler.

**Rotation augmentation (Beetle only, per A2.2).** "due to the inherently high variation
in beetle orientations, we introduce a small degree of random 2D rotation augmentation on
the labeled example during the fine-tuning stage". Applied to the *reference* (image and
labelmap jointly), never the query, and in **model space** (after the square resize):
beetle photographs are non-square, so a native-space rotation would couple with the
anisotropic 1024x1024 resize into a shear — model space is the reading under which the
image/mask pair stays rigid as the model sees it. The magnitude is unquantified in the
paper; the caller supplies it under an a-priori rule recorded in the run report.
"""

from __future__ import annotations

import numpy as np

from ..adapters.sst_sam2 import to_model_tensor
from ..seeding import derive_seed


def foreground_mask(labelmap: np.ndarray, size: int = 1024):
    """Merged foreground as a (1, 1, size, size) float tensor in {0.0, 1.0}.

    This is the *release's* mask — `butterfly_dataset.py:121` `(mask > 0)` — kept byte-for-
    byte reachable as ``mask_mode="merged"``. The paper's per-trait mask is
    :func:`trait_mask_stack`.

    NEAREST resize, because a labelmap is categorical — bilinear would invent labels
    between traits. Binarised *after* resizing so a trait that survives resampling anywhere
    contributes.
    """
    import cv2
    import torch

    lm = cv2.resize(labelmap.astype(np.int32), (size, size),
                    interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((lm > 0).astype(np.float32))[None, None]


def trait_mask_stack(labelmap: np.ndarray, size: int = 1024):
    """Per-trait masks as ``((C, 1, size, size) float tensor, class_ids tuple)``.

    The paper's y₀ ∈ {0,1}^{W×H×C}: one channel per trait *present in this reference*, in
    ascending trait-id order. A trait absent from the reference gets **no channel** —
    exactly the eval adapter's rule (`adapters/sst_sam2.py` skips a class absent from the
    support rather than prompting an empty mask), so training and evaluation agree on what
    an incomplete reference means. The channel dimension is SAM 2's *object batch*
    dimension: C objects tracked through one pass, the same mechanism multi-object
    evaluation already uses.

    Same resize discipline as :func:`foreground_mask`: one NEAREST resize of the integer
    labelmap, channels split afterwards, so the two modes see identical geometry and
    ``per_trait`` at C=1 is bit-identical to ``merged`` when the reference has one trait.
    """
    import cv2
    import torch

    lm = cv2.resize(labelmap.astype(np.int32), (size, size),
                    interpolation=cv2.INTER_NEAREST)
    ids = tuple(int(t) for t in np.unique(lm) if int(t) != 0)
    if not ids:
        raise ValueError("reference labelmap has no traits — nothing to supervise")
    stack = np.stack([(lm == t) for t in ids]).astype(np.float32)
    return torch.from_numpy(stack)[:, None], ids


class CyclePairs:
    """Deterministic (reference, query) pairs over a group's train split.

    Every train image serves as the reference once per epoch, paired with a different train
    image as the query. Sampling without replacement from the remainder means no pair is
    degenerate (reference == query), which would make the cycle trivially satisfiable by
    the identity and teach nothing.
    """

    #: A1 as an AXIS, not just a fix. "imagenet" is ours — the same tensor the inference
    #: path builds, asserted element-for-element identical. "unit" is what SST's own
    #: `butterfly_dataset.py:138-139` hands the trainer: `/255` alone, so the model trains
    #: on [0,1] and is scored on roughly [-2.1, +2.6]. Kept reachable so a replication of
    #: their literal code is possible; a fix that cannot be un-applied cannot be measured.
    NORMALIZATIONS = ("imagenet", "unit", "half")

    #: The supervision axis — see the module docstring. "merged" is the release's
    #: silhouette (`(mask > 0)`); "per_trait" is the paper's y ∈ {0,1}^{W×H×C}.
    MASK_MODES = ("merged", "per_trait")

    def __init__(self, ds, group: str | None = None, size: int = 1024, epoch: int = 0,
                 normalization: str = "imagenet",
                 require_cached: bool = True,
                 mask_mode: str = "merged",
                 reference_id: str | None = None,
                 rotation_aug_deg: float | None = None,
                 mask_size: int | None = None,
                 close_aug_deg: float | None = None):
        """``group=None`` pools every group — the fine-tune is dataset-wide.

        Beetle has a single universal group, so this changes nothing there. Butterfly Major
        has five subspecies and no "all" group at all: asking for one raises
        `FileNotFoundError` on a split file that does not exist, which is how the first
        butterfly fine-tune died.

        Pairs are deranged **within** a subspecies and only then interleaved. Cross-subspecies
        pairs would not crash — the cycle supervises a merged foreground, so the differing
        trait vocabularies (6-10) never surface — but they would ask the model to
        reconstruct a cyrbia silhouette from a malleti prompt, a task the evaluation never
        poses, since supports and queries always come from the same subspecies.

        ``reference_id`` switches to the paper's protocol: that ONE image is the reference
        of every pair, and the queries are all the group's *other* usable train images in
        the seeded permutation order (A2.2: "iteratively sample x₁ from the unlabeled
        training set"). Pairs per epoch = N − 1 instead of N. When ``group`` is None the
        group is resolved to the one containing the reference — the paper fine-tunes per
        species, so a fixed reference never crosses subspecies. The label budget of such a
        run is exactly 1, which is what the paper's headline claims.

        ``rotation_aug_deg`` enables the beetle A2.2 augmentation: per-step angle drawn
        uniform(−θ, +θ) from `derive_seed(ds, "rotaug|ref|query", epoch, i)`, applied to
        the reference image and labelmap jointly in model space. If a rotation clips a
        trait out of frame entirely, that step falls back to 0° and the fallback is
        counted — a supervision channel that exists in the prompt but is empty in the
        target would teach deletion.

        ``close_aug_deg`` (S18 arm `perturb`) enables :meth:`closing`: the CLOSING end of
        the palindrome is the reference under its own independently seeded rotation
        (`derive_seed(ds, "closeaug|ref|query", epoch, i)` — never the ``rotaug`` seed),
        drawn from the raw reference, so the opening ``x0`` the forward leg is prompted
        from is untouched and the closing frame differs from it by one rigid motion the
        forward leg cannot predict. ``__getitem__`` is unchanged.
        """
        self.ds = ds
        self.group = group
        self.size = size
        # S14: sam3 takes 1008 images but 1152 mask prompts (`input_mask_size`);
        # None keeps masks at `size` — byte-stable for every sam2-era caller.
        self.mask_size = size if mask_size is None else mask_size
        self.epoch = epoch
        if normalization not in self.NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {self.NORMALIZATIONS}, "
                             f"got {normalization!r}")
        self.normalization = normalization
        if mask_mode not in self.MASK_MODES:
            raise ValueError(f"mask_mode must be one of {self.MASK_MODES}, "
                             f"got {mask_mode!r}")
        self.mask_mode = mask_mode
        self.reference_id = reference_id
        self.reference_mode = "fixed" if reference_id is not None else "derangement"
        if rotation_aug_deg is not None and rotation_aug_deg <= 0:
            raise ValueError(f"rotation_aug_deg must be positive or None, "
                             f"got {rotation_aug_deg!r}")
        self.rotation_aug_deg = rotation_aug_deg
        if close_aug_deg is not None and close_aug_deg <= 0:
            raise ValueError(f"close_aug_deg must be positive or None, got {close_aug_deg!r}")
        self.close_aug_deg = close_aug_deg
        #: One entry per __getitem__ call, in call order: the pair, the per-trait
        #: channel ids, and the augmentation actually applied (drawn vs effective angle,
        #: fallback flag). `train_lora` reads this after the epoch so the record carries
        #: what ran, not what was configured.
        self.step_log: list[dict] = []
        takes = _takes_group(ds.train_ids)
        if reference_id is not None and group is None:
            owners = [g for g in ds.groups()
                      if reference_id in (ds.train_ids(g) if takes else ds.train_ids())]
            if not owners:
                raise ValueError(f"reference {reference_id!r} is in no group's train split")
            groups = owners[:1]
        else:
            groups = [group] if group is not None else list(ds.groups())
        cache = getattr(ds, "cache", None)

        self.pairs: list[tuple[str, str]] = []
        self.ids: list[str] = []
        self.coverage: dict[str, tuple[int, int]] = {}     # group -> (available, total)
        for g in groups:
            ids = sorted(ds.train_ids(g) if takes else ds.train_ids())
            total = len(ids)
            if require_cached and cache is not None:
                # The image cache is built for the EVALUATION protocol — test queries plus
                # the supports each selection arm draws — not for the whole train split.
                # Beetle's 180 images are fully covered; Butterfly Major has 4,227 train
                # images of which ~1,100 are cached, so an unfiltered fine-tune dies on the
                # first uncached reference.
                #
                # Filtering is the cheap choice: caching the remainder costs ~5.9 GB to
                # build and upload, to enlarge a training set for an effect measured as
                # negative in every arm so far. It IS a deviation, so the fraction is
                # recorded and printed rather than absorbed silently.
                ids = [i for i in ids if cache.has(ds.name, i)]
            self.coverage[g] = (len(ids), total)
            if len(ids) < 2:
                raise ValueError(
                    f"{ds.name}/{g}: need >= 2 usable train images, got {len(ids)} "
                    f"of {total} (cached)")
            self.ids += ids
            rng = np.random.default_rng(derive_seed(ds.name, g, 0, epoch))
            if self.reference_id is not None:
                if self.reference_id not in ids:
                    raise ValueError(
                        f"reference {self.reference_id!r} is not a usable train image of "
                        f"{ds.name}/{g} (not in the train split, or not cached)")
                # The paper's protocol: one fixed labeled x0, queries drawn from the rest.
                # Same seed derivation as the derangement so the two modes differ only in
                # structure, never in seeding discipline.
                others = [i for i in ids if i != self.reference_id]
                order = rng.permutation(len(others))
                self.pairs += [(self.reference_id, others[int(j)]) for j in order]
            else:
                order = rng.permutation(len(ids))
                # Derangement by rotation: pair i with order[(i+1) % n], so query !=
                # reference for every i and each image appears exactly once in each role.
                self.pairs += [(ids[int(order[i])], ids[int(order[(i + 1) % len(order)])])
                               for i in range(len(order))]

        avail = sum(a for a, _ in self.coverage.values())
        tot = sum(t for _, t in self.coverage.values())
        self.cache_fraction = avail / tot if tot else 1.0
        if avail < tot:
            print(f"  CyclePairs: training on {avail}/{tot} train images "
                  f"({100 * self.cache_fraction:.1f}%) — the rest are not in the image "
                  f"cache, which is built for the evaluation protocol. Per group: "
                  + ", ".join(f"{g} {a}/{t}" for g, (a, t) in self.coverage.items()),
                  flush=True)

        if len(groups) > 1:
            # Interleave, or the model trains on ~190 cyrbia pairs, then ~190 lativitta,
            # and so on — a curriculum nobody chose, with the last subspecies seen most
            # recently by the optimiser.
            shuffle = np.random.default_rng(derive_seed(ds.name, "__pairs__", 0, epoch))
            self.pairs = [self.pairs[int(i)] for i in shuffle.permutation(len(self.pairs))]

    def __len__(self) -> int:
        return len(self.pairs)

    def _to_tensor(self, img):
        """A1. `imagenet` matches inference exactly; `unit` is SST's `/255` alone.

        The imagenet branch delegates to `adapters.sst_sam2.to_model_tensor` rather
        than restating the constants — a training-side copy of the transform is what
        created A1 in the first place. `half` (S14, sam3) delegates to
        `adapters.sam3.to_model_tensor_sam3` for exactly the same reason: the
        inference path is the authority.
        """
        if self.normalization == "imagenet":
            return to_model_tensor(img, self.size)
        if self.normalization == "half":
            from ..adapters.sam3 import to_model_tensor_sam3

            return to_model_tensor_sam3(img, self.size)
        import cv2
        import torch

        arr = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return (torch.from_numpy(arr).permute(2, 0, 1).float()
                .div_(255.0).unsqueeze(0))

    def _rotate_reference(self, img: np.ndarray, lm: np.ndarray,
                          ref_id: str, query_id: str, i: int):
        """A2.2's beetle augmentation, in model space. Returns ``(img, lm, aug_record)``.

        Both arrays are first brought to ``(size, size)`` by the SAME resamplers the
        un-augmented path uses (LINEAR image, NEAREST labelmap), then rotated about the
        centre by one seeded angle — so the pair transforms rigidly and the later resizes
        inside `_to_tensor`/`trait_mask_stack` are identities. Border fill is 0
        (background), the standard augmentation default. If the rotation clips away every
        pixel of any trait the reference had, the step falls back to 0° and says so.
        """
        return self._rotate(img, lm, ref_id, query_id, i, self.rotation_aug_deg, "rotaug")

    def _rotate(self, img: np.ndarray, lm: np.ndarray, ref_id: str, query_id: str,
                i: int, max_deg: float, tag: str):
        """One seeded rotation of (image, labelmap) in model space — `_rotate_reference`'s
        body, with the seed's tag a parameter so the S18 closing-end rotation
        (``"closeaug"``) is independent of A2.2's (``"rotaug"``) by construction."""
        import cv2

        rng = np.random.default_rng(
            derive_seed(self.ds.name, f"{tag}|{ref_id}|{query_id}", self.epoch, i))
        drawn = float(rng.uniform(-max_deg, max_deg))

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        lm = cv2.resize(lm.astype(np.int32), (self.size, self.size),
                        interpolation=cv2.INTER_NEAREST)
        c = (self.size / 2 - 0.5, self.size / 2 - 0.5)
        m = cv2.getRotationMatrix2D(c, drawn, 1.0)
        img_r = cv2.warpAffine(img, m, (self.size, self.size),
                               flags=cv2.INTER_LINEAR, borderValue=0)
        lm_r = cv2.warpAffine(lm, m, (self.size, self.size),
                              flags=cv2.INTER_NEAREST, borderValue=0)

        lost = (set(np.unique(lm)) - {0}) - (set(np.unique(lm_r)) - {0})
        if lost:
            return img, lm, {"aug_drawn": drawn, "aug_angle": 0.0, "aug_fallback": True}
        return img_r, lm_r, {"aug_drawn": drawn, "aug_angle": drawn,
                             "aug_fallback": False}

    def closing(self, i: int):
        """S18: ``(x0_close, y0_close, record)`` — the closing end of pair ``i`` under
        ``close_aug_deg``: the reference image and label map rotated together by one
        independently seeded angle (`_rotate`, tag ``closeaug``), tensorised exactly as
        the opening end is. The channels are the opening's by construction (the rotation
        falls back to 0° rather than lose a trait; asserted here, never assumed). The
        record carries ``close_drawn`` / ``close_angle`` / ``close_fallback`` for the
        caller's step log.
        """
        if self.close_aug_deg is None:
            raise ValueError("closing() needs close_aug_deg")
        ref_id, query_id = self.pairs[i]
        img0 = self.ds.load_image(ref_id)
        lm0 = self.ds.load_labelmap(ref_id)
        img_c, lm_c, rec = self._rotate(img0, lm0, ref_id, query_id, i, self.close_aug_deg,
                                        "closeaug")
        x0c = self._to_tensor(img_c)
        if self.mask_mode == "per_trait":
            y0c, ids_c = trait_mask_stack(lm_c, self.mask_size)
            _y0, ids_0 = trait_mask_stack(lm0, self.mask_size)
            if tuple(ids_c) != tuple(ids_0):
                raise RuntimeError(f"pair {i} {ref_id}: the closing rotation changed the "
                                   f"trait channels {tuple(ids_0)} -> {tuple(ids_c)}")
        else:
            y0c = foreground_mask(lm_c, self.mask_size)
        return x0c, y0c, {"close_drawn": rec["aug_drawn"], "close_angle": rec["aug_angle"],
                          "close_fallback": rec["aug_fallback"]}

    def __getitem__(self, i: int):
        """``(x0, xu, y0, (ref_id, query_id))`` — batch dimension already present.

        ``y0`` is ``(1, 1, size, size)`` under ``merged`` and ``(C, 1, size, size)`` under
        ``per_trait`` — the leading dimension is SAM 2's object batch. Each call appends
        one entry to :attr:`step_log` (pair, per-trait channel ids, augmentation applied).
        """
        ref_id, query_id = self.pairs[i]
        img0 = self.ds.load_image(ref_id)
        lm0 = self.ds.load_labelmap(ref_id)
        aug = None
        if self.rotation_aug_deg is not None:
            img0, lm0, aug = self._rotate_reference(img0, lm0, ref_id, query_id, i)
        x0 = self._to_tensor(img0)
        xu = self._to_tensor(self.ds.load_image(query_id))
        if self.mask_mode == "per_trait":
            y0, class_ids = trait_mask_stack(lm0, self.mask_size)
        else:
            y0, class_ids = foreground_mask(lm0, self.mask_size), None
        self.step_log.append({"i": i, "ref": ref_id, "query": query_id,
                              "class_ids": list(class_ids) if class_ids else None,
                              **(aug or {})})
        return x0, xu, y0, (ref_id, query_id)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def _takes_group(fn) -> bool:
    """Beetle's `train_ids()` is universal; ButterflyMajor's takes a group."""
    import inspect

    try:
        return bool(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # builtins / C-implemented
        return False
