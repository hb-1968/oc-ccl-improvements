"""Runtime probes for facts that can only be established on real hardware.

Motivation: SAM 2's attention dispatch is gated by compute capability, and its own
`get_sdpa_settings()` enables **flash only** on any GPU with major >= 8 --
``enable_flash=True, enable_math=False, enable_mem_efficient=False``. There is no second
kernel enabled; the only safety net is a try/except that flips a module-global
``ALLOW_ALL_KERNELS`` after the first failure.

Two of SAM 2's attention shapes are worth probing rather than assuming:

* ``memory_attention.*.self_attn`` is a RoPEAttention with 1 head over internal_dim 256,
  i.e. **head_dim = 256** -- the very top of FlashAttention-2's supported range. FA2's
  *forward* handles it broadly, but its *backward* for head_dim > 192 has historically
  been restricted to sm_80 (A100/H100) and unavailable on sm_86/sm_89 (consumer Ampere,
  Ada/L4). That distinction does not matter for inference, and matters a great deal for
  OC-CCL, which fine-tunes exactly this module.
* The mask decoder's head dims are 32 and 16, which are unremarkable.

Note the image encoder is a different path entirely: Hiera's `MultiScaleAttention` calls
`F.scaled_dot_product_attention` bare, with no kernel context, so PyTorch auto-selects
there and SAM 2's gate never applies.

Everything here is recorded into the stage bundle so the answer is evidence rather than
inference.
"""

from __future__ import annotations

#: (num_heads, head_dim, seq_len) for the attention shapes SAM 2 actually runs outside
#: the image encoder. Sequence lengths are representative, not exact.
SAM2_SHAPES = {
    "memory_attention_self_attn": (1, 256, 4096),
    "mask_decoder_self_attn": (8, 32, 256),
    "mask_decoder_cross_attn_tok": (8, 16, 256),
}


def sdpa_backend_report(device: str = "cuda", dtype=None, train: bool = True) -> dict:
    """Probe which SDPA backends actually dispatch for SAM 2's shapes on this GPU.

    Returns a dict of ``{shape_name: {backend: "ok" | "<error>"}}`` plus a ``backward``
    section when ``train`` is set. A backend that raises here is one SAM 2 would have to
    fall back from at runtime -- slower, but not wrong.
    """
    import torch
    import torch.nn.functional as F

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}

    from torch.nn.attention import SDPBackend, sdpa_kernel

    cap = torch.cuda.get_device_capability(0)
    dtype = dtype or (torch.bfloat16 if cap[0] >= 8 else torch.float16)
    backends = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }

    out: dict = {
        "available": True,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(cap),
        "dtype": str(dtype),
        # This is the exact expression SAM 2 uses; sm_89 (L4) and sm_80 (A100) both pass.
        "sam2_use_flash_attn": cap[0] >= 8,
        "forward": {},
    }
    if train:
        out["backward"] = {}

    for name, (heads, head_dim, seq) in SAM2_SHAPES.items():
        out["forward"][name] = {}
        if train:
            out["backward"][name] = {}
        for bname, backend in backends.items():
            try:
                q = torch.randn(1, heads, seq, head_dim, device=device, dtype=dtype)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, k, v)
                out["forward"][name][bname] = "ok"
            except Exception as e:  # noqa: BLE001
                out["forward"][name][bname] = f"{type(e).__name__}: {str(e)[:120]}"

            if not train:
                continue
            try:
                q = torch.randn(1, heads, seq, head_dim, device=device,
                                dtype=dtype, requires_grad=True)
                k = torch.randn_like(q, requires_grad=True)
                v = torch.randn_like(q, requires_grad=True)
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, k, v).sum().backward()
                out["backward"][name][bname] = "ok"
            except Exception as e:  # noqa: BLE001
                out["backward"][name][bname] = f"{type(e).__name__}: {str(e)[:120]}"

    return out


def allow_all_kernels_state() -> bool | None:
    """Whether SAM 2 has already fallen back off the flash-only path.

    ``True`` after any attention call raised and tripped the global. Read it *after* a
    real forward pass: it is the empirical answer to "did flash actually work here",
    which no amount of capability arithmetic can give you.
    """
    try:
        from sst.segment_anything_2.sam2.modeling.sam import transformer

        return bool(transformer.ALLOW_ALL_KERNELS)
    except Exception:  # noqa: BLE001
        return None
