"""Triton-fused Hillis-Steele parallel prefix scan for the complex-affine
recurrence: (a_t, b_t) ∘ (a_{t-1}, b_{t-1}) = (a_t·a_{t-1}, a_t·b_{t-1} + b_t).

Inputs/outputs are paired real tensors (a_re, a_im, b_re, b_im), each shaped
[B, T, M]. Produces the cumulative scan outputs along the T axis.

STATUS: PROTOTYPE. Ship behind a flag. Validate numerically before routing the
adaptive substrate through it. Fallback to the F.pad-based scan when Triton
isn't available, the kernel raises, or the sequence length exceeds the
single-tile capacity (scan fits in registers/shared-memory only up to ~T=1024
at small BLOCK_M).

Reference: the same associative operator + Hillis-Steele structure as the
F.pad scan in causal_bank_torch.py, but all log(T) levels run in one CUDA
kernel via `tl.associative_scan` with tuple support (Triton 2.2+).
"""
from __future__ import annotations

import torch


def _have_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except Exception:
        return False


def _have_associative_scan_tuple() -> bool:
    """tl.associative_scan with tuple support lands in triton 2.2."""
    try:
        import triton
        # Version parse is approximate; API is what matters
        parts = triton.__version__.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor) >= (2, 2)
    except Exception:
        return False


if _have_triton() and _have_associative_scan_tuple():
    import triton
    import triton.language as tl

    @triton.jit
    def _combine_complex_affine(
        l_ar, l_ai, l_br, l_bi,
        r_ar, r_ai, r_br, r_bi,
    ):
        """Combine left (earlier) and right (later) scan elements.

        (l_a, l_b) is the cumulative so far; (r_a, r_b) is the current element.
        Right-applied:  result = (r_a · l_a, r_a · l_b + r_b).
        """
        new_ar = r_ar * l_ar - r_ai * l_ai
        new_ai = r_ar * l_ai + r_ai * l_ar
        new_br = r_ar * l_br - r_ai * l_bi + r_br
        new_bi = r_ar * l_bi + r_ai * l_br + r_bi
        return new_ar, new_ai, new_br, new_bi

    @triton.jit
    def _scan_kernel(
        a_re_ptr, a_im_ptr, b_re_ptr, b_im_ptr,
        o_ca_re, o_ca_im, o_cb_re, o_cb_im,
        B, T, M,
        stride_b, stride_t, stride_m,
        T_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        mblock = tl.program_id(1)
        offs_t = tl.arange(0, T_POW2)
        offs_m = mblock * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_t = offs_t < T
        mask_m = offs_m < M
        mask = mask_t[:, None] & mask_m[None, :]

        base = batch_idx * stride_b + offs_t[:, None] * stride_t + offs_m[None, :] * stride_m

        # Pad beyond T with identity element (a=1, b=0) so the scan's implicit
        # prefix doesn't alter valid positions when T_POW2 > T.
        ar = tl.load(a_re_ptr + base, mask=mask, other=1.0)
        ai = tl.load(a_im_ptr + base, mask=mask, other=0.0)
        br = tl.load(b_re_ptr + base, mask=mask, other=0.0)
        bi = tl.load(b_im_ptr + base, mask=mask, other=0.0)

        ca_re, ca_im, cb_re, cb_im = tl.associative_scan(
            (ar, ai, br, bi), 0, _combine_complex_affine,
        )

        tl.store(o_ca_re + base, ca_re, mask=mask)
        tl.store(o_ca_im + base, ca_im, mask=mask)
        tl.store(o_cb_re + base, cb_re, mask=mask)
        tl.store(o_cb_im + base, cb_im, mask=mask)


def triton_complex_affine_scan(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused parallel prefix scan of (a, b) complex-affine recurrence.

    All inputs [B, T, M], float32, on same CUDA device. Returns (ca_re, ca_im,
    cb_re, cb_im) with the same shape. At T up to ~1024 and BLOCK_M=16 the
    scan fits in one kernel launch; beyond that this raises (caller must
    fall back to the F.pad-based Python scan until a segmented version lands).
    """
    if not (_have_triton() and _have_associative_scan_tuple()):
        raise RuntimeError("triton or tl.associative_scan tuple API unavailable")

    if not (a_re.is_cuda and a_re.dtype == torch.float32):
        raise RuntimeError("triton scan expects float32 CUDA tensors")

    import triton

    assert a_re.shape == a_im.shape == b_re.shape == b_im.shape
    B, T, M = a_re.shape
    T_POW2 = triton.next_power_of_2(T)
    # Keep the (T, BLOCK_M) tile under ~16K elements to fit register budget.
    # A4000 has 64 KB register file per SM; 16K fp32 = 64 KB is right at the
    # edge. Stay below with BLOCK_M=8 when T_POW2 >= 1024.
    if T_POW2 >= 1024:
        BLOCK_M = 8
    elif T_POW2 >= 512:
        BLOCK_M = 16
    else:
        BLOCK_M = 32

    if T_POW2 > 2048:
        raise RuntimeError(
            f"triton scan prototype supports T_POW2 <= 2048; got {T_POW2}. "
            "Use the F.pad Python scan until segmented Triton is implemented."
        )

    o_ca_re = torch.empty_like(a_re)
    o_ca_im = torch.empty_like(a_re)
    o_cb_re = torch.empty_like(a_re)
    o_cb_im = torch.empty_like(a_re)

    grid = (B, triton.cdiv(M, BLOCK_M))
    _scan_kernel[grid](
        a_re, a_im, b_re, b_im,
        o_ca_re, o_ca_im, o_cb_re, o_cb_im,
        B, T, M,
        a_re.stride(0), a_re.stride(1), a_re.stride(2),
        T_POW2=T_POW2,
        BLOCK_M=BLOCK_M,
    )
    return o_ca_re, o_ca_im, o_cb_re, o_cb_im
