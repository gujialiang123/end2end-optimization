"""Fused Triton kernels for the LFM2.5 gated short-convolution path.

The operator audit (`docs/2026-07-27/lfm_fusion_results.md`) left three glue operations
around `causal_conv1d`, all of them pure data movement:

    proj, _ = self.in_proj(hidden_states)        # [T, 3H]
    B_gate, C_gate, x = proj.chunk(3, dim=-1)    # strided views
    Bx = B_gate * x                              # elementwise  -> [T, H]
    Bx_t = Bx.transpose(0, 1).contiguous()       # materialise  -> [H, T]
    conv_out = causal_conv1d_fn(Bx_t, ...).transpose(0, 1)   # view, [T, H]
    output, _ = self.out_proj(C_gate * conv_out) # elementwise, reads transposed

`causal_conv1d_fn` is an opaque external CUDA op that requires
`x.stride(-1) == 1` on a `[dim, seqlen]` tensor, so it acts as a barrier: the
layout change cannot be avoided, only *absorbed* into the neighbouring
elementwise work.

Measured on the long-prefill audit these glue kernels move ~8.8 GB in 10.3 ms,
i.e. **~0.83 TB/s on a part with ~4.8 TB/s of HBM bandwidth — about 17 % of
peak**. That is the real defect: `Bx.transpose(0,1).contiguous()` and the
transposed read inside `C_gate * conv_out` are uncoalesced. So there are two
separate wins available, and they compound:

  1. fewer passes over HBM (fold three passes into two), and
  2. each remaining pass running coalesced via a tiled shared-memory transpose
     instead of a strided element-at-a-time copy.

Both kernels below are plain Triton, correctness-gated against the stock
PyTorch sequence in `lf_bench_shortconv.py`. Tile shapes come from the measured
sweep in `lf_tune_shortconv.py` (`results/lfm_fusion/microbench/
shortconv_tile_sweep.json`), not from guesswork.

Note the shape dependence: the fused kernels sit on a ~30 us floor for small T
(Triton's Python launch path), so they only pay off once there is enough work to
amortise it — measured crossover T ~= 2048 for the input side and ~3000 for the
output side. The caller guards on this; see `CONV_FUSION_MIN_TOKENS` in
`lfm_fusion_patch.py`.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Input side: chunk + gating multiply + transpose, in one pass.
#   reads  proj[:, 0:H] (B_gate) and proj[:, 2H:3H] (x)
#   writes out[H, T] = (B_gate * x)^T
# The tile is read coalesced along H and written coalesced along T; Triton
# keeps the transpose in registers/shared memory rather than issuing a strided
# global access per element.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_gate_transpose_kernel(
    proj_ptr, out_ptr,
    T, H,
    stride_proj_t, stride_proj_h,
    stride_out_h, stride_out_t,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < T
    mask_h = offs_h < H
    mask = mask_t[:, None] & mask_h[None, :]

    # [BLOCK_T, BLOCK_H] tile, contiguous along H
    base = offs_t[:, None] * stride_proj_t + offs_h[None, :] * stride_proj_h
    b = tl.load(proj_ptr + base, mask=mask, other=0.0)
    x = tl.load(proj_ptr + base + 2 * H * stride_proj_h, mask=mask, other=0.0)

    bx = (b.to(tl.float32) * x.to(tl.float32)).to(b.dtype)

    # write transposed: [BLOCK_H, BLOCK_T], contiguous along T
    out_off = offs_h[:, None] * stride_out_h + offs_t[None, :] * stride_out_t
    tl.store(out_ptr + out_off, tl.trans(bx), mask=mask_h[:, None] & mask_t[None, :])


def fused_gate_transpose(proj: torch.Tensor, H: int) -> torch.Tensor:
    """(B_gate * x)^T as one kernel. `proj` is [T, 3H]; returns [H, T]."""
    T = proj.shape[0]
    out = torch.empty((H, T), device=proj.device, dtype=proj.dtype)
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]),
                         triton.cdiv(H, meta["BLOCK_H"]))
    _fused_gate_transpose_kernel[grid](
        proj, out, T, H,
        proj.stride(0), proj.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=64, BLOCK_H=128, num_warps=8, num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Output side: transpose + gating multiply, in one pass.
#   reads conv_out[H, T] and proj[:, H:2H] (C_gate)
#   writes out[T, H] = C_gate * conv_out^T
# ---------------------------------------------------------------------------
@triton.jit
def _fused_transpose_gate_kernel(
    conv_ptr, proj_ptr, out_ptr,
    T, H,
    stride_conv_h, stride_conv_t,
    stride_proj_t, stride_proj_h,
    stride_out_t, stride_out_h,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < T
    mask_h = offs_h < H

    # read conv_out as [BLOCK_H, BLOCK_T] (coalesced along T), then transpose
    conv_off = offs_h[:, None] * stride_conv_h + offs_t[None, :] * stride_conv_t
    c = tl.load(conv_ptr + conv_off,
                mask=mask_h[:, None] & mask_t[None, :], other=0.0)
    c_t = tl.trans(c)                                  # [BLOCK_T, BLOCK_H]

    # C_gate lives at column offset H, read coalesced along H
    mask = mask_t[:, None] & mask_h[None, :]
    g_off = (offs_t[:, None] * stride_proj_t
             + (offs_h[None, :] + H) * stride_proj_h)
    g = tl.load(proj_ptr + g_off, mask=mask, other=0.0)

    res = (g.to(tl.float32) * c_t.to(tl.float32)).to(g.dtype)
    out_off = offs_t[:, None] * stride_out_t + offs_h[None, :] * stride_out_h
    tl.store(out_ptr + out_off, res, mask=mask)


def fused_transpose_gate(conv_out: torch.Tensor, proj: torch.Tensor,
                         H: int) -> torch.Tensor:
    """C_gate * conv_out^T as one kernel. `conv_out` is [H, T]; returns [T, H]."""
    T = conv_out.shape[1]
    out = torch.empty((T, H), device=conv_out.device, dtype=conv_out.dtype)
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]),
                         triton.cdiv(H, meta["BLOCK_H"]))
    _fused_transpose_gate_kernel[grid](
        conv_out, proj, out, T, H,
        conv_out.stride(0), conv_out.stride(1),
        proj.stride(0), proj.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=128, BLOCK_H=128, num_warps=8, num_stages=3,
    )
    return out


# ---------------------------------------------------------------------------
# Decode side: T is the batch size (1..32), so nothing is bandwidth bound and
# nothing is transposed — `causal_conv1d_update` consumes [T, H] directly. The
# only cost is that two tiny elementwise kernels are launched per conv layer,
# 36 launches per forward. One kernel that computes B*x and leaves C_gate for
# the caller halves that.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_mul_kernel(proj_ptr, out_ptr, T, H,
                     stride_proj_t, stride_proj_h,
                     stride_out_t, stride_out_h,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n = T * H
    mask = offs < n
    t = offs // H
    h = offs % H
    base = t * stride_proj_t + h * stride_proj_h
    b = tl.load(proj_ptr + base, mask=mask, other=0.0)
    x = tl.load(proj_ptr + base + 2 * H * stride_proj_h, mask=mask, other=0.0)
    r = (b.to(tl.float32) * x.to(tl.float32)).to(b.dtype)
    tl.store(out_ptr + t * stride_out_t + h * stride_out_h, r, mask=mask)


def fused_gate_mul(proj: torch.Tensor, H: int) -> torch.Tensor:
    """B_gate * x without materialising the chunk views. Returns [T, H]."""
    T = proj.shape[0]
    out = torch.empty((T, H), device=proj.device, dtype=proj.dtype)
    n = T * H
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _gate_mul_kernel[grid](proj, out, T, H,
                           proj.stride(0), proj.stride(1),
                           out.stride(0), out.stride(1),
                           BLOCK=1024, num_warps=4)
    return out
