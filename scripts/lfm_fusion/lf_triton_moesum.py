"""Fused top-k MoE reduction, residual add, and RMSNorm for LFM2.5.

The stock deferred-residual path materialises the reduced MoE output before the
next layer consumes it:

    partials[T, 4, H] -> moe_sum[T, H] -> fused_add_rmsnorm

This kernel keeps the top-k reduction result in registers, adds the residual,
computes the row RMS, and writes only the updated residual and normalized
activation. LFM2.5 uses BF16, top_k=4, and H=2048.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


TOP_K = 4


@triton.jit
def _moesum_add_rmsnorm_kernel(
    partials_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    H,
    stride_partials_t,
    stride_partials_k,
    stride_partials_h,
    stride_residual_t,
    stride_residual_h,
    stride_output_t,
    stride_output_h,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H
    partial_base = row * stride_partials_t + offs_h * stride_partials_h

    summed = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(0, 4):
        value = tl.load(
            partials_ptr + partial_base + k * stride_partials_k,
            mask=mask,
            other=0.0,
        )
        summed += value.to(tl.float32)

    # The stock MoE reducer writes BF16 before fused_add_rmsnorm reads it.
    summed = summed.to(tl.bfloat16).to(tl.float32)
    residual_offsets = row * stride_residual_t + offs_h * stride_residual_h
    residual = tl.load(
        residual_ptr + residual_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    x = summed + residual

    tl.store(residual_ptr + residual_offsets, x, mask=mask)
    variance = tl.sum(x * x, axis=0) / H
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offs_h, mask=mask, other=0.0).to(tl.float32)
    output_offsets = row * stride_output_t + offs_h * stride_output_h
    tl.store(output_ptr + output_offsets, x * inv_rms * weight, mask=mask)


def fused_moesum_add_rmsnorm(
    partials: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``partials`` and apply residual-add RMSNorm in one kernel."""
    if partials.ndim != 3 or partials.shape[1] != TOP_K:
        raise ValueError(f"expected partials [T,{TOP_K},H], got {partials.shape}")
    T, _, H = partials.shape
    if residual.shape != (T, H):
        raise ValueError(f"expected residual {(T, H)}, got {residual.shape}")
    if weight.shape != (H,):
        raise ValueError(f"expected weight {(H,)}, got {weight.shape}")
    if (
        partials.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
    ):
        raise TypeError("moesum fusion currently supports BF16 only")
    if not partials.is_contiguous() or not residual.is_contiguous():
        raise ValueError("partials and residual must be contiguous")

    output = torch.empty_like(residual)
    block_h = triton.next_power_of_2(H)
    _moesum_add_rmsnorm_kernel[(T,)](
        partials,
        residual,
        weight,
        output,
        H,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        residual.stride(0),
        residual.stride(1),
        output.stride(0),
        output.stride(1),
        eps,
        BLOCK_H=block_h,
        num_warps=8,
        num_stages=1,
    )
    return output, residual
