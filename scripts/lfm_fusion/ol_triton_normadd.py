"""Fused RMSNorm-then-add for OLMo-2.

OLMo-2 is norm-after: each block computes norm(x) and *then* adds the residual,
twice per layer (models/olmo2.py:302-319).

    hidden = self.post_attention_layernorm(hidden)
    hidden = hidden + residual

sglang ships fused_add_rmsnorm, which computes norm(x + residual) -- the
residual goes in *before* the normalisation. That is a different function, so
the primitive that absorbed this pattern on LFM2.5 cannot be reused here, and
the operator audit's own note on the residual_add bucket ("absorbed by
fused_add_rmsnorm") is wrong for this architecture.

Hence an actual new kernel. It reads x once, computes the row RMS, scales by the
weight, adds the residual and writes one output -- replacing an RMSNorm kernel
plus a separate elementwise add, i.e. two kernel launches and one extra full
read/write of the activation per call, 32 times per forward on a 16-layer model.

Numerics follow RMSNorm.forward_native: accumulate in fp32, cast back at the
end, so the result is bit-comparable with the stock path up to the order of the
final add.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_then_add_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    H,
    stride_x_row,
    stride_res_row,
    stride_out_row,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + row * stride_res_row + offs, mask=mask,
                  other=0.0).to(tl.float32)

    # cast the normalised value down before adding, matching the stock path,
    # where the RMSNorm kernel writes bf16 and the add reads it back
    normed = (x * rstd * w).to(out_ptr.dtype.element_ty).to(tl.float32)
    out = normed + res
    tl.store(out_ptr + row * stride_out_row + offs, out.to(out_ptr.dtype.element_ty),
             mask=mask)


def rmsnorm_then_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return norm(x) * weight + residual, in one pass.

    Shapes are [tokens, H] for x and residual, [H] for weight.
    """
    assert x.shape == residual.shape, (x.shape, residual.shape)
    assert x.shape[-1] == weight.shape[0], (x.shape, weight.shape)
    x = x.contiguous() if x.stride(-1) != 1 else x
    residual = residual.contiguous() if residual.stride(-1) != 1 else residual
    tokens, H = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(H)
    num_warps = 4 if block <= 2048 else 8
    _rmsnorm_then_add_kernel[(tokens,)](
        x, residual, weight, out, H,
        x.stride(0), residual.stride(0), out.stride(0),
        eps,
        BLOCK_H=block,
        num_warps=num_warps,
    )
    return out
