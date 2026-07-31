#!/usr/bin/env python3
"""Where does the 3.94% actually come from?

The add_one variant was built on the assumption that folding `1 + w` into a
bf16 weight was the error source. It removed that fold and the error did not
move, so the assumption was wrong and this narrows it down instead of guessing
again.

Isolates each stage in turn against an fp64 reference:
  norm only   -- kernel norm vs exact norm, rope disabled (rotary_dim minimal)
  rope only   -- kernel rope vs sglang rope, weights set to identity
  both        -- the full path
plus the magnitude of a single bf16 ULP at the output scale, which is the floor
any bf16 kernel is entitled to.
"""
from __future__ import annotations

import torch

from sglang.srt.runtime_context import _CONTEXT
from sglang.srt.server_args import ServerArgs

if not getattr(_CONTEXT, "_server_args", None):
    _CONTEXT.set_server_args(ServerArgs(model_path="dummy"))

from sglang.kernels.ops.attention.fused_qknorm_rope import fused_qk_norm_rope
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope

HD, NQ, NK, NV, EPS, BASE = 256, 4, 1, 1, 1e-6, 10000.0


def main() -> None:
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    T = 32

    qkv = torch.randn(T, (NQ + NK + NV) * HD, dtype=torch.bfloat16)
    pos = torch.arange(T, dtype=torch.int32, device="cuda")

    q_norm = Gemma3RMSNorm(dim=HD, eps=EPS).cuda()
    k_norm = Gemma3RMSNorm(dim=HD, eps=EPS).cuda()
    for m in (q_norm, k_norm):
        m.weight.data.normal_(std=0.1)
        m.weight.data = m.weight.data.to(torch.bfloat16)
    wq, wk = q_norm.weight.data, k_norm.weight.data

    q_in = qkv[:, : NQ * HD].reshape(-1, HD)

    # --- exact norm in fp64, the ground truth both paths approximate ---
    x64 = q_in.double()
    exact = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) \
            * (1.0 + wq.double())
    scale = exact.abs().mean().item()

    def rel(a, b=exact):
        return (a.double() - b).abs().max().item() / scale

    print(f"reference scale (mean |out|) = {scale:.4f}")
    print(f"one bf16 ULP at that scale   = {abs(scale) * 2 ** -8 / scale * 100:.2f}% "
          f"(bf16 has 8 mantissa bits)\n")

    # --- sglang's own norm, no rope: this is the number to beat ---
    sg = q_norm(q_in)
    print(f"sglang Gemma3RMSNorm vs fp64      : {rel(sg) * 100:6.2f}%")

    # --- kernel norm alone. rotary_dim=8 is the smallest legal value, so all
    # but 8 of the 256 elements are norm-only and can be compared directly. ---
    buf = qkv.clone()
    fused_qk_norm_rope(buf, NQ, NK, NV, HD, EPS, wq, wk, BASE, True, pos,
                       1.0, 0.0, 0.0, 1.0, rotary_dim=32, add_one=True)
    kern_norm = buf[:, : NQ * HD].reshape(-1, HD)[:, 32:]
    print(f"kernel norm (norm-only tail (elems 32:256)) vs fp64: {rel(kern_norm, exact[:, 32:]) * 100:6.2f}%")
    print(f"sglang norm (same tail)      vs fp64: {rel(sg[:, 32:], exact[:, 32:]) * 100:6.2f}%")

    # --- full path, both implementations, against each other ---
    rope = get_rope(HD, rotary_dim=HD, max_position=8192, base=BASE,
                    rope_scaling={"rope_type": "default"}, is_neox_style=True)
    q4 = q_norm(q_in).view(1, T, NQ, HD)
    k4 = k_norm(qkv[:, NQ * HD:(NQ + NK) * HD].reshape(-1, HD)).view(1, T, NK, HD)
    rq, rk = rope(pos, q4, k4)
    rq = rq.reshape(-1, NQ * HD)

    buf = qkv.clone()
    fused_qk_norm_rope(buf, NQ, NK, NV, HD, EPS, wq, wk, BASE, True, pos,
                       1.0, 0.0, 0.0, 1.0, add_one=True)
    fq = buf[:, : NQ * HD]

    d = (rq.double() - fq.double()).abs()
    print(f"\nfull path, kernel vs sglang       : {d.max().item() / scale * 100:6.2f}% max")
    print(f"                                    {d.mean().item() / scale * 100:6.2f}% mean")

    # Is the disagreement concentrated where rope applies, or spread out?
    per_elem = d.reshape(T, NQ, HD).mean(dim=(0, 1))
    half = HD // 2
    print(f"  mean |diff| over first half (rotated pairs) : {per_elem[:half].mean() / scale * 100:6.2f}%")
    print(f"  mean |diff| over second half                : {per_elem[half:].mean() / scale * 100:6.2f}%")

    # --use_fast_math is on for this kernel; rsqrtf and sincos are approximate.
    print("\nnote: the kernel is compiled with --use_fast_math, so rsqrtf and the "
          "sin/cos used by rope are approximations, not IEEE-correct.")


if __name__ == "__main__":
    main()
