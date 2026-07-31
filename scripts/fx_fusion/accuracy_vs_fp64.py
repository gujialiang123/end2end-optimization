#!/usr/bin/env python3
"""Is the fused kernel as accurate as the path it replaces?

The first pass of this comparison reported ~4% and treated it as a defect. It
was measuring max relative error against a bf16 reference, which is dominated by
whichever single element happens to sit closest to zero. Against an fp64
reference the kernel and sglang's own Gemma3RMSNorm both land at 1.86%, i.e.
they are equally accurate and both are simply quantised to bf16.

So the question is not "is the kernel exact" -- nothing in bf16 is -- but "is it
at least as close to fp64 as what it replaces". That is what this measures:

  sglang_err   Gemma3RMSNorm + rope, vs an fp64 reference
  kernel_err   fused kernel,          vs the same fp64 reference
  ulp          the bf16 quantum at the output scale, as the floor

A kernel_err at or below sglang_err means the fusion costs nothing in accuracy.
Both `add_one` (the `+1` applied in fp32 inside the kernel) and the host-side
bf16 fold are measured, since only one of them needs to survive.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from sglang.srt.runtime_context import _CONTEXT
from sglang.srt.server_args import ServerArgs

if not getattr(_CONTEXT, "_server_args", None):
    _CONTEXT.set_server_args(ServerArgs(model_path="dummy"))

from sglang.kernels.ops.attention.fused_qknorm_rope import fused_qk_norm_rope
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope

HD, NQ, NK, NV, EPS = 256, 4, 1, 1, 1e-6


def rope_fp64(x, positions, base, hd):
    """NeoX rope in fp64: rotate (i, i + hd/2) pairs."""
    half = hd // 2
    inv = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float64,
                                       device=x.device) * 2.0 / hd))
    ang = positions.double()[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    q_norm = Gemma3RMSNorm(dim=HD, eps=EPS).cuda()
    k_norm = Gemma3RMSNorm(dim=HD, eps=EPS).cuda()
    for m in (q_norm, k_norm):
        m.weight.data.normal_(std=0.1)
        m.weight.data = m.weight.data.to(torch.bfloat16)
    wq, wk = q_norm.weight.data, k_norm.weight.data
    wq_fold = (1.0 + wq.float()).to(torch.bfloat16)
    wk_fold = (1.0 + wk.float()).to(torch.bfloat16)

    rows = []
    for base, tag in ((10000.0, "local"), (1000000.0, "global")):
        rope = get_rope(HD, rotary_dim=HD, max_position=8192, base=base,
                        rope_scaling={"rope_type": "default"}, is_neox_style=True)
        print(f"\n=== rope base {base:g} ({tag} layers), error vs an fp64 reference ===")
        print(f"{'tokens':>7}{'bf16_ulp':>10}{'sglang':>9}{'add_one':>9}{'fold':>8}"
              f"{'verdict':>26}")

        for T in a.tokens:
            qkv = torch.randn(T, (NQ + NK + NV) * HD, dtype=torch.bfloat16)
            pos = torch.arange(T, dtype=torch.int32, device="cuda")
            q_in = qkv[:, : NQ * HD].reshape(-1, HD)

            x64 = q_in.double()
            n64 = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) \
                  * (1.0 + wq.double())
            exact = rope_fp64(n64.view(T, NQ, HD), pos, base, HD).reshape(T, NQ * HD)
            scale = exact.abs().mean().item()
            ulp_pct = 2 ** -8 * 100

            def err(v):
                return (v.double().reshape(T, NQ * HD) - exact).abs().mean().item() \
                       / scale * 100

            q4 = q_norm(q_in).view(1, T, NQ, HD)
            k4 = k_norm(qkv[:, NQ * HD:(NQ + NK) * HD].reshape(-1, HD)).view(1, T, NK, HD)
            sq, _ = rope(pos, q4, k4)
            e_sglang = err(sq)

            def kern(w_q, w_k, add_one):
                buf = qkv.clone()
                fused_qk_norm_rope(buf, NQ, NK, NV, HD, EPS, w_q, w_k, base, True,
                                   pos, 1.0, 0.0, 0.0, 1.0, add_one=add_one)
                return buf[:, : NQ * HD]

            e_add = err(kern(wq, wk, True))
            e_fold = err(kern(wq_fold, wk_fold, False))

            best = min(e_add, e_fold)
            verdict = ("no accuracy cost" if best <= e_sglang * 1.05
                       else f"{best / max(e_sglang, 1e-9):.2f}x sglang error")
            print(f"{T:>7}{ulp_pct:>9.2f}%{e_sglang:>8.2f}%{e_add:>8.2f}%"
                  f"{e_fold:>7.2f}%{verdict:>26}")
            rows.append(dict(rope_base=base, layers=tag, tokens=T,
                             bf16_ulp_pct=round(ulp_pct, 4),
                             sglang_err_pct=round(e_sglang, 4),
                             add_one_err_pct=round(e_add, 4),
                             fold_err_pct=round(e_fold, 4),
                             no_accuracy_cost=bool(best <= e_sglang * 1.05)))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
