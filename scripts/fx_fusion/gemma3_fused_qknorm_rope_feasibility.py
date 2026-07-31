#!/usr/bin/env python3
"""Can Gemma-3 use the fused QK-norm + RoPE kernel sglang already ships?

Gemma-3 calls `q_norm`, `k_norm` and `rotary_emb` as three separate ops. sglang
ships `fused_qk_norm_rope`, which does all three in one launch and is already
used by qwen3_moe, deepseek_v4, mellum and interns1pro. Nothing in the Gemma-3
model file names it.

Two things have to hold before that is worth pursuing.

1. The kernel applies `w`; Gemma applies `1 + w`. The kernel accumulates in
   fp32 and casts once at the end, which is Gemma's cast order exactly, so the
   only question is whether folding the `+1` into a bf16 weight costs accuracy.
   Near 1.0 a bf16 ULP is 2^-7, so this is not obviously safe and is measured
   here rather than argued.
2. RoPE has to match: Gemma-3 is NeoX style, rotates the full 256-dim head, and
   uses a different base on sliding vs global layers. `base` is a kernel
   argument, so per-layer theta is fine, but it is checked at both values.

Reports max abs error against the model's own path, plus the device time of
each, and is deliberately silent about whether that error is acceptable -- that
is an end-to-end question, not one a microbenchmark can answer.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.runtime_context import _CONTEXT
from sglang.srt.server_args import ServerArgs

# `get_rope` reads the global server args, which only exist inside a running
# server. Install a default set so the layer can be built standalone.
if not getattr(_CONTEXT, "_server_args", None):
    _CONTEXT.set_server_args(ServerArgs(model_path="dummy"))

from sglang.kernels.ops.attention.fused_qknorm_rope import (
    can_use_fused_qk_norm_rope,
    fused_qk_norm_rope,
)
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope

HD = 256   # head_dim
NQ = 4     # query heads
NK = 1     # key heads
NV = 1     # value heads
EPS = 1e-6


def gpu_us(fn, n: int = 100, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    total = sum(e.self_device_time_total for e in p.key_averages()
                if "CUDA" in str(e.device_type))
    return total / n


def reference(qkv, q_norm, k_norm, rope, positions):
    """Exactly what Gemma3Attention.forward_cuda does today on main.

    The 4-D q/k means both norms miss the fused kernel and run as eager math,
    so this arm carries the dispatch gap of PR #32670 as well as the unfused
    rope. Measuring against it would credit this change with a fix that is
    already in flight.
    """
    q, k, v = qkv.split([NQ * HD, NK * HD, NV * HD], dim=-1)
    q = q.unflatten(-1, (NQ, HD)).unsqueeze(0)
    q = q_norm(q)
    k = k.unflatten(-1, (NK, HD)).unsqueeze(0)
    k = k_norm(k)
    q, k = rope(positions, q, k)
    return q.reshape(-1, NQ * HD), k.reshape(-1, NK * HD)


def reference_flattened(qkv, q_norm, k_norm, rope, positions):
    """The PR #32670 baseline: norms flattened to 2-D so they reach
    `gemma_rmsnorm`, rope still a separate launch. This is what the fused
    kernel has to beat to be worth anything."""
    q, k, v = qkv.split([NQ * HD, NK * HD, NV * HD], dim=-1)
    q = q_norm(q.reshape(-1, HD)).view(1, -1, NQ, HD)
    k = k_norm(k.reshape(-1, HD)).view(1, -1, NK, HD)
    q, k = rope(positions, q, k)
    return q.reshape(-1, NQ * HD), k.reshape(-1, NK * HD)


def norm_with_folded_weight(x, w_folded):
    """Gemma's norm, but multiplying by a pre-folded bf16 `1 + w`.

    Isolates the cost of the fold from the cost of the kernel: comparing the
    fused kernel against *this* is a kernel-vs-eager difference, and comparing
    this against the real Gemma norm is the fold on its own.
    """
    o = x.float()
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
    return (o * w_folded.float()).type_as(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[1, 8, 32, 128, 512, 2048])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    ok = can_use_fused_qk_norm_rope(HD, is_neox=True, dtype=torch.bfloat16)
    print(f"can_use_fused_qk_norm_rope(head_dim={HD}, neox, bf16) = {ok}")
    if not ok:
        raise SystemExit("kernel unavailable; nothing further to measure")

    q_norm = Gemma3RMSNorm(dim=HD, eps=EPS).cuda()
    k_norm = Gemma3RMSNorm(dim=HD, eps=EPS).cuda()
    for m in (q_norm, k_norm):
        m.weight.data.normal_(std=0.1)
        m.weight.data = m.weight.data.to(torch.bfloat16)

    # Gemma applies (1 + w). Fold the +1 into the weight the kernel multiplies
    # by, which is where the bf16 rounding this script is testing comes in.
    wq = (1.0 + q_norm.weight.data.float()).to(torch.bfloat16)
    wk = (1.0 + k_norm.weight.data.float()).to(torch.bfloat16)

    rows = []
    # Gemma-3 alternates a local and a global rope base every 6 layers.
    for base, tag in ((10000.0, "local"), (1000000.0, "global")):
        rope = get_rope(HD, rotary_dim=HD, max_position=8192, base=base,
                        rope_scaling={"rope_type": "default"}, is_neox_style=True)
        print(f"\n=== rope base {base:g} ({tag} layers) ===")
        print(f"{'tokens':>7}{'main_us':>9}{'pr32670_us':>12}{'fused_us':>10}"
              f"{'vs_main':>9}{'vs_pr':>8}{'fold_err':>10}{'kern_err':>10}")

        for T in a.tokens:
            qkv = torch.randn(T, (NQ + NK + NV) * HD, dtype=torch.bfloat16)
            positions = torch.arange(T, dtype=torch.int32, device="cuda")

            rq, rk = reference(qkv.clone(), q_norm, k_norm, rope, positions)

            fused = qkv.clone()
            fused_qk_norm_rope(fused, NQ, NK, NV, HD, EPS, wq, wk,
                               base, True, positions, 1.0, 0.0, 0.0, 1.0)
            fq = fused[:, : NQ * HD]
            fk = fused[:, NQ * HD: (NQ + NK) * HD]

            # Error decomposition, before rope so the rope kernel's own
            # differences do not get attributed to the weight fold.
            q_raw = qkv[:, : NQ * HD].reshape(-1, HD)
            exact_norm = q_norm(q_raw)
            folded_norm = norm_with_folded_weight(q_raw, wq)
            denom = exact_norm.float().abs().mean().clamp_min(1e-6)
            fold_err = ((exact_norm.float() - folded_norm.float()).abs().max()
                        / denom).item()
            total_err = max((rq.float() - fq.float()).abs().max().item(),
                            (rk.float() - fk.float()).abs().max().item())
            rel_err = total_err / denom.item()

            t_main = gpu_us(lambda: reference(
                qkv.clone(), q_norm, k_norm, rope, positions))
            t_pr = gpu_us(lambda: reference_flattened(
                qkv.clone(), q_norm, k_norm, rope, positions))
            t_fus = gpu_us(lambda: fused_qk_norm_rope(
                qkv.clone(), NQ, NK, NV, HD, EPS, wq, wk,
                base, True, positions, 1.0, 0.0, 0.0, 1.0))

            print(f"{T:>7}{t_main:>9.2f}{t_pr:>12.2f}{t_fus:>10.2f}"
                  f"{t_main / t_fus:>8.2f}x{t_pr / t_fus:>7.2f}x"
                  f"{fold_err * 100:>9.2f}%{rel_err * 100:>9.2f}%")
            rows.append(dict(rope_base=base, layers=tag, tokens=T,
                             main_us=round(t_main, 3), pr32670_us=round(t_pr, 3),
                             fused_us=round(t_fus, 3),
                             speedup_vs_main=round(t_main / t_fus, 3),
                             speedup_vs_pr32670=round(t_pr / t_fus, 3),
                             fold_rel_err_pct=round(fold_err * 100, 4),
                             total_rel_err_pct=round(rel_err * 100, 4)))

    # How much of the error is the bf16 weight fold, and how much is the kernel?
    exact = (1.0 + q_norm.weight.data.float())
    folded = wq.float()
    print(f"\nbf16 fold of (1+w): max weight error {(exact - folded).abs().max():.6f} "
          f"({(exact - folded).abs().max() / exact.abs().mean() * 100:.3f}% of mean |1+w|)")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
