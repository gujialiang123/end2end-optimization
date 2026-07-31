#!/usr/bin/env python3
"""Does the add_one kernel variant remove the accuracy cost of folding (1 + w)?

Three arms, all against the model's own Gemma3RMSNorm + rope as reference:

  fold     existing kernel, host-side bf16 `1 + w`   -- the version that costs ~4%
  add_one  patched kernel, `+1` applied in fp32      -- the version under test
  main     what sglang runs today (4-D norms go eager, rope separate)

Reports relative error against the reference and device time for each, so the
accuracy fix and the speed claim are visible in the same table.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.runtime_context import _CONTEXT
from sglang.srt.server_args import ServerArgs

if not getattr(_CONTEXT, "_server_args", None):
    _CONTEXT.set_server_args(ServerArgs(model_path="dummy"))

from sglang.kernels.ops.attention.fused_qknorm_rope import (
    can_use_fused_qk_norm_rope,
    fused_qk_norm_rope,
)
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope

HD, NQ, NK, NV, EPS = 256, 4, 1, 1, 1e-6


def gpu_us(fn, n: int = 100, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total for e in p.key_averages()
               if "CUDA" in str(e.device_type)) / n


def ref_main(qkv, q_norm, k_norm, rope, positions):
    """sglang today: 4-D q/k miss the fused norm, rope is its own launch."""
    q, k, _ = qkv.split([NQ * HD, NK * HD, NV * HD], dim=-1)
    q = q_norm(q.unflatten(-1, (NQ, HD)).unsqueeze(0))
    k = k_norm(k.unflatten(-1, (NK, HD)).unsqueeze(0))
    q, k = rope(positions, q, k)
    return q.reshape(-1, NQ * HD), k.reshape(-1, NK * HD)


def ref_pr32670(qkv, q_norm, k_norm, rope, positions):
    """PR #32670: norms flattened so they reach gemma_rmsnorm; rope separate."""
    q, k, _ = qkv.split([NQ * HD, NK * HD, NV * HD], dim=-1)
    q = q_norm(q.reshape(-1, HD)).view(1, -1, NQ, HD)
    k = k_norm(k.reshape(-1, HD)).view(1, -1, NK, HD)
    q, k = rope(positions, q, k)
    return q.reshape(-1, NQ * HD), k.reshape(-1, NK * HD)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[1, 8, 32, 128, 512, 2048])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    for tag, flag in (("stock", False), ("add_one", True)):
        ok = can_use_fused_qk_norm_rope(HD, True, torch.bfloat16, add_one=flag)
        print(f"can_use(head_dim={HD}, neox, bf16, add_one={flag}) = {ok}")
        if not ok:
            raise SystemExit(f"{tag} kernel unavailable")

    q_norm = Gemma3RMSNorm(dim=HD, eps=EPS).cuda()
    k_norm = Gemma3RMSNorm(dim=HD, eps=EPS).cuda()
    for m in (q_norm, k_norm):
        m.weight.data.normal_(std=0.1)
        m.weight.data = m.weight.data.to(torch.bfloat16)

    wq_raw, wk_raw = q_norm.weight.data, k_norm.weight.data
    wq_fold = (1.0 + wq_raw.float()).to(torch.bfloat16)
    wk_fold = (1.0 + wk_raw.float()).to(torch.bfloat16)

    rows = []
    for base, tag in ((10000.0, "local"), (1000000.0, "global")):
        rope = get_rope(HD, rotary_dim=HD, max_position=8192, base=base,
                        rope_scaling={"rope_type": "default"}, is_neox_style=True)
        print(f"\n=== rope base {base:g} ({tag} layers) ===")
        print(f"{'tokens':>7}{'fold_err%':>11}{'addone_err%':>13}"
              f"{'main_us':>9}{'pr_us':>8}{'fused_us':>10}{'vs_pr':>8}")

        for T in a.tokens:
            qkv = torch.randn(T, (NQ + NK + NV) * HD, dtype=torch.bfloat16)
            pos = torch.arange(T, dtype=torch.int32, device="cuda")

            rq, rk = ref_pr32670(qkv.clone(), q_norm, k_norm, rope, pos)
            denom = rq.float().abs().mean().clamp_min(1e-6).item()

            def run(w_q, w_k, add_one):
                buf = qkv.clone()
                fused_qk_norm_rope(buf, NQ, NK, NV, HD, EPS, w_q, w_k, base,
                                   True, pos, 1.0, 0.0, 0.0, 1.0,
                                   add_one=add_one)
                return (buf[:, : NQ * HD],
                        buf[:, NQ * HD: (NQ + NK) * HD])

            fq_a, fk_a = run(wq_fold, wk_fold, False)
            err_fold = max((rq.float() - fq_a.float()).abs().max().item(),
                           (rk.float() - fk_a.float()).abs().max().item()) / denom

            fq_b, fk_b = run(wq_raw, wk_raw, True)
            err_add = max((rq.float() - fq_b.float()).abs().max().item(),
                          (rk.float() - fk_b.float()).abs().max().item()) / denom

            t_main = gpu_us(lambda: ref_main(qkv.clone(), q_norm, k_norm, rope, pos))
            t_pr = gpu_us(lambda: ref_pr32670(qkv.clone(), q_norm, k_norm, rope, pos))
            t_fus = gpu_us(lambda: run(wq_raw, wk_raw, True))

            print(f"{T:>7}{err_fold * 100:>10.2f}%{err_add * 100:>12.2f}%"
                  f"{t_main:>9.2f}{t_pr:>8.2f}{t_fus:>10.2f}{t_pr / t_fus:>7.2f}x")
            rows.append(dict(rope_base=base, layers=tag, tokens=T,
                             fold_err_pct=round(err_fold * 100, 4),
                             add_one_err_pct=round(err_add * 100, 4),
                             main_us=round(t_main, 3), pr32670_us=round(t_pr, 3),
                             fused_us=round(t_fus, 3),
                             speedup_vs_pr32670=round(t_pr / t_fus, 3),
                             speedup_vs_main=round(t_main / t_fus, 3)))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
