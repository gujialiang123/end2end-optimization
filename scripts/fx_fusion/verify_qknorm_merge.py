#!/usr/bin/env python3
"""Validate the QK-norm merge opportunity found by FX scanning.

Finding: when q_norm and k_norm are applied to slices of one qkv tensor,
Inductor emits two kernels. Two *independent* norms fuse laterally into one, so
it is the slicing that blocks fusion, not the pair of reductions.

Both norms reduce over the same head_dim, so they can be written as a single
norm over a [tokens, heads, head_dim] view with a per-head weight. That is
exactly equivalent, which this script verifies with allclose rather than
assuming.

Hardware-agnostic apart from the profiler used to read device time; any backend
can substitute its own timer.

Note on structure: everything is deliberately at module scope and timed in a
single pass. Defining the compiled functions inside a loop body caused Dynamo to
recompile per shape and the profiler to report zero device time, so the working
shape of this measurement is kept literal.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch._dynamo as dyn
from torch.profiler import ProfilerActivity, profile

HD = 256   # head_dim
QH = 4     # query heads
KH = 1     # key/value heads


def gpu_us(fn, n: int = 200, warmup: int = 30) -> float:
    """Median-ish device time per call, excluding Python/guard overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    total = sum(
        e.self_device_time_total
        for e in p.key_averages()
        if "CUDA" in str(e.device_type)
    )
    if total <= 0:
        raise RuntimeError("profiler reported zero device time")
    return total / n


def sliced(v, a_, b_):
    """Current shape: slice q and k out of qkv, normalise each separately."""
    q = v[..., : QH * HD].reshape(-1, HD)
    k = v[..., QH * HD :].reshape(-1, HD)
    oq = q.float()
    rq = (oq * torch.rsqrt(oq.pow(2).mean(-1, keepdim=True) + 1e-6)
          * a_.float()).type_as(v)
    ok = k.float()
    rk = (ok * torch.rsqrt(ok.pow(2).mean(-1, keepdim=True) + 1e-6)
          * b_.float()).type_as(v)
    return rq, rk


def merged(v, W):
    """One norm over [tokens, heads, head_dim] with a per-head weight."""
    o = v.reshape(-1, QH + KH, HD).float()
    return (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)
            * W.float()).type_as(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[8, 32, 64, 128, 512, 2048, 4096])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    rows = []

    print(f"QK-norm merge: head_dim={HD}, q_heads={QH}, kv_heads={KH}, bf16\n")
    print(f"{'tokens':>7}{'sliced':>10}{'merged':>10}{'speedup':>9}{'equiv':>8}")

    for T in a.tokens:
        qkv = torch.randn(T, (QH + KH) * HD, dtype=torch.bfloat16).contiguous()
        wq = (torch.randn(HD) * 0.1).to(torch.bfloat16)
        wk = (torch.randn(HD) * 0.1).to(torch.bfloat16)

        dyn.reset()
        c1 = torch.compile(sliced, dynamic=False)
        r1 = c1(qkv, wq, wk)

        dyn.reset()
        wcat = torch.cat([wq.repeat(QH, 1), wk.repeat(KH, 1)], 0)
        c2 = torch.compile(merged, dynamic=False)
        r2 = c2(qkv, wcat)

        ref = torch.cat([r1[0].reshape(T, QH, HD), r1[1].reshape(T, KH, HD)], 1)
        equiv = bool(
            torch.allclose(ref, r2.reshape(T, QH + KH, HD), atol=2e-2, rtol=2e-2)
        )

        ts = [gpu_us(lambda: c1(qkv, wq, wk)) for _ in range(a.reps)]
        tm = [gpu_us(lambda: c2(qkv, wcat)) for _ in range(a.reps)]
        s, m = sorted(ts)[len(ts) // 2], sorted(tm)[len(tm) // 2]

        rows.append(dict(tokens=T, sliced_us=round(s, 3), merged_us=round(m, 3),
                         speedup=round(s / m, 3), numerically_equivalent=equiv))
        print(f"{T:>7}{s:9.2f}us{m:9.2f}us{s / m:8.2f}x{str(equiv):>8}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}")

    win = [r for r in rows if r["speedup"] > 1.05]
    print(f"\nfaster at {len(win)}/{len(rows)} token counts, "
          f"best {max(r['speedup'] for r in rows):.2f}x")
    bad = [r for r in rows if not r["numerically_equivalent"]]
    print("numerics: " + ("all equivalent" if not bad else f"MISMATCH: {bad}"))


if __name__ == "__main__":
    main()
