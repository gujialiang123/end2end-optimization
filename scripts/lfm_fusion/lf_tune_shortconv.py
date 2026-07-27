#!/usr/bin/env python3
"""Tile-size sweep for the fused ShortConv kernels.

The default 64x64 tile wins big at T=16000 but loses below T~1536, which
matters because chunked prefill commonly runs at 2048 tokens. This sweeps the
tile and warp count per token count so the shape guard can be set from measured
data rather than from one hand-picked configuration.

Correctness is checked for every configuration before it is timed.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L
import lf_triton_shortconv as K

H = 2048
CONFIGS = [(bt, bh, w, s)
           for bt, bh in [(32, 32), (32, 64), (64, 32), (64, 64),
                          (128, 32), (128, 64), (64, 128), (128, 128)]
           for w in (4, 8)
           for s in (2, 3)]


def timeit(fn, iters=40, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True),
           torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return st.median(sorted(s.elapsed_time(e) for s, e in ev))


def run_in(proj, T, bt, bh, w, s):
    out = torch.empty((H, T), device=proj.device, dtype=proj.dtype)
    grid = (triton.cdiv(T, bt), triton.cdiv(H, bh))
    K._fused_gate_transpose_kernel[grid](
        proj, out, T, H, proj.stride(0), proj.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=bt, BLOCK_H=bh, num_warps=w, num_stages=s)
    return out


def run_out(conv, proj, T, bt, bh, w, s):
    out = torch.empty((T, H), device=conv.device, dtype=conv.dtype)
    grid = (triton.cdiv(T, bt), triton.cdiv(H, bh))
    K._fused_transpose_gate_kernel[grid](
        conv, proj, out, T, H, conv.stride(0), conv.stride(1),
        proj.stride(0), proj.stride(1), out.stride(0), out.stride(1),
        BLOCK_T=bt, BLOCK_H=bh, num_warps=w, num_stages=s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tokens", default="512,1024,2048,4096,16000")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev, dt = f"cuda:{a.gpu}", torch.bfloat16
    torch.manual_seed(0)

    best = {}
    for T in [int(x) for x in a.tokens.split(",")]:
        proj = (torch.randn(T, 3 * H, device=dev) * 0.5).to(dt)
        conv = (torch.randn(H, T, device=dev) * 0.5).to(dt)
        Bg, Cg, x = proj.chunk(3, dim=-1)
        ref_in = (Bg * x).transpose(0, 1).contiguous()
        ref_out = Cg * conv.transpose(0, 1)
        t_stock_in = timeit(lambda: (Bg * x).transpose(0, 1).contiguous())
        t_stock_out = timeit(lambda: Cg * conv.transpose(0, 1))

        rin, rout = [], []
        for bt, bh, w, s in CONFIGS:
            try:
                g = run_in(proj, T, bt, bh, w, s)
                if not torch.equal(g, ref_in):
                    continue
                rin.append((timeit(lambda: run_in(proj, T, bt, bh, w, s)),
                            (bt, bh, w, s)))
            except Exception:
                pass
            try:
                g = run_out(conv, proj, T, bt, bh, w, s)
                if not torch.equal(g, ref_out):
                    continue
                rout.append((timeit(lambda: run_out(conv, proj, T, bt, bh, w, s)),
                             (bt, bh, w, s)))
            except Exception:
                pass
        if not rin or not rout:
            continue
        bi, bo = min(rin), min(rout)
        best[T] = dict(
            stock_in_us=round(t_stock_in * 1000, 1),
            best_in_us=round(bi[0] * 1000, 1), in_cfg=bi[1],
            in_speedup=round(t_stock_in / bi[0], 3),
            stock_out_us=round(t_stock_out * 1000, 1),
            best_out_us=round(bo[0] * 1000, 1), out_cfg=bo[1],
            out_speedup=round(t_stock_out / bo[0], 3),
            n_valid=len(rin))
        print(f"T={T:6d}  in {t_stock_in*1000:7.1f}->{bi[0]*1000:7.1f}us "
              f"{t_stock_in/bi[0]:5.2f}x cfg={bi[1]}   |   "
              f"out {t_stock_out*1000:7.1f}->{bo[0]*1000:7.1f}us "
              f"{t_stock_out/bo[0]:5.2f}x cfg={bo[1]}")

    outdir = L.RESULTS / "microbench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "shortconv_tile_sweep.json").write_text(
        json.dumps(dict(best=best, configs_tried=len(CONFIGS),
                        environment=L.environment()), indent=2, default=str))
    print(f"\nwrote {outdir/'shortconv_tile_sweep.json'}")


if __name__ == "__main__":
    main()
