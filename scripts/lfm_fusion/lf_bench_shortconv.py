#!/usr/bin/env python3
"""Correctness-gated microbenchmark for the fused ShortConv glue kernels.

Isolates exactly the operations the audit flagged, so a kernel can be iterated
on in seconds instead of minutes. Correctness runs *before* timing and a
failing variant is never timed — the same discipline the regime-kernel study
used.

Reports isolated kernel numbers only. Whether any of it converts is a separate
end-to-end question, answered by `lf_e2e.py`.

Usage:
  python scripts/lfm_fusion/lf_bench_shortconv.py --gpu 5
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L
from lf_triton_shortconv import (fused_gate_mul, fused_gate_transpose,
                                 fused_transpose_gate)

H = 2048
CONV_LAYERS = 18


def stock_input_side(proj, H):
    B_gate, C_gate, x = proj.chunk(3, dim=-1)
    Bx = B_gate * x
    return Bx.transpose(0, 1).contiguous()


def stock_output_side(conv_out, proj, H):
    _, C_gate, _ = proj.chunk(3, dim=-1)
    return C_gate * conv_out.transpose(0, 1)


def timeit(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True),
           torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in ev)
    return st.median(ts)


def gb_per_s(bytes_moved, ms):
    return bytes_moved / (ms * 1e-3) / 1024 ** 3


def check(name, got, ref, tol=0.0):
    if got.shape != ref.shape:
        return False, f"shape {got.shape} != {ref.shape}"
    if torch.isnan(got).any() or torch.isinf(got).any():
        return False, "NaN/Inf"
    d = (got.float() - ref.float()).abs()
    mx = d.max().item()
    ok = mx <= tol
    return ok, f"max|diff|={mx:.3e} (tol {tol})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--tokens", default="1,8,32,128,1024,4096,16000")
    ap.add_argument("--out", default="shortconv_bench.json",
                    help="output name; override when running a partial token "
                         "sweep so the full curve on disk is not clobbered")
    a = ap.parse_args()

    torch.cuda.set_device(a.gpu)
    dev, dt = f"cuda:{a.gpu}", torch.bfloat16
    torch.manual_seed(0)

    rows = []
    for T in [int(x) for x in a.tokens.split(",")]:
        proj = (torch.randn(T, 3 * H, device=dev) * 0.5).to(dt)
        conv_out_ht = (torch.randn(H, T, device=dev) * 0.5).to(dt)
        tok_bytes = T * H * 2

        # ---------------- input side ----------------
        ref_in = stock_input_side(proj, H)
        got_in = fused_gate_transpose(proj, H)
        ok_in, msg_in = check("gate+transpose", got_in, ref_in)

        # ---------------- output side ----------------
        ref_out = stock_output_side(conv_out_ht, proj, H)
        got_out = fused_transpose_gate(conv_out_ht, proj, H)
        ok_out, msg_out = check("transpose+gate", got_out, ref_out)

        # ------------- decode-shaped gating only -------------
        B_gate, _, x = proj.chunk(3, dim=-1)
        ref_mul = B_gate * x
        got_mul = fused_gate_mul(proj, H)
        ok_mul, msg_mul = check("gate mul", got_mul, ref_mul)

        print(f"\n=== T={T} ===")
        print(f"  correctness  in:{ok_in} {msg_in} | out:{ok_out} {msg_out} "
              f"| mul:{ok_mul} {msg_mul}")
        if not (ok_in and ok_out and ok_mul):
            print("  CORRECTNESS FAILED -> not timing this shape")
            rows.append(dict(T=T, status="correctness_failed",
                             detail=dict(inp=msg_in, out=msg_out, mul=msg_mul)))
            continue

        t_in_stock = timeit(lambda: stock_input_side(proj, H))
        t_in_fused = timeit(lambda: fused_gate_transpose(proj, H))
        t_out_stock = timeit(lambda: stock_output_side(conv_out_ht, proj, H))
        t_out_fused = timeit(lambda: fused_transpose_gate(conv_out_ht, proj, H))

        # traffic: stock input side = read 2 slices + write Bx, then read Bx
        # transposed + write Bx_t;  fused = read 2 slices + write Bx_t
        b_in_stock, b_in_fused = 5 * tok_bytes, 3 * tok_bytes
        b_out_stock, b_out_fused = 3 * tok_bytes, 3 * tok_bytes

        row = dict(
            T=T, status="ok",
            in_stock_ms=round(t_in_stock, 5), in_fused_ms=round(t_in_fused, 5),
            in_speedup=round(t_in_stock / t_in_fused, 4),
            in_stock_gbs=round(gb_per_s(b_in_stock, t_in_stock), 1),
            in_fused_gbs=round(gb_per_s(b_in_fused, t_in_fused), 1),
            out_stock_ms=round(t_out_stock, 5), out_fused_ms=round(t_out_fused, 5),
            out_speedup=round(t_out_stock / t_out_fused, 4),
            out_stock_gbs=round(gb_per_s(b_out_stock, t_out_stock), 1),
            out_fused_gbs=round(gb_per_s(b_out_fused, t_out_fused), 1),
            total_saved_ms_per_layer=round(
                (t_in_stock - t_in_fused) + (t_out_stock - t_out_fused), 5),
        )
        row["total_saved_ms_per_forward"] = round(
            row["total_saved_ms_per_layer"] * CONV_LAYERS, 4)
        rows.append(row)
        print(f"  input  side  {t_in_stock*1000:8.1f}us -> {t_in_fused*1000:8.1f}us"
              f"  {row['in_speedup']:.2f}x   "
              f"({row['in_stock_gbs']:.0f} -> {row['in_fused_gbs']:.0f} GB/s)")
        print(f"  output side  {t_out_stock*1000:8.1f}us -> {t_out_fused*1000:8.1f}us"
              f"  {row['out_speedup']:.2f}x   "
              f"({row['out_stock_gbs']:.0f} -> {row['out_fused_gbs']:.0f} GB/s)")
        print(f"  saved/forward ({CONV_LAYERS} conv layers): "
              f"{row['total_saved_ms_per_forward']:.3f} ms")

    outdir = L.RESULTS / "microbench"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / a.out).write_text(
        json.dumps(dict(rows=rows, hidden_size=H, conv_layers=CONV_LAYERS,
                        environment=L.environment()), indent=2))
    print(f"\nwrote {outdir/a.out}")


if __name__ == "__main__":
    main()
