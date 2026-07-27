#!/usr/bin/env python3
"""Correctness-gated microbenchmark for MoE sum + residual RMSNorm."""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L
from lf_triton_moesum import TOP_K, fused_moesum_add_rmsnorm

H = 2048


def stock_sequence(partials, residual, weight, eps):
    from sgl_kernel import fused_add_rmsnorm, moe_sum_reduce
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        moe_sum_reduce_torch_compile,
    )

    output = torch.empty_like(residual)
    if partials.shape[0] <= 32:
        moe_sum_reduce_torch_compile(partials, output, 1.0)
    else:
        moe_sum_reduce(partials, output, 1.0)
    fused_add_rmsnorm(output, residual, weight, eps)
    return output, residual


def timeit(fn, reset, iters=50, warmup=20):
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        reset()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return st.median(sorted(start.elapsed_time(end) for start, end in events))


def gb_per_s(bytes_moved, ms):
    return bytes_moved / (ms * 1e-3) / 1024**3


def tensor_diff(got, ref):
    diff = (got.float() - ref.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "different": int(torch.count_nonzero(got != ref).item()),
        "elements": got.numel(),
        "bit_exact": bool(torch.equal(got, ref)),
    }


def logical_device(requested_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return 0 if visible else requested_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--tokens", default="1,8,32,128,1024,4096,16000")
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--output", default="microbench.json")
    a = ap.parse_args()

    device_index = logical_device(a.gpu)
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    torch.manual_seed(20260727)
    torch.cuda.manual_seed_all(20260727)

    rows = []
    for T in [int(x) for x in a.tokens.split(",")]:
        partials = torch.randn(
            (T, TOP_K, H), device=device, dtype=torch.bfloat16
        )
        residual_base = torch.randn((T, H), device=device, dtype=torch.bfloat16)
        weight = (
            1.0
            + 0.1
            * torch.randn((H,), device=device, dtype=torch.bfloat16)
        ).contiguous()

        residual_stock = residual_base.clone()
        residual_fused = residual_base.clone()
        ref, ref_residual = stock_sequence(
            partials, residual_stock, weight, a.eps
        )
        got, got_residual = fused_moesum_add_rmsnorm(
            partials, residual_fused, weight, a.eps
        )
        torch.cuda.synchronize()

        output_diff = tensor_diff(got, ref)
        residual_diff = tensor_diff(got_residual, ref_residual)
        passed = (
            residual_diff["max_abs"] == 0.0
            and output_diff["max_abs"] <= 0.03125
            and not torch.isnan(got).any().item()
            and not torch.isinf(got).any().item()
        )
        print(f"\n=== T={T} ===")
        print(
            "  correctness "
            f"output max={output_diff['max_abs']:.6f} "
            f"mean={output_diff['mean_abs']:.3e} "
            f"bit_exact={output_diff['bit_exact']} | "
            f"residual max={residual_diff['max_abs']:.6f} "
            f"bit_exact={residual_diff['bit_exact']} -> "
            f"{'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            rows.append(
                {
                    "T": T,
                    "status": "correctness_failed",
                    "output_diff": output_diff,
                    "residual_diff": residual_diff,
                }
            )
            print("  CORRECTNESS FAILED -> not timing this shape")
            continue

        timed_stock_residual = residual_base.clone()
        timed_fused_residual = residual_base.clone()
        iters = 30 if T >= 4096 else 50
        stock_ms = timeit(
            lambda: stock_sequence(
                partials, timed_stock_residual, weight, a.eps
            ),
            lambda: timed_stock_residual.copy_(residual_base),
            iters=iters,
        )
        fused_ms = timeit(
            lambda: fused_moesum_add_rmsnorm(
                partials, timed_fused_residual, weight, a.eps
            ),
            lambda: timed_fused_residual.copy_(residual_base),
            iters=iters,
        )

        activation_bytes = T * H * 2
        weight_bytes = H * 2
        stock_bytes = (TOP_K + 5) * activation_bytes + weight_bytes
        fused_bytes = (TOP_K + 3) * activation_bytes + weight_bytes
        row = {
            "T": T,
            "status": "ok",
            "stock_us": round(stock_ms * 1000, 3),
            "fused_us": round(fused_ms * 1000, 3),
            "speedup": round(stock_ms / fused_ms, 4),
            "stock_gbs": round(gb_per_s(stock_bytes, stock_ms), 1),
            "fused_gbs": round(gb_per_s(fused_bytes, fused_ms), 1),
            "output_diff": output_diff,
            "residual_diff": residual_diff,
        }
        rows.append(row)
        print(
            f"  time {row['stock_us']:8.2f} us -> {row['fused_us']:8.2f} us  "
            f"{row['speedup']:.3f}x  "
            f"({row['stock_gbs']:.1f} -> {row['fused_gbs']:.1f} GB/s)"
        )

        del partials, residual_base, weight
        del residual_stock, residual_fused, timed_stock_residual, timed_fused_residual
        del ref, ref_residual, got, got_residual
        torch.cuda.empty_cache()

    outdir = L.RESULTS / "moesum"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / a.output
    out.write_text(
        json.dumps(
            {
                "rows": rows,
                "hidden_size": H,
                "top_k": TOP_K,
                "eps": a.eps,
                "physical_gpu": a.gpu,
                "environment": L.environment(),
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
