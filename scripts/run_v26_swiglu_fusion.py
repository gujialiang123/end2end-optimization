#!/usr/bin/env python3
"""v26: second kernel-level fusion (bandwidth-saving, persists at large batch).
Fuses the SwiGLU activation silu(gate)*up over the shared MLP intermediate [M, N=5632].

Unfused = silu kernel (read+write M*N) + mul kernel (read 2*M*N, write M*N) = 2 launches,
extra intermediate pass. Fused = 1 triton kernel reading gate+up once, writing once.
Unlike the gate-epilogue (launch-overhead bound), this saves real HBM traffic -> win
persists at large batch. Complements v25. bf16, H200.
"""
import argparse, json, os
import torch
import triton
import triton.language as tl

torch.set_default_device("cuda")
torch.manual_seed(0)
N = 5632  # Qwen1.5-MoE-A2.7B shared_expert_intermediate_size


@triton.jit
def swiglu_kernel(g_ptr, u_ptr, o_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu = g / (1.0 + tl.exp(-g))
    tl.store(o_ptr + offs, (silu * u).to(tl.bfloat16), mask=mask)


def fused_swiglu(gate, up):
    o = torch.empty_like(gate)
    n = gate.numel()
    grid = (triton.cdiv(n, 4096),)
    swiglu_kernel[grid](gate, up, o, n, BLOCK=4096, num_warps=8)
    return o


def unfused_swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up


def timed(fn, iters):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    flush = torch.empty(int(256e6 // 4), dtype=torch.int)
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    lat = sorted(a.elapsed_time(b) * 1000 for a, b in zip(s, e))
    return lat[len(lat) // 2]


ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=str, default="1,8,32,128,256,512,1024,4096")
ap.add_argument("--iters", type=int, default=300)
args = ap.parse_args()

rows = []
print(f"{'batch':>6}{'unfused_us':>12}{'fused_us':>10}{'speedup':>9}{'match':>7}", flush=True)
for b in [int(x) for x in args.batches.split(",")]:
    gate = torch.randn(b, N, dtype=torch.bfloat16)
    up = torch.randn(b, N, dtype=torch.bfloat16)
    ref = unfused_swiglu(gate, up)
    got = fused_swiglu(gate, up)
    rel = ((ref.float() - got.float()).abs() / (ref.float().abs() + 1e-3)).max().item()
    tu = timed(lambda: unfused_swiglu(gate, up), args.iters)
    tf = timed(lambda: fused_swiglu(gate, up), args.iters)
    sp = tu / tf
    rows.append({"batch": b, "unfused_us": round(tu, 2), "fused_us": round(tf, 2),
                 "speedup": round(sp, 4), "max_rel_err": round(rel, 5)})
    print(f"{b:>6}{tu:>12.2f}{tf:>10.2f}{sp:>8.3f}x{('OK' if rel<0.02 else 'BAD'):>7}", flush=True)

out = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-20_v25_kernel_fusion/swiglu_activation_fusion.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"kernel": "fused silu(gate)*up (SwiGLU activation), bandwidth-saving",
           "dims": f"N={N}", "results": rows}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
