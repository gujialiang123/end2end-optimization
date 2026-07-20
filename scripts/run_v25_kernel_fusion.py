#!/usr/bin/env python3
"""v25: KERNEL-LEVEL improvement evidence — reproduce #22325 (fuse linear+sigmoid+mul
in shared_experts) by actually WRITING a fused triton kernel and measuring its speedup
vs the unfused 3-op PyTorch baseline. bf16, H200.

Shared-expert gate epilogue (Qwen2-MoE / Qwen3.5 style):
  g = sigmoid( x @ w_gate )      # x:[M,H] w_gate:[H,1] -> [M,1]   (GEMV)
  out = g * shared_out           # broadcast scalar-per-row mul over [M,H]

Unfused = matmul kernel + sigmoid kernel + mul kernel (3 launches).
Fused   = one triton kernel doing GEMV-reduction + sigmoid + broadcast-mul.
This is a genuine kernel-code change (not config tuning) -> the "kernel story".
"""
import argparse, json, os
import torch
import triton
import triton.language as tl

torch.set_default_device("cuda")
torch.manual_seed(0)

HIDDEN = 2048  # Qwen1.5-MoE-A2.7B hidden_size


@triton.jit
def fused_gate_kernel(x_ptr, w_ptr, so_ptr, out_ptr, M, H: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    # GEMV reduction: dot(x[row,:], w[:])
    acc = 0.0
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < H
        xv = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(xv * wv, axis=0)
    g = 1.0 / (1.0 + tl.exp(-acc))  # sigmoid
    # broadcast mul: out[row,:] = g * shared_out[row,:]
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < H
        sov = tl.load(so_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * H + offs, (g * sov).to(tl.bfloat16), mask=mask)


def fused_gate(x, w, shared_out):
    M = x.shape[0]
    out = torch.empty_like(shared_out)
    fused_gate_kernel[(M,)](x, w, shared_out, out, M, H=HIDDEN, BLOCK_H=1024, num_warps=8)
    return out


def unfused_gate(x, w, shared_out):
    g = torch.sigmoid(x @ w)          # [M,1]
    return g * shared_out             # broadcast


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
ap.add_argument("--batches", type=str, default="1,8,32,64,128,256,512,1024,4096")
ap.add_argument("--iters", type=int, default=300)
args = ap.parse_args()

w = torch.randn(HIDDEN, 1, dtype=torch.bfloat16)
rows = []
print(f"{'batch':>6}{'unfused_us':>12}{'fused_us':>10}{'speedup':>9}{'match':>7}", flush=True)
for b in [int(x) for x in args.batches.split(",")]:
    x = torch.randn(b, HIDDEN, dtype=torch.bfloat16)
    so = torch.randn(b, HIDDEN, dtype=torch.bfloat16)
    # correctness
    ref = unfused_gate(x, w, so)
    got = fused_gate(x, w, so)
    max_err = (ref.float() - got.float()).abs().max().item()
    match = max_err < 0.05
    tu = timed(lambda: unfused_gate(x, w, so), args.iters)
    tf = timed(lambda: fused_gate(x, w, so), args.iters)
    sp = tu / tf
    rows.append({"batch": b, "unfused_us": round(tu, 2), "fused_us": round(tf, 2),
                 "speedup": round(sp, 4), "max_err": round(max_err, 5)})
    print(f"{b:>6}{tu:>12.2f}{tf:>10.2f}{sp:>8.3f}x{('OK' if match else 'BAD'):>7}", flush=True)

out = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-20_v25_kernel_fusion/shared_expert_gate_fusion.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"kernel": "fused sigmoid(x@w_gate)*shared_out (reproduces #22325)",
           "dims": f"hidden={HIDDEN}", "results": rows}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
