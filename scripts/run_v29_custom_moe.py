#!/usr/bin/env python3
"""v29: custom small-M decode MoE kernel attempt (bf16). Target: beat sglang's fused_moe
at M=1 (which is only 49% HBM / 24us GEMM + 6us overhead).

Design (specialized for small M, top-k experts):
  Kernel A: for each (selected expert, n_tile) compute SwiGLU act = silu(x@w1_gate)* (x@w1_up)
            -> writes act[n_sel, moe_int]   (parallel over experts x n-tiles)
  Kernel B: for each (selected expert, n_tile) compute y2 = act @ w2.T, scale by topk_weight,
            atomic-add into out[M, hidden]   (parallel over experts x n-tiles)
Fuses activation into A, fuses weighted-sum into B -> removes act_and_mul + moe_sum overhead,
and uses expert x n-tile parallelism to better saturate HBM at tiny M.
"""
import json, os, sys, argparse
import torch, triton
import triton.language as tl

BENCH = "/home/t-jialianggu/work/sglang/benchmark/kernels/fused_moe_triton"
sys.path.insert(0, BENCH)
from common_utils import get_model_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
set_global_server_args_for_scheduler(ServerArgs(model_path="/data/hf/models/Qwen3-30B-A3B-Instruct-2507", tp_size=1))
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKConfig, select_experts

torch.set_default_device("cuda"); torch.manual_seed(0)
MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
HBM = 4.8e12


@triton.jit
def moe_w1_act_kernel(x_ptr, w1_ptr, ids_ptr, act_ptr, M, H: tl.constexpr, I: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr):
    # grid: (n_sel_pairs, I//BN). one program: one (token-expert pair p, n-tile) -> BN act outputs
    p = tl.program_id(0)        # index into selected (token,expert) pairs
    nt = tl.program_id(1)
    tok = p // 999999           # placeholder; for M=1 tok=0
    e = tl.load(ids_ptr + p)
    n0 = nt * BN
    n = n0 + tl.arange(0, BN)   # output cols in [0, I)
    acc_g = tl.zeros((BN,), dtype=tl.float32)
    acc_u = tl.zeros((BN,), dtype=tl.float32)
    tok = p  # M=1 assumption: one pair per expert, token 0
    for k0 in range(0, H, BK):
        koff = k0 + tl.arange(0, BK)
        xv = tl.load(x_ptr + koff).to(tl.float32)                      # [BK]
        wg = tl.load(w1_ptr + e*(2*I*H) + n[:, None]*H + koff[None, :]).to(tl.float32)  # gate rows [BN,BK]
        wu = tl.load(w1_ptr + e*(2*I*H) + (I + n[:, None])*H + koff[None, :]).to(tl.float32)  # up rows
        acc_g += tl.sum(wg * xv[None, :], axis=1)
        acc_u += tl.sum(wu * xv[None, :], axis=1)
    silu = acc_g / (1.0 + tl.exp(-acc_g))
    out = silu * acc_u
    tl.store(act_ptr + p*I + n, out.to(tl.bfloat16))


@triton.jit
def moe_w2_sum_kernel(act_ptr, w2_ptr, ids_ptr, tw_ptr, out_ptr, M, H: tl.constexpr, I: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr):
    # grid: (n_sel_pairs, H//BN). y2 = act[p] @ w2[e].T ; out += tw[p]*y2  (atomic)
    p = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(ids_ptr + p)
    tw = tl.load(tw_ptr + p).to(tl.float32)
    n0 = nt * BN
    n = n0 + tl.arange(0, BN)          # output cols in [0, H)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, I, BK):
        koff = k0 + tl.arange(0, BK)
        av = tl.load(act_ptr + p*I + koff).to(tl.float32)                     # [BK]
        w = tl.load(w2_ptr + e*(H*I) + n[:, None]*I + koff[None, :]).to(tl.float32)  # [BN,BK]
        acc += tl.sum(w * av[None, :], axis=1)
    tl.atomic_add(out_ptr + n, acc * tw)


def custom_moe_m1(x, w1, w2, topk_ids, topk_weights, E, H, I):
    # x:[1,H], selected experts flattened (M=1 -> topk pairs)
    ids = topk_ids.view(-1).to(torch.int32)
    tw = topk_weights.view(-1).float()
    P = ids.numel()
    act = torch.empty(P, I, dtype=torch.bfloat16)
    out = torch.zeros(H, dtype=torch.float32)
    BN1, BK1 = 64, 256
    moe_w1_act_kernel[(P, I // BN1)](x, w1, ids, act, 1, H, I, BN1, BK1, num_warps=8)
    BN2, BK2 = 128, 128
    moe_w2_sum_kernel[(P, H // BN2)](act, w2, ids, tw, out, 1, H, I, BN2, BK2, num_warps=8)
    return out.to(torch.bfloat16).view(1, H)


def timed(fn, iters=300):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    flush = torch.empty(int(256e6 // 4), dtype=torch.int)
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_(); s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    lat = sorted(a.elapsed_time(b) * 1000 for a, b in zip(s, e))
    return lat[len(lat) // 2]


def timed_graph(fn, iters=300):
    # capture 10 invocations into a CUDA graph (like sglang benchmark_config), time replays
    for _ in range(10): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): fn()
    torch.cuda.synchronize()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    flush = torch.empty(int(256e6 // 4), dtype=torch.int)
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_(); s[i].record(); g.replay(); e[i].record()
    torch.cuda.synchronize()
    lat = sorted(a.elapsed_time(b) * 1000 for a, b in zip(s, e))
    return lat[len(lat) // 2] / 10  # per invocation


mc = get_model_config(MODEL, 1)
E, topk, H, shard = mc["num_experts"], mc["topk"], mc["hidden_size"], mc["shard_intermediate_size"]
I = shard // 2  # moe_intermediate = 768
M = 1
SCALE = (1.0 / H) ** 0.5  # realistic init magnitude so bf16 correctness metric is meaningful
x = torch.randn(M, H, dtype=torch.bfloat16)
w1 = (torch.randn(E, shard, H) * SCALE).to(torch.bfloat16)     # [E, 2*I, H]
w2 = (torch.randn(E, H, I) * SCALE).to(torch.bfloat16)          # [E, H, I]
gating = torch.randn(M, E, dtype=torch.float32)
topk_output = select_experts(x, gating, TopKConfig(top_k=topk, renormalize=True))
tw, tid = topk_output.topk_weights, topk_output.topk_ids

# reference (torch)
def ref():
    out = torch.zeros(M, H, dtype=torch.float32)
    for j in range(topk):
        e = tid[0, j].item(); wgt = tw[0, j].item()
        y1 = (x.float() @ w1[e].float().T)          # [1, 2I]
        act = torch.nn.functional.silu(y1[:, :I]) * y1[:, I:]
        y2 = act @ w2[e].float().T                   # [1, H]
        out += wgt * y2
    return out.to(torch.bfloat16)

r = ref()
g_out = custom_moe_m1(x, w1, w2, tid, tw, E, H, I)
rel = ((r.float() - g_out.float()).abs() / (r.float().abs() + 1e-2)).max().item()
print(f"correctness max_rel_err = {rel:.4f}  ({'OK' if rel < 0.05 else 'BAD'})", flush=True)

# preallocated custom output for graph-safe timing
_out = torch.zeros(H, dtype=torch.float32)
_act = torch.empty(tid.numel(), I, dtype=torch.bfloat16)
_ids = tid.view(-1).to(torch.int32)
_tw = tw.view(-1).float()
P = _ids.numel()
def custom_graphsafe():
    _out.zero_()
    moe_w1_act_kernel[(P, I // 64)](x, w1, _ids, _act, 1, H, I, 64, 256, num_warps=8)
    moe_w2_sum_kernel[(P, H // 128)](_act, w2, _ids, _tw, _out, 1, H, I, 128, 128, num_warps=8)
    return _out

cfg = MoeRunnerConfig(inplace=False)
def sglang_moe(): return fused_moe(x, w1, w2, topk_output, moe_runner_config=cfg)

# fair comparison: BOTH under CUDA graph (real serving uses cudagraph)
t_sgl = timed_graph(sglang_moe)
t_cust = timed_graph(custom_graphsafe)
# also non-graphed for reference
t_sgl_ng = timed(sglang_moe)
t_cust_ng = timed(custom_graphsafe)
print(f"[cudagraph]  sglang: {t_sgl:.2f}us | custom: {t_cust:.2f}us | speedup: {t_sgl/t_cust:.3f}x", flush=True)
print(f"[non-graph]  sglang: {t_sgl_ng:.2f}us | custom: {t_cust_ng:.2f}us | speedup: {t_sgl_ng/t_cust_ng:.3f}x", flush=True)

out = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-20_v27_moe_baseline/custom_m1_moe.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"M": M, "max_rel_err": rel,
           "cudagraph": {"sglang_us": round(t_sgl, 2), "custom_us": round(t_cust, 2), "speedup": round(t_sgl / t_cust, 4)},
           "nongraph": {"sglang_us": round(t_sgl_ng, 2), "custom_us": round(t_cust_ng, 2), "speedup": round(t_sgl_ng / t_cust_ng, 4)}},
          open(out, "w"), indent=2)
print(f"wrote {out}")
