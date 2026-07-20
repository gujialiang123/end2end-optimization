#!/usr/bin/env python3
"""v30: small-M decode MoE with tensor cores (tl.dot), skipping align/sort and fusing
act+sum. Target sglang's b1 31.8us (24us GEMM + 6us overhead). Idea: match the GEMM with
tl.dot but remove the align/sort/act/sum overhead by a small-M-specialized single path.
M padded to 16 for tl.dot; only `topk` expert-pairs processed (no expert grouping needed).
"""
import json, os, sys
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
BM = 16  # pad M to 16 for tl.dot


@triton.jit
def w1_act_dot(x_ptr, w1_ptr, ids_ptr, act_ptr, H: tl.constexpr, I: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    p = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(ids_ptr + p)
    n = nt * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        koff = k0 + tl.arange(0, BK)
        xb = tl.load(x_ptr + m[:, None] * H + koff[None, :], mask=m[:, None] < 1, other=0.0)  # [BM,BK]
        wg = tl.load(w1_ptr + e * (2 * I * H) + n[:, None] * H + koff[None, :]).to(tl.bfloat16)  # [BN,BK]
        wu = tl.load(w1_ptr + e * (2 * I * H) + (I + n[:, None]) * H + koff[None, :]).to(tl.bfloat16)
        accg += tl.dot(xb, wg.T)
        accu += tl.dot(xb, wu.T)
    silu = accg / (1.0 + tl.exp(-accg))
    out = (silu * accu)  # [BM,BN], only row 0 real
    r0 = tl.sum(out, axis=0)
    tl.store(act_ptr + p * I + n, r0.to(tl.bfloat16))


@triton.jit
def w2_sum_dot(act_ptr, w2_ptr, ids_ptr, tw_ptr, out_ptr, H: tl.constexpr, I: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    p = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(ids_ptr + p); tw = tl.load(tw_ptr + p).to(tl.float32)
    n = nt * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        koff = k0 + tl.arange(0, BK)
        ab = tl.load(act_ptr + p * I + koff[None, :] + m[:, None] * 0, mask=m[:, None] < 1, other=0.0).to(tl.bfloat16)  # broadcast row0
        w = tl.load(w2_ptr + e * (H * I) + n[:, None] * I + koff[None, :]).to(tl.bfloat16)  # [BN,BK]
        acc += tl.dot(ab, w.T)
    r0 = tl.sum(acc, axis=0) * tw
    tl.atomic_add(out_ptr + n, r0)


def custom(x, w1, w2, ids, tw, H, I, out, act):
    P = ids.numel()
    out.zero_()
    w1_act_dot[(P, I // 64)](x, w1, ids, act, H, I, 64, 64, 16, num_warps=4)
    w2_sum_dot[(P, H // 64)](act, w2, ids, tw, out, H, I, 64, 64, 16, num_warps=4)
    return out


def timed_graph(fn, iters=300):
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
    return lat[len(lat) // 2] / 10


mc = get_model_config(MODEL, 1)
E, topk, H, shard = mc["num_experts"], mc["topk"], mc["hidden_size"], mc["shard_intermediate_size"]
I = shard // 2
SCALE = (1.0 / H) ** 0.5
x = torch.randn(1, H, dtype=torch.bfloat16)
w1 = (torch.randn(E, shard, H) * SCALE).to(torch.bfloat16)
w2 = (torch.randn(E, H, I) * SCALE).to(torch.bfloat16)
gating = torch.randn(1, E, dtype=torch.float32)
to = select_experts(x, gating, TopKConfig(top_k=topk, renormalize=True))
tw, tid = to.topk_weights, to.topk_ids
ids = tid.view(-1).to(torch.int32); twf = tw.view(-1).float(); P = ids.numel()
out = torch.zeros(H, dtype=torch.float32); act = torch.empty(P, I, dtype=torch.bfloat16)

def ref():
    o = torch.zeros(1, H, dtype=torch.float32)
    for j in range(topk):
        e = tid[0, j].item(); wgt = tw[0, j].item()
        y1 = x.float() @ w1[e].float().T
        a = torch.nn.functional.silu(y1[:, :I]) * y1[:, I:]
        o += wgt * (a @ w2[e].float().T)
    return o
r = ref()
g = custom(x, w1, w2, ids, twf, H, I, out, act).view(1, H)
rel = ((r - g.float()).abs() / (r.abs() + 1e-2)).max().item()
print(f"correctness max_rel_err = {rel:.4f} ({'OK' if rel < 0.05 else 'BAD'})", flush=True)

cfg = MoeRunnerConfig(inplace=False)
t_sgl = timed_graph(lambda: fused_moe(x, w1, w2, to, moe_runner_config=cfg))
t_cust = timed_graph(lambda: custom(x, w1, w2, ids, twf, H, I, out, act))
print(f"[cudagraph] sglang {t_sgl:.2f}us | custom(tl.dot) {t_cust:.2f}us | speedup {t_sgl/t_cust:.3f}x", flush=True)

o = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-20_v27_moe_baseline/custom_m1_tldot.json"
json.dump({"max_rel_err": rel, "sglang_us": round(t_sgl, 2), "custom_us": round(t_cust, 2),
           "speedup": round(t_sgl / t_cust, 4)}, open(o, "w"), indent=2)
print(f"wrote {o}")
