#!/usr/bin/env python3
"""v32: generalize the winning small-M MoE kernel to M>1 (token = pair//topk) and find
the crossover batch where sglang (which groups by expert, reading each weight once)
overtakes our per-pair approach (which re-reads an expert weight if reused across tokens).
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


@triton.jit
def w1_act(x_ptr, w1_ptr, ids_ptr, tok_ptr, act_ptr, H: tl.constexpr, I: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    p = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(ids_ptr + p); t = tl.load(tok_ptr + p)
    n = nt * BN + tl.arange(0, BN); m = tl.arange(0, BM)
    accg = tl.zeros((BM, BN), dtype=tl.float32); accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        koff = k0 + tl.arange(0, BK)
        xb = tl.load(x_ptr + t * H + koff[None, :] + m[:, None] * 0, mask=m[:, None] < 1, other=0.0)
        wg = tl.load(w1_ptr + e * (2 * I * H) + n[:, None] * H + koff[None, :]).to(tl.bfloat16)
        wu = tl.load(w1_ptr + e * (2 * I * H) + (I + n[:, None]) * H + koff[None, :]).to(tl.bfloat16)
        accg += tl.dot(xb, wg.T); accu += tl.dot(xb, wu.T)
    silu = accg / (1.0 + tl.exp(-accg))
    tl.store(act_ptr + p * I + n, tl.sum(silu * accu, axis=0).to(tl.bfloat16))


@triton.jit
def w2_sum(act_ptr, w2_ptr, ids_ptr, tok_ptr, tw_ptr, out_ptr, H: tl.constexpr, I: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    p = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(ids_ptr + p); t = tl.load(tok_ptr + p); tw = tl.load(tw_ptr + p).to(tl.float32)
    n = nt * BN + tl.arange(0, BN); m = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        koff = k0 + tl.arange(0, BK)
        ab = tl.load(act_ptr + p * I + koff[None, :] + m[:, None] * 0, mask=m[:, None] < 1, other=0.0).to(tl.bfloat16)
        w = tl.load(w2_ptr + e * (H * I) + n[:, None] * I + koff[None, :]).to(tl.bfloat16)
        acc += tl.dot(ab, w.T)
    tl.atomic_add(out_ptr + t * H + n, tl.sum(acc, axis=0) * tw)


def timed_graph(fn, iters=200):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): fn()
    torch.cuda.synchronize()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    fl = torch.empty(int(256e6 // 4), dtype=torch.int)
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        fl.zero_(); s[i].record(); g.replay(); ev[i].record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b) * 1000 for a, b in zip(s, ev))[iters // 2] / 10


mc = get_model_config(MODEL, 1)
E, topk, H, shard = mc["num_experts"], mc["topk"], mc["hidden_size"], mc["shard_intermediate_size"]
I = shard // 2; SCALE = (1.0 / H) ** 0.5
w1 = (torch.randn(E, shard, H) * SCALE).to(torch.bfloat16)
w2 = (torch.randn(E, H, I) * SCALE).to(torch.bfloat16)
cfg = MoeRunnerConfig(inplace=False)

print(f"{'M':>4}{'sglang_us':>11}{'custom_us':>11}{'speedup':>9}{'rel_err':>9}{'uniqE':>7}", flush=True)
rows = []
for M in [1, 2, 4, 8, 16]:
    torch.manual_seed(M)
    x = torch.randn(M, H, dtype=torch.bfloat16)
    gating = torch.randn(M, E, dtype=torch.float32)
    to = select_experts(x, gating, TopKConfig(top_k=topk, renormalize=True))
    tw, tid = to.topk_weights, to.topk_ids
    ids = tid.view(-1).to(torch.int32)
    tok = torch.arange(M, dtype=torch.int32).repeat_interleave(topk)
    twf = tw.view(-1).float(); P = ids.numel()
    uniqE = len(set(ids.tolist()))
    out = torch.zeros(M, H, dtype=torch.float32); act = torch.empty(P, I, dtype=torch.bfloat16)
    def custom():
        out.zero_()
        w1_act[(P, I // 64)](x, w1, ids, tok, act, H, I, 64, 256, 16, num_warps=4)
        w2_sum[(P, H // 64)](act, w2, ids, tok, twf, out, H, I, 64, 128, 16, num_warps=4)
        return out
    # correctness vs fp32
    r = torch.zeros(M, H)
    for t in range(M):
        for j in range(topk):
            e = tid[t, j].item(); wgt = tw[t, j].item()
            y = x[t:t+1].float() @ w1[e].float().T
            a = torch.nn.functional.silu(y[:, :I]) * y[:, I:]
            r[t:t+1] += wgt * (a @ w2[e].float().T)
    gc = custom()
    rel = ((r - gc.float()).abs() / (r.abs() + 1e-2)).max().item()
    t_sgl = timed_graph(lambda: fused_moe(x, w1, w2, to, moe_runner_config=cfg))
    t_cust = timed_graph(custom)
    rows.append({"M": M, "sglang_us": round(t_sgl, 2), "custom_us": round(t_cust, 2),
                 "speedup": round(t_sgl / t_cust, 4), "rel_err": round(rel, 4), "uniq_experts": uniqE})
    print(f"{M:>4}{t_sgl:>11.2f}{t_cust:>11.2f}{t_sgl/t_cust:>8.3f}x{rel:>9.4f}{uniqE:>7}", flush=True)

o = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-20_v27_moe_baseline/custom_vs_M.json"
json.dump({"results": rows}, open(o, "w"), indent=2)
print(f"wrote {o}")
