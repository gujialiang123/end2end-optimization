#!/usr/bin/env python3
"""v31: tune the custom small-M MoE (v30) tiling to try to actually beat sglang b1 (31.8us).
Sweeps BN/BK/num_warps for both kernels; reports best vs sglang."""
import json, os, sys, itertools
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
def w1_act_dot(x_ptr, w1_ptr, ids_ptr, act_ptr, H: tl.constexpr, I: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    p = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(ids_ptr + p)
    n = nt * BN + tl.arange(0, BN); m = tl.arange(0, BM)
    accg = tl.zeros((BM, BN), dtype=tl.float32); accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        koff = k0 + tl.arange(0, BK)
        xb = tl.load(x_ptr + m[:, None] * H + koff[None, :], mask=m[:, None] < 1, other=0.0)
        wg = tl.load(w1_ptr + e * (2 * I * H) + n[:, None] * H + koff[None, :]).to(tl.bfloat16)
        wu = tl.load(w1_ptr + e * (2 * I * H) + (I + n[:, None]) * H + koff[None, :]).to(tl.bfloat16)
        accg += tl.dot(xb, wg.T); accu += tl.dot(xb, wu.T)
    silu = accg / (1.0 + tl.exp(-accg))
    tl.store(act_ptr + p * I + n, tl.sum(silu * accu, axis=0).to(tl.bfloat16))


@triton.jit
def w2_sum_dot(act_ptr, w2_ptr, ids_ptr, tw_ptr, out_ptr, H: tl.constexpr, I: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    p = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(ids_ptr + p); tw = tl.load(tw_ptr + p).to(tl.float32)
    n = nt * BN + tl.arange(0, BN); m = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        koff = k0 + tl.arange(0, BK)
        ab = tl.load(act_ptr + p * I + koff[None, :] + m[:, None] * 0, mask=m[:, None] < 1, other=0.0).to(tl.bfloat16)
        w = tl.load(w2_ptr + e * (H * I) + n[:, None] * I + koff[None, :]).to(tl.bfloat16)
        acc += tl.dot(ab, w.T)
    tl.atomic_add(out_ptr + n, tl.sum(acc, axis=0) * tw)


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
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        fl.zero_(); s[i].record(); g.replay(); e[i].record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b) * 1000 for a, b in zip(s, e))[iters // 2] / 10


mc = get_model_config(MODEL, 1)
E, topk, H, shard = mc["num_experts"], mc["topk"], mc["hidden_size"], mc["shard_intermediate_size"]
I = shard // 2; SCALE = (1.0 / H) ** 0.5
x = torch.randn(1, H, dtype=torch.bfloat16)
w1 = (torch.randn(E, shard, H) * SCALE).to(torch.bfloat16)
w2 = (torch.randn(E, H, I) * SCALE).to(torch.bfloat16)
gating = torch.randn(1, E, dtype=torch.float32)
to = select_experts(x, gating, TopKConfig(top_k=topk, renormalize=True))
tw, tid = to.topk_weights, to.topk_ids
ids = tid.view(-1).to(torch.int32); twf = tw.view(-1).float(); P = ids.numel()
out = torch.zeros(H, dtype=torch.float32); act = torch.empty(P, I, dtype=torch.bfloat16)

cfg = MoeRunnerConfig(inplace=False)
t_sgl = timed_graph(lambda: fused_moe(x, w1, w2, to, moe_runner_config=cfg))
print(f"sglang b1: {t_sgl:.2f}us", flush=True)

best = None
for bn1, bk1, w1w in itertools.product([64, 128, 192], [64, 128, 256], [4, 8]):
    for bn2, bk2, w2w in itertools.product([64, 128], [128, 256, 384], [4, 8]):
        if I % bn1 or H % bn2 or H % bk1 or I % bk2:
            continue
        def custom():
            out.zero_()
            w1_act_dot[(P, I // bn1)](x, w1, ids, act, H, I, bn1, bk1, 16, num_warps=w1w)
            w2_sum_dot[(P, H // bn2)](act, w2, ids, twf, out, H, I, bn2, bk2, 16, num_warps=w2w)
            return out
        try:
            t = timed_graph(custom, iters=100)
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, (bn1, bk1, w1w, bn2, bk2, w2w))
            print(f"  new best {t:.2f}us  cfg={best[1]}  (sglang {t_sgl:.2f})", flush=True)

print(f"\nBEST custom: {best[0]:.2f}us vs sglang {t_sgl:.2f}us -> speedup {t_sgl/best[0]:.3f}x", flush=True)
o = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-20_v27_moe_baseline/custom_m1_tuned.json"
json.dump({"sglang_us": round(t_sgl, 2), "best_custom_us": round(best[0], 2),
           "best_cfg": best[1], "speedup": round(t_sgl / best[0], 4)}, open(o, "w"), indent=2)
print(f"wrote {o}")
