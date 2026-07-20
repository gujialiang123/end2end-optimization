#!/usr/bin/env python3
"""v28: diagnose b1 decode MoE — can the existing triton kernel hit higher bandwidth
with a different (parallelism-heavy) config? Sweep BLOCK_N / GROUP / warps / stages at
small M. If max achievable >> 49%, the tuning missed it (config fix). If still ~49%,
the triton kernel is fundamentally parallelism-starved at M=1 (needs a rewrite / split-K).
"""
import json, os, sys, argparse
import torch

BENCH = "/home/t-jialianggu/work/sglang/benchmark/kernels/fused_moe_triton"
sys.path.insert(0, BENCH)
from tuning_fused_moe_triton import benchmark_config
from common_utils import get_model_config

torch.set_default_device("cuda")
torch.cuda.manual_seed_all(0)
MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
HBM = 4.8e12

mc = get_model_config(MODEL, tp_size=1)
E, topk, hidden = mc["num_experts"], mc["topk"], mc["hidden_size"]
shard, dtype, block_shape = mc["shard_intermediate_size"], mc["dtype"], mc["block_shape"]
BYTES = (shard*hidden + hidden*(shard//2)) * 2

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--iters", type=int, default=200)
args = ap.parse_args()
M = args.batch
te = E * (1 - (1 - 1/E) ** (M*topk))
gb = te * BYTES / 1e9

# search a parallelism-oriented grid (small BLOCK_M since M tiny; vary N/K/GROUP/warps/stages/splitk-ish)
configs = []
for bn in [32, 64, 128]:
    for bk in [64, 128, 256]:
        for g in [1, 16, 64]:
            for w in [4, 8]:
                for s in [2, 3, 4, 5]:
                    configs.append({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                                    "GROUP_SIZE_M": g, "num_warps": w, "num_stages": s})

best = None
results = []
print(f"M={M} touched_E={te:.1f} weights={gb:.3f}GB  ideal_time={gb*1e9/HBM*1e6:.1f}us", flush=True)
for i, c in enumerate(configs):
    try:
        t = benchmark_config(c, M, E, shard, hidden, topk, dtype, False, False, False, False,
                             block_shape=block_shape, num_iters=args.iters)
    except Exception:
        continue
    bw = gb*1e9 / (t*1e-6)
    pct = bw/HBM*100
    results.append({"cfg": c, "us": round(t,2), "pctHBM": round(pct,1)})
    if best is None or t < best["us"]:
        best = {"cfg": c, "us": round(t,2), "pctHBM": round(pct,1)}
    if i % 20 == 0:
        print(f"  [{i}/{len(configs)}] best so far {best['us']}us {best['pctHBM']}%", flush=True)

results.sort(key=lambda r: r["us"])
print(f"\n== best config at M={M} ==")
for r in results[:5]:
    print(f"  {r['us']}us  {r['pctHBM']}% HBM  {r['cfg']}", flush=True)

out = f"/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-20_v27_moe_baseline/b{M}_config_sweep.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"M": M, "touched_E": te, "ideal_us": gb*1e9/HBM*1e6, "top5": results[:5], "all": results},
          open(out, "w"), indent=2)
print(f"wrote {out}")
