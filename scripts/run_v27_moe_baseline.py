#!/usr/bin/env python3
"""v27: sglang fused_moe real baseline + achieved HBM bandwidth (decode/prefill).

Times sglang's PRODUCTION triton fused_moe (via benchmark_config with the tuned/fallback
config sglang actually loads) on Qwen3-30B-A3B shape, and computes achieved memory
bandwidth to know how much LOSSLESS kernel headroom exists vs the 4.8 TB/s HBM roof.
"""
import json, os, sys, argparse
import torch

BENCH = "/home/t-jialianggu/work/sglang/benchmark/kernels/fused_moe_triton"
sys.path.insert(0, BENCH)
from tuning_fused_moe_triton import benchmark_config
from common_utils import get_model_config
import triton  # noqa

torch.set_default_device("cuda")
torch.cuda.manual_seed_all(0)

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
SGL = "/home/t-jialianggu/work/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs"
FALLBACK = f"{SGL}/triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json"
HBM_TBs = 4.8e12

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=str, default="1,8,32,64,128,256,512,1024,4096")
ap.add_argument("--iters", type=int, default=100)
args = ap.parse_args()

mc = get_model_config(MODEL, tp_size=1)
E, topk, hidden = mc["num_experts"], mc["topk"], mc["hidden_size"]
shard, dtype, block_shape = mc["shard_intermediate_size"], mc["dtype"], mc["block_shape"]
print(f"shape E={E} topk={topk} hidden={hidden} shard={shard}", flush=True)
fb = json.load(open(FALLBACK))

# bytes: w1[E,shard,hidden] + w2[E,hidden,shard/2], bf16=2B. Expected touched experts by
# balls-in-bins for M tokens x topk assignments over E experts.
BYTES_W1 = shard * hidden * 2
BYTES_W2 = hidden * (shard // 2) * 2
def expected_touched(M):
    n = M * topk
    return E * (1 - (1 - 1/E) ** n)

def nearest_cfg(b):
    keys = sorted((int(k) for k in fb), key=lambda k: abs(k - b))
    return fb[str(keys[0])]

rows = []
print(f"\n{'batch':>6}{'time_us':>10}{'touched_E':>11}{'GB_read':>10}{'achieved_TBs':>14}{'%HBM':>7}", flush=True)
for b in [int(x) for x in args.batches.split(",")]:
    t = benchmark_config(nearest_cfg(b), b, E, shard, hidden, topk, dtype,
                         use_fp8_w8a8=False, use_int8_w8a8=False,
                         use_int8_w8a16=False, per_channel_quant=False,
                         block_shape=block_shape, num_iters=args.iters)
    te = expected_touched(b)
    gb = te * (BYTES_W1 + BYTES_W2) / 1e9
    bw = (gb * 1e9) / (t * 1e-6)  # bytes/s
    rows.append({"batch": b, "time_us": round(t, 2), "touched_experts": round(te, 1),
                 "GB_weights_read": round(gb, 3), "achieved_TBs": round(bw/1e12, 3),
                 "pct_HBM": round(bw/HBM_TBs*100, 1)})
    print(f"{b:>6}{t:>10.2f}{te:>11.1f}{gb:>10.3f}{bw/1e12:>13.3f}{bw/HBM_TBs*100:>7.1f}%", flush=True)

out = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-20_v27_moe_baseline/sglang_fused_moe_bandwidth.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"model": MODEL, "note": "achieved bw assumes weight-read-bound (expected touched experts). %HBM near 100 => little lossless headroom",
           "results": rows}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
