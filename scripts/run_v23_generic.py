#!/usr/bin/env python3
"""v23c: model-agnostic config-tuning benchmark. Compares default heuristic vs a
tuned fused_moe config on the real triton kernel, for any MoE model shape.

Usage:
  run_v23_generic.py --model <path> --ours <tuned.json> [--fallback <json>] --batches 1,256,4096
"""
import json, os, sys, argparse
import torch

BENCH_DIR = "/home/t-jialianggu/work/sglang/benchmark/kernels/fused_moe_triton"
sys.path.insert(0, BENCH_DIR)
from tuning_fused_moe_triton import benchmark_config  # noqa: E402
from common_utils import get_model_config  # noqa: E402

torch.set_default_device("cuda")
torch.cuda.manual_seed_all(0)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ours", required=True, help="merged tuned config json")
ap.add_argument("--fallback", default=None)
ap.add_argument("--batches", type=str, default="1,256,4096")
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--out", required=True)
args = ap.parse_args()

mc = get_model_config(args.model, tp_size=1)
E, topk, hidden = mc["num_experts"], mc["topk"], mc["hidden_size"]
shard, dtype, block_shape = mc["shard_intermediate_size"], mc["dtype"], mc["block_shape"]
print(f"shape: E={E} topk={topk} hidden={hidden} shard_intermediate={shard} dtype={dtype}", flush=True)

ours_all = json.load(open(args.ours))
fb_all = json.load(open(args.fallback)) if args.fallback else {}


def default_heuristic(M):
    if M <= E:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}


def bench(cfg, M):
    return benchmark_config(cfg, M, E, shard, hidden, topk, dtype,
                            use_fp8_w8a8=False, use_int8_w8a8=False,
                            use_int8_w8a16=False, per_channel_quant=False,
                            block_shape=block_shape, num_iters=args.iters)


rows = []
print(f"\n{'batch':>6}{'default_us':>12}{'tuned_us':>10}{'tuned/default':>15}", flush=True)
for b in [int(x) for x in args.batches.split(",")]:
    key = str(b)
    if key not in ours_all:
        print(f"batch {b}: no tuned entry, skip"); continue
    t_d = bench(default_heuristic(b), b)
    t_o = bench(ours_all[key], b)
    t_fb = bench(fb_all[key], b) if key in fb_all else None
    best = min([t for t in [t_o, t_fb] if t is not None])
    sp = t_d / best
    rows.append({"batch": b, "default_us": round(t_d, 2), "tuned_us": round(t_o, 2),
                 "fallback_us": round(t_fb, 2) if t_fb else None,
                 "tuned_over_default": round(sp, 4)})
    print(f"{b:>6}{t_d:>12.2f}{t_o:>10.2f}{sp:>14.3f}x", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"model": args.model, "shape": {"E": E, "shard_N": shard, "hidden": hidden, "topk": topk},
           "results": rows}, open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}")
