#!/usr/bin/env python3
"""v50 NCU microbench: isolate the sglang Triton fused_moe grouped-GEMM kernel at a
single token count M so Nsight Compute captures ONE clean launch for roofline.

Qwen3-30B-A3B shape (E=128, shard_intermediate=1536, hidden=2048, topk=8, bf16).
Run under: ncu --set full --kernel-name regex:fused_moe --launch-skip-before-match 0
           --launch-count 1 --export ... python run_v50_ncu_moe_microbench.py --M 32

Decode regime  -> M = batch (e.g. 32, 128): few tokens, re-reads all expert weights
                  -> low arithmetic intensity -> MEMORY-BOUND (DRAM roof).
Prefill regime -> M = batch*seqlen (e.g. 2048, 4096): many tokens amortize the weight
                  read -> high arithmetic intensity -> COMPUTE-BOUND (FP roof).
"""
import argparse, os, sys
import torch

BENCH = "/home/t-jialianggu/work/sglang/benchmark/kernels/fused_moe_triton"
sys.path.insert(0, BENCH)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe  # noqa
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig  # noqa
from sglang.srt.layers.moe.topk import TopKConfig, select_experts  # noqa
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa

torch.set_default_device("cuda")
torch.manual_seed(0)

# fused_moe's config lookup reads the global server args; init a minimal one.
set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

E = 128            # num_experts
N = 1536           # shard_intermediate_size (2*768)
H = 2048           # hidden_size
TOPK = 8
DTYPE = torch.bfloat16


def build(M):
    x = torch.randn(M, H, dtype=DTYPE)
    w1 = torch.randn(E, N, H, dtype=DTYPE)
    w2 = torch.randn(E, H, N // 2, dtype=DTYPE)
    gating = torch.randn(M, E, dtype=torch.float32)
    return x, w1, w2, gating


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--iters", type=int, default=1, help="kernel launches (NCU captures these)")
    args = ap.parse_args()

    x, w1, w2, gating = build(args.M)
    topk_config = TopKConfig(top_k=TOPK, renormalize=True)
    topk_out = select_experts(x, gating, topk_config)
    cfg = MoeRunnerConfig(inplace=True)

    def run():
        fused_moe(x, w1, w2, topk_out, moe_runner_config=cfg,
                  use_fp8_w8a8=False, use_int8_w8a8=False,
                  use_int8_w8a16=False, use_int4_w4a16=False,
                  w1_scale=None, w2_scale=None, a1_scale=None, a2_scale=None,
                  per_channel_quant=False, block_shape=None)

    # JIT + warmup (NCU should skip these via --launch-skip-before-match on the profiled region;
    # we instead warm up here and profile the post-warmup launches with --launch-count)
    run(); torch.cuda.synchronize()
    for _ in range(args.iters):
        run()
    torch.cuda.synchronize()
    print(f"[v50] M={args.M} done: fused_moe E={E} N={N} H={H} topk={TOPK} bf16", flush=True)


if __name__ == "__main__":
    main()
