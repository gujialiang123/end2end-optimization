"""E2E launcher: install custom MoE patch, then run sglang.bench_one_batch.
Toggle CUSTOM_MOE=1 to use the custom small-M decode kernel (captured into cudagraph).
"""
import os, sys, runpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import custom_moe_patch
custom_moe_patch.install()

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
args = [
    "sglang.bench_one_batch",
    "--model-path", MODEL,
    "--trust-remote-code",
    "--batch-size", "1",
    "--input-len", "256",
    "--output-len", "64",
    "--attention-backend", "fa3",
    "--moe-runner-backend", "triton",
    "--mem-fraction-static", "0.85",
    "--log-decode-step", "1",
]
sys.argv = args
runpy.run_module("sglang.bench_one_batch", run_name="__main__")

print(f"[custom_moe_patch] stats = {custom_moe_patch.stats()}", flush=True)
