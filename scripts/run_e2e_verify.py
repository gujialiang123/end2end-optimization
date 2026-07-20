import os, sys, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import custom_moe_patch
custom_moe_patch.install()
MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
sys.argv = ["sglang.bench_one_batch","--model-path",MODEL,"--trust-remote-code",
  "--batch-size","1","--input-len","64","--output-len","6","--attention-backend","fa3",
  "--moe-runner-backend","triton","--mem-fraction-static","0.85","--disable-cuda-graph"]
runpy.run_module("sglang.bench_one_batch", run_name="__main__")
print(f"[custom_moe_patch] stats = {custom_moe_patch.stats()}", flush=True)
