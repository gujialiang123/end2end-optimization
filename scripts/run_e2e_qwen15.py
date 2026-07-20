import os, sys, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen15_gate_patch
qwen15_gate_patch.install()
MODEL = "/home/t-jialianggu/models/Qwen1.5-MoE-A2.7B-Chat"
sys.argv = ["sglang.bench_one_batch","--model-path",MODEL,"--trust-remote-code",
  "--batch-size","1","--input-len","256","--output-len","64",
  "--attention-backend","fa3","--moe-runner-backend","triton","--mem-fraction-static","0.85"]
runpy.run_module("sglang.bench_one_batch", run_name="__main__")
print(f"[qwen15_gate_patch] stats = {qwen15_gate_patch.stats()}", flush=True)
