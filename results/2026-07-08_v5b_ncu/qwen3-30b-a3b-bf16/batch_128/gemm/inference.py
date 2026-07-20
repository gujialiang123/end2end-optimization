
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = '/data/hf/models/Qwen3-30B-A3B-Instruct-2507'
BATCH_SIZE = 128

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
)
model = model.cuda().eval()

# Build batch: same prompt repeated BATCH_SIZE times
prompt = "The quick brown fox jumps over the lazy dog. " * 20
inputs = tok([prompt] * BATCH_SIZE, return_tensors="pt", padding=True).input_ids.cuda()
print(f"input shape: {inputs.shape}", flush=True)

# Warmup 3 forward passes with kv cache
with torch.no_grad():
    for _ in range(3):
        out = model(inputs, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:].argmax(-1)
torch.cuda.synchronize()
print("Warmup done", flush=True)

# Profiled: 5 decode steps
torch.cuda.cudart().cudaProfilerStart()
with torch.no_grad():
    current = next_tok
    for _ in range(5):
        out = model(current, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        current = out.logits[:, -1:].argmax(-1)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("Profiled section done", flush=True)
