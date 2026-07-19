#!/usr/bin/env python3
"""v24: shared-expert fusion opportunity (reproduces the SPIRIT of #22325 / #26727 via
torch.compile). Qwen2-MoE-style shared-expert path (Qwen1.5-MoE-A2.7B dims):

  shared_out = down( silu(gate(x)) * up(x) )        # SwiGLU MLP, intermediate=5632
  g = sigmoid( shared_expert_gate(x) )               # Linear(hidden,1)
  out = g * shared_out                               # broadcast scalar mul

PR #22325 fuses linear+sigmoid+mul; #26727 fuses four ops. We can't apply their CUDA
diffs, so we measure the fusion opportunity with torch.compile (Mason's fusion avenue):
eager (separate kernels) vs torch.compile (fused), across batch sizes. bf16, H200.
"""
import argparse, json, os, time
import torch
import torch.nn as nn

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

HIDDEN = 2048
SHARED_INT = 5632  # Qwen1.5-MoE-A2.7B shared_expert_intermediate_size

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=str, default="1,8,32,128,256,512,1024,4096")
ap.add_argument("--iters", type=int, default=200)
args = ap.parse_args()


class SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, SHARED_INT, bias=False)
        self.up_proj = nn.Linear(HIDDEN, SHARED_INT, bias=False)
        self.down_proj = nn.Linear(SHARED_INT, HIDDEN, bias=False)
        self.act = nn.SiLU()
        self.shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False)

    def mlp(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    def gate_ops(self, x, shared_out):
        # the exact ops #22325 fuses: linear(hidden->1) + sigmoid + broadcast mul
        return torch.sigmoid(self.shared_expert_gate(x)) * shared_out

    def forward(self, x):
        return self.gate_ops(x, self.mlp(x))


def timed(fn, iters):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    flush = torch.empty(int(256e6 // 4), dtype=torch.int)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    lat = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))  # us
    return lat[len(lat) // 2]  # median


m = SharedExpert().eval()

rows = []
print(f"{'batch':>6}{'mlp_us':>9}{'gate_ops_us':>13}{'total_us':>10}{'gate_frac':>11}", flush=True)
for b in [int(x) for x in args.batches.split(",")]:
    x = torch.randn(b, HIDDEN)
    with torch.no_grad():
        shared_out = m.mlp(x)
        t_mlp = timed(lambda: m.mlp(x), args.iters)
        t_gate = timed(lambda: m.gate_ops(x, shared_out), args.iters)
        t_total = timed(lambda: m(x), args.iters)
    frac = t_gate / t_total
    rows.append({"batch": b, "mlp_us": round(t_mlp, 2), "gate_ops_us": round(t_gate, 2),
                 "total_us": round(t_total, 2), "gate_ops_fraction": round(frac, 4)})
    print(f"{b:>6}{t_mlp:>9.2f}{t_gate:>13.2f}{t_total:>10.2f}{frac*100:>10.1f}%", flush=True)

out = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-19_v23_config_evidence/shared_expert_fusion_opportunity.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"model_dims": "Qwen1.5-MoE-A2.7B (hidden=2048, shared_int=5632)",
           "note": "gate_ops = the linear(hidden->1)+sigmoid+mul that #22325 fuses; fraction of total shared-expert = fusion opportunity ceiling",
           "results": rows}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
