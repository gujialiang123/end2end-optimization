#!/usr/bin/env python3
"""v34: build the 'headroom beyond tuning' figures for Qwen3-30B-A3B (decode).
All numbers measured in this project (cited in the doc). Saves PNGs.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results/2026-07-20_v34_figures"
os.makedirs(OUT, exist_ok=True)

# ---------- Figure 1: decode-step composition (measured, v33 audit) ----------
buckets = json.load(open("results/2026-07-20_v33_decode_audit/decode_buckets.json"))
order = ["MoE", "dense_gemm(qkv/o/gate/lm_head)", "attention(+KV+rope)", "norm", "misc/elementwise", "activation", "sampling"]
vals = [buckets.get(k, 0) for k in order]
tot = sum(vals); pct = [100 * v / tot for v in vals]
labels = ["MoE\n(expert wts)", "dense GEMM\n(qkv/o/lm_head)", "attention\n(+KV/rope)", "norm", "misc", "act", "sample"]
fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#7f7f7f", "#9467bd", "#8c564b"]
bars = ax.bar(labels, pct, color=colors)
for b, p in zip(bars, pct):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.5, f"{p:.0f}%", ha="center", fontsize=9)
ax.set_ylabel("% of decode-step GPU kernel time")
ax.set_title("Qwen3-30B-A3B decode step composition (b=1, measured)\nMoE+dense+attn = 89%, all memory-bound weight/KV reads")
ax.set_ylim(0, 48)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_decode_composition.png", dpi=130); plt.close()

# ---------- Figure 2: MoE achieved HBM% vs batch (measured, v27) ----------
bw = json.load(open("results/2026-07-20_v27_moe_baseline/sglang_fused_moe_bandwidth.json"))["results"]
B = [r["batch"] for r in bw]; H = [r["pct_HBM"] for r in bw]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(B, H, "o-", color="#d62728", lw=2)
ax.axhspan(70, 100, color="green", alpha=0.06)
ax.set_xscale("log", base=2); ax.set_xticks(B); ax.set_xticklabels(B)
ax.set_xlabel("MoE batch (tokens)"); ax.set_ylabel("achieved HBM bandwidth (% of 4.8 TB/s)")
ax.set_title("sglang fused_moe: memory-bound at decode, compute-bound at prefill (measured)")
ax.axhline(100, ls="--", color="gray", lw=1)
for x, y in [(1, H[0]), (32, H[2]), (4096, H[-1])]:
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.text(2, 90, "near HBM roof\n(decode, no lossless\nkernel headroom)", fontsize=8, color="green")
ax.text(700, 40, "compute-bound\n(prefill; config-tuning\ngot +50%)", fontsize=8, color="#1f77b4")
ax.set_ylim(0, 105)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_moe_bandwidth_vs_batch.png", dpi=130); plt.close()

# ---------- Figure 3: headroom beyond tuning (decode, normalized speedup) ----------
# measured: baseline (tuned)=1.0; +kernel rewrite (e2e, c1)=1.015; +spec c1=1.066, c32=1.306; roofline ceiling=1.85
fig, ax = plt.subplots(figsize=(9, 4.6))
groups = ["single request (c=1)", "concurrency 32"]
x = np.arange(len(groups)); w = 0.2
base = [1.0, 1.0]
kernel = [1.015, None]        # kernel rewrite e2e only measured at c1
spec = [1.066, 1.306]         # spec decoding e2e (A1b)
ceiling = [1.85, 1.85]        # roofline decode ceiling (upper bound, exact)
ax.bar(x - 1.5*w, base, w, label="best-tuned config (baseline)", color="#7f7f7f")
ax.bar(x - 0.5*w, [k if k else 0 for k in kernel], w, label="+ kernel rewrite (measured e2e)", color="#ff7f0e")
ax.bar(x + 0.5*w, spec, w, label="+ spec decoding (measured e2e, exact)", color="#d62728")
ax.bar(x + 1.5*w, ceiling, w, label="roofline ceiling (theoretical, exact)", color="#2ca02c", alpha=0.5, hatch="//")
for xi, (k, s, c) in zip(x, zip(kernel, spec, ceiling)):
    if k: ax.text(xi - 0.5*w, k + 0.01, f"+{(k-1)*100:.1f}%", ha="center", fontsize=8)
    ax.text(xi + 0.5*w, s + 0.01, f"+{(s-1)*100:.1f}%", ha="center", fontsize=8, color="#d62728")
    ax.text(xi + 1.5*w, c + 0.01, f"{c:.2f}x", ha="center", fontsize=8, color="#2ca02c")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("decode speedup vs best-tuned config")
ax.set_title("Headroom BEYOND config tuning — Qwen3-30B-A3B decode\n(tuning plateaus; spec decoding realizes the biggest exact gain; kernel rewrite tiny)")
ax.axhline(1.0, color="black", lw=0.8)
ax.legend(fontsize=8, loc="upper left"); ax.set_ylim(0.9, 2.0)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_headroom_beyond_tuning.png", dpi=130); plt.close()

print("wrote 3 figures to", OUT)
for f in os.listdir(OUT):
    print("  ", f)
