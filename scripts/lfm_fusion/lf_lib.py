"""Shared definitions for the LFM2.5 kernel-fusion / rewrite study.

Context: the v33 decode audit established that for Qwen3-30B every hot path is
already CUDA-fused, so there is no gap to fill.  LFM2.5-8B-A1B has never been
audited at the operator level, and it is a *different architecture*: 18 of its
24 layers are gated short-convolution layers rather than attention.  This module
holds the model/regime frame and the kernel-name bucketing used by the audit.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results" / "lfm_fusion"
ENVDIR = Path("/home/t-jialianggu/.conda/envs/sglang-dev")
PY = str(ENVDIR / "bin" / "python")

MODELS = {
    "lfm25": dict(
        path="/data/hf/LFM2.5-8B-A1B",
        served="lfm2.5-8b-a1b",
        extra=["--max-prefill-tokens", "16384"],
    ),
    "qwen": dict(
        path="/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
        served="qwen3-30b-a3b",
        extra=[],
    ),
}

# Same three regimes as the regime-kernel study, expressed as bench_one_batch
# shapes.  The regime -> (batch, input_len, output_len) mapping mirrors the
# serving workloads: A = low-batch decode, B = concurrent decode,
# C = long prefill.
REGIME_SHAPES = {
    "A_low_batch_decode": dict(batch=1, input_len=100, output_len=32),
    "B_concurrent_decode": dict(batch=32, input_len=200, output_len=32),
    "C_long_prefill": dict(batch=4, input_len=4000, output_len=8),
}


def run_env(extra: dict | None = None) -> dict:
    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(ENVDIR),
        PATH=f"{ENVDIR/'bin'}:{env.get('PATH','')}",
        HF_HOME=str(REPO / ".hf_cache"),
        TRITON_CACHE_DIR="/tmp/lfm_fusion_triton_cache",
    )
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


# ---------------------------------------------------------------------------
# Kernel-name bucketing.
#
# Order matters: the first matching rule wins.  Rules are substring matches on
# the lowercased CUDA kernel name as reported by torch.profiler.
# ---------------------------------------------------------------------------
BUCKET_RULES = [
    # `norm` must precede `attention`: flashinfer's RMSNorm kernels live in a
    # symbol containing "flashinfer", which the "flash" attention rule matches.
    ("norm", ("rmsnorm", "rms_norm", "layernorm", "layer_norm")),
    ("moe", ("fused_moe", "moe_align", "grouped_topk", "topk_softmax",
             "moe_sum", "silu_and_mul", "moe_fused_gate", "biased_grouped",
             "topkgating", "act_and_mul")),
    ("short_conv", ("causal_conv1d", "conv1d")),
    ("attention", ("flash", "fmha", "attention", "rope", "rotary", "paged",
                   "kv_cache", "set_kv_buffer", "kvcache", "varlen")),
    ("dense_gemm", ("gemm", "cutlass", "sm90", "sm80", "cublas", "nvjet",
                    "matmul", "ampere", "splitk", "gemv")),
    ("elementwise", ("elementwise", "vectorized_elementwise", "mul", "add",
                     "copy", "cat", "transpose", "contiguous", "chunk",
                     "index", "slice", "unrolled")),
    ("sampling", ("sampling", "argmax", "topp", "topk_", "softmax",
                  "multinomial", "sort", "cumsum")),
]


def bucket_of(kernel_name: str) -> str:
    n = kernel_name.lower()
    for bucket, keys in BUCKET_RULES:
        if any(k in n for k in keys):
            return bucket
    return "other"


# ---------------------------------------------------------------------------
# Fusion-gap signatures.
#
# Each entry names a kernel pattern whose *existence* is itself the finding: it
# is work that a fused implementation would not perform at all.  `removable`
# marks gaps where the kernel disappears entirely under fusion (its cost is a
# direct upper bound on the saving); non-removable gaps only get cheaper.
# ---------------------------------------------------------------------------
GAP_SIGNATURES = [
    dict(gap="unfused_rmsnorm", removable=False,
         must=("rmsnormkernel",), must_not=("fusedadd", "gemma"),
         note="RMSNorm invoked without the residual add fused in; sglang's "
              "RMSNorm.forward_cuda(x, residual) would call fused_add_rmsnorm"),
    dict(gap="residual_add", removable=True,
         must=("cudafunctor_add",), must_not=(),
         note="standalone residual addition; absorbed by fused_add_rmsnorm"),
    dict(gap="gating_mul", removable=False,
         must=("binaryfunctor",), must_not=(),
         note="elementwise gating multiply (ShortConv B*x and C*conv_out)"),
    dict(gap="layout_copy", removable=True,
         must=("direct_copy_kernel",), must_not=(),
         note="explicit .contiguous()/transpose materialisation"),
]


def gap_of(kernel_name: str):
    n = kernel_name.lower()
    for g in GAP_SIGNATURES:
        if all(k in n for k in g["must"]) and not any(k in n for k in g["must_not"]):
            return g["gap"]
    return None


def snapshot(outdir: Path, name: str, payload) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


def environment() -> dict:
    def cap(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=60).stdout.strip()
        except Exception as e:  # pragma: no cover
            return f"<{e}>"

    return dict(
        host=cap("hostname"),
        gpu=cap("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
        driver=cap("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        versions=cap(f"{PY} -c \"import torch,triton,sglang;"
                     f"print(torch.__version__, triton.__version__, sglang.__version__)\""),
        sglang_commit=cap("cd /home/t-jialianggu/work/sglang && git rev-parse --short HEAD"),
        python=PY,
    )
