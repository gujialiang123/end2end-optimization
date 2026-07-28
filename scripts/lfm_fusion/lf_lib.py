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
        layers=24, arch="MoE + gated short conv (hybrid)",
        maturity="new", tp=1,
    ),
    "qwen": dict(
        path="/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
        served="qwen3-30b-a3b",
        extra=[],
        layers=48, arch="MoE + full attention",
        maturity="mature", tp=1,
    ),
    # --- added for the cross-architecture audit -----------------------------
    # The hypothesis under test is that fusion headroom tracks how long an
    # architecture has been in SGLang, not the framework itself. These span
    # both axes: dense vs MoE, and mature vs recently added.
    "qwen06": dict(
        path="/data/hf/models/Qwen3-0.6B",
        served="qwen3-0.6b",
        extra=[],
        layers=28, arch="dense, llama-style",
        maturity="very mature", tp=1,
    ),
    "gemma3": dict(
        path="/data/hf/models/gemma-3-1b-it",
        served="gemma-3-1b-it",
        extra=[],
        layers=26, arch="dense + sliding-window attention",
        maturity="mature", tp=1,
    ),
    "olmo2": dict(
        path="/data/hf/gujialiang123/models/OLMo-2-1B-Instruct",
        served="olmo-2-1b-instruct",
        extra=[],
        layers=16, arch="olmo2 dense (AllenAI)",
        maturity="AllenAI", tp=1,
    ),
    "exaone4": dict(
        path="/data/hf/gujialiang123/models/EXAONE-4.0-1.2B",
        served="exaone-4.0-1.2b",
        extra=[],
        layers=30, arch="exaone4 dense (LG)",
        maturity="LG", tp=1,
    ),
    "falconh1": dict(
        path="/data/hf/gujialiang123/models/Falcon-H1-1.5B-Instruct",
        served="falcon-h1-1.5b-instruct",
        extra=[],
        layers=24, arch="falcon_h1 hybrid mamba (TII)",
        maturity="TII", tp=1,
    ),
    "granite": dict(
        path="/data/hf/gujialiang123/models/granite-3.3-2b-instruct",
        served="granite-3.3-2b-instruct",
        extra=[],
        layers=40, arch="granite dense (IBM)",
        maturity="IBM", tp=1,
    ),
    "phi4mini": dict(
        path="/data/hf/gujialiang123/models/Phi-4-mini-instruct",
        served="phi-4-mini-instruct",
        extra=[],
        layers=32, arch="phi3 dense (Microsoft)",
        maturity="Microsoft", tp=1,
    ),
    "olmoe": dict(
        path="/data/hf/gujialiang123/models/OLMoE-1B-7B-Instruct",
        served="olmoe-1b-7b-instruct",
        extra=[],
        layers=16, arch="olmoe MoE 64E (AllenAI)",
        maturity="AllenAI", tp=1,
    ),
    "qwen32": dict(
        path="/data/hf/spec_decode/Qwen3-32B",
        served="qwen3-32b",
        extra=[],
        layers=64, arch="dense, llama-style (large)",
        maturity="mature", tp=1,
    ),
    "qwen3next": dict(
        path="/data/hf/hub/models--Qwen--Qwen3-Coder-Next/snapshots/"
             "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
        served="qwen3-coder-next",
        extra=["--max-prefill-tokens", "16384"],
        layers=48, arch="MoE(512E) + gated DeltaNet linear attention (hybrid)",
        maturity="new", tp=2,
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
        TRITON_CACHE_DIR=env.get(
            "TRITON_CACHE_DIR", str(RESULTS / "moesum" / "triton_cache")
        ),
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
    dict(gap="eager_norm_decomp", removable=True,
         must=("reduce_kernel", "meanops"), must_not=(),
         note="RMSNorm running as eager PyTorch (pow->mean->rsqrt->mul) instead "
              "of a fused kernel; the mean reduction is the unambiguous marker"),
    dict(gap="eager_norm_rsqrt", removable=True,
         must=("rsqrt_kernel",), must_not=(),
         note="rsqrt of the eager RMSNorm decomposition"),
    dict(gap="eager_norm_pow", removable=True,
         must=("pow_tensor_scalar",), must_not=(),
         note="square step of the eager RMSNorm decomposition"),
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
         note="explicit .contiguous()/transpose materialisation, or a dtype "
              "cast introduced by an eager-mode numeric path"),
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
