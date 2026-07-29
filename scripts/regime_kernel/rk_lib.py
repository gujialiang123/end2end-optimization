#!/usr/bin/env python3
"""Regime-aware kernel specialization — shared harness.

Everything in this study operates on the fused-MoE Triton kernel that SGLang
already ships. We never write a new CUDA kernel: we change the *configuration*
that the runtime selects for a given M.

IMPORTANT — what M actually is. `fused_experts_impl` computes

    M = min(num_tokens, CHUNK_SIZE)

i.e. M is the **token count** of the batch entering the MoE layer, NOT
tokens x top_k. (top_k affects the expert assignment, not this dimension.)
Measured traces confirm it: a 100-token prompt produces lookups at M ~ 101-125,
not 400. The runtime then picks a config with
`configs[min(configs.keys(), key=lambda x: abs(x - M))]`. Regimes differ by
orders of magnitude in M, which is the mechanism this study tests.

Default behaviour is never modified: candidate configs are applied through the
existing `override_config()` context manager (microbenchmark) or through the
`SGLANG_MOE_CONFIG_DIR` environment variable (end-to-end).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path("/home/t-jialianggu/work/EndtoEnd-auto-optimization")
ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
RESULTS = REPO / "results/regime_kernel"
CONFIGS = REPO / "configs/regime_kernel"

# --------------------------------------------------------------------------
# Models: the two shapes we already have full serving data for, and which the
# runtime reports as having NO tuned H200 config for Triton 3.5.1.
# --------------------------------------------------------------------------
MODELS: Dict[str, dict] = {
    "lfm25": dict(
        path="/data/hf/LFM2.5-8B-A1B",
        served="lfm2.5-8b-a1b",
        num_experts=32,
        top_k=4,
        hidden_size=2048,
        moe_intermediate_size=1792,
        num_hidden_layers=24,
        extra_server_args=["--max-prefill-tokens", "16384"],
        # runtime says: "Using default MoE kernel config" (no file at all)
        default_config_status="heuristic default (no config file)",
    ),
    "qwen": dict(
        path="/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
        served="qwen3-30b-a3b",
        num_experts=128,
        top_k=8,
        hidden_size=2048,
        moe_intermediate_size=768,
        num_hidden_layers=48,
        extra_server_args=[],
        # runtime says: falls back to the triton_3_2_0 config
        default_config_status="triton_3_2_0 fallback config",
    ),
}

# --------------------------------------------------------------------------
# Regimes. token_batch is the number of tokens the MoE layer sees per
# invocation; M = token_batch * top_k.
# --------------------------------------------------------------------------
@dataclass
class Regime:
    name: str
    workload: str          # frozen workload in the serving campaign
    phase: str             # decode | prefill
    token_batch: int       # tokens per MoE invocation
    note: str


REGIMES: Dict[str, Regime] = {
    "A_low_batch_decode": Regime(
        "A_low_batch_decode", "R_short_decode", "decode", 1,
        "1 active request: smallest M, launch-overhead / low-occupancy regime"),
    "B_concurrent_decode": Regime(
        "B_concurrent_decode", "R_concurrent_decode", "decode", 32,
        "32 active requests: grouped GEMM, weight movement, expert routing"),
    "C_long_prefill": Regime(
        "C_long_prefill", "R_long_prefill", "prefill", 2048,
        "long prompt, chunked prefill: large M, Tensor-Core regime"),
}

# Light sweep used only at kernel level to locate crossovers.
TOKEN_SWEEP = [1, 2, 4, 8, 16, 32, 64]
# Extra token counts representing prefill chunks.
PREFILL_TOKENS = [512, 2048, 8192]

SEED = 20260726


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------
def legal_config(cfg: dict, M: int, N: int, K: int) -> bool:
    """Prune configs that are illegal or provably pointless for this shape.

    Upstream enumerates 1920 combinations per M; most are meaningless at small M
    (a BLOCK_SIZE_M of 128 for M=4 wastes 97 % of the tile) or exceed the shared
    memory budget. Filtering here is what makes the sweep affordable.
    """
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    # do not tile far beyond the actual M
    if bm > max(16, 2 * M):
        return False
    # tiles must not dwarf the matrix dimensions
    if bn > 2 * N or bk > 2 * K:
        return False
    # shared-memory estimate for a BF16 double-ish buffered pipeline
    smem = cfg["num_stages"] * (bm * bk + bn * bk) * 2
    if smem > 227 * 1024:            # H200 max dynamic smem per block
        return False
    # very large tiles with few warps serialise badly
    if bm * bn >= 128 * 256 and cfg["num_warps"] < 8:
        return False
    return True


def build_search_space(M: int, N: int, K: int) -> List[dict]:
    """Candidate configs to sweep.

    BLOCK_SIZE_K includes 32 and GROUP_SIZE_M includes 8 so that the space
    contains sglang's own `get_default_config` values -- the baseline we
    compare against. An earlier version started BK at 64 and omitted GM=8,
    which meant we were beating a baseline whose neighbourhood we had never
    measured. That blind spot hid the reason Triton 3.6 changed the result:
    3.6's gain is concentrated at BK=32 (1.4-1.7x) and is only 0-4% for every
    BK we had been sweeping. See docs/2026-07-29/triton_36_retune_findings.md.
    """
    space = []
    for num_stages in (2, 3, 4, 5):
        for bm in (16, 32, 64, 128, 256):
            for bk in (32, 64, 128, 256):
                for bn in (32, 64, 128, 256):
                    for warps in (4, 8):
                        for gm in (1, 8, 16, 32):
                            cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                                       BLOCK_SIZE_K=bk, GROUP_SIZE_M=gm,
                                       num_warps=warps, num_stages=num_stages)
                            if legal_config(cfg, M, N, K):
                                space.append(cfg)
    return space


def config_key(cfg: dict) -> str:
    return (f"bm{cfg['BLOCK_SIZE_M']}_bn{cfg['BLOCK_SIZE_N']}_bk{cfg['BLOCK_SIZE_K']}"
            f"_gm{cfg['GROUP_SIZE_M']}_w{cfg['num_warps']}_s{cfg['num_stages']}")


# --------------------------------------------------------------------------
# Routing distributions (for the routing-control experiment)
# --------------------------------------------------------------------------
def make_gating(num_tokens: int, num_experts: int, routing: str, torch, gen):
    """Return router logits producing the requested expert-load distribution."""
    if routing == "uniform":
        # near-uniform: plain gaussian logits
        return torch.randn(num_tokens, num_experts, dtype=torch.float32,
                           device="cuda", generator=gen)
    if routing == "skewed":
        # a few hot experts: add a decaying bias so load is strongly imbalanced
        g = torch.randn(num_tokens, num_experts, dtype=torch.float32,
                        device="cuda", generator=gen)
        bias = torch.linspace(4.0, 0.0, num_experts, device="cuda").unsqueeze(0)
        return g + bias
    raise ValueError(routing)


def expert_load_stats(topk_ids, num_experts: int, torch) -> dict:
    counts = torch.bincount(topk_ids.flatten().to(torch.int64),
                            minlength=num_experts).float()
    total = counts.sum().item()
    active = int((counts > 0).sum().item())
    mean = counts.mean().item()
    std = counts.std(unbiased=False).item()
    cv = (std / mean) if mean > 0 else 0.0
    srt, _ = torch.sort(counts)
    n = counts.numel()
    idx = torch.arange(1, n + 1, device=counts.device, dtype=torch.float32)
    gini = ((2 * idx - n - 1) * srt).sum().item() / (n * srt.sum().item()) \
        if srt.sum().item() > 0 else 0.0
    return dict(total_assignments=total, active_experts=active,
                max_expert_load=counts.max().item(),
                mean_expert_load=mean, cv_expert_load=cv, gini_expert_load=gini)


# --------------------------------------------------------------------------
# Environment / provenance
# --------------------------------------------------------------------------
def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""


def environment() -> dict:
    py = f"{ENVDIR}/bin/python"
    return dict(
        git_sha=sh(f"cd {REPO} && git rev-parse HEAD"),
        git_branch=sh(f"cd {REPO} && git rev-parse --abbrev-ref HEAD"),
        host=sh("hostname"),
        gpu=sh("nvidia-smi --query-gpu=name --format=csv,noheader -i 0"),
        driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0"),
        torch=sh(f"{py} -c 'import torch;print(torch.__version__)'"),
        triton=sh(f"{py} -c 'import triton;print(triton.__version__)'"),
        sglang=sh(f"{py} -c 'import sglang;print(sglang.__version__)'"),
        sglang_commit=sh("cd /home/t-jialianggu/work/sglang && git rev-parse HEAD"),
        seed=SEED,
    )


def run_env() -> dict:
    e = dict(os.environ)
    # RK_ENVDIR lets a run use a different interpreter than the default one --
    # needed to measure on a second Triton version, since a tuned config is only
    # valid for the version it was tuned on. RK_SGLANG_SRC puts a matching
    # source tree ahead of whatever that env has installed.
    envdir = e.get("RK_ENVDIR", ENVDIR)
    cuda_home = e.get("RK_CUDA_HOME", envdir)
    e.update(CUDA_HOME=cuda_home,
             PATH=f"{cuda_home}/bin:{envdir}/bin:" + e.get("PATH", ""),
             HF_HOME=str(REPO / ".hf_cache"),
             TRITON_CACHE_DIR=e.get("TRITON_CACHE_DIR",
                                    "/tmp/regime_kernel_triton_cache"))
    for var, extra in (("LD_LIBRARY_PATH", f"{cuda_home}/lib"),
                       ("LIBRARY_PATH", f"{cuda_home}/lib")):
        if cuda_home != envdir:
            e[var] = extra + (os.pathsep + e[var] if e.get(var) else "")
    if e.get("RK_SGLANG_SRC"):
        src = e["RK_SGLANG_SRC"]
        e["PYTHONPATH"] = src + (os.pathsep + e["PYTHONPATH"]
                                 if e.get("PYTHONPATH") else "")
    return e


def run_python() -> str:
    """Interpreter for server/bench subprocesses (see run_env)."""
    envdir = os.environ.get("RK_ENVDIR", ENVDIR)
    return f"{envdir}/bin/python"


def snapshot(outdir: Path, name: str, payload: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------
def summarize(times_ms: List[float]) -> dict:
    s = sorted(times_ms)
    n = len(s)
    def pct(p):
        return s[min(n - 1, max(0, int(round(p / 100 * n)) - 1))] if n else float("nan")
    return dict(median_ms=statistics.median(s) if s else float("nan"),
                mean_ms=statistics.mean(s) if s else float("nan"),
                p95_ms=pct(95), min_ms=s[0] if s else float("nan"),
                max_ms=s[-1] if s else float("nan"),
                std_ms=statistics.pstdev(s) if len(s) > 1 else 0.0,
                n=len(s))
