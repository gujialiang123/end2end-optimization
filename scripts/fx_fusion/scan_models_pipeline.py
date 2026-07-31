#!/usr/bin/env python3
"""Run the fusion-gap pipeline over several models and report what it finds.

Per model:
  1. static scan of the model file for separate q_norm/k_norm/rope calls
  2. config check against what fused_qk_norm_rope can actually compile for
  3. an operator-level count of eager-norm signatures in a profiled decode

The point of running it over a set rather than one model is that the skill's
stated predictor -- how much optimisation attention a model file has had, which
tracks family prominence -- is a claim about a population. One model cannot
confirm or refute it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Kernel constraints, from fused_qknorm_rope.py and the .cuh static_asserts.
SUPPORTED_HEAD_DIMS = (64, 128, 256)


def model_config(path: str) -> dict:
    c = json.loads(Path(path, "config.json").read_text())
    t = c.get("text_config", c)
    hd = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    return dict(
        arch=(c.get("architectures") or ["?"])[0],
        head_dim=hd,
        n_heads=t.get("num_attention_heads"),
        n_kv=t.get("num_key_value_heads"),
        dtype=str(t.get("torch_dtype")),
        rope_theta=t.get("rope_theta"),
        rope_scaling=t.get("rope_scaling"),
        layers=t.get("num_hidden_layers"),
    )


def norm_scope(src_text: str) -> tuple[str, str]:
    """Is q_norm applied per head, or across all heads at once?

    This decides whether fused_qk_norm_rope is even applicable, and no amount of
    looking at config.json reveals it. OLMo-2 builds `RMSNorm(hidden_size)` and
    normalises the whole q projection in one go; Gemma-3 builds
    `Gemma3RMSNorm(head_dim)` and normalises each head separately. The kernel
    reduces over head_dim per head, so it is equivalent to the second and
    silently wrong for the first.
    """
    m = re.search(r"self\.q_norm\s*=\s*\w+\(\s*([^,)]+)", src_text)
    if not m:
        return "unknown", "no q_norm constructor found"
    arg = m.group(1).strip()
    if "head_dim" in arg and "num" not in arg:
        return "per_head", arg
    if "hidden_size" in arg or "num_heads" in arg or "num_attention_heads" in arg:
        return "across_heads", arg
    return "unknown", arg


def eligibility(cfg: dict, src_text: str) -> tuple[bool, list[str]]:
    reasons = []
    if cfg["head_dim"] not in SUPPORTED_HEAD_DIMS:
        reasons.append(f"head_dim={cfg['head_dim']} not in {SUPPORTED_HEAD_DIMS}")
    if cfg["dtype"] not in ("torch.bfloat16", "bfloat16"):
        reasons.append(f"dtype={cfg['dtype']} (kernel is bf16 only)")
    rs = cfg.get("rope_scaling")
    if isinstance(rs, dict):
        rt = rs.get("rope_type", rs.get("type"))
        # The kernel implements plain rope and YaRN. Anything else would be
        # applied with the wrong frequencies, silently.
        if rt not in (None, "default", "yarn"):
            reasons.append(f"rope_type={rt} (kernel has default and yarn only)")
    if "q_norm" not in src_text and "q_layernorm" not in src_text:
        reasons.append("model file has no q_norm")
    if "fused_qk_norm_rope" in src_text:
        reasons.append("already calls the fused kernel")
    scope, arg = norm_scope(src_text)
    if scope == "across_heads":
        reasons.append(f"q_norm normalises across heads (RMSNorm({arg})), "
                       f"kernel is per-head -- NOT equivalent")
    elif scope == "unknown":
        reasons.append(f"cannot tell norm scope from source (arg={arg}); "
                       f"check by hand before trusting this")
    return (not reasons), reasons


def count_gap_kernels(tree: str, model: str, gpu: str, tag: str) -> dict | None:
    """One profiled decode with cuda graphs off, counting eager-norm kernels."""
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    out_dir = Path(f"/tmp/_prof_{tag}")
    env["SGLANG_TORCH_PROFILER_DIR"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "sglang.bench_one_batch",
           "--model-path", model, "--batch-size", "8",
           "--input-len", "128", "--output-len", "8",
           "--attention-backend", "fa3", "--disable-cuda-graph", "--profile"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    traces = sorted(out_dir.glob("*.trace.json*"))
    if not traces:
        return dict(error=(r.stdout + r.stderr)[-400:])

    import gzip
    p = traces[-1]
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as fh:
        ev = json.load(fh).get("traceEvents", [])
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
    tot = sum(e.get("dur", 0) for e in kern)
    # An RMSNorm decomposed into primitives shows up as a mean-reduction next to
    # rsqrt and a pow at matching call counts.
    sig = re.compile(r"MeanOps|rsqrt_kernel|pow_tensor|reduce_kernel", re.I)
    gap = [e for e in kern if sig.search(e.get("name", ""))]
    gap_us = sum(e.get("dur", 0) for e in gap)
    return dict(total_kernel_us=round(tot, 1), n_kernels=len(kern),
                gap_calls=len(gap), gap_us=round(gap_us, 1),
                gap_pct=round(gap_us / tot * 100, 2) if tot else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--models", nargs="+", required=True,
                    help="pairs of name=path")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = Path(a.src)
    rows = []
    for spec in a.models:
        name, path = spec.split("=", 1)
        modfile = src / "python" / "sglang" / "srt" / "models" / f"{name}.py"
        if not modfile.exists():
            print(f"  {name}: no model file at {modfile}")
            continue
        text = modfile.read_text(errors="ignore")
        cfg = model_config(path)
        ok, reasons = eligibility(cfg, text)

        print(f"\n=== {name}  ({cfg['arch']}) ===")
        print(f"  head_dim={cfg['head_dim']} heads={cfg['n_heads']}/{cfg['n_kv']} "
              f"layers={cfg['layers']} dtype={cfg['dtype']}")
        print(f"  eligible for fused_qk_norm_rope: {'YES' if ok else 'no'}")
        for r in reasons:
            print(f"    - {r}")

        prof = None
        if a.profile:
            prof = count_gap_kernels(a.src, path, a.gpu, name)
            if prof and "error" not in prof:
                print(f"  profiled decode: {prof['n_kernels']} kernels, "
                      f"{prof['gap_calls']} eager-norm-signature calls "
                      f"= {prof['gap_pct']}% of kernel time")
            elif prof:
                print(f"  profile failed: {prof['error'][:160]}")

        rows.append(dict(name=name, path=path, eligible=ok,
                         reasons=reasons, config=cfg, profile=prof))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(src=str(src), models=rows), indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
