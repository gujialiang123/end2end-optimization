#!/usr/bin/env python3
"""Operator-level audit of LFM2.5 (and Qwen as control) across the three regimes.

Method mirrors the v33 decode audit that was done for Qwen3-30B: run
`sglang.bench_one_batch --profile` with CUDA graphs disabled so every operator
shows up as its own kernel, then bucket CUDA kernel time by name.

Why redo it for LFM2.5: v33 concluded "for Qwen3-30B every hot path is already
CUDA-fused, there is no gap to fill".  LFM2.5 is a different architecture
(18/24 layers are gated short convolutions) and its sglang implementation does
*not* use the fused residual+RMSNorm path, so the v33 conclusion cannot be
assumed to carry over.

Usage:
  python scripts/lfm_fusion/lf_audit.py --model lfm25 --regime A_low_batch_decode --gpu 5
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L


def run_bench(model: str, regime: str, gpu: int, outdir: Path, disable_graph: bool):
    m = L.MODELS[model]
    shape = L.REGIME_SHAPES[regime]
    prof_dir = outdir / "trace"
    prof_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        L.PY, "-m", "sglang.bench_one_batch",
        "--model-path", m["path"],
        "--batch", str(shape["batch"]),
        "--input-len", str(shape["input_len"]),
        "--output-len", str(shape["output_len"]),
        "--tensor-parallel-size", "1",
        "--mem-fraction-static", "0.85",
        "--trust-remote-code",
        "--profile",
    ] + m["extra"]
    if disable_graph:
        argv.append("--disable-cuda-graph")

    env = L.run_env({"CUDA_VISIBLE_DEVICES": gpu,
                     "SGLANG_TORCH_PROFILER_DIR": str(prof_dir)})
    log = outdir / "bench.log"
    with open(log, "w") as lf:
        p = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=3600)
    return p.returncode, log, prof_dir, argv


def parse_trace(path: Path):
    """Sum CUDA kernel durations from a chrome trace, bucketed by kernel name."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace

    per_kernel = defaultdict(lambda: dict(us=0.0, n=0))
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = e.get("name", "?")
        per_kernel[name]["us"] += float(e.get("dur", 0.0))
        per_kernel[name]["n"] += 1

    rows = []
    for name, v in per_kernel.items():
        rows.append(dict(kernel=name, bucket=L.bucket_of(name),
                         gap=L.gap_of(name),
                         total_us=v["us"], calls=v["n"]))
    rows.sort(key=lambda r: -r["total_us"])
    return rows


def summarize_gaps(rows, total):
    """Aggregate the fusion-gap signatures — the point of the whole audit."""
    agg = defaultdict(lambda: dict(us=0.0, calls=0, kernels=0))
    for r in rows:
        if r["gap"] is None:
            continue
        g = agg[r["gap"]]
        g["us"] += r["total_us"]
        g["calls"] += r["calls"]
        g["kernels"] += 1
    notes = {g["gap"]: g for g in L.GAP_SIGNATURES}
    out = []
    for name, v in sorted(agg.items(), key=lambda kv: -kv[1]["us"]):
        out.append(dict(gap=name, total_us=round(v["us"], 1),
                        pct_of_kernel_time=round(100 * v["us"] / total, 2),
                        calls=v["calls"], distinct_kernels=v["kernels"],
                        removable_by_fusion=notes[name]["removable"],
                        note=notes[name]["note"]))
    return out


def summarize(rows):
    total = sum(r["total_us"] for r in rows) or 1.0
    per_bucket = defaultdict(lambda: dict(us=0.0, calls=0, kernels=0))
    for r in rows:
        b = per_bucket[r["bucket"]]
        b["us"] += r["total_us"]
        b["calls"] += r["calls"]
        b["kernels"] += 1
    out = []
    for b, v in sorted(per_bucket.items(), key=lambda kv: -kv[1]["us"]):
        out.append(dict(bucket=b, total_us=round(v["us"], 1),
                        pct=round(100 * v["us"] / total, 2),
                        calls=v["calls"], distinct_kernels=v["kernels"]))
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm25", choices=list(L.MODELS))
    ap.add_argument("--regime", default="A_low_batch_decode",
                    choices=list(L.REGIME_SHAPES))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--cuda-graph", action="store_true",
                    help="keep CUDA graphs on (default: disabled for attribution)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reparse", action="store_true",
                    help="re-analyse the traces already on disk, no GPU needed")
    a = ap.parse_args()

    outdir = L.RESULTS / "audit" / f"{a.model}_{a.regime}{a.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    prof_dir = outdir / "trace"

    if a.reparse:
        argv = ["<reparse>"]
    else:
        rc, log, prof_dir, argv = run_bench(a.model, a.regime, a.gpu, outdir,
                                            disable_graph=not a.cuda_graph)
        print(f"bench_one_batch rc={rc}, log={log}")
        if rc != 0:
            print(log.read_text(errors="ignore")[-3000:])
            sys.exit(1)

    traces = sorted(prof_dir.glob("*.trace.json*"))
    if not traces:
        print(f"no trace produced in {prof_dir}")
        sys.exit(1)

    report = dict(model=a.model, regime=a.regime,
                  shape=L.REGIME_SHAPES[a.regime],
                  cuda_graph=a.cuda_graph, argv=argv,
                  environment=L.environment(), stages={})

    for t in traces:
        stage = "prefill" if "prefill" in t.name else (
            "decode" if "decode" in t.name else t.stem)
        rows = parse_trace(t)
        buckets, total = summarize(rows)
        gaps = summarize_gaps(rows, total)
        report["stages"][stage] = dict(
            trace=t.name, total_kernel_us=round(total, 1),
            buckets=buckets, fusion_gaps=gaps, top_kernels=rows[:40])
        print(f"\n=== {a.model} / {a.regime} / {stage} "
              f"(total {total/1000:.2f} ms of CUDA kernel time) ===")
        for b in buckets:
            print(f"  {b['bucket']:14s} {b['pct']:6.2f}%  "
                  f"{b['total_us']/1000:8.3f} ms  calls={b['calls']:6d}")
        if gaps:
            print("  -- fusion gaps --")
            for g in gaps:
                flag = "REMOVABLE" if g["removable_by_fusion"] else "reducible"
                print(f"  {g['gap']:20s} {g['pct_of_kernel_time']:6.2f}%  "
                      f"{g['total_us']/1000:8.3f} ms  calls={g['calls']:6d}  {flag}")

    L.snapshot(outdir, "audit", report)
    print(f"\nwrote {outdir/'audit.json'}")


if __name__ == "__main__":
    main()
