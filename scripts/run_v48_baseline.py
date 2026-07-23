#!/usr/bin/env python3
"""v48 cookbook baseline reference — measured SEPARATELY, never enqueued into Optuna.

Runs the frozen cookbook config (cap=32, chunk=-1, sched=lpm, mem=0.85) with the
same fixed path (moe=triton, fa3, cuda-graph on) and the same R_concurrent_decode
workload. Repeats 5 times after warmup; reports mean/std/95% CI.
Writes baseline_reference.json.
"""
from __future__ import annotations
import json, statistics, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v48_lfm25_plateau as H

OUTDIR = H.OUTDIR
COOKBOOK = H.COOKBOOK


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--port", type=int, default=31701)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    bdir = OUTDIR / "baseline_runs"
    bdir.mkdir(parents=True, exist_ok=True)
    rps_list, rows = [], []
    for i in range(args.repeats):
        tdir = bdir / f"run_{i}"
        m, status, reason, notes, startup = H.evaluate(COOKBOOK, args.port, args.gpu, tdir)
        if status != "ok":
            print(f"[baseline run {i}] FAIL: {reason}", flush=True)
            continue
        row = H.metrics_row(m, COOKBOOK, i, -1, startup, str(tdir / "bench_serving.jsonl"))
        rows.append(row)
        rps_list.append(m["request_throughput"])
        print(f"[baseline run {i}] rps={m['request_throughput']:.3f}", flush=True)

    if len(rps_list) < 2:
        print("ERROR: not enough successful baseline runs", flush=True)
        return 1
    mean = statistics.mean(rps_list)
    std = statistics.stdev(rps_list)
    ci95 = 1.96 * std / math.sqrt(len(rps_list))
    out = dict(
        config=COOKBOOK, repeats=len(rps_list),
        request_throughput_runs=rps_list,
        request_throughput_mean=mean,
        request_throughput_std=std,
        request_throughput_ci95=ci95,
        per_run_rows=rows,
        note="measured separately; NOT enqueued into Optuna; external reference only",
    )
    json.dump(out, open(OUTDIR / "baseline_reference.json", "w"), indent=2)
    print(f"\nBASELINE cookbook: {mean:.3f} ± {std:.3f} rps (95% CI ±{ci95:.3f}, n={len(rps_list)})")
    print(f"saved {OUTDIR}/baseline_reference.json")


if __name__ == "__main__":
    sys.exit(main() or 0)
