#!/usr/bin/env python3
"""End-to-end A/B of the fused qk-norm + rope path on Gemma-3.

Runs bench_one_batch against two source trees over several regimes, repeated,
and reports the median with a Welch t-test on the repetitions.

Two things this deliberately does not do:

  - quote the microbenchmark speedup as an expected end-to-end gain. The kernel
    is 1.9-2.3x on the preamble alone, which is a small slice of a decode step;
    the honest number is whatever this measures.
  - reuse an old baseline. Both arms run here, now, on the same commit, so the
    difference is the patch and not four days of upstream movement.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REGIMES = [
    ("decode_bs1",    dict(batch_size=1,  input_len=128,  output_len=64)),
    ("decode_bs32",   dict(batch_size=32, input_len=128,  output_len=64)),
    ("decode_bs64",   dict(batch_size=64, input_len=512,  output_len=64)),
    ("prefill_heavy", dict(batch_size=8,  input_len=2048, output_len=8)),
]


def bench(tree: str, model: str, gpu: str, cfg: dict,
          extra_env: dict | None = None) -> dict | None:
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.update(extra_env or {})
    cmd = [sys.executable, "-m", "sglang.bench_one_batch",
           "--model-path", model,
           "--batch-size", str(cfg["batch_size"]),
           "--input-len", str(cfg["input_len"]),
           "--output-len", str(cfg["output_len"]),
           "--attention-backend", "fa3"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    text = r.stdout + r.stderr
    # The second block is the measured run; the first is warmup.
    dec = re.findall(r"Decode\.\s+median latency:\s+([\d.]+) s", text)
    pre = re.findall(r"Prefill\. latency:\s+([\d.]+) s", text)
    tot = re.findall(r"Total\. latency:\s+([\d.]+) s.*?throughput:\s+([\d.]+)", text)
    if not dec or not tot:
        print(text[-1500:])
        return None
    return dict(decode_median_s=float(dec[-1]),
                prefill_s=float(pre[-1]) if pre else None,
                total_s=float(tot[-1][0]),
                total_tps=float(tot[-1][1]))


def welch_p(xs, ys) -> float:
    from scipy.stats import t as tdist
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return 1.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx, vy = statistics.variance(xs), statistics.variance(ys)
    if vx == 0 and vy == 0:
        return 1.0
    se = (vx / nx + vy / ny) ** 0.5
    tstat = (mx - my) / se
    df = (vx / nx + vy / ny) ** 2 / (
        (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return float(2 * tdist.sf(abs(tstat), df))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-tree", required=True,
                    help="unmodified main")
    ap.add_argument("--ablate", action="store_true",
                    help="add an arm with the fused norm but a separate rope, "
                         "which isolates this change from PR #32670's")
    ap.add_argument("--patched-tree", required=True)
    ap.add_argument("--model", default="/data/hf/models/gemma-3-1b-it")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for name, cfg in REGIMES:
        print(f"\n=== {name}: bs={cfg['batch_size']} in={cfg['input_len']} "
              f"out={cfg['output_len']} ===")
        arms = {}
        # The ablation arm lives in the patched tree behind an env var, so it
        # shares all of this file's layout handling. Two separate trees would
        # also differ in how the norm output is laid out, and that difference
        # alone crashed the attention backend when tried.
        trees = [("baseline", a.baseline_tree, {})]
        if a.ablate:
            trees.append(("fused_norm_only", a.patched_tree,
                          {"SGLANG_GEMMA3_NO_FUSED_ROPE": "1"}))
        trees.append(("fused", a.patched_tree, {}))
        for arm, tree, extra in trees:
            vals = []
            for i in range(a.reps):
                r = bench(tree, a.model, a.gpu, cfg, extra)
                if r is None:
                    print(f"  {arm} rep {i} failed")
                    continue
                vals.append(r)
            if not vals:
                break
            arms[arm] = vals
            dm = statistics.median(v["decode_median_s"] for v in vals)
            tp = statistics.median(v["total_tps"] for v in vals)
            print(f"  {arm:9s} decode {dm * 1000:7.3f} ms   total {tp:9.1f} tok/s"
                  f"   (n={len(vals)})")
        if "baseline" not in arms or "fused" not in arms:
            continue

        b = [v["decode_median_s"] for v in arms["baseline"]]
        f = [v["decode_median_s"] for v in arms["fused"]]
        speedup = statistics.median(b) / statistics.median(f)
        p = welch_p(b, f)
        bt = [v["total_tps"] for v in arms["baseline"]]
        ft = [v["total_tps"] for v in arms["fused"]]
        tp_gain = (statistics.median(ft) / statistics.median(bt) - 1) * 100
        sig = "significant" if p < 0.05 else "n.s."
        print(f"  -> decode {speedup:.4f}x   throughput {tp_gain:+.2f}%   "
              f"p={p:.3f} ({sig})")
        # The increment over a main-equivalent baseline is the number to quote.
        # Against raw main this change also collects the rank-guard fix, which
        # is PR #32670's and already in flight upstream.
        inc = None
        if "fused_norm_only" in arms:
            pr = [v["decode_median_s"] for v in arms["fused_norm_only"]]
            prt = [v["total_tps"] for v in arms["fused_norm_only"]]
            inc = statistics.median(pr) / statistics.median(f)
            inc_tp = (statistics.median(ft) / statistics.median(prt) - 1) * 100
            inc_p = welch_p(pr, f)
            print(f"  -> vs fused-norm-only:  decode {inc:.4f}x  "
                  f"throughput {inc_tp:+.2f}%  p={inc_p:.3f} "
                  f"({'significant' if inc_p < 0.05 else 'n.s.'})   <-- the increment")

        rows.append(dict(regime=name, **cfg,
                         baseline_decode_ms=round(statistics.median(b) * 1000, 4),
                         fused_decode_ms=round(statistics.median(f) * 1000, 4),
                         decode_speedup=round(speedup, 4),
                         throughput_gain_pct=round(tp_gain, 3),
                         welch_p=round(p, 5), significant=bool(p < 0.05),
                         fused_norm_only_decode_ms=(round(statistics.median(pr) * 1000, 4)
                                            if inc else None),
                         increment_vs_fused_norm_only=(round(inc, 4) if inc else None),
                         increment_throughput_pct=(round(inc_tp, 3) if inc else None),
                         increment_p=(round(inc_p, 5) if inc else None),
                         reps=a.reps))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            dict(model=a.model, gpu=a.gpu, reps=a.reps, regimes=rows), indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
