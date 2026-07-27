#!/usr/bin/env python3
"""Build the K1 backend-comparison table from the raw per-run JSON.

Covers every model under `results/regime_kernel/backends/`, so the LFM2.5 runs
and the Qwen cross-model validation land in one table. Ratios are always taken
against the `auto` backend within the same (model, regime) cell, because `auto`
is what a user gets without touching anything.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rk_lib as L

BACKENDS = ["auto", "triton", "triton_kernel", "flashinfer_cutlass"]
SRC = L.RESULTS / "backends"
OUT = L.RESULTS / "processed" / "backend_comparison_all.csv"


def ci95(v):
    return 1.96 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def main():
    rows = []
    for runs_file in sorted(SRC.glob("*/*/backend_runs.json")):
        regime = runs_file.parent.name
        model = runs_file.parent.parent.name
        data = json.loads(runs_file.read_text())
        if not data:
            continue
        base = None
        for b in BACKENDS:
            thr = [r["request_throughput"] for r in data if r.get("backend") == b]
            if not thr:
                continue
            m = st.mean(thr)
            if b == "auto":
                base = m
            tpot = [r["p95_tpot_ms"] for r in data
                    if r.get("backend") == b and "p95_tpot_ms" in r]
            rows.append(dict(
                model=model, regime=regime, backend=b,
                thr=round(m, 4), ci95=round(ci95(thr), 4),
                ratio_vs_auto=round(m / base, 4) if base else "",
                n=len(thr),
                tpot_p95=round(st.mean(tpot), 4) if tpot else ""))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")

    models = sorted({r["model"] for r in rows})
    regimes = sorted({r["regime"] for r in rows})
    print(f"\n{'backend':22s}" + "".join(
        f"{m[:4]}/{rg.split('_')[0]:>4s}" .rjust(14) for rg in regimes for m in models))
    for b in BACKENDS:
        line = f"{b:22s}"
        for rg in regimes:
            for m in models:
                hit = [r for r in rows if r["model"] == m
                       and r["regime"] == rg and r["backend"] == b]
                line += (f"{hit[0]['ratio_vs_auto']:>14.4f}" if hit
                         else f"{'-':>14s}")
        print(line)


if __name__ == "__main__":
    main()
