#!/usr/bin/env python3
"""Plot L3's effect across every real / agentic workload, both metrics.

Two panels rather than one chart with two bars per row, because the two metrics
have different signs and very different magnitudes, and overlaying them is what
made the original table easy to misread.

  python scripts/lfm_fusion/rt_plot_workload_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results/2026-08-07_real_trace_study"


def main() -> None:
    rows = json.loads((OUT / "real_workload_ablation.json").read_text())
    rows = rows[::-1]                      # top-down reading order
    names = [r["workload"] for r in rows]
    thr = [r["throughput_gain_pct"] for r in rows]
    e2e = [-r.get("e2e_mean_ms_gain_pct", 0) for r in rows]
    y = range(len(rows))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)

    a1.barh(y, e2e, color="#ff7f0e")
    a1.set_yticks(list(y))
    a1.set_yticklabels(names, fontsize=9)
    a1.set_xlabel("E2E mean latency reduction (%)")
    a1.set_title("Latency: every workload improves", fontsize=11)
    a1.grid(alpha=0.3, axis="x")
    for i, v in zip(y, e2e):
        a1.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8.5)

    a2.barh(y, thr, color="#1f77b4")
    a2.set_xlabel("throughput gain (%)")
    a2.set_title("Throughput: only where requests queue", fontsize=11)
    a2.grid(alpha=0.3, axis="x")
    for i, v in zip(y, thr):
        a2.text(v + 0.05, i, f"+{v:.2f}%", va="center", fontsize=8.5)

    fig.suptitle("LFM2.5 + L3 on real and agentic workloads "
                 "(counterbalanced, n=16 per arm)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"real_workload_matrix.{ext}", dpi=150)
    print(f"wrote {OUT}/real_workload_matrix.{{png,svg}}")


if __name__ == "__main__":
    main()
