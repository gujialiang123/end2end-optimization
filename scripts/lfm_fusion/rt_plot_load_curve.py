#!/usr/bin/env python3
"""Plot the Tool-Agent arrival-load sweep.

One figure, two y-axes, because the whole point is that the two curves behave
differently: the latency saving is there at every load, while the throughput
gain only appears once the server stops being arrival-limited.

  python scripts/lfm_fusion/rt_plot_load_curve.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results/2026-08-07_real_trace_study"


def main() -> None:
    rows = json.loads((OUT / "toolagent_load_sweep.json").read_text())
    x = [r["offered_req_s"] for r in rows]
    labels = [r["label"] for r in rows]
    thr = [r["throughput_gain_pct"] for r in rows]
    ttft = [r.get("ttft_p50_ms_gain_pct", 0) for r in rows]
    e2e = [r.get("e2e_mean_ms_gain_pct", 0) for r in rows]
    queued = [r["server_queue_peak"] for r in rows]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.15]})

    xi_ = list(range(len(x)))
    ax.plot(xi_, thr, "o-", color="#1f77b4", lw=2.2, ms=8,
            label="throughput gain (higher is better)")
    ax.plot(xi_, [-v for v in ttft], "s--", color="#d62728", lw=2.0, ms=7,
            label="TTFT p50 reduction")
    ax.plot(xi_, [-v for v in e2e], "^--", color="#ff7f0e", lw=2.0, ms=7,
            label="E2E mean reduction")
    ax.axhline(0, color="0.6", lw=0.8)
    for xi, ti in zip(xi_, thr):
        ax.annotate(f"{ti:+.2f}%", (xi, ti), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=9, color="#1f77b4")
    ax.set_ylabel("improvement from L3 (%)")
    ax.set_title("LFM2.5, Mooncake Tool-Agent replay: the same kernel work,\n"
                 "measured as latency at low load and as throughput at "
                 "saturation", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # plot against index, not the raw rate: 5.6/7.1/10.7 crowd together on a
    # linear axis and the labels overlap
    ax2.bar(range(len(x)), queued, color="#7f7f7f", width=0.5)
    ax2.set_xticks(range(len(x)))
    ax2.set_xticklabels([f"{li}\n{xi:.1f} req/s" for li, xi in zip(labels, x)],
                        fontsize=9)
    ax2.set_ylabel("peak queued\nrequests")
    ax2.set_xlabel("arrival load (offered rate comes from the trace timestamps)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"toolagent_load_curve.{ext}", dpi=150)
    print(f"wrote {OUT}/toolagent_load_curve.{{png,svg}}")


if __name__ == "__main__":
    main()
