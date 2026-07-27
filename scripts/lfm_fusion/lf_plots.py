#!/usr/bin/env python3
"""Two figures for the LFM2.5 fusion study.

fig1 — the audit: fusion-gap kernel counts, LFM2.5 against the Qwen control.
       Counts rather than time, because the count is the structural finding
       (48 = 2 residual adds x 24 layers) and it is what an agent can check.
fig2 — the end-to-end gain by regime and arm, with 95 % CI bars.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L

OUT = L.RESULTS / "plots"
GAPS = ["unfused_rmsnorm", "residual_add", "gating_mul", "layout_copy"]
LABEL = {"unfused_rmsnorm": "un-fused\nRMSNorm", "residual_add": "standalone\nresidual add",
         "gating_mul": "gating\nmultiply", "layout_copy": "layout\ncopy"}
ARM_ORDER = ["norm+scale", "conv", "norm+scale+conv"]
REGIME_LABEL = {"A_low_batch_decode": "A\nlow-batch decode",
                "B_concurrent_decode": "B\nconcurrent decode",
                "C_long_prefill": "C\nlong prefill"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/name}.png")


def fig_audit():
    rows = []
    for f in sorted((L.RESULTS / "audit").glob("*/audit.json")):
        d = json.loads(f.read_text())
        for stage, v in d["stages"].items():
            g = {x["gap"]: x for x in v["fusion_gaps"]}
            for gap in GAPS:
                rows.append(dict(model=d["model"], regime=d["regime"],
                                 stage=stage, gap=gap,
                                 calls=g.get(gap, {}).get("calls", 0),
                                 pct=g.get(gap, {}).get("pct_of_kernel_time", 0.0)))
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # kernel counts are identical across regimes (they are structural), so take
    # the max per (model, gap) — showing the per-forward count once
    piv = df.groupby(["model", "gap"])["calls"].max().unstack()[GAPS]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(GAPS))
    w = 0.36
    for i, (model, colour) in enumerate([("lfm25", "#c0392b"), ("qwen", "#2c7fb8")]):
        if model not in piv.index:
            continue
        vals = piv.loc[model].values
        bars = ax.bar([xi + (i - 0.5) * w for xi in x], vals, w,
                      label={"lfm25": "LFM2.5-8B-A1B (24 layers, 18 conv)",
                             "qwen": "Qwen3-30B-A3B (control)"}[model],
                      color=colour)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, str(int(v)),
                    ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[g] for g in GAPS])
    ax.set_ylabel("kernel launches per forward pass")
    ax.set_title("Fusion gaps are architecture-specific, not framework-wide\n"
                 "(Qwen: every hot path already fused — LFM2.5: 61 + 48 + 36 stray kernels)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fusion_gaps_by_model")


def fig_e2e():
    csv = L.RESULTS / "processed" / "fusion_ab_conv.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df = df[(df.metric == "request_throughput") & (df.arm != "baseline")]
    if df.empty:
        return

    regimes = [r for r in REGIME_LABEL
               if any(str(x).endswith(r) for x in df.regime)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(regimes))
    w = 0.26
    colours = {"norm+scale": "#41b6c4", "conv": "#fdae61",
               "norm+scale+conv": "#225ea8", "scale": "#7fcdbb", "norm": "#a1dab4"}
    for i, arm in enumerate(ARM_ORDER):
        vals, errs = [], []
        for rg in regimes:
            s = df[(df.regime.str.endswith(rg)) & (df.arm == arm)]
            vals.append(float(s.gain_pct.iloc[0]) if len(s) else 0.0)
            errs.append(100 * float(s.ci95.iloc[0]) / float(s.baseline_mean.iloc[0])
                        if len(s) else 0.0)
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, w, yerr=errs,
                      capsize=3, label=arm, color=colours[arm])
        for b, v, rg in zip(bars, vals, regimes):
            s = df[(df.regime.str.endswith(rg)) & (df.arm == arm)]
            star = "" if not len(s) or s.verdict.iloc[0] != "improvement" else "*"
            ax.text(b.get_x() + b.get_width() / 2, v + 0.12,
                    f"{v:+.2f}{star}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([REGIME_LABEL[r] for r in regimes])
    ax.set_ylabel("end-to-end request throughput vs baseline (%)")
    ax.set_title("Two complementary fusions: norm/scale is decode-weighted,\n"
                 "the ShortConv kernel is prefill-only "
                 "(6 reps, Welch t vs baseline; * = p < 0.05)", fontsize=10)
    ax.legend(fontsize=9, title="LFM_FUSION_PATCH", loc="upper left",
              framealpha=0.95)
    ax.set_ylim(top=max(6.0, ax.get_ylim()[1] * 1.35))
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fusion_e2e_by_regime")


def fig_final():
    """Final stack by regime, plus the sub-additivity check."""
    import pandas as pd
    f = L.RESULTS / "processed" / "fusion_ab_all.csv"
    fa = L.RESULTS / "processed" / "fusion_ab_allA.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    if fa.exists():
        df = pd.concat([df, pd.read_csv(fa)], ignore_index=True)
    df = df[(df.metric == "request_throughput") & (df.arm != "baseline")]
    arms = ["qkrope", "gate+idx", "norm+scale+conv", "all"]
    colours = {"qkrope": "#e07b39", "gate+idx": "#bbbbbb",
               "norm+scale+conv": "#41b6c4", "all": "#225ea8"}
    regimes = list(REGIME_LABEL)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    x = range(len(regimes))
    w = 0.2
    for i, arm in enumerate(arms):
        vals, errs = [], []
        for rg in regimes:
            s_ = df[(df.regime.str.endswith(rg)) & (df.arm == arm)]
            vals.append(float(s_.gain_pct.iloc[0]) if len(s_) else 0.0)
            errs.append(100 * float(s_.ci95.iloc[0]) / float(s_.baseline_mean.iloc[0])
                        if len(s_) else 0.0)
        bars = ax.bar([xi + (i - 1.5) * w for xi in x], vals, w, yerr=errs,
                      capsize=3, label=arm, color=colours[arm])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:+.2f}",
                    ha="center", fontsize=7.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([REGIME_LABEL[r] for r in regimes])
    ax.set_ylabel("end-to-end request throughput vs baseline (%)")
    ax.set_title("LFM2.5 fusion stack: +4.7 to +5.5 % on every regime\n"
                 "(6 reps, Welch t; all bold arms p<0.005)", fontsize=10)
    ax.set_ylim(top=7.4)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    # sub-additivity
    parts = {"A_low_batch_decode": 4.82, "B_concurrent_decode": 9.72,
             "C_long_prefill": 5.86}
    meas = {"A_low_batch_decode": 4.74, "B_concurrent_decode": 5.54,
            "C_long_prefill": 5.12}
    xs = range(len(regimes))
    ax2.bar([i - 0.19 for i in xs], [parts[r] for r in regimes], 0.38,
            label="sum of parts measured\nseparately", color="#cccccc",
            edgecolor="#888")
    ax2.bar([i + 0.19 for i in xs], [meas[r] for r in regimes], 0.38,
            label="measured together", color="#225ea8")
    for i, r in enumerate(regimes):
        ax2.text(i, max(parts[r], meas[r]) + 0.2,
                 f"{meas[r]/parts[r]:.2f}x", ha="center", fontsize=9,
                 fontweight="bold")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([r.split("_")[0] for r in regimes])
    ax2.set_ylabel("gain (%)")
    ax2.set_ylim(top=13.5)
    ax2.set_title("Wins that remove the same KIND of\ncost do not add up",
                  fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    save(fig, "fusion_final_stack")


def fig_crossover():
    """Isolated kernel speedup vs token count — why the shape guard exists."""
    f = L.RESULTS / "microbench" / "shortconv_bench.json"
    if not f.exists():
        return
    import json
    rows = [r for r in json.loads(f.read_text())["rows"] if r.get("status") == "ok"]
    if len(rows) < 3:
        return
    T = [r["T"] for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax.plot(T, [r["in_speedup"] for r in rows], "o-", label="input side\n(chunk+gate+transpose)")
    ax.plot(T, [r["out_speedup"] for r in rows], "s-", label="output side\n(transpose+gate)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axvline(2048, color="#c0392b", lw=1.2, ls=":",
               label="shape guard (T=2048)")
    ax.set_xscale("log"); ax.set_xlabel("tokens in the forward pass (T)")
    ax.set_ylabel("speedup vs stock PyTorch")
    ax.set_title("Fused kernels only pay off above T~2048", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2.plot(T, [r["in_stock_gbs"] for r in rows], "o--", color="#999",
             label="stock, input")
    ax2.plot(T, [r["in_fused_gbs"] for r in rows], "o-", color="#225ea8",
             label="fused, input")
    ax2.plot(T, [r["out_stock_gbs"] for r in rows], "s--", color="#c9a227",
             label="stock, output")
    ax2.plot(T, [r["out_fused_gbs"] for r in rows], "s-", color="#e07b39",
             label="fused, output")
    ax2.axhline(4800, color="#c0392b", lw=1.0, ls=":", label="H200 HBM peak")
    ax2.set_xscale("log"); ax2.set_xlabel("tokens in the forward pass (T)")
    ax2.set_ylabel("achieved bandwidth (GB/s)")
    ax2.set_title("The defect is coalescing, not traffic:\n"
                  "17 % of peak -> ~70 %", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right"); ax2.grid(alpha=0.3)
    save(fig, "shortconv_crossover")


if __name__ == "__main__":
    fig_audit()
    fig_e2e()
    fig_crossover()
    fig_final()
