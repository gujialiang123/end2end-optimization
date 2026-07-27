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
ARM_ORDER = ["scale", "norm", "norm+scale"]
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
    csv = L.RESULTS / "processed" / "fusion_ab.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df = df[(df.metric == "request_throughput") & (df.arm != "baseline")]
    if df.empty:
        return

    regimes = [r for r in REGIME_LABEL if r in set(df.regime)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(regimes))
    w = 0.26
    colours = {"scale": "#7fcdbb", "norm": "#41b6c4", "norm+scale": "#225ea8"}
    for i, arm in enumerate(ARM_ORDER):
        vals, errs = [], []
        for rg in regimes:
            s = df[(df.regime == rg) & (df.arm == arm)]
            vals.append(float(s.gain_pct.iloc[0]) if len(s) else 0.0)
            errs.append(100 * float(s.ci95.iloc[0]) / float(s.baseline_mean.iloc[0])
                        if len(s) else 0.0)
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, w, yerr=errs,
                      capsize=3, label=arm, color=colours[arm])
        for b, v, rg in zip(bars, vals, regimes):
            s = df[(df.regime == rg) & (df.arm == arm)]
            star = "" if not len(s) or s.verdict.iloc[0] != "improvement" else "*"
            ax.text(b.get_x() + b.get_width() / 2, v + 0.12,
                    f"{v:+.2f}{star}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([REGIME_LABEL[r] for r in regimes])
    ax.set_ylabel("end-to-end request throughput vs baseline (%)")
    ax.set_title("Closing the fusion gaps: decode gains ~4 %, long prefill ~1 %\n"
                 "(5 reps, Welch t vs baseline; * = p < 0.05)", fontsize=10)
    ax.legend(fontsize=9, title="LFM_FUSION_PATCH")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fusion_e2e_by_regime")


if __name__ == "__main__":
    fig_audit()
    fig_e2e()
