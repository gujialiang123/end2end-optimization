#!/usr/bin/env python3
"""Slide-ready figures for the 2026-07-24 serving-ceiling campaign (Phase 9).

Style: white background, dark-navy text, muted blue/green/orange accents,
large fonts, source path in a small footer. PNG + SVG for every figure.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

NAVY = "#1b2a49"
BLUE = "#3d6fb4"
GREEN = "#3f8f6b"
ORANGE = "#d08428"
GREY = "#b9c0cc"
RED = "#b3453c"

WORKLOADS = ["R_short_decode", "R_medium_balanced", "R_long_prefill",
             "R_concurrent_decode", "shared_prefix", "tool_agent"]
NICE = {"R_short_decode": "short decode", "R_medium_balanced": "medium balanced",
        "R_long_prefill": "long prefill", "R_concurrent_decode": "concurrent decode",
        "shared_prefix": "shared-prefix", "tool_agent": "tool-agent"}


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": "#8a94a6",
        "xtick.color": NAVY, "ytick.color": NAVY,
        "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.labelsize": 13, "legend.frameon": False,
        "axes.grid": True, "grid.color": "#e6e9ef", "grid.linewidth": 0.9,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, outdir: Path, name: str, source: str):
    fig.text(0.005, 0.004, f"source: {source}", fontsize=7.5, color="#7c8698")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=190, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("wrote", outdir / f"{name}.png")


# ---------------------------------------------------------------- 1. overview
def fig_search_space(outdir, meta):
    style()
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axis("off")
    ax.set_title("Serving-configuration search: what we enumerate and what we measure",
                 loc="left", pad=18)
    ks = meta["knobs"]
    blocks = [
        ("Search", [f"sampler: {meta['sampler']}",
                    f"search space: {meta['total']} unique configurations",
                    "warm start: none  ·  no seeded cookbook  ·  no reused study"]),
        ("Knobs", [f"max_running_requests: {ks['cap']}",
                   f"chunked_prefill_size: {ks['chunk']}",
                   f"schedule_policy: {ks['policy']}",
                   f"mem_fraction_static: {ks['mem']}"]),
        ("Frozen (not tuned)", ["model · dtype · TP  ·  attention backend (fa3)",
                                "MoE runner backend  ·  CUDA Graph (on)",
                                "speculative decoding  ·  kernel configuration"]),
        ("Workloads (6)", [" · ".join(NICE[w] for w in WORKLOADS[:3]),
                           " · ".join(NICE[w] for w in WORKLOADS[3:])]),
        ("Metrics per cell", ["request / input / output / total throughput",
                              "TTFT, TPOT, E2E  —  mean, p50, p95, p99",
                              "raw per-request records (parquet)"]),
    ]
    y = 0.90
    for title, lines in blocks:
        ax.text(0.01, y, title, fontsize=14, fontweight="bold", color=BLUE,
                transform=ax.transAxes)
        y -= 0.062
        for ln in lines:
            ax.text(0.035, y, ln, fontsize=12.5, color=NAVY, transform=ax.transAxes)
            y -= 0.055
        y -= 0.022
    ax.text(0.62, 0.90, f"models: {meta['models']}", fontsize=13,
            fontweight="bold", color=GREEN, transform=ax.transAxes)
    ax.text(0.62, 0.845, f"coverage runs: {meta['coverage_runs']}", fontsize=12.5,
            color=NAVY, transform=ax.transAxes)
    ax.text(0.62, 0.795, f"validation repeats: {meta['reps']}", fontsize=12.5,
            color=NAVY, transform=ax.transAxes)
    save(fig, outdir, "search_space_overview", meta["source"])


# ------------------------------------------------------- 2. result heat matrix
def fig_result_heatmap(outdir, sm: pd.DataFrame, source):
    style()
    models = sorted(sm.model.unique())
    cols = [("d_request_throughput", "req/s"),
            ("d_output_throughput", "out tok/s"),
            ("d_ttft_p95", "TTFT p95"), ("d_tpot_p95", "TPOT p95"),
            ("d_e2e_p95", "E2E p95")]
    fig, axes = plt.subplots(1, len(models), figsize=(8.6 * len(models), 5.6),
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        d = sm[sm.model == model].set_index("workload").reindex(WORKLOADS)
        M = d[[c for c, _ in cols]].to_numpy(dtype=float) * 100
        vmax = max(5.0, np.nanmax(np.abs(M)))
        im = ax.imshow(M, cmap="RdYlGn", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        ax.set_xticks(range(len(cols)), [n for _, n in cols], fontsize=11.5)
        ax.set_yticks(range(len(WORKLOADS)), [NICE[w] for w in WORKLOADS], fontsize=11.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=11.5,
                        color="black", fontweight="bold")
        # classification printed OUTSIDE the axes so nothing overlaps the cells
        for i, wl in enumerate(WORKLOADS):
            if wl in d.index and isinstance(d.loc[wl, "classification"], str):
                ax.text(len(cols) - 0.35, i, d.loc[wl, "classification"],
                        fontsize=10, color=NAVY, va="center", ha="left",
                        fontweight="bold")
        ax.set_xlim(-0.5, len(cols) + 1.35)
        ax.set_title(f"{model} — best-throughput config vs cookbook")
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.13,
                     label="change (%), green = better")
    fig.suptitle("Serving tuning: gains are regime-specific, not universal",
                 fontsize=17, fontweight="bold", color=NAVY, y=1.03)
    save(fig, outdir, "full_result_matrix_heatmap", source)


# ------------------------------------------------------------ 3. Pareto grid
def fig_pareto_grid(outdir, deltas: pd.DataFrame, model: str, source):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
    for ax, wl in zip(axes.ravel(), WORKLOADS):
        sub = deltas[deltas.workload == wl]
        if sub.empty:
            ax.axis("off"); continue
        ax.scatter(sub.ttft_p95_ms, sub.output_throughput, s=26, c=GREY,
                   edgecolors="none", label="all configs", zorder=2)
        par = sub[sub.pareto_ttft_outthr]
        ax.scatter(par.ttft_p95_ms, par.output_throughput, s=52, c=BLUE,
                   edgecolors="white", linewidths=0.6, label="Pareto", zorder=3)
        cb = sub[sub.is_cookbook == 1]
        if len(cb):
            ax.scatter(cb.ttft_p95_ms, cb.output_throughput, s=190, marker="*",
                       c=ORANGE, edgecolors=NAVY, linewidths=0.8,
                       label="cookbook", zorder=5)
        bw = sub.loc[sub.request_throughput.idxmax()]
        ax.scatter([bw.ttft_p95_ms], [bw.output_throughput], s=125, marker="D",
                   facecolors="none", edgecolors=GREEN, linewidths=2.1,
                   label="throughput winner", zorder=4)
        ax.set_title(NICE[wl])
        ax.set_xlabel("TTFT p95 (ms) — lower is better →")
        ax.set_ylabel("output token throughput (tok/s)")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    axes[0, 0].legend(loc="lower left", fontsize=10.5)
    fig.suptitle(f"{model} — every configuration, per regime  "
                 "(preferred direction: upper-LEFT)",
                 fontsize=17, fontweight="bold", color=NAVY, y=1.0)
    fig.tight_layout()
    save(fig, outdir, f"per_regime_pareto_grid_{model}", source)


# --------------------------------------------------------- 4-7. transfer maps
def fig_transfer(outdir, path: Path, metric: str, model: str, source):
    style()
    m = pd.read_csv(path, index_col=0)
    m = m.reindex(columns=[c for c in WORKLOADS if c in m.columns])
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    M = m.to_numpy(dtype=float)
    lim = max(0.25, np.nanmax(np.abs(M - 1)))
    im = ax.imshow(M, cmap="RdYlGn", norm=TwoSlopeNorm(1.0, 1 - lim, 1 + lim),
                   aspect="auto")
    ax.set_xticks(range(m.shape[1]), [NICE.get(c, c) for c in m.columns],
                  rotation=18, ha="right", fontsize=11.5)
    ax.set_yticks(range(m.shape[0]),
                  [i.replace("_winner", " winner").replace("_", " ") for i in m.index],
                  fontsize=11.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}×", ha="center", va="center",
                        fontsize=11.5, fontweight="bold", color="black")
    ax.set_xlabel("applied to target workload")
    ax.set_ylabel("source configuration")
    ax.set_title(f"{model} — transfer of {metric} (ratio vs target cookbook; "
                 ">1.00 = better)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="ratio vs target cookbook")
    save(fig, outdir, f"transfer_matrix_{metric}_{model}", source)


# ------------------------------------------------------- 8. gain distribution
def fig_gain_distribution(outdir, gd: pd.DataFrame, source):
    style()
    order = ["WIN", "TRADE-OFF", "FLAT", "REGRESSION"]
    colors = {"WIN": GREEN, "TRADE-OFF": ORANGE, "FLAT": GREY, "REGRESSION": RED}
    models = sorted(gd.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.6 * len(models), 5.4),
                             squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        d = gd[gd.model == model]
        tab = (d.groupby(["workload", "cls"]).size().unstack(fill_value=0)
               .reindex(index=WORKLOADS, columns=order, fill_value=0))
        frac = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0) * 100
        bottom = np.zeros(len(WORKLOADS))
        for c in order:
            v = frac[c].to_numpy()
            ax.barh(range(len(WORKLOADS)), v, left=bottom, color=colors[c],
                    label=c, height=0.66)
            for i, (val, b) in enumerate(zip(v, bottom)):
                if val >= 7:
                    ax.text(b + val / 2, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=10.5, color="white", fontweight="bold")
            bottom += np.nan_to_num(v)
        ax.set_yticks(range(len(WORKLOADS)), [NICE[w] for w in WORKLOADS])
        ax.set_xlabel("share of evaluated configurations (%)")
        ax.set_title(f"{model}")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
    axes[0][0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=4,
                      fontsize=11.5)
    fig.suptitle("Outcome of every configuration vs its cookbook baseline",
                 fontsize=17, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, outdir, "gain_distribution", source)


# ------------------------------------------------------ 9. cross-model panel
def fig_cross_model(outdir, sm: pd.DataFrame, source):
    style()
    models = sorted(sm.model.unique())
    if len(models) < 2:
        return
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(WORKLOADS)); w = 0.38
    for k, (model, col) in enumerate(zip(models, [BLUE, GREEN])):
        d = sm[sm.model == model].set_index("workload").reindex(WORKLOADS)
        v = d.d_request_throughput.to_numpy(dtype=float) * 100
        ax.bar(x + (k - 0.5) * w, v, w, color=col, label=model)
        for xi, vi in zip(x + (k - 0.5) * w, v):
            if not np.isnan(vi):
                ax.text(xi, vi + (1.2 if vi >= 0 else -3.0), f"{vi:+.1f}%",
                        ha="center", fontsize=10.5, color=NAVY)
    ax.axhline(0, color=NAVY, linewidth=1.1)
    ax.set_xticks(x, [NICE[w_] for w_ in WORKLOADS], fontsize=12)
    ax.set_ylabel("request-throughput gain of the\nbest config vs that model's cookbook (%)")
    ax.set_title("Same search space, same protocol — the reachable gain is "
                 "model- and regime-dependent")
    ax.legend(fontsize=12)
    fig.tight_layout()
    save(fig, outdir, "cross_model_same_strategy", source)


# --------------------------------------------------------- 10. coverage curve
def fig_coverage(outdir, deltas: pd.DataFrame, model: str, source):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
    for ax, wl in zip(axes.ravel(), WORKLOADS):
        sub = deltas[deltas.workload == wl].sort_values("config_id")
        if sub.empty:
            ax.axis("off"); continue
        v = sub.request_throughput.to_numpy(dtype=float)
        ax.plot(range(1, len(v) + 1), np.maximum.accumulate(v), color=BLUE, lw=2.2,
                label="best so far")
        ax.scatter(range(1, len(v) + 1), v, s=13, c=GREY, zorder=1, label="each config")
        cb = sub[sub.is_cookbook == 1]
        if len(cb):
            ax.axhline(cb.request_throughput.iloc[0], color=ORANGE, ls="--", lw=1.8,
                       label="cookbook")
        ax.set_title(NICE[wl]); ax.set_xlabel("configurations evaluated")
        ax.set_ylabel("request throughput (req/s)")
    axes[0, 0].legend(fontsize=10)
    fig.suptitle(f"{model} — coverage order is a deterministic grid, "
                 "so this is coverage (not adaptive convergence)",
                 fontsize=16, fontweight="bold", color=NAVY, y=1.0)
    fig.tight_layout()
    save(fig, outdir, f"coverage_{model}", source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", required=True)
    args = ap.parse_args()
    root = Path(args.outroot)
    plots = root / "plots"
    src = str(root)

    sm = pd.read_csv(root / "summary_matrix.csv")
    meta_ss = (root / "search_space.yaml").read_text()
    import re
    knobs = {k: re.search(rf"  {k}: (\[.*\])", meta_ss).group(1)
             for k in ("cap", "chunk", "policy", "mem")}
    total = re.search(r"total_configurations: (\d+)", meta_ss).group(1)
    models = sorted(sm.model.unique())
    fig_search_space(plots, dict(
        sampler="Optuna GridSampler (full enumeration)", knobs=knobs, total=total,
        models=" · ".join(models),
        coverage_runs=f"{int(sm.n_configs_evaluated.max())} configs x 6 workloads x "
                      f"{len(models)} models",
        reps=f"{int(sm.repeat_count.max())} (validation pass)", source=src))
    fig_result_heatmap(plots, sm, src)
    fig_cross_model(plots, sm, src)

    gds = pd.concat([pd.read_csv(p) for p in
                     glob.glob(str(root / "analysis" / "*" / "gain_distribution.csv"))],
                    ignore_index=True)
    fig_gain_distribution(plots, gds, src)

    for model in models:
        d = root / "analysis" / model
        deltas = pd.read_csv(d / "per_workload_deltas.csv")
        fig_pareto_grid(plots, deltas, model, src)
        fig_coverage(plots, deltas, model, src)
        for metric in ("request_throughput", "output_throughput", "ttft_p95_ms",
                       "tpot_p95_ms", "e2e_p95_ms"):
            p = d / f"transfer_matrix_{metric}.csv"
            if p.exists():
                fig_transfer(plots, p, metric, model, src)


if __name__ == "__main__":
    main()
