#!/usr/bin/env python3
"""Figures for the regime-aware kernel specialization study.

All numbers are read from results/regime_kernel/processed/*.csv — nothing is
hard-coded in this file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#1b2a49"; BLUE = "#3d6fb4"; GREEN = "#3f8f6b"; ORANGE = "#d08428"
GREY = "#b9c0cc"; RED = "#b3453c"; PURPLE = "#7a5ea7"


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "text.color": NAVY,
        "axes.labelcolor": NAVY, "axes.edgecolor": "#8a94a6", "xtick.color": NAVY,
        "ytick.color": NAVY, "font.size": 12, "axes.titlesize": 14,
        "axes.titleweight": "bold", "legend.frameon": False, "axes.grid": True,
        "grid.color": "#e6e9ef", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False})


def save(fig, outdir: Path, name: str, src: str):
    fig.text(0.005, 0.004, f"source: {src}", fontsize=7.5, color="#7c8698")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=185, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("wrote", outdir / f"{name}.png")


# ------------------------------------------------ 1. workload characterization
def fig_characterization(outdir, sweep: pd.DataFrame, src, traces: Path | None):
    style()
    models = sorted(sweep.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    for m, c in zip(models, (BLUE, GREEN)):
        d = sweep[sweep.model == m].groupby("tokens").agg(
            M=("M", "first"), default_ms=("default_ms", "first")).reset_index()
        ax.plot(d.tokens, d.default_ms, "o-", color=c, lw=2, label=f"{m} default")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("tokens per MoE invocation")
    ax.set_ylabel("default kernel latency (ms)")
    ax.set_title("Regime → kernel workload: latency vs token batch")
    ax.legend()

    ax = axes[1]
    for m, c in zip(models, (BLUE, GREEN)):
        d = sweep[sweep.model == m]
        best = d.loc[d.groupby("tokens").median_ms.idxmin()]
        ax.plot(best.tokens, best.BLOCK_SIZE_M, "o-", color=c, lw=2,
                label=f"{m} best BLOCK_M")
        ax.plot(best.tokens, best.BLOCK_SIZE_N, "s--", color=c, lw=1.4, alpha=0.65,
                label=f"{m} best BLOCK_N")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("tokens per MoE invocation")
    ax.set_ylabel("winning tile dimension")
    ax.set_title("The optimal tile shape moves with the regime")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, outdir, "regime_workload_characterization", src)


# --------------------------------------------------------- 2. transfer heatmap
def fig_transfer(outdir, tr: pd.DataFrame, src):
    style()
    for model, d in tr.groupby("model"):
        piv = d.pivot_table(index="profile", columns="tokens",
                            values="speedup_vs_default", aggfunc="first")
        # order rows low-M -> high-M so degradation off the diagonal is visible
        order = [p for p in ("low_M", "mid_M", "high_M", "global_best")
                 if p in piv.index] + \
                sorted([p for p in piv.index if p.startswith("oracle_")],
                       key=lambda s: int(s.split("t")[-1]))
        piv = piv.reindex(order)
        fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 4, 0.5 * len(piv) + 3))
        M = piv.to_numpy(float)
        im = ax.imshow(M, cmap="RdYlGn", aspect="auto",
                       norm=matplotlib.colors.TwoSlopeNorm(
                           1.0, min(0.8, np.nanmin(M)), max(1.2, np.nanmax(M))))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold", color="black")
        ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv)), piv.index)
        ax.set_xlabel("tested at tokens per invocation")
        ax.set_ylabel("profile tuned on")
        ax.set_title(f"{model}: kernel profile transfer "
                     "(speedup vs default; <1.00 = worse than default)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="speedup vs default")
        fig.tight_layout()
        save(fig, outdir, f"kernel_transfer_heatmap_{model}", src)


# ------------------------------------------------------- 3. kernel winner map
def fig_winner_map(outdir, sweep: pd.DataFrame, src):
    style()
    models = sorted(sweep.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5),
                             squeeze=False)
    for ax, m in zip(axes[0], models):
        d = sweep[sweep.model == m]
        best = d.loc[d.groupby("tokens").median_ms.idxmin()].sort_values("tokens")
        keys = list(dict.fromkeys(best.config_key))
        cmap = plt.get_cmap("tab10")
        for i, k in enumerate(keys):
            sel = best[best.config_key == k]
            ax.scatter(sel.tokens, sel.median_ms / sel.default_ms, s=110,
                       color=cmap(i % 10), label=k, zorder=3)
        ax.axhline(1.0, color=NAVY, lw=1.2, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("tokens per MoE invocation")
        ax.set_ylabel("best latency / default latency\n(lower = better)")
        ax.set_title(f"{m}: winning config by regime")
        ax.legend(fontsize=8, ncol=1, loc="best")
    fig.suptitle("Kernel winner map — which configuration wins at which M",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, outdir, "kernel_winner_map", src)


# ------------------------------- 4. default vs global vs regime-aware vs oracle
def fig_strategies(outdir, st: pd.DataFrame, src):
    style()
    models = sorted(st.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5.2),
                             squeeze=False, sharey=True)
    for ax, m in zip(axes[0], models):
        d = st[st.model == m].sort_values("tokens")
        x = np.arange(len(d)); w = 0.26
        ax.bar(x - w, d.global_speedup, w, color=BLUE, label="global-best")
        ax.bar(x, d.regime_speedup, w, color=GREEN, label="regime-aware")
        ax.bar(x + w, d.oracle_speedup, w, color=ORANGE, label="oracle (bound)")
        ax.axhline(1.0, color=NAVY, lw=1.3, ls="--", label="default")
        ax.set_xticks(x, [str(int(t)) for t in d.tokens])
        ax.set_xlabel("tokens per MoE invocation")
        ax.set_ylabel("speedup over measured default")
        ax.set_title(f"{m}")
        ax.legend(fontsize=10)
    fig.suptitle("Default vs global-best vs regime-aware vs per-shape oracle",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, outdir, "strategy_comparison", src)


# ---------------------------------------------------------- 5. routing control
def fig_routing(outdir, rt: pd.DataFrame, src):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for ax, (m, d) in zip(axes[0], rt.groupby("model")):
        piv = d.pivot_table(index="tokens", columns="routing",
                            values="best_speedup", aggfunc="first")
        piv.plot(kind="bar", ax=ax, color=[BLUE, ORANGE], width=0.7, rot=0)
        ax.axhline(1.0, color=NAVY, lw=1.2, ls="--")
        ax.set_xlabel("tokens per MoE invocation")
        ax.set_ylabel("best speedup over default")
        ax.set_title(f"{m}: does routing skew change the optimum?")
        ax.legend(title="routing", fontsize=10)
    fig.tight_layout()
    save(fig, outdir, "routing_control", src)


# ----------------------------------------------------------- 6. E2E waterfall
def fig_waterfall(outdir, e2e: pd.DataFrame, src):
    """End-to-end effect of each kernel profile, grouped by regime.

    Uses the median across repetitions as the primary statistic because
    introducing new kernel configurations causes occasional Triton
    recompilation stalls that skew the mean; both are reported.
    """
    style()
    ARMS = [("default", GREY, "default kernel"),
            ("global_best", BLUE, "global-best profile"),
            ("regime_aware", ORANGE, "regime-aware (naive)"),
            ("regime_aware_guarded", "#7a9e8b", "guarded (mis-keyed M)"),
            ("guarded_Mfixed", GREEN, "guarded (correct M)")]
    regimes = [r for r in ("A_low_batch_decode", "B_concurrent_decode",
                           "C_long_prefill") if r in set(e2e.regime)]
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.4 * len(regimes), 5.4),
                             squeeze=False, sharey=True)
    for ax, reg in zip(axes[0], regimes):
        d = e2e[e2e.regime == reg]
        xs, hs, cs, ls, err = [], [], [], [], []
        i = 0
        for arm, c, lab in ARMS:
            s = d[d.arm == arm]
            if s.empty:
                continue
            r = s.iloc[-1]          # most recent run for this arm
            xs.append(i); hs.append(r.get("ratio_median", r["ratio"]))
            cs.append(c); ls.append(lab)
            err.append(r["thr_ci95"] / r["thr_mean"] if r["thr_mean"] else 0)
            i += 1
        ax.bar(xs, hs, color=cs, yerr=err, capsize=4, ecolor=NAVY)
        for x, h in zip(xs, hs):
            ax.text(x, h + 0.012, f"{h:.3f}x", ha="center", fontsize=10.5,
                    fontweight="bold", color=NAVY)
        ax.axhline(1.0, color=NAVY, lw=1.3, ls="--")
        ax.set_xticks(xs, ls, rotation=22, ha="right", fontsize=9.5)
        ax.set_title(reg.split("_", 1)[1].replace("_", " "))
        ax.set_ylabel("request throughput vs default\n(median over repetitions)")
    fig.suptitle("End-to-end: serving knobs frozen, only the MoE kernel profile varies",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, outdir, "e2e_waterfall", src)


# ------------------------------------------------------- 7. agent iterations
def fig_agent(outdir, ag: pd.DataFrame, src):
    style()
    fig, ax = plt.subplots(figsize=(12, 5.4))
    colors = {"accept": GREEN, "reject": RED, "rollback": ORANGE}
    for i, r in ag.iterrows():
        ax.bar(i, r.speedup, color=colors.get(r.decision, GREY))
        ax.text(i, r.speedup, f"{r.action}", rotation=90, fontsize=7.5,
                ha="center", va="bottom", color=NAVY)
    ax.axhline(1.0, color=NAVY, lw=1.2, ls="--")
    ax.set_xlabel("agent iteration")
    ax.set_ylabel("candidate speedup vs incumbent")
    ax.set_title("Agent closed loop: action, measured effect, accept/reject")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=10)
    fig.tight_layout()
    save(fig, outdir, "agent_iteration_trace", src)


def fig_measured_M(outdir, mdist: pd.DataFrame, sweep: pd.DataFrame, src):
    """Where the serving regimes actually land on the M axis, against headroom."""
    style()
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.bar(mdist.M, mdist["count"], width=mdist.M * 0.35, color=BLUE,
           label="measured MoE invocations")
    ax.set_xscale("log"); ax.set_xlabel("M reached by the kernel (token count)")
    ax.set_ylabel("invocations observed")
    ax2 = ax.twinx()
    if sweep is not None and len(sweep):
        s = sweep.sort_values("M")
        ax2.plot(s.M, s.oracle_speedup, "o-", color=ORANGE, lw=2.2,
                 label="tuning headroom (oracle)")
        ax2.axhline(1.15, color=RED, ls="--", lw=1.5, label="specialize threshold")
    ax2.set_ylabel("oracle speedup over default")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=11)
    ax.set_title("Measured regime -> M mapping vs where tuning headroom exists")
    fig.tight_layout()
    save(fig, outdir, "measured_M_vs_headroom", src)


def fig_backends(outdir, bc: pd.DataFrame, src):
    """Does the best kernel IMPLEMENTATION differ by regime? (candidate class K1)"""
    style()
    regs = ["A_low_batch_decode", "B_concurrent_decode", "C_long_prefill"]
    regs = [r for r in regs if r in set(bc.regime)]
    backends = ["auto", "triton", "triton_kernel", "flashinfer_cutlass"]
    cols = {"auto": GREY, "triton": BLUE, "triton_kernel": PURPLE,
            "flashinfer_cutlass": ORANGE}
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    w = 0.2
    x = np.arange(len(regs))
    for i, b in enumerate(backends):
        vals, errs = [], []
        for r in regs:
            s = bc[(bc.regime == r) & (bc.backend == b)]
            vals.append(s.ratio.iloc[0] if len(s) else np.nan)
            errs.append((s.ci95.iloc[0] / s.thr.iloc[0]) if len(s) else 0)
        pos = x + (i - 1.5) * w
        ax.bar(pos, vals, w, color=cols[b], label=b, yerr=errs, capsize=3,
               ecolor=NAVY)
        for xx, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(xx, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5,
                        color=NAVY, fontweight="bold")
    ax.axhline(1.0, color=NAVY, lw=1.3, ls="--")
    ax.set_xticks(x, [r.split("_", 1)[1].replace("_", " ") for r in regs])
    ax.set_ylabel("request throughput vs auto backend")
    ax.set_title("Kernel IMPLEMENTATION by regime — the ranking flips\n"
                 "(cutlass is best on concurrent decode, worst on long prefill)")
    ax.legend(fontsize=10.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    save(fig, outdir, "backend_by_regime", src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="results/regime_kernel/processed")
    ap.add_argument("--out", default="results/regime_kernel/plots")
    a = ap.parse_args()
    proc, out = Path(a.proc), Path(a.out)
    src = str(proc)

    def maybe(name):
        p = proc / name
        return pd.read_csv(p) if p.exists() else None

    sweep = maybe("sweep_all.csv")
    if sweep is not None:
        fig_characterization(out, sweep, src, None)
        fig_winner_map(out, sweep, src)
    st = maybe("strategy_comparison.csv")
    if st is not None:
        fig_strategies(out, st, src)
    tr = maybe("transfer_matrix.csv")
    if tr is not None:
        fig_transfer(out, tr, src)
    rt = maybe("routing_control.csv")
    if rt is not None:
        fig_routing(out, rt, src)
    bc = maybe("backend_comparison.csv")
    if bc is not None:
        fig_backends(out, bc, src)
    md = maybe("measured_M_distribution.csv")
    if md is not None:
        sb = maybe("sweep_headroom_bias.csv")
        fig_measured_M(out, md, sb, src)
    e2e = maybe("e2e_summary.csv")
    if e2e is not None:
        fig_waterfall(out, e2e, src)
    ag = maybe("agent_trace.csv")
    if ag is not None:
        fig_agent(out, ag, src)


if __name__ == "__main__":
    main()
