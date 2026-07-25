#!/usr/bin/env python3
"""Phase-6 figures for the alternative-objective study (PNG + SVG)."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#1b2a49"; BLUE = "#3d6fb4"; GREEN = "#3f8f6b"; ORANGE = "#d08428"
GREY = "#b9c0cc"; RED = "#b3453c"; PURPLE = "#7a5ea7"

WORKLOADS = ["R_short_decode", "R_medium_balanced", "R_long_prefill",
             "R_concurrent_decode", "shared_prefix", "tool_agent"]
NICE = {"R_short_decode": "short decode", "R_medium_balanced": "medium balanced",
        "R_long_prefill": "long prefill", "R_concurrent_decode": "concurrent decode",
        "shared_prefix": "shared-prefix", "tool_agent": "tool-agent"}
ROLES = ["cookbook", "request_throughput_best", "ttft_p95_best", "tpot_p95_best",
         "e2e_p95_best", "constrained_throughput_best_3pct",
         "maximin_balanced_best", "pareto_knee_candidate"]
RSHORT = {"cookbook": "cookbook", "request_throughput_best": "req-thr",
          "ttft_p95_best": "TTFT", "tpot_p95_best": "TPOT",
          "e2e_p95_best": "E2E", "constrained_throughput_best_3pct": "SLO-constr",
          "maximin_balanced_best": "maximin", "pareto_knee_candidate": "knee",
          "strict_all_metric_candidate": "strict-all"}
CLS_COLOR = {"WIN": GREEN, "STRICT_ALL_METRIC_WIN": "#2f7d55", "TRADE-OFF": ORANGE,
             "FLAT": GREY, "REGRESSION": RED, "NOT_VALIDATED": "#e3e6ec"}


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


# --------------------------------------------- 1. per-cell winner comparison
def fig_winner_comparison(outdir, v: pd.DataFrame, src):
    style()
    metrics = [("request_throughput_delta", "request thr"),
               ("ttft_p95_ms_delta", "TTFT p95"),
               ("tpot_p95_ms_delta", "TPOT p95"),
               ("e2e_p95_ms_delta", "E2E p95")]
    for (model, wl), g in v[v.objective_role.isin(ROLES)].groupby(["model", "workload"]):
        g = g.set_index("objective_role").reindex([r for r in ROLES if r in set(g.objective_role)])
        g = g.dropna(subset=["classification"])
        if g.empty:
            continue
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.6),
                                 sharey=True)
        y = np.arange(len(g))
        for ax, (col, title) in zip(axes, metrics):
            if col not in g:
                ax.axis("off"); continue
            vals = g[col].to_numpy(float) * 100
            lo = (g[col.replace("_delta", "_ci_lo")].to_numpy(float) * 100
                  if col.replace("_delta", "_ci_lo") in g else np.zeros_like(vals))
            hi = (g[col.replace("_delta", "_ci_hi")].to_numpy(float) * 100
                  if col.replace("_delta", "_ci_hi") in g else np.zeros_like(vals))
            colors = [GREEN if v_ > 0 else RED for v_ in np.nan_to_num(vals)]
            ax.barh(y, vals, color=colors, height=0.62)
            ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none",
                        ecolor=NAVY, elinewidth=1.2, capsize=3)
            ax.axvline(0, color=NAVY, lw=1.1)
            ax.set_title(title)
            ax.set_xlabel("improvement vs cookbook (%)\npositive = better")
        axes[0].set_yticks(y, [f"{RSHORT.get(i,i)}  (cfg {int(r.config_id)})"
                               for i, (_, r) in zip(g.index, g.iterrows())])
        fig.suptitle(f"{model} — {NICE.get(wl, wl)}: each objective picks a "
                     "different configuration", fontsize=15, fontweight="bold",
                     color=NAVY, y=1.03)
        fig.tight_layout()
        save(fig, outdir, f"objective_winner_comparison_{model}_{wl}", src)


# ------------------------------------------------------ 2. config role matrix
def fig_role_matrix(outdir, audit: pd.DataFrame, src):
    style()
    roles = ROLES + ["strict_all_metric_candidate"]
    p = (audit[audit.objective_role.isin(roles)]
         .pivot_table(index=["model", "workload"], columns="objective_role",
                      values="config_id", aggfunc="first"))
    p = p.reindex(columns=[r for r in roles if r in p.columns])
    idx = [(m, w) for m in sorted({i[0] for i in p.index}) for w in WORKLOADS
           if (m, w) in p.index]
    p = p.reindex(idx)
    fig, ax = plt.subplots(figsize=(1.45 * len(p.columns) + 4, 0.62 * len(p) + 3))
    # colour = how many DISTINCT configs that row uses (excluding cookbook)
    nun = p.drop(columns=["cookbook"], errors="ignore").nunique(axis=1)
    M = np.tile(nun.to_numpy(float)[:, None], (1, p.shape[1]))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", vmin=1, vmax=max(2, nun.max()))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            val = p.iat[i, j]
            if pd.isna(val):
                ax.text(j, i, "—", ha="center", va="center", color=NAVY)
            else:
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=11.5, fontweight="bold", color="black")
    ax.set_xticks(range(p.shape[1]), [RSHORT.get(c, c) for c in p.columns],
                  rotation=25, ha="right")
    ax.set_yticks(range(len(p)), [f"{m} · {NICE.get(w,w)}" for m, w in p.index])
    ax.set_title("Which configuration each objective selects (config_id)\n"
                 "cell colour = number of distinct configs chosen in that regime")
    fig.colorbar(im, ax=ax, shrink=0.8, label="distinct configs per regime")
    fig.tight_layout()
    save(fig, outdir, "config_role_matrix", src)


# ------------------------------------------- 3. no-regression feasibility
def fig_feasibility(outdir, v: pd.DataFrame, src):
    style()
    rows = []
    for (model, wl), g in v.groupby(["model", "workload"]):
        cls = set(g.classification.dropna())
        if "STRICT_ALL_METRIC_WIN" in cls:
            status = "all-metric win exists"
        elif "WIN" in cls:
            status = "win with guardrails held"
        elif "TRADE-OFF" in cls:
            status = "only trade-offs"
        else:
            status = "no validated improvement"
        rows.append(dict(model=model, workload=wl, status=status))
    d = pd.DataFrame(rows)
    order = ["all-metric win exists", "win with guardrails held",
             "only trade-offs", "no validated improvement"]
    cmap = {order[0]: "#2f7d55", order[1]: GREEN, order[2]: ORANGE, order[3]: GREY}
    models = sorted(d.model.unique())
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(d) + 2.4))
    ys, labels = [], []
    for k, (_, r) in enumerate(d.sort_values(["model", "workload"]).iterrows()):
        ax.barh(k, 1, color=cmap[r.status], height=0.7)
        ax.text(0.5, k, r.status, ha="center", va="center", color="white",
                fontweight="bold", fontsize=11)
        ys.append(k); labels.append(f"{r.model} · {NICE.get(r.workload, r.workload)}")
    ax.set_yticks(ys, labels)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.grid(False)
    ax.set_title("Is there a configuration that improves without regressing?\n"
                 "(validated, 5 repetitions, bootstrap 95 % CI)")
    fig.tight_layout()
    save(fig, outdir, "no_regression_feasibility", src)


# --------------------------------------- 4. Pareto with objective roles
def fig_pareto_roles(outdir, cov: pd.DataFrame, audit: pd.DataFrame, src):
    style()
    marks = [("request_throughput_best", "D", GREEN, "throughput winner"),
             ("ttft_p95_best", "^", BLUE, "TTFT winner"),
             ("tpot_p95_best", "v", PURPLE, "TPOT winner"),
             ("maximin_balanced_best", "s", ORANGE, "maximin balanced")]
    for model in sorted(cov.model.unique()):
        fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
        for ax, wl in zip(axes.ravel(), WORKLOADS):
            sub = cov[(cov.model == model) & (cov.workload == wl)]
            if sub.empty:
                ax.axis("off"); continue
            ax.scatter(sub.ttft_p95_ms, sub.output_throughput, s=20, c=GREY,
                       edgecolors="none", zorder=1, label="all 192 configs")
            cb = sub[sub.is_cookbook == 1]
            ax.scatter(cb.ttft_p95_ms, cb.output_throughput, s=200, marker="*",
                       c="#d94f43", edgecolors=NAVY, zorder=5, label="cookbook")
            sel = audit[(audit.model == model) & (audit.workload == wl)]
            for role, mk, col, lab in marks:
                rr = sel[sel.objective_role == role]
                if rr.empty:
                    continue
                cid = int(rr.iloc[0].config_id)
                pt = sub[sub.config_id == cid]
                ax.scatter(pt.ttft_p95_ms, pt.output_throughput, s=125, marker=mk,
                           facecolors="none", edgecolors=col, linewidths=2.2,
                           zorder=4, label=lab)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
            ax.set_title(NICE.get(wl, wl))
            ax.set_xlabel("TTFT p95 (ms) — lower is better →")
            ax.set_ylabel("output token throughput (tok/s)")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, fontsize=11.5,
                   bbox_to_anchor=(0.5, -0.015))
        fig.suptitle(f"{model} — different objectives land on different points of "
                     "the same frontier (preferred: upper-LEFT)",
                     fontsize=16, fontweight="bold", color=NAVY, y=1.0)
        fig.tight_layout()
        save(fig, outdir, f"pareto_with_objective_roles_{model}", src)


# ------------------------------------------- 5. outcome counts by objective
def fig_outcome_counts(outdir, v: pd.DataFrame, src):
    style()
    order = ["STRICT_ALL_METRIC_WIN", "WIN", "TRADE-OFF", "FLAT", "REGRESSION"]
    roles = [r for r in ROLES + ["strict_all_metric_candidate"] if r != "cookbook"]
    d = v[v.objective_role.isin(roles)]
    tab = (d.groupby(["objective_role", "classification"]).size()
           .unstack(fill_value=0).reindex(index=roles, columns=order, fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    bottom = np.zeros(len(tab))
    for c in order:
        vals = tab[c].to_numpy(float)
        ax.bar(range(len(tab)), vals, bottom=bottom, color=CLS_COLOR[c],
               label=c, width=0.66)
        for i, (val, b) in enumerate(zip(vals, bottom)):
            if val:
                ax.text(i, b + val / 2, int(val), ha="center", va="center",
                        color="white", fontweight="bold", fontsize=10.5)
        bottom += vals
    ax.set_xticks(range(len(tab)), [RSHORT.get(r, r) for r in tab.index],
                  rotation=18, ha="right")
    ax.set_ylabel("model × workload cells")
    ax.set_title("Validated outcome of each objective policy across all 12 regimes")
    ax.legend(ncol=5, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save(fig, outdir, "outcome_counts_by_objective", src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/2026-07-26_alternative_objectives")
    ap.add_argument("--coverage", default="results/2026-07-24_serving_ceiling")
    a = ap.parse_args()
    root, cov_root = Path(a.root), Path(a.coverage)
    plots = root / "plots"
    src = str(root)

    audit = pd.read_csv(root / "candidate_validation_audit.csv")
    audit = audit[audit.config_id.notna()].copy()
    audit["config_id"] = audit.config_id.astype(int)
    cov = pd.read_csv(cov_root / "per_config_workload_metrics.csv")

    fig_role_matrix(plots, audit, src)
    fig_pareto_roles(plots, cov, audit, src)

    vp = root / "objective_winners_validated.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        fig_winner_comparison(plots, v, src)
        fig_feasibility(plots, v, src)
        fig_outcome_counts(plots, v, src)
    else:
        print("validated results not present yet; skipped figures 1/3/5")


if __name__ == "__main__":
    main()
