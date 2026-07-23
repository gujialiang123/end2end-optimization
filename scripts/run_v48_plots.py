#!/usr/bin/env python3
"""v48 plots + plateau analysis. Reads per_trial_log.csv, baseline_reference.json,
best_validated.json. Emits 3 figures (png+svg) and prints plateau statistics that
feed summary.md.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path("/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-22_lfm25_plateau_100")


def load():
    rows = list(csv.DictReader(open(OUTDIR / "per_trial_log.csv")))
    rows.sort(key=lambda r: int(r["completed_index"]))
    rps = [float(r["request_throughput"]) for r in rows]
    ttft95 = [float(r["ttft_p95"]) for r in rows]
    base = json.load(open(OUTDIR / "baseline_reference.json"))
    base_mean = base["request_throughput_mean"]
    val = None
    p = OUTDIR / "best_validated.json"
    if p.exists():
        val = json.load(open(p))
    return rows, rps, ttft95, base_mean, val


def cummax(xs):
    out, m = [], -1e18
    for x in xs:
        m = max(m, x); out.append(m)
    return out


def plateau_stats(rps, base_mean, validated_best):
    n = len(rps)
    cm = cummax(rps)
    def best_through(k):
        return max(rps[:k]) if k <= n else max(rps)
    ref = validated_best if validated_best else cm[-1]
    # first index within 1% of final validated best
    first_within1 = next((i + 1 for i, v in enumerate(cm) if v >= 0.99 * ref), None)
    last20_improve = cm[-1] - cm[-21] if n >= 21 else cm[-1] - cm[0]
    last20_count = sum(1 for i in range(max(1, n - 20), n) if cm[i] > cm[i - 1])
    within = lambda p: sum(1 for x in rps if x >= (1 - p) * ref) / n
    return dict(
        n=n,
        best_through_10=best_through(10), best_through_20=best_through(20),
        best_through_50=best_through(50), best_through_75=best_through(75),
        best_through_100=best_through(100),
        final_cummax=cm[-1], validated_best=ref, baseline_mean=base_mean,
        first_within_1pct_index=first_within1,
        improvement_final_20=last20_improve,
        improvement_final_20_pct=(last20_improve / cm[-1] * 100) if cm[-1] else 0,
        n_final20_improving=last20_count,
        frac_within_1pct=within(0.01), frac_within_3pct=within(0.03),
        frac_within_5pct=within(0.05),
        cummax=cm,
    )


def plot_convergence_raw(rps, base_mean, ps, validated_best):
    x = np.arange(1, len(rps) + 1)
    cm = ps["cummax"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, rps, s=28, color="#7f8c8d", alpha=0.6, label="individual trial")
    ax.plot(x, cm, color="#c0392b", lw=2.5, label="cumulative best-so-far")
    ax.axhline(base_mean, color="#2980b9", ls="--", lw=1.8, label=f"cookbook baseline ({base_mean:.1f})")
    fw = ps["first_within_1pct_index"]
    if fw:
        ax.axvline(fw, color="#27ae60", ls=":", lw=1.5)
        ax.annotate(f"within 1% of final best\n@ config {fw}", xy=(fw, cm[fw - 1]),
                    xytext=(fw + 5, cm[fw - 1] * 0.85), fontsize=9, color="#27ae60",
                    arrowprops=dict(arrowstyle="->", color="#27ae60"))
    for k in (20, 50, 100):
        if k <= len(rps):
            ax.annotate(f"best@{k}={ps[f'best_through_{k}']:.1f}", xy=(k, ps[f'best_through_{k}']),
                        fontsize=8, color="#c0392b",
                        xytext=(k - 3, ps[f'best_through_{k}'] + 0.5))
    ax.set_xlabel("successfully evaluated unique configurations")
    ax.set_ylabel("request throughput (req/s)")
    ax.set_title("LFM2.5 serving-knob autotuning — convergence (no warm start)\n"
                 "R_concurrent_decode, H200, fixed Triton MoE + FA3 + CUDA graph on")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "convergence_req_throughput.png", dpi=130)
    fig.savefig(OUTDIR / "convergence_req_throughput.svg")
    plt.close(fig)


def plot_convergence_norm(rps, base_mean, ps):
    x = np.arange(1, len(rps) + 1)
    cm = np.array(ps["cummax"]) / base_mean
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, np.array(rps) / base_mean, s=28, color="#7f8c8d", alpha=0.6, label="individual trial")
    ax.plot(x, cm, color="#c0392b", lw=2.5, label="cumulative best-so-far")
    ax.axhline(1.0, color="#2980b9", ls="--", lw=1.8, label="cookbook baseline = 1.0x")
    ax.set_xlabel("successfully evaluated unique configurations")
    ax.set_ylabel("request throughput / cookbook-baseline mean")
    ax.set_title("LFM2.5 serving-knob autotuning — normalized convergence (no warm start)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "convergence_normalized.png", dpi=130)
    fig.savefig(OUTDIR / "convergence_normalized.svg")
    plt.close(fig)


def pareto_front(ttft, rps):
    idx = sorted(range(len(rps)), key=lambda i: (ttft[i], -rps[i]))
    front, best_rps = [], -1e18
    for i in idx:
        if rps[i] > best_rps:
            front.append(i); best_rps = rps[i]
    return front


def plot_pareto(rows, rps, ttft95, base_mean, validated_best_cfg):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(ttft95, rps, s=30, color="#7f8c8d", alpha=0.6, label="all configs")
    front = pareto_front(ttft95, rps)
    fx = [ttft95[i] for i in front]; fy = [rps[i] for i in front]
    ax.plot(fx, fy, "-o", color="#c0392b", lw=2, ms=6, label="Pareto frontier")
    ax.axhline(base_mean, color="#2980b9", ls="--", lw=1.5, label=f"cookbook baseline rps ({base_mean:.1f})")
    ax.set_xlabel("TTFT p95 (ms) — lower is better")
    ax.set_ylabel("request throughput (req/s) — higher is better")
    ax.set_title("LFM2.5 serving knobs — TTFT p95 vs throughput Pareto (100 configs)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "ttft_throughput_pareto.png", dpi=130)
    fig.savefig(OUTDIR / "ttft_throughput_pareto.svg")
    plt.close(fig)


def main():
    rows, rps, ttft95, base_mean, val = load()
    validated_best = None
    if val:
        validated_best = val.get("best_validated", {}).get("mean")
    ps = plateau_stats(rps, base_mean, validated_best)
    plot_convergence_raw(rps, base_mean, ps, validated_best)
    plot_convergence_norm(rps, base_mean, ps)
    plot_pareto(rows, rps, ttft95, base_mean, None)
    ps.pop("cummax")
    json.dump(ps, open(OUTDIR / "plateau_stats.json", "w"), indent=2)
    print(json.dumps(ps, indent=2))
    print(f"\nsaved plots + plateau_stats.json to {OUTDIR}")


if __name__ == "__main__":
    main()
