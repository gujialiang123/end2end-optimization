#!/usr/bin/env python3
"""Analysis for the 2026-07-24 serving-ceiling campaign.

Builds, per model:
  * summary_matrix.csv / .md   — one row per workload (Phase 7)
  * pareto_points.csv          — non-dominance flags (Phase 6)
  * transfer_matrix_*.csv      — 6x6 regime-winner transfer (Phase 5)
  * gain_distribution.csv      — win / flat / regression / trade-off counts
  * validated_candidates.csv   — populated from the 5-rep validation pass

Sign conventions (Phase 6):
  throughput improvement = candidate / baseline - 1
  latency  improvement   = 1 - candidate / baseline      (positive == faster)
"""
from __future__ import annotations
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

WORKLOADS = ["R_short_decode", "R_medium_balanced", "R_long_prefill",
             "R_concurrent_decode", "shared_prefix", "tool_agent"]
COOKBOOK_HASH = "cap32_chunk-1_pollpm_mem0.85"

HIGHER_BETTER = ["request_throughput", "output_throughput", "total_throughput",
                 "input_throughput"]
LOWER_BETTER = ["ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "ttft_mean_ms",
                "tpot_p50_ms", "tpot_p95_ms", "tpot_mean_ms",
                "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "e2e_mean_ms"]


def load(outroot: Path, pattern="per_run_metrics_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(str(outroot / pattern)))
    if not files:
        raise SystemExit(f"no {pattern} under {outroot}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def collapse_reps(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over repetitions -> one row per (model, config, workload) + CI."""
    keys = ["model", "config_id", "hash", "cap", "chunk", "policy", "mem",
            "is_cookbook", "workload"]
    metrics = [c for c in HIGHER_BETTER + LOWER_BETTER if c in df.columns]
    g = df.groupby(keys, dropna=False)
    out = g[metrics].mean().reset_index()
    out["n_rep"] = g.size().values
    # 95% CI half-width on the primary metric
    sd = g["request_throughput"].std(ddof=1).values
    n = out["n_rep"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        out["request_throughput_ci95"] = 1.96 * np.where(n > 1, sd / np.sqrt(n), np.nan)
    return out


def pareto_mask(points: np.ndarray, maximize: list[bool]) -> np.ndarray:
    """Boolean mask of non-dominated rows. points: (n, k)."""
    p = points.copy().astype(float)
    for j, mx in enumerate(maximize):
        if not mx:
            p[:, j] = -p[:, j]           # convert to maximize
    n = len(p)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if j >= i everywhere and > somewhere
        dom = np.all(p >= p[i], axis=1) & np.any(p > p[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep


def classify(delta_thr, ci_thr, delta_lat_list, ci_lat_list, noise=0.03):
    """WIN / REGRESSION / TRADE-OFF / FLAT using CI where available."""
    def sig(d, ci):
        if d is None or (isinstance(d, float) and math.isnan(d)):
            return 0
        if ci is not None and not (isinstance(ci, float) and math.isnan(ci)) and ci > 0:
            return 1 if d - ci > 0 else (-1 if d + ci < 0 else 0)
        return 1 if d > noise else (-1 if d < -noise else 0)

    s_thr = sig(delta_thr, ci_thr)
    lat_sigs = [sig(d, c) for d, c in zip(delta_lat_list, ci_lat_list)]
    improved = s_thr > 0 or any(s > 0 for s in lat_sigs)
    worsened = s_thr < 0 or any(s < 0 for s in lat_sigs)
    if improved and worsened:
        return "TRADE-OFF"
    if improved:
        return "WIN"
    if worsened:
        return "REGRESSION"
    return "FLAT"


def analyze_model(df: pd.DataFrame, model: str, outroot: Path):
    d = df[df.model == model].copy()
    if d.empty:
        return None
    res_dir = outroot / "analysis" / model
    res_dir.mkdir(parents=True, exist_ok=True)

    # ---- baselines: cookbook row per workload ----
    cb = d[d.hash == COOKBOOK_HASH].set_index("workload")
    if cb.empty:
        print(f"[{model}] WARNING: cookbook config missing, skipping")
        return None

    # ---- per-workload deltas vs cookbook ----
    rows = []
    for wl in WORKLOADS:
        sub = d[d.workload == wl].copy()
        if sub.empty or wl not in cb.index:
            continue
        b = cb.loc[wl]
        for m in HIGHER_BETTER:
            if m in sub:
                sub[f"d_{m}"] = sub[m] / b[m] - 1.0
        for m in LOWER_BETTER:
            if m in sub:
                sub[f"d_{m}"] = 1.0 - sub[m] / b[m]
        # pareto: (TTFT p95 lower better, output throughput higher better)
        pts = sub[["ttft_p95_ms", "output_throughput"]].to_numpy()
        ok = ~np.isnan(pts).any(axis=1)
        sub["pareto_ttft_outthr"] = False
        if ok.sum():
            m2 = pareto_mask(pts[ok], [False, True])
            sub.loc[sub.index[ok], "pareto_ttft_outthr"] = m2
        # pareto2: (E2E p95 lower, request throughput higher)
        pts2 = sub[["e2e_p95_ms", "request_throughput"]].to_numpy()
        ok2 = ~np.isnan(pts2).any(axis=1)
        sub["pareto_e2e_reqthr"] = False
        if ok2.sum():
            sub.loc[sub.index[ok2], "pareto_e2e_reqthr"] = pareto_mask(pts2[ok2], [False, True])
        # full multi-metric non-dominance
        cols5 = ["request_throughput", "output_throughput", "ttft_p95_ms",
                 "tpot_p95_ms", "e2e_p95_ms"]
        pts5 = sub[cols5].to_numpy()
        ok5 = ~np.isnan(pts5).any(axis=1)
        sub["pareto_full"] = False
        if ok5.sum():
            sub.loc[sub.index[ok5], "pareto_full"] = pareto_mask(
                pts5[ok5], [True, True, False, False, False])
        rows.append(sub)
    allw = pd.concat(rows, ignore_index=True)
    allw.to_csv(res_dir / "per_workload_deltas.csv", index=False)

    # ---- summary matrix (one row per workload) ----
    summary = []
    for wl in WORKLOADS:
        sub = allw[allw.workload == wl]
        if sub.empty:
            continue
        b = cb.loc[wl]
        best_thr = sub.loc[sub.request_throughput.idxmax()]
        best_out = sub.loc[sub.output_throughput.idxmax()]
        lo_ttft = sub.loc[sub.ttft_p95_ms.idxmin()]
        lo_tpot = sub.loc[sub.tpot_p95_ms.idxmin()]
        lo_e2e = sub.loc[sub.e2e_p95_ms.idxmin()]
        # balanced = pareto point maximizing (norm thr) - (norm ttft)
        par = sub[sub.pareto_ttft_outthr]
        if len(par):
            z = ((par.output_throughput - sub.output_throughput.mean()) / (sub.output_throughput.std() or 1)
                 - (par.ttft_p95_ms - sub.ttft_p95_ms.mean()) / (sub.ttft_p95_ms.std() or 1))
            balanced = par.loc[z.idxmax()]
        else:
            balanced = best_thr
        cls = classify(best_thr["d_request_throughput"],
                       best_thr.get("request_throughput_ci95_rel", np.nan),
                       [best_thr["d_ttft_p95_ms"], best_thr["d_tpot_p95_ms"],
                        best_thr["d_e2e_p95_ms"]],
                       [np.nan, np.nan, np.nan])
        n = len(sub)
        summary.append(dict(
            model=model, workload=wl,
            cookbook_config_id=int(b["config_id"]), cookbook_hash=COOKBOOK_HASH,
            cookbook_request_throughput=b["request_throughput"],
            cookbook_output_throughput=b["output_throughput"],
            cookbook_ttft_p50_ms=b["ttft_p50_ms"], cookbook_ttft_p95_ms=b["ttft_p95_ms"],
            cookbook_tpot_p50_ms=b["tpot_p50_ms"], cookbook_tpot_p95_ms=b["tpot_p95_ms"],
            cookbook_e2e_p50_ms=b["e2e_p50_ms"], cookbook_e2e_p95_ms=b["e2e_p95_ms"],
            best_throughput_config_id=int(best_thr["config_id"]),
            best_throughput_hash=best_thr["hash"],
            best_throughput_knobs=f"cap={best_thr['cap']},chunk={best_thr['chunk']},"
                                  f"policy={best_thr['policy']},mem={best_thr['mem']}",
            d_request_throughput=best_thr["d_request_throughput"],
            d_output_throughput=best_thr["d_output_throughput"],
            d_ttft_p50=best_thr["d_ttft_p50_ms"], d_ttft_p95=best_thr["d_ttft_p95_ms"],
            d_tpot_p50=best_thr["d_tpot_p50_ms"], d_tpot_p95=best_thr["d_tpot_p95_ms"],
            d_e2e_p50=best_thr["d_e2e_p50_ms"], d_e2e_p95=best_thr["d_e2e_p95_ms"],
            best_output_config_id=int(best_out["config_id"]),
            lowest_ttft_p95_config_id=int(lo_ttft["config_id"]),
            lowest_tpot_p95_config_id=int(lo_tpot["config_id"]),
            lowest_e2e_p95_config_id=int(lo_e2e["config_id"]),
            balanced_config_id=int(balanced["config_id"]),
            n_configs_evaluated=n,
            n_pareto=int(sub.pareto_ttft_outthr.sum()),
            n_pareto_full=int(sub.pareto_full.sum()),
            classification=cls,
            # distribution: is the search space broadly good, or one cliff?
            worst_request_throughput=sub.request_throughput.min(),
            median_request_throughput=sub.request_throughput.median(),
            worst_vs_cookbook=sub.request_throughput.min() / b["request_throughput"] - 1,
            median_vs_cookbook=sub.request_throughput.median() / b["request_throughput"] - 1,
            pct_configs_improve_throughput=float((sub.d_request_throughput > 0).mean()),
            pct_configs_improve_ttft_p95=float((sub.d_ttft_p95_ms > 0).mean()),
            pct_configs_improve_all=float(((sub.d_request_throughput > 0) &
                                           (sub.d_ttft_p95_ms > 0) &
                                           (sub.d_tpot_p95_ms > 0) &
                                           (sub.d_e2e_p95_ms > 0)).mean()),
            pct_dominated_by_cookbook=float(((sub.d_request_throughput <= 0) &
                                             (sub.d_ttft_p95_ms <= 0) &
                                             (sub.d_e2e_p95_ms <= 0)).mean()),
            repeat_count=int(sub.n_rep.max()) if "n_rep" in sub else 1,
        ))
    sm = pd.DataFrame(summary)
    sm.to_csv(res_dir / "summary_matrix.csv", index=False)

    # ---- pareto points ----
    par = allw[allw.pareto_ttft_outthr | allw.pareto_full | allw.pareto_e2e_reqthr]
    par.to_csv(res_dir / "pareto_points.csv", index=False)

    # ---- gain distribution (per config x workload) ----
    gd = []
    for wl in WORKLOADS:
        sub = allw[allw.workload == wl]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            gd.append(dict(model=model, workload=wl, config_id=int(r.config_id),
                           hash=r["hash"],
                           cls=classify(r["d_request_throughput"], np.nan,
                                        [r["d_ttft_p95_ms"], r["d_tpot_p95_ms"],
                                         r["d_e2e_p95_ms"]], [np.nan] * 3)))
    gdf = pd.DataFrame(gd)
    gdf.to_csv(res_dir / "gain_distribution.csv", index=False)

    # ---- transfer matrix: winner of each source regime applied to all targets ----
    winners = {wl: int(sm[sm.workload == wl].best_throughput_config_id.iloc[0])
               for wl in sm.workload}
    cb_id = int(cb.iloc[0]["config_id"])
    src = {"cookbook": cb_id, **{f"{wl}_winner": winners[wl] for wl in winners}}
    for metric, higher in [("request_throughput", True), ("output_throughput", True),
                           ("ttft_p95_ms", False), ("tpot_p95_ms", False),
                           ("e2e_p95_ms", False)]:
        mat = pd.DataFrame(index=list(src), columns=WORKLOADS, dtype=float)
        for sname, cid in src.items():
            for wl in WORKLOADS:
                cell = allw[(allw.config_id == cid) & (allw.workload == wl)]
                if cell.empty or wl not in cb.index:
                    continue
                v = cell.iloc[0][metric]; base = cb.loc[wl][metric]
                if base and not math.isnan(base):
                    # ratio vs target cookbook; >1 always == better
                    mat.loc[sname, wl] = (v / base) if higher else (base / v)
        mat.to_csv(res_dir / f"transfer_matrix_{metric}.csv")

    json.dump({"model": model, "source_configs": src,
               "n_rows_per_run": int(len(df[df.model == model]))},
              open(res_dir / "transfer_sources.json", "w"), indent=2)
    return sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--pattern", default="per_run_metrics_*.csv")
    args = ap.parse_args()
    outroot = Path(args.outroot)
    raw = load(outroot, args.pattern)
    print(f"loaded {len(raw)} per-run rows; "
          f"{raw.groupby('model').config_id.nunique().to_dict()} configs/model")
    coll = collapse_reps(raw)
    coll.to_csv(outroot / "per_config_workload_metrics.csv", index=False)
    alls = []
    for model in sorted(raw.model.unique()):
        sm = analyze_model(coll, model, outroot)
        if sm is not None:
            alls.append(sm)
            print(f"\n=== {model} ===")
            print(sm[["workload", "d_request_throughput", "d_ttft_p95",
                      "d_e2e_p95", "classification", "n_configs_evaluated",
                      "n_pareto", "pct_configs_improve_throughput"]].to_string(index=False))
    if alls:
        cat = pd.concat(alls, ignore_index=True)
        cat.to_csv(outroot / "summary_matrix.csv", index=False)
        with open(outroot / "summary_matrix.md", "w") as f:
            f.write("# Serving-ceiling summary matrix (2026-07-24)\n\n")
            f.write("Sign convention: throughput delta = cand/base - 1; "
                    "latency delta = 1 - cand/base (positive == faster).\n\n")
            f.write(cat.to_markdown(index=False))
        print(f"\nwrote {outroot/'summary_matrix.csv'}")


if __name__ == "__main__":
    main()
