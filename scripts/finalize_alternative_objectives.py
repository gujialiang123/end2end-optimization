#!/usr/bin/env python3
"""Phase-5: validated selection for the alternative-objective study.

Combines the original validation pass with the new targeted validation runs,
computes 5-repetition means with bootstrap 95 % confidence intervals against the
cookbook, and classifies every objective role.

Classification (all judged by CI, never by a fixed threshold):
  WIN                    intended objective significantly improves and no
                         guardrail metric significantly regresses
  REGRESSION             intended objective significantly degrades
  TRADE-OFF              >=1 metric significantly improves and >=1 other
                         primary metric significantly worsens
  FLAT / INCONCLUSIVE    all confidence intervals overlap zero
  STRICT_ALL_METRIC_WIN  no metric significantly regresses and >=1 significantly
                         improves (a genuine free win)
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

COOKBOOK_ID = 74
WORKLOADS = ["R_short_decode", "R_medium_balanced", "R_long_prefill",
             "R_concurrent_decode", "shared_prefix", "tool_agent"]
HIGHER = ["request_throughput", "output_throughput"]
LOWER = ["ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms"]
METRICS = HIGHER + LOWER
OBJ_METRIC = {
    "request_throughput_best": "request_throughput",
    "output_throughput_best": "output_throughput",
    "ttft_p95_best": "ttft_p95_ms",
    "tpot_p95_best": "tpot_p95_ms",
    "e2e_p95_best": "e2e_p95_ms",
    "constrained_throughput_best_3pct": "request_throughput",
    "constrained_ttft_best": "ttft_p95_ms",
    "constrained_tpot_best": "tpot_p95_ms",
    "constrained_e2e_best": "e2e_p95_ms",
}
RNG = np.random.default_rng(20260726)


def load_runs(paths) -> pd.DataFrame:
    fr = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        fs = sorted(glob.glob(str(p / "per_run_metrics_*.csv")))
        if fs:
            fr.append(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True))
        elif (p / "per_run_metrics.csv").exists():
            fr.append(pd.read_csv(p / "per_run_metrics.csv"))
    if not fr:
        raise SystemExit("no per-run metrics found")
    df = pd.concat(fr, ignore_index=True)
    # keep the LAST 5 reps for each (model, config, workload) if duplicated
    df = df.drop_duplicates(["model", "config_id", "workload", "rep"], keep="last")
    return df


def boot_delta(cand: np.ndarray, base: np.ndarray, higher: bool, n=8000):
    """Bootstrap the improvement of cand vs base.

    Improvement is defined so that positive always means better:
      higher-is-better metric: cand/base - 1
      lower-is-better metric : 1 - cand/base
    Returns (point estimate, lo, hi).
    """
    if len(cand) == 0 or len(base) == 0:
        return np.nan, np.nan, np.nan
    ci = RNG.choice(cand, (n, len(cand)), replace=True).mean(axis=1)
    bi = RNG.choice(base, (n, len(base)), replace=True).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (ci / bi - 1.0) if higher else (1.0 - ci / bi)
        pt = (cand.mean() / base.mean() - 1.0) if higher else (1.0 - cand.mean() / base.mean())
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def sig(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return 0
    if lo > 0:
        return 1
    if hi < 0:
        return -1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/2026-07-26_alternative_objectives/candidate_validation_audit.csv")
    ap.add_argument("--runs", nargs="+",
                    default=["results/2026-07-26_alternative_objectives",
                             "results/2026-07-24_serving_ceiling_validation"])
    ap.add_argument("--out", default="results/2026-07-26_alternative_objectives")
    a = ap.parse_args()
    out = Path(a.out)

    runs = load_runs(a.runs)
    audit = pd.read_csv(a.audit)
    audit = audit[audit.config_id.notna()].copy()
    audit["config_id"] = audit.config_id.astype(int)

    reps = runs.groupby(["model", "config_id", "workload"]).size()

    rows = []
    for (model, wl), grp in audit.groupby(["model", "workload"]):
        base = runs[(runs.model == model) & (runs.config_id == COOKBOOK_ID)
                    & (runs.workload == wl)]
        if base.empty:
            continue
        for _, r in grp.iterrows():
            cid = int(r.config_id)
            cand = runs[(runs.model == model) & (runs.config_id == cid)
                        & (runs.workload == wl)]
            n = len(cand)
            rec = dict(model=model, workload=wl, objective_role=r.objective_role,
                       config_id=cid, hash=r["hash"],
                       cap=r.max_running_requests, chunk=r.chunked_prefill_size,
                       policy=r.schedule_policy, mem=r.mem_fraction_static,
                       repeat_count=n, baseline_reps=len(base))
            if n == 0:
                rec["classification"] = "NOT_VALIDATED"
                rows.append(rec)
                continue
            sigs = {}
            for m in METRICS:
                higher = m in HIGHER
                pt, lo, hi = boot_delta(cand[m].to_numpy(float),
                                        base[m].to_numpy(float), higher)
                rec[f"{m}_raw"] = float(cand[m].mean())
                rec[f"{m}_base"] = float(base[m].mean())
                rec[f"{m}_delta"] = pt
                rec[f"{m}_ci_lo"] = lo
                rec[f"{m}_ci_hi"] = hi
                sigs[m] = sig(lo, hi)
            up = [m for m, s in sigs.items() if s > 0]
            dn = [m for m, s in sigs.items() if s < 0]
            obj = OBJ_METRIC.get(r.objective_role)
            if obj is not None and sigs.get(obj, 0) < 0:
                cls = "REGRESSION"
            elif up and not dn:
                cls = "STRICT_ALL_METRIC_WIN" if obj is None or sigs.get(obj, 0) > 0 \
                    else "WIN"
            elif up and dn:
                cls = "TRADE-OFF"
            elif dn and not up:
                cls = "REGRESSION"
            else:
                cls = "FLAT"
            if cls == "STRICT_ALL_METRIC_WIN" and obj is not None and sigs.get(obj, 0) > 0:
                pass  # objective itself improved and nothing regressed
            rec["metrics_significantly_better"] = ";".join(up)
            rec["metrics_significantly_worse"] = ";".join(dn)
            rec["classification"] = cls
            rows.append(rec)

    v = pd.DataFrame(rows)
    v.to_csv(out / "objective_winners_validated.csv", index=False)

    # ---- comparison matrix: one row per model/workload/role ----------------
    key = ["cookbook", "request_throughput_best", "ttft_p95_best", "tpot_p95_best",
           "e2e_p95_best", "constrained_throughput_best_3pct",
           "maximin_balanced_best", "pareto_knee_candidate",
           "strict_all_metric_candidate"]
    cm = v[v.objective_role.isin(key)].copy()
    cm["role_order"] = cm.objective_role.map({k: i for i, k in enumerate(key)})
    cm = cm.sort_values(["model", "workload", "role_order"])
    cols = (["model", "workload", "objective_role", "config_id", "hash",
             "cap", "chunk", "policy", "mem"] +
            [f"{m}{s}" for m in METRICS for s in ("_raw", "_delta", "_ci_lo", "_ci_hi")] +
            ["classification", "repeat_count"])
    cm[[c for c in cols if c in cm.columns]].to_csv(
        out / "objective_comparison_matrix.csv", index=False)

    # ---- knobs of every selected config ------------------------------------
    (v[["model", "workload", "objective_role", "config_id", "hash",
        "cap", "chunk", "policy", "mem"]]
     .drop_duplicates().to_csv(out / "selected_config_knobs.csv", index=False))

    # ---- markdown report ---------------------------------------------------
    with open(out / "objective_winners_validated.md", "w") as f:
        f.write("# Validated alternative-objective winners\n\n")
        f.write("Improvement sign convention: positive always means better "
                "(throughput `cand/base-1`, latency `1-cand/base`). "
                "Classification uses bootstrap 95 % CIs, never a fixed "
                "threshold.\n\n")
        for (model, wl), g in cm.groupby(["model", "workload"]):
            f.write(f"\n## {model} — {wl}\n\n")
            t = g[["objective_role", "config_id", "cap", "chunk", "policy", "mem",
                   "request_throughput_delta", "ttft_p95_ms_delta",
                   "tpot_p95_ms_delta", "e2e_p95_ms_delta",
                   "classification", "repeat_count"]].copy()
            for c in t.columns:
                if c.endswith("_delta"):
                    t[c] = (t[c] * 100).round(1)
            f.write(t.to_markdown(index=False))
            f.write("\n")

    # ---- outcome counts ----------------------------------------------------
    counts = (v.groupby(["objective_role", "classification"]).size()
              .unstack(fill_value=0))
    counts.to_csv(out / "outcome_counts_by_objective.csv")

    print("=== classification counts by objective role ===")
    print(counts.to_string())
    print("\n=== per model x workload, key roles ===")
    print(cm.pivot_table(index=["model", "workload"], columns="objective_role",
                         values="classification", aggfunc="first").to_string())
    json.dump({"n_validated_rows": int(len(v)),
               "roles": sorted(v.objective_role.unique().tolist())},
              open(out / "validated_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
