#!/usr/bin/env python3
"""Alternative-objective re-analysis of the completed serving-ceiling campaign.

The full 192-point grid is already measured for every model x workload, so
changing the optimization objective does NOT require a new search — it only
changes which measured configuration is selected. This script therefore:

  Phase 1  audit which configurations already have 5 valid repetitions
  Phase 2  apply 8 alternative selection policies to the warmed coverage grid
  Phase 3  emit a validation plan containing ONLY the missing configurations

Sign convention (benefit ratios, >1 always means better):
    r_req  = cand_request_throughput / cookbook_request_throughput
    r_out  = cand_output_throughput  / cookbook_output_throughput
    r_ttft = cookbook_ttft_p95 / cand_ttft_p95
    r_tpot = cookbook_tpot_p95 / cand_tpot_p95
    r_e2e  = cookbook_e2e_p95  / cand_e2e_p95
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

COOKBOOK_HASH = "cap32_chunk-1_pollpm_mem0.85"
WORKLOADS = ["R_short_decode", "R_medium_balanced", "R_long_prefill",
             "R_concurrent_decode", "shared_prefix", "tool_agent"]

HIGHER = {"request_throughput": "r_req", "output_throughput": "r_out"}
LOWER = {"ttft_p95_ms": "r_ttft", "tpot_p95_ms": "r_tpot", "e2e_p95_ms": "r_e2e"}
RATIOS = ["r_req", "r_out", "r_ttft", "r_tpot", "r_e2e"]

KNOBS = ["cap", "chunk", "policy", "mem"]
RAW = ["request_throughput", "output_throughput", "ttft_p50_ms", "ttft_p95_ms",
       "tpot_p50_ms", "tpot_p95_ms", "e2e_p50_ms", "e2e_p95_ms"]


def add_ratios(sub: pd.DataFrame, cb: pd.Series) -> pd.DataFrame:
    d = sub.copy()
    for col, name in HIGHER.items():
        d[name] = d[col] / cb[col]
    for col, name in LOWER.items():
        d[name] = cb[col] / d[col]
    d["maximin_score"] = d[RATIOS].min(axis=1)
    d["geometric_score"] = np.exp(np.log(d[RATIOS].clip(lower=1e-9)).mean(axis=1))
    return d


def pick(d: pd.DataFrame, by: str, ascending: bool):
    """Deterministic argmin/argmax with config_id as final tiebreak."""
    if d.empty:
        return None
    s = d.sort_values([by, "config_id"], ascending=[ascending, True])
    return s.iloc[0]


def feasible(d: pd.DataFrame, exclude: str, tol: float) -> pd.DataFrame:
    """Rows where every guardrail ratio except `exclude` is >= the tolerance.

    tol is expressed as a benefit ratio: a 3% allowance means ratio >= 1/1.03
    for latency guardrails and >= 0.99 for throughput guardrails, matching the
    plan's '<= 1.03 x cookbook' / '>= 0.99 x cookbook' wording.
    """
    m = pd.Series(True, index=d.index)
    for name in RATIOS:
        if name == exclude:
            continue
        floor = 0.99 if name in ("r_req", "r_out") else 1.0 / (1.0 + tol)
        m &= d[name] >= floor
    return d[m]


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Non-dominated rows; all columns are benefit ratios (higher == better)."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dom = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep


def select_all(d: pd.DataFrame, cb: pd.Series) -> dict:
    """Apply every objective policy to one model x workload slice."""
    roles: dict[str, object] = {}
    roles["cookbook"] = d[d.hash == COOKBOOK_HASH].iloc[0]

    # --- B. pure single-objective winners --------------------------------
    roles["request_throughput_best"] = pick(d, "request_throughput", False)
    roles["output_throughput_best"] = pick(d, "output_throughput", False)
    roles["ttft_p95_best"] = pick(d, "ttft_p95_ms", True)
    roles["tpot_p95_best"] = pick(d, "tpot_p95_ms", True)
    roles["e2e_p95_best"] = pick(d, "e2e_p95_ms", True)
    roles["ttft_p50_best"] = pick(d, "ttft_p50_ms", True)
    roles["tpot_p50_best"] = pick(d, "tpot_p50_ms", True)
    roles["e2e_p50_best"] = pick(d, "e2e_p50_ms", True)

    # --- Policy 1: SLO-constrained throughput (+ sensitivity) ------------
    for tol, tag in ((0.01, "1pct"), (0.03, "3pct"), (0.05, "5pct")):
        f = feasible(d, exclude="r_req", tol=tol)
        roles[f"constrained_throughput_best_{tag}"] = (
            pick(f, "request_throughput", False) if len(f) else None)

    # --- Policies 2-4: latency-first with throughput guardrail -----------
    for excl, col, name in (("r_ttft", "ttft_p95_ms", "constrained_ttft_best"),
                            ("r_tpot", "tpot_p95_ms", "constrained_tpot_best"),
                            ("r_e2e", "e2e_p95_ms", "constrained_e2e_best")):
        f = feasible(d, exclude=excl, tol=0.03)
        roles[name] = pick(f, col, True) if len(f) else None

    # --- Policy 5: maximin ------------------------------------------------
    s = d.sort_values(["maximin_score", "geometric_score", "request_throughput",
                       "config_id"], ascending=[False, False, False, True])
    roles["maximin_balanced_best"] = s.iloc[0]

    # --- Policy 6: geometric mean ----------------------------------------
    s = d.sort_values(["geometric_score", "request_throughput", "config_id"],
                      ascending=[False, False, True])
    roles["geometric_balanced_best"] = s.iloc[0]

    # --- Policy 7: strict / noise-tolerant all-metric ---------------------
    strict = d[(d[RATIOS] >= 1.00).all(axis=1)]
    roles["strict_all_metric_candidate"] = (
        pick(strict, "request_throughput", False) if len(strict) else None)
    tolset = d[(d[RATIOS] >= 0.97).all(axis=1)]
    roles["noise_tolerant_all_metric_candidate"] = (
        pick(tolset, "request_throughput", False) if len(tolset) else None)

    # --- Policy 8: Pareto knee -------------------------------------------
    P = d[RATIOS].to_numpy(dtype=float)
    nd = pareto_mask(P)
    par = d[nd]
    if len(par):
        Q = par[RATIOS].to_numpy(dtype=float)
        # min-max normalize each ratio over the Pareto set, utopia = all 1s
        lo, hi = Q.min(axis=0), Q.max(axis=0)
        rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        Z = (Q - lo) / rng
        dist = np.sqrt(((1.0 - Z) ** 2).sum(axis=1))
        order = np.lexsort((par.config_id.to_numpy(), dist))
        roles["pareto_knee_candidate"] = par.iloc[order[0]]
        ties = int((np.abs(dist - dist[order[0]]) < 1e-9).sum())
        roles["_pareto_knee_ties"] = ties
        roles["_pareto_n"] = int(nd.sum())
    else:
        roles["pareto_knee_candidate"] = None
        roles["_pareto_knee_ties"] = 0
        roles["_pareto_n"] = 0
    return roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", default="results/2026-07-24_serving_ceiling")
    ap.add_argument("--validation",
                    default="results/2026-07-24_serving_ceiling_validation")
    ap.add_argument("--out", default="results/2026-07-26_alternative_objectives")
    a = ap.parse_args()
    cov_root, val_root, out = Path(a.coverage), Path(a.validation), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cov = pd.read_csv(cov_root / "per_config_workload_metrics.csv")
    val = pd.read_csv(val_root / "per_run_metrics.csv")
    val_reps = (val.groupby(["model", "config_id", "workload"]).size()
                .groupby(level=[0, 1]).min().rename("valid_reps"))

    rows, notes = [], []
    for model in sorted(cov.model.unique()):
        for wl in WORKLOADS:
            sub = cov[(cov.model == model) & (cov.workload == wl)]
            if sub.empty:
                continue
            cbrow = sub[sub.hash == COOKBOOK_HASH]
            if cbrow.empty:
                notes.append(f"{model}/{wl}: cookbook missing")
                continue
            cb = cbrow.iloc[0]
            d = add_ratios(sub, cb)
            roles = select_all(d, cb)
            meta = {k: roles.pop(k) for k in list(roles) if k.startswith("_")}
            for role, r in roles.items():
                if r is None:
                    rows.append(dict(model=model, workload=wl, objective_role=role,
                                     config_id=None, hash="NO_FEASIBLE_CONFIG",
                                     validation_status="N/A",
                                     reason="no configuration satisfies the constraints"))
                    continue
                rec = dict(model=model, workload=wl, objective_role=role,
                           config_id=int(r.config_id), hash=r["hash"],
                           max_running_requests=int(r.cap),
                           chunked_prefill_size=int(r.chunk),
                           schedule_policy=r.policy,
                           mem_fraction_static=float(r.mem))
                for c in RAW:
                    rec[f"cov_{c}"] = float(r[c])
                for c in RATIOS:
                    rec[c] = float(r[c])
                rec["maximin_score"] = float(r.maximin_score)
                rec["geometric_score"] = float(r.geometric_score)
                rec["min_benefit_ratio"] = float(min(r[c] for c in RATIOS))
                n = int(val_reps.get((model, int(r.config_id)), 0))
                rec["valid_validation_reps"] = n
                rec["in_validation_dataset"] = n > 0
                rec["validation_status"] = ("ALREADY_VALIDATED" if n >= 5
                                            else "PARTIALLY_VALIDATED" if n > 0
                                            else "NOT_VALIDATED")
                rec["reason"] = (f"{n}/5 valid repetitions in {val_root.name}"
                                 if n else "absent from the validation dataset")
                rec["validation_path"] = str(val_root) if n else ""
                rec["pareto_n"] = meta.get("_pareto_n")
                rec["pareto_knee_ties"] = meta.get("_pareto_knee_ties")
                rows.append(rec)

    audit = pd.DataFrame(rows)
    audit.to_csv(out / "candidate_validation_audit.csv", index=False)
    audit.to_json(out / "candidate_validation_audit.json",
                  orient="records", indent=2)

    # ---------------- validation plan: unique missing configs -------------
    sel = audit[audit.config_id.notna()].copy()
    sel["config_id"] = sel.config_id.astype(int)
    grp = (sel.groupby(["model", "config_id", "hash"])
           .agg(roles=("objective_role", lambda s: "; ".join(sorted(set(s)))),
                workloads=("workload", lambda s: "; ".join(sorted(set(s)))),
                valid_reps=("valid_validation_reps", "min"))
           .reset_index())
    grp["additional_reps_required"] = (5 - grp.valid_reps).clip(lower=0)
    grp["needs_run"] = grp.additional_reps_required > 0
    # ~40 s startup + ~44 s warm-up + ~144 s per scored repetition set
    grp["est_runtime_s"] = np.where(grp.needs_run,
                                    84 + 144 * grp.additional_reps_required, 0)
    grp["reason"] = np.where(grp.needs_run,
                             "selected by an objective policy but lacking 5 valid repetitions",
                             "already has 5 valid repetitions under the canonical protocol")
    grp.sort_values(["model", "config_id"]).to_csv(out / "validation_plan.csv",
                                                   index=False)

    miss = grp[grp.needs_run]
    gpu_h = miss.est_runtime_s.sum() / 3600.0

    def cnt(mask):
        s = sel[mask]
        u = s.drop_duplicates(["model", "config_id"])
        return int((u.valid_validation_reps >= 5).sum()), int(len(u))

    pure = sel.objective_role.isin(["request_throughput_best", "output_throughput_best",
                                    "ttft_p95_best", "tpot_p95_best", "e2e_p95_best"])
    newp = sel.objective_role.str.startswith(("constrained_", "maximin_", "geometric_",
                                              "strict_", "noise_", "pareto_"))
    lines = []
    a1, b1 = cnt(pure)
    a2, b2 = cnt(newp)
    lines.append(f"pure single-objective winners already validated: {a1} / {b1}")
    lines.append(f"new constrained/maximin/pareto candidates already validated: {a2} / {b2}")
    lines.append(f"unique configs requiring new runs: {len(miss)}")
    lines.append(f"estimated H200 GPU-hours: {gpu_h:.2f}")
    lines.append("")
    lines.append("missing config hashes:")
    for _, r in miss.iterrows():
        lines.append(f"  {r.model:6s} cfg{r.config_id:<4d} {r['hash']:<34s} "
                     f"+{r.additional_reps_required} reps  <- {r.roles}")
    summary = "\n".join(lines)
    print(summary)

    with open(out / "validation_plan.md", "w") as f:
        f.write("# Alternative-objective validation plan\n\n```\n")
        f.write(summary)
        f.write("\n```\n\n## Full plan\n\n")
        f.write(grp.sort_values(["model", "config_id"]).to_markdown(index=False))
    json.dump({"unique_missing": int(len(miss)),
               "estimated_gpu_hours": round(gpu_h, 3),
               "missing": miss[["model", "config_id", "hash",
                                "additional_reps_required"]].to_dict("records")},
              open(out / "validation_plan.json", "w"), indent=2)
    if notes:
        print("\nNOTES:", *notes, sep="\n  ")


if __name__ == "__main__":
    main()
