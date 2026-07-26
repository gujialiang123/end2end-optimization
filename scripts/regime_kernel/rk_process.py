#!/usr/bin/env python3
"""Turn raw regime-kernel results into the tidy tables the plots consume."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rk_lib as L

RAW = L.RESULTS / "raw"
PROC = L.RESULTS / "processed"


def transfer_matrix() -> pd.DataFrame:
    """Every tuned profile evaluated at every token count (RQ2)."""
    rows = []
    for model in L.MODELS:
        prof_file = L.CONFIGS / f"{model}_profiles.json"
        if not prof_file.exists():
            continue
        profiles = json.loads(prof_file.read_text())
        # map config -> profile name so we can label the benchmark rows
        key_to_name = {L.config_key(cfg): name for name, cfg in profiles.items()}
        for f in sorted((RAW / "transfer" / model).glob("profiles_t*_uniform.json")):
            d = json.loads(f.read_text())
            base = d["default_baseline"]["median_ms"]
            tokens = d["plan"]["tokens"]
            for r in d["results"]:
                name = key_to_name.get(r["config_key"])
                if name is None:
                    continue
                rows.append(dict(model=model, profile=name, tokens=tokens,
                                 M=r["M"], median_ms=r["median_ms"],
                                 p95_ms=r["p95_ms"], default_ms=base,
                                 speedup_vs_default=base / r["median_ms"],
                                 regression_pct=(r["median_ms"] / base - 1) * 100,
                                 correctness_ok=r["correctness_ok"]))
    return pd.DataFrame(rows)


def routing_control() -> pd.DataFrame:
    rows = []
    for model in L.MODELS:
        for f in sorted((RAW / "routing" / model).glob("ctrl_t*_*.json")):
            d = json.loads(f.read_text())
            if not d["results"]:
                continue
            best = d["results"][0]
            base = d["default_baseline"]["median_ms"]
            rs = d["routing_stats"]
            rows.append(dict(model=model, tokens=d["plan"]["tokens"],
                             M=best["M"], routing=d["plan"]["routing"],
                             cv_expert_load=rs["cv_expert_load"],
                             gini_expert_load=rs["gini_expert_load"],
                             active_experts=rs["active_experts"],
                             default_ms=base, best_ms=best["median_ms"],
                             best_speedup=best["speedup_vs_default"],
                             best_key=best["config_key"],
                             BLOCK_SIZE_M=best["BLOCK_SIZE_M"],
                             BLOCK_SIZE_N=best["BLOCK_SIZE_N"],
                             BLOCK_SIZE_K=best["BLOCK_SIZE_K"],
                             GROUP_SIZE_M=best["GROUP_SIZE_M"]))
    return pd.DataFrame(rows)


def agent_trace() -> pd.DataFrame:
    rows = []
    for f in sorted((L.RESULTS / "agent").rglob("agent_result.json")):
        d = json.loads(f.read_text())
        for it in d["iterations"]:
            rec = dict(it)
            rec.pop("bottleneck", None)
            rows.append(dict(model=d["model"], tokens=d["tokens"], M=d["M"],
                             bottleneck=d["diagnosis"]["bottleneck"], **rec))
    return pd.DataFrame(rows)


def workload_characterization() -> pd.DataFrame:
    """Per-regime kernel workload facts measured during the sweep."""
    rows = []
    for model in L.MODELS:
        for f in sorted((RAW / "sweep" / model).glob("full_t*_uniform.json")):
            d = json.loads(f.read_text())
            p, rs = d["plan"], d["routing_stats"]
            shape = L.MODELS[model]
            M = p["M"]
            N, K, E = (shape["moe_intermediate_size"], shape["hidden_size"],
                       shape["num_experts"])
            flops = 2 * M * N * K * 2
            wbytes = E * (2 * N * K + K * N) * 2
            best = d["results"][0] if d["results"] else None
            rows.append(dict(
                model=model, tokens=p["tokens"], M=M, top_k=p["top_k"],
                num_experts=E, active_experts=rs["active_experts"],
                max_expert_load=rs["max_expert_load"],
                mean_expert_load=rs["mean_expert_load"],
                cv_expert_load=rs["cv_expert_load"],
                gini_expert_load=rs["gini_expert_load"],
                arithmetic_intensity=flops / wbytes,
                default_ms=d["default_baseline"]["median_ms"],
                best_ms=best["median_ms"] if best else None,
                best_speedup=best["speedup_vs_default"] if best else None,
                best_key=best["config_key"] if best else "",
                n_candidates_ok=len(d["results"]),
                n_candidates_failed=len(d["failures"])))
    return pd.DataFrame(rows)


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    for name, fn in (("transfer_matrix", transfer_matrix),
                     ("routing_control", routing_control),
                     ("agent_trace", agent_trace),
                     ("workload_characterization", workload_characterization)):
        df = fn()
        if df.empty:
            print(f"[skip] {name}: no data")
            continue
        df.to_csv(PROC / f"{name}.csv", index=False)
        print(f"wrote {PROC/name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
