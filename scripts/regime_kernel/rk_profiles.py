#!/usr/bin/env python3
"""Build kernel profiles and compare selection strategies (RQ2 / RQ3).

Reads the sweep results and produces:

  configs/regime_kernel/<model>_profiles.json   candidate profiles for the
                                                transfer matrix
  configs/regime_kernel/profiles/<name>/        SGLANG_MOE_CONFIG_DIR trees for
                                                the end-to-end stage
  results/regime_kernel/processed/*.csv         tidy tables for plotting

Strategies compared:
  default        what the runtime does today (measured, not assumed)
  global_best    one config chosen over all token counts, objective =
                 geometric mean of per-M normalized latency (weights recorded)
  regime_aware   one config per regime cluster (2-3 profiles)
  oracle         best config at each individual M (upper bound, not deployable)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rk_lib as L

RAW = L.RESULTS / "raw"
PROC = L.RESULTS / "processed"


def load_sweep(model: str, stage: str = "sweep") -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / stage / model).glob("full_t*_uniform.json")):
        d = json.loads(f.read_text())
        base = d["default_baseline"]["median_ms"]
        for r in d["results"]:
            rows.append(dict(model=model, tokens=r["tokens"], M=r["M"],
                             config_key=r["config_key"],
                             BLOCK_SIZE_M=r["BLOCK_SIZE_M"],
                             BLOCK_SIZE_N=r["BLOCK_SIZE_N"],
                             BLOCK_SIZE_K=r["BLOCK_SIZE_K"],
                             GROUP_SIZE_M=r["GROUP_SIZE_M"],
                             num_warps=r["num_warps"], num_stages=r["num_stages"],
                             median_ms=r["median_ms"], p95_ms=r["p95_ms"],
                             std_ms=r["std_ms"],
                             default_ms=base,
                             speedup=base / r["median_ms"],
                             max_abs_err=r.get("max_abs_err")))
    return pd.DataFrame(rows)


def cfg_from_key(df: pd.DataFrame, key: str) -> dict:
    r = df[df.config_key == key].iloc[0]
    return {k: int(r[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                                   "GROUP_SIZE_M", "num_warps", "num_stages")}


def pick_global_best(df: pd.DataFrame, weights: dict | None = None):
    """Config minimizing the (weighted) geometric mean of normalized latency.

    Normalization is per token count against the best latency seen there, so no
    single large-M point can dominate the objective by magnitude alone.
    """
    best_per_t = df.groupby("tokens").median_ms.min().rename("best_ms")
    d = df.join(best_per_t, on="tokens")
    d["norm"] = d.median_ms / d.best_ms
    if weights:
        d["w"] = d.tokens.map(weights).fillna(1.0)
    else:
        d["w"] = 1.0
    # geometric mean of normalized latency, weighted
    g = d.groupby("config_key").apply(
        lambda x: math.exp((x.w * x.norm.apply(math.log)).sum() / x.w.sum()),
        include_groups=False).rename("geo_norm")
    # only consider configs measured at every token count
    complete = d.groupby("config_key").tokens.nunique()
    valid = complete[complete == d.tokens.nunique()].index
    g = g.loc[g.index.intersection(valid)]
    return g.sort_values().index[0], g.sort_values()


def regime_clusters(df: pd.DataFrame):
    """Map token counts onto the three regimes plus the crossover sweep."""
    ts = sorted(df.tokens.unique())
    low = [t for t in ts if t <= 4]
    mid = [t for t in ts if 4 < t <= 64]
    high = [t for t in ts if t > 64]
    return {"low_M": low, "mid_M": mid, "high_M": high}


def pick_for_subset(df: pd.DataFrame, tokens: list):
    sub = df[df.tokens.isin(tokens)]
    if sub.empty:
        return None, None
    key, ranked = pick_global_best(sub)
    return key, ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lfm25,qwen")
    ap.add_argument("--stage", default="sweep",
                    help="sweep (no-bias) or bias (the variant the server runs)")
    ap.add_argument("--suffix", default="",
                    help="suffix for profile names, e.g. _bias")
    a = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)
    L.CONFIGS.mkdir(parents=True, exist_ok=True)

    all_strategy, all_sweep = [], []
    for model in a.models.split(","):
        df = load_sweep(model, a.stage)
        if df.empty:
            print(f"[skip] no sweep results for {model}")
            continue
        all_sweep.append(df)
        shape = L.MODELS[model]

        # ---- oracle: best config at each token count ----------------------
        oracle = df.loc[df.groupby("tokens").median_ms.idxmin()]
        # ---- global best ---------------------------------------------------
        gkey, granked = pick_global_best(df)
        # ---- regime-aware --------------------------------------------------
        clusters = regime_clusters(df)
        rkeys = {}
        for name, toks in clusters.items():
            k, _ = pick_for_subset(df, toks)
            if k:
                rkeys[name] = k

        # ---- assemble the comparison table ---------------------------------
        for _, orow in oracle.iterrows():
            t = int(orow.tokens)
            rec = dict(model=model, tokens=t, M=int(orow.M),
                       default_ms=float(orow.default_ms),
                       oracle_ms=float(orow.median_ms),
                       oracle_key=orow.config_key)
            g = df[(df.tokens == t) & (df.config_key == gkey)]
            rec["global_ms"] = float(g.median_ms.iloc[0]) if len(g) else float("nan")
            rec["global_key"] = gkey
            cl = next((n for n, ts in clusters.items() if t in ts), None)
            rk = rkeys.get(cl)
            r = df[(df.tokens == t) & (df.config_key == rk)] if rk else None
            rec["regime_cluster"] = cl
            rec["regime_key"] = rk
            rec["regime_ms"] = float(r.median_ms.iloc[0]) if r is not None and len(r) else float("nan")
            for k in ("oracle", "global", "regime"):
                rec[f"{k}_speedup"] = rec["default_ms"] / rec[f"{k}_ms"] \
                    if rec.get(f"{k}_ms") else float("nan")
            # how much of the oracle gain does each strategy recover?
            og = rec["oracle_speedup"] - 1.0
            for k in ("global", "regime"):
                rec[f"{k}_pct_of_oracle"] = ((rec[f"{k}_speedup"] - 1.0) / og * 100
                                             if og > 1e-9 else float("nan"))
            all_strategy.append(rec)

        # ---- profiles for the transfer matrix ------------------------------
        profiles = {}
        for name, key in rkeys.items():
            profiles[name] = cfg_from_key(df, key)
        profiles["global_best"] = cfg_from_key(df, gkey)
        for _, orow in oracle.iterrows():
            profiles[f"oracle_t{int(orow.tokens)}"] = cfg_from_key(df, orow.config_key)
        (L.CONFIGS / f"{model}{a.suffix}_profiles.json").write_text(
            json.dumps(profiles, indent=2))

        # ---- SGLANG_MOE_CONFIG_DIR trees for E2E ---------------------------
        E, N = shape["num_experts"], shape["moe_intermediate_size"]
        fname = f"E={E},N={N},device_name=NVIDIA_H200.json"
        for pname, sel in (("global_best", {"global_best": None}),
                           ("regime_aware", None)):
            outdir = L.CONFIGS / "profiles" / f"{model}{a.suffix}_{pname}" / "configs" / "triton_3_5_1"
            outdir.mkdir(parents=True, exist_ok=True)
            table = {}
            for _, orow in oracle.iterrows():
                M = int(orow.M)
                if pname == "global_best":
                    table[str(M)] = cfg_from_key(df, gkey)
                else:
                    cl = next((n for n, ts in clusters.items()
                               if int(orow.tokens) in ts), None)
                    table[str(M)] = cfg_from_key(df, rkeys.get(cl, gkey))
            (outdir / fname).write_text(json.dumps(table, indent=2))
        # oracle profile dir (upper bound; per-M best)
        outdir = L.CONFIGS / "profiles" / f"{model}{a.suffix}_oracle" / "configs" / "triton_3_5_1"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / fname).write_text(json.dumps(
            {str(int(o.M)): cfg_from_key(df, o.config_key)
             for _, o in oracle.iterrows()}, indent=2))

        print(f"\n=== {model} ===")
        print(f"global best : {gkey}")
        for n, k in rkeys.items():
            print(f"regime {n:7s}: {k}")

    if all_sweep:
        pd.concat(all_sweep, ignore_index=True).to_csv(
            PROC / f"sweep_all{a.suffix}.csv", index=False)
    if all_strategy:
        s = pd.DataFrame(all_strategy).sort_values(["model", "tokens"])
        s.to_csv(PROC / f"strategy_comparison{a.suffix}.csv", index=False)
        print("\n=== strategy comparison (speedup over measured default) ===")
        print(s[["model", "tokens", "M", "default_ms", "oracle_speedup",
                 "global_speedup", "regime_speedup",
                 "regime_pct_of_oracle"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
