#!/usr/bin/env python3
"""Does the regime change the optimal kernel config BEYOND the shape it implies?

The runtime already dispatches per-M. So the interesting question is not "does
the config depend on M" (it does, and that is handled), but:

    at a FIXED M, does a different routing distribution change which config is
    optimal by an amount that actually costs performance?

Design: for each M, take the config tuned under uniform routing and the config
tuned under skewed routing, then measure BOTH under BOTH routings. A 2x2 matrix
per M. If the off-diagonal cells lose nothing, then kernel tuning is purely
shape-dependent and "regime-aware" adds nothing beyond "shape-aware".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rk_lib as L

HERE = Path(__file__).resolve().parent
PY = f"{L.ENVDIR}/bin/python"
RAW = L.RESULTS / "raw" / "routing"
OUT = L.RESULTS / "raw" / "routing_cross"


def tuned_config(model: str, tokens: int, routing: str) -> dict | None:
    f = RAW / model / f"ctrl_t{tokens}_{routing}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if not d["results"]:
        return None
    r = d["results"][0]
    return {k: int(r[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                                   "GROUP_SIZE_M", "num_warps", "num_stages")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm25")
    ap.add_argument("--tokens", default="8,32,64,512")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    env = L.run_env()
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    for tok in [int(x) for x in a.tokens.split(",")]:
        cfgs, names = [], []
        for src in ("uniform", "skewed"):
            c = tuned_config(a.model, tok, src)
            if c is not None:
                cfgs.append(c)
                names.append(f"tuned_on_{src}")
        if len(cfgs) < 2:
            print(f"[skip] t={tok}: need both routing tunings")
            continue
        cf = OUT / f"{a.model}_t{tok}_cands.json"
        cf.write_text(json.dumps(cfgs))
        for test in ("uniform", "skewed"):
            out = OUT / f"{a.model}_t{tok}_testedon_{test}.json"
            if out.exists():
                continue
            cmd = [PY, str(HERE / "rk_microbench.py"), "--model", a.model,
                   "--tokens", str(tok), "--routing", test, "--bias",
                   "--configs", str(cf), "--out", str(out),
                   "--warmup", str(a.warmup), "--iters", str(a.iters),
                   "--repeats", str(a.repeats)]
            log = out.with_suffix(".log")
            with open(log, "w") as lf:
                subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            print(f"t={tok} tested_on={test} -> {out.name}")

    # ---- assemble the cross matrix -------------------------------------
    rows = []
    for tok in [int(x) for x in a.tokens.split(",")]:
        key = {}
        for src in ("uniform", "skewed"):
            c = tuned_config(a.model, tok, src)
            if c:
                key[L.config_key(c)] = f"tuned_on_{src}"
        for test in ("uniform", "skewed"):
            f = OUT / f"{a.model}_t{tok}_testedon_{test}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            base = d["default_baseline"]["median_ms"]
            best = min((r["median_ms"] for r in d["results"]), default=None)
            for r in d["results"]:
                nm = key.get(r["config_key"])
                if nm is None:
                    continue
                rows.append(dict(model=a.model, tokens=tok, tested_on=test,
                                 profile=nm, median_ms=r["median_ms"],
                                 speedup_vs_default=base / r["median_ms"],
                                 vs_best_here=best / r["median_ms"]))
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        p = L.RESULTS / "processed" / "routing_cross_matrix.csv"
        df.to_csv(p, index=False)
        print(f"\nwrote {p}")
        print(df.pivot_table(index=["tokens", "profile"], columns="tested_on",
                             values="speedup_vs_default").round(4).to_string())
        print("\n=== cost of using the config tuned on the OTHER routing ===")
        for tok, g in df.groupby("tokens"):
            for test in ("uniform", "skewed"):
                s = g[g.tested_on == test]
                if len(s) < 2:
                    continue
                match = s[s.profile == f"tuned_on_{test}"]
                other = s[s.profile != f"tuned_on_{test}"]
                if match.empty or other.empty:
                    continue
                loss = (1 - other.speedup_vs_default.iloc[0] /
                        match.speedup_vs_default.iloc[0]) * 100
                print(f"  M={tok:>4} tested_on={test:8s}: mismatched config "
                      f"costs {loss:+.2f}%")


if __name__ == "__main__":
    main()
