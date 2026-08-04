#!/usr/bin/env python3
"""Latency for the cells where throughput is the wrong yardstick.

On the tool-agent trace, request throughput is set by the trace's own think
time, not by how fast the server is: both arms finish the same 200 requests in
the same wall clock and the kernel work shows up as +0.4 %. That is not "no
effect", it is the wrong metric. The same run's TTFT moves by 7-8 %.

Reports every latency field lf_e2e records, pooled across arm orders, with the
same exact Student-t tail exp3_analyze uses.

  python scripts/lfm_fusion/exp3_latency.py --regime F_tool_agent
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import exp3_analyze as A

FIELDS = ["ttft_p50_ms", "ttft_p95_ms", "ttft_mean_ms",
          "tpot_p50_ms", "tpot_p95_ms", "tpot_mean_ms",
          "e2e_p50_ms", "e2e_p95_ms", "e2e_mean_ms",
          "request_throughput"]


def load_field(tag: str, field: str) -> dict[str, list[float]]:
    import json
    rows = json.loads(
        (A.E2E / f"lfm25_exp3_{A.PREFIX}{tag}" / A.REGIME / "e2e_runs.json").read_text())
    out: dict[str, list[float]] = {}
    for r in rows:
        if r.get("status") == "ok" and r.get(field) is not None:
            out.setdefault(r["arm"], []).append(float(r[field]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--suite", default="")
    ap.add_argument("--level", default="nocfg", choices=["nocfg", "cfg"])
    a = ap.parse_args()
    A.REGIME = a.regime
    A.PREFIX = a.suite + ("" if a.regime == "C_long_prefill"
                          else a.regime.split("_")[0] + "_")

    print(f"regime {a.regime}, level {a.level}, orders pooled\n")
    print(f"{'metric':<20}{'baseline':>12}{'all7':>12}{'change':>10}"
          f"{'t':>8}{'p':>11}   better")
    for f in FIELDS:
        try:
            fwd, rev = load_field(f"{a.level}_fwd", f), load_field(f"{a.level}_rev", f)
        except FileNotFoundError:
            print(f"{f:<20}  (cells not present)")
            continue
        if "baseline" not in fwd or "all7" not in fwd:
            continue
        b = fwd["baseline"] + rev["baseline"]
        k = fwd["all7"] + rev["all7"]
        ratio, t, p = A.welch(b, k)
        pct = (ratio - 1) * 100
        lower_is_better = f != "request_throughput"
        good = (pct < 0) if lower_is_better else (pct > 0)
        mark = ("better" if good else "worse") if p < 0.05 else "n.s."
        print(f"{f:<20}{st.mean(b):>12.3f}{st.mean(k):>12.3f}{pct:>9.2f}%"
              f"{t:>8.2f}{p:>11.1e}   {mark}")

    print("\nlower is better for every metric except request_throughput.")


if __name__ == "__main__":
    main()
