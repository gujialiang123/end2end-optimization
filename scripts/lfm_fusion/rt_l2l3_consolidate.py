#!/usr/bin/env python3
"""Consolidate the L2/L3/L2+L3 ablation on the real/agentic RT_ workloads.

For each workload the exp3 harness produced four cells (2x2 of {MoE config
off/on} x {arm order fwd/rev}), each holding a `baseline` and an `all7` arm.
Mapping onto the ablation the mentor asked for:

    S0     = nocfg / baseline      (bare cookbook serving, no kernel work)
    L2     = cfg   / baseline      (tuned MoE config only)
    L3     = nocfg / all7          (kernel rewrite only)
    L2+L3  = cfg   / all7          (both)

Every delta below is reported against S0, orders pooled (n=16 per arm), with
the same exact Student-t tail exp3_analyze/exp3_latency use. Throughput is the
arrival-limited yardstick on these traces, so TTFT/E2E are reported alongside
it -- that is where an arrival-limited workload shows the win.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import exp3_analyze as A

E2E = A.E2E
OUTDIR = Path(__file__).resolve().parents[2] / "results/2026-08-10_rt_l2l3"

# (regime, label, kind)
WORKLOADS = [
    ("RT_tool_agent_x1", "Tool-Agent 1.0x", "real trace, arrival-limited"),
    ("RT_tool_agent_x2", "Tool-Agent 2.0x", "real trace"),
    ("RT_tool_agent_x3", "Tool-Agent 3.0x", "real trace, near knee"),
    ("RT_tool_agent_x4", "Tool-Agent 4.0x", "real trace, saturated"),
    ("RT_conversation_x2", "Conversation 2.0x", "real trace, generation-heavy"),
    ("RT_conversation_x4", "Conversation 4.0x", "real trace, generation-heavy"),
    ("RT_mooncake_generic_x2", "Mooncake arxiv 2.0x", "sibling of Tool-Agent"),
    ("RT_sharegpt_rate8", "ShareGPT 8 req/s", "real prompts, Poisson arrivals"),
    ("RT_sharegpt_rate16", "ShareGPT 16 req/s", "real prompts, Poisson arrivals"),
]

FIELDS = ["request_throughput", "ttft_p50_ms", "ttft_p95_ms",
          "tpot_p50_ms", "e2e_mean_ms", "e2e_p95_ms"]


def load_cell(prefix: str, regime: str, tag: str, field: str) -> dict[str, list[float]]:
    p = E2E / f"lfm25_exp3_{prefix}{tag}" / regime / "e2e_runs.json"
    rows = json.loads(p.read_text())
    out: dict[str, list[float]] = {}
    for r in rows:
        if r.get("status") == "ok" and r.get(field) is not None:
            out.setdefault(r["arm"], []).append(float(r[field]))
    return out


def pooled(prefix: str, regime: str, level: str, field: str) -> dict[str, list[float]]:
    fwd = load_cell(prefix, regime, f"{level}_fwd", field)
    rev = load_cell(prefix, regime, f"{level}_rev", field)
    # Union the arms across orders. A cell whose server launch_failed leaves one
    # arm with no ok rows in one order; fall back to whatever order has it rather
    # than dropping the whole workload. Reduced n is surfaced by delta()'s n.
    arms = set(fwd) | set(rev)
    return {arm: fwd.get(arm, []) + rev.get(arm, []) for arm in arms}


def cells_present(regime: str) -> bool:
    prefix = regime.split("_")[0] + "_"
    for level in ("nocfg", "cfg"):
        for order in ("fwd", "rev"):
            if not (E2E / f"lfm25_exp3_{prefix}{level}_{order}" / regime
                    / "e2e_runs.json").exists():
                return False
    return True


def delta(a: list[float], b: list[float]) -> dict:
    ratio, t, p = A.welch(a, b)
    return {"baseline": st.mean(a), "value": st.mean(b),
            "pct": (ratio - 1) * 100, "t": t, "p": p, "n": len(b)}


def main() -> None:
    results = []
    for regime, label, kind in WORKLOADS:
        if not cells_present(regime):
            print(f"skip {regime}: cells incomplete")
            continue
        prefix = regime.split("_")[0] + "_"
        row = {"regime": regime, "label": label, "kind": kind}
        for field in FIELDS:
            nocfg = pooled(prefix, regime, "nocfg", field)
            cfg = pooled(prefix, regime, "cfg", field)
            s0 = nocfg["baseline"]
            row[field] = {
                "S0_mean": st.mean(s0),
                "L2": delta(s0, cfg["baseline"]),
                "L3": delta(s0, nocfg["all7"]),
                "L2L3": delta(s0, cfg["all7"]),
            }
        results.append(row)
        print(f"ok  {label}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "rt_l2l3_ablation.json").write_text(json.dumps(results, indent=2))

    # flat CSV: one row per (workload, arm) with the six metrics' pct deltas
    with (OUTDIR / "rt_l2l3_ablation.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "kind", "arm"] +
                   [f"{f}_pct" for f in FIELDS] +
                   [f"{f}_p" for f in FIELDS])
        for row in results:
            for arm in ("L2", "L3", "L2L3"):
                w.writerow([row["label"], row["kind"], arm] +
                           [f"{row[f][arm]['pct']:.3f}" for f in FIELDS] +
                           [f"{row[f][arm]['p']:.2e}" for f in FIELDS])

    # markdown: throughput + TTFT p50 + E2E mean, the three the report uses
    lines = ["| Workload | arm | throughput | TTFT p50 | E2E mean |",
             "|---|---|---:|---:|---:|"]
    for row in results:
        for arm, name in (("L2", "L2"), ("L3", "L3"), ("L2L3", "L2+L3")):
            def cell(f):
                d = row[f][arm]
                mark = "" if d["p"] < 0.05 else " n.s."
                return f"{d['pct']:+.2f}%{mark}"
            lines.append(f"| {row['label']} | {name} | {cell('request_throughput')} "
                         f"| {cell('ttft_p50_ms')} | {cell('e2e_mean_ms')} |")
    (OUTDIR / "rt_l2l3_ablation.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUTDIR}/rt_l2l3_ablation.{{json,csv,md}}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
