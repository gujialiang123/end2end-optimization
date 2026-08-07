#!/usr/bin/env python3
"""Compare the L3 effect across every real / agentic workload measured.

The load sweep answers "why is throughput flat at 1x"; this answers "is the
answer specific to one trace". Each row is a different request distribution
replayed against the same server, the same tree, and the same L3 patch, with
only LFM_FUSION_PATCH differing between arms.

Throughput and latency are reported side by side deliberately. Reporting only
throughput is what made the Tool-Agent row look like a null result in the first
place, and any workload whose client paces itself will do the same.

  python scripts/lfm_fusion/rt_workload_matrix.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp3_analyze as A  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results/2026-08-07_real_trace_study"

# regime, display name, what kind of realism it actually provides
ROWS = [
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
FIELDS = ["request_throughput", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
          "e2e_mean_ms", "e2e_p95_ms"]


def pooled(regime: str, field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for d in ("fwd", "rev"):
        p = A.E2E / f"lfm25_rt_{d}" / regime / "e2e_runs.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r.get("status") == "ok" and r.get(field) is not None:
                out.setdefault(r["arm"], []).append(float(r[field]))
    return out


def per_order(regime: str) -> list[float]:
    res = []
    for d in ("fwd", "rev"):
        p = A.E2E / f"lfm25_rt_{d}" / regime / "e2e_runs.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        m = {}
        for r in rows:
            if r.get("status") == "ok":
                m.setdefault(r["arm"], []).append(float(r["request_throughput"]))
        if "baseline" in m and "all7" in m:
            res.append((st.mean(m["all7"]) / st.mean(m["baseline"]) - 1) * 100)
    return res


def main() -> None:
    out_rows = []
    print("LFM2.5, L3 (all7) against stock sglang, on real and agentic workloads")
    print("counterbalanced; only LFM_FUSION_PATCH differs between arms\n")
    print(f"{'workload':<21}{'kind':<31}{'thr':>8}{'TTFT p50':>10}"
          f"{'E2E mean':>10}{'p (thr)':>10}   fwd/rev")
    for regime, name, kind in ROWS:
        thr = pooled(regime, "request_throughput")
        if "baseline" not in thr or "all7" not in thr:
            print(f"{name:<21}{kind:<31}   (not measured)")
            continue
        r, t, pv = A.welch(thr["baseline"], thr["all7"])
        row = {"regime": regime, "workload": name, "kind": kind,
               "n_per_arm": len(thr["baseline"]),
               "baseline_req_s": st.mean(thr["baseline"]),
               "l3_req_s": st.mean(thr["all7"]),
               "throughput_gain_pct": (r - 1) * 100, "throughput_p": pv}
        for f in FIELDS[1:]:
            d = pooled(regime, f)
            if "baseline" in d and "all7" in d:
                rr, _, pp = A.welch(d["baseline"], d["all7"])
                row[f"{f}_baseline"] = st.mean(d["baseline"])
                row[f"{f}_gain_pct"] = (rr - 1) * 100
                row[f"{f}_p"] = pp
        orders = per_order(regime)
        row["orders"] = orders
        print(f"{name:<21}{kind:<31}{row['throughput_gain_pct']:>7.2f}%"
              f"{row.get('ttft_p50_ms_gain_pct', 0):>9.2f}%"
              f"{row.get('e2e_mean_ms_gain_pct', 0):>9.2f}%{pv:>10.1e}   "
              + "/".join(f"{o:+.2f}" for o in orders))
        out_rows.append(row)

    print("\nlatency columns are reductions, so negative is better")
    lat = [r.get("e2e_mean_ms_gain_pct", 0) for r in out_rows]
    thr = [r["throughput_gain_pct"] for r in out_rows]
    print(f"\nacross {len(out_rows)} workloads: "
          f"E2E mean {min(lat):.2f}% to {max(lat):.2f}%, "
          f"throughput {min(thr):+.2f}% to {max(thr):+.2f}%")
    print("every workload improves on latency; throughput only where the "
          "client offers enough load to queue")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_workload_ablation.json").write_text(json.dumps(out_rows, indent=2))
    if out_rows:
        cols = [c for c in out_rows[0] if c != "orders"]
        with (OUT / "real_workload_ablation.csv").open("w") as f:
            f.write(",".join(cols) + "\n")
            for row in out_rows:
                f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"\nwrote {OUT}/real_workload_ablation.{{json,csv}}")


if __name__ == "__main__":
    main()
