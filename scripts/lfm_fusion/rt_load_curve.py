#!/usr/bin/env python3
"""Turn the Tool-Agent arrival-load sweep into a saturation curve.

The question this answers is why a real trace shows +0.4 % throughput while
every synthetic workload shows +6 to +8 %. At 1x the Mooncake replay is
arrival-limited -- the client sleeps until each record's own timestamp, so the
server spends most of the run idle and cannot retire requests it has not been
given. The kernel work still makes each request faster, which shows up in
latency and nowhere else.

If that reading is right, then raising the arrival rate should convert the same
saving into throughput, and the crossover should line up with the point where
the server stops being idle. So this reports, per load level:

  * offered vs achieved req/s, which diverge once the server is the bottleneck;
  * in-flight requests from Little's Law, against the client concurrency cap;
  * the running/queued depth the server itself logged, which is the direct
    evidence rather than an inference;
  * the L3 delta on throughput and on latency, side by side.

  python scripts/lfm_fusion/rt_load_curve.py
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp3_analyze as A  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results/2026-08-07_real_trace_study"
# regime -> (label, offered req/s from the trace characterisation)
LEVELS = [
    ("RT_tool_agent_x1", "1.0x", 5.6),
    ("RT_tool_agent_x133", "1.33x", 7.1),
    ("RT_tool_agent_x2", "2.0x", 10.7),
    ("RT_tool_agent_x3", "3.0x", 16.8),
    ("RT_tool_agent_x4", "4.0x", 22.2),
]
CAP = 128
RUNNING = re.compile(r"#running-req: (\d+)")
QUEUE = re.compile(r"#queue-req: (\d+)")


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


def server_depth(regime: str) -> dict[str, tuple[float, float, float]]:
    """Peak and mean running/queued depth, read from the server's own log."""
    res = {}
    for arm in ("baseline", "all7"):
        runs, queues = [], []
        for d in ("fwd", "rev"):
            log = A.E2E / f"lfm25_rt_{d}" / regime / f"server_{arm}.log"
            if not log.exists():
                continue
            txt = log.read_text(errors="ignore")
            runs += [int(m) for m in RUNNING.findall(txt)]
            queues += [int(m) for m in QUEUE.findall(txt)]
        if runs:
            res[arm] = (st.mean(runs), max(runs), max(queues) if queues else 0)
    return res


def main() -> None:
    rows = []
    print("Tool-Agent arrival-load sweep on LFM2.5, S0 vs S0+L3 (all7)")
    print("baseline = stock sglang; only LFM_FUSION_PATCH differs\n")
    print(f"{'load':<7}{'offered':>9}{'achieved':>10}{'in-flight':>11}"
          f"{'srv run':>9}{'queued':>8}{'thr gain':>10}{'p':>10}")
    for regime, label, offered in LEVELS:
        thr = pooled(regime, "request_throughput")
        if "baseline" not in thr or "all7" not in thr:
            print(f"{label:<7}   (incomplete)")
            continue
        e2e = pooled(regime, "e2e_mean_ms")
        base_thr, base_e2e = st.mean(thr["baseline"]), st.mean(e2e["baseline"])
        n = base_thr * base_e2e / 1000.0
        depth = server_depth(regime)
        srun, speak, squeue = depth.get("baseline", (0, 0, 0))
        r, t, pv = A.welch(thr["baseline"], thr["all7"])
        print(f"{label:<7}{offered:>9.1f}{base_thr:>10.2f}{n:>11.2f}"
              f"{srun:>9.1f}{squeue:>8.0f}{(r - 1) * 100:>9.2f}%{pv:>10.1e}")

        row = {"regime": regime, "label": label, "offered_req_s": offered,
               "achieved_req_s": base_thr, "in_flight": n,
               "client_cap": CAP, "utilisation": n / CAP,
               "server_running_mean": srun, "server_running_peak": speak,
               "server_queue_peak": squeue,
               "throughput_gain_pct": (r - 1) * 100, "throughput_p": pv,
               "n_per_arm": len(thr["baseline"])}
        for fld in ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
                    "e2e_mean_ms", "e2e_p95_ms"):
            d = pooled(regime, fld)
            if "baseline" in d and "all7" in d:
                rr, _, pp = A.welch(d["baseline"], d["all7"])
                row[f"{fld}_baseline"] = st.mean(d["baseline"])
                row[f"{fld}_gain_pct"] = (rr - 1) * 100
                row[f"{fld}_p"] = pp
        rows.append(row)

    print(f"\n{'load':<7}{'TTFT p50':>11}{'E2E mean':>11}{'thr':>10}"
          f"   <- negative latency is better")
    for row in rows:
        print(f"{row['label']:<7}{row.get('ttft_p50_ms_gain_pct', 0):>10.2f}%"
              f"{row.get('e2e_mean_ms_gain_pct', 0):>10.2f}%"
              f"{row['throughput_gain_pct']:>9.2f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toolagent_load_sweep.json").write_text(json.dumps(rows, indent=2))
    if rows:
        cols = list(rows[0])
        with (OUT / "toolagent_load_sweep.csv").open("w") as f:
            f.write(",".join(cols) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"\nwrote {OUT}/toolagent_load_sweep.{{json,csv}}")


if __name__ == "__main__":
    main()
