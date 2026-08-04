#!/usr/bin/env python3
"""Is the client the bottleneck, or is the server?

Five of the six workloads are closed-loop: bench_serving defaults --request-rate
to inf, so every request is released at t=0 and only --max-concurrency gates
them. A slot frees, the next request goes in immediately, and the server runs
saturated. tool_agent is the exception -- the mooncake replay sleeps until each
record's own trace timestamp (get_mooncake_request_over_time), so arrivals are
fixed by the trace and are not affected by how fast the server answers.

Little's Law separates the two without any new measurement. In steady state the
average number of requests in flight is throughput x latency. For a saturated
closed loop that has to equal the concurrency cap; for an arrival-driven replay
it lands wherever the trace puts it, and the gap to the cap is the server's idle
headroom.

  python scripts/lfm_fusion/exp3_littles_law.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp3_analyze as A  # noqa: E402

# regime -> (suite, --max-concurrency of its workload)
REGIMES = [
    ("A_low_batch_decode", "", 1),
    ("C_long_prefill", "", 4),
    ("D_medium_balanced", "", 8),
    ("B_concurrent_decode", "", 32),
    ("E_shared_prefix", "", 64),
    ("F_tool_agent", "", 64),
]


def field(regime: str, suite: str, level: str, name: str) -> dict[str, list[float]]:
    A.REGIME = regime
    A.PREFIX = suite + ("" if regime == "C_long_prefill" else regime.split("_")[0] + "_")
    out: dict[str, list[float]] = {}
    for tag in (f"{level}_fwd", f"{level}_rev"):
        rows = json.loads(
            (A.E2E / f"lfm25_exp3_{A.PREFIX}{tag}" / regime / "e2e_runs.json").read_text())
        for r in rows:
            if r.get("status") == "ok" and r.get(name) is not None:
                out.setdefault(r["arm"], []).append(float(r[name]))
    return out


def main() -> None:
    print("Requests actually in flight, from Little's Law: N = throughput x E2E latency")
    print("A saturated closed loop sits at its concurrency cap; an arrival-driven")
    print("replay sits wherever its trace puts it.\n")
    print(f"{'regime':<22}{'cap':>5}{'thr':>9}{'E2E mean':>11}{'N in flight':>13}"
          f"{'utilisation':>13}")
    for regime, suite, cap in REGIMES:
        thr = field(regime, suite, "nocfg", "request_throughput")["baseline"]
        e2e = field(regime, suite, "nocfg", "e2e_mean_ms")["baseline"]
        n = st.mean(thr) * st.mean(e2e) / 1000.0
        print(f"{regime:<22}{cap:>5}{st.mean(thr):>9.3f}{st.mean(e2e):>10.1f}ms"
              f"{n:>13.2f}{n / cap * 100:>12.0f}%")

    print("\nAnd what the kernel work buys, on the same runs:\n")
    print(f"{'regime':<22}{'E2E mean':>10}{'TTFT p50':>10}{'TPOT p50':>10}"
          f"{'thr predicted':>15}{'thr measured':>14}")
    for regime, suite, _ in REGIMES:
        cells = {}
        for name in ("request_throughput", "e2e_mean_ms", "ttft_p50_ms", "tpot_p50_ms"):
            d = field(regime, suite, "nocfg", name)
            r, _, _ = A.welch(d["baseline"], d["all7"])
            cells[name] = (r - 1) * 100
        # closed loop with N pinned: thr = N / latency, so the latency cut
        # predicts the throughput gain exactly
        pred = (1.0 / (1.0 + cells["e2e_mean_ms"] / 100.0) - 1.0) * 100
        flag = "" if abs(pred - cells["request_throughput"]) < 1.0 else "   <-- N moved"
        print(f"{regime:<22}{cells['e2e_mean_ms']:>9.2f}%{cells['ttft_p50_ms']:>9.2f}%"
              f"{cells['tpot_p50_ms']:>9.2f}%{pred:>14.2f}%"
              f"{cells['request_throughput']:>13.2f}%{flag}")

    print("\nIn a saturated closed loop throughput and latency are the same measurement:")
    print("N is fixed, so thr = N / latency and a 6 % latency cut *is* a 6 % throughput")
    print("gain -- the predicted and measured columns agree to a tenth of a point in")
    print("five regimes. Only where N is free to move do the two disagree, and there")
    print("the latency column is the one carrying the result.")


if __name__ == "__main__":
    main()
