#!/usr/bin/env python3
"""What the serving-knob ceiling actually is on LFM2.5, per regime.

The layered story treats the sglang cookbook config as "the best autotuning
config". On four of the six workloads that is defensible; on long prefill and
shared prefix it is not, and the difference is large enough to decide which
regime the deliverable should be built on.

Source is the 2026-07-24 campaign: all 192 serving configs at n=1, then the top
35 re-measured at n=5. Only the n=5 validation is reported here, because
R_long_prefill runs for ~0.3 s and taking a max over 192 single measurements is
mostly a max over noise.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAL = REPO / "results/2026-07-24_serving_ceiling_validation/per_config_workload_metrics.csv"
OUT = REPO / "results/2026-07-24_serving_ceiling_validation/analysis/lfm25/ceiling_per_regime.json"


def main() -> None:
    by_wl: dict[str, list[dict]] = defaultdict(list)
    for r in csv.DictReader(VAL.open()):
        if r["model"] == "lfm25":
            by_wl[r["workload"]].append(r)

    report = {}
    print(f"{'workload':<22}{'cookbook':>10}{'ceiling':>10}{'gain':>9}"
          f"{'ceiling config':<34}{'TTFT p95 ms':>22}")
    for wl, rows in by_wl.items():
        cook = next(r for r in rows if r["is_cookbook"] == "1")
        best = max(rows, key=lambda r: float(r["request_throughput"]))
        c, b = float(cook["request_throughput"]), float(best["request_throughput"])
        gain = b / c - 1
        print(f"{wl:<22}{c:>10.3f}{b:>10.3f}{gain*100:>8.1f}%  {best['hash']:<34}"
              f"{float(cook['ttft_p95_ms']):>10.1f} ->{float(best['ttft_p95_ms']):>9.1f}")
        report[wl] = {
            "n_configs_validated": len(rows),
            "cookbook": {"hash": cook["hash"],
                         "req_per_s": c,
                         "ci95": float(cook["request_throughput_ci95"]),
                         "ttft_p95_ms": float(cook["ttft_p95_ms"]),
                         "tpot_p95_ms": float(cook["tpot_p95_ms"])},
            "ceiling": {"hash": best["hash"],
                        "req_per_s": b,
                        "ci95": float(best["request_throughput_ci95"]),
                        "ttft_p95_ms": float(best["ttft_p95_ms"]),
                        "tpot_p95_ms": float(best["tpot_p95_ms"])},
            "gain_over_cookbook": gain,
        }

    print("\nlong prefill, ranked by the knob that actually matters")
    lp = sorted(by_wl["R_long_prefill"], key=lambda r: -float(r["request_throughput"]))
    per_chunk: dict[str, list[float]] = defaultdict(list)
    for r in lp:
        per_chunk[r["chunk"]].append(float(r["request_throughput"]))
    for chunk, vals in sorted(per_chunk.items(), key=lambda kv: -max(kv[1])):
        print(f"  chunked_prefill_size={chunk:<6} n={len(vals):<3} "
              f"best={max(vals):6.3f}  worst={min(vals):6.3f} req/s"
              f"{'   <== cookbook disables chunking' if chunk == '-1' else ''}")
    report["R_long_prefill"]["by_chunked_prefill_size"] = {
        k: {"n": len(v), "best": max(v), "worst": min(v)} for k, v in per_chunk.items()
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
