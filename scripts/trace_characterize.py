#!/usr/bin/env python3
"""Characterise the Mooncake replay traces before spending any GPU time.

Every one of these traces is replayed by timestamp, so the offered load is a
property of the file, not of the benchmark flags. Two numbers decide whether a
run is usable at all:

  * how many records are needed for a >=30 s scored window at a given slowdown
    factor -- the 200-record default spans 36 s at 1x and only 9 s at 4x;
  * what the offered arrival rate is, which says whether a load level will
    saturate the server or merely tickle it.

The prompt text is built from `hash_ids`, not from the `input_length` field
(bench_serving.py:1126-1131 emits 128 "hi" tokens per hash id), so input size
has to be estimated from the hash-id count rather than read off the record.

  python scripts/trace_characterize.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

TRACES = {
    "toolagent": "/tmp/toolagent_trace.jsonl",
    "conversation": "/tmp/conversation_trace.jsonl",
    "mooncake": "/tmp/mooncake_trace.jsonl",
}
OUT = Path(__file__).resolve().parents[1] / "results/2026-08-07_real_trace_study"
# slowdown factor -> arrival multiplier; the replay sleeps to
# trace_ts * slowdown, so a smaller factor means requests arrive sooner
LOADS = [(1.0, "1.0x"), (0.75, "1.33x"), (0.5, "2.0x"), (0.33, "3.0x"),
         (0.25, "4.0x")]
TARGET_WINDOW_S = 30.0
CANDIDATE_N = [200, 400, 800, 1600, 3200, 6400]


def pct(vals: list, p: float) -> float:
    s = sorted(vals)
    return s[min(int(len(s) * p), len(s) - 1)]


def characterise(name: str, path: str) -> dict:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs.sort(key=lambda r: r["timestamp"])
    t0 = recs[0]["timestamp"]
    span_s = (recs[-1]["timestamp"] - t0) / 1000.0

    hid = [len(r.get("hash_ids", [])) for r in recs]
    out = [r.get("output_length", 0) for r in recs]
    inp = [r.get("input_length", 0) for r in recs]

    # arrivals per wall-clock second, over the first 3200 records
    head = recs[: min(3200, len(recs))]
    per_sec = Counter(int((r["timestamp"] - t0) / 1000) for r in head)
    counts = list(per_sec.values())

    # smallest record count that still gives a usable window at each load
    plan = {}
    for sd, label in LOADS:
        chosen = None
        for n in CANDIDATE_N:
            if n > len(recs):
                break
            w = (recs[n - 1]["timestamp"] - t0) / 1000.0 * sd
            if w >= TARGET_WINDOW_S:
                chosen = {"num_prompts": n, "window_s": round(w, 1),
                          "offered_req_s": round(n / w, 2)}
                break
        plan[label] = chosen or {"num_prompts": None,
                                 "note": "trace too short for a 30 s window"}

    rec = {
        "records": len(recs),
        "span_s": round(span_s, 1),
        "mean_arrival_req_s": round(len(recs) / span_s, 2),
        "arrival_per_s_p50": st.median(counts),
        "arrival_per_s_peak": max(counts),
        "hash_ids_p50": pct(hid, 0.5), "hash_ids_p95": pct(hid, 0.95),
        "est_input_tokens_p50": pct(hid, 0.5) * 128,
        "est_input_tokens_p95": pct(hid, 0.95) * 128,
        "output_len_p50": pct(out, 0.5), "output_len_p95": pct(out, 0.95),
        "input_len_field_p50": pct(inp, 0.5),
        "load_plan": plan,
    }
    print(f"\n=== {name}: {rec['records']} records over {rec['span_s']}s "
          f"({rec['mean_arrival_req_s']} req/s mean, "
          f"peak {rec['arrival_per_s_peak']}/s)")
    print(f"    hash_ids p50={rec['hash_ids_p50']} p95={rec['hash_ids_p95']}"
          f"  -> est input ~{rec['est_input_tokens_p50']} / "
          f"{rec['est_input_tokens_p95']} tok")
    print(f"    output_length p50={rec['output_len_p50']} "
          f"p95={rec['output_len_p95']}")
    print(f"    {'load':<8}{'slowdown':>10}{'num_prompts':>13}{'window':>9}"
          f"{'offered':>10}")
    for sd, label in LOADS:
        p = plan[label]
        if p.get("num_prompts"):
            print(f"    {label:<8}{sd:>10}{p['num_prompts']:>13}"
                  f"{p['window_s']:>8.0f}s{p['offered_req_s']:>9.1f}/s")
        else:
            print(f"    {label:<8}{sd:>10}     -- too short --")
    return rec


def main() -> None:
    report = {}
    for name, path in TRACES.items():
        if not Path(path).exists():
            print(f"{name}: MISSING at {path}")
            continue
        report[name] = characterise(name, path)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "trace_characterization.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
