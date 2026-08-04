#!/usr/bin/env python3
"""Cross-model gap table for the 2026-08-04 campaign, against the Qwen control.

Counts, not time shares. A count that divides evenly by the layer count means
every layer makes the same mistake, which is what separates an implementation
gap from a hot kernel that is simply expensive. The control column is what makes
the absolute numbers mean anything: Qwen3-30B is the framework's most-optimised
model, so whatever it does not do is what the others are doing needlessly.

  python scripts/lfm_fusion/gap_table_2026_08_04.py
"""
from __future__ import annotations

import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[2] / "results/lfm_fusion/audit"
OUT = AUDIT.parent / "processed/gap_table_2026_08_04.json"
MODELS = ["olmo2", "falconh1", "qwen"]
REGIMES = ["A_low_batch_decode", "B_concurrent_decode", "C_long_prefill"]


def load(model: str, regime: str, tag: str = ""):
    p = AUDIT / f"{model}_{regime}{tag}" / "audit.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    report: dict = {}
    for regime in REGIMES:
        for stage in ("decode", "prefill"):
            rows = {}
            layers = {}
            for m in MODELS:
                d = load(m, regime)
                if not d or stage not in d["stages"]:
                    continue
                layers[m] = d["layers"]
                rows[m] = {g["gap"]: g for g in d["stages"][stage]["fusion_gaps"]}
            if len(rows) < 2:
                continue
            print(f"\n########## {regime} / {stage}")
            print("   layers: " + ", ".join(f"{m}={layers[m]}" for m in rows))
            allgaps = sorted(
                {k for r in rows.values() for k in r},
                key=lambda k: -max(r.get(k, {}).get("pct_of_kernel_time", 0)
                                   for r in rows.values()),
            )
            hdr = f"{'gap':<20}"
            for m in rows:
                hdr += f"{m + ' calls':>14}{'/layer':>8}{'%time':>7}"
            print(hdr)
            for k in allgaps:
                line = f"{k:<20}"
                for m in rows:
                    g = rows[m].get(k)
                    if g:
                        line += (f"{g['calls']:>14}{g['calls_per_layer']:>8.2f}"
                                 f"{g['pct_of_kernel_time']:>6.2f}%")
                    else:
                        line += f"{0:>14}{0:>8.2f}{0:>6.2f}%"
                print(line)
            report[f"{regime}/{stage}"] = {
                m: {k: dict(calls=g["calls"], per_layer=g["calls_per_layer"],
                            pct=g["pct_of_kernel_time"],
                            removable=g["removable_by_fusion"])
                    for k, g in r.items()}
                for m, r in rows.items()
            }

    # the CUDA-graph pair is the decisive experiment for olmo2
    off, on = load("olmo2", "A_low_batch_decode"), load("olmo2", "A_low_batch_decode", "_cg")
    if off and on:
        o = {g["gap"]: g for g in off["stages"]["decode"]["fusion_gaps"]}
        n = {g["gap"]: g for g in on["stages"]["decode"]["fusion_gaps"]}
        t_off = off["stages"]["decode"]["total_kernel_us"]
        t_on = on["stages"]["decode"]["total_kernel_us"]
        print("\n########## olmo2 decode: CUDA graph off vs on")
        print("   The capture-mode branch in _apply_qk_norm is the only thing that")
        print("   differs, so whatever disappears here shares that one root cause.")
        print(f"{'gap':<20}{'graph off':>12}{'graph on':>12}")
        for k in sorted(set(o) | set(n)):
            print(f"{k:<20}{o.get(k, {}).get('calls', 0):>12}"
                  f"{n.get(k, {}).get('calls', 0):>12}")
        print(f"{'total kernel us':<20}{t_off:>12.0f}{t_on:>12.0f}"
              f"   -> {(1 - t_on / t_off) * 100:.1f}% of decode kernel time")
        report["olmo2_cuda_graph_pair"] = {
            "off": {k: g["calls"] for k, g in o.items()},
            "on": {k: g["calls"] for k, g in n.items()},
            "total_kernel_us_off": t_off, "total_kernel_us_on": t_on,
            "share_of_decode_kernel_time": 1 - t_on / t_off,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
