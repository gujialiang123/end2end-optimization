#!/usr/bin/env python3
"""Build a tuned MoE config from sweep results, using the guarded policy.

Specialise a bucket only where the measured speedup over the runtime default
clears a threshold; everywhere else write the default heuristic's own values, so
those buckets are field-for-field identical to what `get_default_config` would
have produced and the runtime behaviour there is unchanged.

Usage:
  rk_build_config.py --sweeps DIR --out FILE.json [--threshold 1.15]
                     [--buckets 1,2,4,...] [--report CSV]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
]
DEFAULT_BUCKETS = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96,
    128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192,
]


def default_config(M: int, E: int) -> dict:
    """Mirror of sglang's get_default_config for the non-quantised branch."""
    if M <= E:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True, help="dir of sweep_M*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=1.15)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--buckets", default=",".join(str(b) for b in DEFAULT_BUCKETS))
    ap.add_argument("--report", help="write a per-bucket provenance CSV")
    a = ap.parse_args()

    buckets = [int(b) for b in a.buckets.split(",")]
    sweeps = Path(a.sweeps)
    out, rows, missing = {}, [], []

    for M in buckets:
        f = sweeps / f"sweep_M{M}.json"
        if not f.exists():
            missing.append(M)
            continue
        j = json.loads(f.read_text())
        base = j["default_baseline"]["median_ms"]
        best = min(j["results"], key=lambda r: r["median_ms"])
        speedup = base / best["median_ms"]
        specialise = speedup >= a.threshold

        cfg = (
            {k: best[k] for k in FIELDS if k in best}
            if specialise
            else default_config(M, a.experts)
        )
        out[str(M)] = cfg
        rows.append(
            dict(
                M=M,
                default_ms=round(base, 5),
                best_ms=round(best["median_ms"], 5),
                speedup=round(speedup, 4),
                specialised=specialise,
                equals_default=cfg == default_config(M, a.experts),
                n_candidates=len(j["results"]) + len(j.get("failures", [])),
            )
        )

    if missing:
        raise SystemExit(f"missing sweeps for M={missing}")

    Path(a.out).write_text(json.dumps(out, indent=4) + "\n")

    print(f"{'M':>6} {'default':>9} {'best':>9} {'speedup':>9} {'specialised':>12}")
    for r in rows:
        print(
            f"{r['M']:>6} {r['default_ms']:9.4f} {r['best_ms']:9.4f} "
            f"{r['speedup']:8.3f}x {str(r['specialised']):>12}"
        )
    n_spec = sum(r["specialised"] for r in rows)
    print(f"\n{n_spec}/{len(rows)} buckets specialised (threshold {a.threshold}x)")
    print(f"wrote {a.out}")

    if a.report:
        with open(a.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
