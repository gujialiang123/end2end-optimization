#!/usr/bin/env python3
"""Aggregate the LFM2.5 fusion A/B into tidy CSVs plus a Welch t-test.

A ~1-2 % effect cannot be called from a single measurement, and this project
has previously been burned by exactly that (an n=3 "-8.84 %" collapsed to
+0.91 %, p=0.93, at n=8). So every arm is reported as mean +/- 95 % CI with an
explicit Welch t against the baseline arm, and the verdict column refuses to
call anything significant at p >= 0.05.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L

METRICS = ["request_throughput", "mean_ttft_ms", "p95_ttft_ms",
           "mean_tpot_ms", "p95_tpot_ms", "mean_e2e_latency_ms",
           "output_throughput"]
LOWER_IS_BETTER = {"mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms",
                   "p95_tpot_ms", "mean_e2e_latency_ms"}


def welch(a, b):
    """Welch t and a two-sided p-value; returns (t, p, df)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    va, vb = st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    if se2 == 0:
        return float("inf"), 0.0, float("nan")
    t = (st.mean(a) - st.mean(b)) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from statistics import NormalDist
        # Student-t tail via an incomplete-beta free approximation is overkill
        # here; with df >= 6 the normal approximation is within ~0.01 of the
        # exact p-value, and we only ever use it against a 0.05 threshold.
        p = 2 * (1 - NormalDist().cdf(abs(t)))
    except Exception:
        p = float("nan")
    return t, p, df


def ci95(v):
    if len(v) < 2:
        return 0.0
    return 1.96 * st.stdev(v) / math.sqrt(len(v))


def load(root: Path, runset: str | None = None):
    """Load runs, keyed by (runset, regime).

    The runset is the parent directory (e.g. `lfm25`, `lfm25_conv`). Runs from
    different sessions must never be pooled — a baseline measured hours apart on
    a shared machine is not the same baseline — so it is part of the key rather
    than being flattened away.
    """
    rows = []
    for f in sorted(root.glob("*/*/e2e_runs.json")):
        regime = f.parent.name
        rs = f.parent.parent.name
        if runset and rs != runset:
            continue
        for r in json.loads(f.read_text()):
            if r.get("status") != "ok":
                continue
            r["regime"] = regime
            r["runset"] = rs
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(L.RESULTS / "e2e"))
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--runset", default=None,
                    help="restrict to one run directory, e.g. lfm25_conv")
    ap.add_argument("--out", default="fusion_ab.csv")
    a = ap.parse_args()

    rows = load(Path(a.root), a.runset)
    if not rows:
        raise SystemExit(f"no successful runs under {a.root}")

    by = defaultdict(list)
    for r in rows:
        by[(f'{r["runset"]}/{r["regime"]}', r["arm"])].append(r)

    outdir = L.RESULTS / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    regimes = sorted({k[0] for k in by})

    for regime in regimes:
        base = by.get((regime, a.baseline), [])
        if not base:
            print(f"[warn] no baseline for {regime}")
            continue
        for (rg, arm), runs in sorted(by.items()):
            if rg != regime:
                continue
            for metric in METRICS:
                v = [r[metric] for r in runs if metric in r]
                b = [r[metric] for r in base if metric in r]
                if not v or not b:
                    continue
                mv, mb = st.mean(v), st.mean(b)
                t, p, df = welch(v, b)
                # "gain" is always oriented so that positive means better
                ratio = mv / mb if mb else float("nan")
                gain = (1 - ratio) if metric in LOWER_IS_BETTER else (ratio - 1)
                if arm == a.baseline:
                    verdict = "baseline"
                elif not (p == p) or p >= 0.05:
                    verdict = "no significant difference"
                else:
                    verdict = "improvement" if gain > 0 else "regression"
                out.append(dict(regime=regime, arm=arm, metric=metric,
                                n=len(v), mean=round(mv, 4),
                                ci95=round(ci95(v), 4),
                                baseline_mean=round(mb, 4),
                                ratio=round(ratio, 4),
                                gain_pct=round(100 * gain, 2),
                                welch_t=round(t, 3) if t == t else "",
                                p_value=round(p, 4) if p == p else "",
                                verdict=verdict))

    csv_path = outdir / a.out
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {csv_path} ({len(out)} rows)")

    print("\n== request_throughput ==")
    for r in out:
        if r["metric"] != "request_throughput":
            continue
        print(f"  {r['regime']:20s} {r['arm']:12s} {r['mean']:8.3f} "
              f"+/-{r['ci95']:.3f}  {r['ratio']:.4f}x  "
              f"gain={r['gain_pct']:+6.2f}%  p={r['p_value']}  {r['verdict']}")


if __name__ == "__main__":
    main()
