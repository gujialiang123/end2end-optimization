#!/usr/bin/env python3
"""Analyse the 2x2 layered experiment on LFM2.5 regime C.

Cells: {tuned MoE config off, on} x {arm order forward, reversed}. Orders are
pooled, so each arm of each config level has n=16 scored repetitions drawn from
two independent server lifetimes -- which is what makes the comparison robust
to the position effect this harness is known to have.

Reports, for each config level, the kernel increment (all7 vs baseline), and
across levels the config increment and the full stack, with exact Student-t
tails rather than a normal approximation (n is small enough that the normal
tail is anti-conservative).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

E2E = Path(__file__).resolve().parents[2] / "results/lfm_fusion/e2e"
METRIC = "request_throughput"
REGIME = "C_long_prefill"
PREFIX = ""          # regime C cells were tagged before the script took a regime


def load(tag: str) -> dict[str, list[float]]:
    rows = json.loads(
        (E2E / f"lfm25_exp3_{PREFIX}{tag}" / REGIME / "e2e_runs.json").read_text())
    out: dict[str, list[float]] = {}
    for r in rows:
        if r.get("status") == "ok":
            out.setdefault(r["arm"], []).append(float(r[METRIC]))
    return out


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Return (ratio b/a, t, two-sided p) with the exact Student-t tail."""
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    t = (mb - ma) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    # two-sided tail of Student-t via the regularised incomplete beta
    x = df / (df + t * t)
    p = _betainc(df / 2, 0.5, x)
    return mb / ma, t, p


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b), continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    r = front * (f - 1.0)
    return r if x < (a + 1) / (a + b + 2) else 1.0 - _betainc(b, a, 1 - x)


def pooled(level: str) -> dict[str, list[float]]:
    fwd, rev = load(f"{level}_fwd"), load(f"{level}_rev")
    return {arm: fwd[arm] + rev[arm] for arm in fwd}


def fmt(name: str, a: list[float], b: list[float]) -> str:
    ratio, t, p = welch(a, b)
    return (f"{name:<46} {st.mean(a):7.3f} -> {st.mean(b):7.3f} req/s   "
            f"{(ratio - 1) * 100:+6.2f}%   t={t:6.2f}  p={p:.2e}   "
            f"n={len(a)}/{len(b)}")


def main() -> None:
    global REGIME, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="C_long_prefill")
    ap.add_argument("--suite", default="",
                    help="tag prefix of an extra arm suite, e.g. 'six_'")
    ap.add_argument("--arms", default="",
                    help="comma-separated arms to compare, baseline first; "
                         "defaults to every arm present in the cells")
    a = ap.parse_args()
    REGIME = a.regime
    PREFIX = a.suite + ("" if REGIME == "C_long_prefill"
                        else REGIME.split("_")[0] + "_")
    print(f"regime {REGIME}\n")
    nocfg, cfg = pooled("nocfg"), pooled("cfg")

    print("per-cell means (each from one server lifetime, n=8)")
    for level in ("nocfg", "cfg"):
        for order in ("fwd", "rev"):
            d = load(f"{level}_{order}")
            cells = "  ".join(
                f"{k}={st.mean(v):7.3f}+/-{st.pstdev(v):5.3f}" for k, v in d.items()
            )
            print(f"  {level:<6} {order:<4} {cells}")

    print("\ncounterbalanced comparisons (orders pooled, n=16 per arm)")
    other = a.arms.split(",")[1] if a.arms else \
        next(k for k in nocfg if k != "baseline")
    print(fmt(f"{other} increment, NO tuned MoE config",
              nocfg["baseline"], nocfg[other]))
    print(fmt(f"{other} increment, WITH tuned MoE config",
              cfg["baseline"], cfg[other]))
    print(fmt("tuned MoE config alone (baseline arm)", nocfg["baseline"], cfg["baseline"]))
    print(fmt(f"tuned MoE config alone ({other} arm)", nocfg[other], cfg[other]))
    print(fmt("full stack: bare default -> config+kernel",
              nocfg["baseline"], cfg[other]))

    g_cfg = st.mean(cfg["baseline"]) / st.mean(nocfg["baseline"])
    g_ker = st.mean(nocfg[other]) / st.mean(nocfg["baseline"])
    g_all = st.mean(cfg[other]) / st.mean(nocfg["baseline"])
    print(f"\nadditivity: config {g_cfg:.4f}x * kernel {g_ker:.4f}x = "
          f"{g_cfg * g_ker:.4f}x predicted, {g_all:.4f}x observed, "
          f"realisation {(g_all - 1) / (g_cfg * g_ker - 1):.2f}")

    out = {
        "metric": METRIC, "regime": REGIME,
        "cells": {f"{lv}_{od}": {k: v for k, v in load(f"{lv}_{od}").items()}
                  for lv in ("nocfg", "cfg") for od in ("fwd", "rev")},
        "pooled": {"nocfg": nocfg, "cfg": cfg},
        "summary": {
            "kernel_nocfg": welch(nocfg["baseline"], nocfg[other]),
            "kernel_cfg": welch(cfg["baseline"], cfg[other]),
            "config_on_baseline": welch(nocfg["baseline"], cfg["baseline"]),
            f"config_on_{other}": welch(nocfg[other], cfg[other]),
            "full_stack": welch(nocfg["baseline"], cfg[other]),
        },
    }
    dest = E2E / f"exp3_layered_{PREFIX}{REGIME}_summary.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
