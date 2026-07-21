#!/usr/bin/env python3
"""v42 analysis: compare baseline vs ours e2e A/B, per cell, with Welch t-test.

Reads the raw jsonl produced by run_v42_e2e_config_ab.py (both labels appended to
the same file, or two files) and reports, per (kind, batch, input_len):
  - decode: median decode-step latency (lower=better) -> TPOT
  - prefill: prefill throughput (higher=better)
with n, median, mean, std, %delta (ours vs baseline) and Welch's t / p (two-sided).
"""
import argparse, json, math
from collections import defaultdict


def welch_t(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("inf"), 0.0
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else 1
    # two-sided p via survival of t-dist approximated with normal for df>=large,
    # else use a simple incomplete-beta based CDF.
    p = 2 * _t_sf(abs(t), df)
    return t, p


def _t_sf(t, df):
    # survival function of Student t via regularized incomplete beta
    x = df / (df + t * t)
    return 0.5 * _betai(df / 2.0, 0.5, x)


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    return front * _betacf(a, b, x)


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab = a + b; qap = a + 1; qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    rows = []
    for fn in args.files:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    # group by cell
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("rc") != 0:
            continue
        key = (r["kind"], r["batch"], r["input_len"])
        metric = r["decode_med_lat_s"] if r["kind"] == "decode" else r["prefill_tput"]
        if metric is not None:
            cells[key][r["label"]].append(metric)

    print(f"{'cell':<28}{'metric':<16}{'base_med':>12}{'ours_med':>12}{'delta%':>9}{'t':>8}{'p':>9}{'n':>6}")
    for key in sorted(cells):
        kind, b, il = key
        base = cells[key].get("baseline", [])
        ours = cells[key].get("ours", [])
        if not base or not ours:
            continue
        mb, mo = median(base), median(ours)
        if kind == "decode":
            metric = "decode_lat(s)"  # lower better
            delta = (mb - mo) / mb * 100  # +% = ours faster
        else:
            metric = "prefill_tput"    # higher better
            delta = (mo - mb) / mb * 100  # +% = ours faster
        t, p = welch_t(ours, base)
        cell = f"{kind} b={b} in={il}"
        tp = f"{t:>8.2f}" if t is not None else f"{'-':>8}"
        pp = f"{p:>9.4f}" if p is not None else f"{'-':>9}"
        print(f"{cell:<28}{metric:<16}{mb:>12.5f}{mo:>12.5f}{delta:>8.2f}%{tp}{pp}{len(base):>6}")


if __name__ == "__main__":
    main()
