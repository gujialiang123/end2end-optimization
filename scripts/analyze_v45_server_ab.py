#!/usr/bin/env python3
"""v45 analysis: server-level ours vs fallback, per regime, Welch t-test.

Metrics per regime (mean over repeats), gain sign = ours better:
  - TTFT (ms, lower better), TPOT (ms, lower better),
  - E2E latency (ms, lower better), output throughput (tok/s, higher better).
"""
import argparse, json, math
from collections import defaultdict


def welch_t(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return (float("inf") if ma != mb else 0.0), (0.0 if ma != mb else 1.0)
    t = (ma - mb) / se
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else 1
    p = 2 * _t_sf(abs(t), df)
    return t, p


def _t_sf(t, df):
    x = df / (df + t * t)
    return 0.5 * _betai(df / 2.0, 0.5, x)


def _betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    return front * _betacf(a, b, x)


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d; h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < eps: break
    return h


def mean(xs): return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    rows = []
    for fn in args.files:
        for line in open(fn):
            line = line.strip()
            if line: rows.append(json.loads(line))

    # regime -> arm -> metric -> [values]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    order = []
    for r in rows:
        reg, arm = r["regime"], r["arm"]
        if reg not in order: order.append(reg)
        for m in ("ttft_ms", "tpot_ms", "e2e_ms", "out_tput"):
            if r.get(m) is not None:
                data[reg][arm][m].append(r[m])

    metrics = [("ttft_ms", True), ("tpot_ms", True), ("e2e_ms", True), ("out_tput", False)]
    print(f"{'regime':<22}{'metric':<10}{'fb_med':>11}{'ours_med':>11}{'gain%':>9}{'t':>8}{'p':>9}{'n':>5}")
    print("-" * 95)
    for reg in order:
        fb, ou = data[reg].get("fallback", {}), data[reg].get("ours", {})
        for m, lower_better in metrics:
            if m not in fb or m not in ou or not fb[m] or not ou[m]:
                continue
            mf, mo = median(fb[m]), median(ou[m])
            gain = (mf / mo - 1) * 100 if lower_better else (mo / mf - 1) * 100  # + = ours better
            t, p = welch_t(ou[m], fb[m])
            tp = f"{t:>8.2f}" if t is not None else f"{'-':>8}"
            pp = f"{p:>9.4f}" if p is not None else f"{'-':>9}"
            sig = " *" if (p is not None and p < 0.05) else ""
            print(f"{reg:<22}{m:<10}{mf:>11.2f}{mo:>11.2f}{gain:>8.2f}%{tp}{pp}{len(fb[m]):>5}{sig}")
        print()


if __name__ == "__main__":
    main()
