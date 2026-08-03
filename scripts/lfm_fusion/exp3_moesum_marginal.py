#!/usr/bin/env python3
"""Marginal contribution of `moesum`, at both baseline levels, on regime C.

`moesum` is the one item of the seven that reaches inside FusedMoE: it makes the
layer return four unreduced expert outputs and fuses the reduction with the
residual add and the following RMSNorm. The other six sit outside the MoE. If
the superadditivity seen in exp3 comes from that interaction, `moesum` should be
worth ~nothing on the untuned MoE and clearly positive once the tuned config is
in, which is what `all7 - all` measures.

The two arm suites were separate runs, so their baselines differ slightly. Both
suites are reported against their own baseline, and the `all7 - all` step is
computed on the absolute throughputs, which were measured on the same GPU in the
same session, half an hour apart.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import exp3_analyze as A

E2E = A.E2E


def cells(suite: str, level: str) -> dict[str, list[float]]:
    A.PREFIX, A.REGIME = suite, "C_long_prefill"
    fwd, rev = A.load(f"{level}_fwd"), A.load(f"{level}_rev")
    return {arm: fwd[arm] + rev[arm] for arm in fwd}


def main() -> None:
    seven = {lv: cells("", lv) for lv in ("nocfg", "cfg")}
    six = {lv: cells("six_", lv) for lv in ("nocfg", "cfg")}

    print("regime C_long_prefill, orders pooled, n=16 per arm\n")
    print(f"{'':<34}{'no tuned MoE config':>24}{'with tuned MoE config':>26}")
    rows = []
    for name, base, arm in (
        ("baseline (seven-item suite)", None, ("seven", "baseline")),
        ("baseline (six-item suite)", None, ("six", "baseline")),
        ("all  = six items", "own", ("six", "all")),
        ("all7 = six items + moesum", "own", ("seven", "all7")),
    ):
        suite, key = arm
        src = seven if suite == "seven" else six
        a, b = src["nocfg"][key], src["cfg"][key]
        rows.append((name, st.mean(a), st.mean(b)))
        print(f"{name:<34}{st.mean(a):>17.3f} req/s{st.mean(b):>19.3f} req/s")

    print("\nkernel increment over its own baseline")
    for label, src in (("six items  (all)", six), ("seven items (all7)", seven)):
        key = "all" if src is six else "all7"
        r0, _, p0 = A.welch(src["nocfg"]["baseline"], src["nocfg"][key])
        r1, _, p1 = A.welch(src["cfg"]["baseline"], src["cfg"][key])
        print(f"  {label:<20} {(r0-1)*100:+6.2f}% (p={p0:.1e})"
              f"   ->   {(r1-1)*100:+6.2f}% (p={p1:.1e})")

    print("\nmarginal contribution of moesum (all7 vs all, cross-suite)")
    for lv in ("nocfg", "cfg"):
        r, t, p = A.welch(six[lv]["all"], seven[lv]["all7"])
        tag = "no config" if lv == "nocfg" else "with config"
        print(f"  {tag:<12} {st.mean(six[lv]['all']):7.3f} -> "
              f"{st.mean(seven[lv]['all7']):7.3f} req/s   {(r-1)*100:+6.2f}%"
              f"   t={t:6.2f}  p={p:.2e}")

    print("\ndecomposition of the +6.18% -> +9.73% move (percentage points)")
    six_n = (st.mean(six["nocfg"]["all"]) / st.mean(six["nocfg"]["baseline"]) - 1) * 100
    six_c = (st.mean(six["cfg"]["all"]) / st.mean(six["cfg"]["baseline"]) - 1) * 100
    sev_n = (st.mean(seven["nocfg"]["all7"]) / st.mean(seven["nocfg"]["baseline"]) - 1) * 100
    sev_c = (st.mean(seven["cfg"]["all7"]) / st.mean(seven["cfg"]["baseline"]) - 1) * 100
    print(f"  six items, Amdahl only      {six_n:+6.2f} -> {six_c:+6.2f}"
          f"   ({six_c - six_n:+.2f} pt)")
    print(f"  moesum, genuine interaction {sev_n - six_n:+6.2f} -> {sev_c - six_c:+6.2f}"
          f"   ({(sev_c - six_c) - (sev_n - six_n):+.2f} pt)")
    print(f"  total                       {sev_n:+6.2f} -> {sev_c:+6.2f}"
          f"   ({sev_c - sev_n:+.2f} pt)")

    # a fixed absolute saving per request becomes a larger fraction when the
    # request gets cheaper -- check whether the six items behave that way
    def sec(x: float) -> float:
        return 1.0 / x
    d_n = sec(st.mean(six["nocfg"]["baseline"])) - sec(st.mean(six["nocfg"]["all"]))
    d_c = sec(st.mean(six["cfg"]["baseline"])) - sec(st.mean(six["cfg"]["all"]))
    print(f"\n  six items save {d_n*1e3:.3f} ms/req untuned and {d_c*1e3:.3f} ms/req "
          f"tuned (ratio {d_c/d_n:.2f})")
    d7n = sec(st.mean(seven["nocfg"]["baseline"])) - sec(st.mean(seven["nocfg"]["all7"]))
    d7c = sec(st.mean(seven["cfg"]["baseline"])) - sec(st.mean(seven["cfg"]["all7"]))
    print(f"  seven items save {d7n*1e3:.3f} ms/req untuned and {d7c*1e3:.3f} ms/req "
          f"tuned (ratio {d7c/d7n:.2f})")

    out = {
        "moesum_marginal": {
            lv: A.welch(six[lv]["all"], seven[lv]["all7"]) for lv in ("nocfg", "cfg")
        },
        "six_item_increment": {
            "nocfg": A.welch(six["nocfg"]["baseline"], six["nocfg"]["all"]),
            "cfg": A.welch(six["cfg"]["baseline"], six["cfg"]["all"]),
        },
        "points": {"six_nocfg": six_n, "six_cfg": six_c,
                   "seven_nocfg": sev_n, "seven_cfg": sev_c},
        "ms_saved_per_req": {"six_nocfg": d_n * 1e3, "six_cfg": d_c * 1e3,
                             "seven_nocfg": d7n * 1e3, "seven_cfg": d7c * 1e3},
    }
    dest = E2E / "exp3_moesum_marginal_C_long_prefill.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
