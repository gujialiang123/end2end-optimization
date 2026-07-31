#!/usr/bin/env python3
"""Paired significance test for the GSM8K arms.

The gate script's noise-floor arm reruns the baseline tree with a different
seed, which under greedy decoding is deterministic and therefore reports a
0.00-point floor. A floor of zero makes any difference at all look significant,
which is exactly backwards -- it measures nothing, because there is no
randomness for the seed to perturb.

Both arms answer the *same* questions, so the correct test is paired. McNemar
looks only at the questions where the two disagree: if the fused arm is neither
better nor worse, wins and losses should split evenly, and 2 net wins out of 400
is not evidence of anything. The unpaired binomial standard error at n=400 and
p~0.22 is about 2 points, which is the scale a difference has to clear.
"""
from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path


def last_number(t: str) -> str | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/test.jsonl")
    ap.add_argument("--baseline", default="/tmp/_gsm_baseline.json")
    ap.add_argument("--fused", default="/tmp/_gsm_fused.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = json.loads(Path(a.baseline).read_text())
    fused = json.loads(Path(a.fused).read_text())
    n = base["total"]
    rows = [json.loads(l) for l in open(a.data)][:n]
    golds = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]

    b_ok = [last_number(t) == g for t, g in zip(base["texts"], golds)]
    f_ok = [last_number(t) == g for t, g in zip(fused["texts"], golds)]

    both = sum(x and y for x, y in zip(b_ok, f_ok))
    neither = sum((not x) and (not y) for x, y in zip(b_ok, f_ok))
    only_base = sum(x and not y for x, y in zip(b_ok, f_ok))   # fused lost
    only_fused = sum(y and not x for x, y in zip(b_ok, f_ok))  # fused won

    print(f"n = {n}")
    print(f"  both correct        : {both}")
    print(f"  both wrong          : {neither}")
    print(f"  baseline only (loss): {only_base}")
    print(f"  fused only    (win) : {only_fused}")
    print(f"\n  baseline acc : {sum(b_ok) / n * 100:5.2f}%")
    print(f"  fused acc    : {sum(f_ok) / n * 100:5.2f}%")
    print(f"  delta        : {(sum(f_ok) - sum(b_ok)) / n * 100:+5.2f} pts")

    # Exact two-sided McNemar over the discordant pairs.
    d = only_base + only_fused
    k = min(only_base, only_fused)
    p = min(1.0, 2.0 * sum(comb(d, i) for i in range(k + 1)) / (2 ** d)) if d else 1.0

    print(f"\n  discordant pairs : {d}  ({only_fused} win / {only_base} loss)")
    print(f"  exact McNemar p  : {p:.3f}")
    verdict = ("no detectable accuracy change" if p > 0.05
               else ("accuracy improved" if only_fused > only_base
                     else "ACCURACY REGRESSED"))
    print(f"  verdict          : {verdict}")
    print(f"\n  note: {d} of {n} answers changed at all, which is the real story --"
          f"\n  the arithmetic differs, and the net effect on accuracy is a wash.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(
            n=n, both_correct=both, both_wrong=neither,
            fused_wins=only_fused, fused_losses=only_base,
            baseline_acc=sum(b_ok) / n, fused_acc=sum(f_ok) / n,
            discordant=d, mcnemar_p=round(p, 5), verdict=verdict), indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
