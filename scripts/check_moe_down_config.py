#!/usr/bin/env python3
"""Can an up-projection and a down-projection MoE config be used together?

A MoE layer runs two grouped GEMMs, w13 (up) and w2 (down), and sglang tunes
them with two separate files keyed on a `down_moe` flag. It then asserts that
the two configs agree on BLOCK_SIZE_M (fused_moe_triton_config.py:264), because
the same block partition has to serve both.

The trap is that the two files are looked up *independently*, each by nearest
bucket. Comparing them bucket-by-bucket therefore proves nothing: if one file
has a bucket the other lacks, some M will resolve to two buckets whose
BLOCK_SIZE_M differ and the server dies at startup. This script walks the M
values a run will actually see and reports the ones that would fire.

  python scripts/check_moe_down_config.py <up.json> <down.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def pick(cfgs: dict[int, dict], m: int) -> dict:
    return cfgs[min(cfgs, key=lambda x: abs(x - m))]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    up = {int(k): v for k, v in json.loads(Path(sys.argv[1]).read_text()).items()}
    dn = {int(k): v for k, v in json.loads(Path(sys.argv[2]).read_text()).items()}

    print(f"up buckets   ({len(up):2d}): {sorted(up)}")
    print(f"down buckets ({len(dn):2d}): {sorted(dn)}")
    missing = sorted(set(up) - set(dn))
    if missing:
        print(f"buckets present up-side only: {missing}")

    probe = sorted(set(range(1, 4200)) | {4096, 8192, 16000, 16384})
    bad = [(m, pick(up, m)["BLOCK_SIZE_M"], pick(dn, m)["BLOCK_SIZE_M"])
           for m in probe
           if pick(up, m)["BLOCK_SIZE_M"] != pick(dn, m)["BLOCK_SIZE_M"]]

    if not bad:
        print("\nOK: every probed M resolves to a matching BLOCK_SIZE_M.")
        return 0

    runs = []
    for m, u, d in bad:
        if runs and m == runs[-1][1] + 1 and (u, d) == runs[-1][2]:
            runs[-1][1] = m
        else:
            runs.append([m, m, (u, d)])
    print(f"\nASSERT WOULD FIRE for {len(bad)} values of M, in {len(runs)} range(s):")
    for lo, hi, (u, d) in runs:
        span = f"M={lo}" if lo == hi else f"M={lo}..{hi}"
        print(f"  {span:<20} up BLOCK_SIZE_M={u:<4} down={d}")
    print("\nThe server raises AssertionError at startup for any of these, so the"
          "\ndown config cannot be dropped in as-is. It has to be re-tuned on the"
          "\nup config's bucket set, with BLOCK_SIZE_M constrained to match.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
