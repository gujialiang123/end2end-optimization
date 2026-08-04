#!/usr/bin/env python3
"""Which MoE config bucket does each forward actually land in?

The tuned MoE config was swept under the cookbook serving knobs, where long
prefill arrives as one 4000-token chunk. The L1 ceiling config turns chunked
prefill on at 2048, which changes the distribution of M -- the token count of a
forward -- and therefore which bucket of the config file gets selected.

That claim can be settled without timing anything. sglang logs `#new-token` for
every prefill batch, and `get_moe_configs` picks the bucket nearest to M, so the
mapping from log line to bucket is deterministic. This reads the server logs of
a run and reports the histogram, which is the evidence a 0.2-second measurement
window cannot provide.

  python scripts/analyze_moe_bucket_usage.py <server.log> [<server.log> ...]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / (
    "configs/regime_kernel/profiles/lfm25_pr_candidate/configs/triton_3_5_1/"
    "E=32,N=1792,device_name=NVIDIA_H200.json"
)
NEW_TOKEN = re.compile(r"Prefill batch.*?#new-token: (\d+)")
# The benchmark's own prompts are 4000 tokens; the correctness probe and the
# health check issue tiny ones, and those are not part of the measured load.
MIN_REAL = 64


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cfg = json.loads(CONFIG.read_text())
    buckets = sorted(int(k) for k in cfg)

    for path in sys.argv[1:]:
        toks = [int(m.group(1)) for m in NEW_TOKEN.finditer(
            Path(path).read_text(errors="ignore"))]
        real = [t for t in toks if t >= MIN_REAL]
        if not real:
            print(f"{path}: no prefill batches above {MIN_REAL} tokens")
            continue

        hist = Counter(real)
        chosen = Counter()
        for m, n in hist.items():
            chosen[min(buckets, key=lambda b: abs(b - m))] += n

        print(f"\n=== {Path(path).parent.name}/{Path(path).name}")
        print(f"    {len(real)} prefill batches >= {MIN_REAL} tokens")
        print(f"    token counts: {sorted(hist)[:8]}"
              f"{' ...' if len(hist) > 8 else ''}")
        print(f"    {'bucket':>8}  {'batches':>8}  {'BLOCK_M':>8} {'BLOCK_N':>8}"
              f" {'BLOCK_K':>8} {'warps':>6} {'stages':>7}")
        for b, n in sorted(chosen.items()):
            c = cfg[str(b)]
            print(f"    {b:>8}  {n:>8}  {c['BLOCK_SIZE_M']:>8} {c['BLOCK_SIZE_N']:>8}"
                  f" {c['BLOCK_SIZE_K']:>8} {c.get('num_warps', '-'):>6}"
                  f" {c.get('num_stages', '-'):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
