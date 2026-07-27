#!/usr/bin/env python3
"""Build a guardrailed regime-aware profile.

The naive regime-aware profile replaces the kernel configuration at EVERY M
bucket. Measurement showed why that is wrong:

  * CUDA-graph capture pins decode to batch sizes [1,2,4,8,12,16,24,32], so
    decode only ever touches M in {4 ... 128};
  * in exactly that range the with-bias oracle is 0.98-1.09x, i.e. there is no
    real headroom, so "tuning" there selects measurement noise;
  * deploying that noise cost 12-25 % end-to-end throughput in decode;
  * the genuine headroom (1.39-1.64x) lives at M >= 256, which is reached by
    prefill.

The guardrail: specialize only where the oracle beats the default by more than a
threshold, and write the DEFAULT heuristic configuration everywhere else, so the
nearest-M lookup can never pull a large-M tile into a decode bucket.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rk_lib as L


def default_config_for(M: int, E: int, N: int, K: int, topk: int) -> dict:
    """Reproduce the runtime's own heuristic default for this shape."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        get_default_config)
    cfg = get_default_config(M, E, N, K, topk, None, False, None)
    return {k: int(v) for k, v in cfg.items()
            if k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                     "GROUP_SIZE_M", "num_warps", "num_stages")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm25")
    ap.add_argument("--strategy-csv",
                    default="results/regime_kernel/processed/strategy_comparison_bias.csv")
    ap.add_argument("--threshold", type=float, default=1.15,
                    help="only specialize where oracle speedup exceeds this")
    ap.add_argument("--name", default="regime_aware_guarded")
    a = ap.parse_args()

    from sglang.srt.server_args import (ServerArgs,
                                        set_global_server_args_for_scheduler)
    set_global_server_args_for_scheduler(ServerArgs(model_path=L.MODELS[a.model]["path"]))

    shape = L.MODELS[a.model]
    E, N, K = shape["num_experts"], shape["moe_intermediate_size"], shape["hidden_size"]
    topk = shape["top_k"]
    df = pd.read_csv(a.strategy_csv)
    df = df[df.model == a.model]

    prof_file = L.CONFIGS / f"{a.model}_bias_profiles.json"
    profiles = json.loads(prof_file.read_text())

    table, decisions = {}, []
    for _, r in df.iterrows():
        M = int(r.M)
        if r.oracle_speedup >= a.threshold:
            cfg = {k: int(v) for k, v in
                   json.loads(json.dumps(profiles[f"oracle_t{int(r.tokens)}"])).items()}
            why = f"oracle {r.oracle_speedup:.3f}x >= {a.threshold} -> specialize"
        else:
            cfg = default_config_for(M, E, N, K, topk)
            why = (f"oracle {r.oracle_speedup:.3f}x < {a.threshold} -> keep the "
                   f"runtime default (tuning here would deploy noise)")
        table[str(M)] = cfg
        decisions.append(dict(M=M, tokens=int(r.tokens),
                              oracle_speedup=float(r.oracle_speedup),
                              specialized=r.oracle_speedup >= a.threshold,
                              reason=why, **cfg))

    outdir = (L.CONFIGS / "profiles" / f"{a.model}_bias_{a.name}" /
              "configs" / "triton_3_5_1")
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"E={E},N={N},device_name=NVIDIA_H200.json"
    (outdir / fname).write_text(json.dumps(table, indent=2))

    dec = pd.DataFrame(decisions)
    dec.to_csv(L.RESULTS / "processed" / f"{a.model}_guarded_profile_decisions.csv",
               index=False)
    print(f"wrote {outdir/fname}")
    print(dec[["M", "oracle_speedup", "specialized", "BLOCK_SIZE_M",
               "BLOCK_SIZE_N", "num_warps", "num_stages"]].to_string(index=False))
    n = int(dec.specialized.sum())
    print(f"\nspecialized {n}/{len(dec)} M buckets "
          f"(threshold {a.threshold}x); the rest keep the runtime default")


if __name__ == "__main__":
    main()
