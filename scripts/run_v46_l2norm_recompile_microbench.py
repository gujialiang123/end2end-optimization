#!/usr/bin/env python3
"""v46 microbench: does PR #31558 (do_not_specialize=["T"]) actually remove
per-token-count recompilation of the FLA l2norm Triton kernel?

Mechanism under test: the l2norm_fwd_kernel had T (flattened token count) as a
tl.constexpr, so each distinct token count JIT-compiles a NEW kernel variant. In a
VLM, different image resolutions -> different token counts -> a fresh cubin each
time. PR #31558 makes T a runtime scalar (do_not_specialize=["T"]) so ONE kernel
serves all token counts.

We call l2norm_fwd on a sweep of distinct token counts and count how many distinct
compiled kernels Triton ends up caching. Baseline should compile ~N (one per unique
T); patched should compile 1.

Run once per arm (baseline / patched l2norm.py in place); the arm label + counts are
printed and appended to a jsonl.
"""
import argparse, json, os, time
import torch

from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd, l2norm_fwd_kernel

D = 128  # linear_key_head_dim for Qwen3.6 (<=512 -> hits the patched kernel path)


def count_cached_kernels():
    """Number of distinct compiled variants Triton holds for l2norm_fwd_kernel."""
    dc = getattr(l2norm_fwd_kernel, "device_caches", None)
    if dc is None:
        return None
    n = 0
    for dev, v in dc.items():
        # triton 3.6: device_caches[dev] is a tuple; v[0] is the compiled-kernel dict
        try:
            n += len(v[0])
        except Exception:
            try:
                n += len(v)
            except Exception:
                pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="baseline | patched")
    ap.add_argument("--out", required=True)
    # token counts mimic different image-token payloads (varying resolution/#images)
    ap.add_argument("--token-counts", type=str,
                    default="777,1234,1600,2048,2500,3072,4096,5000,6144,7777")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    dev = "cuda"
    torch.cuda.init()
    tcs = [int(x) for x in args.token_counts.split(",")]

    # warmup a trivial call to make sure module-level jit is realized
    _ = l2norm_fwd(torch.randn(16, D, device=dev, dtype=torch.bfloat16))
    torch.cuda.synchronize()

    n0 = count_cached_kernels()
    t0 = time.time()
    per_call = []
    for T in tcs:
        x = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
        tc0 = time.time()
        for _ in range(args.repeats):
            y = l2norm_fwd(x)
        torch.cuda.synchronize()
        per_call.append({"T": T, "ms_mean": round((time.time() - tc0) / args.repeats * 1000, 3),
                         "cached_after": count_cached_kernels()})
    total_wall = time.time() - t0
    n1 = count_cached_kernels()

    row = {
        "label": args.label,
        "unique_token_counts": len(tcs),
        "kernels_cached_before": n0,
        "kernels_cached_after": n1,
        "kernels_compiled_during_sweep": (n1 - n0) if (n0 is not None and n1 is not None) else None,
        "sweep_wall_s": round(total_wall, 3),
        "per_call": per_call,
        "token_counts": tcs,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{args.label}] unique_T={len(tcs)}  cached before={n0} after={n1}  "
          f"compiled_during_sweep={row['kernels_compiled_during_sweep']}  "
          f"sweep_wall={total_wall:.2f}s", flush=True)
    for pc in per_call:
        print(f"   T={pc['T']:>5}  {pc['ms_mean']:>8.3f} ms  cached={pc['cached_after']}", flush=True)


if __name__ == "__main__":
    main()
