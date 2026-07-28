#!/usr/bin/env python3
"""PR-grade numeric verification of the Gemma3RMSNorm CUDA fix.

Checks the patched `forward_cuda` against the unpatched `forward_native`
reference across the shapes, dtypes and magnitudes that actually occur, plus
the edge cases a reviewer would ask about. Run against the PATCHED tree via
PYTHONPATH, so it exercises the real source change rather than a monkeypatch.

Reported per case: max absolute deviation, max relative deviation, and whether
any NaN/Inf appeared. The pass bar is bf16 round-off — the fused kernel keeps
the weight multiply in the activation dtype while the reference does it in
fp32, so exact equality is not expected and not required. `GemmaRMSNorm`
(gemma/gemma2) already ships with the identical trade-off.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from sglang.srt.layers.layernorm import Gemma3RMSNorm

# (name, dims) — 1152 is gemma-3-1b hidden, 256 its head_dim, 2560/3840 are the
# hidden sizes of the 4b/12b checkpoints.
SHAPES = [
    ("decode b1", (1, 1152)),
    ("decode b32", (32, 1152)),
    ("prefill short", (128, 1152)),
    ("prefill long", (16000, 1152)),
    ("q_norm 3-D", (37, 4, 256)),
    ("k_norm 3-D", (1, 1, 256)),
    ("q_norm 3-D large", (4096, 8, 256)),
    ("hidden 2560", (64, 2560)),
    ("hidden 3840", (64, 3840)),
    ("odd tokens", (1023, 1152)),
]
DTYPES = [("bf16", torch.bfloat16), ("fp16", torch.float16)]
SCALES = [("normal", 0.5), ("large", 50.0), ("tiny", 1e-3)]


def run_case(dim, shape, dtype, scale, dev, weight_dtype):
    torch.manual_seed(0)
    n = Gemma3RMSNorm(dim, eps=1e-6).to(dev)
    with torch.no_grad():
        n.weight.copy_((torch.randn(dim, device=dev) * 0.1).to(n.weight.dtype))
    if weight_dtype is not None:
        n.weight.data = n.weight.data.to(weight_dtype)

    x = (torch.randn(*shape, device=dev) * scale).to(dtype)
    ref = n.forward_native(x)
    got = n.forward_cuda(x)

    bad = bool(torch.isnan(got).any() or torch.isinf(got).any())
    d = (ref.float() - got.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    return dict(
        shape_ok=(got.shape == ref.shape),
        nan_or_inf=bad,
        max_abs=d.max().item(),
        max_rel=(d / denom).max().item(),
        mean_rel=(d / denom).mean().item(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--tol-rel", type=float, default=0.02,
                    help="max tolerated relative deviation (bf16 is ~2^-8)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"

    print(f"{'case':22s} {'dtype':6s} {'scale':7s} {'wdtype':7s} "
          f"{'max_abs':>10s} {'max_rel':>10s} {'NaN':>5s}  verdict")
    rows, failures = [], 0
    for wname, wdtype in [("fp32", torch.float32), ("same", None)]:
        for sname, scale in SCALES:
            for dname, dtype in DTYPES:
                for cname, shape in SHAPES:
                    r = run_case(shape[-1], shape, dtype, scale, dev, wdtype)
                    ok = (r["shape_ok"] and not r["nan_or_inf"]
                          and r["max_rel"] <= a.tol_rel)
                    failures += (not ok)
                    rows.append(dict(case=cname, shape=list(shape), dtype=dname,
                                     scale=sname, weight_dtype=wname, ok=ok, **r))
                    print(f"{cname:22s} {dname:6s} {sname:7s} {wname:7s} "
                          f"{r['max_abs']:10.3e} {r['max_rel']:10.3e} "
                          f"{str(r['nan_or_inf']):>5s}  {'PASS' if ok else 'FAIL'}")

    worst = max(rows, key=lambda r: r["max_rel"])
    print(f"\n{len(rows)} cases, {failures} failures")
    print(f"worst relative deviation: {worst['max_rel']:.3e} "
          f"({worst['case']}, {worst['dtype']}, {worst['scale']}, "
          f"weight={worst['weight_dtype']})")
    print(f"tolerance: {a.tol_rel}  ->  "
          f"{'ALL PASS' if failures == 0 else 'FAILURES PRESENT'}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(rows=rows, failures=failures, tol_rel=a.tol_rel,
                           worst=worst,
                           torch=torch.__version__,
                           gpu=torch.cuda.get_device_name(a.gpu)), f, indent=2)
        print(f"wrote {a.out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
