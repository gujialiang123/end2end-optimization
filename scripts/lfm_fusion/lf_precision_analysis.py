#!/usr/bin/env python3
"""Precision analysis: is a fused-kernel swap lossy, or just a different rounding?

The question this answers is not "how far apart are the two paths" — that is
easy and uninformative — but **"which of them is closer to the truth"**. So it
computes the same normalisation in fp64, treats that as ground truth, and
measures both paths against it, together with the best a bf16 result could
possibly be (fp64 truth rounded once to bf16).

If both paths sit at the fp64-rounded-to-bf16 bound, the difference between
them is a rounding-order artefact of the storage format, not a loss of
precision introduced by the kernel.

Usage:
  python scripts/lfm_fusion/lf_precision_analysis.py --gpu 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L


def analyse(case, dev, dtype=torch.bfloat16):
    """Returns errors of both paths against an fp64 reference."""
    from sglang.srt.layers.layernorm import Gemma3RMSNorm
    from sgl_kernel import gemma_rmsnorm

    T, H = case
    torch.manual_seed(0)
    n = Gemma3RMSNorm(H, eps=1e-6).to(dev)
    with torch.no_grad():
        n.weight.copy_(torch.randn(H, device=dev) * 0.1)
    x = (torch.randn(T, H, device=dev) * 0.5).to(dtype)

    eager = n.forward_native(x)
    fused = gemma_rmsnorm(x.contiguous(),
                          n.weight.data.to(dtype).contiguous(), n.eps)

    xd = x.double()
    truth = (xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + 1e-6)) \
        * (1.0 + n.weight.double())
    best_possible = truth.to(dtype)          # the fp64 answer rounded once

    def rel(a):
        d = (a.double() - truth).abs()
        return (d / truth.abs().clamp_min(1e-9)).max().item()

    diff = (eager.float() - fused.float()).abs().max().item()
    eps = torch.finfo(dtype).eps
    return dict(
        tokens=T, hidden=H, dtype=str(dtype).split(".")[-1],
        eager_rel_err=rel(eager),
        fused_rel_err=rel(fused),
        best_possible_rel_err=rel(best_possible),
        max_abs_diff=diff,
        diff_in_ulp=diff / max(truth.abs().max().item(), 1e-9) / eps,
        identical_frac=(eager == fused).float().mean().item(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"

    rows = []
    for case in [(1, 1152), (64, 1152), (4096, 1152), (64, 256)]:
        for dt in (torch.bfloat16, torch.float16):
            rows.append(analyse(case, dev, dt))

    print(f"{'shape':>14s} {'dt':>5s} {'eager':>10s} {'fused':>10s} "
          f"{'bf16 bound':>11s} {'ulp':>6s} {'same':>7s}")
    for r in rows:
        print(f"{r['tokens']:6d}x{r['hidden']:<7d} {r['dtype'][:4]:>5s} "
              f"{r['eager_rel_err']:10.3e} {r['fused_rel_err']:10.3e} "
              f"{r['best_possible_rel_err']:11.3e} {r['diff_in_ulp']:6.2f} "
              f"{100*r['identical_frac']:6.1f}%")

    print("\nReading: if 'eager' equals 'bf16 bound', the reference path is "
          "already at the\nstorage format's limit and cannot be more accurate. "
          "'ulp' is the gap between the\ntwo paths in units of the last "
          "representable digit.")

    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(rows=rows, environment=L.environment()), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
