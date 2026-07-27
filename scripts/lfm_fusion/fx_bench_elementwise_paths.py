"""Decompose the ShortConv glue into its three elementwise passes and compare
each against a fully contiguous baseline of identical HBM traffic.

Separates the two mechanisms:
  * uncoalesced access  (transpose copy, transposed read in C_gate * conv_out)
  * non-vectorizable    (B_gate * x: coalesced, but rows are 3H apart, so
                         TensorIterator cannot vectorize)
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx_common as C  # noqa: E402
from fx_bench_fusions import bench_graph  # noqa: E402

H = C.H
dev = "cuda"
dt = torch.bfloat16


def main():
    torch.manual_seed(0)
    rows = []
    for T in (4096, 8192, 16000):
        proj = torch.randn(T, 3 * H, device=dev, dtype=dt)
        a = torch.randn(T, H, device=dev, dtype=dt)
        b = torch.randn(T, H, device=dev, dtype=dt)
        convt = torch.randn(H, T, device=dev, dtype=dt)
        Bg, Cg, x = proj[:, :H], proj[:, H : 2 * H], proj[:, 2 * H :]
        Bx = Bg * x

        r = {
            "T": T,
            "contiguous_add_us": bench_graph(lambda: a + b),
            "contiguous_mul_us": bench_graph(lambda: a * b),
            "b_gate_mul_strided_us": bench_graph(lambda: Bg * x),
            "c_gate_mul_transposed_us": bench_graph(lambda: Cg * convt.transpose(0, 1)),
            "transpose_contiguous_us": bench_graph(lambda: Bx.transpose(0, 1).contiguous()),
        }
        three_pass = 3 * T * H * 2
        two_pass = 2 * T * H * 2
        r["gbps"] = {
            "contiguous_add": round(three_pass / r["contiguous_add_us"] * 1e-3),
            "contiguous_mul": round(three_pass / r["contiguous_mul_us"] * 1e-3),
            "b_gate_mul_strided": round(three_pass / r["b_gate_mul_strided_us"] * 1e-3),
            "c_gate_mul_transposed": round(
                three_pass / r["c_gate_mul_transposed_us"] * 1e-3
            ),
            "transpose_contiguous": round(
                two_pass / r["transpose_contiguous_us"] * 1e-3
            ),
        }
        rows.append(r)
        print(json.dumps(r))

    outp = os.path.join(C.outdir(), "bench_elementwise_paths.json")
    with open(outp, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
