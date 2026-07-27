"""Correctness verification for the ShortConv fusion candidates.

Checks the fused kernels against the exact op sequence in
sglang/srt/models/lfm2_moe.py::Lfm2MoeShortConv.forward on realistic
multi-sequence (varlen) prefill batches and on decode with non-trivial
conv_state and scattered conv_state_indices.
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx_common as C  # noqa: E402
from fx_bench_fusions import gate_only, gate_transpose, out_gate_untranspose  # noqa: E402
from fx_common import causal_conv1d_fn, causal_conv1d_update  # noqa: E402

H = C.H
dev = "cuda"
dt = torch.bfloat16
checks = []


def check(name, cond, detail=""):
    checks.append(dict(name=name, ok=bool(cond), detail=detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond


def stock_prefill(hs, w_in, w_out, conv_w, conv_state, qsl, ci):
    proj = torch.nn.functional.linear(hs, w_in)
    B_gate, C_gate, x = proj.chunk(3, dim=-1)
    Bx = B_gate * x
    Bx_t = Bx.transpose(0, 1).contiguous()
    conv_out = causal_conv1d_fn(
        Bx_t, conv_w, None, query_start_loc=qsl, cache_indices=ci,
        has_initial_state=None, conv_states=conv_state, activation=None,
    ).transpose(0, 1)
    return torch.nn.functional.linear(C_gate * conv_out, w_out)


def fused_prefill(hs, w_in, w_out, conv_w, conv_state, qsl, ci, fuse_out=True):
    T = hs.shape[0]
    proj = torch.nn.functional.linear(hs, w_in)
    Bx_t = gate_transpose(proj, T)
    assert Bx_t.stride(-1) == 1, "causal_conv1d_fn requires x.stride(-1)==1"
    conv_out = causal_conv1d_fn(
        Bx_t, conv_w, None, query_start_loc=qsl, cache_indices=ci,
        has_initial_state=None, conv_states=conv_state, activation=None,
    )
    if fuse_out:
        y = out_gate_untranspose(proj, conv_out, T)
    else:
        y = proj[:, H : 2 * H] * conv_out.transpose(0, 1)
    return torch.nn.functional.linear(y, w_out)


def stock_decode(hs, w_in, w_out, conv_w, conv_state, rpi):
    proj = torch.nn.functional.linear(hs, w_in)
    B_gate, C_gate, x = proj.chunk(3, dim=-1)
    Bx = B_gate * x
    conv_out = causal_conv1d_update(
        Bx, conv_state, conv_w, None, activation=None,
        conv_state_indices=rpi.to(torch.int32),
    )
    return torch.nn.functional.linear(C_gate * conv_out, w_out)


def fused_decode(hs, w_in, w_out, conv_w, conv_state, rpi32):
    T = hs.shape[0]
    proj = torch.nn.functional.linear(hs, w_in)
    conv_out = causal_conv1d_update(
        gate_only(proj, T), conv_state, conv_w, None, activation=None,
        conv_state_indices=rpi32,
    )
    return torch.nn.functional.linear(proj[:, H : 2 * H] * conv_out, w_out)


def main():
    torch.manual_seed(1234)
    w_in = torch.randn(3 * H, H, device=dev, dtype=dt) * 0.02
    w_out = torch.randn(H, H, device=dev, dtype=dt) * 0.02
    conv_w = torch.randn(H, 3, device=dev, dtype=dt) * 0.2

    print("\n--- prefill, varlen multi-sequence (the real SGLang extend path) ---")
    for seqlens in ([4000, 4000, 4000, 4000], [1, 137, 999, 2048, 63], [3], [7, 9]):
        T = sum(seqlens)
        hs = torch.randn(T, H, device=dev, dtype=dt) * 0.05
        starts = [0]
        for s in seqlens:
            starts.append(starts[-1] + s)
        qsl = torch.tensor(starts, device=dev, dtype=torch.int32)
        ci = torch.arange(len(seqlens), device=dev, dtype=torch.int32)
        init = torch.randn(len(seqlens) + 4, H, 3, device=dev, dtype=dt) * 0.1

        cs = init.clone()
        r0 = stock_prefill(hs, w_in, w_out, conv_w, cs, qsl, ci)
        s0 = cs.clone()
        cs = init.clone()
        r1 = fused_prefill(hs, w_in, w_out, conv_w, cs, qsl, ci, fuse_out=False)
        s1 = cs.clone()
        cs = init.clone()
        r2 = fused_prefill(hs, w_in, w_out, conv_w, cs, qsl, ci, fuse_out=True)
        s2 = cs.clone()
        tag = f"T={T} seqs={len(seqlens)}"
        check(f"prefill fuse-in bit-exact output      {tag}", torch.equal(r0, r1))
        check(f"prefill fuse-in+out bit-exact output  {tag}", torch.equal(r0, r2))
        check(f"prefill conv_state identical          {tag}",
              torch.equal(s0, s1) and torch.equal(s0, s2))

    print("\n--- decode, scattered conv_state_indices, non-zero initial state ---")
    for B in (1, 4, 8, 33, 128):
        n_slots = 256
        hs = torch.randn(B, H, device=dev, dtype=dt) * 0.05
        perm = torch.randperm(n_slots, device=dev)[:B]
        rpi = perm.to(torch.int64)
        rpi32 = rpi.to(torch.int32)
        init = torch.randn(n_slots, H, 3, device=dev, dtype=dt) * 0.1

        cs = init.clone()
        r0 = stock_decode(hs, w_in, w_out, conv_w, cs, rpi)
        s0 = cs.clone()
        cs = init.clone()
        r1 = fused_decode(hs, w_in, w_out, conv_w, cs, rpi32)
        s1 = cs.clone()
        check(f"decode fused-gate bit-exact output    B={B}", torch.equal(r0, r1))
        check(f"decode conv_state identical           B={B}", torch.equal(s0, s1))

    print("\n--- layout constraint checks ---")
    proj = torch.randn(777, 3 * H, device=dev, dtype=dt)
    bxt = gate_transpose(proj, 777)
    check("gate_transpose output is [H,T] with stride(-1)==1",
          bxt.shape == (H, 777) and bxt.stride(-1) == 1,
          f"shape={tuple(bxt.shape)} strides={bxt.stride()}")
    ref = (proj[:, :H] * proj[:, 2 * H :]).transpose(0, 1).contiguous()
    check("gate_transpose == eager mul+transpose+contiguous (bit-exact)",
          torch.equal(bxt, ref))
    conv_t = torch.randn(H, 777, device=dev, dtype=dt)
    got = out_gate_untranspose(proj, conv_t, 777)
    ref2 = proj[:, H : 2 * H] * conv_t.transpose(0, 1)
    check("out_gate_untranspose == eager C_gate*conv_out (bit-exact)",
          torch.equal(got, ref2))
    g = gate_only(proj, 777)
    check("gate_only == eager B_gate*x (bit-exact)",
          torch.equal(g, proj[:, :H] * proj[:, 2 * H :]))

    n_fail = sum(1 for c in checks if not c["ok"])
    outp = os.path.join(C.outdir(), "verify_fusion.json")
    with open(outp, "w") as f:
        json.dump({"n_checks": len(checks), "n_fail": n_fail, "checks": checks}, f,
                  indent=2)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed -> {outp}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
