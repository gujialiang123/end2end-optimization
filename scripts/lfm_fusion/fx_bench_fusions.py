"""Microbenchmarks for the ShortConv fusion candidates found by the FX/Inductor
graph study.

All shapes are LFM2.5-8B-A1B: hidden_size=2048, conv width 3, bf16, TP1.
Runs on a single decoder layer's worth of random weights - no checkpoint needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx_common as C  # noqa: E402
from fx_common import causal_conv1d_fn, causal_conv1d_update  # noqa: E402

H = C.H
BF = 2  # bytes per bf16 element


# =========================================================================
# Triton kernels
# =========================================================================
@triton.jit
def _gate_transpose_kernel(
    proj_ptr,
    out_ptr,
    T,
    H_: tl.constexpr,
    stride_pt,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """proj [T, 3H] -> out [H, T],  out[h, t] = proj[t, h] * proj[t, 2H + h]

    Fuses chunk(3) + B_gate*x + transpose + contiguous into one pass.
    Tiled so both the load (along 3H) and the store (along T) are coalesced;
    triton materialises the transpose in shared memory.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mt = offs_t < T
    mh = offs_h < H_
    m = mt[:, None] & mh[None, :]
    base = proj_ptr + offs_t[:, None] * stride_pt + offs_h[None, :]
    b = tl.load(base, mask=m, other=0.0)
    x = tl.load(base + 2 * H_, mask=m, other=0.0)
    bx = (b * x).to(out_ptr.dtype.element_ty)
    out_ptrs = out_ptr + offs_h[:, None] * T + offs_t[None, :]
    tl.store(out_ptrs, tl.trans(bx), mask=mh[:, None] & mt[None, :])


def gate_transpose(proj: torch.Tensor, T: int, out: torch.Tensor | None = None):
    if out is None:
        out = torch.empty((H, T), device=proj.device, dtype=proj.dtype)
    BLOCK_T, BLOCK_H = 64, 64
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
    _gate_transpose_kernel[grid](
        proj, out, T, H, proj.stride(0), BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, num_warps=4
    )
    return out


@triton.jit
def _gate_kernel(proj_ptr, out_ptr, n, H_: tl.constexpr, BLOCK: tl.constexpr):
    """proj [T, 3H] -> out [T, H]; out[t,h] = proj[t,h] * proj[t,2H+h] (decode)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    t = offs // H_
    h = offs % H_
    src = t * (3 * H_) + h
    b = tl.load(proj_ptr + src, mask=mask, other=0.0)
    x = tl.load(proj_ptr + src + 2 * H_, mask=mask, other=0.0)
    tl.store(out_ptr + offs, b * x, mask=mask)


def gate_only(proj, T, out=None):
    if out is None:
        out = torch.empty((T, H), device=proj.device, dtype=proj.dtype)
    n = T * H
    _gate_kernel[(triton.cdiv(n, 1024),)](proj, out, n, H, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _out_gate_untranspose_kernel(
    proj_ptr, conv_ptr, out_ptr, T, H_: tl.constexpr, BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """out[t,h] = proj[t, H + h] * conv[h, t]   (conv is [H, T])."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mt = offs_t < T
    mh = offs_h < H_
    c = tl.load(
        conv_ptr + offs_h[:, None] * T + offs_t[None, :],
        mask=mh[:, None] & mt[None, :], other=0.0,
    )
    cT = tl.trans(c)
    g = tl.load(
        proj_ptr + offs_t[:, None] * (3 * H_) + H_ + offs_h[None, :],
        mask=mt[:, None] & mh[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr + offs_t[:, None] * H_ + offs_h[None, :], g * cT,
        mask=mt[:, None] & mh[None, :],
    )


def out_gate_untranspose(proj, conv_out_t, T, out=None):
    if out is None:
        out = torch.empty((T, H), device=proj.device, dtype=proj.dtype)
    BLOCK_T, BLOCK_H = 64, 64
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
    _out_gate_untranspose_kernel[grid](
        proj, conv_out_t, out, T, H, BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, num_warps=4
    )
    return out


# =========================================================================
# timing helper
# =========================================================================
def bench(fn, warmup=25, iters=100):
    """Wall-clock per-iteration time. At small shapes this is CPU-dispatch bound."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # microseconds


def bench_graph(fn, warmup=8, inner=32, iters=30, reps=3):
    """True GPU time: capture `inner` calls into a CUDA graph and replay.

    This is the number that matters for SGLang decode, which runs the whole
    forward inside a captured CUDA graph, so per-op launch overhead is gone.
    Returns microseconds per single call.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
    except Exception as ex:
        print(f"    [graph capture failed: {type(ex).__name__}: {ex}]")
        return float("nan")
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / (iters * inner))
    return best


def rel(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


# =========================================================================
def run(args):
    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)
    results = []

    w_in = torch.randn(3 * H, H, device=dev, dtype=dt) * 0.02
    w_out = torch.randn(H, H, device=dev, dtype=dt) * 0.02
    conv_w = torch.randn(H, 3, device=dev, dtype=dt) * 0.2

    print("NOTE: 'wall' = per-call wall time (CPU-dispatch bound at small T).")
    print("      'gpu'  = CUDA-graph replay time = true kernel time, which is what")
    print("               SGLang decode sees (it runs the forward in a CUDA graph).")

    # ------------------------------------------------------------------
    # A. prefill input side:  chunk + B*x + transpose + contiguous
    # ------------------------------------------------------------------
    print("\n=== A. prefill input side: chunk + (B*x) + transpose().contiguous() ===")
    print(f"{'T':>7} | {'eager wall':>10} {'eager gpu':>10} | {'triton wall':>11} "
          f"{'triton gpu':>10} | {'inductor gpu':>12} | {'gpu speedup':>11} | "
          f"{'HBM before':>10} {'HBM after':>9}")
    fused_compiled = torch.compile(
        lambda p: (p[:, :H] * p[:, 2 * H:]).transpose(0, 1).contiguous(), dynamic=False
    )
    for T in args.prefill_T:
        proj = torch.randn(T, 3 * H, device=dev, dtype=dt)
        out_t = torch.empty(H, T, device=dev, dtype=dt)

        def eager():
            B, Cg, x = proj.chunk(3, dim=-1)
            return (B * x).transpose(0, 1).contiguous()

        ref = eager()
        got = gate_transpose(proj, T, out_t)
        err = rel(got, ref)
        err_i = rel(fused_compiled(proj), ref)
        e_w, e_g = bench(eager), bench_graph(eager)
        t_w = bench(lambda: gate_transpose(proj, T, out_t))
        t_g = bench_graph(lambda: gate_transpose(proj, T, out_t))
        i_g = bench_graph(lambda: fused_compiled(proj))
        before = 5 * T * H * BF
        after = 3 * T * H * BF
        print(f"{T:>7} | {e_w:>8.1f}us {e_g:>8.1f}us | {t_w:>9.1f}us {t_g:>8.1f}us | "
              f"{i_g:>10.1f}us | {e_g/t_g:>10.2f}x | {before/1e6:>7.2f}MB "
              f"{after/1e6:>6.2f}MB   err={err:.1e}/{err_i:.1e}")
        results.append(dict(bench="A_prefill_input", T=T, eager_wall_us=e_w,
                            eager_gpu_us=e_g, triton_wall_us=t_w, triton_gpu_us=t_g,
                            inductor_gpu_us=i_g, hbm_before_B=before,
                            hbm_after_B=after, relerr_triton=err,
                            relerr_inductor=err_i))

    # ------------------------------------------------------------------
    # B. prefill output side: C_gate * conv_out(transposed view)
    # ------------------------------------------------------------------
    print("\n=== B. prefill output side: C_gate * conv_out  (conv_out is [H,T]) ===")
    print(f"{'T':>7} | {'eager gpu':>10} {'triton gpu':>11} {'inductor gpu':>12} "
          f"{'speedup':>8}")
    out_compiled = torch.compile(
        lambda p, co: p[:, H:2 * H] * co.transpose(0, 1), dynamic=False
    )
    for T in args.prefill_T:
        proj = torch.randn(T, 3 * H, device=dev, dtype=dt)
        conv_t = torch.randn(H, T, device=dev, dtype=dt)
        obuf = torch.empty(T, H, device=dev, dtype=dt)

        def eager():
            B, Cg, x = proj.chunk(3, dim=-1)
            return Cg * conv_t.transpose(0, 1)

        ref = eager()
        err = rel(out_gate_untranspose(proj, conv_t, T, obuf), ref)
        err_i = rel(out_compiled(proj, conv_t), ref)
        e_g = bench_graph(eager)
        t_g = bench_graph(lambda: out_gate_untranspose(proj, conv_t, T, obuf))
        i_g = bench_graph(lambda: out_compiled(proj, conv_t))
        print(f"{T:>7} | {e_g:>8.1f}us {t_g:>9.1f}us {i_g:>10.1f}us {e_g/t_g:>7.2f}x  "
              f"err={err:.1e}/{err_i:.1e}")
        results.append(dict(bench="B_prefill_output", T=T, eager_gpu_us=e_g,
                            triton_gpu_us=t_g, inductor_gpu_us=i_g,
                            relerr_triton=err))

    # ------------------------------------------------------------------
    # C. full ShortConv prefill:  eager vs fused-input vs fused-in+out
    # ------------------------------------------------------------------
    print("\n=== C. full ShortConv prefill forward (gpu time) ===")
    print(f"{'T':>7} | {'stock':>9} {'+fusedIn':>9} {'+fusedInOut':>11} {'saved':>8} "
          f"{'gain%':>7}")
    for T in args.prefill_T:
        hs = torch.randn(T, H, device=dev, dtype=dt) * 0.05
        conv_state = torch.zeros(4, H, 3, device=dev, dtype=dt)
        qsl = torch.tensor([0, T], device=dev, dtype=torch.int32)
        ci = torch.zeros(1, device=dev, dtype=torch.int32)

        def stock():
            proj = torch.nn.functional.linear(hs, w_in)
            B, Cg, x = proj.chunk(3, dim=-1)
            Bx_t = (B * x).transpose(0, 1).contiguous()
            co = causal_conv1d_fn(Bx_t, conv_w, None, query_start_loc=qsl,
                                  cache_indices=ci, has_initial_state=None,
                                  conv_states=conv_state, activation=None)
            return torch.nn.functional.linear(Cg * co.transpose(0, 1), w_out)

        def fused_in():
            proj = torch.nn.functional.linear(hs, w_in)
            Bx_t = gate_transpose(proj, T)
            co = causal_conv1d_fn(Bx_t, conv_w, None, query_start_loc=qsl,
                                  cache_indices=ci, has_initial_state=None,
                                  conv_states=conv_state, activation=None)
            return torch.nn.functional.linear(proj[:, H:2 * H] * co.transpose(0, 1),
                                              w_out)

        def fused_both():
            proj = torch.nn.functional.linear(hs, w_in)
            Bx_t = gate_transpose(proj, T)
            co = causal_conv1d_fn(Bx_t, conv_w, None, query_start_loc=qsl,
                                  cache_indices=ci, has_initial_state=None,
                                  conv_states=conv_state, activation=None)
            return torch.nn.functional.linear(out_gate_untranspose(proj, co, T), w_out)

        conv_state.zero_(); r0 = stock()
        conv_state.zero_(); r1 = fused_in()
        conv_state.zero_(); r2 = fused_both()
        t0, t1, t2 = bench_graph(stock), bench_graph(fused_in), bench_graph(fused_both)
        print(f"{T:>7} | {t0:>7.1f}us {t1:>7.1f}us {t2:>9.1f}us {t0-t2:>6.1f}us "
              f"{100*(t0-t2)/t0:>6.1f}%  err {rel(r1,r0):.1e} {rel(r2,r0):.1e}")
        results.append(dict(bench="C_shortconv_prefill", T=T, stock_gpu_us=t0,
                            fused_in_gpu_us=t1, fused_both_gpu_us=t2,
                            relerr_in=rel(r1, r0), relerr_both=rel(r2, r0)))

    # ------------------------------------------------------------------
    # D. decode input side (no transpose) + the int32 index cast
    # ------------------------------------------------------------------
    print("\n=== D. decode: B*x gate, and req_pool_indices.to(int32) ===")
    print(f"{'T':>7} | {'gate eager gpu':>14} {'gate triton gpu':>15} "
          f"{'int32 cast gpu':>14} {'cast wall':>10}")
    for T in args.decode_T:
        proj = torch.randn(T, 3 * H, device=dev, dtype=dt)
        rpi = torch.arange(T, device=dev, dtype=torch.int64)
        obuf = torch.empty(T, H, device=dev, dtype=dt)

        def eager():
            B, Cg, x = proj.chunk(3, dim=-1)
            return B * x

        ref = eager()
        err = rel(gate_only(proj, T, obuf), ref)
        e_g = bench_graph(eager)
        t_g = bench_graph(lambda: gate_only(proj, T, obuf))
        c_g = bench_graph(lambda: rpi.to(torch.int32))
        c_w = bench(lambda: rpi.to(torch.int32))
        print(f"{T:>7} | {e_g:>12.2f}us {t_g:>13.2f}us {c_g:>12.2f}us {c_w:>8.2f}us  "
              f"err={err:.1e}")
        results.append(dict(bench="D_decode_gate", T=T, eager_gpu_us=e_g,
                            triton_gpu_us=t_g, int32_cast_gpu_us=c_g,
                            int32_cast_wall_us=c_w))

    # ------------------------------------------------------------------
    # E. full ShortConv decode
    # ------------------------------------------------------------------
    print("\n=== E. full ShortConv decode forward (18 conv layers per fwd) ===")
    print(f"{'T':>7} | {'stock gpu':>10} {'hoisted cast':>12} {'+fused gate':>12} "
          f"{'saved/layer':>11} {'x18':>8}")
    for T in args.decode_T:
        hs = torch.randn(T, H, device=dev, dtype=dt) * 0.05
        conv_state = torch.zeros(max(T, 8), H, 3, device=dev, dtype=dt)
        rpi = torch.arange(T, device=dev, dtype=torch.int64)
        rpi32 = rpi.to(torch.int32)

        def stock():
            proj = torch.nn.functional.linear(hs, w_in)
            B, Cg, x = proj.chunk(3, dim=-1)
            co = causal_conv1d_update(B * x, conv_state, conv_w, None, activation=None,
                                      conv_state_indices=rpi.to(torch.int32))
            return torch.nn.functional.linear(Cg * co, w_out)

        def hoisted():
            proj = torch.nn.functional.linear(hs, w_in)
            B, Cg, x = proj.chunk(3, dim=-1)
            co = causal_conv1d_update(B * x, conv_state, conv_w, None, activation=None,
                                      conv_state_indices=rpi32)
            return torch.nn.functional.linear(Cg * co, w_out)

        def hoisted_fused():
            proj = torch.nn.functional.linear(hs, w_in)
            co = causal_conv1d_update(gate_only(proj, T), conv_state, conv_w, None,
                                      activation=None, conv_state_indices=rpi32)
            return torch.nn.functional.linear(proj[:, H:2 * H] * co, w_out)

        conv_state.zero_(); r0 = stock()
        conv_state.zero_(); r2 = hoisted_fused()
        t0 = bench_graph(stock)
        t1 = bench_graph(hoisted)
        t2 = bench_graph(hoisted_fused)
        print(f"{T:>7} | {t0:>8.2f}us {t1:>10.2f}us {t2:>10.2f}us "
              f"{t0-t1:>9.2f}us {18*(t0-t1):>6.1f}us  err={rel(r2,r0):.1e}")
        results.append(dict(bench="E_shortconv_decode", T=T, stock_gpu_us=t0,
                            hoisted_cast_gpu_us=t1, hoisted_fused_gpu_us=t2))

    # ------------------------------------------------------------------
    # F. moe_sum_reduce_torch_compile  (== triton_poi_fused_copy__mul_sum_0)
    # ------------------------------------------------------------------
    print("\n=== F. moe_sum_reduce_torch_compile (triton_poi_fused_copy__mul_sum_0) ===")

    @torch.compile
    def moe_sum_reduce_torch_compile(x, out, rsf):
        torch.sum(x, dim=1, out=out)
        out.mul_(rsf)

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_sum_reduce

    print(f"{'T':>7} | {'compiled gpu':>12} {'compiled wall':>13} {'moe_sum_reduce':>14} "
          f"{'add-pair':>9} | {'traffic':>8} {'eff BW':>9}")
    for T in args.decode_T:
        ic3 = torch.randn(T, C.TOP_K, H, device=dev, dtype=dt)
        outb = torch.empty(T, H, device=dev, dtype=dt)
        moe_sum_reduce_torch_compile(ic3, outb, 1.0)
        c_g = bench_graph(lambda: moe_sum_reduce_torch_compile(ic3, outb, 1.0))
        c_w = bench(lambda: moe_sum_reduce_torch_compile(ic3, outb, 1.0))
        try:
            moe_sum_reduce(ic3, outb, 1.0)
            m_g = bench_graph(lambda: moe_sum_reduce(ic3, outb, 1.0))
        except Exception as ex:
            print(f"    [moe_sum_reduce failed: {type(ex).__name__}: {ex}]")
            m_g = float("nan")
        a_g = bench_graph(lambda: torch.add(ic3[:, 0] + ic3[:, 1],
                                            ic3[:, 2] + ic3[:, 3], out=outb))
        traffic = (T * C.TOP_K * H + T * H) * BF
        print(f"{T:>7} | {c_g:>10.2f}us {c_w:>11.2f}us {m_g:>12.2f}us {a_g:>7.2f}us | "
              f"{traffic/1e6:>6.3f}MB {traffic/c_g*1e-3:>7.1f}GB/s")
        results.append(dict(bench="F_moe_sum_reduce", T=T, compiled_gpu_us=c_g,
                            compiled_wall_us=c_w, moe_sum_reduce_gpu_us=m_g,
                            addpair_gpu_us=a_g, traffic_B=traffic))

    outp = os.path.join(C.outdir(), "bench_fusions.json")
    with open(outp, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0),
                   "torch": torch.__version__, "triton": triton.__version__,
                   "results": results}, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-T", type=int, nargs="+",
                    default=[512, 1000, 2048, 4096, 8192, 16000])
    ap.add_argument("--decode-T", type=int, nargs="+", default=[1, 4, 8, 32, 64, 128])
    run(ap.parse_args())
