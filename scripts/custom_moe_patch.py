"""Monkeypatch sglang's TritonRunner.run to use a custom small-M (decode) MoE kernel
when M<=SMALL_M_MAX and the case is bf16/gated/no-quant/shape-matched. Falls back to
the original sglang implementation otherwise.

Install: `import custom_moe_patch; custom_moe_patch.install()` BEFORE model load.
Enable via env CUSTOM_MOE=1.
"""
import os
import torch
import triton
import triton.language as tl

SMALL_M_MAX = int(os.environ.get("CUSTOM_MOE_MAX_M", "4"))
_stats = {"custom": 0, "fallback": 0}


@triton.jit
def _w1_act(x_ptr, w1_ptr, ids_ptr, tok_ptr, act_ptr, H: tl.constexpr, I: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    p = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(ids_ptr + p); t = tl.load(tok_ptr + p)
    n = nt * BN + tl.arange(0, BN); m = tl.arange(0, BM)
    accg = tl.zeros((BM, BN), dtype=tl.float32); accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        koff = k0 + tl.arange(0, BK)
        xb = tl.load(x_ptr + t * H + koff[None, :] + m[:, None] * 0, mask=m[:, None] < 1, other=0.0)
        wg = tl.load(w1_ptr + e * (2 * I * H) + n[:, None] * H + koff[None, :]).to(tl.bfloat16)
        wu = tl.load(w1_ptr + e * (2 * I * H) + (I + n[:, None]) * H + koff[None, :]).to(tl.bfloat16)
        accg += tl.dot(xb, wg.T); accu += tl.dot(xb, wu.T)
    silu = accg / (1.0 + tl.exp(-accg))
    tl.store(act_ptr + p * I + n, tl.sum(silu * accu, axis=0).to(tl.bfloat16))


@triton.jit
def _w2_sum(act_ptr, w2_ptr, ids_ptr, tok_ptr, tw_ptr, out_ptr, rsf,
            H: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    p = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(ids_ptr + p); t = tl.load(tok_ptr + p); tw = tl.load(tw_ptr + p).to(tl.float32)
    n = nt * BN + tl.arange(0, BN); m = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        koff = k0 + tl.arange(0, BK)
        ab = tl.load(act_ptr + p * I + koff[None, :] + m[:, None] * 0, mask=m[:, None] < 1, other=0.0).to(tl.bfloat16)
        w = tl.load(w2_ptr + e * (H * I) + n[:, None] * I + koff[None, :]).to(tl.bfloat16)
        acc += tl.dot(ab, w.T)
    tl.atomic_add(out_ptr + t * H + n, tl.sum(acc, axis=0) * tw * rsf)


def _custom_moe(hidden_states, w13, w2, topk_weights, topk_ids, routed_scaling_factor):
    M, H = hidden_states.shape
    E, N2, _ = w13.shape          # N2 = 2*I
    I = N2 // 2
    topk = topk_ids.shape[1]
    ids = topk_ids.reshape(-1).to(torch.int32)
    tok = torch.arange(M, device=hidden_states.device, dtype=torch.int32).repeat_interleave(topk)
    tw = topk_weights.reshape(-1).float()
    P = ids.numel()
    act = torch.empty(P, I, dtype=torch.bfloat16, device=hidden_states.device)
    out = torch.zeros(M, H, dtype=torch.float32, device=hidden_states.device)
    _w1_act[(P, I // 64)](hidden_states, w13, ids, tok, act, H, I, 64, 256, 16, num_warps=4)
    _w2_sum[(P, H // 64)](act, w2, ids, tok, tw, out, float(routed_scaling_factor),
                          H, I, 64, 128, 16, num_warps=4)
    return out.to(hidden_states.dtype)


def install():
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fm
    if getattr(fm, "_custom_patched", False):
        return
    orig_impl = fm.fused_experts_impl

    def _arg(args, kwargs, idx, name, default):
        if len(args) > idx:
            return args[idx]
        return kwargs.get(name, default)

    def patched_impl(*args, **kwargs):
        hs = _arg(args, kwargs, 0, "hidden_states", None)
        w1 = _arg(args, kwargs, 1, "w1", None)
        w2 = _arg(args, kwargs, 2, "w2", None)
        tw = _arg(args, kwargs, 3, "topk_weights", None)
        tid = _arg(args, kwargs, 4, "topk_ids", None)
        b1 = _arg(args, kwargs, 5, "b1", None)
        b2 = _arg(args, kwargs, 6, "b2", None)
        inplace = _arg(args, kwargs, 7, "inplace", False)
        activation = _arg(args, kwargs, 8, "activation", "silu")
        is_gated = _arg(args, kwargs, 9, "is_gated", True)
        arwoi = _arg(args, kwargs, 10, "apply_router_weight_on_input", False)
        fp8 = _arg(args, kwargs, 11, "use_fp8_w8a8", False)
        i8a8 = _arg(args, kwargs, 12, "use_int8_w8a8", False)
        i8a16 = _arg(args, kwargs, 13, "use_int8_w8a16", False)
        i4a16 = _arg(args, kwargs, 14, "use_int4_w4a16", False)
        block_shape = _arg(args, kwargs, 22, "block_shape", None)
        rsf = _arg(args, kwargs, 24, "routed_scaling_factor", None)
        g_alpha = _arg(args, kwargs, 25, "gemm1_alpha", None)
        g_limit = _arg(args, kwargs, 26, "gemm1_limit", None)
        M = hs.shape[0] if hs is not None else 999
        ok = (
            os.environ.get("CUSTOM_MOE", "0") == "1"
            and M <= SMALL_M_MAX
            and hs is not None and hs.dtype == torch.bfloat16
            and is_gated and activation == "silu" and not arwoi
            and g_alpha is None and g_limit is None
            and not (fp8 or i8a8 or i8a16 or i4a16)
            and b1 is None and b2 is None and block_shape is None
        )
        if ok:
            H = hs.shape[1]; E, N2, K = w1.shape; I = N2 // 2
            if K == H and tuple(w2.shape) == (E, H, I) and I % 64 == 0 and H % 64 == 0:
                out = _custom_moe(hs, w1, w2, tw, tid, rsf if rsf is not None else 1.0)
                _stats["custom"] += 1
                if os.environ.get("CUSTOM_MOE_VERIFY", "0") == "1" and _stats["custom"] <= 20:
                    ref = orig_impl(*args, **kwargs)
                    rel = ((ref.float() - out.float()).abs() / (ref.float().abs() + 1e-2)).max().item()
                    _stats.setdefault("max_rel_err", 0.0)
                    _stats["max_rel_err"] = max(_stats["max_rel_err"], rel)
                if inplace:
                    hs.copy_(out); return hs
                return out
        _stats["fallback"] += 1
        return orig_impl(*args, **kwargs)

    fm.fused_experts_impl = patched_impl
    fm._custom_patched = True
    print(f"[custom_moe_patch] installed on fused_experts_impl (SMALL_M_MAX={SMALL_M_MAX}, CUSTOM_MOE={os.environ.get('CUSTOM_MOE','0')})", flush=True)


def stats():
    return dict(_stats)
