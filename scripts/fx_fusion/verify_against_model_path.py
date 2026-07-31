#!/usr/bin/env python3
"""Does the fused kernel reproduce Gemma-3's *actual* attention preamble?

The earlier feasibility run compared against `get_rope`, but the CUDA path in
gemma3_causal.py does not use it: `forward_native` receives precomputed cos/sin
from Gemma3RotaryEmbedding and calls apply_rotary_pos_emb, after transposing to
[b, h, s, d]. Those are three chances for a convention mismatch -- the layout,
the attention_scaling factor, and whether cos/sin are built from the same base
the kernel is told to use.

So this replays the model's own code with the model's own rotary module, per
layer type, before anything is patched into the model.

Both paths are scored against an fp64 reference rather than against each other.
An earlier version diffed the two bf16 outputs directly and reported 87% at the
global rope base, which was an artefact of that comparison: measured against
fp64, the model path and the kernel are 0.19% and 0.14% off respectively, i.e.
both are simply quantised and the kernel is the closer of the two. Comparing two
approximations to each other cannot tell you which one is wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.runtime_context import _CONTEXT
from sglang.srt.server_args import ServerArgs

if not getattr(_CONTEXT, "_server_args", None):
    _CONTEXT.set_server_args(ServerArgs(model_path="dummy"))

from transformers import AutoConfig

from sglang.kernels.ops.attention.fused_qknorm_rope import fused_qk_norm_rope
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb
from sglang.srt.models.gemma3_causal import Gemma3RotaryEmbedding


def model_path_preamble(qkv, q_norm, k_norm, cos, sin, nq, nk, nv, hd):
    """Verbatim from Gemma3Attention.forward_native."""
    q, k, _ = qkv.split([nq * hd, nk * hd, nv * hd], dim=-1)
    q = q.unflatten(-1, (nq, hd)).transpose(0, 1).unsqueeze(0)
    q = q_norm(q)
    k = k.unflatten(-1, (nk, hd)).transpose(0, 1).unsqueeze(0)
    k = k_norm(k)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    # [b, h, s, d] -> [b, s, h, d], then flatten to the kernel's layout
    q = q.permute(0, 2, 1, 3).reshape(-1, nq * hd)
    k = k.permute(0, 2, 1, 3).reshape(-1, nk * hd)
    return q, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/hf/models/gemma-3-1b-it")
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    cfg = AutoConfig.from_pretrained(a.model)
    tcfg = getattr(cfg, "text_config", cfg)
    hd = tcfg.head_dim
    nq = tcfg.num_attention_heads
    nk = nv = tcfg.num_key_value_heads
    eps = tcfg.rms_norm_eps
    print(f"{a.model}: head_dim={hd} q_heads={nq} kv_heads={nk} eps={eps}")

    q_norm = Gemma3RMSNorm(dim=hd, eps=eps).cuda()
    k_norm = Gemma3RMSNorm(dim=hd, eps=eps).cuda()
    for m in (q_norm, k_norm):
        m.weight.data.normal_(std=0.1)
        m.weight.data = m.weight.data.to(torch.bfloat16)

    local_base = getattr(tcfg, "rope_local_base_freq", 10000.0)
    global_base = getattr(tcfg, "rope_theta", 1000000.0)

    results = []
    for base, tag in ((local_base, "sliding/local"), (global_base, "full/global")):
        import copy
        c = copy.deepcopy(tcfg)
        c.rope_theta = base
        c.rope_parameters = {"rope_type": "default", "rope_theta": base}
        rot = Gemma3RotaryEmbedding(config=c).cuda()
        print(f"\n=== base {base:g} ({tag}) | attention_scaling="
              f"{getattr(rot, 'attention_scaling', 1.0)} ===")
        print(f"{'tokens':>7}{'model_err':>11}{'kernel_err':>12}{'verdict':>12}")

        for T in a.tokens:
            qkv = torch.randn(T, (nq + nk + nv) * hd, dtype=torch.bfloat16)
            pos = torch.arange(T, device="cuda")
            cos, sin = rot(torch.zeros(1, T, hd, dtype=torch.bfloat16),
                           pos.unsqueeze(0))

            mq, mk = model_path_preamble(qkv, q_norm, k_norm, cos, sin,
                                         nq, nk, nv, hd)

            buf = qkv.clone()
            fused_qk_norm_rope(buf, nq, nk, nv, hd, eps,
                               q_norm.weight.data, k_norm.weight.data,
                               float(base), True, pos.to(torch.int32),
                               1.0, 0.0, 0.0, 1.0, add_one=True)
            fq = buf[:, : nq * hd]

            # fp64 reference for q, so model and kernel are each scored against
            # ground truth rather than against each other. Comparing the two
            # bf16 paths directly conflates their independent rounding.
            x = qkv[:, : nq * hd].reshape(T, nq, hd).double()
            n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) \
                * (1.0 + q_norm.weight.data.double())
            half = hd // 2
            inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float64,
                                               device="cuda") / hd))
            ang = pos.double()[:, None] * inv[None, :]
            co, si = ang.cos()[:, None, :], ang.sin()[:, None, :]
            exact = torch.cat([n[..., :half] * co - n[..., half:] * si,
                               n[..., half:] * co + n[..., :half] * si],
                              dim=-1).reshape(T, nq * hd)
            scale = exact.abs().mean().clamp_min(1e-6)

            e_model = ((mq.double() - exact).abs().mean() / scale * 100).item()
            e_kern = ((fq.double() - exact).abs().mean() / scale * 100).item()
            # The kernel is acceptable if it is no worse than the path it
            # replaces, with a little slack for the two roundings differing.
            ok = e_kern <= e_model * 1.10 + 0.02
            print(f"{T:>7}{e_model:>10.3f}%{e_kern:>11.3f}%"
                  f"{'OK' if ok else 'WORSE':>12}")
            results.append(dict(base=float(base), layers=tag, tokens=T,
                                model_err_pct=round(e_model, 4),
                                kernel_err_pct=round(e_kern, 4),
                                kernel_no_worse=bool(ok)))

    all_ok = all(r["kernel_no_worse"] for r in results)
    print(f"\n{'kernel is no less accurate than the model path everywhere'
                if all_ok else 'KERNEL WORSE SOMEWHERE'}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            dict(model=a.model, head_dim=hd, all_ok=all_ok, results=results),
            indent=1))
        print(f"wrote {a.out}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
