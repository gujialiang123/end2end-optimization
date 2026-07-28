"""Opt-in fusion patch for SGLang's Gemma-3 implementation.

Found by extending the operator-level audit (`lf_audit.py`) to a fourth
architecture. The signature that found it is the *static* one: enumerate the
fused primitives the codebase already ships, then find models whose call sites
never call them.

**The gap.** `sglang/srt/layers/layernorm.py` defines two Gemma norm classes
about a hundred lines apart:

* `GemmaRMSNorm` (line ~402) calls `gemma_rmsnorm` / `gemma_fused_add_rmsnorm`
  — pre-built fused CUDA kernels shipped in `sgl_kernel`;
* `Gemma3RMSNorm` (line ~505) has

  ```python
  def forward_cuda(self, x):
      return self.forward_native(x)          # <- eager PyTorch
  def _norm(self, x):
      return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
  ```

so every Gemma-3 norm runs as `pow -> mean -> add -> rsqrt -> mul -> mul` plus
the fp32 up/down casts, i.e. roughly **six kernels instead of one**. Note the
CPU path *does* dispatch to a fused kernel
(`torch.ops.sgl_kernel.gemma3_rmsnorm_cpu`) and so does NPU
(`torch_npu.npu_gemma_rms_norm`) — CUDA is the one that falls through.

Gemma-3-1B has 26 layers x 6 norms (input, post_attention, pre_feedforward,
post_feedforward, q_norm, k_norm) = **157 norm calls per forward**. Measured
cost of just the mean/rsqrt/pow part of the decomposition:

| regime / stage | share of CUDA kernel time |
|---|---|
| low-batch decode | **15.98 %** |
| decode, long-prefill workload | **16.35 %** |
| prefill (short) | **19.31 %** |
| prefill (T=16000) | **11.07 %** |

Every other model audited (LFM2.5, Qwen3-30B, Qwen3-0.6B) shows **0.00 %** and
zero such calls, so this is specific to Gemma-3 rather than a framework-wide
property.

**One caveat that matters.** `Gemma3RMSNorm.weight` is created by
`nn.Parameter(torch.zeros(dim))` and so can be fp32 while activations are bf16.
The fused kernel requires the weight to match the input dtype — passing an fp32
weight against a bf16 input silently produces NaNs. The cast is therefore done
once per module and cached, never per call.

The eager path computes the whole normalisation in fp32 and casts once at the
end; the fused kernel keeps the weight multiply in the activation dtype. The
result is not bit-identical — measured at 1-2 bf16 ulp — which is the same
trade-off SGLang already accepts for `GemmaRMSNorm` on gemma/gemma2.

Activation (default off; without the variable the stock path is untouched):

    GEMMA_FUSION_PATCH=norm   python -m sglang.launch_server ...
"""
from __future__ import annotations

import os

import torch

_APPLIED: list[str] = []


def _enabled() -> set[str]:
    raw = os.environ.get("GEMMA_FUSION_PATCH", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def _fused_weight(self, like: torch.Tensor) -> torch.Tensor:
    """Weight in the activation dtype, computed once per module.

    Guarded on dtype identity rather than on a plain `hasattr`, so a module
    that is later moved or re-cast cannot keep serving a stale buffer.
    """
    cached = getattr(self, "_fused_w", None)
    if cached is not None and cached.dtype == like.dtype:
        return cached
    w = self.weight.data
    w = w.to(like.dtype).contiguous() if w.dtype != like.dtype else w.contiguous()
    self._fused_w = w
    return w


# Reproduces upstream main's norm coverage exactly: 2-D fused, higher rank
# falls back to eager. Used as the A/B baseline so the increment attributable to
# the high-rank fix is measured directly rather than inferred across runs.
def _patched_gemma3_forward_cuda_2d_only(self, x: torch.Tensor) -> torch.Tensor:
    from sgl_kernel import gemma_rmsnorm

    if x.dtype not in (torch.bfloat16, torch.float16) or x.dim() != 2:
        return self.forward_native(x)
    return gemma_rmsnorm(x.contiguous(), _fused_weight(self, x), self.eps)


def _patched_gemma3_forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
    from sgl_kernel import gemma_rmsnorm

    if x.dtype not in (torch.bfloat16, torch.float16):
        return self.forward_native(x)

    # The kernel wants [tokens, hidden]. RMSNorm normalises over the last
    # dimension only, so any higher-rank tensor can be flattened to 2-D and
    # restored afterwards without changing the result. This matters: q_norm and
    # k_norm are called with [tokens, heads, head_dim], and they are 2 of the 6
    # norms per layer — a rank guard alone would leave a third of them eager.
    if x.dim() == 2:
        return gemma_rmsnorm(x.contiguous(), _fused_weight(self, x), self.eps)
    if x.dim() > 2 and x.shape[-1] == self.weight.numel():
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = gemma_rmsnorm(flat, _fused_weight(self, x), self.eps)
        return out.view_as(x)
    return self.forward_native(x)


# ---------------------------------------------------------------------------
# `residual` — fold the post-attention residual add into the following norm
# ---------------------------------------------------------------------------
# After the `norm` component lands, the audit's next-largest gap in Gemma-3 is
# 52 standalone residual adds per forward (2.00 per layer, 3.00 % of decode
# kernel time). The layer does:
#
#     h = post_attention_layernorm(attn_out)
#     h = residual + h                        # <- standalone add
#     residual = h
#     h = pre_feedforward_layernorm(h)        # <- immediately followed by a norm
#
# `add then norm` is exactly `gemma_fused_add_rmsnorm`, which is already in the
# CUDA build. Only the FIRST of the two adds is fused here: the second one's
# following norm is the *next layer's* `input_layernorm`, so fusing it would
# mean carrying a residual across the layer boundary and changing the layer's
# return signature. That is deliberately left alone.
def _patched_gemma3_layer_forward(
    self, positions, hidden_states, position_embeddings_global,
    position_embeddings_local, forward_batch, **kwargs,
):
    from sgl_kernel import gemma_fused_add_rmsnorm

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    position_embeddings = (position_embeddings_local if self.self_attn.is_sliding
                           else position_embeddings_global)
    hidden_states = self.self_attn(
        positions=positions, hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        forward_batch=forward_batch, **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)

    ffn_norm = self.pre_feedforward_layernorm
    if _gemma3_fused_add_ok(hidden_states, residual, ffn_norm):
        # in-place: residual := residual + hidden_states,
        #           hidden_states := pre_feedforward_layernorm(residual)
        hidden_states = hidden_states.contiguous()
        residual = residual.contiguous()
        gemma_fused_add_rmsnorm(
            hidden_states, residual,
            ffn_norm._fused_weight(hidden_states.dtype), ffn_norm.eps)
    else:
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = ffn_norm(hidden_states)

    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    return (hidden_states,)


def _gemma3_fused_add_ok(x, residual, norm) -> bool:
    return (
        x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and residual.shape == x.shape
        and residual.dtype == x.dtype
        and hasattr(norm, "_fused_weight")
        and x.shape[-1] == norm.weight.numel()
    )


def apply_for(module_name: str) -> list[str]:
    """Apply the components owned by `module_name`, once it has finished importing."""
    want = _enabled()
    if not want:
        return []
    unknown = want - {"norm", "norm2d", "residual", "olmo2_qknorm"}
    if unknown:
        raise ValueError(f"unknown GEMMA_FUSION_PATCH components: {sorted(unknown)}")

    if module_name == "sglang.srt.models.gemma3_causal":
        if "residual" in want:
            from sglang.srt.models import gemma3_causal as G

            G.Gemma3DecoderLayer.forward = _patched_gemma3_layer_forward
            _APPLIED.append("residual")
            print(f"[gemma_fusion_patch] applied: {_APPLIED}", flush=True)
        return list(_APPLIED)

    from sglang.srt.layers import layernorm as LN

    if want & {"norm", "norm2d"}:
        LN.Gemma3RMSNorm.forward_cuda = (
            _patched_gemma3_forward_cuda_2d_only if "norm2d" in want
            else _patched_gemma3_forward_cuda)
        # MultiPlatformOp binds the dispatch target at construction time, so
        # replacing the method alone is not enough for modules that already
        # exist; patch the constructor too.
        orig_init = LN.Gemma3RMSNorm.__init__

        def patched_init(self, dim, eps=1e-6):
            orig_init(self, dim, eps)
            self._forward_method = self.forward_cuda

        LN.Gemma3RMSNorm.__init__ = patched_init
        _APPLIED.append("norm2d" if "norm2d" in want else "norm")

    print(f"[gemma_fusion_patch] applied: {_APPLIED}", flush=True)
    return list(_APPLIED)


# ---------------------------------------------------------------------------
# OLMo-2 — `_apply_qk_norm` calls forward_native explicitly on the non-capture
# path, bypassing the fused kernel.
# ---------------------------------------------------------------------------
# Found by the same static signature. `models/olmo2.py:190-191`:
#
#     else:
#         q = self.q_norm.forward_native(q)     # <- explicit eager call
#         k = self.k_norm.forward_native(k)
#
# The `if` branch immediately above already proves the fused call works — it
# reshapes to 2-D and calls `self.q_norm(...)`, which dispatches to the fused
# CUDA kernel. Only the fallback path was left eager. Measured at 7.71 % of
# decode CUDA kernel time (32 calls, 2.00/layer) on OLMo-2-1B.
def _patched_olmo2_apply_qk_norm(self, q, k):
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank, tensor_model_parallel_all_gather)
    from sglang.srt.layers.dp_attention import get_is_capture_mode
    from sglang.srt.utils import split_tensor_along_last_dim
    from functools import partial
    import torch

    if self.tp_size > 1:
        q = tensor_model_parallel_all_gather(q.contiguous())
        k = tensor_model_parallel_all_gather(k.contiguous())

    if self.alt_stream is not None and get_is_capture_mode():
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        q_shape, k_shape = q.shape, k.shape
        q_by_last = self.q_norm(q.reshape(-1, q_shape[-1]))
        with torch.cuda.stream(self.alt_stream):
            k_by_last = self.k_norm(k.reshape(-1, k_shape[-1]))
        current_stream.wait_stream(self.alt_stream)
        q, k = q_by_last.view(q_shape), k_by_last.view(k_shape)
    else:
        # was: self.q_norm.forward_native(q). RMSNorm reduces over the last
        # dimension only, so reshaping to 2-D and back is exact, and is exactly
        # what the capture-mode branch above already does.
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(q.reshape(-1, q_shape[-1])).view(q_shape)
        k = self.k_norm(k.reshape(-1, k_shape[-1])).view(k_shape)

    if self.tp_size > 1:
        splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
        q = splitter(q)[self.tp_rank]
        k = splitter(k)[self.tp_rank]
    return q, k


def apply_olmo2() -> bool:
    if "olmo2_qknorm" not in _enabled():
        return False
    from sglang.srt.models import olmo2 as M

    M.Olmo2Attention._apply_qk_norm = _patched_olmo2_apply_qk_norm
    print("[gemma_fusion_patch] applied: ['olmo2_qknorm']", flush=True)
    return True
