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


def apply() -> list[str]:
    want = _enabled()
    if not want:
        return []
    from sglang.srt.layers import layernorm as LN

    unknown = want - {"norm"}
    if unknown:
        raise ValueError(f"unknown GEMMA_FUSION_PATCH components: {sorted(unknown)}")

    if "norm" in want:
        LN.Gemma3RMSNorm.forward_cuda = _patched_gemma3_forward_cuda
        # MultiPlatformOp binds the dispatch target at construction time, so
        # replacing the method alone is not enough for modules that already
        # exist; patch the constructor too.
        orig_init = LN.Gemma3RMSNorm.__init__

        def patched_init(self, dim, eps=1e-6):
            orig_init(self, dim, eps)
            self._forward_method = self.forward_cuda

        LN.Gemma3RMSNorm.__init__ = patched_init
        _APPLIED.append("norm")

    print(f"[gemma_fusion_patch] applied: {_APPLIED}", flush=True)
    return list(_APPLIED)
