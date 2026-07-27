"""Opt-in fusion patches for SGLang's LFM2.5 (Lfm2Moe) implementation.

Three gaps were found by the operator-level audit
(`scripts/lfm_fusion/lf_audit.py`, results in `results/lfm_fusion/audit/`).
Each is enabled independently so the end-to-end A/B can attribute the effect.

G1 `norm`  — the decoder layer calls RMSNorm without a residual and then adds
             the residual with a separate elementwise kernel.  SGLang's
             `RMSNorm.forward_cuda(x, residual)` dispatches to
             `fused_add_rmsnorm`, which does both in one pass.  The model loop
             already threads a `residual` through the layers, but the layer
             ignores it.  Rewriting to the deferred-residual convention used by
             every other SGLang model removes 2 kernels per layer (48 total).

G2 `scale` — `Lfm2MoeSparseMoeBlock.forward` ends with
             `final_hidden_states * self.routed_scaling_factor`.  For LFM2.5
             `routed_scaling_factor == 1.0`, so this is a full-tensor
             elementwise multiply by one: 22 no-op kernel launches per forward.
             Skipping it when the factor is exactly 1.0 is bit-exact.

G3 `conv`  — `Lfm2MoeShortConv.forward` materialises `Bx.transpose(0, 1)
             .contiguous()` on the prefill path.  Not yet implemented; the
             audit is recorded so the cost is known.

Activation (default is fully off — the stock code path is untouched):

    LFM_FUSION_PATCH=norm,scale   python -m sglang.launch_server ...

Import happens through `sitecustomize.py` in this directory, which is put on
PYTHONPATH by the A/B driver.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

_APPLIED: list[str] = []


def _enabled() -> set[str]:
    raw = os.environ.get("LFM_FUSION_PATCH", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


# ---------------------------------------------------------------------------
# G1 — deferred residual + fused_add_rmsnorm
# ---------------------------------------------------------------------------
def _patched_layer_forward(
    self,
    layer_id: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    forward_batch,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deferred-residual form.

    Invariant: the mathematical activation entering the layer is
    ``hidden_states + residual`` (or just ``hidden_states`` when residual is
    None).  `RMSNorm(x, residual)` performs ``residual += x`` and normalises the
    updated residual in a single kernel, so the separate add disappears.

    Equivalence to the stock layer, writing ``x`` for the incoming activation:
        stock:  a = op(rms(x));  h1 = a + x;  out = h1 + ffn(rms(h1))
        here:   r' = x;          n = rms(x);   a = op(n)
                r'' = a + x = h1; n2 = rms(h1); out_pair = (ffn(n2), h1)
    and the next layer / final norm consumes ``ffn(n2) + h1`` — identical.
    """
    if forward_batch.forward_mode.is_idle():
        return hidden_states, residual

    if residual is None:
        residual = hidden_states
        normed = self.operator_norm(hidden_states)
    else:
        normed, residual = self.operator_norm(hidden_states, residual)

    if self.is_attention_layer:
        hidden_states = self.self_attn(positions, normed, forward_batch)
    else:
        hidden_states = self.conv(normed, forward_batch)

    hidden_states, residual = self.ffn_norm(hidden_states, residual)
    hidden_states = self.feed_forward(hidden_states)
    return hidden_states, residual


def _patched_model_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    hidden_states = (
        inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
    )
    residual = None
    for i in range(len(self.layers)):
        hidden_states, residual = self.layers[i](
            layer_id=i,
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )
    if residual is None:
        return self.embedding_norm(hidden_states)
    hidden_states, _ = self.embedding_norm(hidden_states, residual)
    return hidden_states


# ---------------------------------------------------------------------------
# G2 — skip the multiply by routed_scaling_factor == 1.0
# ---------------------------------------------------------------------------
def _patched_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    router_logits, _ = self.gate(hidden_states)
    topk_output = self.topk(hidden_states, router_logits)
    final_hidden_states = self.experts(hidden_states, topk_output)
    if self.routed_scaling_factor == 1.0:
        return final_hidden_states
    return final_hidden_states * self.routed_scaling_factor


def apply() -> list[str]:
    want = _enabled()
    if not want:
        return []
    from sglang.srt.models import lfm2_moe as M

    if "norm" in want:
        M.Lfm2MoeDecoderLayer.forward = _patched_layer_forward
        M.Lfm2MoeModel.forward = _patched_model_forward
        _APPLIED.append("norm")
    if "scale" in want:
        M.Lfm2MoeSparseMoeBlock.forward = _patched_moe_forward
        _APPLIED.append("scale")

    unknown = want - {"norm", "scale"}
    if unknown:
        raise ValueError(f"unknown LFM_FUSION_PATCH components: {sorted(unknown)}")
    print(f"[lfm_fusion_patch] applied: {_APPLIED}", flush=True)
    return list(_APPLIED)
