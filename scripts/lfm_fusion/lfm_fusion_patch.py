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
             .contiguous()` on the prefill path, and the output-side
             `C_gate * conv_out` reads a transposed view. Both are uncoalesced:
             measured together they move ~8.8 GB in 10.3 ms on long prefill,
             i.e. ~0.83 TB/s against ~4.8 TB/s of HBM — **17 % of peak**.
             `lf_triton_shortconv.py` folds the chunk, the gating multiply and
             the transpose into one tiled Triton kernel on each side of the
             conv. Isolated at T=16000: 5.7x (input) and 4.2x (output),
             ~980 -> ~3350 GB/s, saving 7.8 ms per forward. Bit-exact.
             **Shape-guarded**: below T~1536 the fused kernel is *slower*
             (launch overhead dominates and the tile is mostly masked), so the
             stock path is kept there. Decode never transposes at all, so this
             component is prefill-only by construction.

Activation (default is fully off — the stock code path is untouched):

    LFM_FUSION_PATCH=norm,scale,conv   python -m sglang.launch_server ...

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


# ---------------------------------------------------------------------------
# G3 — fuse chunk + gating multiply + transpose on both sides of the conv
# ---------------------------------------------------------------------------
# Below this token count the fused kernel loses to the stock elementwise ops:
# the work per launch is too small to amortise the tile setup, and a 64x64 tile
# is mostly masked. Measured crossover is between T=1024 (0.94x) and T=2048
# (1.2x); 2048 is chosen so the guard only fires where the win is unambiguous.
CONV_FUSION_MIN_TOKENS = int(os.environ.get("LFM_FUSION_CONV_MIN_TOKENS", "2048"))

_APPLIED_SET: set[str] = set()


def _cached_int32_indices(forward_batch, req_pool_indices):
    """`req_pool_indices.to(torch.int32)` once per forward, not once per layer.

    Every one of the 18 conv layers re-casts the same tensor. The cast moves
    12 bytes, so it is pure launch overhead — but 18 launches per forward is
    ~1.3 % of low-batch decode kernel time, which is the same order as the gaps
    this study set out to close. Cached on the ForwardBatch, keyed by the
    identity of the source tensor so a stale cache can never be returned.
    """
    if "idx" not in _APPLIED_SET:
        return req_pool_indices.to(torch.int32)
    cached = getattr(forward_batch, "_lfm_int32_idx", None)
    if cached is not None and cached[0] is req_pool_indices:
        return cached[1]
    out = req_pool_indices.to(torch.int32)
    try:
        forward_batch._lfm_int32_idx = (req_pool_indices, out)
    except AttributeError:      # slotted ForwardBatch — fall back silently
        pass
    return out


# ---------------------------------------------------------------------------
# G4 — reuse the shipped fused QK-norm + RoPE CUDA kernel
# ---------------------------------------------------------------------------
# `sgl_kernel.fused_qk_norm_rope` combines both head-wise RMSNorms and RoPE into
# one in-place kernel over the packed QKV tensor. Qwen3-MoE already calls it;
# LFM2.5 instead splits QKV, reshapes, runs two separate RMSNorm kernels and
# then a separate RoPE. Same pattern as G1: the fused primitive exists and the
# call site does not use it.
def _patched_attention_forward(self, positions, hidden_states, forward_batch):
    from sgl_kernel import fused_qk_norm_rope

    T = hidden_states.shape[0]
    qkv, _ = self.qkv_proj(hidden_states)

    q_size = self.num_local_q_heads * self.head_dim
    kv_size = self.num_local_kv_heads * self.head_dim

    if qkv.dtype == torch.bfloat16 and self.head_dim == 64:
        pos = positions.view(-1).to(dtype=torch.int32,
                                    device=qkv.device).contiguous()
        # LFM2.5 declares rope_type "default" with no rope_scaling, so the yarn
        # parameters degenerate to the identity (factor 1, no ramp).
        fused_qk_norm_rope(
            qkv, self.num_local_q_heads, self.num_local_kv_heads,
            self.num_local_kv_heads, self.head_dim,
            self.q_layernorm.variance_epsilon,
            self.q_layernorm.weight, self.k_layernorm.weight,
            self._lfm_rope_theta, self.rotary_emb.is_neox_style, pos,
            1.0, 0, 0, 1.0)
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        attn_out = self.attn(q, k, v, forward_batch)
        out, _ = self.out_proj(attn_out)
        return out

    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
    q = q.reshape(T, self.num_local_q_heads, self.head_dim)
    k = k.reshape(T, self.num_local_kv_heads, self.head_dim)
    q = self.q_layernorm(q.reshape(-1, self.head_dim)).reshape(
        T, self.num_local_q_heads, self.head_dim)
    k = self.k_layernorm(k.reshape(-1, self.head_dim)).reshape(
        T, self.num_local_kv_heads, self.head_dim)
    q, k = self.rotary_emb(positions, q, k)
    attn_out = self.attn(q.reshape(T, -1), k.reshape(T, -1), v, forward_batch)
    out, _ = self.out_proj(attn_out)
    return out


def _patched_shortconv_forward(self, hidden_states: torch.Tensor,
                               forward_batch) -> torch.Tensor:
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn, causal_conv1d_update)

    import lf_triton_shortconv as K

    if forward_batch.forward_mode.is_idle():
        return hidden_states

    layer_cache = forward_batch.req_to_token_pool.mamba2_layer_cache(self.layer_idx)
    conv_state = layer_cache.conv[0]
    req_pool_indices = forward_batch.req_pool_indices

    proj, _ = self.in_proj(hidden_states)
    H = self.hidden_size_per_partition

    if forward_batch.forward_mode.is_decode():
        # Decode consumes [T, H] directly — there is no transpose to absorb, and
        # T is the batch size, so the tiled kernel would only add overhead.
        # The gate multiply is still worth replacing: `B_gate * x` reads two
        # *strided rows* of `proj`, which defeats TensorIterator's vectorisation
        # and leaves it at roughly half the bandwidth of an equivalent
        # contiguous multiply. One Triton kernel reading `proj` directly avoids
        # that without changing the number of launches.
        if "gate" in _APPLIED_SET:
            Bx = K.fused_gate_mul(proj, H)
            C_gate = proj[:, H:2 * H]
        else:
            B_gate, C_gate, x = proj.chunk(3, dim=-1)
            Bx = B_gate * x
        conv_out = causal_conv1d_update(
            Bx, conv_state, self.conv_weight, self.conv_bias, activation=None,
            conv_state_indices=_cached_int32_indices(forward_batch,
                                                     req_pool_indices))
        output, _ = self.out_proj(C_gate * conv_out)
        return output

    T = hidden_states.shape[0]
    if "conv" not in _APPLIED_SET or T < CONV_FUSION_MIN_TOKENS:
        return _stock_shortconv_prefill(self, hidden_states, forward_batch,
                                        proj, causal_conv1d_fn)

    # one kernel: read proj, multiply the gate, write [H, T] coalesced
    Bx_t = K.fused_gate_transpose(proj, H)

    extend_start_loc = forward_batch.extend_start_loc
    if extend_start_loc is not None and len(extend_start_loc) > 1:
        query_start_loc = extend_start_loc.new_empty(len(extend_start_loc) + 1)
        query_start_loc[:-1] = extend_start_loc
        query_start_loc[-1] = T
        cache_indices = _cached_int32_indices(forward_batch, req_pool_indices)
    else:
        query_start_loc = hidden_states.new_tensor([0, T], dtype=torch.int32)
        cache_indices = req_pool_indices[:1].to(torch.int32)

    conv_out_ht = causal_conv1d_fn(
        Bx_t, self.conv_weight, self.conv_bias,
        query_start_loc=query_start_loc, cache_indices=cache_indices,
        has_initial_state=None, conv_states=conv_state, activation=None)

    # one kernel: tiled transpose of conv_out, multiply by C_gate, write [T, H]
    gated = K.fused_transpose_gate(conv_out_ht, proj, H)
    output, _ = self.out_proj(gated)
    return output


def _stock_shortconv_prefill(self, hidden_states, forward_batch, proj,
                             causal_conv1d_fn):
    """Unfused prefill path, used below the shape guard."""
    layer_cache = forward_batch.req_to_token_pool.mamba2_layer_cache(self.layer_idx)
    conv_state = layer_cache.conv[0]
    req_pool_indices = forward_batch.req_pool_indices

    B_gate, C_gate, x = proj.chunk(3, dim=-1)
    Bx = B_gate * x
    T = hidden_states.shape[0]
    Bx_t = Bx.transpose(0, 1).contiguous()

    extend_start_loc = forward_batch.extend_start_loc
    if extend_start_loc is not None and len(extend_start_loc) > 1:
        query_start_loc = extend_start_loc.new_empty(len(extend_start_loc) + 1)
        query_start_loc[:-1] = extend_start_loc
        query_start_loc[-1] = T
        cache_indices = req_pool_indices.to(torch.int32)
    else:
        query_start_loc = hidden_states.new_tensor([0, T], dtype=torch.int32)
        cache_indices = req_pool_indices[:1].to(torch.int32)

    conv_out = causal_conv1d_fn(
        Bx_t, self.conv_weight, self.conv_bias,
        query_start_loc=query_start_loc, cache_indices=cache_indices,
        has_initial_state=None, conv_states=conv_state,
        activation=None).transpose(0, 1)

    output, _ = self.out_proj(C_gate * conv_out)
    return output


def apply() -> list[str]:
    want = _enabled()
    if not want:
        return []
    from sglang.srt.models import lfm2_moe as M

    known = {"norm", "scale", "conv", "gate", "idx", "qkrope"}
    unknown = want - known
    if unknown:
        raise ValueError(f"unknown LFM_FUSION_PATCH components: {sorted(unknown)}")
    _APPLIED_SET.update(want)

    if "norm" in want:
        M.Lfm2MoeDecoderLayer.forward = _patched_layer_forward
        M.Lfm2MoeModel.forward = _patched_model_forward
        _APPLIED.append("norm")
    if "scale" in want:
        M.Lfm2MoeSparseMoeBlock.forward = _patched_moe_forward
        _APPLIED.append("scale")
    # `gate` and `idx` both live inside the patched ShortConv, so that method
    # has to be installed whenever any of the three is requested.
    if want & {"conv", "gate", "idx"}:
        M.Lfm2MoeShortConv.forward = _patched_shortconv_forward
        if "conv" in want:
            _APPLIED.append(f"conv(min_tokens={CONV_FUSION_MIN_TOKENS})")
        if "gate" in want:
            _APPLIED.append("gate")
        if "idx" in want:
            _APPLIED.append("idx")
    if "qkrope" in want:
        _install_rope_theta(M)
        M.Lfm2MoeAttention.forward = _patched_attention_forward
        _APPLIED.append("qkrope")

    print(f"[lfm_fusion_patch] applied: {_APPLIED}", flush=True)
    return list(_APPLIED)


def _install_rope_theta(M):
    """Cache rope theta on the attention module at construction time."""
    orig_init = M.Lfm2MoeAttention.__init__

    def patched_init(self, config, layer_id, quant_config=None, prefix=""):
        orig_init(self, config, layer_id, quant_config=quant_config,
                  prefix=prefix)
        rp = getattr(config, "rope_parameters", None)
        if rp is not None and "rope_theta" in rp:
            self._lfm_rope_theta = float(rp["rope_theta"])
        else:
            self._lfm_rope_theta = float(getattr(config, "rope_theta", 1000000.0))

    M.Lfm2MoeAttention.__init__ = patched_init
