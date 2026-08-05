"""Opt-in fusion patches for SGLang's OLMo-2 implementation.

Two gaps, both found by the operator audit and both confirmed against the
Qwen3-30B control, which shows zero calls in every one of these buckets.

O1 `qknorm` — `Olmo2Attention._apply_qk_norm` (models/olmo2.py:164-191) takes a
             fused path only when `get_is_capture_mode()` is true, and calls
             `forward_native` otherwise. `forward_native` decomposes one RMSNorm
             into roughly seven eager kernels (cast, pow, mean, rsqrt, mul,
             cast, mul), twice per layer.

             The decisive detail is that CUDA graphs only capture *decode*. So
             on the decode path the fast branch is taken and the gap is
             invisible, while **every prefill pays the eager decomposition** --
             20.2 %, 21.6 % and 29.2 % of prefill kernel time on the three
             regimes, measured with CUDA graphs enabled, i.e. exactly what a
             real deployment pays. An audit that only looks at decode concludes
             this gap is already handled. It is not.

             The fix is to call the normal `self.q_norm(...)` dispatch, which
             reaches the fused RMSNorm kernel, rather than reaching past it to
             `forward_native`. The alt-stream overlap of the capture path is
             kept where it applies.

O2 `normadd` — the decoder layer (models/olmo2.py:302-319) runs
             `norm(x)` and then `x + residual` as two separate kernels, twice
             per layer. sglang's `fused_add_rmsnorm` cannot absorb this: it
             computes `norm(x + residual)`, and OLMo-2 is norm-after, so the
             residual must be added *after* normalisation. Different function,
             so this needs a real kernel -- `ol_triton_normadd.py`, verified
             bit-identical to the stock pair.

             Worth 1.8-2.2 % of prefill and 2.2-2.6 % of decode.

Activation (default fully off — the stock path is untouched):

    OLMO2_FUSION_PATCH=qknorm,normadd  python -m sglang.launch_server ...

Import happens through `sitecustomize.py` in `ol_inject/`, put on PYTHONPATH.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Tuple

import torch

_APPLIED_SET: set[str] = set()


def _enabled() -> set[str]:
    raw = os.environ.get("OLMO2_FUSION_PATCH", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


# --- O1: stop reaching past the dispatch to forward_native -------------------
def _patched_apply_qk_norm(
    self, q: torch.Tensor, k: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.distributed import (
        split_tensor_along_last_dim,
        tensor_model_parallel_all_gather,
    )
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if self.tp_size > 1:
        q = tensor_model_parallel_all_gather(q.contiguous())
        k = tensor_model_parallel_all_gather(k.contiguous())

    if self.alt_stream is not None and get_is_capture_mode():
        # unchanged: overlap the two norms on a second stream during capture
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        q_shape, k_shape = q.shape, k.shape
        q_by_last = self.q_norm(q.reshape(-1, q_shape[-1]))
        with torch.cuda.stream(self.alt_stream):
            k_by_last = self.k_norm(k.reshape(-1, k_shape[-1]))
        current_stream.wait_stream(self.alt_stream)
        q, k = q_by_last.view(q_shape), k_by_last.view(k_shape)
    else:
        # was: self.q_norm.forward_native(q) -- seven eager kernels per call.
        # The normal dispatch reaches the fused RMSNorm kernel instead. The
        # reshape mirrors the capture branch: RMSNorm normalises the last
        # dimension, and these tensors are already [tokens, heads*head_dim].
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(q.reshape(-1, q_shape[-1])).view(q_shape)
        k = self.k_norm(k.reshape(-1, k_shape[-1])).view(k_shape)

    if self.tp_size > 1:
        splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
        q = splitter(q)[self.tp_rank]
        k = splitter(k)[self.tp_rank]
    return q, k


# --- O2: fuse the post-norm residual add -------------------------------------
def _patched_layer_forward(self, positions, hidden_states, forward_batch):
    import ol_triton_normadd as K

    # Attention block
    residual = hidden_states
    hidden_states = self.self_attn(positions, hidden_states, forward_batch)
    hidden_states = K.rmsnorm_then_add(
        hidden_states, residual,
        self.post_attention_layernorm.weight.data,
        self.post_attention_layernorm.variance_epsilon,
    )

    # MLP block
    residual = hidden_states
    hidden_states = self.mlp(hidden_states)
    hidden_states = K.rmsnorm_then_add(
        hidden_states, residual,
        self.post_feedforward_layernorm.weight.data,
        self.post_feedforward_layernorm.variance_epsilon,
    )
    return hidden_states


def apply(module) -> None:
    """Patch the already-imported `sglang.srt.models.olmo2` module."""
    want = _enabled()
    if not want:
        return
    applied = []

    if "qknorm" in want:
        module.Olmo2Attention._apply_qk_norm = _patched_apply_qk_norm
        applied.append("qknorm")

    if "normadd" in want:
        module.Olmo2DecoderLayer.forward = _patched_layer_forward
        applied.append("normadd")

    _APPLIED_SET.update(applied)
    print(f"[olmo2_fusion_patch] applied: {applied}", flush=True)
