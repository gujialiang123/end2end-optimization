"""Shared helpers for the LFM2.5 FX / dynamo graph-export study.

Everything here is deliberately dependency-light so the scripts can be run on a
single GPU with a single decoder layer's worth of random weights instead of the
full 8B checkpoint.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

# --------------------------------------------------------------------------
# LFM2.5-8B-A1B config constants (from the HF config; see docs/2026-07-27/lfm_fusion_results.md)
# --------------------------------------------------------------------------
H = 2048  # hidden_size
CONV_L = 3  # conv_L_cache -> conv kernel width
CONV_BIAS = False
N_LAYERS = 24
N_EXPERTS = 32
TOP_K = 4
MOE_INTERMEDIATE = 1792
NORM_EPS = 1e-5
ROUTED_SCALING = 1.0
DTYPE = torch.bfloat16


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def outdir(sub: str = "") -> str:
    d = os.path.join(repo_root(), "results", "lfm_fusion", "fx", sub)
    os.makedirs(d, exist_ok=True)
    return d


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def init_sglang_distributed() -> bool:
    """Bring up a 1-rank TP group so sglang's parallel Linear layers construct."""
    try:
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{free_port()}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        return True
    except Exception as e:  # pragma: no cover - reported by the caller
        print(f"[fx_common] distributed init failed: {type(e).__name__}: {e}")
        return False


# --------------------------------------------------------------------------
# Leaf wrappers so torch.fx.symbolic_trace can keep the external CUDA conv op
# as a single opaque node (the real wrapper does `if x.stride(-1) != 1`, which
# is untraceable on a Proxy).
# --------------------------------------------------------------------------
from sglang.srt.layers.attention.mamba.causal_conv1d import (  # noqa: E402
    causal_conv1d_fn,
    causal_conv1d_update,
)


@torch.fx.wrap
def conv1d_fn_leaf(x, weight, bias, query_start_loc, cache_indices, conv_states):
    return causal_conv1d_fn(
        x,
        weight,
        bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=None,
        conv_states=conv_states,
        activation=None,
    )


@torch.fx.wrap
def conv1d_update_leaf(x, conv_state, weight, bias, conv_state_indices):
    return causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=None,
        conv_state_indices=conv_state_indices,
    )


# --------------------------------------------------------------------------
# Faithful op-for-op replicas of the sglang modules.
#
# in_proj  == MergedColumnParallelLinear(H, [H,H,H], bias=False) at TP=1
#             -> a single F.linear with a [3H, H] weight
# out_proj == RowParallelLinear(H, H, bias=False, input_is_parallel=True) at TP=1
#             -> a single F.linear with a [H, H] weight
# (verified against sglang/srt/layers/linear.py: at tp_size==1 both reduce to
#  self.quant_method.apply(self, x, bias) == F.linear(x, layer.weight, bias))
# --------------------------------------------------------------------------
class ShortConvRepro(nn.Module):
    """Op-for-op replica of Lfm2MoeShortConv.forward (sglang lfm2_moe.py:321)."""

    def __init__(self, hidden: int = H, kernel: int = CONV_L, mode: str = "decode"):
        super().__init__()
        self.hidden = hidden
        self.kernel = kernel
        self.mode = mode
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(hidden, kernel))
        self.conv_bias = None

    def forward(
        self,
        hidden_states,
        conv_state,
        req_pool_indices,
        query_start_loc=None,
        cache_indices=None,
    ):
        proj = self.in_proj(hidden_states)
        B_gate, C_gate, x = proj.chunk(3, dim=-1)
        Bx = B_gate * x

        if self.mode == "decode":
            conv_out = conv1d_update_leaf(
                Bx,
                conv_state,
                self.conv_weight,
                self.conv_bias,
                req_pool_indices.to(torch.int32),
            )
        else:
            Bx_t = Bx.transpose(0, 1).contiguous()
            conv_out = conv1d_fn_leaf(
                Bx_t,
                self.conv_weight,
                self.conv_bias,
                query_start_loc,
                cache_indices,
                conv_state,
            ).transpose(0, 1)

        return self.out_proj(C_gate * conv_out)


class MoERepro(nn.Module):
    """Torch-native stand-in for Lfm2MoeSparseMoeBlock.

    Reproduces the *sequence of tensor ops* (router GEMM -> sigmoid scoring ->
    fp32 expert-bias correction -> top-k -> renormalise -> per-expert SwiGLU ->
    weighted sum -> routed_scaling multiply), which is what the graph study
    needs.  It is NOT the production FusedMoE Triton path.
    """

    def __init__(
        self,
        hidden: int = H,
        n_experts: int = N_EXPERTS,
        top_k: int = TOP_K,
        inter: int = MOE_INTERMEDIATE,
        routed_scaling: float = ROUTED_SCALING,
        apply_scaling: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.routed_scaling = routed_scaling
        self.apply_scaling = apply_scaling
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts, dtype=torch.float32))
        self.w13 = nn.Parameter(torch.empty(n_experts, 2 * inter, hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, hidden, inter))

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        scores = torch.sigmoid(router_logits.float())
        scores_for_choice = scores + self.expert_bias
        _, topk_ids = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        topk_w = scores.gather(1, topk_ids)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        topk_w = topk_w.to(hidden_states.dtype)

        # dense-equivalent expert application (shape-faithful, not perf-faithful)
        gate_up = torch.einsum("th,eih->tei", hidden_states, self.w13[topk_ids[0]])
        g, u = gate_up.chunk(2, dim=-1)
        act = torch.nn.functional.silu(g) * u
        per_expert = torch.einsum("tei,ehi->teh", act, self.w2[topk_ids[0]])
        out = (per_expert * topk_w.unsqueeze(-1)).sum(dim=1)
        if self.apply_scaling:
            out = out * self.routed_scaling
        return out


@dataclass
class LayerInputs:
    hidden_states: torch.Tensor
    conv_state: torch.Tensor
    req_pool_indices: torch.Tensor
    query_start_loc: Optional[torch.Tensor]
    cache_indices: Optional[torch.Tensor]


def make_inputs(T: int, mode: str, device="cuda", dtype=DTYPE, n_slots: int = 64):
    hs = torch.randn(T, H, device=device, dtype=dtype) * 0.05
    conv_state = torch.zeros(n_slots, H, CONV_L, device=device, dtype=dtype)
    if mode == "decode":
        req_pool_indices = torch.arange(T, device=device, dtype=torch.int64)
        return LayerInputs(hs, conv_state, req_pool_indices, None, None)
    req_pool_indices = torch.zeros(1, device=device, dtype=torch.int64)
    qsl = torch.tensor([0, T], device=device, dtype=torch.int32)
    cache_indices = torch.zeros(1, device=device, dtype=torch.int32)
    return LayerInputs(hs, conv_state, req_pool_indices, qsl, cache_indices)


def init_weights(mod: nn.Module, dtype=DTYPE, device="cuda"):
    for p in mod.parameters():
        if p.dtype.is_floating_point and p.dim() > 0:
            with torch.no_grad():
                p.normal_(0, 0.02)
    mod.to(device=device, dtype=dtype)
    # expert bias stays fp32 in the real model
    for name, p in mod.named_parameters():
        if name.endswith("expert_bias"):
            p.data = p.data.float()
    return mod
