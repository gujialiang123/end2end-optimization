"""Opt-in kernel patches for Falcon-H1 (and any mamba2 hybrid).

F1 `convtriton` — the prefill path materialises a transposed copy of the
             activations before the causal conv:

                 x = hidden_states_B_C_p.transpose(0, 1)   # mamba.py:501
                 ...
                 causal_conv1d_fn(x, ...)                  # causal_conv1d.py:60
                     if x.stride(-1) != 1: x = x.contiguous()

             The transpose guarantees the stride test fires, so **every layer
             pays a full copy of the activations**, twice (in and out). The
             profiler attributes 96 direct_copy launches, exactly 4 per layer,
             to aten::contiguous under aten::clone -- 4.3 % of prefill kernel
             time on the stock baseline and **7.7 % once the SSD tiles are
             fixed**, because the copies cost the same while everything around
             them got 39 % faster.

             The CUDA implementation needs contiguity. The Triton one does not:
             it reads x.stride(0) and x.stride(1) directly, asserts only that
             *one* of them is 1, and has an explicit `is_channel_last` branch
             for precisely this layout (causal_conv1d_triton.py:441-449). Two
             other models in the tree -- granitemoehybrid, nemotron_h -- already
             pass use_triton_causal_conv=True.

             So this is not a new kernel, it is a call-site choice that Falcon-H1
             never made. Nothing is invented here.

Activation (default fully off):

    FALCON_FUSION_PATCH=convtriton  python -m sglang.launch_server ...
"""
from __future__ import annotations

import os

_APPLIED: list[str] = []


def _enabled() -> set[str]:
    raw = os.environ.get("FALCON_FUSION_PATCH", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def apply(module) -> None:
    """Patch the already-imported hybrid_linear_attn_backend module."""
    want = _enabled()
    if not want:
        return

    if "convtriton" in want:
        # The flag is threaded from Mamba2AttnBackend.forward into
        # mixer.forward, and Falcon-H1 never sets it; forcing it True is the
        # whole change. Signature copied verbatim from
        # hybrid_linear_attn_backend.py:1120-1141.
        target = module.Mamba2AttnBackend.forward

        def patched(self, mixer, hidden_states, output, layer_id,
                    mup_vector=None, use_triton_causal_conv=False):
            return target(self, mixer, hidden_states, output, layer_id,
                          mup_vector=mup_vector, use_triton_causal_conv=True)

        module.Mamba2AttnBackend.forward = patched
        _APPLIED.append("convtriton")

    print(f"[falcon_fusion_patch] applied: {_APPLIED}", flush=True)
