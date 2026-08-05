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

F2 `foldmul`  — the decoder layer scales whole activation tensors by four
             constants from the config, once each per layer
             (models/falcon_h1.py:334-355):

                 self_attention(hidden_states * attention_in_multiplier)
                 attention_hidden_states * attn_out_multiplier
                 mamba(hidden_states * ssm_in_multiplier)
                 mamba_hidden_states * ssm_out_multiplier

             That is 4 full-tensor elementwise multiplies per layer, 96 kernel
             launches on a 24-layer model, and the audit puts them at 4.11 % of
             prefill kernel time once the SSD tiles are fixed.

             Every one of the four sits immediately next to a linear layer, and
             the multipliers are constants, so `(x * a) @ W == x @ (a * W)`.
             Folding each constant into the neighbouring weight **removes the
             kernels entirely** rather than fusing them into something else:

                 attention_in  -> qkv_proj.weight
                 attn_out      -> o_proj.weight
                 ssm_in        -> mamba.in_proj.weight
                 ssm_out       -> mamba.out_proj.weight

             All four projections are bias-free on Falcon-H1 (attention_bias,
             mamba_proj_bias and projectors_bias are all false in the config),
             so no bias term needs the same treatment; the code asserts this
             rather than assuming it.

             The fold happens once, on the first forward, because weights are
             loaded after the module is constructed.

Activation (default fully off):

    FALCON_FUSION_PATCH=convtriton,foldmul  python -m sglang.launch_server ...
"""
from __future__ import annotations

import os

_APPLIED: list[str] = []


def _enabled() -> set[str]:
    raw = os.environ.get("FALCON_FUSION_PATCH", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def _install_foldmul(F) -> None:
    """Fold the four constant multipliers into the neighbouring weights.

    `F` is the already-executed falcon_h1 module handed over by the import
    hook. Re-importing it here would fetch the still-initialising sys.modules
    entry and raise AttributeError, since the import statement that triggered
    this has not returned yet.
    """
    import torch

    orig_forward = F.FalconH1HybridAttentionDecoderLayer.forward

    def _fold_once(self) -> None:
        if getattr(self, "_falcon_mults_folded", False):
            return
        pairs = [
            (self.qkv_proj, self.attention_in_multiplier),
            (self.o_proj, self.attn_out_multiplier),
            (self.mamba.in_proj, self.ssm_in_multiplier),
            (self.mamba.out_proj, self.ssm_out_multiplier),
        ]
        with torch.no_grad():
            for proj, mult in pairs:
                assert getattr(proj, "bias", None) is None, (
                    "a bias would need the same scaling; Falcon-H1 has none, so "
                    "this path was never written for it"
                )
                proj.weight.mul_(mult)
        self._falcon_mults_folded = True

    def patched_forward(self, positions, hidden_states, residual,
                        forward_batch, **kwargs):
        """Original body with the four multiplies removed.

        Setting the multipliers to 1.0 would not help -- `x * 1.0` still
        launches a kernel and still reads and writes the whole tensor. The
        multiplies have to be absent from the code path, which is why this is a
        rewritten forward rather than a constant substitution.
        """
        _fold_once(self)
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            attention_hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,          # was * attention_in_mult
                forward_batch=forward_batch,
            )                                          # was * attn_out_mult

            attn_backend = forward_batch.attn_backend
            mamba_hidden_states = torch.empty_like(hidden_states)
            attn_backend.linear_attn_backend.forward(
                self.mamba,
                hidden_states,                         # was * ssm_in_mult
                mamba_hidden_states,
                layer_id=self.layer_id,
                mup_vector=self.mup_vector,
            )                                          # was * ssm_out_mult

            hidden_states = attention_hidden_states + mamba_hidden_states

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.feed_forward(
            hidden_states, forward_batch, use_reduce_scatter
        )
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual

    F.FalconH1HybridAttentionDecoderLayer.forward = patched_forward


def apply(module) -> None:
    """Patch the already-imported hybrid_linear_attn_backend module."""
    want = _enabled()
    if not want:
        return

    if "foldmul" in want:
        _install_foldmul(module)
        _APPLIED.append("foldmul")

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
