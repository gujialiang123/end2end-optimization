"""Patch Qwen2MoeSparseMoeBlock._forward_shared_experts to FUSE the shared-expert gate
(linear + sigmoid + mul) into ONE triton kernel on CUDA — filling sglang's CUDA gap
(sglang only has a CPU/AMX fused_linear_sigmoid_mul; GPU path runs 3 separate kernels).

Install before model load. Enable via env FUSE_GATE=1. Verify via FUSE_GATE_VERIFY=1.
"""
import os
import torch
import triton
import triton.language as tl

_stats = {"fused": 0, "fallback": 0, "max_rel_err": 0.0}


@triton.jit
def _fused_gate(x_ptr, w_ptr, so_ptr, out_ptr, H: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < H
        xv = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(xv * wv, axis=0)
    g = 1.0 / (1.0 + tl.exp(-acc))
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < H
        sov = tl.load(so_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * H + offs, (g * sov).to(tl.bfloat16), mask=mask)


def fused_gate(x, w_gate, shared_out):
    # x:[M,H], w_gate:[1,H] or [H], shared_out:[M,H] -> sigmoid(x@w_gate.T) * shared_out
    M, H = x.shape
    out = torch.empty_like(shared_out)
    bh = 1024 if H % 1024 == 0 else (512 if H % 512 == 0 else 256)
    _fused_gate[(M,)](x.contiguous(), w_gate.reshape(-1).contiguous(),
                      shared_out.contiguous(), out, H, bh, num_warps=8)
    return out


def install():
    from sglang.srt.models import qwen2_moe as m
    import torch.nn.functional as F
    Blk = m.Qwen2MoeSparseMoeBlock
    if getattr(Blk, "_gate_patched", False):
        return
    orig = Blk._forward_shared_experts

    def patched(self, hidden_states):
        if (os.environ.get("FUSE_GATE", "0") == "1"
                and self.shared_expert is not None and self.shared_expert_gate is not None
                and hidden_states.dtype == torch.bfloat16
                and isinstance(self.shared_expert_gate, torch.nn.Linear)
                and self.shared_expert_gate.bias is None):
            shared_output = self.shared_expert(hidden_states)
            out = fused_gate(hidden_states, self.shared_expert_gate.weight, shared_output)
            _stats["fused"] += 1
            if os.environ.get("FUSE_GATE_VERIFY", "0") == "1" and _stats["fused"] <= 20:
                ref = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
                rel = ((ref.float() - out.float()).abs() / (ref.float().abs() + 1e-2)).max().item()
                _stats["max_rel_err"] = max(_stats["max_rel_err"], rel)
            return out
        _stats["fallback"] += 1
        return orig(self, hidden_states)

    Blk._forward_shared_experts = patched
    Blk._gate_patched = True
    print(f"[qwen15_gate_patch] installed (FUSE_GATE={os.environ.get('FUSE_GATE','0')})", flush=True)


def stats():
    return dict(_stats)
