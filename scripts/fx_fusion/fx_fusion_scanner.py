#!/usr/bin/env python3
"""Find fusable elementwise chains in a torch.compile FX graph.

Hardware-agnostic by construction: it reads the post-grad FX graph that
`torch.compile` produces, which is the same IR on any backend that goes through
Dynamo/AOTAutograd. Nothing here inspects CUDA, kernels, or device timings, so
the same pass runs against a CPU, CUDA or a vendor accelerator.

What it looks for
-----------------
Maximal runs of elementwise / reduction ops where every intermediate is consumed
exactly once. Those are precisely the runs a fused kernel can collapse: each op
in the chain otherwise round-trips its output through memory, and a single-user
intermediate is safe to keep in registers because nobody else needs it.

Why single-user matters: if an intermediate has two consumers, fusing it away
means recomputing it or spilling it anyway, so the win evaporates. `num_users`
is carried on every FX node, so this safety condition is free.

Usage
-----
    from fx_fusion_scanner import scan_module
    report = scan_module(model, example_inputs)
"""
from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from typing import Any, Callable, Iterable, Optional

import torch
import torch.fx as fx

# Ops that are cheap per element and memory-bound in a chain: fusing them is
# almost always a win. Deliberately expressed as aten/prims targets rather than
# Python methods, because that is what survives into the post-grad graph.
_POINTWISE = {
    "add", "sub", "mul", "div", "neg", "reciprocal", "rsqrt", "sqrt", "exp",
    "log", "pow", "tanh", "sigmoid", "silu", "gelu", "relu", "erf", "abs",
    "clamp", "clamp_min", "clamp_max", "maximum", "minimum", "where",
    "convert_element_type", "to_copy", "sin", "cos", "sign", "rsub",
}
# Reductions can anchor a chain (RMSNorm's mean, softmax's max/sum) but a chain
# made only of reductions is not interesting.
_REDUCTION = {"mean", "sum", "var", "amax", "amin", "prod"}
# Ops that only relabel memory. They neither cost nor block fusion, so a chain
# may pass straight through them.
_VIEWLIKE = {
    "view", "reshape", "permute", "transpose", "expand", "squeeze",
    "unsqueeze", "slice", "select", "broadcast_in_dim", "_unsafe_view",
    "contiguous", "clone", "detach", "alias",
}


def _op_name(node: fx.Node) -> str:
    """Short name of the aten/prims op a node calls, or '' for non-calls."""
    if node.op != "call_function":
        return ""
    t = node.target
    name = getattr(t, "_opname", None) or getattr(t, "__name__", "") or str(t)
    name = name.split(".")[-1] if "." in name and not name.startswith("_") else name
    for prefix in ("aten.", "prims.", "torch.ops."):
        name = name.replace(prefix, "")
    return name.strip("_") if name.endswith("_") else name


def _classify(node: fx.Node) -> str:
    n = _op_name(node)
    base = n.split(".")[0]
    if base in _POINTWISE:
        return "pointwise"
    if base in _REDUCTION:
        return "reduction"
    if base in _VIEWLIKE:
        return "view"
    return "other"


@dataclasses.dataclass
class Chain:
    """A run of fusable ops."""

    ops: list[str]
    n_pointwise: int
    n_reduction: int
    shapes: list[tuple]
    est_bytes_saved: int
    node_names: list[str]

    def signature(self) -> str:
        return "->".join(self.ops)

    def to_dict(self) -> dict:
        return dict(
            signature=self.signature(),
            length=len(self.ops),
            n_pointwise=self.n_pointwise,
            n_reduction=self.n_reduction,
            shapes=[list(s) for s in self.shapes],
            est_bytes_saved=self.est_bytes_saved,
            nodes=self.node_names,
        )


def _tensor_meta(node: fx.Node) -> Optional[torch.Tensor]:
    v = node.meta.get("val", None)
    return v if isinstance(v, torch.Tensor) else None


def _nbytes(node: fx.Node) -> int:
    t = _tensor_meta(node)
    if t is None:
        return 0
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


def find_chains(graph: fx.Graph, min_len: int = 2) -> list[Chain]:
    """Maximal chains of fusable ops whose intermediates have a single user.

    The estimate of bytes saved counts every intermediate that would otherwise
    be written and re-read: fusing a chain of k ops removes k-1 round trips.
    It is an upper bound -- the backend may already be fusing some of it -- and
    is used only to rank candidates, never as a claim of speedup.
    """
    chains: list[Chain] = []
    visited: set[fx.Node] = set()

    for node in graph.nodes:
        if node in visited or _classify(node) not in ("pointwise", "reduction"):
            continue

        run = [node]
        visited.add(node)
        cur = node
        # Walk forward while the single consumer is also fusable.
        while len(cur.users) == 1:
            nxt = next(iter(cur.users))
            kind = _classify(nxt)
            if kind == "view":
                # A view neither costs nor blocks; step over it.
                if len(nxt.users) != 1:
                    break
                cur = nxt
                visited.add(nxt)
                continue
            if kind not in ("pointwise", "reduction"):
                break
            if nxt in visited:
                break
            run.append(nxt)
            visited.add(nxt)
            cur = nxt

        real = [n for n in run if _classify(n) in ("pointwise", "reduction")]
        if len(real) < min_len:
            continue

        saved = sum(_nbytes(n) for n in real[:-1]) * 2  # write + read
        shapes = []
        for n in (real[0], real[-1]):
            t = _tensor_meta(n)
            shapes.append(tuple(t.shape) if t is not None else ())

        chains.append(
            Chain(
                ops=[_op_name(n) for n in real],
                n_pointwise=sum(1 for n in real if _classify(n) == "pointwise"),
                n_reduction=sum(1 for n in real if _classify(n) == "reduction"),
                shapes=shapes,
                est_bytes_saved=saved,
                node_names=[n.name for n in real],
            )
        )
    return chains


def scan_module(
    mod: torch.nn.Module,
    example_inputs: tuple,
    min_len: int = 2,
) -> dict:
    """Compile `mod` and report fusable chains found in its post-grad graph."""
    captured: dict[str, Any] = {}

    from torch._inductor import config as ind_cfg

    def _capture(g: fx.Graph):
        captured.setdefault("graphs", []).append(g)

    old = getattr(ind_cfg, "post_grad_custom_post_pass", None)
    ind_cfg.post_grad_custom_post_pass = _capture
    try:
        with torch.no_grad():
            torch.compile(mod, dynamic=False)(*example_inputs)
    finally:
        ind_cfg.post_grad_custom_post_pass = old

    all_chains: list[Chain] = []
    for g in captured.get("graphs", []):
        all_chains.extend(find_chains(g, min_len=min_len))

    all_chains.sort(key=lambda c: -c.est_bytes_saved)
    by_sig: dict[str, int] = defaultdict(int)
    for c in all_chains:
        by_sig[c.signature()] += 1

    return dict(
        n_graphs=len(captured.get("graphs", [])),
        n_chains=len(all_chains),
        total_bytes_saved=sum(c.est_bytes_saved for c in all_chains),
        by_signature=dict(sorted(by_sig.items(), key=lambda kv: -kv[1])),
        chains=[c.to_dict() for c in all_chains],
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        class RMSNorm(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(d))

            def forward(self, x):
                o = x.float()
                o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)
                return (o * (1.0 + self.w.float())).type_as(x)

        m = RMSNorm(128)
        rep = scan_module(m, (torch.randn(32, 128, dtype=torch.bfloat16),))
        print(json.dumps({k: v for k, v in rep.items() if k != "chains"}, indent=2))
        for c in rep["chains"]:
            print(f"  {c['signature']}  len={c['length']} saved={c['est_bytes_saved']}B")
