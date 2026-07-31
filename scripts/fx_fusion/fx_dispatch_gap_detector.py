#!/usr/bin/env python3
"""Detect dispatch gaps by comparing a module's FX graph across input shapes.

The signal is asymmetry, not the presence of an elementwise chain.

An op that dispatches to a fused kernel appears in the traced graph as an opaque
call. An op that falls through to an eager implementation appears as an expanded
chain of pointwise/reduction ops. So if the *same module* traces to an opaque
call for one input shape and to an expanded chain for another, the second shape
is missing a fusion the framework already has.

This is what distinguishes a real gap from a model that simply has no fused
kernel at all. Scanning a stock HF model finds expanded chains everywhere and
cannot tell you which of them a serving framework fails to fuse; scanning the
framework's own module across shapes can.

Hardware-agnostic: it reads traced graphs, not kernels or timings.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Iterable

import torch
import torch._dynamo as dyn

# Ops that indicate an eager, unfused implementation was traced through.
_EAGER_MARKERS = {
    "pow", "mean", "rsqrt", "sqrt", "var", "sigmoid", "tanh", "erf", "exp",
}
# Namespaces that are part of PyTorch itself. An op outside these is a kernel
# somebody registered, i.e. positive evidence that a fused path was taken.
_STOCK_NAMESPACES = {"aten", "prims", "prim", "profiler", "inductor", "_c10d"}


def _custom_ops(gm) -> list[str]:
    """Names of non-PyTorch registered ops in the graph (fused kernels)."""
    found = []
    for n in gm.graph.nodes:
        t = n.target
        ns = getattr(t, "namespace", None)
        if ns is None:
            overload = getattr(t, "_overloadpacket", None)
            ns = getattr(overload, "namespace", None)
        if ns is not None and ns not in _STOCK_NAMESPACES:
            found.append(str(t))
    return found


def _op_names(gm) -> list[str]:
    out = []
    for n in gm.graph.nodes:
        if n.op not in ("call_function", "call_method"):
            continue
        t = n.target
        name = getattr(t, "__name__", None) or str(t)
        out.append(name.split(".")[-1] if "." in name else name)
    return out


def trace_ops(fn: Callable, *args) -> dict[str, Any]:
    dyn.reset()
    e = dyn.explain(fn)(*args)
    ops: list[str] = []
    custom: list[str] = []
    for g in e.graphs:
        ops.extend(_op_names(g))
        custom.extend(_custom_ops(g))
    eager_hits = sorted({o for o in ops if o in _EAGER_MARKERS})
    # Positive evidence (a registered kernel in the graph) outranks the
    # marker heuristic, which can only ever say "I did not see eager math".
    fused = bool(custom)
    return dict(
        n_graphs=e.graph_count,
        n_breaks=e.graph_break_count,
        ops=ops,
        custom_ops=sorted(set(custom)),
        eager_markers=eager_hits,
        fused=fused,
        looks_expanded=(not fused) and len(eager_hits) >= 2,
    )


def compare_shapes(module: torch.nn.Module, shapes: Iterable[tuple],
                   dtype=torch.bfloat16, device="cuda") -> dict:
    """Trace `module` once per input shape and report which look unfused."""
    results = {}
    for shape in shapes:
        x = torch.randn(*shape, dtype=dtype, device=device)
        r = trace_ops(lambda t: module(t), x)
        r["shape"] = list(shape)
        r["rank"] = len(shape)
        results[str(list(shape))] = r

    expanded = [k for k, v in results.items() if v["looks_expanded"]]
    opaque = [k for k, v in results.items() if v["fused"]]
    return dict(
        results=results,
        expanded_shapes=expanded,
        opaque_shapes=opaque,
        # The gap exists only if the same module reaches a registered kernel for
        # some shapes and expands to eager math for others. All-expanded means
        # there is no fused kernel to miss.
        dispatch_gap=bool(expanded and opaque),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-src", default=None,
                    help="prepend a sglang source tree to sys.path")
    ap.add_argument("--dim", type=int, default=1152)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.sglang_src:
        import sys
        sys.path.insert(0, a.sglang_src)

    from sglang.srt.layers.layernorm import Gemma3RMSNorm

    torch.set_default_device("cuda")
    m = Gemma3RMSNorm(a.dim).cuda()
    m.weight.data.normal_(std=0.1)
    m.weight.data = m.weight.data.to(torch.bfloat16)

    shapes = [(64, a.dim), (1, 64, a.dim), (1, 64, 4, a.dim)]
    rep = compare_shapes(m, shapes)

    print(f"Gemma3RMSNorm(dim={a.dim}) traced at {len(shapes)} input ranks\n")
    for k, v in rep["results"].items():
        if v["fused"]:
            kind = "FUSED  (registered kernel)"
        elif v["looks_expanded"]:
            kind = "EAGER  (expanded math)"
        else:
            kind = "unclear"
        print(f"  rank={v['rank']} shape={k:<18} {kind}")
        if v["custom_ops"]:
            print(f"      kernel: {v['custom_ops']}")
        if v["eager_markers"]:
            print(f"      eager markers: {v['eager_markers']}")

    print()
    if rep["dispatch_gap"]:
        print("DISPATCH GAP FOUND")
        print(f"  fused at:   {rep['opaque_shapes']}")
        print(f"  eager at:   {rep['expanded_shapes']}")
        print("  -> the fused kernel exists but some ranks do not reach it")
    else:
        print("no gap: all shapes trace the same way")

    if a.out:
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
