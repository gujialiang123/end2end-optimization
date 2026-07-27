"""Export FX / dynamo / AOT graphs and Inductor output code for the LFM2.5
ShortConv module and a full decoder layer.

Usage:
    python scripts/lfm_fusion/fx_export_graphs.py --outdir results/lfm_fusion/fx
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import traceback

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx_common as C  # noqa: E402


def write(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)
    print(f"  wrote {path}  ({len(text)} bytes)")


# --------------------------------------------------------------------------
# 1. torch.fx.symbolic_trace  (pure python-level trace, conv kept as leaf)
# --------------------------------------------------------------------------
class LeafNormTracer(torch.fx.Tracer):
    """sglang's RMSNorm.forward_cuda does `if x.numel() == 0`, which is untraceable
    on a Proxy; treat it as a leaf module so the layer graph still comes out."""

    def is_leaf_module(self, m, qualname):
        return type(m).__name__ in ("RMSNorm",) or super().is_leaf_module(m, qualname)


def do_symbolic_trace(mod: nn.Module, name: str, out: str):
    print(f"[symbolic_trace] {name}")
    try:
        tracer = LeafNormTracer()
        gm = torch.fx.GraphModule(mod, tracer.trace(mod))
    except Exception:
        write(os.path.join(out, f"symbolic_{name}.ERROR.txt"), traceback.format_exc())
        print(f"  FAILED (see symbolic_{name}.ERROR.txt)")
        return None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gm.graph.print_tabular()
    body = [
        f"# torch.fx.symbolic_trace of {name}",
        "# (causal_conv1d_* wrapped with torch.fx.wrap -> single opaque leaf node)",
        "",
        "==== graph.print_tabular() ====",
        buf.getvalue(),
        "==== print_readable() ====",
        gm.print_readable(print_output=False),
    ]
    write(os.path.join(out, f"symbolic_{name}.txt"), "\n".join(body))
    return gm


# --------------------------------------------------------------------------
# 2. dynamo capture (bytecode level) + AOT (post-decomposition, aten level)
# --------------------------------------------------------------------------
def make_dump_backend(name: str, out: str, lower_to_inductor: bool):
    from torch._functorch.aot_autograd import aot_module_simplified

    state = {"dynamo_graphs": 0, "aot_graphs": 0}

    def dump(gm: torch.fx.GraphModule, tag: str, idx: int):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gm.graph.print_tabular()
        body = [
            f"# {tag} graph #{idx} for {name}",
            f"# nodes={len(gm.graph.nodes)}",
            "",
            "==== graph.print_tabular() ====",
            buf.getvalue(),
            "==== print_readable() ====",
            gm.print_readable(print_output=False),
        ]
        write(os.path.join(out, f"{tag}_{name}_{idx}.txt"), "\n".join(body))

    def backend(gm: torch.fx.GraphModule, example_inputs):
        i = state["dynamo_graphs"]
        state["dynamo_graphs"] += 1
        dump(gm, "dynamo", i)

        def fw_compiler(agm: torch.fx.GraphModule, ainputs):
            j = state["aot_graphs"]
            state["aot_graphs"] += 1
            dump(agm, "aot", j)
            if lower_to_inductor:
                from torch._inductor.compile_fx import compile_fx_inner

                return compile_fx_inner(agm, ainputs)
            return agm.forward

        kwargs = {}
        if lower_to_inductor:
            # inductor's lowering table assumes its own decomps have already run
            from torch._inductor.decomposition import select_decomp_table

            kwargs["decompositions"] = select_decomp_table()
        return aot_module_simplified(
            gm, example_inputs, fw_compiler=fw_compiler, **kwargs
        )

    return backend, state


def do_dynamo(mod, args, name, out, lower_to_inductor):
    print(f"[dynamo{'+inductor' if lower_to_inductor else ''}] {name}")
    torch._dynamo.reset()
    backend, state = make_dump_backend(name, out, lower_to_inductor)
    compiled = torch.compile(mod, backend=backend, fullgraph=False, dynamic=False)
    err = None
    try:
        with torch.inference_mode():
            compiled(*args)
    except Exception:
        err = traceback.format_exc()
        write(os.path.join(out, f"dynamo_{name}.ERROR.txt"), err)
        print("  RUN FAILED (see .ERROR.txt)")
    return state, err


def do_explain(mod, args, name, out):
    print(f"[explain] {name}")
    torch._dynamo.reset()
    try:
        with torch.inference_mode():
            exp = torch._dynamo.explain(mod)(*args)
        txt = str(exp)
    except Exception:
        txt = traceback.format_exc()
    write(os.path.join(out, f"explain_{name}.txt"), txt)
    # graph-break count is the headline number
    for line in txt.splitlines():
        if "Graph Break Count" in line or "Graph Count" in line or "Op Count" in line:
            print("   ", line.strip())
    return txt


# --------------------------------------------------------------------------
# 3. plain inductor compile, capturing the generated triton via the
#    `output_code` / `schedule` / `fusion` artifact loggers
# --------------------------------------------------------------------------
@contextlib.contextmanager
def capture_inductor_logs(path: str):
    import logging

    torch._logging.set_logs(output_code=True, fusion=True, schedule=True)
    handler = logging.FileHandler(path, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    loggers = [logging.getLogger("torch._inductor")]
    for lg in loggers:
        lg.addHandler(handler)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeHandler(handler)
        handler.close()
        torch._logging.set_logs()


def do_plain_inductor(mod, args, name, out):
    print(f"[inductor] {name}")
    torch._dynamo.reset()
    logpath = os.path.join(out, f"inductor_{name}.log")
    ok = True
    with capture_inductor_logs(logpath):
        compiled = torch.compile(mod, backend="inductor", dynamic=False)
        try:
            with torch.inference_mode():
                compiled(*args)
                torch.cuda.synchronize()
        except Exception:
            ok = False
            write(
                os.path.join(out, f"inductor_{name}.ERROR.txt"), traceback.format_exc()
            )
            print("  INDUCTOR RUN FAILED")
    if ok:
        sz = os.path.getsize(logpath)
        n_triton = 0
        with open(logpath) as f:
            for line in f:
                if line.lstrip().startswith("def triton_") or "async_compile.triton(" in line:
                    n_triton += 1
        print(f"  wrote {logpath} ({sz} bytes), triton kernel defs ~ {n_triton}")
    return ok


# --------------------------------------------------------------------------
# decoder layer wrappers (stock vs already-patched residual convention)
# --------------------------------------------------------------------------
class DecoderLayerStock(nn.Module):
    """Verbatim Lfm2MoeDecoderLayer.forward (sglang lfm2_moe.py:433), conv variant."""

    def __init__(self, mode="decode", moe=True):
        super().__init__()
        from sglang.srt.layers.layernorm import RMSNorm

        self.operator_norm = RMSNorm(C.H, eps=C.NORM_EPS)
        self.ffn_norm = RMSNorm(C.H, eps=C.NORM_EPS)
        self.conv = C.ShortConvRepro(mode=mode)
        self.feed_forward = C.MoERepro() if moe else nn.Linear(C.H, C.H, bias=False)

    def forward(self, hidden_states, conv_state, rpi, qsl=None, ci=None):
        residual = hidden_states
        normed = self.operator_norm(hidden_states)
        hidden_states = self.conv(normed, conv_state, rpi, qsl, ci)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, residual


class DecoderLayerPatched(nn.Module):
    """The G1-patched (deferred-residual / fused_add_rmsnorm) convention."""

    def __init__(self, mode="decode", moe=True):
        super().__init__()
        from sglang.srt.layers.layernorm import RMSNorm

        self.operator_norm = RMSNorm(C.H, eps=C.NORM_EPS)
        self.ffn_norm = RMSNorm(C.H, eps=C.NORM_EPS)
        self.conv = C.ShortConvRepro(mode=mode)
        self.feed_forward = (
            C.MoERepro(apply_scaling=False) if moe else nn.Linear(C.H, C.H, bias=False)
        )

    def forward(self, hidden_states, residual, conv_state, rpi, qsl=None, ci=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)
        hidden_states = self.conv(hidden_states, conv_state, rpi, qsl, ci)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--decode-T", type=int, default=8)
    ap.add_argument("--prefill-T", type=int, default=2048)
    args = ap.parse_args()

    out = args.outdir or C.outdir("graphs")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)

    summary = {}

    for mode, T in (("decode", args.decode_T), ("prefill", args.prefill_T)):
        print(f"\n================= ShortConv / {mode} T={T} =================")
        sc = C.init_weights(C.ShortConvRepro(mode=mode))
        inp = C.make_inputs(T, mode)
        call = (
            inp.hidden_states,
            inp.conv_state,
            inp.req_pool_indices,
            inp.query_start_loc,
            inp.cache_indices,
        )
        # eager sanity
        with torch.inference_mode():
            ref = sc(*call)
        print(f"  eager ok, out={tuple(ref.shape)} {ref.dtype}")

        nm = f"shortconv_{mode}"
        do_symbolic_trace(sc, nm, out)
        do_explain(sc, call, nm, out)
        st, err = do_dynamo(sc, call, nm, out, lower_to_inductor=False)
        ok = do_plain_inductor(sc, call, nm, out)
        summary[nm] = {"T": T, **st, "dynamo_error": bool(err), "inductor_ok": ok}

    # ---- full decoder layer ----
    for mode, T in (("decode", args.decode_T), ("prefill", 512)):
        for variant, cls in (("stock", DecoderLayerStock), ("patched", DecoderLayerPatched)):
            name = f"layer_{variant}_{mode}"
            print(f"\n================= {name} T={T} =================")
            layer = C.init_weights(cls(mode=mode, moe=True))
            inp = C.make_inputs(T, mode)
            if variant == "stock":
                call = (
                    inp.hidden_states,
                    inp.conv_state,
                    inp.req_pool_indices,
                    inp.query_start_loc,
                    inp.cache_indices,
                )
            else:
                resid = torch.randn_like(inp.hidden_states) * 0.05
                call = (
                    inp.hidden_states,
                    resid,
                    inp.conv_state,
                    inp.req_pool_indices,
                    inp.query_start_loc,
                    inp.cache_indices,
                )
            try:
                with torch.inference_mode():
                    layer(*call)
                print("  eager ok")
            except Exception:
                write(os.path.join(out, f"eager_{name}.ERROR.txt"), traceback.format_exc())
                print("  EAGER FAILED")
                continue
            do_symbolic_trace(layer, name, out)
            do_explain(layer, call, name, out)
            st, err = do_dynamo(layer, call, name, out, lower_to_inductor=False)
            ok = do_plain_inductor(layer, call, name, out)
            summary[name] = {"T": T, **st, "dynamo_error": bool(err), "inductor_ok": ok}

    write(os.path.join(out, "summary.json"), json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
