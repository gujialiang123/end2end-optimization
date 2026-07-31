#!/usr/bin/env python3
"""Static scan for un-called fused primitives. Seconds, no GPU.

Implements the two scans described in `../SKILL.md`:

  1a. ops whose CUDA path falls through to the reference implementation while a
      sibling backend (cpu/hip/npu) calls a fused kernel;
  1b. model files that never name a primitive their peers use.

Deliberately high-recall / low-precision — it produces *candidates*, and the
operator audit is the arbiter. To keep the false-positive rate down it does the
one check that would have rejected the biggest false positive of the 2026-07
run: verifying the primitive is actually importable on the CUDA build rather
than merely present somewhere in the tree. `gelu_quick` looks exactly like the
real Gemma-3 finding until you notice it is imported only under `elif _is_hip`.

Usage:
  python scan_fusion_gaps.py --src /path/to/sglang --out candidates.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

BACKENDS = ("cpu", "hip", "npu", "xpu")


def cuda_available_symbols(python: str) -> set[str]:
    """Symbols the CUDA build of sgl_kernel actually exports."""
    try:
        out = subprocess.run(
            [python, "-c", "import sgl_kernel; print('\\n'.join(dir(sgl_kernel)))"],
            capture_output=True, text=True, timeout=300)
        return {l.strip() for l in out.stdout.splitlines() if l.strip()}
    except Exception:
        return set()


def scan_fallthrough(src: Path, cuda_syms: set[str]) -> list[dict]:
    """1a: forward_cuda that just calls forward_native."""
    layers = src / "python" / "sglang" / "srt" / "layers"
    hits = []
    for f in sorted(layers.rglob("*.py")):
        text = f.read_text(errors="ignore")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "def forward_cuda" not in line:
                continue
            body = "\n".join(lines[i + 1:i + 6])
            if "return self.forward_native" not in body:
                continue

            cls = _enclosing_class(lines, i)
            body_all = _class_body(text, cls) or ""
            sibling_calls = {}
            for b in BACKENDS:
                m = re.search(rf"def forward_{b}\b(.*?)(?=\n    def |\nclass |\Z)",
                              body_all, re.S)
                if not m:
                    continue
                called = _kernel_calls(m.group(1))
                if called:
                    sibling_calls[b] = called

            # A sibling backend doing real work is the signal. Match it to a
            # CUDA symbol by normalised name, because the sibling spells it
            # differently: `torch.ops.sgl_kernel.gemma3_rmsnorm_cpu` and
            # `torch_npu.npu_gemma_rms_norm` are both `gemma_rmsnorm` on CUDA.
            candidates = sorted({c for v in sibling_calls.values() for c in v})
            on_cuda = sorted({m for c in candidates for m in _match_cuda(c, cuda_syms)})
            hits.append(dict(
                kind="cuda_fallthrough",
                file=str(f.relative_to(src)), line=i + 1, cls=cls,
                sibling_fused=sibling_calls,
                primitive_available_on_cuda=on_cuda,
                # this is the whole point: a sibling has it AND cuda ships it
                verdict=("LIKELY REAL" if on_cuda else
                         "probably not actionable (no CUDA-side primitive)"),
            ))
    return hits


_KERNEL_LIKE = set()

# Backend/vendor decorations that do not carry meaning when matching a sibling
# backend's kernel against the CUDA symbol table.
_NOISE = ("torch", "ops", "sgl", "kernel", "npu", "cpu", "hip", "cuda", "aiter",
          "fwd", "forward", "out", "impl", "2d", "v2")


def _callname(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def _terminates(body) -> bool:
    """True if this block always leaves, so following code is the else-branch."""
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise,
                                                ast.Continue, ast.Break))


def _collect_calls(stmts, cond_stack, acc) -> None:
    """Record every call together with the `if` conditions guarding it.

    Early returns are propagated: after `if C: ...; return`, the rest of the
    block really is guarded by `not C`, and that is precisely the region we need
    to inspect. Without this the scanner cannot tell a shape gap from a chain of
    guards that each end in a kernel -- see the Ernie4_5 false positive in WHY.
    """
    stack = list(cond_stack)
    for st in stmts:
        if isinstance(st, ast.If):
            test = ast.unparse(st.test)
            _collect_calls(st.body, stack + [test], acc)
            if st.orelse:
                _collect_calls(st.orelse, stack + [f"not ({test})"], acc)
            elif _terminates(st.body):
                stack = stack + [f"not ({test})"]
        elif isinstance(st, (ast.For, ast.While, ast.With, ast.Try)):
            _collect_calls(st.body, stack, acc)
        else:
            for n in ast.walk(st):
                if isinstance(n, ast.Call):
                    acc.append((_callname(n), list(stack)))


# A guard mentioning one of these constrains the *input tensor*, so it can send
# some inputs down the eager path. `residual is not None` does not qualify --
# that selects an algorithm, not a subset of shapes.
_INPUT_PROPS = (".dim(", ".ndim", ".shape", ".size(", ".dtype",
                ".is_contiguous(", ".stride(", "len(")


def scan_guarded_fallthrough(src: Path) -> list[dict]:
    """1c: forward_cuda that reaches a fused kernel only for *some* inputs.

    Scan 1a catches a `forward_cuda` that hands everything to `forward_native`.
    The more common and much quieter shape is a body that calls a real kernel on
    one branch and falls back on another, gated by a property of the input. Then
    the op is fused in the profile *and* eager in the profile, depending on who
    calls it, so neither the source nor an aggregate profile reads as broken.

    This is the Gemma-3 case on current main: `gemma_rmsnorm` runs for rank-2
    input and every higher rank silently takes the eager path.
    """
    layers = src / "python" / "sglang" / "srt" / "layers"
    hits = []
    for f in sorted(layers.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body
                       if isinstance(n, ast.FunctionDef)
                       and n.name.startswith("forward")
                       and n.name != "forward_native"]:
                acc: list[tuple[str, list[str]]] = []
                _collect_calls(fn.body, [], acc)

                kernels = [(n, g) for n, g in acc
                           if n.split(".")[-1] in _KERNEL_LIKE
                           or n.startswith(("torch.ops.sgl_kernel.", "torch.ops.sgl_kernels."))]
                natives = [(n, g) for n, g in acc if n.endswith("forward_native")]
                if not kernels or not natives:
                    continue

                # Only guards that can route on the input itself matter here.
                guards = sorted({g for _, gs in kernels for g in gs
                                 if any(p in g for p in _INPUT_PROPS)})
                if not guards:
                    continue

                # A guard is a gap only if the region where it is FALSE reaches
                # no kernel of its own. Ernie4_5_VLRotaryEmbedding guards a
                # kernel on `positions.ndim == 2` and then calls a *different*
                # kernel for rank 1, so nothing is missing a fused path.
                gap_guards = []
                for g in guards:
                    neg = f"not ({g})"
                    if any(neg in gs for _, gs in kernels):
                        continue
                    if any(neg in gs for _, gs in natives):
                        gap_guards.append(g)
                if not gap_guards:
                    continue

                hits.append(dict(
                    kind="guarded_fallthrough",
                    file=str(f.relative_to(src)), line=fn.lineno, cls=cls.name,
                    backend=fn.name, 
                    fused_kernels=sorted({n.split(".")[-1] for n, _ in kernels}),
                    input_guards=gap_guards,
                    verdict="CANDIDATE - confirm with a rank/dtype sweep "
                            "(scripts/fx_fusion/fx_dispatch_gap_detector.py)",
                ))
    return hits


def _norm_tokens(name: str) -> set[str]:
    toks = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    # rms_norm and rmsnorm must compare equal
    toks = ["rmsnorm" if t in ("rms", "rmsnorm") else t for t in toks]
    out = set()
    for t in toks:
        if t in _NOISE:
            continue
        out.add(t)
        if t.startswith("gemma"):      # gemma3 -> gemma
            out.add("gemma")
    return out - {"norm"} | ({"rmsnorm"} if "rmsnorm" in toks or
                             ("rms" in toks and "norm" in toks) else set())


def _match_cuda(sibling_symbol: str, cuda_syms: set[str]) -> list[str]:
    """CUDA symbols plausibly equivalent to a sibling backend's kernel."""
    want = _norm_tokens(sibling_symbol)
    if not want:
        return []
    hits = []
    for s in cuda_syms:
        if s.startswith("_"):
            continue
        have = _norm_tokens(s)
        if want and have and want <= have | {"gemma"} and len(want & have) >= 2:
            hits.append(s)
    return sorted(hits)


def _kernel_calls(body: str) -> list[str]:
    """Kernel-ish calls in a backend body, including attribute forms.

    Catches `foo(...)`, `torch.ops.sgl_kernel.foo(...)` and `torch_npu.foo(...)`,
    which is what the first version of this scanner got wrong -- it only looked
    at plain imported identifiers and therefore missed the very case it was
    written from.
    """
    found = set()
    for m in re.finditer(r"([A-Za-z_][\w.]*)\s*\(", body):
        full = m.group(1)
        leaf = full.split(".")[-1]
        if leaf in ("forward_native", "getattr", "print", "super", "range",
                    "len", "int", "float", "bool", "type", "empty", "empty_like"):
            continue
        if full.startswith(("torch.ops.", "torch_npu.", "aiter.")) or leaf in _KERNEL_LIKE:
            found.add(leaf)
        elif re.search(r"norm|gelu|silu|rope|moe|attn|gemm|quick|fused", leaf):
            found.add(leaf)
    return sorted(found)


def _seed_kernel_names(src: Path) -> None:
    """Names that look like fused kernels, harvested from the imports.

    Covers both the `sgl_kernel` extension and the in-tree `sglang.kernels`
    package: restricting this to bare `from sgl_kernel import` saw 63 names
    where the two together see 629, and the missing ones are exactly the Triton
    kernels the newer model files call.
    """
    global _KERNEL_LIKE
    pat = re.compile(
        r"from (?:sgl_kernel[\w.]*|sglang\.kernels[\w.]*) import \(([^)]*)\)"
        r"|from (?:sgl_kernel[\w.]*|sglang\.kernels[\w.]*) import ([^\n]*)")
    names = set()
    for f in (src / "python" / "sglang" / "srt").rglob("*.py"):
        for m in pat.finditer(f.read_text(errors="ignore")):
            blob = m.group(1) or m.group(2) or ""
            names |= {x.strip().strip(",").split(" as ")[0].strip()
                      for x in blob.replace("\n", " ").split(",")}
    _KERNEL_LIKE = {n for n in names if n and n.isidentifier()}


def _enclosing_class(lines, i):
    for j in range(i, -1, -1):
        m = re.match(r"class (\w+)", lines[j])
        if m:
            return m.group(1)
    return "?"


def _class_body(text, cls):
    m = re.search(rf"class {cls}\b.*?(?=\nclass |\Z)", text, re.S)
    return m.group(0) if m else None


def scan_models(src: Path, primitive: str, requires: list[str]) -> list[dict]:
    """1b: model files that use `requires` but never name `primitive`."""
    models = src / "python" / "sglang" / "srt" / "models"
    hits = []
    for f in sorted(models.glob("*.py")):
        t = f.read_text(errors="ignore")
        if all(re.search(r, t) for r in requires) and primitive not in t:
            hits.append(dict(kind="model_missing_primitive",
                             file=str(f.relative_to(src)), primitive=primitive,
                             verdict="CANDIDATE - confirm with an operator audit"))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="fusion_gap_candidates.json")
    a = ap.parse_args()
    src = Path(a.src)

    _seed_kernel_names(src)
    cuda_syms = cuda_available_symbols(a.python)

    hits = scan_fallthrough(src, cuda_syms)
    hits += scan_guarded_fallthrough(src)
    hits += scan_models(src, "fused_qk_norm_rope",
                        [r"q_norm|q_layernorm", r"rotary_emb"])

    real = [h for h in hits if h.get("verdict", "").startswith("LIKELY")]
    print(f"{len(hits)} candidates, {len(real)} likely real\n")
    for h in hits:
        if h["kind"] != "cuda_fallthrough":
            continue
        print(f"  {h['cls']:22s} {h['file']}:{h['line']}")
        print(f"      siblings with a fused kernel : {h['sibling_fused'] or '-'}")
        print(f"      available on the CUDA build  : {h['primitive_available_on_cuda'] or '-'}")
        print(f"      -> {h['verdict']}")

    guarded = [h for h in hits if h["kind"] == "guarded_fallthrough"]
    print(f"\n  {len(guarded)} guarded fall-throughs "
          f"(a fused kernel is called, but only for some inputs):\n")
    for h in guarded:
        print(f"  {h['cls']}.{h['backend']}  {h['file']}:{h['line']}")
        print(f"      kernel : {', '.join(h['fused_kernels'])}")
        for g in h["input_guards"]:
            print(f"      guard  : {g}")

    n_model = sum(h["kind"] == "model_missing_primitive" for h in hits)
    print(f"\n  + {n_model} model files never naming fused_qk_norm_rope "
          f"(low precision — many dispatch to it via a helper; audit to confirm)")

    Path(a.out).write_text(json.dumps(
        dict(src=str(src), n_cuda_symbols=len(cuda_syms), hits=hits), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
