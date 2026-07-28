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
    """Names that look like fused kernels, harvested from the imports."""
    global _KERNEL_LIKE
    pat = re.compile(r"from sgl_kernel import \(([^)]*)\)|from sgl_kernel import ([^\n]*)")
    names = set()
    for f in (src / "python" / "sglang" / "srt").rglob("*.py"):
        for m in pat.finditer(f.read_text(errors="ignore")):
            blob = m.group(1) or m.group(2) or ""
            names |= {x.strip().strip(",") for x in blob.replace("\n", " ").split(",")}
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
    n_model = sum(h["kind"] == "model_missing_primitive" for h in hits)
    print(f"\n  + {n_model} model files never naming fused_qk_norm_rope "
          f"(low precision — many dispatch to it via a helper; audit to confirm)")

    Path(a.out).write_text(json.dumps(
        dict(src=str(src), n_cuda_symbols=len(cuda_syms), hits=hits), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
