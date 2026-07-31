#!/usr/bin/env python3
"""Which models could call fused_qk_norm_rope but do not?

Scan 1b from the fusion-gap-hunting skill, applied to one specific primitive
across the whole model directory, then filtered by what the kernel actually
supports so the output is a list worth measuring rather than a list of names.

Naming the primitive is not the test -- several models reach it through a helper
-- so this reports what it can see statically and marks each candidate with the
reason it would or would not be eligible. The audit remains the arbiter.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

# What the kernel is compiled for. Anything outside this cannot use it as-is.
SUPPORTED_HEAD_DIMS = (64, 128, 256)


def uses_qk_norm(text: str) -> bool:
    return bool(re.search(r"self\.q_norm\s*=|self\.q_layernorm\s*=", text))


def calls_separately(text: str) -> bool:
    """q_norm and k_norm applied as two separate calls, then a rope call."""
    return bool(re.search(r"self\.q_norm\(", text)
                and re.search(r"self\.k_norm\(", text))


def rope_style(text: str) -> str:
    if "MRotaryEmbedding" in text:
        return "mrope (kernel does not cover)"
    if re.search(r"is_neox_style\s*=\s*False", text):
        return "gptj-interleave (kernel has a variant)"
    return "neox (kernel default)"


def norm_flavour(path: Path, text: str) -> str:
    """Gemma-style (1 + w) needs add_one; plain w does not."""
    if re.search(r"1\.0\s*\+\s*self\.weight|1\s*\+\s*self\.weight", text):
        return "gemma (1+w)"
    if "Gemma" in path.stem.title() or "gemma" in path.stem:
        return "gemma (1+w), inherited"
    return "plain w"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = Path(a.src)
    models = src / "python" / "sglang" / "srt" / "models"
    layers_txt = "\n".join(
        p.read_text(errors="ignore")
        for p in (src / "python" / "sglang" / "srt" / "layers").rglob("*.py"))

    users, candidates = [], []
    for f in sorted(models.glob("*.py")):
        t = f.read_text(errors="ignore")
        if not uses_qk_norm(t):
            continue
        if "fused_qk_norm_rope" in t:
            users.append(f.name)
            continue
        if not calls_separately(t):
            continue
        candidates.append(dict(
            file=f.name,
            rope=rope_style(t),
            norm=norm_flavour(f, t),
            # A model that names its own fused primitive is already handled.
            has_other_fusion=bool(re.search(r"fused_qknorm|qknorm_rope|fused_gemma", t)),
        ))

    print(f"models already calling fused_qk_norm_rope : {len(users)}")
    for u in users:
        print(f"    {u}")
    print(f"\nmodels with separate q_norm/k_norm calls and no fused rope: "
          f"{len(candidates)}\n")
    print(f"  {'file':28s}{'norm':22s}{'rope':32s}{'other fusion'}")
    for c in candidates:
        print(f"  {c['file']:28s}{c['norm']:22s}{c['rope']:32s}"
              f"{'yes' if c['has_other_fusion'] else '-'}")

    actionable = [c for c in candidates
                  if "mrope" not in c["rope"] and not c["has_other_fusion"]]
    print(f"\n{len(actionable)} look actionable without a new kernel variant")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            dict(src=str(src), users=users, candidates=candidates,
                 n_actionable=len(actionable)), indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
