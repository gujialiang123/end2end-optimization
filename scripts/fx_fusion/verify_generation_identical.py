#!/usr/bin/env python3
"""Does the fused path change what the model generates?

Runs the same prompts through two source trees and compares outputs token for
token. Correctness first: a latency number from a model whose generations have
drifted is not a result.

Greedy decoding, so any divergence is a real difference in the argmax rather
than sampling noise. Token identity is the right gate here because Gemma-3 is
dense -- there is no routed expert whose selection could flip discontinuously on
a bf16-level perturbation, which is what makes token identity unusable for MoE.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "Write a haiku about autumn rain.",
    "Explain in one sentence why the sky is blue.",
    "List three prime numbers greater than 50.",
    "def fibonacci(n):",
    "The three laws of thermodynamics are",
    "Translate to French: 'Where is the library?'",
    "Q: What is 17 times 23? A:",
]

RUNNER = r'''
import json, sys, os


def main():
    os.environ.setdefault("SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "1")
    import sglang as sgl

    prompts = json.loads(sys.argv[1])
    model = sys.argv[2]
    out_path = sys.argv[3]

    llm = sgl.Engine(model_path=model, log_level="error",
                     attention_backend="fa3", mem_fraction_static=0.6)
    try:
        outs = llm.generate(prompts, {"temperature": 0.0, "max_new_tokens": 48})
        json.dump([o["text"] for o in outs], open(out_path, "w"))
    finally:
        llm.shutdown()


# sglang spawns its scheduler with multiprocessing, which re-imports this file
# in the child. Without the guard the child builds another Engine and dies, and
# the parent only reports "scheduler died during initialization".
if __name__ == "__main__":
    main()
'''


def run(tree: str, model: str, out: Path, gpu: str) -> list[str]:
    """Run generation in a subprocess with `tree` on PYTHONPATH.

    The whole environment is inherited rather than rebuilt: the JIT kernels need
    CUDA_HOME, PATH and LD_LIBRARY_PATH pointing at the toolchain shipped inside
    the conda env, and reconstructing only a few of those silently produced a
    scheduler that died during init.
    """
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    script = Path("/tmp/_gen_runner.py")
    script.write_text(RUNNER)
    if out.exists():
        out.unlink()
    r = subprocess.run([sys.executable, str(script), json.dumps(PROMPTS), model, str(out)],
                       env=env, capture_output=True, text=True, timeout=3600)
    if not out.exists():
        print(r.stdout[-2500:])
        print(r.stderr[-2500:])
        raise SystemExit(f"generation failed for {tree}")
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-tree", required=True)
    ap.add_argument("--patched-tree", required=True)
    ap.add_argument("--model", default="/data/hf/models/gemma-3-1b-it")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import os
    base = run(a.baseline_tree, a.model, Path("/tmp/_gen_base.json"), a.gpu)
    patch = run(a.patched_tree, a.model, Path("/tmp/_gen_patch.json"), a.gpu)

    same = 0
    rows = []
    for p, b, q in zip(PROMPTS, base, patch):
        ok = b == q
        same += ok
        rows.append(dict(prompt=p, identical=ok, baseline=b, patched=q))
        mark = "same" if ok else "DIFFERS"
        print(f"  [{mark}] {p[:44]}")
        if not ok:
            print(f"      base : {b[:110]!r}")
            print(f"      patch: {q[:110]!r}")

    print(f"\n{same}/{len(PROMPTS)} prompts identical under greedy decoding")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            dict(model=a.model, identical=same, total=len(PROMPTS), rows=rows), indent=1))
        print(f"wrote {a.out}")
    raise SystemExit(0 if same == len(PROMPTS) else 1)


if __name__ == "__main__":
    main()
