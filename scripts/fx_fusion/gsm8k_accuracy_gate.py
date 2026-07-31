#!/usr/bin/env python3
"""Does the fused path cost any task accuracy?

Token identity is the wrong gate here. The fused kernel keeps norm and rotate in
fp32 registers where the model rounds to bf16 in between, so the two are not
bit-identical by construction -- 6 of 8 greedy continuations match and the two
that differ are both coherent. That says the arithmetic changed, which we
already knew, and nothing about whether the model got worse.

GSM8K accuracy is a gate the change can actually be held to. Crucially it is run
with a noise-floor arm: the *baseline tree against itself*, twice, under the same
harness. Any difference smaller than that spread is not a result. Without it a
one-point move looks meaningful and usually is not.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUNNER = r'''
import json, os, re, sys


def main():
    os.environ.setdefault("SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "1")
    import sglang as sgl

    data_path, model, out_path, n, seed = sys.argv[1:6]
    n = int(n)

    rows = [json.loads(l) for l in open(data_path)][:n]

    def few_shot(q):
        return ("Question: " + q + "\nAnswer:")

    prompts = [few_shot(r["question"]) for r in rows]
    golds = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]

    llm = sgl.Engine(model_path=model, log_level="error",
                     attention_backend=os.environ.get("SGLANG_AB_BACKEND", "fa3"), mem_fraction_static=0.6,
                     random_seed=int(seed))
    try:
        outs = llm.generate(prompts, {"temperature": 0.0, "max_new_tokens": 256})
    finally:
        llm.shutdown()

    def last_number(t):
        m = re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
        return m[-1].rstrip(".") if m else None

    correct = sum(last_number(o["text"]) == g for o, g in zip(outs, golds))
    json.dump(dict(correct=correct, total=len(rows),
                   acc=correct / len(rows),
                   texts=[o["text"] for o in outs]), open(out_path, "w"))


if __name__ == "__main__":
    main()
'''


def run(tree: str, model: str, data: str, n: int, seed: int,
        out: Path, gpu: str) -> dict:
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    script = Path("/tmp/_gsm_runner.py")
    script.write_text(RUNNER)
    if out.exists():
        out.unlink()
    r = subprocess.run([sys.executable, str(script), data, model, str(out),
                        str(n), str(seed)],
                       env=env, capture_output=True, text=True, timeout=7200)
    if not out.exists():
        print(r.stdout[-2500:])
        print(r.stderr[-2500:])
        raise SystemExit(f"gsm8k run failed for {tree}")
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-tree", required=True)
    ap.add_argument("--patched-tree", required=True)
    ap.add_argument("--model", default="/data/hf/models/gemma-3-1b-it")
    ap.add_argument("--data", default="/tmp/test.jsonl")
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arms = [
        ("baseline", a.baseline_tree, 1),
        ("baseline_seed2", a.baseline_tree, 2),   # the noise floor
        ("fused", a.patched_tree, 1),
    ]
    res = {}
    for name, tree, seed in arms:
        r = run(tree, a.model, a.data, a.n, seed, Path(f"/tmp/_gsm_{name}.json"), a.gpu)
        res[name] = dict(correct=r["correct"], total=r["total"], acc=r["acc"])
        print(f"  {name:16s} {r['correct']:4d}/{r['total']}  = {r['acc'] * 100:5.2f}%")

    noise = abs(res["baseline"]["acc"] - res["baseline_seed2"]["acc"]) * 100
    delta = (res["fused"]["acc"] - res["baseline"]["acc"]) * 100
    print(f"\n  baseline vs itself, different seed : {noise:+.2f} pts")
    print(f"  fused - baseline                   : {delta:+.2f} pts")
    print("\n  Do NOT read the first line as a noise floor. Greedy decoding is")
    print("  deterministic, so reseeding changes nothing and it is 0 by")
    print("  construction -- which would make any difference at all look")
    print("  significant. Both arms answer the same questions, so run")
    print("  gsm8k_paired_test.py for the paired McNemar test that actually")
    print("  decides this.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            dict(model=a.model, n=a.n, arms=res,
                 seed_spread_pts=round(noise, 4),
                 delta_pts=round(delta, 4),
                 verdict="see gsm8k_paired_test.py; seed spread is not a noise floor"),
            indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
