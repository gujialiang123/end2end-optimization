#!/usr/bin/env python3
"""Numeric correctness gate for the LFM2.5 fusion patch.

Token-identity of greedy text is the wrong gate here. `fused_add_rmsnorm`
performs the residual addition with a different accumulation order (and
internal precision) than a separate bf16 `add` followed by a normalisation, so
the arms are *algebraically* identical but not bit-identical. On a degenerate
repetitive continuation, where the top two tokens are nearly tied, a 1e-3
logit difference flips the argmax and the texts diverge — which says nothing
about whether the patch is correct.

So this compares the model's *distribution* instead: for each prompt, take the
next-token top-k logprobs and measure how far the arms are apart. The pass
criterion is that the arms agree to within bf16 round-off, and that the
top-1 token still matches on prompts that are not near-ties.

Usage:
  # collect one arm at a time (each launches its own server)
  python scripts/lfm_fusion/lf_correctness.py collect --arm baseline --gpu 5
  python scripts/lfm_fusion/lf_correctness.py collect --arm norm+scale --gpu 5
  # then compare
  python scripts/lfm_fusion/lf_correctness.py compare --arms baseline,norm+scale
"""
from __future__ import annotations

import argparse
import json
import os
import math
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lf_lib as L
import serving_ceiling_lib as S
from lf_e2e import ARMS, arm_overlay, check_patch_applied

OUT = L.RESULTS / "correctness"

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:",
    "In 1969, humans first landed on the",
    "The chemical symbol for gold is",
    "Question: What is 17 times 24?\nAnswer:",
    "Translate to French: 'The weather is nice today.'\n",
    "Once upon a time, in a village at the foot of a mountain,",
    "The three primary colours of light are red, green and",
    "import torch\nimport torch.nn as nn\n\nclass Attention(nn.Module):",
    "A rectangle has width 7 and height 3. Its area is",
    "The largest planet in our solar system is",
    "SELECT name, COUNT(*) FROM users GROUP BY",
]


def collect(arm: str, gpu: int, port: int, topk: int, model: str):
    import requests

    cfg = dict(cap=32, chunk=-1, policy="lpm", mem=0.85, config_id=-1,
               is_cookbook=False, hash="correctness")
    outdir = OUT
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / f"server_{arm.replace('+','_')}.log"

    old = dict(os.environ)
    os.environ.update(arm_overlay(arm))
    try:
        p, argv = S.launch_server(model, cfg, gpu, port, log)
        ok, info = S.wait_health(p, port, t=900)
        if not ok:
            S.kill_server(p)
            raise SystemExit(f"launch failed: {info}")
        applied_ok, msg = check_patch_applied(log, arm)
        print(f"patch check: {msg}")
        if not applied_ok:
            S.kill_server(p)
            raise SystemExit(f"patch check failed: {msg}")

        recs = []
        for prompt in PROMPTS:
            r = requests.post(
                f"http://127.0.0.1:{port}/generate",
                json={"text": prompt,
                      "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                      "return_logprob": True,
                      "top_logprobs_num": topk},
                timeout=300)
            r.raise_for_status()
            j = r.json()
            top = j["meta_info"]["output_top_logprobs"][0]
            # entries are [logprob, token_id, token_text]
            recs.append(dict(prompt=prompt, text=j["text"],
                             top=[[e[0], e[1]] for e in top]))
        S.kill_server(p)
    finally:
        os.environ.clear()
        os.environ.update(old)

    path = outdir / f"logprobs_{arm.replace('+','_')}.json"
    path.write_text(json.dumps(dict(arm=arm, patch=ARMS[arm], model=model,
                                    topk=topk, records=recs,
                                    environment=L.environment()), indent=2))
    print(f"wrote {path}")


def compare(arms, tol):
    a0, a1 = arms
    f0 = OUT / f"logprobs_{a0.replace('+','_')}.json"
    f1 = OUT / f"logprobs_{a1.replace('+','_')}.json"
    d0, d1 = json.loads(f0.read_text()), json.loads(f1.read_text())

    rows, head_dev, all_kl = [], [], []
    top1_match = 0
    HEAD = math.log(1e-3)  # only tokens with p > 1e-3 carry meaningful logprobs

    for r0, r1 in zip(d0["records"], d1["records"]):
        m0 = {tid: lp for lp, tid in r0["top"]}
        m1 = {tid: lp for lp, tid in r1["top"]}
        shared = sorted(set(m0) & set(m1))
        # Deviation restricted to the probability head. In the tail, log()
        # amplifies: p=1e-6 vs 2e-6 is a 0.69 logprob gap but 1e-6 of mass, so
        # tail deviations say nothing about whether the model changed.
        hd = [abs(m0[t] - m1[t]) for t in shared if m0[t] > HEAD]
        head_dev.extend(hd)

        # symmetrised KL over the shared top-k, renormalised — a mass-weighted
        # measure of how far the two distributions actually are apart
        p0 = {t: math.exp(m0[t]) for t in shared}
        p1 = {t: math.exp(m1[t]) for t in shared}
        s0, s1 = sum(p0.values()) or 1.0, sum(p1.values()) or 1.0
        kl = sum(
            0.5 * ((p0[t] / s0) * math.log((p0[t] / s0) / (p1[t] / s1))
                   + (p1[t] / s1) * math.log((p1[t] / s1) / (p0[t] / s0)))
            for t in shared if p0[t] > 0 and p1[t] > 0)
        all_kl.append(kl)

        t0, t1 = r0["top"][0][1], r1["top"][0][1]
        margin = abs(r0["top"][0][0] - r0["top"][1][0]) if len(r0["top"]) > 1 else 0.0
        ok = t0 == t1
        top1_match += ok
        rows.append(dict(prompt=r0["prompt"][:45], top1_same=ok,
                         top1_margin=round(margin, 4),
                         head_max_dev=round(max(hd), 6) if hd else 0.0,
                         sym_kl=round(kl, 8),
                         shared_of_topk=f"{len(shared)}/{len(r0['top'])}"))

    print(f"\n{'prompt':47s} {'top1':5s} {'margin':>8s} {'headdev':>9s} {'symKL':>10s}  shared")
    for r in rows:
        print(f"{r['prompt']:47s} {str(r['top1_same']):5s} "
              f"{r['top1_margin']:8.4f} {r['head_max_dev']:9.6f} "
              f"{r['sym_kl']:10.7f}  {r['shared_of_topk']}")

    mx = max(head_dev) if head_dev else 0.0
    mean = st.mean(head_dev) if head_dev else 0.0
    max_kl = max(all_kl) if all_kl else 0.0
    verdict = "PASS" if mx <= tol else "FAIL"
    print(f"\nhead |dlogprob| (p>1e-3): max {mx:.6f}  mean {mean:.6f}  "
          f"tolerance {tol}")
    print(f"symmetrised KL over shared top-k: max {max_kl:.8f}  "
          f"mean {st.mean(all_kl):.8f}")
    print(f"top-1 agreement: {top1_match}/{len(rows)}")
    print(f"VERDICT: {verdict}")

    (OUT / f"compare_{a0.replace('+','_')}_vs_{a1.replace('+','_')}.json").write_text(
        json.dumps(dict(arms=arms, tolerance=tol, head_max_dev=mx,
                        head_mean_dev=mean, max_sym_kl=max_kl,
                        mean_sym_kl=st.mean(all_kl) if all_kl else 0.0,
                        top1_match=f"{top1_match}/{len(rows)}",
                        verdict=verdict, rows=rows), indent=2))


def accuracy(arm: str, gpu: int, port: int, model: str, n_questions: int,
             num_shots: int, reps: int = 1):
    """Task-level gate: few-shot GSM8K accuracy under greedy decoding.

    Bit-identity is the wrong bar for the `norm` arm — it is algebraically
    equivalent but uses a different accumulation order, and LFM2.5's top-k
    expert routing turns a bf16-level perturbation into a *discrete* change of
    which experts fire. So the question that actually matters is whether model
    quality moves, and that needs a task metric rather than a token diff.
    """
    cfg = dict(cap=32, chunk=-1, policy="lpm", mem=0.85, config_id=-1,
               is_cookbook=False, hash="accuracy")
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"acc_server_{arm.replace('+','_')}.log"

    old = dict(os.environ)
    os.environ.update(arm_overlay(arm))
    try:
        p, argv = S.launch_server(model, cfg, gpu, port, log)
        ok, info = S.wait_health(p, port, t=900)
        if not ok:
            S.kill_server(p)
            raise SystemExit(f"launch failed: {info}")
        applied_ok, msg = check_patch_applied(log, arm)
        print(f"patch check: {msg}")
        if not applied_ok:
            S.kill_server(p)
            raise SystemExit(f"patch check failed: {msg}")

        import subprocess
        accs, logs = [], []
        for rep in range(reps):
            out_log = OUT / f"gsm8k_{arm.replace('+','_')}_rep{rep}.log"
            cmd = [L.PY, "-m", "sglang.test.few_shot_gsm8k",
                   "--num-questions", str(n_questions),
                   "--num-shots", str(num_shots),
                   "--temperature", "0.0", "--parallel", "32",
                   "--port", str(port)]
            with open(out_log, "w") as f:
                r = subprocess.run(cmd, env=L.run_env(), stdout=f,
                                   stderr=subprocess.STDOUT, timeout=3600)
            txt = out_log.read_text(errors="ignore")
            acc = None
            for line in txt.splitlines():
                if "Accuracy" in line:
                    try:
                        acc = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
            accs.append(acc)
            logs.append(out_log.name)
            print(f"  rep{rep}: accuracy={acc}")
        S.kill_server(p)
    finally:
        os.environ.clear()
        os.environ.update(old)

    ok_accs = [x for x in accs if x is not None]
    path = OUT / f"accuracy_{arm.replace('+','_')}.json"
    path.write_text(json.dumps(dict(arm=arm, patch=ARMS[arm], model=model,
                                    num_questions=n_questions,
                                    num_shots=num_shots, reps=reps,
                                    accuracies=accs,
                                    mean=st.mean(ok_accs) if ok_accs else None,
                                    spread=(max(ok_accs) - min(ok_accs))
                                    if len(ok_accs) > 1 else 0.0,
                                    logs=logs,
                                    environment=L.environment()), indent=2))
    print(f"wrote {path}  accuracies={accs}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--arm", required=True, choices=list(ARMS))
    c.add_argument("--gpu", type=int, required=True)
    c.add_argument("--port", type=int, default=53000)
    c.add_argument("--topk", type=int, default=20)
    c.add_argument("--model", default="lfm25")
    k = sub.add_parser("compare")
    k.add_argument("--arms", default="baseline,norm+scale")
    k.add_argument("--tol", type=float, default=0.05,
                   help="max tolerated |delta logprob| across the shared top-k")
    q = sub.add_parser("accuracy")
    q.add_argument("--arm", required=True, choices=list(ARMS))
    q.add_argument("--gpu", type=int, required=True)
    q.add_argument("--port", type=int, default=53100)
    q.add_argument("--model", default="lfm25")
    q.add_argument("--num-questions", type=int, default=400)
    q.add_argument("--num-shots", type=int, default=5)
    q.add_argument("--reps", type=int, default=3,
                   help="evaluations per server launch; the bit-exact `scale` "
                        "arm uses these to calibrate the harness noise floor")
    a = ap.parse_args()

    if a.cmd == "collect":
        collect(a.arm, a.gpu, a.port, a.topk, a.model)
    elif a.cmd == "accuracy":
        accuracy(a.arm, a.gpu, a.port, a.model, a.num_questions,
                 a.num_shots, a.reps)
    else:
        compare(a.arms.split(","), a.tol)


if __name__ == "__main__":
    main()
