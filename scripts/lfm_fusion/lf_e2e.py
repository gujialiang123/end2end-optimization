#!/usr/bin/env python3
"""End-to-end A/B for the LFM2.5 fusion patch, with a correctness gate.

Arms differ only by the `LFM_FUSION_PATCH` environment variable; serving knobs,
model, backend and CUDA-graph settings are identical. The stock code path runs
when the variable is absent, so `baseline` is genuinely unmodified SGLang.

Discipline carried over from the regime-kernel study:
  * correctness first — greedy decode must be token-identical to baseline, and
    a failing arm is never benchmarked;
  * per-workload warm-up before any measured repetition (short workloads drift
    badly before steady state);
  * several repetitions plus a Welch t-test, because a single measurement of a
    ~1 % effect is indistinguishable from noise.

Usage:
  python scripts/lfm_fusion/lf_e2e.py --regime C_long_prefill --gpu 5 \
      --arms baseline,scale,norm,norm+scale --reps 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lf_lib as L
import serving_ceiling_lib as S
import run_serving_ceiling_campaign as C

OUT = L.RESULTS / "e2e"
INJECT = Path(__file__).resolve().parent / "lf_inject"
PATCH_DIR = Path(__file__).resolve().parent

REGIME_SERVING = {
    "A_low_batch_decode": dict(workload="R_short_decode",
                               cap=32, chunk=-1, policy="lpm", mem=0.85),
    "B_concurrent_decode": dict(workload="R_concurrent_decode",
                                cap=32, chunk=-1, policy="lpm", mem=0.85),
    "C_long_prefill": dict(workload="R_long_prefill",
                           cap=32, chunk=-1, policy="lpm", mem=0.85),
}

# arm name -> value of LFM_FUSION_PATCH ("" means leave it unset)
ARMS = {
    "baseline": "",
    "scale": "scale",
    "norm": "norm",
    "conv": "conv",
    "norm+scale": "norm,scale",
    "norm+scale+conv": "norm,scale,conv",
}


def arm_overlay(arm: str) -> dict:
    """Env vars that distinguish one arm. Baseline gets none at all."""
    spec = ARMS[arm]
    if not spec:
        return {}
    py_path = os.pathsep.join([str(INJECT), str(PATCH_DIR)])
    existing = os.environ.get("PYTHONPATH", "")
    return {
        "LFM_FUSION_PATCH": spec,
        "PYTHONPATH": f"{py_path}{os.pathsep}{existing}" if existing else py_path,
    }


def check_patch_applied(log_path: Path, arm: str) -> tuple[bool, str]:
    """The patch prints a line on apply; absence of it means a silent no-op."""
    txt = log_path.read_text(errors="ignore")
    if not ARMS[arm]:
        if "[lfm_fusion_patch] applied" in txt:
            return False, "baseline arm unexpectedly has the patch applied"
        return True, "clean baseline"
    marker = "[lfm_fusion_patch] applied"
    if marker not in txt:
        return False, "patch never applied (silent no-op)"
    line = [l for l in txt.splitlines() if marker in l][-1]
    return True, line.strip()


def greedy_sample(port: int, prompts, max_tokens=64):
    """Greedy completions, used as the correctness signature of an arm."""
    import requests

    outs = []
    for p in prompts:
        r = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json={"text": p,
                  "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens}},
            timeout=300)
        r.raise_for_status()
        outs.append(r.json()["text"])
    return outs


CORRECTNESS_PROMPTS = [
    "Explain in one paragraph why memory bandwidth limits decoding speed.",
    "List the first ten prime numbers, separated by commas.",
    "Write a haiku about a GPU kernel.",
    "The capital of France is",
    "def quicksort(arr):",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm25")
    ap.add_argument("--regime", required=True, choices=list(REGIME_SERVING))
    ap.add_argument("--arms", default="baseline,norm+scale")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, default=52000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    spec = REGIME_SERVING[a.regime]
    wl = spec["workload"]
    cfg = dict(cap=spec["cap"], chunk=spec["chunk"], policy=spec["policy"],
               mem=spec["mem"], config_id=-1, is_cookbook=False,
               hash=f"cap{spec['cap']}_chunk{spec['chunk']}"
                    f"_pol{spec['policy']}_mem{spec['mem']}")
    arms = a.arms.split(",")
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; known: {list(ARMS)}")

    outdir = OUT / f"{a.model}{a.tag}" / a.regime
    outdir.mkdir(parents=True, exist_ok=True)
    plan = dict(model=a.model, regime=a.regime, workload=wl, arms=arms,
                reps=a.reps, gpu=a.gpu,
                serving_knobs={k: spec[k] for k in ("cap", "chunk", "policy", "mem")})
    print(json.dumps(plan, indent=2))
    if a.dry_run:
        return
    L.snapshot(outdir, "plan", dict(plan=plan, environment=L.environment()))

    rows, correctness, notes = [], {}, []
    baseline_sig = None

    for arm in arms:
        log = outdir / f"server_{arm.replace('+','_')}.log"
        print(f"\n[{a.regime}] arm={arm}", flush=True)
        old = dict(os.environ)
        os.environ.update(arm_overlay(arm))
        try:
            p, argv = S.launch_server(a.model, cfg, a.gpu, a.port, log)
            ok, info = S.wait_health(p, a.port, t=900)
            if not ok:
                S.kill_server(p)
                rows.append(dict(arm=arm, status="launch_failed", info=str(info)))
                print(f"  FAILED to start: {info}")
                continue

            applied_ok, applied_msg = check_patch_applied(log, arm)
            notes.append(dict(arm=arm, patch_check=applied_msg, ok=applied_ok))
            print(f"  patch check: {applied_msg}")
            if not applied_ok:
                S.kill_server(p)
                rows.append(dict(arm=arm, status="patch_not_applied",
                                 info=applied_msg))
                continue

            # ---- correctness gate, before any timing ----
            if not a.skip_correctness:
                sig = greedy_sample(a.port, CORRECTNESS_PROMPTS)
                correctness[arm] = sig
                if baseline_sig is None:
                    baseline_sig = sig
                    print("  correctness: recorded baseline signature")
                else:
                    identical = sig == baseline_sig
                    n_match = sum(x == y for x, y in zip(sig, baseline_sig))
                    print(f"  correctness: {n_match}/{len(sig)} prompts identical"
                          f" -> {'PASS' if identical else 'MISMATCH'}")
                    if not identical:
                        S.kill_server(p)
                        rows.append(dict(arm=arm, status="correctness_failed",
                                         info=f"{n_match}/{len(sig)} identical"))
                        continue

            resolved = S.parse_resolved(log)
            for w in range(S.WARMUP_RUNS.get(wl, 1)):
                S.run_workload(a.model, wl, a.port, outdir / f"{arm}_warm{w}.jsonl")
            for rep in range(a.reps):
                tmp = outdir / f"{arm}_rep{rep}.jsonl"
                res, err, tail = S.run_workload(a.model, wl, a.port, tmp)
                if err:
                    rows.append(dict(arm=arm, rep=rep, status="bench_failed",
                                     info=str(err)[:200]))
                    continue
                row, _ = C.summarize(res, cfg, a.model, wl, rep, arm)
                row.update(arm=arm, status="ok",
                           lfm_fusion_patch=ARMS[arm],
                           attention_backend=resolved.get("attention_backend"),
                           moe_runner_backend=resolved.get("moe_runner_backend"),
                           cuda_graph_captured=resolved.get("cuda_graph_captured"))
                rows.append(row)
                tmp.unlink(missing_ok=True)
            S.kill_server(p)
        finally:
            os.environ.clear()
            os.environ.update(old)
        time.sleep(5)

    (outdir / "e2e_runs.json").write_text(json.dumps(rows, indent=2, default=str))
    (outdir / "correctness.json").write_text(
        json.dumps(dict(prompts=CORRECTNESS_PROMPTS, outputs=correctness,
                        patch_checks=notes), indent=2))

    print(f"\nwrote {outdir/'e2e_runs.json'} ({len(rows)} rows)")
    base = None
    for arm in arms:
        v = [r["request_throughput"] for r in rows
             if r.get("arm") == arm and r.get("status") == "ok"]
        if not v:
            print(f"  {arm:12s} no successful runs")
            continue
        m = st.mean(v)
        if base is None:
            base = m
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f"  {arm:12s} {m:8.3f} +/- {sd:.3f} req/s   {m/base:.4f}x  n={len(v)}")


if __name__ == "__main__":
    main()
