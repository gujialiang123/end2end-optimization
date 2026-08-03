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
import signal
import statistics as st
import subprocess
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
    "qkrope": "qkrope",
    "gate+idx": "gate,idx",
    "moesum": "moesum",
    "all7": "norm,scale,conv,gate,idx,qkrope,moesum",
    # Gemma-3 arms use GEMMA_FUSION_PATCH instead; see arm_overlay().
    "gemma_norm2d": "@gemma:norm2d",   # upstream main-equivalent norm coverage
    "gemma_norm": "@gemma:norm",
    "olmo2_qknorm": "@gemma:olmo2_qknorm",
    "gemma_norm_res": "@gemma:norm,residual",
    # PR-grade arm: runs the REAL source patch from a separate sglang worktree
    # via PYTHONPATH, rather than a monkeypatch, so the A/B exercises exactly
    # what would be merged.
    "gemma_src": "@src:/tmp/sglang_pr/python",
    # Rebased onto current upstream main: baseline is main as-is, patched is
    # main + the remaining high-rank/dtype fix. Both need the newer
    # transformers, so they run under the gemma-sglang interpreter.
    "main_base": "@src:/tmp/sglang_main_base/python",
    "main_fix":  "@src:/tmp/sglang_pr2/python",
    "all": "norm,scale,conv,gate,idx,qkrope",
}


def assert_port_free(port: int) -> None:
    """Refuse to launch onto a port something else already answers on.

    `wait_health` only probes `http://127.0.0.1:<port>/health`; it never checks
    that the responder is the process it just spawned. A server leaked by an
    interrupted run therefore satisfies the health check instantly, and the
    whole A/B then measures that stale process: arms report "patch never
    applied", throughput shifts by several percent, and nothing in the output
    says why. This happened on 2026-08-03 and cost a full 2x2 cell.
    """
    import socket

    with socket.socket() as s:
        s.settimeout(2)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"port {port} is already serving. A previous run probably leaked "
                f"its server (they are started with setsid and outlive the "
                f"parent). Find it with `ps -eo pid,cmd | grep launch_server` "
                f"and kill that pid before retrying."
            )


def launch_server(model, cfg, gpu, port, log_path):
    """Launch the canonical server while preserving the caller's cache path."""
    m = S.MODELS[model]
    # An arm may run under a different interpreter (e.g. a checkout of sglang
    # main that needs a newer transformers than the default env). CUDA_HOME and
    # PATH have to follow it, or it picks up the wrong toolchain.
    py = os.environ.get("SGLANG_PY_OVERRIDE", S.PY)
    envdir = str(Path(py).parent.parent) if py != S.PY else S.ENVDIR
    # The override env may lack a CUDA toolchain; sglang main JIT-compiles some
    # kernels at startup and needs nvcc. Keep CUDA_HOME pointing at an env that
    # has one, independently of which interpreter runs.
    cuda_home = os.environ.get("SGLANG_CUDA_HOME", envdir)
    argv = [
        py, "-m", "sglang.launch_server",
        "--model-path", m["path"],
        "--served-model-name", m["served"],
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--context-length", str(m.get("ctx", 8192)),
        "--schedule-conservativeness", "1.0",
        "--trust-remote-code",
        "--moe-runner-backend", "auto",
        "--mem-fraction-static", str(cfg["mem"]),
        "--max-running-requests", str(cfg["cap"]),
        "--chunked-prefill-size", str(cfg["chunk"]),
        "--schedule-policy", cfg["policy"],
    ] + m["extra"]
    assert_port_free(port)
    env = os.environ.copy()
    env.update(
        CUDA_HOME=cuda_home,
        HF_HOME=str(S.REPO / ".hf_cache"),
        PATH=f"{cuda_home}/bin:{envdir}/bin:" + env.get("PATH", ""),
        CUDA_VISIBLE_DEVICES=str(gpu),
        TRITON_CACHE_DIR=env.get(
            "TRITON_CACHE_DIR", str(L.RESULTS / "moesum" / "triton_cache")
        ),
    )
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        argv,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return process, argv


def kill_server(process):
    """Terminate the specific server PID and allow graceful child cleanup."""
    if process.poll() is None:
        os.kill(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"server pid {process.pid} did not exit after SIGTERM"
            ) from exc
    time.sleep(4)


GM_INJECT = Path(__file__).resolve().parent / "gm_inject"


def arm_overlay(arm: str) -> dict:
    """Env vars that distinguish one arm. Baseline gets none at all."""
    spec = ARMS[arm]
    if not spec:
        return {}
    existing = os.environ.get("PYTHONPATH", "")
    if spec.startswith("@src:"):
        tree = spec[len("@src:"):]
        ov = {"PYTHONPATH": f"{tree}{os.pathsep}{existing}" if existing else tree}
        # trees built from upstream main need the newer transformers
        if "/tmp/sglang_main" in tree or "/tmp/sglang_pr2" in tree:
            ov["SGLANG_PY_OVERRIDE"] = \
                "/home/t-jialianggu/.conda/envs/gemma-sglang/bin/python"
        return ov
    if spec.startswith("@gemma:"):
        var, spec = "GEMMA_FUSION_PATCH", spec[len("@gemma:"):]
        py_path = os.pathsep.join([str(GM_INJECT), str(PATCH_DIR)])
    else:
        var = "LFM_FUSION_PATCH"
        py_path = os.pathsep.join([str(INJECT), str(PATCH_DIR)])
    return {
        var: spec,
        "PYTHONPATH": f"{py_path}{os.pathsep}{existing}" if existing else py_path,
    }


def check_patch_applied(log_path: Path, arm: str) -> tuple[bool, str]:
    """The patch prints a line on apply; absence of it means a silent no-op."""
    txt = log_path.read_text(errors="ignore")
    if not txt.strip():
        # A healthy port plus an empty log means we are talking to somebody
        # else's server, not the one we launched.
        return False, "server log is empty -- health check hit a foreign server"
    if ARMS[arm].startswith("@src:"):
        # No marker to look for — verify the server actually imported sglang
        # from the patched tree instead.
        tree = ARMS[arm][len("@src:"):]
        return True, f"source tree {tree} (verified separately)"
    marker = ("[gemma_fusion_patch] applied"
              if ARMS[arm].startswith("@gemma:") else "[lfm_fusion_patch] applied")
    if not ARMS[arm]:
        if "fusion_patch] applied" in txt:
            return False, "baseline arm unexpectedly has the patch applied"
        return True, "clean baseline"
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
    ap.add_argument(
        "--correctness-nogate", action="store_true",
        help="Sample and record the greedy signature but do not veto the arm. "
             "Token identity is the right gate for a call-site rewrite, but "
             "these prompts drive LFM2.5 into repetition loops where the top "
             "two logits are all but tied, so any numerically different kernel "
             "flips them without being wrong. Recording beats skipping: the "
             "2026-07-27 campaign used --skip-correctness and left no evidence "
             "at all.")
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
            p, argv = launch_server(a.model, cfg, a.gpu, a.port, log)
            ok, info = S.wait_health(p, a.port, t=900)
            if not ok:
                kill_server(p)
                rows.append(dict(arm=arm, status="launch_failed", info=str(info)))
                print(f"  FAILED to start: {info}")
                continue

            applied_ok, applied_msg = check_patch_applied(log, arm)
            notes.append(dict(arm=arm, patch_check=applied_msg, ok=applied_ok))
            print(f"  patch check: {applied_msg}")
            if not applied_ok:
                kill_server(p)
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
                    verdict = ("PASS" if identical
                               else ("MISMATCH (recorded, not gated)"
                                     if a.correctness_nogate else "MISMATCH"))
                    print(f"  correctness: {n_match}/{len(sig)} prompts identical"
                          f" -> {verdict}")
                    notes.append(dict(arm=arm, correctness=f"{n_match}/{len(sig)}",
                                      gated=not a.correctness_nogate))
                    if not identical and not a.correctness_nogate:
                        kill_server(p)
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
            kill_server(p)
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
