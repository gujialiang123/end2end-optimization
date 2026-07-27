#!/usr/bin/env python3
"""K1: does the best MoE kernel IMPLEMENTATION differ by regime?

Everything measured so far varied the *configuration* (tile shape) of one
kernel. This experiment varies the kernel itself: SGLang ships several MoE
runner backends (triton, triton_kernel, flashinfer_cutlass, cutlass, ...), which
are genuinely different implementations, not the same kernel with different
parameters.

For each regime we launch a server per backend, with serving knobs frozen, and
measure end-to-end. A backend that fails to load or run is recorded as such
rather than being silently dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rk_lib as L
import serving_ceiling_lib as S
import run_serving_ceiling_campaign as C

OUT = L.RESULTS / "backends"

REGIME_WORKLOAD = {
    "A_low_batch_decode": "R_short_decode",
    "B_concurrent_decode": "R_concurrent_decode",
    "C_long_prefill": "R_long_prefill",
}


def launch_with_backend(model, cfg, gpu, port, log_path, backend):
    """Same launch path as the canonical harness, with the backend overridden."""
    m = S.MODELS[model]
    argv = [S.PY, "-m", "sglang.launch_server", "--model-path", m["path"],
            "--served-model-name", m["served"], "--host", "127.0.0.1",
            "--port", str(port), "--tensor-parallel-size", "1",
            "--context-length", "8192", "--schedule-conservativeness", "1.0",
            "--trust-remote-code", "--moe-runner-backend", backend,
            "--mem-fraction-static", str(cfg["mem"]),
            "--max-running-requests", str(cfg["cap"]),
            "--chunked-prefill-size", str(cfg["chunk"]),
            "--schedule-policy", cfg["policy"]] + m["extra"]
    env = L.run_env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lf = open(log_path, "w")
    p = subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    return p, argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm25")
    ap.add_argument("--regime", required=True, choices=list(REGIME_WORKLOAD))
    ap.add_argument("--backends", default="auto,triton,triton_kernel,flashinfer_cutlass")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, default=50000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=-1)
    ap.add_argument("--policy", default="lpm")
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    wl = REGIME_WORKLOAD[a.regime]
    cfg = dict(cap=a.cap, chunk=a.chunk, policy=a.policy, mem=a.mem,
               config_id=-1, is_cookbook=False,
               hash=f"cap{a.cap}_chunk{a.chunk}_pol{a.policy}_mem{a.mem}")
    outdir = OUT / a.model / a.regime
    outdir.mkdir(parents=True, exist_ok=True)
    backends = a.backends.split(",")

    plan = dict(model=a.model, regime=a.regime, workload=wl, backends=backends,
                serving_knobs={k: cfg[k] for k in ("cap", "chunk", "policy", "mem")},
                reps=a.reps, gpu=a.gpu)
    print(json.dumps(plan, indent=2))
    if a.dry_run:
        return
    L.snapshot(outdir, "plan", dict(plan=plan, environment=L.environment()))

    rows, failures = [], []
    for backend in backends:
        log = outdir / f"server_{backend}.log"
        print(f"\n[{a.regime}] backend={backend}", flush=True)
        p, argv = launch_with_backend(a.model, cfg, a.gpu, a.port, log, backend)
        ok, info = S.wait_health(p, a.port, t=700)
        if not ok:
            S.kill_server(p)
            tail = log.read_text(errors="ignore")[-500:]
            failures.append(dict(backend=backend, stage="launch",
                                 info=str(info), tail=tail[-300:]))
            print(f"  FAILED to start: {info}")
            continue
        resolved = S.parse_resolved(log)
        try:
            for w in range(S.WARMUP_RUNS.get(wl, 1)):
                S.run_workload(a.model, wl, a.port, outdir / f"{backend}_warm{w}.jsonl")
            for rep in range(a.reps):
                tmp = outdir / f"{backend}_rep{rep}.jsonl"
                res, err, tail = S.run_workload(a.model, wl, a.port, tmp)
                if err:
                    failures.append(dict(backend=backend, stage=f"bench_rep{rep}",
                                         info=str(err)[:200]))
                    continue
                row, _ = C.summarize(res, cfg, a.model, wl, rep, backend)
                row.update(backend=backend,
                           resolved_moe_backend=resolved.get("moe_runner_backend"),
                           attention_backend=resolved.get("attention_backend"),
                           cuda_graph_captured=resolved.get("cuda_graph_captured"))
                rows.append(row)
                tmp.unlink(missing_ok=True)
        finally:
            S.kill_server(p)
        time.sleep(5)

    (outdir / "backend_runs.json").write_text(json.dumps(rows, indent=2, default=str))
    (outdir / "backend_failures.json").write_text(json.dumps(failures, indent=2))
    print(f"\nwrote {outdir/'backend_runs.json'} ({len(rows)} rows, "
          f"{len(failures)} failures)")
    if rows:
        import statistics as st
        base = None
        for b in backends:
            v = [r["request_throughput"] for r in rows if r["backend"] == b]
            if not v:
                continue
            m = st.mean(v)
            if base is None:
                base = m
            print(f"  {b:22s} {m:8.3f} req/s  {m/base:.4f}x  n={len(v)}")


if __name__ == "__main__":
    main()
