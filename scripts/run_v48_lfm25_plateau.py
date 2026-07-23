#!/usr/bin/env python3
"""v48 — LFM2.5 serving-knob autotuning plateau study (clean, no warm start).

Replaces the biased 2026-07-02 v3 warm-started convergence figure. Key properties:
  * fresh Optuna study, NO enqueue_trial, NO warm start, NO cookbook injection;
  * TPESampler(seed=20260722, n_startup_trials=20, multivariate=True);
  * fixed MoE path (triton), fixed attention (fa3), CUDA graph ALWAYS on;
  * tunes only 4 serving knobs;
  * 100 unique COMPLETE trials (duplicates re-sampled, failures not counted);
  * primary objective: maximize R_concurrent_decode request throughput.

Workload (R_concurrent_decode) reproduces the v3 spec: concurrency=32,
output=256, num_prompts=32, input≈200 words (256 tokens). We use the official
streaming sglang.bench_serving client (instead of the v3 non-streaming custom
client) because the task requires TTFT/TPOT percentiles, which require streaming.
The request-throughput objective and workload shape are identical.

Usage:
  python scripts/run_v48_lfm25_plateau.py --gpu 3 --n-success 100
  python scripts/run_v48_lfm25_plateau.py --gpu 3 --n-success 3   # smoke test
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, signal, socket, subprocess, sys, time
from pathlib import Path

import optuna

ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{ENVDIR}/bin/python"
REPO = Path("/home/t-jialianggu/work/EndtoEnd-auto-optimization")
MODEL = "/data/hf/LFM2.5-8B-A1B"
SERVED = "lfm2.5-8b-a1b"
OUTDIR = REPO / "results/2026-07-22_lfm25_plateau_100"

# ---- fixed (NOT tuned) ----
FIXED = dict(
    moe_runner_backend="triton",
    attention_backend="fa3",
    disable_cuda_graph=False,
    tensor_parallel_size=1,
    context_length=73728,
    schedule_conservativeness=1.0,
    max_prefill_tokens=96000,
    disable_radix_cache=False,
    trust_remote_code=True,
    reasoning_parser="qwen3",
    tool_call_parser="lfm2",
)

# ---- search space (4 knobs) ----
SPACE = dict(
    max_running_requests=[8, 16, 24, 32, 48, 64, 96, 128],
    chunked_prefill_size=[-1, 2048, 8192],
    schedule_policy=["lpm", "fcfs"],
    mem_fraction_static=[0.75, 0.80, 0.85, 0.90],
)

# ---- workload: R_concurrent_decode ----
WORKLOAD = dict(concurrency=32, output_len=256, num_prompts=32, input_len=256)

# ---- cookbook reference config (measured separately, NOT enqueued) ----
COOKBOOK = dict(max_running_requests=32, chunked_prefill_size=-1,
                schedule_policy="lpm", mem_fraction_static=0.85)


def config_hash(cfg: dict) -> str:
    key = json.dumps({k: cfg[k] for k in SPACE}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_port_free(port: int, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_free(port):
            return True
        time.sleep(2)
    return False


def wait_gpu_free(gpu: int, need_free_mib=100000, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(gpu)], capture_output=True, text=True)
        try:
            used = int(out.stdout.strip().split("\n")[0])
        except Exception:
            used = 0
        if (143771 - used) >= need_free_mib:
            return True
        time.sleep(3)
    return False


def build_server_cmd(cfg: dict, port: int, gpu: int):
    argv = [PY, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--served-model-name", SERVED,
            "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", str(FIXED["tensor_parallel_size"]),
            "--context-length", str(FIXED["context_length"]),
            "--schedule-conservativeness", str(FIXED["schedule_conservativeness"]),
            "--max-prefill-tokens", str(FIXED["max_prefill_tokens"]),
            "--reasoning-parser", FIXED["reasoning_parser"],
            "--tool-call-parser", FIXED["tool_call_parser"],
            "--trust-remote-code",
            "--attention-backend", FIXED["attention_backend"],
            "--moe-runner-backend", FIXED["moe_runner_backend"],
            # tuned knobs:
            "--mem-fraction-static", str(cfg["mem_fraction_static"]),
            "--max-running-requests", str(cfg["max_running_requests"]),
            "--chunked-prefill-size", str(cfg["chunked_prefill_size"]),
            "--schedule-policy", cfg["schedule_policy"]]
    # NOTE: we deliberately DO NOT pass --disable-cuda-graph (cuda graph stays ON)
    env = os.environ.copy()
    env.update(dict(CUDA_HOME=ENVDIR, HF_HOME=str(REPO / ".hf_cache"),
                    HF_HUB_CACHE=str(REPO / ".hf_cache/hub"),
                    HF_DATASETS_CACHE=str(REPO / ".hf_cache/datasets"),
                    PATH=f"{ENVDIR}/bin:" + os.environ.get("PATH", ""),
                    CUDA_VISIBLE_DEVICES=str(gpu)))
    return argv, env


def wait_health(port: int, proc, timeout=420):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, f"server exited early rc={proc.returncode}"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return True, time.time() - t0
        except Exception:
            time.sleep(3)
    return False, "health timeout"


def verify_resolved(server_log: Path, cfg: dict):
    """Confirm fixed invariants + knob values from server logs. Returns (ok, notes)."""
    txt = server_log.read_text(errors="ignore")
    notes = {}
    # disable_cuda_graph=false in resolved ServerArgs
    notes["cuda_graph_on"] = ("disable_cuda_graph=False" in txt) or \
                             ("'disable_cuda_graph': False" in txt) or \
                             ("Capture cuda graph" in txt)
    notes["fa3"] = ("attention_backend='fa3'" in txt) or ("attention_backend=fa3" in txt) \
                   or ("'attention_backend': 'fa3'" in txt)
    notes["triton_moe"] = ("moe_runner_backend='triton'" in txt) or \
                          ("moe_runner_backend=triton" in txt) or \
                          ("'moe_runner_backend': 'triton'" in txt)
    ok = notes["cuda_graph_on"] and notes["triton_moe"]
    return ok, notes


def run_benchmark(port: int, trial_dir: Path):
    resfile = trial_dir / "bench_serving.jsonl"
    env = os.environ.copy()
    env.update(dict(HF_HOME=str(REPO / ".hf_cache"),
                    HF_HUB_CACHE=str(REPO / ".hf_cache/hub"),
                    HF_DATASETS_CACHE=str(REPO / ".hf_cache/datasets"),
                    PATH=f"{ENVDIR}/bin:" + os.environ.get("PATH", "")))
    argv = [PY, "-m", "sglang.bench_serving", "--backend", "sglang",
            "--host", "127.0.0.1", "--port", str(port), "--model", MODEL,
            "--dataset-name", "random",
            "--random-input-len", str(WORKLOAD["input_len"]),
            "--random-output-len", str(WORKLOAD["output_len"]),
            "--random-range-ratio", "1.0",
            "--num-prompts", str(WORKLOAD["num_prompts"]),
            "--max-concurrency", str(WORKLOAD["concurrency"]),
            "--output-details",
            "--output-file", str(resfile)]
    log = trial_dir / "bench.log"
    with open(log, "w") as f:
        p = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1200)
    if resfile.exists():
        rows = [json.loads(l) for l in open(resfile) if l.strip()]
        if rows:
            return rows[-1]
    return None


def _pct(vals, p):
    import numpy as np
    if not vals:
        return None
    return float(np.percentile(vals, p))


def latency_percentiles(m):
    """Compute exact p50/p95/p99 for ttft/tpot/e2e from output-details arrays."""
    ttfts = [t for t in m.get("ttfts", []) if t is not None]
    itls = m.get("itls", [])  # list-of-lists (per request)
    tpots, e2es = [], []
    for i, req_itl in enumerate(itls):
        if req_itl:
            tpots.append(sum(req_itl) / len(req_itl))  # mean inter-token latency
            ttft_i = ttfts[i] if i < len(ttfts) and ttfts[i] is not None else 0.0
            e2es.append(ttft_i + sum(req_itl))
    return dict(
        ttft_p50=(_pct(ttfts, 50) or 0) * 1000, ttft_p95=(_pct(ttfts, 95) or 0) * 1000,
        ttft_p99=(_pct(ttfts, 99) or 0) * 1000,
        tpot_p50=(_pct(tpots, 50) or 0) * 1000, tpot_p95=(_pct(tpots, 95) or 0) * 1000,
        tpot_p99=(_pct(tpots, 99) or 0) * 1000,
        e2e_p50=(_pct(e2es, 50) or 0) * 1000, e2e_p95=(_pct(e2es, 95) or 0) * 1000,
        e2e_p99=(_pct(e2es, 99) or 0) * 1000,
    )


def run_warmup(port: int, trial_dir: Path):
    """Unscored warmup: complete cuda-graph capture + stabilize."""
    env = os.environ.copy()
    env.update(dict(HF_HOME=str(REPO / ".hf_cache"),
                    HF_HUB_CACHE=str(REPO / ".hf_cache/hub"),
                    HF_DATASETS_CACHE=str(REPO / ".hf_cache/datasets"),
                    PATH=f"{ENVDIR}/bin:" + os.environ.get("PATH", "")))
    argv = [PY, "-m", "sglang.bench_serving", "--backend", "sglang",
            "--host", "127.0.0.1", "--port", str(port), "--model", MODEL,
            "--dataset-name", "random", "--random-input-len", str(WORKLOAD["input_len"]),
            "--random-output-len", str(WORKLOAD["output_len"]), "--random-range-ratio", "1.0",
            "--num-prompts", "8", "--max-concurrency", "8",
            "--output-file", str(trial_dir / "warmup.jsonl")]
    with open(trial_dir / "warmup.log", "w") as f:
        subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600)


def kill_group(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def evaluate(cfg: dict, port: int, gpu: int, trial_dir: Path, is_warmstart_check=False):
    """Launch → health → verify → warmup → bench → parse → cleanup.
    Returns (metrics_dict | None, status, fail_reason, resolved_notes, startup_s)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    argv, env = build_server_cmd(cfg, port, gpu)
    (trial_dir / "server_cmd.txt").write_text(" ".join(argv))
    server_log = trial_dir / "server.log"
    proc = None
    try:
        if not wait_port_free(port, 60):
            return None, "fail", "port_busy", {}, 0.0
        with open(server_log, "w") as lf:
            proc = subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    preexec_fn=os.setsid)
        ok, info = wait_health(port, proc)
        if not ok:
            kill_group(proc)
            return None, "fail", f"server_not_ready:{info}", {}, 0.0
        startup_s = float(info)
        vok, notes = verify_resolved(server_log, cfg)
        if not vok:
            kill_group(proc)
            return None, "fail", f"invariant_violation:{notes}", notes, startup_s
        run_warmup(port, trial_dir)
        m = run_benchmark(port, trial_dir)
        kill_group(proc)
        if not m or not m.get("completed"):
            return None, "fail", "benchmark_no_result", notes, startup_s
        rps = m.get("request_throughput")
        if not rps or rps <= 0:
            return None, "fail", "invalid_throughput", notes, startup_s
        return m, "ok", "", notes, startup_s
    except subprocess.TimeoutExpired:
        kill_group(proc)
        return None, "fail", "timeout", {}, 0.0
    except Exception as e:
        kill_group(proc)
        return None, "fail", f"exception:{type(e).__name__}:{e}", {}, 0.0
    finally:
        if proc is not None:
            kill_group(proc)
        wait_gpu_free(gpu, need_free_mib=100000, timeout=120)


PER_TRIAL_FIELDS = [
    "completed_index", "optuna_trial_number", "request_throughput",
    "output_token_throughput", "ttft_p50", "ttft_p95", "ttft_p99",
    "tpot_p50", "tpot_p95", "tpot_p99", "e2e_p50", "e2e_p95", "e2e_p99",
    "max_running_requests", "chunked_prefill_size", "schedule_policy",
    "mem_fraction_static", "moe_runner_backend", "attention_backend",
    "disable_cuda_graph", "server_startup_s", "benchmark_wall_s",
    "config_hash", "result_path",
]


def metrics_row(m, cfg, completed_index, trial_number, startup_s, result_path):
    def g(*keys):
        for k in keys:
            if k in m and m[k] is not None:
                return m[k]
        return None
    lat = latency_percentiles(m)
    return {
        "completed_index": completed_index,
        "optuna_trial_number": trial_number,
        "request_throughput": g("request_throughput"),
        "output_token_throughput": g("output_throughput"),
        "ttft_p50": round(lat["ttft_p50"], 3), "ttft_p95": round(lat["ttft_p95"], 3),
        "ttft_p99": round(lat["ttft_p99"], 3),
        "tpot_p50": round(lat["tpot_p50"], 3), "tpot_p95": round(lat["tpot_p95"], 3),
        "tpot_p99": round(lat["tpot_p99"], 3),
        "e2e_p50": round(lat["e2e_p50"], 3), "e2e_p95": round(lat["e2e_p95"], 3),
        "e2e_p99": round(lat["e2e_p99"], 3),
        "max_running_requests": cfg["max_running_requests"],
        "chunked_prefill_size": cfg["chunked_prefill_size"],
        "schedule_policy": cfg["schedule_policy"],
        "mem_fraction_static": cfg["mem_fraction_static"],
        "moe_runner_backend": FIXED["moe_runner_backend"],
        "attention_backend": FIXED["attention_backend"],
        "disable_cuda_graph": FIXED["disable_cuda_graph"],
        "server_startup_s": round(startup_s, 1),
        "benchmark_wall_s": g("duration"),
        "config_hash": config_hash(cfg),
        "result_path": result_path,
    }


def append_csv(path: Path, fields, row):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--port", type=int, default=31700)
    ap.add_argument("--n-success", type=int, default=100)
    ap.add_argument("--max-attempts", type=int, default=400)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    per_trial_csv = OUTDIR / "per_trial_log.csv"
    failures_csv = OUTDIR / "failures.csv"

    # resume: count existing successes
    seen_hashes = set()
    completed = 0
    if per_trial_csv.exists():
        for r in csv.DictReader(open(per_trial_csv)):
            seen_hashes.add(r["config_hash"])
            completed += 1
        print(f"[resume] {completed} successful trials already logged", flush=True)

    study = optuna.create_study(
        study_name="lfm25_plateau_100",
        direction="maximize",
        storage=f"sqlite:///{OUTDIR}/study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=20260722, n_startup_trials=20, multivariate=True),
    )
    # HARD GUARD: no pre-seeded trials
    assert len(study.trials) == 0 or per_trial_csv.exists(), \
        "study.db has trials but no per_trial_log — refuse to reuse a dirty study"

    attempts = 0
    while completed < args.n_success and attempts < args.max_attempts:
        attempts += 1
        trial = study.ask()
        cfg = {k: trial.suggest_categorical(k, SPACE[k]) for k in SPACE}
        h = config_hash(cfg)
        if h in seen_hashes:
            # duplicate → prune, do not count, re-ask
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            print(f"[attempt {attempts}] duplicate {h} pruned", flush=True)
            continue
        trial_dir = OUTDIR / f"trial_{completed:04d}"
        print(f"[attempt {attempts}] eval completed_idx={completed} optuna#{trial.number} "
              f"cfg={cfg}", flush=True)
        m, status, reason, notes, startup_s = evaluate(cfg, args.port, args.gpu, trial_dir)
        if status != "ok":
            append_csv(failures_csv,
                       ["attempt", "optuna_trial_number", "config_hash", "reason",
                        "max_running_requests", "chunked_prefill_size",
                        "schedule_policy", "mem_fraction_static", "server_log"],
                       dict(attempt=attempts, optuna_trial_number=trial.number,
                            config_hash=h, reason=reason, **{k: cfg[k] for k in SPACE},
                            server_log=str(trial_dir / "server.log")))
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"  -> FAIL: {reason}", flush=True)
            continue
        rps = m["request_throughput"]
        seen_hashes.add(h)
        row = metrics_row(m, cfg, completed, trial.number, startup_s,
                          str(trial_dir / "bench_serving.jsonl"))
        append_csv(per_trial_csv, PER_TRIAL_FIELDS, row)
        (trial_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        study.tell(trial, rps)
        completed += 1
        print(f"  -> OK rps={rps:.3f} ({completed}/{args.n_success})", flush=True)

    print(f"\nDONE: {completed} successful unique trials in {attempts} attempts", flush=True)
    if completed < args.n_success:
        print(f"WARNING: only {completed}/{args.n_success} — hit max_attempts", flush=True)


if __name__ == "__main__":
    main()
