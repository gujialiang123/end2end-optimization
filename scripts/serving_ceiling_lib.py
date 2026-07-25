#!/usr/bin/env python3
"""Shared library for the 2026-07-24 Qwen/LFM serving-ceiling campaign.

Unified STREAMING client = sglang.bench_serving (--output-details) for ALL six
canonical workloads, so every regime yields per-request TTFT / ITL / E2E and we
compute p50/p95/p99 ourselves from the raw arrays.

Server backend (moe-runner / attention) is left at `auto` and RECORDED from the
launch log; it does not depend on the four serving knobs, so it is constant
across the grid (isolates serving-level tuning per the campaign spec).
"""
from __future__ import annotations
import itertools
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{ENVDIR}/bin/python"
REPO = Path("/home/t-jialianggu/work/EndtoEnd-auto-optimization")
GPU_MEM_TOTAL_MIB = 143771

MODELS = {
    "qwen": dict(
        path="/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
        served="qwen3-30b-a3b", extra=[],
    ),
    "lfm25": dict(
        path="/data/hf/LFM2.5-8B-A1B",
        served="lfm2.5-8b-a1b", extra=["--max-prefill-tokens", "16384"],
    ),
}

# ---- Canonical serving search space (192 unique configs) --------------------
KNOBS = dict(
    cap=[8, 16, 24, 32, 48, 64, 96, 128],           # max_running_requests
    chunk=[-1, 2048, 8192],                          # chunked_prefill_size
    policy=["lpm", "fcfs"],                          # schedule_policy
    mem=[0.75, 0.80, 0.85, 0.90],                    # mem_fraction_static
)
COOKBOOK = dict(cap=32, chunk=-1, policy="lpm", mem=0.85)  # frozen reference

# ---- Six canonical workloads, unified under bench_serving -------------------
# The four synthetic regimes reuse the exact request-shape parameters recovered
# from results/2026-06-25_autotuning/.../regimes_resolved.yaml (prompt_words is
# mapped 1:1 to random-input-len tokens; documented approximation). shared_prefix
# and tool_agent reuse the exact v7 agentic definitions (scripts/run_v7_agentic_bench.py).
SEED = 20260724
WORKLOADS = {
    "R_short_decode": dict(
        args=["--dataset-name", "random-ids", "--random-input-len", "100",
              "--random-output-len", "256", "--random-range-ratio", "1.0",
              "--num-prompts", "8", "--max-concurrency", "1"],
        note="synthetic: in~100 tok, out 256, conc 1, n 8 (v4 R_short_decode)",
    ),
    "R_medium_balanced": dict(
        args=["--dataset-name", "random-ids", "--random-input-len", "800",
              "--random-output-len", "256", "--random-range-ratio", "1.0",
              "--num-prompts", "16", "--max-concurrency", "8"],
        note="synthetic: in~800 tok, out 256, conc 8, n 16 (v4 R_medium_balanced)",
    ),
    "R_long_prefill": dict(
        args=["--dataset-name", "random-ids", "--random-input-len", "4000",
              "--random-output-len", "32", "--random-range-ratio", "1.0",
              "--num-prompts", "4", "--max-concurrency", "4"],
        note="synthetic: in~4000 tok, out 32, conc 4, n 4 (v4 R_long_prefill)",
    ),
    "R_concurrent_decode": dict(
        args=["--dataset-name", "random-ids", "--random-input-len", "200",
              "--random-output-len", "256", "--random-range-ratio", "1.0",
              "--num-prompts", "32", "--max-concurrency", "32"],
        note="synthetic: in~200 tok, out 256, conc 32, n 32 (v4 R_concurrent_decode)",
    ),
    "shared_prefix": dict(
        args=["--dataset-name", "generated-shared-prefix",
              "--gsp-num-groups", "8", "--gsp-prompts-per-group", "16",
              "--gsp-system-prompt-len", "2048", "--gsp-question-len", "128",
              "--gsp-output-len", "256", "--max-concurrency", "64"],
        note="agentic: 8 groups x16, sys 2048 / q 128 / out 256 (v7 shared_prefix)",
    ),
    "tool_agent": dict(
        args=["--dataset-name", "mooncake", "--mooncake-workload", "toolagent",
              "--num-prompts", "200", "--max-concurrency", "64"],
        note="agentic: mooncake toolagent trace, n 200 (v7 toolagent)",
    ),
}

# Unscored warm-up passes per workload, run before the scored repetitions.
# Short workloads are dominated by first-touch effects (Triton JIT, radix-cache
# population): R_long_prefill runs for only ~0.33 s with 4 requests and was
# measured drifting +36.5 % between the first and fifth scored repetition, while
# the long agentic traces (~40 s) drift < 1 %. Warm-up count is therefore scaled
# to the measurement window rather than applied uniformly.
WARMUP_RUNS = {
    "R_short_decode": 1,
    "R_medium_balanced": 2,
    "R_long_prefill": 4,
    "R_concurrent_decode": 2,
    "shared_prefix": 2,
    "tool_agent": 1,
}


def build_configs():
    """Deterministic ordered list of the 192 unique serving configs."""
    cfgs = []
    for i, (cap, chunk, policy, mem) in enumerate(itertools.product(
            KNOBS["cap"], KNOBS["chunk"], KNOBS["policy"], KNOBS["mem"])):
        c = dict(cap=cap, chunk=chunk, policy=policy, mem=mem)
        c["config_id"] = i
        c["hash"] = config_hash(c)
        c["is_cookbook"] = (cap == COOKBOOK["cap"] and chunk == COOKBOOK["chunk"]
                            and policy == COOKBOOK["policy"] and mem == COOKBOOK["mem"])
        cfgs.append(c)
    return cfgs


def config_hash(c):
    return f"cap{c['cap']}_chunk{c['chunk']}_pol{c['policy']}_mem{c['mem']}"


# ---- server lifecycle -------------------------------------------------------
def port_free(p):
    """True only if we can actually BIND the port.

    A connect() probe is not sufficient: a shutting-down server can still hold
    the listening socket while refusing new connections, which later produces
    `[Errno 98] address already in use` in the next server. Binding is the same
    operation the next server performs, so it is the correct test.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", p))
            return True
        except OSError:
            return False


def wait_port_free(p, t=120):
    t0 = time.time()
    while time.time() - t0 < t:
        if port_free(p):
            return True
        time.sleep(2)
    return False


def wait_gpu_free(gpu, need_free_mib=110000, t=240):
    t0 = time.time()
    while time.time() - t0 < t:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits", "-i", str(gpu)],
                           capture_output=True, text=True)
        try:
            used = int(o.stdout.strip().split("\n")[0])
        except Exception:
            used = 0
        if (GPU_MEM_TOTAL_MIB - used) >= need_free_mib:
            return True
        time.sleep(3)
    return False


def launch_server(model, cfg, gpu, port, log_path):
    m = MODELS[model]
    argv = [PY, "-m", "sglang.launch_server", "--model-path", m["path"],
            "--served-model-name", m["served"], "--host", "127.0.0.1",
            "--port", str(port), "--tensor-parallel-size", "1",
            "--context-length", "8192", "--schedule-conservativeness", "1.0",
            "--trust-remote-code", "--moe-runner-backend", "auto",
            "--mem-fraction-static", str(cfg["mem"]),
            "--max-running-requests", str(cfg["cap"]),
            "--chunked-prefill-size", str(cfg["chunk"]),
            "--schedule-policy", cfg["policy"]] + m["extra"]
    env = os.environ.copy()
    env.update(dict(CUDA_HOME=ENVDIR, HF_HOME=str(REPO / ".hf_cache"),
                    PATH=f"{ENVDIR}/bin:" + env.get("PATH", ""),
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    TRITON_CACHE_DIR=f"/tmp/sgl_triton_ceiling_gpu{gpu}"))
    lf = open(log_path, "w")
    p = subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    return p, argv


def wait_health(p, port, t=600):
    t0 = time.time()
    while time.time() - t0 < t:
        if p.poll() is not None:
            return False, f"server exited rc={p.returncode}"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return True, round(time.time() - t0, 1)
        except Exception:
            time.sleep(3)
    return False, "health-timeout"


def kill_server(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=40)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    # let the kernel finish tearing the listening socket down
    time.sleep(4)


def parse_resolved(log_path):
    """Extract resolved server args (backends, cuda graph, knobs) from log."""
    txt = Path(log_path).read_text(errors="ignore")
    out = {}
    for key in ("attention_backend", "moe_runner_backend", "disable_cuda_graph",
                "max_running_requests", "chunked_prefill_size", "schedule_policy",
                "mem_fraction_static"):
        # server_args are printed like key='value' or key=value
        import re
        m = re.search(rf"{key}=([^,\s\)]+)", txt)
        if m:
            out[key] = m.group(1).strip("'\"")
    out["cuda_graph_captured"] = ("Capture cuda graph" in txt
                                  or "CUDA Graph is enabled" in txt
                                  or out.get("disable_cuda_graph") == "False")
    return out


# ---- benchmark client (bench_serving) ---------------------------------------
def run_workload(model, workload, port, out_jsonl, seed=SEED, timeout=1800):
    """Run one bench_serving invocation; return parsed result dict or error."""
    m = MODELS[model]
    argv = [PY, "-m", "sglang.bench_serving", "--backend", "sglang",
            "--host", "127.0.0.1", "--port", str(port),
            "--model", m["path"], "--tokenizer", m["path"],
            "--seed", str(seed), "--output-details",
            "--output-file", str(out_jsonl)] + WORKLOADS[workload]["args"]
    env = os.environ.copy()
    env.update(dict(CUDA_HOME=ENVDIR, HF_HOME=str(REPO / ".hf_cache"),
                    PATH=f"{ENVDIR}/bin:" + env.get("PATH", "")))
    try:
        r = subprocess.run(argv, env=env, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "bench-timeout", ""
    if r.returncode != 0:
        return None, f"bench-rc={r.returncode}", (r.stderr or r.stdout)[-2000:]
    # bench_serving appends the result as the LAST jsonl line
    try:
        lines = [l for l in Path(out_jsonl).read_text().splitlines() if l.strip()]
        res = json.loads(lines[-1])
    except Exception as e:
        return None, f"parse-fail:{e}", (r.stdout or "")[-2000:]
    return res, None, ""


if __name__ == "__main__":
    cfgs = build_configs()
    print(f"{len(cfgs)} configs; cookbook config_id=",
          [c["config_id"] for c in cfgs if c["is_cookbook"]])
    print("workloads:", list(WORKLOADS))
