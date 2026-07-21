#!/usr/bin/env python
"""v45: server-level END-TO-END A/B — OURS (re-tuned on triton 3.6.0) vs the
FALLBACK config sglang actually loads — across all regimes + the agent dataset.

This is the server/bench_serving counterpart of v44 (which used bench_one_batch),
and the "right leg" of the config-tuning story vs v43 (which compared default
heuristic vs fallback). Here BOTH arms have a real tuned config; we isolate the
marginal value of re-tuning for the current Triton version.

Arms (config placement in the sglang source tree, verified in v44):
  * fallback : stock sglang. triton_3_6_0/ has NO config for E=128,N=768,H200,
               so the loader falls back to triton_3_2_0 (prints "sub-optimal").
  * ours     : our config re-tuned on triton 3.6.0 dropped into triton_3_6_0/ so
               the loader picks it first (version match).

For each (arm, regime) we launch one sglang server and run bench_serving REPEATS
times; we record TTFT / TPOT / E2E latency / output throughput per run to a jsonl
so signal-vs-noise can be judged with a t-test afterwards (analyze_v45).
"""
import os, sys, subprocess, json, time, signal, urllib.request, shutil, argparse

ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{ENVDIR}/bin/python"
REPO = "/home/t-jialianggu/work/end2end-optimization"
SGLANG = "/home/t-jialianggu/work/sglang"
MODEL = "/home/t-jialianggu/work/models/Qwen3-30B-A3B-Instruct-2507"
OUTDIR = f"{REPO}/results/2026-07-21_v45_server_ours_vs_fallback"
PORT = "31577"
GPU = os.environ.get("GPU", "0")

# where sglang looks for triton_3_6_0 configs, and our tuned artifact
CFG_DIR_360 = f"{SGLANG}/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0"
OURS_CFG = f"{REPO}/results/2026-07-20_v44_retune_e2e_ab/E=128,N=768,device_name=NVIDIA_H200.json"
TARGET_CFG = f"{CFG_DIR_360}/E=128,N=768,device_name=NVIDIA_H200.json"

os.makedirs(OUTDIR, exist_ok=True)

# regimes mirror v43 (artificial sweep + mooncake agent), covering all input shapes.
REGIMES = [
    ("tiny_latency",        ["--dataset-name","random","--random-input-len","8","--random-output-len","4","--max-concurrency","1","--num-prompts","32"]),
    ("short_in_short_out",  ["--dataset-name","random","--random-input-len","128","--random-output-len","32","--max-concurrency","16","--num-prompts","128"]),
    ("sched_overhead_hiconc",["--dataset-name","random","--random-input-len","128","--random-output-len","16","--max-concurrency","64","--num-prompts","256"]),
    ("prefill_medium",      ["--dataset-name","random","--random-input-len","4096","--random-output-len","16","--max-concurrency","4","--num-prompts","48"]),
    ("prefill_long",        ["--dataset-name","random","--random-input-len","16384","--random-output-len","16","--max-concurrency","2","--num-prompts","16"]),
    ("decode_medium",       ["--dataset-name","random","--random-input-len","128","--random-output-len","512","--max-concurrency","16","--num-prompts","96"]),
    ("decode_heavy",        ["--dataset-name","random","--random-input-len","128","--random-output-len","1024","--max-concurrency","32","--num-prompts","96"]),
    ("agent_toolagent",     ["--dataset-name","mooncake","--mooncake-workload","toolagent","--max-concurrency","32","--num-prompts","96"]),
]

base_env = os.environ.copy()
base_env.update(dict(
    CUDA_HOME=ENVDIR,
    HF_HOME=f"{REPO}/.hf_cache",
    HF_HUB_CACHE=f"{REPO}/.hf_cache/hub",
    HF_DATASETS_CACHE=f"{REPO}/.hf_cache/datasets",
    PYTHONPATH=f"{SGLANG}/python",
    PATH=f"{ENVDIR}/bin:" + base_env.get("PATH", ""),
    CUDA_VISIBLE_DEVICES=GPU,
))


def set_arm(arm):
    """fallback = ensure ours config absent; ours = place ours config."""
    if arm == "ours":
        shutil.copy(OURS_CFG, TARGET_CFG)
        assert os.path.exists(TARGET_CFG)
    else:
        if os.path.exists(TARGET_CFG):
            os.remove(TARGET_CFG)
        assert not os.path.exists(TARGET_CFG)


def wait_ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return True
        except Exception:
            time.sleep(3)
    return False


def launch_server(arm):
    log = open(f"{REPO}/logs/v45_server_{arm}.log", "w")
    argv = [PY, "-m", "sglang.launch_server", "--model-path", MODEL, "--trust-remote-code",
            "--host", "127.0.0.1", "--port", PORT, "--mem-fraction-static", "0.85",
            "--attention-backend", "fa3", "--moe-runner-backend", "triton"]
    p = subprocess.Popen(argv, env=base_env, stdout=log, stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    return p, log


def run_bench(arm, name, args, rep):
    resfile = f"{OUTDIR}/{arm}_{name}_r{rep}.jsonl"
    if os.path.exists(resfile):
        os.remove(resfile)
    argv = [PY, "-m", "sglang.bench_serving", "--backend", "sglang",
            "--host", "127.0.0.1", "--port", PORT, "--model", MODEL,
            *args, "--output-file", resfile]
    log = f"{REPO}/logs/v45_bench_{arm}_{name}_r{rep}.log"
    with open(log, "w") as f:
        subprocess.run(argv, env=base_env, stdout=f, stderr=subprocess.STDOUT)
    if os.path.exists(resfile):
        rows = [json.loads(l) for l in open(resfile) if l.strip()]
        return rows[-1] if rows else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--arms", type=str, default="fallback,ours")
    ap.add_argument("--only", type=str, default="", help="comma-separated regime names to run (default all)")
    args = ap.parse_args()

    only = set(x for x in args.only.split(",") if x)
    regimes = [(n, a) for n, a in REGIMES if not only or n in only]

    raw_path = f"{OUTDIR}/server_ab.jsonl"
    for arm in args.arms.split(","):
        set_arm(arm)
        print(f"\n===== arm={arm} : launching server =====", flush=True)
        p, log = launch_server(arm)
        if not wait_ready():
            print(f"[{arm}] server failed to start", flush=True)
            try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception: pass
            log.close(); continue
        print(f"[{arm}] server ready", flush=True)
        for name, bargs in regimes:
            for rep in range(args.repeats):
                t0 = time.time()
                r = run_bench(arm, name, bargs, rep)
                if r:
                    row = {"arm": arm, "regime": name, "repeat": rep,
                           "ttft_ms": r.get("median_ttft_ms") or r.get("mean_ttft_ms"),
                           "tpot_ms": r.get("median_tpot_ms") or r.get("mean_tpot_ms"),
                           "e2e_ms": r.get("median_e2e_latency_ms") or r.get("mean_e2e_latency_ms"),
                           "out_tput": r.get("output_throughput"),
                           "in_tput": r.get("input_throughput"),
                           "wall_s": round(time.time() - t0, 1)}
                    with open(raw_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[{arm}] {name} r{rep}: TTFT={row['ttft_ms']:.1f} "
                          f"TPOT={row['tpot_ms']:.2f} E2E={row['e2e_ms']:.0f} "
                          f"out_tput={row['out_tput']:.0f} ({row['wall_s']:.0f}s)", flush=True)
                else:
                    print(f"[{arm}] {name} r{rep}: FAILED", flush=True)
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
        log.close()
        time.sleep(10)

    # restore stock sglang (remove ours)
    set_arm("fallback")
    print(f"\nraw -> {raw_path}", flush=True)


if __name__ == "__main__":
    main()
