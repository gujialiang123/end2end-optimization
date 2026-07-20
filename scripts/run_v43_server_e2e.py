#!/usr/bin/env python
"""v43: server-level END-TO-END A/B (default heuristic vs tuned MoE config) across
our artificial regimes + the sglang agent dataset (mooncake toolagent).

For each regime: launch sglang server (config A=default via empty
SGLANG_MOE_CONFIG_DIR, B=tuned=sglang default fallback), run bench_serving,
record e2e metrics (TTFT, TPOT, E2E latency, output throughput). 2 servers per
regime group are avoided by looping config outer, regime inner (server reused).
"""
import os, sys, subprocess, json, time, signal, urllib.request

ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{ENVDIR}/bin/python"
MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
REPO = "/home/t-jialianggu/work/EndtoEnd-auto-optimization"
OUTDIR = f"{REPO}/results/2026-07-20_v43_server_e2e"
EMPTY = f"{REPO}/results/2026-07-20_v42_kernel_e2e/emptyconfig"
PORT = "31555"
GPU = os.environ.get("GPU", "0")

os.makedirs(OUTDIR, exist_ok=True)

# regimes: (name, bench_serving args). Keep num_prompts modest for time.
REGIMES = [
    ("tiny_latency",       ["--dataset-name","random","--random-input-len","8","--random-output-len","4","--max-concurrency","1","--num-prompts","32"]),
    ("short_in_short_out", ["--dataset-name","random","--random-input-len","128","--random-output-len","32","--max-concurrency","16","--num-prompts","128"]),
    ("sched_overhead_hiconc",["--dataset-name","random","--random-input-len","128","--random-output-len","16","--max-concurrency","64","--num-prompts","256"]),
    ("prefill_medium",     ["--dataset-name","random","--random-input-len","4096","--random-output-len","16","--max-concurrency","4","--num-prompts","48"]),
    ("prefill_long",       ["--dataset-name","random","--random-input-len","16384","--random-output-len","16","--max-concurrency","2","--num-prompts","16"]),
    ("decode_medium",      ["--dataset-name","random","--random-input-len","128","--random-output-len","512","--max-concurrency","16","--num-prompts","96"]),
    ("decode_heavy",       ["--dataset-name","random","--random-input-len","128","--random-output-len","1024","--max-concurrency","32","--num-prompts","96"]),
    ("agent_toolagent",    ["--dataset-name","mooncake","--mooncake-workload","toolagent","--max-concurrency","32","--num-prompts","96"]),
]

base_env = os.environ.copy()
base_env.update(dict(CUDA_HOME=ENVDIR, HF_HOME=f"{REPO}/.hf_cache",
                     HF_HUB_CACHE=f"{REPO}/.hf_cache/hub",
                     HF_DATASETS_CACHE=f"{REPO}/.hf_cache/datasets",
                     PATH=f"{ENVDIR}/bin:" + base_env.get("PATH",""),
                     CUDA_VISIBLE_DEVICES=GPU))

def wait_ready(timeout=300):
    t0=time.time()
    while time.time()-t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return True
        except Exception:
            time.sleep(3)
    return False

def launch_server(config):
    env = base_env.copy()
    if config == "default":
        env["SGLANG_MOE_CONFIG_DIR"] = EMPTY
    log = open(f"{REPO}/logs/v43_server_{config}.log","w")
    argv = [PY,"-m","sglang.launch_server","--model-path",MODEL,"--trust-remote-code",
            "--host","127.0.0.1","--port",PORT,"--mem-fraction-static","0.85",
            "--attention-backend","fa3","--moe-runner-backend","triton"]
    p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    return p, log

def run_bench(config, name, args):
    resfile = f"{OUTDIR}/{config}_{name}.jsonl"
    if os.path.exists(resfile): os.remove(resfile)
    env = base_env.copy()
    argv = [PY,"-m","sglang.bench_serving","--backend","sglang",
            "--host","127.0.0.1","--port",PORT,"--model",MODEL,
            *args,"--output-file",resfile]
    log = f"{REPO}/logs/v43_bench_{config}_{name}.log"
    with open(log,"w") as f:
        subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
    if os.path.exists(resfile):
        rows=[json.loads(l) for l in open(resfile) if l.strip()]
        return rows[-1] if rows else None
    return None

results = {}
for config in ["default","tuned"]:
    print(f"\n===== launching {config} server =====", flush=True)
    p, log = launch_server(config)
    if not wait_ready():
        print(f"[{config}] server failed to start", flush=True)
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
        log.close()
        continue
    print(f"[{config}] server ready", flush=True)
    for name, args in REGIMES:
        t0=time.time()
        r = run_bench(config, name, args)
        if r:
            results.setdefault(name,{})[config] = r
            print(f"[{config}] {name}: TPOT={r.get('median_tpot_ms') or r.get('mean_tpot_ms'):.2f}ms "
                  f"out_tput={r.get('output_throughput'):.0f} ({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"[{config}] {name}: FAILED", flush=True)
    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception: pass
    log.close()
    time.sleep(10)

# summarize
print("\n===== SERVER E2E: tuned vs default (per regime) =====", flush=True)
summary=[]
for name,_ in REGIMES:
    d=results.get(name,{})
    if "default" not in d or "tuned" not in d: continue
    de,tu=d["default"],d["tuned"]
    def g(field, lower_better=True):
        a,b=de.get(field),tu.get(field)
        if a is None or b is None or a==0: return (a,b,None)
        gain=(a/b-1)*100 if lower_better else (b/a-1)*100
        return (a,b,gain)
    tpot=g("median_tpot_ms") if de.get("median_tpot_ms") else g("mean_tpot_ms")
    ttft=g("median_ttft_ms") if de.get("median_ttft_ms") else g("mean_ttft_ms")
    e2e=g("median_e2e_latency_ms") if de.get("median_e2e_latency_ms") else g("mean_e2e_latency_ms")
    otp=g("output_throughput", lower_better=False)
    print(f"{name:22s} TPOT {tpot[2]:+.1f}%  TTFT {ttft[2]:+.1f}%  E2E {e2e[2]:+.1f}%  out_tput {otp[2]:+.1f}%")
    summary.append(dict(regime=name, tpot=tpot, ttft=ttft, e2e=e2e, out_tput=otp))
json.dump(dict(summary=summary, raw=results), open(f"{OUTDIR}/summary.json","w"), indent=2, default=str)
print(f"\nsaved {OUTDIR}/summary.json")
