#!/usr/bin/env python3
"""v51 orchestrator — high-concurrency TTFT rerun (2 models x 3 configs x 2 regimes).

Faithfully reproduces the v4 slide workload (200-word seed=2026 prompts, closed-loop
ThreadPoolExecutor(concurrency), temp=0, ignore_eos, fixed max_new) but with a STREAMING
client to capture TTFT. 6 reps/regime (drop rep0). Randomized regime order per config.
"""
from __future__ import annotations
import json, os, random, signal, socket, statistics, subprocess, sys, time, csv, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v51_stream_bench as B

ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{ENVDIR}/bin/python"
REPO = Path("/home/t-jialianggu/work/EndtoEnd-auto-optimization")
OUTDIR = REPO / "results/2026-07-23_high_concurrency_ttft_rerun"
GPU = os.environ.get("GPU", "7")
PORT = int(os.environ.get("PORT", "32207"))
N_REP = 6  # rep0 dropped, reps 1-5 retained

MODELS = {
    "qwen": dict(dir="qwen", path="/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
                 served="qwen3-30b-a3b", extra=[]),
    "lfm25": dict(dir="lfm25", path="/data/hf/LFM2.5-8B-A1B",
                  served="lfm2.5-8b-a1b", extra=["--max-prefill-tokens", "16384"]),
}
CONFIGS = {
    "baseline":            dict(cap=32,  chunk=-1,   sched="lpm",  mem=0.85),
    "cap_only":            dict(cap=128, chunk=-1,   sched="lpm",  mem=0.85),
    "full_high_concurrency": dict(cap=128, chunk=2048, sched="fcfs", mem=0.90),
}
REGIMES = {
    "C64_O512":  dict(num_prompts=64,  words=200, max_new=512, conc=64),
    "C128_O256": dict(num_prompts=128, words=200, max_new=256, conc=128),
}


def port_free(p):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", p)) != 0


def wait_port_free(p, t=120):
    t0 = time.time()
    while time.time() - t0 < t:
        if port_free(p): return True
        time.sleep(2)
    return False


def wait_gpu_free(gpu, need_free=100000, t=180):
    t0 = time.time()
    while time.time() - t0 < t:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits", "-i", str(gpu)],
                           capture_output=True, text=True)
        try: used = int(o.stdout.strip().split("\n")[0])
        except: used = 0
        if (143771 - used) >= need_free: return True
        time.sleep(3)
    return False


def launch(model, cfg, log):
    m = MODELS[model]; c = CONFIGS[cfg]
    argv = [PY, "-m", "sglang.launch_server", "--model-path", m["path"],
            "--served-model-name", m["served"], "--host", "127.0.0.1",
            "--port", str(PORT), "--tensor-parallel-size", "1",
            "--context-length", "8192", "--schedule-conservativeness", "1.0",
            "--trust-remote-code", "--moe-runner-backend", "auto",
            "--mem-fraction-static", str(c["mem"]),
            "--max-running-requests", str(c["cap"]),
            "--chunked-prefill-size", str(c["chunk"]),
            "--schedule-policy", c["sched"]] + m["extra"]
    env = os.environ.copy()
    env.update(dict(CUDA_HOME=ENVDIR, HF_HOME=str(REPO/".hf_cache"),
                    PATH=f"{ENVDIR}/bin:"+env.get("PATH",""), CUDA_VISIBLE_DEVICES=str(GPU)))
    lf = open(log, "w")
    p = subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return p, argv


def wait_health(p, t=420):
    import urllib.request
    t0 = time.time()
    while time.time()-t0 < t:
        if p.poll() is not None: return False, f"exited rc={p.returncode}"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return True, time.time()-t0
        except: time.sleep(3)
    return False, "timeout"


def kill(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=30)
    except:
        try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except: pass


def verify_cfg(log, cfg):
    txt = Path(log).read_text(errors="ignore")
    c = CONFIGS[cfg]
    ok = (f"max_running_requests={c['cap']}" in txt.replace("'","").replace(" ","") or
          f"'max_running_requests': {c['cap']}" in txt or
          f"max_running_requests={c['cap']}" in txt)
    cg = ("disable_cuda_graph=False" in txt or "Capture cuda graph" in txt)
    return cg  # cuda graph presence is the key invariant; cap parsed loosely


def pctl(xs, p):
    return float(statistics.quantiles(sorted(xs), n=100)[p-1]) if len(xs) > 1 else (xs[0] if xs else 0.0)


def agg_run(res):
    """Per-run aggregates from a single streaming bench result."""
    ok = [r for r in res["records"] if r["ok"] and r["ttft_s"] is not None]
    ttft = [r["ttft_s"]*1000 for r in ok]
    tpot = [ (sum(r["itls"])/len(r["itls"]))*1000 for r in ok if r["itls"] ]
    e2e = [r["e2e_s"]*1000 for r in ok]
    return dict(
        req_per_s=res["req_per_s"], output_tok_per_s=res["output_tokens_per_s"],
        num_ok=res["num_ok"], num_total=res["num_total"],
        completion_rate=res["completion_rate"],
        ttft_mean=statistics.mean(ttft) if ttft else 0,
        ttft_p50=statistics.median(ttft) if ttft else 0,
        ttft_p95=pctl(ttft,95), ttft_p99=pctl(ttft,99),
        tpot_mean=statistics.mean(tpot) if tpot else 0,
        tpot_p50=statistics.median(tpot) if tpot else 0,
        tpot_p95=pctl(tpot,95), tpot_p99=pctl(tpot,99),
        e2e_mean=statistics.mean(e2e) if e2e else 0,
        e2e_p50=statistics.median(e2e) if e2e else 0,
        e2e_p95=pctl(e2e,95), e2e_p99=pctl(e2e,99),
    )


def ci95(xs):
    if len(xs) < 2: return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def write_env():
    import torch
    def sh(c): 
        try: return subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
        except: return "?"
    env = dict(
        recorded_at=sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
        repo_git_sha=sh("git rev-parse HEAD"),
        sglang_src_sha=sh("git -C /home/t-jialianggu/work/sglang rev-parse HEAD"),
        gpu=torch.cuda.get_device_name(0), gpu_index_used=GPU,
        gpu_uuid=sh(f"nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i {GPU}"),
        torch=torch.__version__, cuda=torch.version.cuda,
        triton=__import__("triton").__version__,
        sglang=__import__("sglang").__version__,
        flashinfer=sh(f"{PY} -c 'import flashinfer;print(flashinfer.__version__)'"),
        driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        models={k: v["path"] for k,v in MODELS.items()},
        dtype="bf16", tp=1, cuda_graph="on",
        original_experiment="results/2026-07-07_v4_decode_sweep (non-streaming, no TTFT)",
        note="streaming client v51_stream_bench.py; identical workload/seed; TTFT includes admission queueing",
    )
    json.dump(env, open(OUTDIR/"environment.json","w"), indent=2)
    return env


PER_RUN_FIELDS = ["model","regime","configuration","repeat","req_per_s","output_tok_per_s",
    "num_ok","num_total","completion_rate","ttft_mean","ttft_p50","ttft_p95","ttft_p99",
    "tpot_mean","tpot_p50","tpot_p95","tpot_p99","e2e_mean","e2e_p50","e2e_p95","e2e_p99"]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_env()
    per_run_rows = []
    per_req_rows = []
    failures = []
    rng = random.Random(20260723)

    for model in ["qwen", "lfm25"]:
        for cfg in ["baseline", "cap_only", "full_high_concurrency"]:
            cdir = OUTDIR / MODELS[model]["dir"] / cfg
            cdir.mkdir(parents=True, exist_ok=True)
            slog = cdir / "server.log"
            if not wait_port_free(PORT, 60):
                failures.append(dict(model=model, cfg=cfg, reason="port_busy")); continue
            proc, argv = launch(model, cfg, slog)
            (cdir/"server_cmd.txt").write_text(" ".join(argv))
            ok, info = wait_health(proc)
            if not ok:
                kill(proc); failures.append(dict(model=model, cfg=cfg, reason=f"server_not_ready:{info}"))
                wait_gpu_free(GPU); continue
            if not verify_cfg(slog, cfg):
                kill(proc); failures.append(dict(model=model, cfg=cfg, reason="cuda_graph_not_confirmed"))
                wait_gpu_free(GPU); continue
            print(f"[{model}/{cfg}] server ready ({info:.0f}s)", flush=True)
            url = f"http://127.0.0.1:{PORT}"
            # warmup each regime once (unscored)
            for reg in REGIMES:
                r = REGIMES[reg]
                B.run_regime_once(url, r["num_prompts"], r["words"], r["max_new"], r["conc"])
            # 6 reps, randomized regime order each rep; drop rep0
            for rep in range(N_REP):
                order = list(REGIMES.keys()); rng.shuffle(order)
                for reg in order:
                    r = REGIMES[reg]
                    res = B.run_regime_once(url, r["num_prompts"], r["words"], r["max_new"], r["conc"])
                    # save raw per-request for rep>=1
                    if rep >= 1:
                        raw = cdir / f"{reg}_rep{rep}.json"
                        json.dump(res, open(raw,"w"))
                        a = agg_run(res)
                        row = dict(model=model, regime=reg, configuration=cfg, repeat=rep, **{k:round(a[k],3) for k in a})
                        per_run_rows.append(row)
                        for rec in res["records"]:
                            if rec["ok"] and rec["ttft_s"] is not None:
                                per_req_rows.append(dict(model=model, regime=reg, configuration=cfg,
                                    repeat=rep, idx=rec["idx"], ttft_ms=round(rec["ttft_s"]*1000,3),
                                    e2e_ms=round(rec["e2e_s"]*1000,3), output_tokens=rec["output_tokens"]))
                    print(f"  [{model}/{cfg}] {reg} rep{rep}: req/s={res['req_per_s']:.2f} "
                          f"ttft_p50={statistics.median([x['ttft_s']*1000 for x in res['records'] if x['ok'] and x['ttft_s']]):.0f}ms", flush=True)
            kill(proc)
            wait_gpu_free(GPU, need_free=100000, t=120)
            time.sleep(5)

    # write per_run_metrics.csv
    with open(OUTDIR/"per_run_metrics.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_RUN_FIELDS); w.writeheader()
        for r in per_run_rows: w.writerow(r)
    # write per_request_metrics.csv
    with open(OUTDIR/"per_request_metrics.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","regime","configuration","repeat","idx","ttft_ms","e2e_ms","output_tokens"])
        w.writeheader()
        for r in per_req_rows: w.writerow(r)
    # write failures
    if failures:
        with open(OUTDIR/"failures.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model","cfg","reason"]); w.writeheader()
            for r in failures: w.writerow(r)

    # summary.csv: aggregate reps 1-5 (mean + ci95 on req/s; mean on latency percentiles)
    SUMM = ["model","regime","configuration","repeat_count","request_throughput_mean","request_throughput_ci95",
            "output_throughput_mean","ttft_mean_ms","ttft_p50_ms","ttft_p95_ms","ttft_p99_ms",
            "tpot_mean_ms","tpot_p50_ms","tpot_p95_ms","tpot_p99_ms",
            "e2e_mean_ms","e2e_p50_ms","e2e_p95_ms","e2e_p99_ms"]
    from collections import defaultdict
    groups = defaultdict(list)
    for r in per_run_rows:
        groups[(r["model"],r["regime"],r["configuration"])].append(r)
    with open(OUTDIR/"summary.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMM); w.writeheader()
        for (model,reg,cfg),rows in groups.items():
            def m(k): return statistics.mean([x[k] for x in rows])
            rps=[x["req_per_s"] for x in rows]
            w.writerow(dict(model=model, regime=reg, configuration=cfg, repeat_count=len(rows),
                request_throughput_mean=round(m("req_per_s"),3), request_throughput_ci95=round(ci95(rps),3),
                output_throughput_mean=round(m("output_tok_per_s"),1),
                ttft_mean_ms=round(m("ttft_mean"),1), ttft_p50_ms=round(m("ttft_p50"),1),
                ttft_p95_ms=round(m("ttft_p95"),1), ttft_p99_ms=round(m("ttft_p99"),1),
                tpot_mean_ms=round(m("tpot_mean"),2), tpot_p50_ms=round(m("tpot_p50"),2),
                tpot_p95_ms=round(m("tpot_p95"),2), tpot_p99_ms=round(m("tpot_p99"),2),
                e2e_mean_ms=round(m("e2e_mean"),1), e2e_p50_ms=round(m("e2e_p50"),1),
                e2e_p95_ms=round(m("e2e_p95"),1), e2e_p99_ms=round(m("e2e_p99"),1)))
    print(f"\nDONE. {len(per_run_rows)} per-run rows, {len(per_req_rows)} per-request rows, {len(failures)} failures")


if __name__ == "__main__":
    main()
