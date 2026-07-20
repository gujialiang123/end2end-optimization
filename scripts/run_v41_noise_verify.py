#!/usr/bin/env python
"""Chendi verification: is the custom-MoE-kernel b1 +1.4% e2e real or noise?
Runs MANY interleaved separate bench_one_batch launches (cudagraph ON, matching
the ORIGINAL measurement condition) for baseline (CUSTOM_MOE=0) vs custom
(CUSTOM_MOE=1), across batches {1,2,4}, and computes per-batch statistics:
mean, std, 95% CI, Welch t-test, and whether the delta exceeds the noise band.
"""
import os, sys, subprocess, re, json, statistics, math, time

ENV = os.environ.copy()
ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{ENVDIR}/bin/python"
MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
REPO = "/home/t-jialianggu/work/EndtoEnd-auto-optimization"
N = int(os.environ.get("N_REPEAT", "12"))
BATCHES = ["1", "2", "4"]
OUTLEN = "64"

ENV.update(dict(
    CUDA_HOME=ENVDIR, HF_HOME=f"{REPO}/.hf_cache",
    PATH=f"{ENVDIR}/bin:" + ENV.get("PATH", ""),
    CUDA_VISIBLE_DEVICES="0",
    CUSTOM_MOE_MAX_M="4",
))

DRIVER = f"""
import os, sys, runpy
sys.path.insert(0, "{REPO}/scripts")
import custom_moe_patch
custom_moe_patch.install()
sys.argv = ["sglang.bench_one_batch","--model-path","{MODEL}","--trust-remote-code",
  "--batch-size",{",".join(f'"{b}"' for b in BATCHES)},
  "--input-len","256","--output-len","{OUTLEN}",
  "--attention-backend","fa3","--moe-runner-backend","triton",
  "--mem-fraction-static","0.85"]
runpy.run_module("sglang.bench_one_batch", run_name="__main__")
print("STATS", custom_moe_patch.stats(), flush=True)
"""

def run_once(custom: bool):
    env = ENV.copy()
    env["CUSTOM_MOE"] = "1" if custom else "0"
    p = subprocess.run([PY, "-c", DRIVER], env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    # parse per-batch: bench prints results in batch order; median decode latency lines
    lats = re.findall(r"Decode\.  median latency:\s*([0-9.]+)\s*s", out)
    # bench prints one warmup median (bs[0]) + one per batch; keep the last len(BATCHES)
    lats = lats[-len(BATCHES):]
    stats = re.search(r"STATS (\{.*\})", out)
    return [float(x) for x in lats], (stats.group(1) if stats else "")

# collect
data = {b: {"base": [], "custom": []} for b in BATCHES}
kernel_fired = None
log = open(f"{REPO}/logs/v41_noise_verify.log", "w")
for rep in range(N):
    for custom in ([False, True] if rep % 2 == 0 else [True, False]):  # alternate order
        t0 = time.time()
        lats, st = run_once(custom)
        tag = "custom" if custom else "base"
        if len(lats) == len(BATCHES):
            for b, l in zip(BATCHES, lats):
                data[b][tag].append(l * 1000.0)  # ms
        msg = f"rep{rep} {tag}: lats(ms)={[round(l*1000,3) for l in lats]} {st[:80]} ({time.time()-t0:.0f}s)"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if custom and st:
            kernel_fired = st

def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return ma, mb, 0.0, float("inf")
    t = (ma - mb) / se
    return ma, mb, se, t

print("\n===== RESULTS (baseline vs custom, decode median latency ms) =====")
log.write("\n===== RESULTS =====\n")
summary = {}
for b in BATCHES:
    base, cust = data[b]["base"], data[b]["custom"]
    if len(base) < 2 or len(cust) < 2:
        continue
    mb_, mc_, se, t = welch(base, cust)
    sb, sc = statistics.stdev(base), statistics.stdev(cust)
    delta_pct = (mb_ - mc_) / mb_ * 100  # +ve => custom faster
    # 95% CI of baseline noise band (relative)
    ci_base = 1.96 * sb / math.sqrt(len(base)) / mb_ * 100
    line = (f"b={b}: base {mb_:.3f}±{sb:.3f} (n={len(base)})  "
            f"custom {mc_:.3f}±{sc:.3f} (n={len(cust)})  "
            f"delta={delta_pct:+.2f}%  |t|={abs(t):.2f}  "
            f"baseline_95%CI=±{ci_base:.2f}%  "
            f"verdict={'SIGNAL' if abs(t)>2.0 else 'NOISE'}")
    print(line); log.write(line + "\n")
    summary[b] = dict(base_mean=mb_, base_std=sb, custom_mean=mc_, custom_std=sc,
                      delta_pct=delta_pct, t=t, n_base=len(base), n_custom=len(cust),
                      verdict=("SIGNAL" if abs(t) > 2.0 else "NOISE"))
print("kernel_fired:", kernel_fired)
json.dump(dict(summary=summary, kernel_fired=kernel_fired, N=N, outlen=OUTLEN),
          open(f"{REPO}/results/2026-07-20_v41_noise/summary.json", "w"), indent=2)
log.close()
