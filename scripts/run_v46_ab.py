#!/usr/bin/env python
"""v46 A/B: OURS (retuned for triton 3.5.1) vs FALLBACK (sglang现状, triton_3_2_0).

This is the HONEST comparison: baseline = what sglang actually loads today
(the triton_3_2_0 fallback config), treatment = our freshly retuned config.

Switch mechanism: the ours config lives at configs/triton_3_5_1/<name>.json.
- OURS run: file present -> sglang loads it.
- FALLBACK run: we point SGLANG_MOE_CONFIG_DIR at a dir whose triton_3_5_1 has
  NO such file, but triton_3_2_0 does (copied), so sglang falls back exactly like
  production does today.

bench_one_batch, batch x input sweep, N repeats, median. Reports prefill
throughput, decode latency, total e2e.
"""
import os, sys, subprocess, json, statistics, shutil

ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{ENVDIR}/bin/python"
MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
REPO = "/home/t-jialianggu/work/EndtoEnd-auto-optimization"
SGLANG_CFG = "/home/t-jialianggu/work/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs"
OUTDIR = f"{REPO}/results/2026-07-21_v46_retune/ab"
N = int(os.environ.get("N_REPEAT", "3"))

BATCHES = ["1", "32", "64"]
INLENS = ["256", "2048", "4096"]
OUTLEN = "32"
NAME = "E=128,N=768,device_name=NVIDIA_H200.json"

os.makedirs(OUTDIR, exist_ok=True)

# Build a fallback config dir: triton_3_5_1 WITHOUT our file, triton_3_2_0 WITH the
# stock fallback -> reproduces sglang production behavior (falls back to 3.2.0).
FB_DIR = f"{OUTDIR}/fallback_cfgdir/configs"
os.makedirs(f"{FB_DIR}/triton_3_5_1", exist_ok=True)
os.makedirs(f"{FB_DIR}/triton_3_2_0", exist_ok=True)
# copy the stock 3.2.0 fallback so lookup finds it
shutil.copy(f"{SGLANG_CFG}/triton_3_2_0/{NAME}", f"{FB_DIR}/triton_3_2_0/{NAME}")
# also copy the down/fp8 siblings if present (harmless)
for f in os.listdir(f"{SGLANG_CFG}/triton_3_2_0"):
    if f.startswith("E=128,N=768"):
        shutil.copy(f"{SGLANG_CFG}/triton_3_2_0/{f}", f"{FB_DIR}/triton_3_2_0/{f}")
# ensure NO E=128,N=768 file in the fallback triton_3_5_1 dir
assert not os.path.exists(f"{FB_DIR}/triton_3_5_1/{NAME}")

base_env = os.environ.copy()
base_env.update(dict(CUDA_HOME=ENVDIR, HF_HOME=f"{REPO}/.hf_cache",
                     PATH=f"{ENVDIR}/bin:" + base_env.get("PATH", ""),
                     CUDA_VISIBLE_DEVICES="0"))

def run(config, rep):
    resfile = f"{OUTDIR}/res_{config}_rep{rep}.jsonl"
    if os.path.exists(resfile):
        os.remove(resfile)
    env = base_env.copy()
    if config == "fallback":
        # point at our fallback dir -> sglang won't find 3.5.1 config, falls back to 3.2.0
        env["SGLANG_MOE_CONFIG_DIR"] = f"{OUTDIR}/fallback_cfgdir"
    # ours: default env -> sglang loads the triton_3_5_1 file we installed
    argv = [PY, "-m", "sglang.bench_one_batch",
            "--model-path", MODEL, "--trust-remote-code",
            "--batch-size", *BATCHES, "--input-len", *INLENS, "--output-len", OUTLEN,
            "--attention-backend", "fa3", "--moe-runner-backend", "triton",
            "--mem-fraction-static", "0.85", "--result-filename", resfile]
    log = f"{REPO}/logs/v46ab_{config}_rep{rep}.log"
    with open(log, "w") as f:
        subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
    rows = []
    if os.path.exists(resfile):
        rows = [json.loads(l) for l in open(resfile) if l.strip()]
    # sanity: record which config sglang actually loaded
    loaded = "?"
    for line in open(log, errors="ignore"):
        if "Using MoE kernel config from" in line:
            loaded = "triton_3_5_1(OURS)" if "triton_3_5_1" in line else "other"
            break
        if "Fallback to triton version 3.2.0" in line:
            loaded = "triton_3_2_0(FALLBACK)"
            break
    return rows, loaded

data = {}
for rep in range(N):
    for config in ["fallback", "ours"]:
        rows, loaded = run(config, rep)
        print(f"[rep{rep}] {config}: {len(rows)} rows, loaded={loaded}", flush=True)
        for r in rows:
            key = (r["batch_size"], r["input_len"])
            data.setdefault(key, {}).setdefault(config, []).append(r)

def med(lst, field):
    v = [x[field] for x in lst if x.get(field) is not None]
    return statistics.median(v) if v else None

print("\n===== v46 A/B: OURS (retuned 3.5.1) vs FALLBACK (sglang现状 3.2.0), median of N =====")
print(f"{'regime':22s} {'prefill tput fb->ours':30s} {'decode lat fb->ours':28s} {'total e2e':22s}")
summary = []
for (b, il) in sorted(data.keys(), key=lambda k: (int(k[0]), int(k[1]))):
    d = data[(b, il)]
    if "fallback" not in d or "ours" not in d:
        continue
    pf_fb = med(d["fallback"], "prefill_throughput"); pf_ou = med(d["ours"], "prefill_throughput")
    dc_fb = med(d["fallback"], "median_decode_latency"); dc_ou = med(d["ours"], "median_decode_latency")
    tt_fb = med(d["fallback"], "total_latency"); tt_ou = med(d["ours"], "total_latency")
    pf_g = (pf_ou/pf_fb - 1)*100 if pf_fb else float('nan')
    dc_g = (dc_fb/dc_ou - 1)*100 if (dc_fb and dc_ou) else float('nan')
    tt_g = (tt_fb/tt_ou - 1)*100 if (tt_fb and tt_ou) else float('nan')
    print("b=%s,in=%-5s  %.0f->%.0f (%+.2f%%)   %.3f->%.3fms (%+.2f%%)   %+.2f%%" % (
          b, il, pf_fb, pf_ou, pf_g, dc_fb*1000, dc_ou*1000, dc_g, tt_g))
    summary.append(dict(batch=b, input_len=il,
                        prefill_fb=pf_fb, prefill_ours=pf_ou, prefill_gain_pct=pf_g,
                        decode_fb_ms=dc_fb*1000, decode_ours_ms=dc_ou*1000, decode_gain_pct=dc_g,
                        total_gain_pct=tt_g))
json.dump(dict(N=N, note="ours(retuned triton3.5.1) vs fallback(sglang现状 triton3.2.0)",
               summary=summary), open(f"{OUTDIR}/summary.json", "w"), indent=2)
print(f"\nsaved {OUTDIR}/summary.json")
