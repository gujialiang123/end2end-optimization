#!/usr/bin/env python
"""v42: END-TO-END validation of MoE kernel-config tuning across ALL regimes.
Answers Chendi/user: the §1.6 kernel-level +35-54% (tuned config vs default
heuristic) — does it translate to END-TO-END prefill/decode?

A/B via SGLANG_MOE_CONFIG_DIR:
  - DEFAULT : empty config dir -> get_moe_configs returns None -> default heuristic
  - TUNED   : sglang as-is -> loads fallback triton_3_2_0 tuned config (all batches)

Sweeps (batch x input_len), each config x 3 repeats, parses prefill throughput +
decode median latency from --result-filename jsonl, computes per-regime deltas.
"""
import os, sys, subprocess, json, statistics, itertools

ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-dev"
PY = f"{ENVDIR}/bin/python"
MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
REPO = "/home/t-jialianggu/work/EndtoEnd-auto-optimization"
OUTDIR = f"{REPO}/results/2026-07-20_v42_kernel_e2e"
EMPTY = f"{OUTDIR}/emptyconfig"
N = int(os.environ.get("N_REPEAT", "3"))

# regimes: decode M=batch (small); prefill M~=batch*input (large)
BATCHES = ["1", "32", "64"]
INLENS = ["256", "2048", "4096"]
OUTLEN = "32"

base_env = os.environ.copy()
base_env.update(dict(CUDA_HOME=ENVDIR, HF_HOME=f"{REPO}/.hf_cache",
                     PATH=f"{ENVDIR}/bin:" + base_env.get("PATH", ""),
                     CUDA_VISIBLE_DEVICES="0"))

def run(config, rep):
    resfile = f"{OUTDIR}/res_{config}_rep{rep}.jsonl"
    if os.path.exists(resfile):
        os.remove(resfile)
    env = base_env.copy()
    if config == "default":
        env["SGLANG_MOE_CONFIG_DIR"] = EMPTY
    argv = [PY, "-m", "sglang.bench_one_batch",
            "--model-path", MODEL, "--trust-remote-code",
            "--batch-size", *BATCHES, "--input-len", *INLENS, "--output-len", OUTLEN,
            "--attention-backend", "fa3", "--moe-runner-backend", "triton",
            "--mem-fraction-static", "0.85",
            "--result-filename", resfile]
    log = f"{REPO}/logs/v42_{config}_rep{rep}.log"
    with open(log, "w") as f:
        subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
    rows = []
    if os.path.exists(resfile):
        for line in open(resfile):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# collect: data[(b,il)][config] = list of dicts
data = {}
for rep in range(N):
    for config in ["default", "tuned"]:
        rows = run(config, rep)
        print(f"[rep{rep}] {config}: {len(rows)} rows", flush=True)
        for r in rows:
            key = (r["batch_size"], r["input_len"])
            data.setdefault(key, {}).setdefault(config, []).append(r)

def med(lst, field):
    vals = [x[field] for x in lst if field in x and x[field] is not None]
    return statistics.median(vals) if vals else None

print("\n===== E2E: tuned config vs default heuristic (median of 3) =====")
print(f"{'regime':28s} {'prefill_tput default->tuned':32s} {'decode_lat default->tuned':32s}")
summary = []
for (b, il) in sorted(data.keys(), key=lambda k: (int(k[0]), int(k[1]))):
    d = data[(b, il)]
    if "default" not in d or "tuned" not in d:
        continue
    # prefill throughput (higher=better); decode median latency ms (lower=better)
    pf_def = med(d["default"], "prefill_throughput")
    pf_tun = med(d["tuned"], "prefill_throughput")
    dc_def = med(d["default"], "median_decode_latency")
    dc_tun = med(d["tuned"], "median_decode_latency")
    pf_gain = (pf_tun / pf_def - 1) * 100 if pf_def else float('nan')
    dc_gain = (dc_def / dc_tun - 1) * 100 if (dc_def and dc_tun) else float('nan')  # +ve=tuned faster
    prefill_M = int(b) * int(il)
    regime = f"b={b},in={il} (pfM={prefill_M})"
    pf_str = f"{pf_def:.0f}->{pf_tun:.0f} ({pf_gain:+.1f}%)" if pf_def else "n/a"
    dc_str = f"{dc_def*1000:.3f}->{dc_tun*1000:.3f}ms ({dc_gain:+.1f}%)" if (dc_def and dc_tun) else "n/a"
    print(f"{regime:28s} {pf_str:32s} {dc_str:32s}")
    summary.append(dict(batch=b, input_len=il, prefill_M=prefill_M,
                        prefill_tput_default=pf_def, prefill_tput_tuned=pf_tun, prefill_gain_pct=pf_gain,
                        decode_lat_default_ms=dc_def*1000 if dc_def else None,
                        decode_lat_tuned_ms=dc_tun*1000 if dc_tun else None,
                        decode_gain_pct=dc_gain))
json.dump(dict(N=N, batches=BATCHES, inlens=INLENS, summary=summary),
          open(f"{OUTDIR}/summary.json", "w"), indent=2)
print(f"\nsaved {OUTDIR}/summary.json")
