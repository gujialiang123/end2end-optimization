#!/usr/bin/env python3
"""v42: End-to-end A/B of fused_moe kernel-config tuning (the gap left by §1.6).

Compares sglang's REAL loaded config for the uncovered Qwen3-30B-A3B shape
(E=128,N=768,H200) under two variants:

  * baseline  -> stock sglang: triton_3_6_0 has no config for this shape, so it
                 falls back to triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json
                 ("Performance might be sub-optimal!").
  * ours      -> our config re-tuned ON triton 3.6.0 for this exact shape, placed in
                 triton_3_6_0/ so sglang loads it first (version match).

Unlike §1.6 (isolated kernel micro-time), this measures END-TO-END:
  - decode TPOT (median decode-step latency) at batch = 1/8/32
  - prefill throughput at input-len = 512/1024/2048/4096 (batch 1)
Each point is repeated N times; we report median and a Welch t-test (baseline vs ours).

This script only RUNS bench_one_batch and parses stdout. Config-file placement is
handled by the caller (bash), which sets --label for provenance. Raw per-run numbers
are written to a jsonl so new metrics can be recomputed post-hoc.
"""
import argparse, json, os, re, subprocess, sys, time

MODEL = os.environ.get("MODEL",
    "/home/t-jialianggu/work/models/Qwen3-30B-A3B-Instruct-2507")

PREFILL_RE = re.compile(r"Prefill\.\s*latency:\s*([0-9.]+)\s*s,\s*throughput:\s*([0-9.]+)\s*token/s")
DECODE_MED_RE = re.compile(r"Decode\.\s*median latency:\s*([0-9.]+)\s*s,\s*median throughput:\s*([0-9.]+)\s*token/s")
TOTAL_RE = re.compile(r"Total\.\s*latency:\s*([0-9.]+)\s*s,\s*throughput:\s*([0-9.]+)\s*token/s")


def run_one(batch, input_len, output_len, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable, "-m", "sglang.bench_one_batch",
        "--model-path", MODEL, "--trust-remote-code",
        "--batch-size", str(batch),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--attention-backend", "fa3",
        "--moe-runner-backend", "triton",
        "--mem-fraction-static", "0.85",
        "--log-decode-step", "1",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    out = p.stdout + "\n" + p.stderr
    # The benchmark prints TWO passes (warmup + measured); take the LAST match of each.
    pf = PREFILL_RE.findall(out)
    dm = DECODE_MED_RE.findall(out)
    tot = TOTAL_RE.findall(out)
    fallback = "Fallback to triton version" in out or "sub-optimal" in out
    used_ours = ("Using MoE kernel config" in out) and not fallback
    res = {
        "prefill_lat_s": float(pf[-1][0]) if pf else None,
        "prefill_tput": float(pf[-1][1]) if pf else None,
        "decode_med_lat_s": float(dm[-1][0]) if dm else None,
        "decode_med_tput": float(dm[-1][1]) if dm else None,
        "total_lat_s": float(tot[-1][0]) if tot else None,
        "wall_s": round(time.time() - t0, 1),
        "fallback_warning": fallback,
        "rc": p.returncode,
    }
    if p.returncode != 0 and not pf:
        res["stderr_tail"] = out[-2000:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="baseline | ours")
    ap.add_argument("--out", required=True, help="output jsonl path (appended)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decode-batches", type=str, default="1,8,32")
    ap.add_argument("--prefill-inputs", type=str, default="512,1024,2048,4096")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    matrix = []
    # decode-focused: short input, longer decode to get a stable decode median
    for b in [int(x) for x in args.decode_batches.split(",") if x]:
        matrix.append(("decode", b, 256, 64))
    # prefill-focused: batch 1, sweep input length, tiny decode
    for il in [int(x) for x in args.prefill_inputs.split(",") if x]:
        matrix.append(("prefill", 1, il, 8))

    for kind, b, il, ol in matrix:
        for r in range(args.repeats):
            res = run_one(b, il, ol)
            row = {"label": args.label, "kind": kind, "batch": b,
                   "input_len": il, "output_len": ol, "repeat": r,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **res}
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            tag = f"[{args.label}] {kind} b={b} in={il} r={r}"
            print(f"{tag}: prefill_tput={res['prefill_tput']} "
                  f"decode_med_lat={res['decode_med_lat_s']} "
                  f"fallback_warn={res['fallback_warning']} rc={res['rc']}",
                  flush=True)


if __name__ == "__main__":
    main()
