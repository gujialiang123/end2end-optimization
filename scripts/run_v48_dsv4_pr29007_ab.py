#!/usr/bin/env python3
"""v48: DeepSeek-V4-Flash-FP8 e2e A/B for PR #29007 (MoE TP allreduce via NCCL
symmetric memory / in-pool output allocation) on 8×H200 TP8.

Both arms run with --enable-symm-mem; the ONLY difference is whether the PR patch is
applied (MoE output allocated in the symmetric pool so the downstream TP all-reduce
takes the low-latency NCCL symmetric path). Baseline = stock v0.5.15.post1; patched =
PR #29007 ported to v0.5.15.post1 (5 pure-Python files; see patches/pr29007/).

The server lifecycle is handled by the caller (bash); this script just drives
bench_serving against an already-running server and records metrics. Metric focus:
mean/median TPOT + output throughput + E2E (the PR reported mean E2E -6.58%, output
tput +7.05%, TPOT -6.55% on DeepSeek-V4-Flash-FP8 4K/1536/conc1).
"""
import argparse, json, os, subprocess, sys, time

ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-v515"
PY = f"{ENVDIR}/bin/python"
SGLANG = "/home/t-jialianggu/work/sglang-v515"
MODEL = "/home/t-jialianggu/work/models/DeepSeek-V4-Flash-FP8"
REPO = "/home/t-jialianggu/work/end2end-optimization"


def run_bench(arm, port, in_len, out_len, conc, nprompts, rep, outdir):
    resfile = f"{outdir}/{arm}_in{in_len}_out{out_len}_c{conc}_r{rep}.jsonl"
    if os.path.exists(resfile):
        os.remove(resfile)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SGLANG}/python"
    argv = [PY, "-m", "sglang.bench_serving", "--backend", "sglang",
            "--host", "127.0.0.1", "--port", str(port), "--model", MODEL,
            "--dataset-name", "random-ids",
            "--random-input-len", str(in_len),
            "--random-output-len", str(out_len),
            "--random-range-ratio", "1.0",
            "--max-concurrency", str(conc),
            "--num-prompts", str(nprompts),
            "--output-file", resfile]
    log = f"{REPO}/logs/v48_bench_{arm}_in{in_len}_out{out_len}_c{conc}_r{rep}.log"
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
    if os.path.exists(resfile):
        rows = [json.loads(l) for l in open(resfile) if l.strip()]
        return rows[-1] if rows else None, time.time() - t0
    return None, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["baseline", "patched"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    # cells: (in_len, out_len, conc, nprompts)
    ap.add_argument("--cells", type=str,
                    default="4096:1024:1:12,4096:1024:8:24,4096:512:16:32")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cells = []
    for c in args.cells.split(","):
        il, ol, cc, npr = [int(x) for x in c.split(":")]
        cells.append((il, ol, cc, npr))

    for (il, ol, cc, npr) in cells:
        for rep in range(args.repeats):
            r, wall = run_bench(args.arm, args.port, il, ol, cc, npr, rep, args.outdir)
            if r:
                row = {"arm": args.arm, "in_len": il, "out_len": ol, "conc": cc,
                       "nprompts": npr, "repeat": rep,
                       "mean_tpot_ms": r.get("mean_tpot_ms"),
                       "median_tpot_ms": r.get("median_tpot_ms"),
                       "mean_ttft_ms": r.get("mean_ttft_ms"),
                       "mean_e2e_ms": r.get("mean_e2e_latency_ms"),
                       "median_e2e_ms": r.get("median_e2e_latency_ms"),
                       "output_throughput": r.get("output_throughput"),
                       "wall_s": round(wall, 1), "ts": time.strftime("%H:%M:%S")}
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{args.arm}] in{il} out{ol} c{cc} r{rep}: "
                      f"TPOT={row['mean_tpot_ms']:.2f} E2E={row['mean_e2e_ms']:.0f} "
                      f"out_tput={row['output_throughput']:.1f} ({wall:.0f}s)", flush=True)
            else:
                print(f"[{args.arm}] in{il} out{ol} c{cc} r{rep}: FAILED", flush=True)


if __name__ == "__main__":
    main()
