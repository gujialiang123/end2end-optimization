# v44 — Kernel-config tuning, layer 3: re-tune vs the fallback sglang actually loads (e2e)

> 2026-07-20 (new host `aifx-clou000001`, 8×H200) · triton **3.6.0** / torch 2.11 / CUDA 13.0
> This is the **third and final layer** of the kernel-config e2e story, run on the
> migrated machine. It closes a specific sub-gap that report §1.6.3 asserted but
> never measured end-to-end.

## Where this fits (relative to the already-pushed v42/v43)

Three distinct A/B comparisons, three distinct questions:

| exp | comparison | question | result |
|---|---|---|---|
| **v42/v43** (remote, §1.7/§1.8) | **default heuristic** vs tuned/fallback config | is having *any* tuned config worth it? | **YES**: prefill **+34~43%** e2e, agent load **+17.5%** |
| **v44** (this doc) | fallback config vs **ours re-tuned** on triton 3.6.0 | once you already load the fallback, does re-tuning per Triton version add more? | **NO**: e2e ≈0 |

So the full picture is: **the entire config-tuning payoff comes from not falling
into the default heuristic** (v42/v43). For a shape that already has a decent
version-mismatched fallback (ours: `triton_3_2_0` covers `E=128,N=768,H200`),
**re-tuning it for the current Triton version buys nothing end-to-end** — this doc.
Report §1.6.3 stated this from a single isolated-kernel number (ours vs fallback
+0.6% at b=32); v44 confirms it across decode+prefill with real e2e latency/throughput.

## 0. TL;DR

**Re-tuning the `fused_moe` Triton meta-config for our uncovered Qwen3-30B-A3B
shape (`E=128,N=768,H200`) on triton 3.6.0 yields NO measurable end-to-end gain
over the config sglang already falls back to.** Of 7 cells (3 decode + 4 prefill),
none show a significant *speedup*; the only two statistically-significant cells are
tiny *regressions* (decode b=1 −1.64%, prefill in=1024 −2.75%), everything else is
within noise. This confirms the earlier isolated-kernel finding (ours vs fallback
was only +0.6% at b=32) and answers the open question with real e2e evidence:
**for this shape, "re-tune the config per Triton version" is an e2e no-op.**

## 1. What baseline vs ours actually are

sglang looks up `fused_moe` configs keyed by Triton version. On this host
(`triton 3.6.0`) the directory `triton_3_6_0/` has **no** config for our shape
`E=128,N=768,device_name=NVIDIA_H200.json`, so the loader walks back and loads
`triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json`, printing
*"Performance might be sub-optimal!"*.

- **baseline** = stock sglang → loads the `triton_3_2_0` fallback (18 batch buckets).
- **ours** = we ran sglang's official tuning script
  (`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, 1920 configs ×
  18 buckets, 8-GPU ray, 2149 s) **on triton 3.6.0** for this exact shape, and
  dropped the result into `triton_3_6_0/` so the loader picks it first (version match).

Config load was verified from the server log:
`Using MoE kernel config from .../configs/triton_3_6_0/E=128,N=768,device_name=NVIDIA_H200.json`
(the residual "sub-optimal" line refers only to the *down_moe* auxiliary config,
which neither arm has — so it is constant across the A/B and does not affect the comparison).

### Config diff (selected buckets)

| bucket | fallback (triton_3_2_0) | ours (triton_3_6_0 retuned) |
|---|---|---|
| b=1  | M16 N64 K64  G1  w4 s5 | M16 N32 K256 G64 w4 s3 |
| b=8  | M16 N64 K128 G1  w4 s3 | M16 N64 K128 G32 w4 s3 |
| b=32 | M16 N64 K128 G16 w4 s2 | M16 N64 K128 G64 w4 s3 |

(The b=32 ours config is identical to what the old machine produced on triton 3.5.1,
differing from fallback only in GROUP_SIZE_M and num_stages — hence the ~0.6% isolated delta.)

## 2. Method

- Harness: `scripts/run_v44_e2e_config_ab.py` → launches `sglang.bench_one_batch`
  (tp=1, GPU 0, `--attention-backend fa3 --moe-runner-backend triton`,
  `--mem-fraction-static 0.85`). Each run is an independent process; bench_one_batch
  itself warms up (runs twice) and we parse the **measured** (2nd) pass.
- **decode** cells: batch ∈ {1,8,32}, input-len 256, output-len 64 → metric =
  median decode-step latency (TPOT proxy), lower is better. **n=8** repeats.
- **prefill** cells: batch 1, input-len ∈ {512,1024,2048,4096}, output-len 8 →
  metric = prefill throughput (tok/s), higher is better. **n=3** repeats.
- Stats: per cell, median + Welch's two-sided t-test (`scripts/analyze_v44_config_ab.py`).
- Raw per-run rows: `results/2026-07-20_v44_retune_e2e_ab/e2e_ab.jsonl` (72 rows).

Measurement-discipline note (Chendi standard): an initial n=3 pass showed decode
b=8 at −8.84% which looked alarming; extending to n=8 collapsed it to +0.91%
(p=0.93) — it was pure noise. This is exactly why signal-vs-noise needs repeats + a t-test.

## 3. Results

| cell | metric (dir) | baseline med | ours med | Δ% (ours) | t | p | verdict |
|---|---|---|---|---|---|---|---|
| decode b=1  | decode lat (↓) | 0.00427 s | 0.00434 s | **−1.64%** | 13.39 | <0.0001 | ours slightly **slower** (real, ~0.07 ms) |
| decode b=8  | decode lat (↓) | 0.00606 s | 0.00600 s | +0.91% | −0.09 | 0.93 | noise |
| decode b=32 | decode lat (↓) | 0.00797 s | 0.00827 s | −3.76% | 0.85 | 0.41 | noise |
| prefill in=512  | tput (↑) | 27132 tok/s | 26916 tok/s | −0.79% | −0.13 | 0.91 | noise |
| prefill in=1024 | tput (↑) | 41164 tok/s | 40032 tok/s | **−2.75%** | −5.76 | 0.028 | ours slightly **slower** (real) |
| prefill in=2048 | tput (↑) | 54176 tok/s | 54513 tok/s | +0.62% | 0.37 | 0.74 | noise |
| prefill in=4096 | tput (↑) | 59353 tok/s | 60331 tok/s | +1.65% | 0.32 | 0.77 | noise |

(Δ% sign convention: positive = ours faster. decode uses latency, prefill uses throughput.)

## 4. Interpretation

- **No end-to-end win.** Not a single cell shows a significant speedup for the
  re-tuned config. The two significant cells are both small *regressions*.
- **Why:** the `triton_3_2_0` fallback is already a well-tuned config for this
  shape; re-tuning on 3.6.0 lands on essentially equivalent meta-params (the
  isolated-kernel delta was only +0.6% at b=32). At the e2e level, the fused_moe
  GEMM is one of several memory-bound decode ops (MoE ~41%, dense GEMM ~32%,
  attention ~16% per the v33 decode audit), so even a real isolated-kernel edge is
  diluted; here there is no edge to dilute.
- **Relation to §1.6 / v42-v43.** §1.6's headline "+35~54% (Qwen) / +47~67%
  (DeepSeek)" is **isolated kernel time of tuned-config vs the *default heuristic***,
  and the remote v42/v43 already confirmed that gap converts e2e (**prefill +34~43%**,
  agent **+17.5%**). But sglang never runs the default heuristic for *this* shape —
  it runs the fallback. v44 measures the remaining question — fallback vs a fresh
  re-tune — and finds the e2e delta is ≈0.
- **Where config-tuning does matter** (unchanged from prior conclusions): shapes
  with **no** fallback at all (default heuristic path) — there the "add a tuned
  config" PR class is a real double-digit e2e win (v42/v43). Our shape simply isn't
  one of those, because triton_3_2_0 covers it, so re-tuning it is an e2e no-op.

## 5. Artifacts

- Tuned config (triton 3.6.0, all 18 buckets, upstream-PR-ready):
  `results/2026-07-20_v44_retune_e2e_ab/E=128,N=768,device_name=NVIDIA_H200.json`
- Tuning log: `results/2026-07-20_v44_retune_e2e_ab/tune.log` (2149 s)
- Raw e2e rows: `results/2026-07-20_v44_retune_e2e_ab/e2e_ab.jsonl` (72 runs)
- Per-run stdout summaries: `ab_baseline.log`, `ab_ours.log`, `ab_baseline_sup.log`, `ab_ours_sup.log`
- Analysis table: `results/2026-07-20_v44_retune_e2e_ab/ab_analysis.txt`
- Scripts: `scripts/run_v44_e2e_config_ab.py`, `scripts/analyze_v44_config_ab.py`

## 6. Environment (reproduce)

- Host `aifx-clou000001`, 8×H200; conda env `sglang-dev` (python 3.12).
- sglang **v0.5.12.post1** editable @ `/home/t-jialianggu/work/sglang`.
- **triton 3.6.0 / torch 2.11.0+cu130 / CUDA 13.0** (older machine used triton 3.5.1;
  the fallback baseline is identical either way — both walk back to triton_3_2_0).
- Env gotchas (else `import sglang` aborts): pin `kernels==0.12.3` (sglang leaves it
  unpinned → pip pulls 0.16 which breaks transformers 5.6.0), and set
  `CUDA_HOME=$CONDA_PREFIX` with a real nvcc (deep_gemm JIT-builds a `_C` module at import).
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` at
  `/home/t-jialianggu/work/models/Qwen3-30B-A3B-Instruct-2507`.
