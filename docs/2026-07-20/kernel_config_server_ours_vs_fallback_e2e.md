# v45 — Server-level e2e A/B: ours (re-tuned) vs fallback, all regimes + agent dataset

> 2026-07-21 · new host `aifx-clou000001`, 8×H200 · triton 3.6.0 / torch 2.11 / CUDA 13.0
> User request: "give me ours-vs-fallback across all inputs, including the agent dataset."
> This is the **server + bench_serving** counterpart of v44 (which used
> bench_one_batch), and the mirror of the remote v43 (which compared default
> heuristic vs fallback). Here both arms carry a real tuned config; we isolate the
> **marginal value of re-tuning for the current Triton version** under realistic serving.

## 0. TL;DR

**Across all 8 regimes (7 synthetic + the mooncake `toolagent` agent trace),
re-tuning the fused_moe config on triton 3.6.0 gives NO end-to-end gain over the
`triton_3_2_0` fallback sglang already loads.** Every regime sits within ±2% on
TTFT / TPOT / E2E / output-throughput, none of it a meaningful speedup. The only
"significant" cells are `decode_heavy` at ~+1% (tiny, low-variance) and a couple of
sub-2% regressions. The agent trace is a dead heat (TTFT −5% p=0.18, E2E −1%). This
matches v44 (bench_one_batch ≈0) and report §1.6.5 (isolated kernel +0.6%): all
three measurement layers agree.

## 1. Arms

Same config-placement mechanism verified in v44:
- **fallback** = stock sglang → `triton_3_6_0/` has no config for our shape
  `E=128,N=768,H200`, loader falls back to `triton_3_2_0` ("sub-optimal" warning).
- **ours** = our config re-tuned on triton 3.6.0 (v44's 18-bucket artifact) dropped
  into `triton_3_6_0/` so the loader picks it first.

## 2. Method

- `scripts/run_v45_server_ours_vs_fallback.py`: for each arm, launch one sglang
  server (`--attention-backend fa3 --moe-runner-backend triton --mem-fraction-static 0.85`,
  GPU 0), run `bench_serving` on each regime **3× repeats**, record median TTFT /
  TPOT / E2E latency / output throughput per run.
- Regimes mirror the remote v43 sweep (covers the full input space):

  | regime | in_len | out_len | conc | prompts | stresses |
  |---|---|---|---|---|---|
  | tiny_latency | 8 | 4 | 1 | 32 | launch overhead |
  | short_in_short_out | 128 | 32 | 16 | 128 | balanced small |
  | sched_overhead_hiconc | 128 | 16 | 64 | 256 | scheduler / high concurrency |
  | prefill_medium | 4096 | 16 | 4 | 48 | prefill-bound |
  | prefill_long | 16384 | 16 | 2 | 16 | long prefill |
  | decode_medium | 128 | 512 | 16 | 96 | decode-bound |
  | decode_heavy | 128 | 1024 | 32 | 96 | heavy decode |
  | **agent_toolagent** | mooncake toolagent trace | | 32 | 96 | **real agent load** |

- Stats: `scripts/analyze_v45_server_ab.py` — per regime, **median** across repeats
  (robust to warmup outliers) + Welch's two-sided t-test. gain% sign = ours better.
- Raw per-run rows: `results/2026-07-21_v45_server_ours_vs_fallback/server_ab.jsonl` (48 rows).

## 3. Results (median over 3 repeats; + = ours better)

| regime | TTFT | TPOT | E2E | out_tput |
|---|---|---|---|---|
| tiny_latency | −0.6% | −2.0%\* | +0.3% | −1.0% |
| short_in_short_out | −0.3% | −0.1% | −0.2% | −0.0% |
| sched_overhead_hiconc | +1.0% | +1.8% | +0.4% | +2.1% |
| prefill_medium | +0.7% | −1.3% | +0.3% | −0.2% |
| prefill_long | +4.4% (p=0.97) | −0.2% | +0.9% | +0.8% |
| decode_medium | +0.1% | −0.1%\* | −0.1% | −0.0% |
| decode_heavy | +1.2%\* | +1.3%\* | +1.3%\* | +1.0%\* |
| **agent_toolagent** | −5.0% (p=0.18) | +0.5% | −1.0% (p=0.31) | −1.2% |

(\* = p<0.05. Full t/p in `ab_analysis.txt`.)

## 4. Interpretation

- **No regime shows a meaningful e2e speedup.** Nothing exceeds ~+2% with
  significance; `decode_heavy`'s significant +1.1–1.3% is real but negligible (long
  low-variance run → large t on a tiny effect). A few sub-2% regressions
  (`tiny_latency` TPOT, `decode_medium` TPOT) are equally negligible.
- **The agent trace is a dead heat.** Median TTFT −5% (p=0.18), E2E −1% (p=0.31),
  TPOT +0.5%, out_tput −1.2% — all within noise. Note the raw ours-r0 was a
  cold-start outlier (TTFT 250 ms vs steady-state ~52 ms); using the median (not
  mean) correctly discounts it. Steady-state ours ≈ fallback (TTFT 53.9 vs 51.1,
  E2E 1226 vs 1214).
- **Consistent across all three measurement layers:**
  isolated kernel (§1.6.5) +0.6% · bench_one_batch (v44) ≈0 · server+agent (v45) ≈0.
- **The takeaway is unchanged and now airtight:** for a shape already covered by a
  fallback config, re-tuning per Triton version is an end-to-end no-op — even under
  a real agent workload. The entire config-tuning payoff (remote v42/v43: prefill
  +34~43%, agent +17.5%) comes from **not falling into the default heuristic**, not
  from re-tuning the fallback.

## 5. Harness fix (mooncake on sglang v0.5.12.post1)

`sglang.bench_serving` on this tag crashed on `--dataset-name mooncake` because two
sync code paths assume `input_requests[i]` is a `DatasetRow` while mooncake yields
raw trace dicts:
- line ~1230 `input_requests[0].prompt` → guarded with `hasattr`.
- line ~1464 `calculate_metrics(input_requests=...)` → pass `None` for mooncake
  (same as the multi-turn path); it only skips input-token accounting, leaving
  TTFT/TPOT/E2E/output-throughput intact.

These are in the sglang tree (independent clone), saved as
`patches/sglang_bench_serving_mooncake_v0.5.12.post1.patch` for reproduction.

## 6. Artifacts

- Raw: `results/2026-07-21_v45_server_ours_vs_fallback/server_ab.jsonl` (48 runs)
- Analysis: `results/2026-07-21_v45_server_ours_vs_fallback/ab_analysis.txt`
- Scripts: `scripts/run_v45_server_ours_vs_fallback.py`, `scripts/analyze_v45_server_ab.py`
- Per-run bench logs: `logs/v45_bench_*.log`; tuned config reused from v44.
