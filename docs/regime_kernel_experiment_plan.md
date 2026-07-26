# Regime-aware Kernel Specialization — experiment plan

**Scope:** 1–2 weeks, single H200, no new CUDA kernels, no serving-runtime
refactor. Default code paths stay unchanged; specialization is opt-in through
`SGLANG_MOE_CONFIG_DIR` (E2E) or `override_config()` (microbenchmark).

Status and feasibility audit: `docs/regime_kernel_status.md`.
Running log and results: `docs/regime_kernel_results.md`.

---

## 1. Hypothesis

> Different serving regimes produce different kernel workload shapes and routing
> patterns, so the optimal kernel configuration differs by regime on the same
> model and GPU — and an agent can select, tune, validate and deploy
> regime-specific kernel profiles.

The mechanism is concrete and already present in SGLang: the fused-MoE Triton
config is a map `M → kernel config`, selected by nearest M, where
`M = tokens × top_k`. Regimes differ by orders of magnitude in M.

## 2. Research questions and the experiment that answers each

| RQ | question | experiment | primary output |
|---|---|---|---|
| **RQ1** | do regimes produce distinct kernel workloads? | opt-in MoE trace over the three frozen workloads; record phase, M, tokens, active requests, per-expert counts, CV/Gini, latency, GPU-time share | `regime_workload_characterization` figure + `traces/*.parquet` |
| **RQ2** | does a config tuned on one regime degrade on another? | transfer matrix: every tuned profile × every M bucket, ≥5 repeats, median/p95 | `kernel_transfer_heatmap` + absolute latency table |
| **RQ3** | do a few regime profiles beat one global profile and approach a per-shape oracle? | strategy comparison: default / global-best / regime-aware (2–3 profiles) / oracle; profile-count sweep | `strategy_comparison` figure, % of oracle recovered, crossover M |
| **RQ4** | can diagnosis → candidate → correctness → benchmark → accept/reject be closed automatically? | budgeted controller with structured bottleneck rules over the trace + benchmark results | `agent_iteration_trace` + dispatch profile + rejected-candidate log |

## 3. Fixed experimental frame

| axis | value |
|---|---|
| hardware | 1× NVIDIA H200, TP1 |
| dtype | BF16 |
| software | sglang 0.5.12.post1 @ `17f7a1da1`, torch 2.9.1+cu128, Triton 3.5.1, CUDA 12.8, driver 580.105.08 |
| models | LFM2.5-8B-A1B (`E=32, N=1792, top_k=4`) and Qwen3-30B-A3B (`E=128, N=768, top_k=8`) |
| hot kernel | fused MoE Triton (`fused_experts`) |
| workloads | the frozen definitions in `results/2026-07-24_serving_ceiling/workloads.yaml` |
| seeds / sampling | fixed seed, `temperature=0`, `ignore_eos`, fixed output length |

### Regimes

| regime | workload | active batch | M = tokens × top_k (LFM / Qwen) |
|---|---|---:|---|
| **A. low-batch decode** | `R_short_decode` | 1 | 4 / 8 |
| **B. concurrent decode** | `R_concurrent_decode` | 32 | 128 / 256 |
| **C. long prefill** | `R_long_prefill` | 1–4 | thousands (chunk-dependent) |

Plus a light M sweep `1, 2, 4, 8, 16, 32, 64` (extended to the real prefill M
values) used only at kernel level to locate crossovers.

## 4. Kernel candidate pool

No new kernels. Candidates are configurations of the existing fused-MoE Triton
kernel, plus the existing alternative backends.

| id | candidate |
|---|---|
| **K0** | production default — exactly what the server does today (LFM: heuristic default; Qwen: triton-3.2.0 fallback) |
| **K1** | existing alternative backend already in the repo/runtime |
| **K2** | **low-M profile** — tuned at M ∈ {1,2,4,8} |
| **K3** | **high-M profile** — tuned at large M (concurrent decode and prefill) |
| **K4** | existing fusion candidate, only if already implemented in the repo |

Search space, pruned from upstream's 1920 points to a legality-filtered set:

```
BLOCK_SIZE_M  ∈ {16, 32, 64, 128}        (256 only for large M)
BLOCK_SIZE_N  ∈ {32, 64, 128, 256}
BLOCK_SIZE_K  ∈ {64, 128, 256}
GROUP_SIZE_M  ∈ {1, 16, 32}
num_warps     ∈ {4, 8}
num_stages    ∈ {2, 3, 4, 5}
```

with M-dependent pruning (large BLOCK_M is meaningless at M=4) and a shared-memory
legality filter, so invalid combinations are never launched.

## 5. Measurement protocol

* warm-up ≥ 20 iterations, ≥ 100 measured iterations, ≥ 5 independent repeats;
* report median, p95 and variance; all raw timings retained;
* CUDA events for timing, `torch.cuda.synchronize()` around each repeat;
* every deviation recorded in the run metadata;
* **correctness gates performance**: a candidate that fails numerical comparison
  against the default path (BF16 tolerance, NaN/Inf check, several seeds and real
  traced shapes) never enters the selector or any table.

Failure taxonomy kept distinct in `failures.csv`: correctness failure, runtime
failure, OOM, performance regression.

## 6. Routing control experiment

At **fixed M**, compare synthetic uniform routing, synthetic skewed routing, real
traced decode routing and real traced prefill routing. This separates "M decides
the winner" from "routing distribution also decides the winner". A null result
here is reported as a null result.

## 7. Strategy comparison (RQ3)

| strategy | definition |
|---|---|
| **Default** | current runtime behaviour |
| **Global-best** | one profile chosen over all regimes mixed, objective = geometric mean of per-regime normalized latency (weights recorded explicitly) |
| **Regime-aware** | 2–3 profiles, one per regime cluster |
| **Oracle** | best profile per individual shape — an upper bound, not deployable |

Reported: where global-best regresses, regime-aware gain over global-best,
fraction of oracle recovered, number of profiles needed, and the crossover M.

## 8. End-to-end stage (gated)

Run **only** if the microbenchmark shows a crossover or a regime-specific winner.

* static profile per server launch via `SGLANG_MOE_CONFIG_DIR`;
* serving knobs frozen at the **validated per-regime winner** from
  `results/2026-07-24_serving_ceiling_validation/` — so the only varying factor
  is the kernel profile;
* three arms per regime: tuned serving + default kernel / + global-best kernel /
  + regime-specific kernel;
* metrics: TTFT, TPOT, request latency, request and output-token throughput,
  wall time, p50/p95, GPU kernel time and hot-kernel GPU share;
* warm-up + ≥5 valid repeats, bootstrap 95 % CI, raw data retained.

Final artifact: a **waterfall** — cookbook → regime-specific serving tuning →
serving tuning + kernel specialization — to show whether the two levels are
complementary.

**Cost gate:** E2E is estimated at 4–6 GPU-hours, above the 2 GPU-hour
threshold, so its exact scope is re-reported in `docs/regime_kernel_results.md`
before launch.

## 9. Minimal agent closed loop (P1)

Input: workload trace, model/hardware metadata, candidate pool, budget.
Loop: analyse trace + profiler summary → identify hot kernel → classify
bottleneck (launch-bound / memory-bound / compute-bound / low occupancy /
routing imbalance) → choose an action class (tune config, switch backend, pick
low-M or high-M specialization) → generate candidates → correctness → benchmark
→ accept/reject/rollback → update history.
Output: selected profile, dispatch table, performance and correctness summaries,
attempted and rejected candidates with reasons, remaining gap to oracle.

The controller is rule-based over structured diagnoses; it is only called an
agent because it owns the decision and the accept/reject, not because it wraps a
parameter sweep.

## 10. Deliverables

```
scripts/regime_kernel/     collect_traces · tune · transfer · select · e2e · agent · plots
configs/regime_kernel/     generated kernel profiles (JSON, SGLANG_MOE_CONFIG_DIR layout)
results/regime_kernel/raw/        per-run raw timings, traces, correctness logs
results/regime_kernel/processed/  tidy tables consumed by plotting
analysis/regime_kernel/    strategy comparison, transfer matrices
docs/regime_kernel_results.md     running log: commands, files, results, blockers, next
```

Every entry point supports `--dry-run`, prints the exact command it will run,
snapshots config + git commit + environment, is resume-safe, and never
overwrites existing results.

Required figures: regime workload characterization · kernel transfer heatmap ·
kernel winner map over M (and routing skew) · default vs global vs regime-aware
vs oracle · E2E waterfall · agent iteration trace. All plotted from processed
data; no numbers hard-coded in plotting code.

## 11. Priorities

**P0 (week 1):** audit ✓ → trace 3 regimes → ≥3 runnable profiles → correctness
framework → microbenchmark sweep → transfer matrix → strategy comparison → ≥1
regime static E2E → figures + report.

P0 succeeds if we can state definitively whether a kernel crossover exists, every
performance number is correctness-backed, microbenchmark and E2E speedups are
clearly separated, and at least one regime shows a stable kernel-level
improvement **or** a credible negative result.

**P1 (week 2):** runtime selector · CUDA-graph-compatible bucket dispatch · full
agent loop · second workload family (shared-prefix / agentic) · deeper NCU.

Explicitly out of scope for P0: writing new CUDA kernels, refactoring the serving
runtime, re-running old autotuning, any performance search without correctness,
and presenting synthetic microbenchmarks without real traces.
