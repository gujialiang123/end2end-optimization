# Alternative-objective validation audit (Phases 1–3)

**Date:** 2026-07-26
**Question:** if the autotuner optimizes something other than request
throughput — TTFT, TPOT, E2E, or a balanced score — does it pick a *different*
configuration, and do we already have validated data for those picks?

**Method:** no new search. The warmed 192-point grid already measures every
configuration on every workload, so changing the objective only changes which
measured point is selected. Selection is re-run offline; only the missing
configurations are re-measured.

Inputs (verified to exist, warmed campaign only):

| file | rows |
|---|---|
| `results/2026-07-24_serving_ceiling/per_config_workload_metrics.csv` | 2 304 (2 models × 192 configs × 6 workloads) |
| `results/2026-07-24_serving_ceiling_validation/per_run_metrics.csv` | 1 860 (62 configs × 6 workloads × 5 reps) |

The un-warmed passes (`*_nowarmup/`) are **excluded**, per the data rules.

Baseline: `config_id 74` = `cap32_chunk-1_pollpm_mem0.85`.

Benefit-ratio convention (>1 always better):

```
r_req  = cand_request_throughput / cookbook_request_throughput
r_out  = cand_output_throughput  / cookbook_output_throughput
r_ttft = cookbook_ttft_p95 / cand_ttft_p95
r_tpot = cookbook_tpot_p95 / cand_tpot_p95
r_e2e  = cookbook_e2e_p95  / cand_e2e_p95
```

---

## 1. Headline finding — the objective really does change the answer

Across the 12 model × workload cells, eight objective policies select on
average **4.8 distinct configurations**, and up to **7 distinct** ones:

| model | workload | distinct configs chosen by the 8 objectives |
|---|---|---:|
| lfm25 | R_concurrent_decode | 3 |
| lfm25 | R_long_prefill | 3 |
| lfm25 | R_medium_balanced | 5 |
| lfm25 | R_short_decode | 6 |
| lfm25 | shared_prefix | 5 |
| lfm25 | tool_agent | 6 |
| qwen | R_concurrent_decode | 4 |
| qwen | R_long_prefill | 4 |
| qwen | R_medium_balanced | 4 |
| qwen | R_short_decode | 5 |
| qwen | shared_prefix | 5 |
| qwen | tool_agent | **7** |

Selected `config_id` per objective (coverage-grade selection):

| model | workload | cookbook | req-thr | TTFT p95 | TPOT p95 | E2E p95 | SLO-constr. 3 % | maximin | strict all-metric | Pareto knee |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lfm25 | concurrent decode | 74 | 100 | 125 | 5 | 100 | 100 | 100 | 100 | 100 |
| lfm25 | long prefill | 74 | 60 | 60 | 42 | 60 | 60 | 42 | 60 | 59 |
| lfm25 | medium balanced | 74 | 32 | 170 | 168 | 36 | 118 | 118 | 118 | 118 |
| lfm25 | short decode | 74 | 167 | 21 | 105 | 5 | 167 | 9 | 78 | 5 |
| lfm25 | shared-prefix | 74 | 152 | 152 | 8 | 152 | 106 | 83 | 106 | 179 |
| lfm25 | tool-agent | 74 | 105 | 41 | 17 | 191 | 68 | 68 | 68 | 25 |
| qwen | concurrent decode | 74 | 174 | 169 | 4 | 169 | 174 | 179 | 174 | 174 |
| qwen | long prefill | 74 | 120 | 98 | 71 | 98 | 120 | 120 | 120 | 137 |
| qwen | medium balanced | 74 | 165 | 165 | 72 | 83 | 165 | 83 | 102 | 83 |
| qwen | short decode | 74 | 102 | 139 | 67 | 39 | 102 | 39 | 102 | 98 |
| qwen | shared-prefix | 74 | 184 | 169 | 9 | 165 | 77 | 77 | 77 | 169 |
| qwen | tool-agent | 74 | 139 | 115 | 5 | 98 | 73 | 75 | 75 | 48 |

Immediate observations:

* **The throughput winner is never the TPOT winner.** In 12/12 cells the
  `tpot_p95_best` configuration differs from `request_throughput_best`.
* **TTFT and throughput sometimes agree** (lfm25 long-prefill 60/60, lfm25
  shared-prefix 152/152, qwen medium-balanced 165/165) — where the bottleneck is
  a single admission cliff, one configuration fixes both.
* **The SLO-constrained winner frequently differs from the raw throughput
  winner** (e.g. qwen shared-prefix 184 → 77, lfm25 shared-prefix 152 → 106),
  which is precisely the "throughput winner violates a latency guardrail" case.
* **`tpot_p95_best` repeatedly selects `cap = 8`-class configurations** (lfm25
  concurrent-decode 5, qwen concurrent-decode 4, qwen tool-agent 5). Starving the
  batch minimises per-token latency while destroying throughput — the mirror
  image of the throughput-first failure mode.

---

## 2. Selection policies implemented

Implemented in `scripts/analyze_alternative_serving_objectives.py`.

| # | role | rule |
|---|---|---|
| B | `request_throughput_best`, `output_throughput_best`, `ttft_p95_best`, `tpot_p95_best`, `e2e_p95_best` | pure single-metric optimum (p50 variants also recorded) |
| 1 | `constrained_throughput_best_{1,3,5}pct` | max request throughput s.t. output-thr ≥ 0.99×, and TTFT/TPOT/E2E p95 ≤ (1+tol)× cookbook |
| 2 | `constrained_ttft_best` | min TTFT p95 s.t. throughput guardrails ≥ 0.99× and other latencies ≤ 1.03× |
| 3 | `constrained_tpot_best` | min TPOT p95 under the same guardrail family |
| 4 | `constrained_e2e_best` | min E2E p95 under the same guardrail family |
| 5 | `maximin_balanced_best` | max `min(r_req, r_out, r_ttft, r_tpot, r_e2e)`; ties broken by geometric mean, then throughput, then config_id |
| 6 | `geometric_balanced_best` | max geometric mean of the five ratios; **`min_benefit_ratio` is always reported alongside**, because this policy can hide one severe regression |
| 7 | `strict_all_metric_candidate` / `noise_tolerant_all_metric_candidate` | all five ratios ≥ 1.00 (resp. ≥ 0.97); highest throughput among them; emits `NO_FEASIBLE_CONFIG` rather than silently relaxing |
| 8 | `pareto_knee_candidate` | five-metric non-dominated set, each ratio min–max normalised **over the Pareto set**, closest point to the utopia corner by Euclidean distance; number of ties recorded |

All tie-breaks end at `config_id` so the output is deterministic.

**Feasibility result:** a `strict_all_metric_candidate` exists in **12/12**
cells at coverage grade — no `NO_FEASIBLE_CONFIG`. Whether those survive
statistical validation is the question Phase 5 answers; a coverage-grade
all-metric win with ratios like 1.001 is well inside noise.

---

## 3. Validation coverage audit — the important negative

```
pure single-objective winners already validated:             6 / 34
new constrained/maximin/pareto candidates already validated:  8 / 42
unique configs requiring new runs:                           56
estimated H200 GPU-hours:                                    12.51
```

**The existing 62-configuration validation subset is largely NOT reusable for
this question, and the reason matters.** That subset was selected from the
*un-warmed* coverage grid, before the steady-state defect was found and fixed.
Once the grid was re-measured with warm-up, the winners moved — so the
previously validated configurations are, for many cells, no longer the ones any
objective selects. Only 14 of 76 selected (model, config) pairs already carry
five valid repetitions.

This is exactly why the plan's instruction to *verify rather than trust* the
methodology's claim about validation coverage was the right call.

Per-config detail, including knob values, all coverage metrics, all five benefit
ratios, maximin/geometric scores, repetition counts and status
(`ALREADY_VALIDATED` / `PARTIALLY_VALIDATED` / `NOT_VALIDATED`):

* `results/2026-07-26_alternative_objectives/candidate_validation_audit.csv`
* `results/2026-07-26_alternative_objectives/candidate_validation_audit.json`

---

## 4. Validation plan

Configurations are deduplicated by `(model, config_hash)`, because the canonical
harness measures **all six workloads from one server launch** — so one
validation of a configuration serves every workload that selected it.

* 56 unique configurations require 5 repetitions each;
* 2 cookbook anchors (one per model) are queued **first**, so the new run window
  can be checked against the original validation baseline CI;
* estimated 12.5 H200 GPU-hours ≈ 1.8 h wall-clock on 7 workers.

Plan files: `validation_plan.csv`, `validation_plan.md`, `validation_plan.json`.

Because 56 configurations is a modest, bounded set and the runner reuses the
already-validated canonical harness unchanged, the run was launched
automatically per the plan's allowance.

### Protocol (identical to the canonical campaign)

1× H200 · TP1 · BF16 · FA3 · MoE runner `auto` · CUDA Graph on · context 8192 ·
same sglang/Triton/PyTorch/CUDA build · same six workload definitions · same
streaming client (`sglang.bench_serving --output-details`) · **same per-workload
warm-up protocol** · same seeds and request payloads. Resolved knobs and CUDA
graph capture are re-verified from each server log. Infrastructure failures are
retried once and never fake-scored; partial benchmarks do not count as valid
repetitions.

---

## 5. What Phase 5 must still decide

Coverage-grade selection answers "which configuration would this objective
pick". It cannot answer "is that pick real". The validated pass will therefore
re-classify every role with 95 % confidence intervals into
`WIN` / `REGRESSION` / `TRADE-OFF` / `FLAT` / `STRICT_ALL_METRIC_WIN`, and in
particular test whether the 12/12 coverage-grade strict all-metric candidates
survive — the honest expectation is that several will collapse to `FLAT`.
