# Does the serving objective change which configuration wins?

**Date:** 2026-07-26
**Short answer: yes, decisively — and the objective also changes whether the
"win" is real.**

| | |
|---|---|
| coverage grid | 2 models × 192 configs × 6 workloads, warmed, already measured |
| new validation | **58 configurations × 6 workloads × 5 repetitions** (1 740 runs), 0 unresolved failures |
| reused validation | the compatible subset of the original 62-config pass |
| statistics | bootstrap 95 % CI (8 000 resamples) on every metric vs cookbook |

Artifacts: `results/2026-07-26_alternative_objectives/`
Audit and plan: `docs/2026-07-26/alternative_objective_validation_audit.md`

Sign convention — **positive always means better**:
`throughput = cand/base − 1`, `latency = 1 − cand/base`.

---

## 0. The run window is trustworthy

The cookbook (config 74) was re-measured as an interleaved anchor inside the new
run window. It reproduces the original validation baseline almost exactly:

| model | workload | new anchor (req/s) | original validation | ratio |
|---|---|---:|---:|---:|
| lfm25 | short decode | 1.69 ± 0.00 | 1.68 | 1.004 |
| lfm25 | medium balanced | 7.13 ± 0.00 | 7.11 | 1.002 |
| lfm25 | long prefill | 12.78 ± 0.11 | 12.60 | 1.014 |
| lfm25 | concurrent decode | 21.88 ± 0.15 | 21.99 | 0.995 |
| lfm25 | shared-prefix | 14.06 ± 0.36 | 14.08 | 0.998 |
| lfm25 | tool-agent | 5.26 ± 0.01 | 5.26 | 1.000 |
| qwen | short decode | 0.88 ± 0.00 | 0.88 | 1.000 |
| qwen | medium balanced | 4.67 ± 0.00 | 4.65 | 1.003 |
| qwen | long prefill | 11.72 ± 0.52 | 11.24 | 1.043 |
| qwen | concurrent decode | 12.03 ± 0.02 | 12.03 | 1.000 |
| qwen | shared-prefix | 14.81 ± 0.06 | 14.82 | 1.000 |
| qwen | tool-agent | 4.86 ± 0.00 | 4.86 | 1.000 |

No time-window drift, so every classification below is attributable to the
configuration, not to the machine.

---

## 1. Do different objectives select different configurations?

**Yes.** Across the 12 model × workload cells the eight policies select on
average **4.8 distinct configurations**, up to **7** (qwen tool-agent).

Figure: `plots/config_role_matrix.png`.

## 2. How often is the throughput winner also the latency winner?

| comparison | cells where the same config wins both |
|---|---:|
| request throughput **and** TTFT p95 | **3 / 12** |
| request throughput **and** E2E p95 | **3 / 12** |
| request throughput **and** TPOT p95 | **0 / 12** |

**The throughput winner is never the TPOT winner.** This is not a coincidence:
`tpot_p95_best` repeatedly selects `max_running_requests = 8`. Starving the
batch minimises per-token latency and wrecks throughput — the exact mirror image
of the throughput-first failure mode:

| model | workload | TPOT winner | cap | throughput Δ | TPOT p95 Δ | class |
|---|---|---:|---:|---:|---:|:--|
| lfm25 | shared-prefix | cfg 8 | 8 | **−44.9 %** | +46.8 % | TRADE-OFF |
| lfm25 | concurrent decode | cfg 5 | 8 | **−64.1 %** | +24.3 % | TRADE-OFF |
| qwen | shared-prefix | cfg 9 | 8 | **−63.7 %** | +27.4 % | TRADE-OFF |
| qwen | concurrent decode | cfg 4 | 8 | **−61.9 %** | +30.5 % | TRADE-OFF |
| qwen | tool-agent | cfg 5 | 8 | −11.8 % | +7.5 % | TRADE-OFF |

If you hand the tuner a TPOT objective without a throughput guardrail, it will
happily give away two thirds of your throughput.

## 3. Where does a validated all-metric winner exist?

**10 / 12 cells** contain at least one configuration that, over 5 repetitions,
significantly improves at least one metric while significantly regressing none
(`STRICT_ALL_METRIC_WIN`).

The two exceptions — where **every** improvement is a trade-off — are
**lfm25 short decode** and **qwen tool-agent**.

Figure: `plots/no_regression_feasibility.png`.

Caveat that matters: "all-metric win" is a statistical statement, not a large
one. Most of these wins are small; the genuinely large ones are concentrated in
the two cliff regimes (long-prefill, shared-prefix). The largest validated
all-metric win is lfm25 long-prefill, **+63.1 % request throughput and +55.5 %
TTFT p95** (Pareto-knee config 59).

## 4. Which policy is most reliable?

This is the most actionable result of the study.

| policy | STRICT_ALL_METRIC_WIN | WIN | TRADE-OFF | FLAT | **REGRESSION** |
|---|---:|---:|---:|---:|---:|
| **maximin balanced** | **8** | 1 | 1 | 1 | **2** |
| strict all-metric | 7 | 1 | 1 | 0 | 4 |
| E2E p95 first | 6 | 0 | 3 | 2 | 1 |
| SLO-constrained throughput (3 %) | 6 | 1 | 1 | 0 | 4 |
| Pareto knee | 5 | 0 | 5 | 1 | 1 |
| **request throughput first** | 4 | 1 | 2 | 0 | **5** |
| TTFT p95 first | 4 | 1 | 2 | 1 | 4 |
| **TPOT p95 first** | 2 | 1 | **6** | 1 | 2 |

**The maximin policy dominates throughput-first**: twice as many clean wins
(8 vs 4) and less than half the regressions (2 vs 5). Optimising the *worst*
relative metric is markedly more robust than optimising the *best* one.

Figure: `plots/outcome_counts_by_objective.png`.

## 5. Why do coverage winners regress under validation?

Because a single coverage measurement can select a point that repeated
measurement does not support. `request_throughput_best` regresses in 5 of 12
cells — the coverage grid's top point was not reproducibly the top point.

This is *not* the pre-warm-up steady-state defect (that was fixed, and the anchor
above confirms the run window). It is ordinary selection-on-noise: picking the
argmax of 192 noisy measurements systematically over-estimates that point. It is
the quantitative reason the campaign has a separate validation stage at all, and
the reason **no serving-tuning result should ever be published from a
single-measurement grid search**.

The regression concentrates in the saturated regimes, where the true spread
between configurations is ~1 % and therefore comparable to the noise. In the
cliff regimes (long-prefill, shared-prefix), where the real effect is 20–90 %,
the coverage winner survives validation comfortably.

## 6. Regime taxonomy (validated)

| type | regimes | evidence |
|---|---|---|
| **configuration cliff** — large real headroom | lfm25 long-prefill (+63 %), lfm25 shared-prefix, qwen long-prefill (+20 %), qwen shared-prefix | validated all-metric or large trade-off wins |
| **serving-saturated** — nothing reliable to gain | lfm25 medium-balanced, qwen medium-balanced, lfm25/qwen short-decode | winners regress or go flat under validation; ~1 % true spread |
| **trade-off dominated** — you must choose | qwen tool-agent, lfm25 short-decode, both concurrent-decode cells for latency objectives | every improvement costs another metric |

## 7. Does changing the objective change the main conclusion?

**No — it strengthens it.**

Changing the serving objective changes which configuration is selected, and the
full-grid data lets us choose a throughput-, latency-, or balanced operating
point **without rerunning the search**. But:

* in 2 of 12 cells no objective yields a non-trade-off improvement;
* in the saturated regimes the winners of every objective collapse to FLAT or
  REGRESSION under repetition;
* the large validated wins are confined to the two regimes where the cookbook is
  structurally mismatched, and they are cliff repairs, not frontier motion;
* every latency-first policy pays for its gain in throughput unless a guardrail
  is imposed.

> **Serving tuning selects an operating point on the existing frontier.
> Changing the objective moves you along that frontier — it does not move the
> frontier outward. Profiling and kernel/backend optimization are required for
> that.**

---

## 8. Surprises and contradictions, reported honestly

1. **`strict_all_metric_candidate` regresses in 4 cells.** A configuration whose
   five coverage benefit ratios were all ≥ 1.00 can still fail validation,
   because ratios of 1.001 are noise. The coverage-grade feasibility count
   (12/12) overstates reality; the validated count is 10/12 and comes from
   *different* roles.
2. **The Pareto knee is the most trade-off-prone policy** (5 of 12 cells).
   Choosing the point closest to the utopia corner deliberately balances
   metrics, which in practice means accepting a small loss somewhere.
3. **The geometric-mean policy can hide a severe regression**, which is why
   `min_benefit_ratio` is reported next to it in every output file. Use maximin
   instead when a guarantee is wanted.
4. **The existing 62-config validation subset was largely unusable** for this
   study (only 14 of 76 selected pairs had 5 valid repetitions), because it had
   been chosen from the pre-warm-up grid. Selection sets are not transferable
   across a measurement-protocol change.

---

## 9. Files

| file | contents |
|---|---|
| `candidate_validation_audit.{csv,json}` | every role × cell: config, knobs, all coverage metrics, five benefit ratios, maximin/geometric scores, validation status |
| `validation_plan.{csv,md,json}` | which configs needed runs and why |
| `objective_winners_validated.csv` | validated means, deltas, 95 % CIs, classification |
| `objective_comparison_matrix.csv` | slide-ready table for the key roles |
| `objective_winners_validated.md` | per-cell markdown tables |
| `outcome_counts_by_objective.csv` | the policy-reliability table above |
| `selected_config_knobs.csv` | knob values of every selected configuration |
| `per_run_metrics_a*.csv`, `raw/` | new validation runs |
| `plots/*.{png,svg}` | 17 figures |

Scripts: `scripts/analyze_alternative_serving_objectives.py`,
`scripts/run_alternative_objective_validation.py`,
`scripts/finalize_alternative_objectives.py`,
`scripts/render_alternative_objective_figures.py`.
