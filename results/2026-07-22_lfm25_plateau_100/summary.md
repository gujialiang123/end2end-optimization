# Summary — LFM2.5 serving-knob autotuning plateau (clean, no warm start)

**Date:** 2026-07-22 · **Model:** LFM2.5-8B-A1B (bf16) · **HW:** 1×H200 · **Engine:** sglang 0.5.12.post1 (Triton MoE, FA3, CUDA graph on)
**Study:** fresh Optuna `TPESampler(seed=20260722, n_startup_trials=20, multivariate=True)`, **no warm start, no enqueue_trial, no cookbook injection**. 100 unique COMPLETE trials, 0 failures, 26 duplicates pruned (126 attempts).

## Headline

Starting from **no known-good configuration**, TPE discovers the useful region within **~7 configurations**. The final 20 configurations improve best-so-far throughput by **0.0%**. The validated best config is within measurement noise of the cookbook baseline (**+0.4%**, overlapping 95% CIs). This is a genuine plateau of the 4-knob serving search space for R_concurrent_decode — **not** evidence that all inference optimization is exhausted.

## Numbers

| quantity | value |
|---|---|
| cookbook baseline (separate 5-run reference) | **19.49 ± 0.59 req/s** (95% CI ±0.52) |
| best raw single-trial (trial_50) | 19.98 req/s |
| best **validated** (trial_41, 5 interleaved repeats) | **19.80 req/s → 1.004× (+0.4%)** |
| trial throughput spread (100 configs) | 7.09 → 19.98 req/s |
| first config within 1% of final best | **config 7** |
| best-through-10 / 20 / 50 / 75 / 100 | 19.89 / 19.89 / 19.89 / 19.98 / 19.98 |
| improvement in final 20 configs | **0.0%** |
| # of final-20 configs that improve best-so-far | **0** |
| fraction of configs within 1% / 3% / 5% of best | **0.42 / 0.70 / 0.72** |

## Validation (interleaved ×5, cookbook + top-5 raw)

| candidate | validated mean (req/s) | speedup vs cookbook |
|---|---:|---:|
| cookbook | 19.72 | 1.000× |
| trial_41 | **19.80** | **1.004×** |
| trial_26 | 19.72 | 1.000× |
| trial_7 | 19.70 | 0.999× |
| trial_30 | 19.65 | 0.997× |
| trial_50 (raw #1) | 19.64 | 0.996× |

Raw ranking `[50, 30, 7, 41, 26]` ≠ validated ranking `[41, 26, 7, 30, 50]` → **ranking is not stable**; the top configs differ only by noise. The single-trial "best" (trial_50) drops to last on re-validation.

## Interpretation

The only knob that matters in this regime is **not starving batching**: configs with `max_running_requests = 8` collapse to ~7 req/s. Once `max_running_requests ≥ ~24`, every config lands within a few % of the cookbook. TPE finds this within the 20-trial startup phase; the remaining 80 trials are diminishing returns.

## Justified claim for the slide

> "Without a seeded warm start, autotuning discovers the useful region after **7** configurations. The final **20** configurations improve the best-so-far throughput by only **0%**, indicating diminishing returns within this fixed serving-knob search space."

## Why this replaces the old v3 figure

The 2026-07-02 v3 curve was biased: trials 0–3 were **manually enqueued warm-start** configs already containing cookbook-equivalent settings, and the study jointly tuned MoE backend + CUDA-graph enablement (disabled-CUDA-graph configs created artificially poor points). This study removes all of that: single fixed MoE path (triton), CUDA graph always on, no warm start, 4 clean serving knobs.

## Artifacts
`study.db`, `per_trial_log.csv` (100 rows), `failures.csv` (empty), `baseline_reference.json`,
`best_raw.json`, `best_validated.json`, `validation_repeats.csv`, `plateau_stats.json`,
`convergence_req_throughput.{png,svg}`, `convergence_normalized.{png,svg}`,
`ttft_throughput_pareto.{png,svg}`, `environment.json`, `search_space.md`, `reproduce.sh`.
