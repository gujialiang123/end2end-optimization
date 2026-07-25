# Slide claims — provenance and limits

For every claim that appears on slides 1–6 of the performance-gap deck, this
document records the exact wording, the source file and fields, the calculation,
the limitations, and whether it is **direct evidence** or an **inference**.

Deck draft: `results/2026-07-24_serving_ceiling/performance_gap_slides_1to6_draft.pptx`
Generator: `scripts/update_performance_gap_slides.py`

Sign conventions used everywhere:

```
throughput improvement = candidate / baseline − 1
latency    improvement = 1 − candidate / baseline     (positive == lower latency)
```

---

## C1 — "192 serving configurations per model, six workload regimes"

* **Slides:** 1, 4
* **Source:** `results/2026-07-24_serving_ceiling/search_space.yaml`,
  `workloads.yaml`, `campaign.db` (384 tasks, all `done`)
* **Calculation:** 8 `max_running_requests` × 3 `chunked_prefill_size` × 2
  `schedule_policy` × 4 `mem_fraction_static` = 192; × 2 models = 384 tasks;
  each task runs all six workloads ⇒ 2 304 per-run measurements.
* **Limitation:** `chunked_prefill_size = 8192` is effectively identical to `−1`
  at `--context-length 8192`, so the space contains 2 genuinely distinct
  chunking levels, not 3. This is stated in the speaker notes.
* **Type:** direct evidence.

## C2 — "The cookbook is a fair baseline, measured under the identical protocol"

* **Slide:** 3
* **Source:** `baseline_definition.json` (config_id 74,
  `cap32_chunk-1_pollpm_mem0.85`), `per_config_log.csv`
* **Calculation:** none; the cookbook configuration is one of the 192 grid
  points and is measured by the same harness, same client, same seeds.
* **Limitation:** the cookbook values used are the LFM2.5 cookbook serving knobs
  applied to both models; Qwen has no separate published cookbook cell in
  `sglang_cookbook_deployment_baselines.md` for this GPU/dtype combination.
  It is therefore a *common reference configuration*, and the slide says
  "frozen baseline", not "the official Qwen cookbook".
* **Type:** direct evidence.

## C3 — "Our earlier 5–9× was a CUDA-Graph-disabled artifact; against the true default it is 1.00–1.05×"

* **Slide:** 3
* **Source:** `docs/2026-06-25/autotuning_honest_results.md` (baseline table A/B/C)
* **Calculation:** C/B, where B is the strict zero-flag default (CUDA graph ON)
  and C the Optuna winner. `R_short_decode` 0.886/0.888 = 1.00×;
  `R_medium_balanced` 4.652/4.629 = 1.00×.
* **Limitation:** historical study, non-streaming client, different sglang
  branch. Quoted only as a methodological warning, never merged into the
  canonical matrix.
* **Type:** direct evidence (for the retraction), inference (for "baselines must
  be frozen").

## C4 — "Where the cookbook matches the traffic, the ceiling is ≈1 %"

* **Slide:** 5
* **Source:** `results/2026-07-24_serving_ceiling/summary_matrix.csv`, rows
  `R_medium_balanced` and `R_long_prefill`, column `d_request_throughput`
* **Values:** LFM2.5 +1.1 % / +0.5 %; Qwen +0.7 % / +0.5 %. Supporting columns:
  `pct_dominated_by_cookbook` = 90.1 % / 97.9 % (LFM) and 69.8 % / 97.9 % (Qwen).
* **Calculation:** best of 192 configurations divided by cookbook, minus 1.
* **Limitation:** coverage-pass values are single measurements; the ±1 % figure
  is inside the noise band, which is exactly the point — it is reported as
  **FLAT / inconclusive**, not as a gain. Validation repeats confirm the band.
* **Type:** direct evidence.

## C5 — "Where it is mismatched, the cliff is large: shared-prefix +78.6 % (LFM2.5) / +27.7 % (Qwen)"

* **Slide:** 5
* **Source:** `summary_matrix.csv`, row `shared_prefix`
* **Values:** `d_request_throughput` 0.786 / 0.277; `d_ttft_p95` 0.840 / 0.796;
  `d_tpot_p95` −0.215 / −0.317.
* **Winning knobs:** `cap96 · chunk2048 · lpm · mem0.90` (LFM) and
  `cap96 · chunk2048 · fcfs · mem0.90` (Qwen).
* **Limitation:** **this is a multi-knob configuration.** It must not be
  described as "chunked prefill gives +78.6 %". Both winners raise the admission
  cap from 32 to 96 *and* set chunking to 2048 *and* raise the memory fraction.
* **Type:** direct evidence.

## C6 — "The shared-prefix winner is a trade-off, not a free win"

* **Slide:** 5
* **Source:** same row; `d_tpot_p95` = −21.5 % (LFM) and −31.7 % (Qwen)
* **Calculation:** classification rule — one primary metric improves
  (throughput, TTFT p95) while another worsens (TPOT p95) ⇒ TRADE-OFF.
* **Limitation:** a trade-off is only claimed because both directions are large
  and consistent across two models; noise-level differences are never labelled
  trade-offs.
* **Type:** direct evidence.

## C7 — "Honest negative: the LFM tool-agent throughput winner gains +0.5 % and costs 82.6 % of TPOT p95"

* **Slide:** 5
* **Source:** `summary_matrix.csv`, row `tool_agent`, model `lfm25`
* **Values:** `d_request_throughput` +0.0046, `d_tpot_p95` −0.826 ⇒ REGRESSION.
* **Limitation:** this is the *throughput-selected* configuration; the
  lowest-TPOT configuration for the same regime is different and is reported in
  `lowest_tpot_p95_config_id`. The claim is about the danger of optimising a
  single objective, not that the regime is untunable.
* **Type:** direct evidence.

## C8 — "The downside is far larger than the upside"

* **Slide:** 5 (speaker notes) / 2 (gain-distribution chart)
* **Source:** `summary_matrix.csv` column `worst_vs_cookbook`;
  `analysis/*/gain_distribution.csv`
* **Values:** worst configuration loses 60.3 % (LFM long-prefill), 67.8 % (LFM
  concurrent decode), 72.4 % (Qwen long-prefill), 70.6 % (Qwen concurrent
  decode), against best gains of +0.5 % to +2.9 %.
* **Calculation:** min over the 192 configurations, divided by cookbook, −1.
* **Limitation:** the cliff is dominated by `max_running_requests = 8`; naming
  that mechanism is an **inference** supported by the v48 plateau study, which
  isolated the same driver independently.
* **Type:** direct evidence (the magnitudes), inference (the mechanism).

## C9 — "A regime winner can be catastrophic elsewhere: 0.36× under concurrent decode"

* **Slide:** 6
* **Source:** `analysis/lfm25/transfer_matrix_request_throughput.csv`, row
  `R_long_prefill_winner`, column `R_concurrent_decode`
* **Calculation:** value / target-regime cookbook value. Every cell is a real
  measurement, because a single server launch evaluates all six workloads.
* **Limitation:** the long-prefill winner is `cap = 8`, so this is largely the
  admission cliff seen from the transfer direction; it is not evidence that
  every winner is dangerous, which is why the full matrix is shown.
* **Type:** direct evidence.

## C10 — "Off-diagonal transfer is almost never > 1.00×"

* **Slide:** 6
* **Source:** the five transfer matrices under `analysis/*/`
* **Limitation:** the `shared_prefix` **column** is the systematic exception —
  almost any higher-capacity configuration beats the cookbook there, because the
  cookbook is badly mismatched for that regime. The slide shows the whole matrix
  so this exception is visible rather than hidden.
* **Type:** direct evidence.

## C11 — "The reachable gain is regime-dominated, not model-dominated"

* **Slide:** 5 / `cross_model_same_strategy.png`
* **Source:** `summary_matrix.csv`, both models
* **Reasoning:** both models agree on which regimes are saturated
  (long-prefill, medium-balanced: +0.5 %…+1.1 %) and which has a cliff
  (shared-prefix), but differ ~3× in the size of that cliff (78.6 % vs 27.7 %).
* **Limitation:** raw values for the two models are never merged; only ratios
  against each model's own cookbook are compared. Two models is not a
  population — this is a consistent observation, not a law.
* **Type:** inference from direct measurements.

## C12 — "High-concurrency stress: throughput 1.40–2.44× with TTFT p50/p95 −85…−96 %"

* **Slide:** 5 (supporting panel)
* **Source:** `results/2026-07-23_high_concurrency_ttft_rerun/comparison.md`,
  `summary.csv`
* **Limitation:** **different workload definition** (C64/O512, C128/O256) and a
  separate campaign. It is labelled a high-concurrency *stress study*, never a
  cell of the canonical six-regime matrix. Its cap-only ablation shows the gain
  is dominated by `max_running_requests` (full-vs-cap residual −2.0 %…+0.7 %).
* **Type:** direct evidence, from a separate study.

## C13 — "The same chunking candidate helps LFM2.5 shared-prefix and is neutral on Qwen"

* **Slide:** 5 (supporting panel)
* **Source:** `results/consolidated_v7_config_sweep.csv`
* **Values:** LFM2.5 shared-prefix 14.09→18.12 req/s (+28.6 %), median TTFT
  2 758→1 287 ms (−53.3 %); Qwen shared-prefix −2.9 % req/s, +1.3 % TTFT.
* **Limitation:** **n = 1 per cell**, so no confidence interval. Used for
  direction only. Independently corroborated by C5/C11 in this campaign.
* **Type:** direct evidence (single-run), corroborated.

## C14 — "Serving search selects points on the frontier; profiling is needed to move it"

* **Slide:** 6 (hand-off banner)
* **Reasoning:** in four of twelve model × regime cells the full 192-point grid
  moves request throughput by < 2 %, and the transfer matrices show no
  configuration that is good everywhere. Therefore the residual end-to-end gap
  is not reachable by serving configuration.
* **Limitation:** this is scoped to **these four knobs, these six regimes, this
  hardware and this software stack**. It is not a claim that the serving
  optimisation space in general is exhausted.
* **Type:** **inference**, clearly marked as such in the speaker notes.

---

## Wording that must NOT be used

| forbidden | why | use instead |
|---|---|---|
| "serving tuning has no value" | contradicted by +78.6 % shared-prefix and the high-concurrency stress result | "serving tuning removes workload-specific cliffs" |
| "all serving optimization space is exhausted" | only 4 knobs and 6 regimes were searched | "within this four-knob space and these six regimes, the ceiling is ~1 % in the saturated regimes" |
| "chunked prefill gives +78.6 %" | the winner changes cap, chunk and memory fraction together | "a long-context/high-capacity tuned configuration gives +78.6 %" |
| "5–9× from autotuning" | retracted; CUDA-Graph-disabled baseline artifact | "1.00–1.05× against a correctly configured default" |
| "Qwen long-input tuning gives 1.36–2.21×" | that data is LFM2.5, not Qwen | cite it as LFM2.5, multi-knob |
| "the agent discovered this configuration" | the search is a deterministic grid plus rule-based scoring, not an LLM decision | "grid enumeration identified" |
