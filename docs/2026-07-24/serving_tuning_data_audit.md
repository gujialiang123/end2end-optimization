# Serving-Tuning Data Audit (Phase 0)

**Date:** 2026-07-24 · **Repo commit at audit time:** `915f636`
**Purpose:** establish exactly which existing artifacts are reusable for the
performance-gap presentation, *before* spending GPU hours. This audit is
read-only with respect to historical results; nothing is edited or deleted.

Companion machine-readable file: `serving_tuning_data_audit.json`.

---

## 0. Summary verdict

| # | Dataset | Class | One-line reason |
|---|---|---|---|
| E1 | `results/2026-06-25_autotuning/` | **(3) candidate selection only** | Non-streaming client → no TTFT/TPOT; tuned CUDA-graph & backend together with serving knobs. |
| E2 | `results/consolidated_v4_by_model_config.csv` | **(4) historical/appendix** | Non-streaming client → throughput/TPOT only, no TTFT; hand-picked configs, not a search. |
| E3 | `results/consolidated_v7_config_sweep.csv` | **(2) supporting evidence** | Real agentic workloads with TTFT, but n=1 per cell and only 2 candidate configs. |
| E4 | `results/consolidated_v8_tuning.csv` | **(2) supporting evidence** | 2-knob sweep (chunk × cap) on agentic datasets, n=1 per cell. |
| E5 | `results/2026-07-22_lfm25_plateau_100/` (v48) | **(2) supporting evidence** | Clean, no-warm-start, 100/192 configs, streaming — but **LFM only, 1 regime**, and long-context server args. |
| E6 | `results/2026-07-23_high_concurrency_ttft_rerun/` (v51) | **(2) supporting evidence / stress panel** | Repeated (5×) streaming TTFT, but stress regimes (C64/O512, C128/O256) and only 3 configs. |
| E7 | `results/2026-07-02_lfm2.5_v3/` | **(4) historical/appendix** | Warm-started (`enqueue_trial`), tuned MoE backend + CUDA graph jointly → confounded. |
| **NEW** | `results/2026-07-24_serving_ceiling/` | **(1) canonical matrix** | This campaign: 2 models × 192 configs × 6 workloads, one streaming client, clean grid. |

**Conclusion: no existing dataset can serve as the canonical matrix.** Every
historical artifact fails at least one invariant (streaming client, clean search,
both models, all six regimes, or knob isolation). Hence the new campaign.

---

## 1. Required audit checks (explicitly answered)

### 1.1 Which historical Qwen experiments used a non-streaming client (→ no TTFT/TPOT)?

- **E1 (2026-06-25 autotuning)** and **E2 (v4 consolidated CSV)** both used the
  custom closed-loop client in `.github/skills/e2e-bench-runner/impl/run_bench.py`,
  which POSTs to `/generate` **without** `"stream": true`. It therefore records
  only wall-clock throughput and a derived TPOT; **client-observed TTFT is
  structurally absent**, which is why `consolidated_v4_by_model_config.csv` has no
  TTFT column.
- This was independently confirmed in the v51 rerun (2026-07-23), whose entire
  motivation was to recover TTFT for the four v4 slide points.

### 1.2 Which early Qwen gains used a miscalibrated baseline (CUDA Graph disabled)?

- **The 6/11 `cutlass-bf16-patched` study and the 6/25 "morning headline" numbers.**
  Their baseline passed `--disable-cuda-graph true` (a workaround for a Triton
  3.5.1 `KeyError: 'cubin'` bug). Against that handicapped reference the reported
  speedup was **4.7–8.4×**.
- `docs/2026-06-25/autotuning_honest_results.md` retracts this: measured against
  the true zero-flag default (CUDA graph ON), the same Optuna winner is
  **1.00–1.05×**, i.e. essentially flat (`R_short_decode` 1.00×,
  `R_medium_balanced` 1.00×).
- **Consequence for the slides:** the "5–9×" figure must never be shown. The
  honest early-Qwen result is a *negative/flat* result and is valuable as such.

### 1.3 Which LFM studies used enqueued warm-start configurations?

- **E7 (2026-07-02 v3)**: trials 0–3 were injected via `study.enqueue_trial` with
  cookbook-equivalent configs, so its convergence curve "starts high" by
  construction. Its convergence figure is **not** usable as evidence of search
  quality.
- **E5 (v48, 2026-07-22)** was explicitly built to replace it: **no warm start,
  no enqueued trials, fresh study**. E5's convergence curve is the only clean one.

### 1.4 Which studies changed backend or CUDA Graph together with serving knobs?

- **E1 (6/25)**: search space included `--disable-cuda-graph {true,false}` and the
  MoE runner backend → kernel-path choice is confounded with serving tuning.
- **E7 (v3)**: tuned MoE backend (auto / triton / flashinfer_cutlass) **and**
  CUDA-graph enablement alongside serving knobs.
- **Clean (serving knobs only, backend + CUDA graph fixed):** E5 (v48), E6 (v51),
  and the new canonical campaign.

### 1.5 Which long-input experiments changed multiple knobs (→ cannot be attributed to chunked prefill alone)?

- The 8K/16K/32K/50K long-input result (≈1.36× / 1.55× / 1.47× / 2.21×) is in
  `results/consolidated_config_spreadsheet.csv` + `results/2026-07-02_lfm2.5_v3/`.
- Two corrections already documented in
  `docs/2026-07-23/serving_tuning_slide_verified_data.md`:
  1. **The model is LFM2.5-8B-A1B, not Qwen.** There is *no* verified Qwen
     long-input tuning curve in this repository.
  2. The winning configuration changed **four** knobs simultaneously
     (`chunk=8192` **+** `fcfs` **+** `mem=0.75` **+** `fa3`). It must be labelled
     a **"long-context tuned configuration"**, never "chunked prefill gives 2.21×".

### 1.6 Which agentic results contain only one measurement per configuration?

- **E3 (v7)** and **E4 (v8)**: every `(model, config, dataset)` cell is a single
  run. No confidence interval can be computed, so these support *direction*
  ("chunking helps LFM shared-prefix, is neutral on Qwen") but not a precise
  effect size.
- The new campaign fixes this by re-running selected configs 5×.

### 1.7 Which experiments must not be combined in one numerical matrix?

| Reason | Datasets |
|---|---|
| Different benchmark client (non-streaming vs streaming) | E1, E2 vs E3–E6 |
| Different server context length / prefill budget | E5 used `--context-length 73728`, `--max-prefill-tokens 96000`; E6 and the new campaign use `8192` |
| Different fixed backend policy | E5/E7 pinned `--moe-runner-backend triton`; E6 and the new campaign use `auto` (resolves to the same path on H200, but is *recorded*, not asserted) |
| Confounded knobs | E1, E7 (CUDA graph / backend tuned) |
| Different workload definitions | E2/E6 stress regimes (C64/O512, C128/O256) vs the six canonical regimes |

Therefore Qwen and LFM raw values are **never** merged into a single numeric
matrix; cross-model comparison is done only on **ratios vs each model's own
cookbook baseline**, which is dimensionless and stated as such.

---

## 2. Per-experiment audit detail

### E1 — `results/2026-06-25_autotuning/` (Qwen, Optuna TPE)

| field | value |
|---|---|
| model | Qwen3-30B-A3B BF16 (also `true-default-bf16` variant) |
| GPU / TP / dtype | 1× H200 · TP1 · BF16 |
| software | sglang study/v0.5.9 + local flashinfer_cutlass allowlist patch; torch 2.9.1; Triton 3.5.1 |
| search algorithm | Optuna 4.9.0 `TPESampler`, seed 2026 |
| knobs | 5 flags incl. `--disable-cuda-graph` and MoE runner backend → **96 combinations** |
| workloads | 4 synthetic regimes (`regimes_resolved.yaml`) |
| client | custom closed-loop, **non-streaming** |
| metrics | request throughput only (no TTFT; TPOT derived) |
| repeats | `num_runs=3`, run 0 dropped |
| raw per-request data | **no** |
| verdict | **class 3** — usable to *select candidate knob values*, not for the metric matrix |
| key honest finding | true speedup vs correct default = **1.00–1.05×**, not 5–9× |

### E2 — `results/consolidated_v4_by_model_config.csv`

| field | value |
|---|---|
| models | `lfm2.5-8b-a1b`, `qwen3-30b-a3b` BF16 |
| configs | hand-picked (`cookbook_baseline`, `v3_best_chunk8k`, high-concurrency candidate, …) — **not a search** |
| workloads | C1/O2K, C32/O256, C32/O1K, C64/O512, C128/O256 (stress regimes) |
| client | non-streaming → **no TTFT** |
| metrics | tokens/s, req/s, MFU, MBU, TPOT, decode-step ms, HBM peak, power |
| verdict | **class 4** — appendix; superseded for TTFT by E6 |

### E3 — `results/consolidated_v7_config_sweep.csv`

| field | value |
|---|---|
| datasets | `shared_prefix` (generated-shared-prefix), `toolagent` (Mooncake FAST'25 trace) |
| client | `sglang.bench_serving`, **streaming** → TTFT/TPOT/E2E present |
| repeats | **n = 1 per cell** → no CI |
| verified numbers | LFM2.5 shared-prefix 14.09→18.12 req/s (+28.6 %), median TTFT 2758→1287 ms (−53.3 %); tool-agent 16.13→17.20 req/s (+6.6 %), TTFT 1583→209 ms (−86.8 %); Qwen shared-prefix −2.9 % req/s, +1.3 % TTFT |
| verdict | **class 2** — supports "same candidate behaves differently across models" |

### E4 — `results/consolidated_v8_tuning.csv`

2-knob sweep (`chunked ∈ {2048, …}` × `cap ∈ {32, 64, …}`) on the two agentic
datasets, streaming metrics, **n = 1 per cell**. **Class 2.**

### E5 — `results/2026-07-22_lfm25_plateau_100/` (v48)

| field | value |
|---|---|
| model | LFM2.5-8B-A1B BF16, TP1, H200 |
| fixed path | `--moe-runner-backend triton`, `--attention-backend fa3`, CUDA graph ON, `--context-length 73728`, `--max-prefill-tokens 96000` |
| search | Optuna TPE, seed 20260722, `n_startup_trials=20`, **no warm start** |
| space | **the same 4 serving knobs / 192 combinations** used by this campaign |
| completed | 100/100 unique, 0 failures, 26 duplicates pruned (126 attempts) |
| workload | **`R_concurrent_decode` only** (1 of 6) |
| client | `sglang.bench_serving`, streaming, `--output-details` |
| key finding | plateau: best-so-far within 1 % of final by config 7; **0 % improvement over the last 20 configs**; spread 7.09→19.98 req/s driven entirely by `cap=8` starvation |
| verdict | **class 2** — the cleanest existing plateau evidence, but single regime, single model, different context length |

### E6 — `results/2026-07-23_high_concurrency_ttft_rerun/` (v51)

| field | value |
|---|---|
| models | Qwen3-30B-A3B + LFM2.5, BF16, TP1, H200 |
| configs | 3 (cookbook / cap-only / full high-concurrency) |
| workloads | C64/O512, C128/O256 (**stress**, not canonical) |
| repeats | 6 reps, rep 0 dropped → **n = 5**, 5 760 per-request records |
| key finding | throughput 1.40–2.44× **and** TTFT p50/p95 −85 %…−96 % ⇒ removal of an admission bottleneck, **not** a Pareto trade-off; cap-only ablation reproduces nearly all of it (full-vs-cap residual −2.0 %…+0.7 %) |
| verdict | **class 2** — use as a labelled *high-concurrency stress panel* |

### E7 — `results/2026-07-02_lfm2.5_v3/`

Warm-started (`enqueue_trial` trials 0–3 with cookbook-equivalent configs), MoE
backend and CUDA graph tuned jointly. Source of the (mis-attributed) 8K–50K
long-input numbers. **Class 4 — appendix only, never a convergence claim.**

---

## 3. Gap analysis → what the new campaign must add

| Requirement for the six slides | Covered by existing data? | Action |
|---|---|---|
| Both models, identical protocol | ✗ (E5 LFM-only, E1 Qwen-only) | run both in one campaign |
| All six canonical regimes | ✗ (E5 = 1 regime; E1 = 4 synthetic; E3/E4 = 2 agentic) | one server launch × 6 workloads |
| Streaming TTFT everywhere | ✗ (E1/E2 non-streaming) | `sglang.bench_serving --output-details` for **all** six |
| Full-grid coverage (Pareto, negatives) | ✗ (E5 = 100 of 192, 1 regime) | full 192 grid × 2 models |
| Knob isolation (no backend/CUDA-graph tuning) | partly (E5, E6 clean) | freeze backend/CUDA graph; record resolved values |
| Repeats + CI on final claims | ✗ (E3/E4 n=1) | 5-rep validation pass on selected configs |
| Cross-regime transfer matrix | ✗ (never measured) | reuse coverage cells (every config already sees all 6 regimes) |

---

## 4. Execution plan and measured GPU-time estimate

Estimates come from a real smoke test (3 diverse configs × 6 workloads on GPU 5),
not from guesswork.

| quantity | measured |
|---|---|
| LFM2.5 server startup (incl. CUDA-graph capture) | **25 s** |
| Qwen3-30B server startup | **37–40 s** |
| LFM2.5 full task (launch + 6 workloads + shutdown) | **133–142 s** |
| Qwen3-30B full task (launch + 6 workloads + shutdown) | **152–181 s** |
| ⇒ coverage pass, LFM (192 configs) | ≈ **7.3 GPU-h** |
| ⇒ coverage pass, Qwen (192 configs) | ≈ **8.9 GPU-h** |
| **total coverage (384 tasks)** | ≈ **16.5 GPU-h** |

Per the plan's decision rule (full grid if ≲ 24 H200 GPU-hours), the campaign runs
the **full 192-configuration grid for both models**; the Sobol/QMC fallback is
**not** needed. With 3 workers (GPU 4/5/6) this is ≈ **5.5 h wall-clock**. The
5-rep validation pass on selected configs adds ≈ 2 GPU-h.

### Smoke-test evidence that the harness is sound

All six workloads completed with full per-request TTFT/ITL, and the resolved
server configuration was verified from the log for every config
(`attention_backend=fa3`, `moe_runner_backend=auto`, `disable_cuda_graph=False`,
`cuda_graph_captured=True`, and all four knobs echoed at their requested values).

A real effect was already visible in the smoke data (LFM2.5, 1 run each):

| config | `shared_prefix` req/s | `R_concurrent_decode` req/s |
|---|---|---|
| cfg 74 cookbook (cap 32) | 14.2 | 21.8 |
| cfg 0 (cap 8) | **6.6** | **7.9** |

i.e. a genuine **admission cliff** when `max_running_requests` is too small —
the positive half of the evidence chain.

---

## 5. Known limitations recorded up front

1. **`chunked_prefill_size = 8192` ≈ `-1`** for this campaign, because
   `--context-length` is 8192 and the longest canonical input is ~4 000 tokens.
   This is *not* a bug: the pair acts as a built-in duplicate measurement and
   gives a free estimate of run-to-run noise. Only `chunk = 2048` actually chunks
   the long-prefill and shared-prefix workloads. This must be stated on the slide
   rather than presenting three distinct chunk settings.
2. **Synthetic regimes are short** (`R_long_prefill` ≈ 0.3 s, `R_concurrent_decode`
   ≈ 1.5 s per run). The definitions are kept for fidelity with the recovered v4
   spec; statistical strength comes from the 5-rep validation pass, not from
   longer runs.
3. **`prompt_words` → token mapping.** The recovered v4 regimes are specified in
   *words*; the unified client specifies *tokens*. We map 1 word → 1 token
   (100/800/4000/200 tokens). This is an approximation of the original text
   prompts and is documented as such; it is applied identically to every
   configuration, so within-campaign comparisons are unaffected.
4. **`moe_runner_backend=auto`** rather than a hard `triton` pin. The resolved
   value is parsed from every server log and stored in `per_config_log*.csv`, so
   the invariant is *verified* rather than *assumed*.
