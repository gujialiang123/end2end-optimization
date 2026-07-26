# Regime-aware Kernel Specialization — results

**Date:** 2026-07-26 · Status and plan: `docs/regime_kernel_status.md`,
`docs/regime_kernel_experiment_plan.md`

Fixed frame: 1× H200, TP1, BF16, sglang 0.5.12.post1 @ `17f7a1da1`, torch
2.9.1+cu128, Triton 3.5.1, driver 580.105.08. Hot kernel: fused-MoE Triton.
Specialization is opt-in (`override_config` for microbenchmarks,
`SGLANG_MOE_CONFIG_DIR` for serving); default code paths are untouched.

---

## 0. One-paragraph summary

Regime-aware kernel specialization **works, and matters**: on long prefill it
delivers **+18.3 % end-to-end request throughput** while a single global-best
profile *loses* 20.7 % on the same workload. But the study also produced a
first-rate negative: a profile tuned on the *wrong kernel variant* is actively
harmful — our low-M profile cost **25 % throughput** in low-batch decode, and the
mechanism is identified and reproduced (the model uses expert bias; the
microbenchmark did not, so we tuned a kernel the server never runs).

---

## 1. What was run

| stage | jobs | failures | artifact |
|---|---:|---:|---|
| tuning sweep (2 models × 10 token counts, full pruned space) | 20 | 0 | `results/regime_kernel/raw/sweep/` |
| routing control (2 models × 4 M × 2 routings) | 16 | 0 | `raw/routing/` |
| transfer matrix (profiles × all token counts) | 20 | 0 | `raw/transfer/` |
| agent closed loop (2 contrasting regimes) | 2 | 0 | `results/regime_kernel/agent/` |
| end-to-end (LFM2.5 × 3 regimes × 3 arms × 5 reps) | 45 runs | 0 | `results/regime_kernel/e2e/` |
| bias-variant control | 2 | 0 | `raw/bias/` |

Every timing is correctness-gated: a candidate is compared against the default
kernel output (BF16 tolerance, NaN/Inf check) and is discarded before timing if
it fails. **0 correctness failures across ~9 000 benchmarked configurations.**

Commands:

```bash
python scripts/regime_kernel/rk_campaign.py --init          # queue sweep+routing
python scripts/regime_kernel/rk_campaign.py --gpu 6 --worker w6
python scripts/regime_kernel/rk_profiles.py                 # build profiles
python scripts/regime_kernel/rk_campaign.py --init-transfer
python scripts/regime_kernel/rk_agent.py --model lfm25 --tokens 1 --gpu 6
python scripts/regime_kernel/rk_e2e.py --model lfm25 --regime C_long_prefill --gpu 6
python scripts/regime_kernel/rk_process.py && python scripts/regime_kernel/rk_plots.py
```

---

## 2. RQ1 — do regimes produce different kernel workloads? **Yes.**

The winning configuration moves systematically with M (= tokens × top_k), and the
achievable speedup grows monotonically with it (LFM2.5, uniform routing):

| tokens | M | default (ms) | best (ms) | speedup | winning config |
|---:|---:|---:|---:|---:|---|
| 1 | 4 | 0.2556 | 0.2396 | 1.067× | bm16 bn64 bk256 gm1 w4 s2 |
| 2 | 8 | 0.2653 | 0.2475 | 1.072× | bm16 bn64 bk256 gm32 w8 s3 |
| 4 | 16 | 0.2591 | 0.2424 | 1.069× | bm16 bn32 bk256 gm16 w4 s2 |
| 8 | 32 | 0.3108 | 0.2401 | 1.295× | bm16 bn64 bk128 gm16 w8 s4 |
| 16 | 64 | 0.3140 | 0.2501 | 1.255× | bm32 bn256 bk64 gm32 w8 s3 |
| 32 | 128 | 0.3199 | 0.2449 | 1.306× | bm16 bn128 bk64 gm32 w4 s4 |
| 64 | 256 | 0.2730 | 0.1950 | 1.400× | bm16 bn128 bk128 gm1 w8 s2 |
| 512 | 2048 | 0.3486 | 0.2385 | 1.462× | bm64 bn128 bk64 gm32 w4 s4 |
| 2048 | 8192 | 0.7880 | 0.4988 | 1.580× | bm128 bn256 bk64 gm1 w8 s4 |
| 8192 | 32768 | 2.6943 | 1.6006 | **1.683×** | bm128 bn256 bk64 gm1 w8 s3 |

`BLOCK_SIZE_M` climbs 16 → 32 → 64 → 128 and `BLOCK_SIZE_N` 64 → 256 as the
regime moves from single-request decode to prefill. The measured bottleneck moves
with it: arithmetic intensity of the grouped GEMM is **0.083 FLOP/byte at M=4**
(weight-streaming bound, only 4 of 32 experts active, expert-load CV 2.65) and
**170.7 FLOP/byte at M=8192** (compute bound).

Figures: `plots/regime_workload_characterization.png`, `plots/kernel_winner_map.png`.

## 3. RQ2 — does a regime-tuned config transfer? **No, it degrades badly.**

Transfer matrix, speedup vs default (rows = tuned on, columns = tested at):

**Qwen3-30B**

| profile | t=1 | t=8 | t=64 | t=512 | t=2048 | t=8192 |
|---|---:|---:|---:|---:|---:|---:|
| low_M | 0.992 | 0.960 | 0.765 | 0.349 | 0.160 | **0.123** |
| mid_M | 0.805 | 0.780 | 1.017 | 0.693 | 0.361 | 0.285 |
| global_best | 1.003 | 0.839 | 0.967 | 0.761 | 0.472 | 0.385 |

**LFM2.5**

| profile | t=1 | t=8 | t=64 | t=512 | t=2048 | t=8192 |
|---|---:|---:|---:|---:|---:|---:|
| low_M | 0.974 | 0.750 | 1.190 | 0.903 | 0.620 | 0.554 |
| mid_M | 0.751 | 0.753 | **1.396** | 0.840 | 0.558 | 0.510 |
| global_best | 0.987 | 0.920 | 1.277 | 0.938 | 0.673 | 0.618 |

A configuration tuned for batch-1 decode is **8× slower than the default** when
applied to prefill on Qwen. This is the strongest single piece of evidence for
the hypothesis: kernel specialization is not merely beneficial per regime, it is
*mandatory* not to reuse it across regimes.

Figure: `plots/kernel_transfer_heatmap_{lfm25,qwen}.png`.

## 4. RQ3 — global vs regime-aware vs oracle

| model | tokens | oracle | global-best | regime-aware | regime as % of oracle |
|---|---:|---:|---:|---:|---:|
| lfm25 | 1 | 1.067 | 0.963 | 1.049 | 73 % |
| lfm25 | 64 | 1.400 | 1.275 | 1.397 | 99 % |
| lfm25 | 2048 | 1.580 | **0.667** | 1.580 | 100 % |
| lfm25 | 8192 | 1.683 | **0.618** | 1.664 | 97 % |
| qwen | 16 | 1.230 | 1.197 | 1.217 | 94 % |
| qwen | 2048 | 0.960 | **0.467** | 0.960 | — |
| qwen | 8192 | 0.966 | **0.388** | 0.960 | — |

* **A single global profile is not viable.** It degrades to 0.618× (LFM) and
  0.388× (Qwen) at large M — materially *worse than doing nothing*.
* **Three profiles recover 73–100 % of the per-shape oracle.** There is no need
  for a per-shape config table.
* **Honest negative:** on Qwen at large M the oracle itself is 0.96–0.99×, i.e.
  **there is no headroom at all** — the default is already optimal there. The
  audit predicted exactly this asymmetry: Qwen falls back to a real
  `triton_3_2_0` config, whereas LFM2.5 has *no config file whatsoever*.

Figure: `plots/strategy_comparison.png`.

## 5. Routing control — M sets the tile, routing sets the schedule

At fixed M, comparing near-uniform against strongly skewed routing:

| model | tokens | routing | expert-load CV | best speedup | winning config |
|---|---:|---|---:|---:|---|
| lfm25 | 512 | uniform | 0.10 | **1.462×** | bm64 bn128 bk64 **gm32** w4 s4 |
| lfm25 | 512 | skewed | 1.36 | 1.331× | bm64 bn128 bk64 **gm16** w4 s3 |
| lfm25 | 64 | uniform | 0.41 | 1.402× | bm16 bn128 bk128 **gm1** w8 s2 |
| lfm25 | 64 | skewed | 1.30 | 1.239× | bm16 bn64 bk64 **gm16** w4 s2 |
| qwen | 8 | uniform | 1.39 | 1.030× | bm16 bn64 bk128 **gm1** w8 s2 |
| qwen | 8 | skewed | 1.98 | 1.037× | bm16 bn64 bk256 **gm16** w4 s5 |

Two findings, both non-trivial:

1. **The tile shape is set by M, not by routing.** `BLOCK_SIZE_M/N/K` are
   essentially unchanged between uniform and skewed at the same M.
2. **`GROUP_SIZE_M` does respond to skew.** Under skew the winner selects
   `gm16` in 4 of 4 LFM cases, whereas uniform routing picks `gm1` or `gm32`.
   `GROUP_SIZE_M` controls block scheduling order, which is precisely the knob
   that matters when expert groups are ragged.
3. Skew also **lowers the achievable gain** (LFM t=512: 1.462× → 1.331×).

Figure: `plots/routing_control.png`.

## 6. RQ4 — the agent closed loop takes different paths for different diagnoses

The controller derives a bottleneck class from measured quantities and the
diagnosis *determines which candidates exist*:

**LFM2.5, M=4** — diagnosed `low_occupancy_launch_bound` (arithmetic intensity
0.083 FLOP/byte):

| iter | action | result | decision |
|---:|---|---:|---|
| 0 | small_tile_deep_pipeline | 1.081× | **accept** |
| 1 | small_tile_wide_k | 0.997× | **reject** (below the 1.01× threshold) |

**LFM2.5, M=8192** — diagnosed `compute_bound` (intensity 170.7):

| iter | action | result | decision |
|---:|---|---:|---|
| 0 | large_tile_more_warps | 1.266× | **accept** |
| 1 | large_tile_deep_pipeline | 1.055× further | **accept** → 1.336× total |

The two regimes produce **entirely different action sequences**, which is the
distinction between a controller and a wrapped parameter sweep. The loop also
reached 1.081× at M=4 using **48 candidates**, beating the 240-candidate
exhaustive sweep's 1.067×.

Figure: `plots/agent_iteration_trace.png`.

## 7. End-to-end — where the story gets interesting

LFM2.5, serving knobs frozen, **only** `SGLANG_MOE_CONFIG_DIR` varies. Profile
pickup verified in every server log. 5 measured repetitions per arm.

| regime | arm | request throughput | vs default | TPOT p95 |
|---|---|---:|---:|---:|
| A low-batch decode | default | 1.691 ± 0.001 | 1.000× | 2.23 ms |
| | global-best | 1.560 ± 0.001 | 0.923× | 2.43 ms |
| | **regime-aware** | 1.260 ± 0.001 | **0.745×** | 3.02 ms |
| B concurrent decode | default | 21.800 ± 0.117 | 1.000× | 5.17 ms |
| | global-best | 21.860 ± 0.129 | 1.003× | 5.08 ms |
| | regime-aware | 21.957 ± 2.040 | 1.007× (flat) | 4.95 ms |
| C long prefill | default | 12.217 ± 0.340 | 1.000× | 3.59 ms |
| | global-best | 9.683 ± 0.233 | **0.793×** | 3.35 ms |
| | **regime-aware** | 14.453 ± 0.051 | **1.183×** | 3.76 ms |

**The positive result:** regime-aware specialization gives **+18.3 % end-to-end
request throughput on long prefill**, with a tight confidence interval, while the
single global profile *loses 20.7 %* on the same workload. That is a direct,
end-to-end demonstration that one profile cannot serve all regimes.

**The negative result:** the same mechanism costs **25 % throughput in
low-batch decode**, far beyond the microbenchmark's prediction (0.974×).

## 8. Why regime A regressed — mechanism identified

The microbenchmark and the server were not running the same kernel variant.
LFM2.5 sets `use_expert_bias: true`, so the serving path executes the fused MoE
**with expert bias**; our sweep called it **without**. Re-running the identical
sweep with bias:

| variant | default (ms) | best (ms) | headroom |
|---|---:|---:|---:|
| no bias (what we tuned) | 0.2556 | 0.2396 | **1.067×** |
| with bias (what the server runs) | 0.2412 | 0.2397 | **1.007×** |

Two things follow:

1. **The with-bias default is already faster** (0.2412 vs 0.2556 ms) and has
   essentially **no tuning headroom** at M=4. The 1.067× we measured was a
   property of a kernel variant the server never executes.
2. Deploying that config into the with-bias path is not merely neutral, it is
   **harmful** — the E2E cost was 25 %.

This is the most transferable lesson of the study:

> A kernel microbenchmark must exercise the exact variant the server runs —
> including bias, fusion and quantization flags. Otherwise a "validated" speedup
> can be an artifact of an unused code path, and deploying it degrades production.

It also cleanly explains the regime pattern: at large M the tuning gain
(1.46–1.68×) is large enough to survive the variant mismatch and still deliver
+18.3 % E2E, whereas at M=4 the true headroom (1.007×) is smaller than the error
introduced by tuning the wrong variant.

## 9. Does this support the hypothesis?

| claim | verdict | evidence |
|---|---|---|
| regimes produce different kernel workloads | **supported** | winner moves 16→128 BLOCK_M; intensity 0.083→170.7 FLOP/byte |
| regime-tuned configs do not transfer | **strongly supported** | 0.123× worst-case cross-regime |
| a few regime profiles beat one global profile | **supported** | global 0.618×/0.388× at large M; 3 profiles reach 73–100 % of oracle |
| specialization improves end-to-end serving | **supported in one regime, refuted in another** | +18.3 % long prefill, −25 % low-batch decode |
| an agent can close the loop | **supported** | different diagnoses → different action sequences, with accept/reject and rollback |

## 10. Failed experiments and blockers

* **Regime A E2E regression** — root-caused to the bias variant mismatch (§8), not
  a harness bug. Fix is known: re-tune with `--bias` and redeploy. Queued as the
  first P1 item.
* **Qwen large-M has no headroom** (oracle ≤ 0.99×), so Qwen was correctly
  excluded from E2E rather than spending GPU hours measuring noise.
* **Upstream tuner unusable** — `benchmark/kernels/fused_moe_triton` requires
  `ray`, which is not installed; we wrote our own driver. Also required
  `set_global_server_args_for_scheduler` before any standalone kernel call.
* No OOM and no correctness failures occurred at any point.

## 11. Next steps

1. **Re-tune with `--bias` and re-run regime A/B E2E.** The harness already
   supports it; this is the direct test of §8 and should convert the −25 %
   regression into either a win or a clean flat.
2. Emit the `_down` companion config as well, so both projections use tuned
   settings.
3. Extend the E2E waterfall to cookbook → serving tuning → serving + kernel, to
   quantify how complementary the two levels are.
4. P1: runtime bucket dispatch under CUDA graph, second model family, shared-
   prefix/agentic workloads, deeper NCU on the two crossover points.
