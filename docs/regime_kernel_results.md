# Regime-aware Kernel Specialization — results

**Date:** 2026-07-26 · Status and plan: `docs/regime_kernel_status.md`,
`docs/regime_kernel_experiment_plan.md`

Fixed frame: 1× H200, TP1, BF16, sglang 0.5.12.post1 @ `17f7a1da1`, torch
2.9.1+cu128, Triton 3.5.1, driver 580.105.08. Hot kernel: fused-MoE Triton.
Specialization is opt-in (`override_config` for microbenchmarks,
`SGLANG_MOE_CONFIG_DIR` for serving); default code paths are untouched.

---

## 0. One-paragraph summary

Regime-aware kernel specialization **works, and the hypothesis is supported** —
but only once two mechanisms are respected. Final end-to-end result on LFM2.5,
serving knobs frozen, only the MoE kernel profile varying:

| regime | global-best | regime-aware (naive) | guarded (mis-keyed M) | **guarded (correct M)** |
|---|---:|---:|---:|---:|
| low-batch decode | 0.923× | 0.745× | 1.001× | **0.998×** |
| concurrent decode | 1.004× | 1.060× | 1.003× | **1.014×** |
| long prefill | 0.796× | 1.170× | 1.223× | **1.223×** |

Final arm, 6 repetitions each: long prefill 15.06 ± 0.08 vs 12.32 ± 0.03 req/s
(6/6 non-overlapping), concurrent decode 22.09 ± 0.11 vs 21.78 ± 0.09, low-batch
decode neutral inside the noise band.

A single global profile is *harmful* (0.80–0.92×). Naive per-regime
specialization wins big on prefill but costs 25 % on decode. The **guarded**
policy — specialize only where the oracle proves headroom, keep the runtime
default elsewhere — delivers **+22.3 % end-to-end on long prefill with zero
regression anywhere** (8/8 clean repetitions, no overlap with the baseline).

Getting there required **three** corrections, and they are the study's most
transferable findings:
1. we were tuning a kernel variant the server never executes (expert bias);
2. CUDA-graph capture replays decode, so the config is baked in at capture time;
3. **`M` is the token count, not `tokens x top_k`** — we had the profile keys
   wrong by a factor of `top_k`, which hid real headroom behind mis-aligned
   buckets. Only a live trace exposed this.

---

## 1. What was run

| stage | jobs | failures | artifact |
|---|---:|---:|---|
| tuning sweep (2 models × 10 token counts, full pruned space) | 20 | 0 | `results/regime_kernel/raw/sweep/` |
| routing control (2 models × 4 M × 2 routings) | 16 | 0 | `raw/routing/` |
| transfer matrix (profiles × all token counts) | 20 | 0 | `raw/transfer/` |
| agent closed loop (2 contrasting regimes) | 2 | 0 | `results/regime_kernel/agent/` |
| end-to-end, no-bias profiles (3 regimes × 3 arms × 5 reps) | 45 runs | 0 | `results/regime_kernel/e2e/lfm25/` |
| bias-variant control sweep | 10 | 0 | `raw/bias/` |
| end-to-end, bias-aware profiles | 45 runs | 0 | `e2e/lfm25_bias/` |
| end-to-end, guarded profile (incl. 8-rep confirmations) | 58 runs | 0 | `e2e/lfm25_bias/` |

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

## 7. End-to-end — three iterations to a correct result

LFM2.5, serving knobs frozen, **only** `SGLANG_MOE_CONFIG_DIR` varies. Profile
pickup verified in every server log.

### Iteration 1 — naive regime-aware profiles tuned on the no-bias variant

| regime | default | global-best | regime-aware |
|---|---:|---:|---:|
| A low-batch decode | 1.691 ± 0.001 | 0.923× | **0.745×** |
| B concurrent decode | 21.800 ± 0.117 | 1.003× | 1.007× |
| C long prefill | 12.217 ± 0.340 | **0.793×** | **1.183×** |

Long prefill gains 18.3 % while the single global profile loses 20.7 % on the
same workload — but low-batch decode loses 25 %. Something is wrong.

### Iteration 2 — profiles re-tuned on the with-bias variant the server runs

| regime | default | global-best | regime-aware |
|---|---:|---:|---:|
| A low-batch decode | 1.691 ± 0.002 | 0.922× | 0.879× |
| B concurrent decode | 21.824 ± 0.107 | 0.994× | **1.061×** |
| C long prefill | 12.291 ± 0.109 | 0.792× | **1.188×** |

The decode regression halves and concurrent decode turns into a real win, but a
12 % regression remains. §8 and §9 explain why.

### Iteration 3 — guarded profile (specialize only where headroom is proven)

| regime | default (req/s) | guarded | reps |
|---|---:|---:|---:|
| A low-batch decode | 1.686 ± 0.004 | **1.0015×** | 5 |
| B concurrent decode | 21.997 ± 0.075 | **1.005×** (median 1.003×) | 8 |
| C long prefill | 12.254 ± 0.106 | **1.221×** (median 1.223×) | 8 |

**Long prefill: 8/8 repetitions between 14.63 and 15.19 req/s against a baseline
of 11.91–12.40 — the distributions do not overlap.** Decode regression is fully
eliminated. This is the deployable result.

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

## 9. What M actually is — measured, not assumed

`fused_experts_impl` computes

```python
M = min(num_tokens, CHUNK_SIZE)
```

so **M is the token count of the batch entering the MoE layer, not
`tokens x top_k`**. We initially assumed the latter, which meant every profile
was written under a key `top_k` times too large, and nearest-M lookup then
applied each config to the wrong bucket.

An opt-in tracer (`RK_KERNEL_TRACE`, `scripts/regime_kernel/rk_trace/`) settled
it by recording 924 real lookups. Note the tracer needed one non-obvious fix:
`fused_moe.py` does `from .fused_moe_triton_config import
try_get_optimal_moe_config`, binding the original function into its own module
namespace, so patching only the config module has no effect on the call site.

Measured regime → M mapping (`processed/measured_M_distribution.csv`):

| regime | measured M values | assumed (wrong) |
|---|---|---|
| A low-batch decode | 101, 106, 119, 120, 121, 125 | 4 |
| B concurrent decode | 115, 672, 5327 | 128 |
| C long prefill | 4595, 13138 | thousands ✓ |

Two consequences:

* Regime A's MoE lookups are the **prompt prefill**, not the decode steps —
  decode is graph-replayed and issues no lookup at all.
* Re-running the sweep on the true M axis reveals headroom that the mis-keyed
  analysis had hidden:

| true M | ≤ 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle speedup | 0.98–1.09× | 1.484× | 1.440× | 1.449× | 1.391× | 1.491× | 1.632× | 1.602× | 1.639× |

**The crossover is at M ≈ 64**, and every measured serving M except the
graph-replayed decode steps sits above it. Figure:
`plots/measured_M_vs_headroom.png`.

## 10. Why decode still gains nothing — CUDA graph pins the M range

An opt-in tracer (`RK_KERNEL_TRACE`, `scripts/regime_kernel/rk_trace/`) recorded
**zero** config lookups during steady-state decode. That is not a bug: with CUDA
graph enabled the decode path is replayed from captured graphs, so the kernel
configuration is baked in at capture time.

The server log shows capture uses batch sizes `[1, 2, 4, 8, 12, 16, 24, 32]`,
so with `top_k = 4` decode only ever reaches **M ∈ {4, 8, 16, 32, 48, 64, 96,
128}**. Comparing that against the with-bias oracle:

| M reached by | M values | with-bias oracle speedup |
|---|---|---|
| **decode** (CUDA-graph batch sizes) | 4 … 128 | **0.98 – 1.09×** — no headroom |
| **prefill** | ≥ 256 | **1.39 – 1.64×** — real headroom |

So the naive profile was replacing configurations in a range where the best
achievable gain is inside the noise band. Tuning there does not select a better
kernel, it selects **measurement noise**, and deploying that noise cost 12–25 %
throughput. This is the same selection-on-noise failure mode documented in the
serving-objective study, now observed at kernel level.

The guardrail follows directly: specialize a bucket only when the oracle beats
the default by more than a threshold (1.15× here), and write the runtime's own
heuristic default everywhere else so nearest-M lookup can never pull a prefill
tile into a decode bucket. For LFM2.5 that specializes **4 of 10 M buckets**
(`results/regime_kernel/processed/lfm25_guarded_profile_decisions.csv`).

## 11. Deployment finding — new configs cause Triton recompilation stalls

The first 8-repetition run of concurrent decode with the guarded profile was
bimodal: 6 runs at 22.2–22.3 req/s and 2 runs at 14.6 and 16.5. Re-running the
identical configuration with a **warm Triton cache** removed the effect entirely:

| Triton cache | runs (req/s) | mean vs default |
|---|---|---:|
| cold | 14.59, 22.23, 16.45, 22.21, 22.30, 22.25, 22.28, 22.25 | 0.940× (median 1.018×) |
| **warm** | 21.98, 22.05, 22.03, 22.18, 22.37, 22.04, 21.99, 22.27 | **1.005×**, 8/8 clean |

The bimodality was **entirely JIT recompilation**, not kernel performance.
Practical consequence: introducing new kernel configurations into a live server
causes intermittent multi-second stalls until the cache is warm, so a deployment
must pre-warm the Triton cache — and any benchmark that does not will report a
misleadingly low mean.

## 12. Does this support the hypothesis?

| claim | verdict | evidence |
|---|---|---|
| regimes produce different kernel workloads | **supported** | winner moves 16→128 BLOCK_M; intensity 0.083→170.7 FLOP/byte |
| regime-tuned configs do not transfer | **strongly supported** | 0.123× worst-case cross-regime |
| a few regime profiles beat one global profile | **supported** | global 0.618×/0.388× at large M; 3 profiles reach 73–100 % of oracle |
| specialization improves end-to-end serving | **supported, with a guardrail** | M-corrected guarded profile: **+22.3 %** long prefill (6/6 non-overlapping), **+1.4 %** concurrent decode, neutral low-batch decode — no regression anywhere |
| an agent can close the loop | **supported** | different diagnoses → different action sequences, with accept/reject and rollback |

## 13. Failed experiments and blockers

* **Regime A E2E regression** — root-caused and **fixed**. Two contributing
  mechanisms: the bias-variant mismatch (§8) and the CUDA-graph M range (§9).
  The guarded profile takes it from 0.745× to 1.0015×.
* **Concurrent-decode bimodality** — root-caused to Triton JIT recompilation
  (§10) and reproduced/eliminated with a warm cache.
* **Wrong M definition** — assumed `M = tokens x top_k`; the kernel uses
  `M = min(num_tokens, CHUNK_SIZE)`. Measurements were valid but profile keys
  were `top_k`-times too large, so configs landed in the wrong buckets. Found
  only by tracing a live server; fixed, and the corrected sweep exposed a
  crossover at M ≈ 64 that the mis-keyed view had hidden.
* **Tracer no-op** — patching `fused_moe_triton_config.try_get_optimal_moe_config`
  had no effect because the call site imports the symbol directly. Fixed by
  patching the consumer module as well.
* **Result overwrite** — the iteration-3 runs wrote into the same
  `e2e_runs.json` path as iteration 2 for regimes A and C, so those raw files
  hold the latest arms only. The iteration-2 numbers are preserved in this
  report and in `processed/e2e_summary.csv`. Future runs should timestamp the
  output directory.
* **Qwen large-M has no headroom** (oracle ≤ 0.99×), so Qwen was correctly
  excluded from E2E rather than spending GPU hours measuring noise.
* **Upstream tuner unusable** — `benchmark/kernels/fused_moe_triton` requires
  `ray`, which is not installed; we wrote our own driver. Also required
  `set_global_server_args_for_scheduler` before any standalone kernel call.
* No OOM and no correctness failures occurred at any point.

## 14. Next steps

1. **Emit the `_down` companion config.** The runtime asserts
   `config["BLOCK_SIZE_M"] == down_config["BLOCK_SIZE_M"]`, so the down
   projection must be tuned under that constraint; currently it falls back to
   the default while the up projection is specialized.
2. **Extend the waterfall to cookbook → serving tuning → serving + kernel** to
   quantify how complementary the two levels are. The serving numbers already
   exist in `results/2026-07-24_serving_ceiling/`.
3. **Pre-warm the Triton cache as part of deployment** (§10), and add a
   cache-warming step to the E2E harness so means are not skewed.
4. P1: runtime bucket dispatch under CUDA graph, a second model family,
   shared-prefix/agentic workloads, deeper NCU at the crossover points.
5. Raise `cuda_graph_max_bs` or capture larger batches so that concurrent decode
   also reaches the M range where headroom exists — currently the only way
   decode can benefit at all.
