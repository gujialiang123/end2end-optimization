# Regime-aware Kernel Specialization — repository status (Step 1–2)

**Date:** 2026-07-26 · read-only audit, no experiments run yet.

**Verdict: the experiment is feasible, cheap, and lands on a real gap.** The two
models we already have full serving data for are both running on **untuned MoE
Triton kernel configurations on this H200**, and SGLang's MoE config format is
*already keyed by M (token batch size)* — i.e. regime-aware kernel
specialization is a first-class concept in the runtime that nobody has populated
for our shapes. That makes this a config-specialization study, not a
write-a-new-CUDA-kernel study, which is what fits the 1–2 week budget.

---

## 1. The gap this experiment exploits

Both models emit this at every server start (verified in the campaign server
logs, `results/2026-07-24_serving_ceiling/raw/*/config_074/server.log`):

| model | MoE shape | what the runtime says |
|---|---|---|
| LFM2.5-8B-A1B | `E=32, N=1792` | **"Using default MoE kernel config. Performance might be sub-optimal!"** — no config file exists at all |
| Qwen3-30B-A3B | `E=128, N=768` | falls back to the **triton 3.2.0** config; no `triton_3_5_1` config for H200 |

We run Triton **3.5.1**. So:

* LFM2.5 uses a heuristic default for *every* M;
* Qwen3 uses a config tuned for a *different Triton version*.

Neither is regime-aware, and neither is tuned for this GPU + Triton build.

## 2. Why the runtime already supports the hypothesis

`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`:

```python
config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
```

The config JSON is a **map from M to a kernel configuration**. An existing tuned
file shows how strongly the optimum moves with M:

| M | BLOCK_M | BLOCK_N | BLOCK_K | GROUP_M | warps | stages |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 64 | 64 | 1 | 4 | 5 |
| 16 | 16 | 64 | 256 | 1 | 4 | 2 |
| 64 | 16 | 256 | 128 | 1 | 8 | 2 |
| 4096 | 128 | 256 | 64 | 16 | 8 | 4 |

That is an 8× swing in BLOCK_M and a 16× swing in GROUP_M across the M range our
three regimes span. **This is the mechanism by which a regime changes the
optimal kernel**, and it is directly measurable.

Mapping regime → M is unambiguous for a fused MoE kernel:

```
M = num_tokens_in_batch * top_k         (per layer, per invocation)
```

* **A. low-batch decode** (1 active request): LFM M = 1×4 = **4**; Qwen M = 1×8 = **8**
* **B. concurrent decode** (32 active): LFM M = **128**; Qwen M = **256**
* **C. long prefill** (16K prompt, chunked): M = chunk_tokens × top_k, i.e. **thousands**

So the three regimes land in three different, well-separated regions of the very
table that selects the kernel config. This is the cleanest possible test of RQ1.

## 3. Switching profiles without patching SGLang

Two mechanisms exist, both already in the runtime:

| mechanism | use |
|---|---|
| `SGLANG_MOE_CONFIG_DIR` env var (read in `get_moe_configs`) | point a whole server at an alternative profile directory — **this is how E2E profile swapping will be done**, default path unchanged |
| `override_config()` context manager (`fused_moe_triton/__init__.py`) | force one exact config inside a microbenchmark — used for the transfer matrix |

This satisfies the requirement that default behaviour is untouched and
specialization is opt-in via an explicit flag.

## 4. What already exists and is reusable

| asset | location | reuse |
|---|---|---|
| warmed serving grid, 2 models × 192 configs × 6 regimes, streaming TTFT/TPOT/E2E | `results/2026-07-24_serving_ceiling/` | **serving knobs are already tuned per regime** — the kernel study can hold them fixed at the validated per-regime winner, which is exactly the waterfall the plan asks for |
| 5-rep validated winners + CIs | `results/2026-07-24_serving_ceiling_validation/`, `results/2026-07-26_alternative_objectives/` | baseline anchors and the statistical protocol (bootstrap CI, WIN/TRADE-OFF/FLAT classification) |
| canonical benchmark harness | `scripts/serving_ceiling_lib.py`, `run_serving_ceiling_campaign.py` | **reuse verbatim for E2E** — server lifecycle, resolved-knob verification from logs, per-workload warm-up, sqlite work queue, failure policy |
| six frozen workload definitions | `results/2026-07-24_serving_ceiling/workloads.yaml` | regimes A/B/C map onto `R_short_decode`, `R_concurrent_decode`, `R_long_prefill` |
| NCU / nsys profiling runs | `results/2026-07-08_v6_ncu/`, `2026-07-10_v9_ncu_realworkload/`, `2026-07-15_v19b_ncu_decode/` | bottleneck evidence; avoids re-deriving that MoE is the hot kernel |
| prior kernel-change regime sweep | `docs/2026-07-20/regime_sweep_kernel_changes.md` | **the key negative control** (see §6) |
| SGLang MoE tuning tooling | `sglang/benchmark/kernels/fused_moe_triton/` | search-space definition and config file naming; the driver itself needs `ray`, which is not installed → we write our own driver |

## 5. What is missing and must be built

1. **A microbenchmark driver without `ray`.** Upstream's tuner enumerates
   4×5×3×4×2×4 = **1920 configs per M**, which for 18 M buckets × 2 models is
   ~69 k kernel benchmarks. Too expensive and unnecessary. We need a pruned,
   legality-filtered search space and our own driver calling `fused_experts`
   directly under `override_config`.
2. **A correctness harness** comparing candidate output against the default
   kernel path (BF16 tolerance, NaN/Inf checks, multiple seeds and real shapes)
   that gates every performance number.
3. **Lightweight MoE trace instrumentation** (opt-in) to record per-invocation
   phase, layer, M, tokens, active requests, top-k, per-expert token counts,
   load imbalance (CV / Gini) and latency — so the regime → shape → routing →
   bottleneck chain is measured, not assumed.
4. **Profile assembly + strategy comparison**: default / global-best /
   regime-aware / per-shape oracle, plus the transfer matrix.

## 6. The negative control we must respect

`docs/2026-07-20/regime_sweep_kernel_changes.md` and `plan.md` record an earlier,
carefully validated result: **kernel changes that looked good in isolation did
not transfer end-to-end** (a 1.23× isolated-layer win became +1.17 % E2E, not
significant; a custom MoE kernel was +1.4 % at batch 1 and −7 % under
concurrency).

This study must therefore:

* report microbenchmark and E2E speedups as **separate** quantities and never
  present the former as the latter;
* run E2E only where the microbenchmark shows a real crossover;
* treat "kernel config specialization helps the kernel but not the service" as a
  **publishable result**, not a failure.

The difference from the earlier work is meaningful, though: that study changed
kernel *implementations*; this one changes kernel *configurations* for shapes the
vendor never tuned on this GPU/Triton build. The prior negative does not predict
this outcome, but it does set the evidential bar.

## 7. Minimum viable experiment

Fixed: 1× H200, TP1, BF16, same SGLang/Triton/PyTorch/CUDA as the serving
campaign (sglang 0.5.12.post1 @ `17f7a1da1`, torch 2.9.1+cu128, triton 3.5.1,
driver 580.105.08).

1. **Trace** the three regimes with instrumentation on; extract the real
   (M, routing) distribution per regime.
2. **Tune** a pruned config search over an M sweep `1,2,4,8,16,32,64` plus the
   real per-regime M values, for both MoE shapes.
3. **Transfer matrix**: every tuned profile × every M, plus a routing control
   (uniform / skewed / real-decode / real-prefill at fixed M) to separate "M
   decides the winner" from "routing decides the winner".
4. **Strategies**: default vs global-best vs regime-aware (2–3 profiles) vs
   per-shape oracle.
5. **E2E**: only for regimes with a demonstrated crossover; swap profiles via
   `SGLANG_MOE_CONFIG_DIR` with serving knobs frozen at the validated per-regime
   winner; 5 reps + bootstrap CI.

## 8. Cost estimate

| stage | unit cost | count | total |
|---|---|---|---|
| trace collection | ~3 min/regime/model | 3 × 2 | ~0.3 GPU-h |
| pruned tuning sweep | ~0.4 s/config (100 iters) | ~120 configs × 11 M × 2 shapes | **~0.3 GPU-h** |
| transfer matrix + repeats | 5 repeats × 100 iters | ~10 profiles × 11 M × 2 | ~0.4 GPU-h |
| routing control | as above | 4 routings × 5 M × 2 | ~0.1 GPU-h |
| correctness | fast | all candidates | ~0.1 GPU-h |
| **P0 kernel-level total** | | | **≈1.2 GPU-h** |
| E2E (per regime: 3 profiles × 5 reps) | ~4 min/rep incl. startup | 3 × 5 × 3 regimes × 2 models | ~4–6 GPU-h |

Kernel-level P0 is well under the 2 GPU-hour reporting threshold and can start
immediately. E2E exceeds it and will be scoped and reported before launching,
per the plan's rule.

## 9. Principal risks

| risk | mitigation |
|---|---|
| **Micro win does not transfer to E2E** (precedent exists, §6) | designed for: E2E is a separate, gated stage; a negative is reported honestly with the waterfall |
| MoE is not the dominant kernel in some regime (e.g. long prefill is attention-bound) | trace measures GPU-time share per kernel first; if MoE share is small in a regime, that itself answers RQ1 and we say so |
| CUDA graph capture may pin one config per captured batch size | E2E uses static per-server profiles (plan's stage 1), which is graph-compatible; dynamic dispatch is explicitly P1 |
| Tuning overfits to a synthetic routing distribution | routing control experiment is part of P0, and real traced routing is used |
| Config JSON nearest-M lookup makes "regime-aware" and "oracle" collapse | measured directly by the profile-count sweep in strategy comparison |

## 10. Decision

Proceed to P0 with **both** models (both are cheap at kernel level and they have
very different MoE shapes — E=32/N=1792/top-4 vs E=128/N=768/top-8 — which
strengthens RQ1/RQ3), and the three regimes A/B/C mapped onto the existing frozen
workloads. No new CUDA kernels, no serving-runtime refactor, no re-running old
autotuning.
