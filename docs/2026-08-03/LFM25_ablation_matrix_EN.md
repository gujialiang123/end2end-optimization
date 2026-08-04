# LFM2.5-8B-A1B — Optimization Ablation Matrix

**Model** LFM2.5-8B-A1B · **Hardware** 1×H200 · **Precision** BF16 · **TP** 1
**SGLang** 0.5.12.post1 @ `17f7a1da1` · torch 2.9.1+cu128 · Triton 3.5.1 · CUDA 12.8
**Date** 2026-08-03 · **Metric** request throughput (req/s)

Three optimization layers are applied independently and in combination, giving a
**2³ factorial**. Every cell is a separate serving benchmark against the same
model and workload; only the layer under test varies.

| | Layer | What it changes | Touches source? |
|---|---|---|:--:|
| **L1** | Serving config tuning | 4 server flags: `max_running_requests`, `chunked_prefill_size`, `schedule_policy`, `mem_fraction_static` | No |
| **L2** | Kernel config tuning | `fused_moe_kernel` tile parameters (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`) for `E=32, N=1792` on H200 | No |
| **L3** | Kernel rewrite / fusion | 7 code changes, **4 of them hand-written Triton kernels** | **Yes** |

---

## 1. The matrix

Each cell: **absolute req/s** and **(Δ vs the cookbook baseline of that regime)**.
⬜ = not measured. Blank cells are left blank rather than interpolated.

**Cell markers**: † = measured in a *different campaign* with its own cookbook baseline, so
only the ratio transfers, not the absolute value. ‡ = older n=6 measurement taken before
arm-order counterbalancing was adopted. § = throughput is the wrong yardstick for this
regime; see the latency table below the matrix.

| Regime | Workload | **S0** cookbook | **L1** only | **L2** only | **L3** only | **L1+L2** | **L1+L3** | **L2+L3** | **L1+L2+L3** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** low-batch decode | synthetic · in≈100, out=256, conc=1 | **1.6863**<br>±0.0027 | **1.6767**<br>±0.0025<br>(−0.57%) ³ | 1.6872<br>(+0.05%) **n.s.** | 1.7992<br>**(+6.70%)** | **1.6791**<br>±0.0021<br>(−0.43%) | **1.8018**<br>±0.0018<br>**(+6.85%)** | **1.7944**<br>**(+6.41%)** | **1.8025**<br>±0.0019<br>**(+6.89%)** |
| **B** concurrent decode | synthetic · in≈200, out=256, conc=32 | **21.661**<br>±0.17 | 22.234 †<br>(+1.11%) | **22.026**<br>±0.18<br>**(+1.68%)** | **23.118**<br>±0.16<br>**(+6.72%)** | ⬜ | ⬜ | **23.537**<br>±0.18<br>**(+8.66%)** | ⬜ |
| **C** long prefill | synthetic · in≈4000, out=32, conc=4 | **12.119**<br>±0.116 | **21.530**<br>±1.37<br>**(+77.7%)** | **14.939**<br>±0.123<br>**(+23.27%)** | 12.869<br>±0.182<br>(+6.19%) | **20.413**<br>±1.9<br>(+68.4%) ² | **22.879**<br>±1.07<br>**(+88.8%)** | **16.392**<br>±0.200<br>**(+35.26%)** | **21.717**<br>±0.5<br>(+79.2%) ² |
| **D** medium balanced | synthetic · in≈800, out=256, conc=8 | **6.900**<br>±0.019 | **7.074**<br>±0.085<br>**(+2.52%)** | **7.027**<br>±0.013<br>**(+1.85%)** | **7.472**<br>±0.018<br>**(+8.29%)** | **7.162**<br>±0.096<br>**(+3.80%)** | **7.668**<br>±0.089<br>**(+11.13%)** | **7.598**<br>±0.044<br>**(+10.13%)** | **7.762**<br>±0.091<br>**(+12.49%)** |
| **E** shared prefix | agentic · 8 groups × 16, sys 2048 / q 128 / out 256 | **14.220**<br>±0.024 | **27.552**<br>±0.36<br>**(+93.76%)** | **15.995**<br>±0.031<br>**(+12.49%)** | **15.249**<br>±0.031<br>**(+7.24%)** | **31.430**<br>±0.63<br>**(+121.0%)** | **28.059**<br>±0.40<br>**(+97.3%)** | **17.237**<br>±0.038<br>**(+21.22%)** | **32.218**<br>±1.02<br>**(+126.6%)** |
| **F** tool agent | **real trace** · mooncake toolagent, n=200, conc=64 | **5.2646**<br>±0.0085 § | 5.280 †<br>(+0.31%) | **5.2724**<br>±0.0080<br>(+0.15%) § | **5.2857**<br>±0.0084<br>(+0.40%) § | ⬜ | ⬜ | **5.2952**<br>±0.0053<br>(+0.58%) § | ⬜ |

🔄 = in flight. ¹ ² ³ see the footnotes below.

### Regime F, the only real trace, measured on latency

Throughput on an agentic trace is set by the client's think time between turns, not by the
server: both arms retire the same 200 requests in nearly the same wall clock. The same runs,
pooled across both arm orders, n=16 per arm:

| metric | S0 cookbook | + **L3** | change | p |
|---|---:|---:|---:|---|
| TTFT p50 | 321.2 ms | 295.8 ms | **−7.91%** | 1.8e-12 |
| TTFT p95 | 537.2 ms | 496.8 ms | **−7.53%** | 5.2e-18 |
| TPOT p50 | 3.294 ms | 3.147 ms | **−4.47%** | 1.2e-13 |
| TPOT p95 | 23.56 ms | 26.13 ms | +10.91% | 0.40 **n.s.** |
| E2E p50 | 511.8 ms | 470.9 ms | **−7.98%** | 3.1e-06 |
| E2E p95 | 2104.1 ms | 1972.6 ms | **−6.25%** | 2.0e-24 |
| E2E mean | 901.7 ms | 847.2 ms | **−6.04%** | 9.5e-22 |
| request throughput | 5.2646 | 5.2857 | +0.40% | 3.4e-08 |

And with the tuned MoE config underneath, i.e. **L2 → L2+L3**:

| metric | **L2** | + **L3** | change | p |
|---|---:|---:|---:|---|
| TTFT p50 | 225.3 ms | 203.4 ms | **−9.73%** | 4.4e-10 |
| TTFT p95 | 380.3 ms | 345.9 ms | **−9.03%** | 1.7e-09 |
| TPOT p50 | 3.254 ms | 3.085 ms | **−5.17%** | 2.3e-16 |
| TPOT p95 | 30.56 ms | 30.09 ms | −1.55% | 0.71 **n.s.** |
| E2E p95 | 1978.8 ms | 1856.3 ms | **−6.19%** | 6.8e-17 |
| E2E mean | 797.9 ms | 745.8 ms | **−6.53%** | 3.4e-18 |
| request throughput | 5.2724 | 5.2952 | +0.43% | 1.2e-08 |

**The kernel work is worth 6–10% of latency on the one workload we did not design.** Reporting
only throughput would have recorded this regime as a null result. TPOT p95 is the single
metric moving the wrong way and it is not significant in either baseline; a few turns with
long tool gaps dominate that tail.

Two further points come out of the second table. The kernel gain is **larger** on the tuned
baseline (−9.03% vs −7.53% on TTFT p95), the same direction regime C showed on throughput.
And **L2 is far from neutral here even though it barely moves throughput**: it takes TTFT p95
from 537 ms to 380 ms, −29%, while request throughput goes from 5.2646 to 5.2724, +0.15%. On
a self-paced trace every layer is a latency effect.

> **★ The kernel layers cover only A, B and C.** L1 was run over all six regimes
> (192 configurations × 6 workloads × 2 models); L2 and L3 were only ever run on the three
> synthetic ones. **In particular, the only workload in the suite that is a real trace —
> F, tool agent — has no kernel-layer measurement at all.** See §5, gap #0.
>
> Partial coverage does exist at the profiling level: the NCU hardware-counter study
> (`results/2026-07-10_v9_ncu_realworkload/`) was run on an agentic workload (in≈2700),
> under the tuned config, and is the source of the "occupancy is 12–25 % even after
> tuning" observation in §5. But no end-to-end kernel A/B was ever run there.

**Sample sizes and significance**

| Cell | n | p |
|---|---:|---|
| A: L2 only | 16 vs 16 | 0.34 (**not significant** — by design, see §3.2) |
| A: L3 only | 16 vs 16 | 2.1e-41 |
| A: L2+L3 (increment over L2) | 16 vs 16 | 1.8e-34 |
| B: L3 only | 6 vs 6 | 2.4e-08 |
| C: L2 only | 16 vs 16 | 1.1e-33 |
| C: L3 only | 16 vs 16 | 4.5e-13 |
| C: L2+L3 (increment over L2) | 16 vs 16 | 9.5e-19 |
| L1 (all regimes) | 5 reps × 35 configs validated | CI-backed, see §3.1 |

**Footnotes**

† **The L1 column comes from a different campaign** (`2026-07-24_serving_ceiling_validation`)
with its own cookbook baseline (A=1.6814, B=21.990, C=12.604, D=7.108, E=14.081, F=5.264).
**Only the ratios are comparable across columns, not the absolute values.** An internally
consistent L1 column for regime C is being measured now (see §5, gap #4).

‡ **Regime B's L3 cell is the older n=6 measurement** taken before we adopted arm-order
counterbalancing. A and C have since been re-measured at n=16. See §5, gap #1.

¹ A separate study measured L2 on regime B at 1.005× (n=8), but without order
counterbalancing, which we later found produces a 1.7% position effect — larger than half
the effect being measured. Now superseded by the counterbalanced n=16 measurement in the
matrix (+1.68%, p=8.5e-07).

² **Do not quote these two cells as an L2 effect.** They are sound as a baseline for the
L3 comparison beside them, but the L2 step itself — L1 → L1+L2, reading −5.19% — is a
*between-server-lifetime* comparison on the shortest measurement window in the matrix.
`R_long_prefill` under the L1 config runs for **0.196 s** per repetition; the two no-config
lifetimes differ by 1.86 req/s while the two config lifetimes differ by 0.17, so the 1.12
gap between the pooled means is smaller than one lifetime's own spread, and the p-value
assumes an independence the repetitions inside a lifetime do not have.

The mechanism behind it is nonetheless real and was confirmed without timing anything.
sglang logs `#new-token` for every prefill batch and picks the config bucket by nearest
neighbour, so `scripts/analyze_moe_bucket_usage.py` reads the bucket histogram straight off
the server logs:

| serving config | prefill batches | buckets selected |
|---|---:|---|
| cookbook (`chunk=-1`) | 36 | **4096** (24), **8192** (12) |
| L1 ceiling (`chunk=2048`) | 171 | 512 (38), **1024 (117)**, 1536 (6), 2048 (10) |

**The two distributions do not overlap.** The config was swept where M ≥ 4000; under L1
those buckets are never selected at all, and 68% of forwards land in the 1024 bucket, whose
`BLOCK_SIZE_M=128` gives eight blocks of work to 132 SMs. So **a tuned kernel config is
tied to the serving config it was tuned under** — the same shape of claim as our earlier
"regime→backend rules do not transfer across models".

³ **The L1 "ceiling" on regime A does not survive re-measurement.** `cap8/chunk−1/fcfs/
mem0.85` won the 2026-07-24 sweep at +0.38% over the cookbook; measured here in the same
harness as everything else in this row, at n=24 with order counterbalancing, it is
**−0.57%**. Both orders agree (1.6788 and 1.6745 vs the cookbook's 1.6863). The reading is
that serving autotuning on this regime buys **nothing at all** — the best of 192
configurations cannot beat the cookbook across an independent re-measurement — which is a
stronger form of "autotuning has run out" than a small positive number would have been.

### 1.1 The L1 ceilings are not throughput-for-latency trades

A reasonable objection to stacking on top of L1 is that its large wins might be bought by
destroying latency. **On the validated (n=5) ceilings they are not.** Sign convention:
**positive = better** (higher throughput, lower latency).

| Regime | req/s | TTFT p95 | TPOT p95 | Verdict |
|---|---:|---:|---:|---|
| A low-batch decode | +0.4 % | +11.8 % | −0.0 % | flat |
| B concurrent decode | +1.1 % | +1.2 % | +1.4 % | flat |
| **C long prefill** | **+56.9 %** | **208.5 → 94.0 ms (+54.9 %)** | 3.56 → 3.89 ms (−9.3 %) | **favourable** |
| D medium balanced | +1.8 % | −14.2 % | −3.0 % | mild regression |
| **E shared prefix** | **+93.6 %** | **7450 → 389 ms (+94.8 %)** | 7.66 → 8.03 ms (−4.8 %) | **strongly favourable** |
| F tool agent | +0.3 % | 530.7 → 310.7 ms (+41.4 %) | 24.5 → 18.2 ms (+25.8 %) | everything improves |

On regime C, throughput rises 57 % **and TTFT p95 more than halves**; the only cost is 9 %
on TPOT p95. On shared prefix, TTFT p95 drops from 7.4 s to 0.39 s. On the real trace,
every one of the three metrics improves.

> **Correction.** An earlier version of this document reported the tool-agent ceiling as
> costing **−221 % TPOT p95**. That number is real but belongs to a **different
> configuration** (`cap48·chunk2048·lpm·mem0.80`) selected from the **n=1 coverage pass**,
> and that configuration **did not survive validation** — the n=5 winner is
> `cap128·chunk8192·lpm·mem0.75`, which improves TPOT by 25.8 %. Taking a maximum over 192
> single-shot measurements is largely a maximum over noise, which is exactly why the
> validation pass exists. **The finding that single-objective optimization *can* produce a
> config that is strictly worse for the user still stands** — it is just a statement about
> the search, not about the validated ceiling.

**Consequence for the baseline choice.** Because the validated L1 ceilings are not bad
trades, "cookbook" is *not* automatically the right base to stack on. For regimes C and E
the cookbook is demonstrably **not** the best serving configuration, and Debadeepta's
framing — *beyond what the best auto tuning config provides* — points at the L1 ceiling,
not at the cookbook. The deliverable should therefore report **both**:

- **cookbook → L2 → L3** — what a default deployment gains (fully measured today, §2)
- **L1 ceiling → L2 → L3** — what remains after serving config is already optimal
  (in flight, §5 gap #4)

and print TTFT p95 / TPOT p95 next to every bar, so no reader has to wonder whether the
throughput was bought with latency.

---

## 2. The measured waterfall (regime C, long prefill)

This is the one path through the matrix that is fully measured and internally consistent
(same campaign, same tree, n=16 per cell, arm order counterbalanced).

| Stage | Configuration | req/s | vs previous | vs cookbook |
|---|---|---:|---:|---:|
| **S0** | cookbook: `cap32 / chunk−1 / lpm / mem0.85`, stock MoE config, stock kernels | **12.119 ± 0.116** | — | 1.000× |
| **S2′** | **+ L2** tuned MoE config | **14.939 ± 0.123** | **+23.26%** (p=1.1e-33) | 1.233× |
| **S3′** | **+ L3** kernel rewrite | **16.392 ± 0.200** | **+9.73%** (p=9.5e-19) | **1.353×** |
| *(control)* | L3 **without** L2 | 12.869 ± 0.182 | +6.18% (p=4.5e-13) | 1.062× |

### The headline result

> **After the best available kernel autotuning, kernel rewriting still contributes
> +9.73 % end-to-end (p = 9.5e-19, 8/8 repetitions non-overlapping in both arm orders).**

### And the increment *grows* rather than shrinks

The same kernel rewrite is worth **+6.18 % on the untuned baseline** and **+9.73 % on the
tuned one**. Some of that is Amdahl — L2 only touches the fused-MoE GEMM, which is 73.6 %
of long-prefill kernel time, and L3 touches only the remaining 26.4 %, so shrinking the
denominator raises the relative gain. But the measured value **exceeds** the orthogonal
Amdahl bound (+6.62 %), which it should not.

Adding a six-component arm isolates where the excess comes from:

| Source | without L2 | with L2 | contribution to the +3.54 pt difference |
|---|---:|---:|---:|
| six components (all avoid the MoE GEMM) | +6.41 % | +8.47 % | **+2.06 pt — Amdahl** |
| `moesum` marginal (`all7 − six`) | **−0.08 %** (p=0.88, neutral) | **+1.69 %** (p=2.8e-04) | **+1.49 pt — real interaction** |
| total (`all7`) | +6.18 % | +9.73 % | +3.54 pt |

`moesum` is the one component that changes what `FusedMoE` *returns* (un-combined
`[T, top_k, H]` partials instead of a reduced tensor), so it shares the
`intermediate_cache` layout with the very GEMM whose tiling L2 retunes. It is worth
nothing on the untuned MoE and +1.69 % on the tuned one.

> **How we state this, and how we do not.**
> We report: *kernel rewriting contributes +9.73 % on top of the best tuned kernel config;
> this exceeds the increment on the untuned baseline (+6.18 %), and the difference is
> attributable to a single component with a structural contact point to the MoE GEMM
> output layout, whose interaction mechanism is not yet confirmed at the profile level.*
> We do **not** claim the layers are generally super-additive: n = 1 regime, 1 component.
> **The six-component arm (+8.47 %) excludes this interaction and is the conservative
> number.**
>
> Every other composition we have measured is strongly **sub**-additive (§3.3).

---

## 3. What each layer is

### 3.1 L1 — Serving config tuning (no source changes)

Four server flags. The search is a **full grid enumeration, 8 × 3 × 2 × 4 = 192
configurations** — not sampling, so there is no sampling bias — followed by a 5-repetition
validation pass over the top 35.

| Knob | Values |
|---|---|
| `max_running_requests` | 8, 16, 24, 32, 48, 64, 96, 128 |
| `chunked_prefill_size` | −1, 2048, 8192 |
| `schedule_policy` | lpm, fcfs |
| `mem_fraction_static` | 0.75, 0.80, 0.85, 0.90 |

Per-regime validated ceiling:

| Regime | Winning knobs | cookbook | ceiling | gain |
|---|---|---:|---:|---:|
| A low-batch decode | `cap8 · chunk−1 · fcfs · mem0.85` | 1.6814 ± 0.006 | 1.6878 ± 0.002 | +0.38 % |
| B concurrent decode | `cap64 · chunk8192 · fcfs · mem0.75` | 21.990 ± 0.081 | 22.234 ± 0.166 | +1.11 % |
| **C long prefill** | **`cap8 · chunk2048 · fcfs · mem0.90`** | 12.604 ± 0.382 | **19.781 ± 0.295** | **+56.94 %** |
| *(medium balanced)* | `cap8 · chunk2048 · fcfs · mem0.90` | 7.108 | 7.235 | +1.79 % |
| *(shared prefix)* | `cap96 · chunk2048 · lpm · mem0.90` | 14.081 | 27.262 | +93.61 % |
| *(tool agent, real trace)* | `cap128 · chunk8192 · lpm · mem0.75` | 5.264 | 5.280 | +0.31 % |

Three findings worth keeping:

- **Three of six regimes are a genuine plateau** (+0.3 % to +1.1 %). An independent
  100-trial Optuna study with **no warm start** reaches within 1 % of its final best at
  configuration 7, and the last 20 configurations improve the best-so-far by **0.0 %**.
- **Two regimes have a real cliff, and it is a *capacity* cliff.** Both winners change
  batching and enable chunking. This is a multi-knob effect and must not be attributed to
  chunked prefill alone. Unlike the n=1 coverage pass suggested, the *validated* winners
  are **not** throughput-for-latency trades — see §1.1.
- **The downside is an order of magnitude larger than the upside.** Worst configuration on
  concurrent decode is **−64.9 %** against a best of +1.1 %. Serving knobs are a
  *don't-fall-off-the-cliff* lever, not a speed lever.
- **Single-shot rankings are not stable.** The n=1 coverage pass and the n=5 validation
  pass disagree on the winning knobs for every cliff regime (long prefill:
  `cap24·mem0.75` vs `cap8·mem0.90`; tool agent: `cap48·chunk2048·mem0.80` vs
  `cap128·chunk8192·mem0.75`). Only `chunked_prefill_size=2048`+`fcfs` survives both on
  long prefill. Any config picked from a single measurement should be re-validated before
  it is believed.

### 3.2 L2 — Kernel config tuning (no source changes)

Retunes the tile parameters of the **existing** upstream `fused_moe_kernel`. LFM2.5's MoE
shape is `E=32, N=1792`; upstream PR #22791 already ships tuned configs for H100 / B200 /
MI325X but **not H200**, so larger prefill shapes fall back to a two-tier heuristic — the
server itself logs `Performance might be sub-optimal!`.

- 468–894 candidates swept per token-count bucket, 19 buckets (aligned with upstream's
  H100/B200 files)
- **Every candidate is correctness-gated before it is timed** — ~9 000 benchmarked
  configurations, **0 correctness failures**
- **Guarded policy**: for `M ≤ 32` the emitted config is field-for-field identical to the
  default. CUDA-graph-captured decode batches all fall in that range, so **L2 is neutral on
  decode by construction** — which is exactly what row A of the matrix shows (+0.05 %,
  p = 0.34).

Getting here required three corrections that are themselves transferable findings:
we were tuning a kernel variant the server never executes; CUDA-graph capture bakes the
config in at capture time so decode cannot be retuned afterwards; and **`M` is the token
count, not `tokens × top_k`** — the profile keys were off by a factor of `top_k`, hiding
real headroom behind misaligned buckets. Only a live trace exposed that.

### 3.3 L3 — Kernel rewrite / fusion (source changes)

Seven changes in two classes.

#### Call-site fixes — the fused primitive already shipped, the model does not call it

| Component | The defect | The fix |
|---|---|---|
| **`norm`** | The decoder layer takes a `residual` argument and **overwrites it on the first line**, so `RMSNorm` never receives it and never dispatches to `fused_add_rmsnorm`; both adds run as their own elementwise kernels | Switch to the deferred-residual convention every other SGLang model uses. The residual is carried as a debt and settled by the next layer's norm kernel. **−2 kernels per layer × 24 layers = 48** |
| **`qkrope`** | `sgl_kernel.fused_qk_norm_rope` merges both head-wise RMSNorms and RoPE into one in-place CUDA kernel and **Qwen3-MoE already calls it**; LFM2.5 runs all three separately — 1.65 % of decode and 3.61 % of prefill kernel time | Call the fused kernel on the packed QKV, guarded on bf16, `head_dim == 64`, unscaled RoPE |
| **`scale`** | `config.json` ships `routed_scaling_factor: 1.0`, but the multiply is unconditional — **22 kernels per forward read and rewrite the entire `[T, 2048]` activation to multiply by one** | Skip when the factor is exactly 1.0. **Bit-exact** |
| **`idx`** | `req_pool_indices.to(int32)` is recomputed in **each of the 18 conv layers** for a 12-byte tensor — pure launch overhead, ~1.3 % of low-batch decode kernel time | Cache per forward, keyed on source-tensor identity so a stale cache cannot be returned |

#### Hand-written Triton kernels

| Component | The defect | The fix | Isolated |
|---|---|---|---|
| **`conv`** (2 kernels) | `causal_conv1d_fn` requires `[dim, seqlen]` with unit last stride, so **the layout change cannot be avoided, only absorbed**. Both the materialised transpose and the transposed read in `C_gate * conv_out` are uncoalesced: 18 layers move 8.79 GB in 10.3 ms = **0.83 TB/s against 4.8 TB/s of HBM — 17 % of peak** | One tiled kernel per side folding chunk + gating multiply + transpose into a single pass, with the transpose held in registers/shared memory via `tl.trans` | **5.93× / 4.33×** at T=16000, **0.98 → 3.46 TB/s (17 % → 72 % of peak)**, bit-exact at every shape. Guarded below T=2048 |
| **`moesum`** (1 kernel) | The MoE top-k reduction writes `[T, H]` to HBM and the **next layer's** `fused_add_rmsnorm` reads it straight back. Both are row-wise — a wasted round trip | `FusedMoE` returns its four weighted expert outputs; one kernel does reduction + residual add + RMSNorm | **2.46×** at T=1, **2.68×** at T=8, 1.30× at T=16000 — but **0.72–0.74×** at T=128..1024. Guarded to `T ≤ 32 or T ≥ 4096` |
| **`gate`** (1 kernel) | On decode, `B_gate * x` reads **strided rows** of `proj`. The access is coalesced, but the strided rows stop `TensorIterator` from vectorising — the trace shows the scalar `elementwise_kernel` instead of `vectorized_elementwise_kernel<8>` | A Triton kernel reading `proj` directly | Bit-exact |

> `conv` and `moesum` have **opposite** shape dependence: `conv` needs large T to amortise
> Triton's ~30 µs launch floor, `moesum` *saves* launch overhead plus a round trip and so
> wins at small T. Together they cover the whole range.

#### Per-component attribution (on the cookbook baseline, n=6, each against its own paired baseline)

| Component | Class | A low-batch decode | B concurrent decode | C long prefill |
|---|---|---:|---:|---:|
| `norm` | wiring | +2.35 % | +2.89 % | +1.42 % |
| `scale` | wiring | +1.40 % | +1.02 % | +0.73 % **n.s.** |
| `norm+scale` | wiring | +4.20 % | +3.68 % | +1.60 % |
| `conv` | **Triton ×2** | +0.13 % **n.s.** | −0.03 % **n.s.** | **+2.33 %** |
| `norm+scale+conv` | mixed | +3.89 % | +3.65 % | +3.47 % |
| `qkrope` | wiring | +0.93 % | **+5.42 %** | +1.99 % |
| `gate+idx` | Triton ×1 + cache | −0.00 % **n.s.** | +0.65 % **n.s.** | +0.40 % **n.s.** |
| `moesum` | **Triton ×1** | **+4.55 %** | +3.08 % | ⬜ never measured alone |
| **six components** | | +4.60 % | +6.01 % | +5.81 % |
| **all seven** | | **+6.57 %** | **+6.21 %** | **+5.30 %** |

**Four different shapes of gain.** `norm+scale` removes a fixed number of kernels per
forward regardless of how much work that forward does, so it dominates on decode and is
diluted on long prefill. `conv` removes traffic that grows with token count and needs
T ≥ 2048, so decode never reaches it. `qkrope` removes work in the 6 attention layers, so
concurrent decode benefits most. `moesum` removes launch overhead plus a round trip, so
low-batch decode benefits most. **Measuring one regime would have shown none of this.**

**`gate+idx` is an honest negative** — the mechanism is real and measurable at kernel level
(1–2 %) but does not survive to end-to-end in any regime.

> ⚠️ **This whole table is measured on the untuned baseline, and we now know that matters.**
> `moesum` moves from −0.08 % (p=0.88, neutral) to +1.69 % (p=2.8e-04) between the untuned
> and tuned baselines — it changes sign and significance. **The attribution of the other six
> components on the tuned baseline is therefore unverified.** See §5, gap #2.

#### Combined gains are strongly sub-additive

| Regime | Sum of components measured individually | Measured together | Realization |
|---|---:|---:|---:|
| C long prefill | 5.86 % | 5.30 % | 0.90 |
| A low-batch decode | 9.37 % | 6.57 % | 0.70 |
| B concurrent decode | 12.80 % | 6.21 % | **0.49** |

On concurrent decode, `qkrope` alone is worth +5.42 %; adding a group worth +3.65 % on its
own buys **0.12 points**. The components are removing overlapping per-forward overhead, and
**the realization rate tracks how saturated the regime is** — long prefill has the most work
per forward to hide overhead behind and loses the least.

> **Never report the sum of individually measured components. Any combination that will
> actually be deployed must be measured as a combination.**

---

## 4. Experimental configuration

### 4.1 Cookbook baseline — full launch command

```bash
python -m sglang.launch_server \
    --model-path /data/hf/LFM2.5-8B-A1B \
    --served-model-name lfm2.5-8b-a1b \
    --host 127.0.0.1 --port <PORT> \
    --tensor-parallel-size 1 \
    --context-length 8192 \
    --schedule-conservativeness 1.0 \
    --trust-remote-code \
    --moe-runner-backend auto \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --chunked-prefill-size -1 \
    --schedule-policy lpm \
    --max-prefill-tokens 16384
```

Resolved server args, **transcribed from a real server log** rather than inferred:

| Argument | Value | |
|---|---|---|
| `disable_cuda_graph` | **False** | **CUDA graph is ON in every arm** |
| `cuda_graph_max_bs` | 256 | captured `bs [1,2,4,8,12,16,24,32]` — equals `max_running_requests`, so the whole decode path is graph replay |
| `enable_torch_compile` | False | |
| `enable_piecewise_cuda_graph` | False | |
| `disable_radix_cache` | False | radix cache on |
| `disable_overlap_schedule` | False | overlap scheduling on |
| `enable_fused_qk_norm_rope` | **False** | upstream has since added a server-level switch for this fusion, defaulting off; our `qkrope` change wires it at the model call site, a different path |
| `attention_backend` | `fa3` | |
| `moe_runner_backend` | `auto` | |
| `dtype` / `kv_cache_dtype` | `auto` / `auto` | BF16 |
| `quantization` / `speculative_algorithm` | `None` / `None` | |
| `page_size` | 1 | |

### 4.2 A/B methodology

- **L3 toggled by** the `LFM_FUSION_PATCH` environment variable. Unset means the
  **byte-for-byte unmodified SGLang path** — same tree, same commit, same server args. The
  baseline is a real baseline, not a rebuild.
- **L2 toggled by** `SGLANG_MOE_CONFIG_DIR`. Pickup is verified in the server log.
- **Arm order is counterbalanced.** The harness runs arms sequentially, which produces a
  measurable position effect: on regime C the baseline reads 12.020 in forward order and
  12.219 in reverse — **1.7 %, larger than half the effect under test**. Every cell in §2
  pools `{forward, reverse}` × 8 repetitions from 2 independent server lifetimes, n=16.
- **Effect verification.** The server log is checked for the patch marker and the config-load
  line; otherwise a silently-inactive patch would be recorded as "identical to baseline".
- **Statistics.** Welch t with an **exact Student-t tail**. (A normal approximation is
  anti-conservative at n=6; switching to the exact tail changed no conclusion but did change
  individual p-values.)

### 4.3 Correctness

**Token-identity is structurally unusable for this model.** LFM2.5 routes top-4-of-32 and
expert selection is a discrete `argmax`, so a bf16-level perturbation can flip which expert
is chosen and change the output discontinuously. Any change that is not bit-identical trips
this gate.

Validated instead on **full GSM8K, 1319 questions, greedy decoding**:

| Arm | Runs | Mean |
|---|---|---:|
| baseline | 0.348 / 0.349 / 0.344 | 0.3470 |
| **`scale` (provably bit-exact)** | 0.338 / 0.339 / 0.340 | **0.3390** |
| `norm` | 0.362 / 0.368 / 0.361 | 0.3637 |
| `conv` (bit-exact) | 0.342 / 0.350 | 0.3460 |
| `qkrope` | 0.352 / 0.346 | 0.3490 |
| `moesum` (bit-exact) | 0.343 / 0.347 | 0.3450 |
| all seven | 0.371 / 0.364 / 0.370 | 0.3683 |

The `scale` arm is **mathematically identical to baseline** and still reads **0.8 points
lower**. That is not a defect — it measures the harness noise floor for free
(`--parallel 32` varies batch composition between server instances, and batch-dependent
reductions change greedy output). All 8 arms span 2.5 points, inside that floor and inside
the ±2.6-point binomial error at n=1319, p≈0.35.

> **Claim: no quality regression detected.** Not "quality improved" — the experiment cannot
> resolve a difference that small, and the bit-exact arm is the proof of that.

### 4.4 Known defects in the experimental record

These are disclosed rather than left for a reviewer to find.

1. **The 7/27 end-to-end runs all ran with `--skip-correctness`.** Their `correctness.json`
   files contain `outputs: []`. The correctness evidence is the separately-run GSM8K
   campaign above — which was measured on the **cookbook** baseline, **without** the tuned
   MoE config. The delivered combination (L2+L3) has not been quality-tested.
2. **The SGLang working tree carries one uncommitted patch** (a flashinfer_cutlass autotune
   allowlist change, 2026-06-11). Both arms share it, so the A/B is unaffected, but strictly
   the baseline is "`17f7a1da1` + that patch".
3. **L2 has not reached its own ceiling.** A MoE layer runs **two** grouped GEMMs — the
   up projection (`w13`) and the down projection (`w2`) — and SGLang tunes them with
   **two separate config files**, `E=32,N=1792,device_name=NVIDIA_H200.json` and the same
   name with a `_down` suffix
   (`fused_moe_triton_config.py:33` builds the filename from a `down_moe` flag). We only
   produced the first one. When the `_down` file is missing, SGLang falls back to reusing
   the up-projection config for the down GEMM and logs
   `Using MoE kernel config with down_moe=False. Performance might be sub-optimal!`
   — which is exactly what our server logs show. **So roughly half the MoE GEMM work is
   still running on a config that was not tuned for it.**
   Tuning it is constrained rather than free: the runtime asserts
   `config["BLOCK_SIZE_M"] == down_config["BLOCK_SIZE_M"]`
   (`fused_moe_triton_config.py:265`), so the down config must be swept **subject to**
   the up config's `BLOCK_SIZE_M`, not independently.
4. **Two measurements of the same quantity disagree.** L3 alone on regime C reads **+5.30 %**
   in the 7/27 report (n=6) and **+6.18 %** in the current campaign (n=16, counterbalanced,
   after fixing a leaked-server bug that had a previous batch benchmarking a stale process).
   The current number is the more reliable one.

---

## 5. Open cells

| # | Missing | Which cells | Est. | Why it matters |
|---|---|---|---|---|
| **0** | **L2 and L3 on regimes D, E, F** — in particular **F, the only real trace** | 3 whole rows, 18 cells | ~2 h / regime | ★★ Mason's ask is explicitly that the shapes come from real end-to-end runs. Right now every kernel-layer number in this document is from a synthetic workload. E also has the largest L1 cliff (+93.6 %) and exercises the radix cache heavily, so its prefill/decode mix is different from anything we have measured |
| **1** | Regime **B** with the tuned config (L2, L2+L3) | row B | ~30 min | B is one of the three headline regimes and the only one without a clean baseline |
| **2** | **Per-component** L3 attribution on the tuned baseline | §3.3 table | ~2–3 h / regime | ★ `moesum` already changed sign between baselines; the other six are unverified |
| **3** | `moesum` measured alone on regime C | §3.3 table | ~30 min | never measured in isolation there |
| **4** | **L1 stacking** (L1, L1+L2, L1+L3, L1+L2+L3) | 4 columns | in flight | running now for regime C |
| **5** | Sweep the `_down` companion config (subject to the up config's `BLOCK_SIZE_M`), then re-measure L2 and the L3 increment | column L2, L2+L3 | ~1–2 h | ⚠️ **may shrink the +9.73 %** — but without it "beyond the best autotuning" is not defensible |
| **6** | GSM8K on the delivered stack (L2+L3) | §4.3 | ~2 h | current quality evidence does not cover the tuned config |
| **7** | Profile-level evidence for the `moesum` × config interaction | §2 | ~1 h | the super-additivity currently has a plausible mechanism and no measurement |
| **8** | NCU on the final stack (remaining headroom) | new section | ~1 h | closes the "what is left" question |

### Not needed

**Hardening the L1 autotuning ceiling.** That space is already **enumerated exhaustively**
(192 of 192 configurations, plus a 5-repetition validation pass over the top 35). The
"25-trial TPE search may simply have failed" objection applies only to a superseded earlier
study, not to the numbers in this document.

---

## 5.1 How to fill the blank cells

**29 of 48 cells are blank.** They are not 29 separate experiments.

### The unit of work is one harness invocation = 4 cells

`scripts/lfm_fusion/exp3_layered.sh` runs a 2 × 2 × 2 design at a **fixed serving config**:
`{L2 off, L2 on} × {L3 off, L3 on} × {forward, reverse arm order}`, 8 repetitions each,
8 server lifetimes. That fills **four cells of one row**.

Run it **twice per regime** — once at the cookbook serving config and once at that regime's
L1 ceiling — and the **entire row of 8 cells is filled and internally consistent** (same
tree, same campaign, same protocol, so the absolute values are comparable and the `†`
marker goes away).

```
invocation at cookbook serving  →  S0,  L2,     L3,     L2+L3
invocation at L1-ceiling serving →  L1,  L1+L2,  L1+L3,  L1+L2+L3
```

### What remains

| Regime | cookbook invocation | L1-ceiling invocation | L1 winner knobs to add | Est. |
|---|---|---|---|---|
| **A** low-batch decode | ✅ done (exp3) | ⬜ needed | `cap8 · chunk−1 · fcfs · mem0.85` | ~35 min |
| **B** concurrent decode | ⬜ needed | ⬜ needed | `cap64 · chunk8192 · fcfs · mem0.75` | ~70 min |
| **C** long prefill | ✅ done (exp3) | 🔄 in flight (exp5) | `cap8 · chunk2048 · fcfs · mem0.90` ✅ already in harness | — |
| **D** medium balanced | ⬜ needed | ⬜ needed | `cap8 · chunk2048 · fcfs · mem0.90` | ~65 min |
| **E** shared prefix | ⬜ needed | ⬜ needed | `cap96 · chunk2048 · lpm · mem0.90` | ~105 min |
| **F** tool agent | ⬜ needed | ⬜ needed | `cap128 · chunk8192 · lpm · mem0.75` | ~145 min |

**≈ 7 GPU-hours serial.** The invocations are independent, so with six free GPUs this is
**≈ 2 hours wall clock**.

Timing is dominated by server startup (8 lifetimes × ~3.5 min) for the short workloads and
by benchmark time for E (~20 s/run) and F (~42 s/run).

### Code changes needed first (small)

`scripts/lfm_fusion/lf_e2e.py` currently defines only `A/B/C` plus `C_long_prefill_tuned`.
To run the table above it needs:

1. Base entries for `D_medium_balanced`, `E_shared_prefix`, `F_tool_agent`
   (workload names already exist in `serving_ceiling_lib.WORKLOADS`).
2. `*_tuned` entries for A, B, D, E, F carrying the L1 winner knobs above.

No harness logic changes — `run_workload` already dispatches on the workload name, and
`shared_prefix` / `tool_agent` are already wired into the campaign library.

### ⚠️ Sequence this correctly

**Decide gap #5 (the `_down` companion config) *before* running any of the above.** Adding
it changes what the "L2 on" arm *is*, so every cell measured beforehand would have to be
re-measured. The order that avoids rework is:

```
1. sweep the _down config (gap #5, ~1–2 h)          ← changes the L2 arm definition
2. re-measure regime C at cookbook + L1 ceiling      ← re-anchors the headline result
3. fan out A, B, D, E, F across the free GPUs        ← ~2 h wall clock
4. per-component ablation on the tuned baseline      ← gap #2, only for the chosen regime
5. GSM8K on the delivered stack                      ← gap #6
```

### What each new row is expected to teach

These are predictions to be tested, not results.

- **D medium balanced** — the only regime whose token counts sit **between** the two
  Triton kernels' guards (`conv` needs T ≥ 2048, `moesum` wants T ≤ 32 or T ≥ 4096). It is
  the regime where we expect the *fewest* components to fire, and therefore a useful
  negative control for the "four different shapes of gain" claim.
- **E shared prefix** — heavy radix-cache reuse means the actual prefilled token count per
  request is far below the nominal 2048-token system prompt. Whether `conv` still clears
  its T ≥ 2048 guard here is an open question, and the answer determines whether shape
  guards tuned on synthetic workloads survive contact with prefix caching.
- **F tool agent** — **the only real trace.** Every kernel-layer number in this document is
  currently from a synthetic workload, and Mason's stated requirement is that the shapes
  come from real end-to-end runs. This row is what converts the study from "measured on
  benchmarks we designed" to "measured on traffic we did not design".

---

## 6. Data provenance

| Content | Path |
|---|---|
| L1 per-regime ceiling | `results/2026-07-24_serving_ceiling_validation/analysis/lfm25/ceiling_per_regime.json` |
| L1 full grid (192 configs) | `results/2026-07-24_serving_ceiling/` |
| L1 convergence study (100 trials, no warm start) | `results/2026-07-22_lfm25_plateau_100/` |
| L2 | `results/regime_kernel/` |
| L3 per-component (paired baselines) | `results/lfm_fusion/processed/fusion_ab*.csv` |
| L2 × L3 factorial | `results/lfm_fusion/e2e/exp3_layered_*_summary.json` |
| `moesum` marginal | `results/lfm_fusion/e2e/exp3_moesum_marginal_C_long_prefill.json` |
| L1 stacking (in flight) | `results/lfm_fusion/e2e/lfm25_exp3_l1_C_*/` |
| Kernel time breakdown (nsys) | `results/lfm_fusion/nsys/FINDINGS.md` |
| NCU | `results/2026-07-10_v9_ncu_realworkload/` |
| GSM8K | `results/lfm_fusion/correctness/accuracy_*.json` |
| Hand-written Triton kernels | `scripts/lfm_fusion/lf_triton_shortconv.py`, `lf_triton_moesum.py` |
| Source-level port of all 7 changes | `gujialiang123/sglang` PR #1 |
