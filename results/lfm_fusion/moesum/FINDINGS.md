# MoE top-k reduction + residual-add RMSNorm fusion

**Date:** 2026-07-27  
**GPU:** GPU 5, NVIDIA H200  
**Model:** `/data/hf/LFM2.5-8B-A1B`  
**SGLang:** `17f7a1da1`, 0.5.12.post1  
**Kernel:** `scripts/lfm_fusion/lf_triton_moesum.py`

## Verdict

**The premise holds, correctness passes, and the candidate is worth keeping.**

The fused kernel is enabled for decode (`T <= 32`) and long prefill
(`T >= 4096`). It is deliberately disabled for `33 <= T < 4096`, where the
stock CUDA sequence is faster. Against the unmodified serving baseline it
improved request throughput by:

- **+4.55%** in A low-batch decode, Welch `t=61.555`, `p=1.54e-13`;
- **+3.08%** in B concurrent decode, Welch `t=3.857`, `p=0.00318`.

These are standalone results. They must not be added to the prior component
wins: §6c of the main report already showed only 0.57 of separately measured
overhead-removal gains surviving together in concurrent decode.

The requested A/B compares against the **unmodified** baseline. At a fused MoE
boundary this standalone component therefore collapses the baseline's reduction,
multiply-by-one, residual add, and following RMSNorm into one kernel. The nsys
3.835% chain was measured after `norm,scale`, where only the reduction and
already-fused residual RMSNorm remain. No incremental `norm,scale` versus
`norm,scale,moesum` result is claimed here.

## 1. Premise verification

The down-projection writes one weighted output per selected expert:

```text
intermediate_cache3: [T, top_k, H] = [T, 4, 2048]
```

Evidence in the read-only upstream checkout:

- `fused_moe.py:572-602` writes the down-GEMM results to
  `intermediate_cache3`;
- `fused_moe.py:619-631` performs the final reduction;
- decode takes `tokens_in_chunk <= 32` and calls
  `moe_sum_reduce_torch_compile`;
- prefill takes the `>32` branch and calls the CUDA `moe_sum_reduce`;
- `lfm2_moe.py:145-165` configures `top_k=4`, `hidden_size=2048`, and invokes
  `FusedMoE`;
- there are `24 - num_dense_layers(2) = 22` MoE layers.

In the deferred-residual form, each reduced `[T,H]` output is consumed by the
next layer's `operator_norm(hidden_states, residual)`; the last MoE output is
consumed by `embedding_norm(hidden_states, residual)`. Thus all 22 materialized
MoE sums have the required adjacent residual-add RMSNorm consumer.

**Premise verdict: confirmed for both decode and prefill.**

## 2. Implementation

One Triton program handles one token row:

1. load and FP32-accumulate the four BF16 expert outputs;
2. round the sum to BF16, matching the stock reducer's materialized boundary;
3. add the BF16 residual in FP32;
4. write the updated BF16 residual;
5. compute the RMS over `H=2048`;
6. multiply by the norm weight and write the normalized BF16 output.

The patch makes `FusedMoE` use its existing `no_combine=True` mode only at
profitable shapes. At intermediate shapes it restores the original combined
MoE path, so the guard does not retain a large `[T,4,H]` output unnecessarily.

Logical traffic per token row, excluding the amortized weight read:

```text
stock: (top_k + 5) * H * 2 B = 9 * H * 2 B
fused: (top_k + 3) * H * 2 B = 7 * H * 2 B
saved: 2 * H * 2 B = 8192 B
```

Across 22 layers this is 180,224 B at decode `T=1` and 2,883,584,000 B at
prefill `T=16000`.

## 3. Primitive correctness gate

Reference: SGLang's stock decode/prefill MoE reducer followed by
`sgl_kernel.fused_add_rmsnorm`, on random BF16 tensors. No shape was timed
before its correctness check passed.

| T | normalized max \|diff\| | updated residual max \|diff\| | verdict |
|---:|---:|---:|---|
| 1 | 0 | 0 | bit-exact |
| 8 | 0 | 0 | bit-exact |
| 32 | 0 | 0 | bit-exact |
| 128 | 0 | 0 | bit-exact |
| 1024 | 0 | 0 | bit-exact |
| 4096 | 0 | 0 | bit-exact |
| 16000 | 0.00048828125 | 0 | within BF16 rounding |

At `T=16000`, exactly **1 of 32,768,000** normalized elements differed; the
updated residual remained bit-exact. The difference comes from the rowwise
FP32 RMS reduction order, not from the top-k sum or residual update.

## 4. Isolated benchmark and crossover

Times are median isolated sequence latency from CUDA events around the Python
call, so they include submission gaps. This is intentional: it exposes the
same roughly 30–36 us Triton launch floor used to set the shape guard.
Bandwidth is logical traffic divided by measured time.

| T | stock | fused | speedup | stock GB/s | fused GB/s | patch decision |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 90.50 us | 36.77 us | **2.461x** | 0.4 | 0.8 | fuse |
| 8 | 95.58 us | 35.68 us | **2.679x** | 2.9 | 6.1 | fuse |
| 32 | 94.85 us | 35.95 us | **2.638x** | 11.6 | 23.9 | fuse |
| 128 | 24.86 us | 33.78 us | 0.736x | 176.9 | 101.3 | stock |
| 1024 | 24.80 us | 34.27 us | 0.724x | 1417.7 | 798.0 | stock |
| 4096 | 42.18 us | 36.94 us | **1.142x** | 3334.3 | 2960.7 | fuse |
| 16000 | 145.68 us | 111.81 us | **1.303x** | 3770.7 | 3821.3 | fuse |

The supplementary sweep found `1.024x` at `T=3072`, too close to noise to use
as a production crossover. `T=4096` is the conservative prefill threshold.
The lower achieved bandwidth at `T=4096` is not a contradiction: the fused
kernel moves 7 activation-equivalents rather than 9 and still finishes sooner.

Raw data:

- `microbench.json`
- `crossover.json`

## 5. GSM8K quality gate

Required full-set run, five-shot greedy, 1319 questions, two repetitions:

| arm | accuracies | mean |
|---|---|---:|
| established baseline | — | 0.347 |
| `moesum` | 0.343 / 0.347 | **0.345** |

The difference is **-0.2 percentage points**, well inside the measured
0.8-point between-server harness noise floor. **No quality regression was
detected.**

Artifact: `results/lfm_fusion/correctness/accuracy_moesum.json`.

## 6. End-to-end A/B

Both workloads used CUDA graphs, six repetitions per arm, identical serving
knobs, and a checked patch marker.

| regime | baseline req/s | `moesum` req/s | gain | Welch t | p |
|---|---:|---:|---:|---:|---:|
| A low-batch decode | 1.6822 ± 0.0015 | 1.7588 ± 0.0019 | **+4.55%** | 61.555 | 1.54e-13 |
| B concurrent decode | 21.6642 ± 0.2421 | 22.3315 ± 0.2374 | **+3.08%** | 3.857 | 0.00318 |

The `±` values are 95% confidence half-widths from the project analyzer.
Both improvements are statistically significant. The smaller gain under
concurrency is directionally consistent with the previously measured
sub-additivity of fixed-overhead removals.

Artifacts:

- `results/lfm_fusion/e2e/lfm25_moesum/`
- `results/lfm_fusion/processed/fusion_ab_moesum.csv`

## Recommendation

**Keep the component and its shape guard.** It passes primitive and task-level
correctness, wins at both requested serving regimes, and directly removes the
targeted HBM round-trip. Before adding it to the existing `all` deployment arm,
measure that exact combination; the standalone +3.08% concurrent-decode gain
should not be assumed additive with `qkrope` or `norm+scale`.
