# LFM2.5-8B-A1B adjacent-kernel fusion findings

**Date:** 2026-07-27  
**GPU:** GPU 5 only, NVIDIA H200 (`GPU-50a5b4b6-74d9-f89d-32c7-1908ca1a161b`)  
**Model:** `/data/hf/LFM2.5-8B-A1B`  
**SGLang:** `17f7a1da1a371e8fac398032682b8f47f74a6ec6`, 0.5.12.post1  
**Profiler:** Nsight Systems 2025.1.1.0

The runs used the already validated `LFM_FUSION_PATCH=norm,scale` path. Thus
the findings below are what remains after residual-add/RMSNorm fusion and
removal of the routed-scaling multiply by 1.0.

## Bottom line

Ranked by expected value relative to engineering risk:

| rank | opportunity | net kernels removed / forward (decode, prefill) | shape-derived HBM traffic saved (decode, prefill) | measured current chain, % of kernel time (CUDA-graph decode, default prefill) | implementation |
|---:|---|---:|---:|---:|---|
| 1 | Reuse fused packed-QKV **Q/K layout + RMSNorm + RoPE** | **11, 23** | **61,428 B; 1,965,888,000 B** | **1.653%, 3.607%** | Model call-site change plus one hoisted positions cast; existing CUDA kernel |
| 2 | Fuse both ShortConv gates and prefill layout conversion into `causal_conv1d` | **36, 54** | **294,912 B; 7,077,888,000 B** | **1.977%, 6.268%** | New/extended `sgl-kernel` CUDA kernels |
| 3 | Fuse MoE top-4 reduction with the next residual-add RMSNorm | **22, 22** | **180,224 B; 2,883,584,000 B** | **3.835%, 1.971%** | New Triton or CUDA reduction+norm kernel |
| 4 | Apply `SiLU(gate)*up` in the MoE down-GEMM input prologue | **22, 22** | **630,784 B; 10,092,544,000 B** | **1.632%, 2.204%** | Modify/new Triton down-GEMM kernel |
| 5 | Fuse RoPE with KV-cache store using the existing fused buffer argument | **6, 6** | **12,288 B; 196,608,000 B** | **0.335%, 0.064%** | Small call-site change; low ceiling |

The percentages are the **measured footprint of the current chain**, not a
predicted saving. Every proposed fused kernel still has to perform arithmetic.
The corresponding whole-forward hard ceilings are, respectively:

| opportunity | decode hard ceiling vs measured graph wall time | prefill hard ceiling vs measured wall time |
|---|---:|---:|
| Fused Q/K layout+norm+RoPE | 1.244% | 3.470% |
| ShortConv gates+layout | 1.488% | 6.030% |
| MoE sum+next RMSNorm | 2.886% | 1.896% |
| MoE activation in down GEMM | 1.228% | 2.120% |
| RoPE+KV store | 0.252% | 0.061% |

Those are deliberately unrealistic ceilings obtained by assuming the complete
current chain time vanishes. Actual end-to-end gains must be lower and require
an implementation benchmark.

## Profiling runs

`nsys` was restricted to the `cudaProfilerStart/Stop` region emitted by
`bench_one_batch --profile --profile-activities CUDA_PROFILER`.

| run | workload/stage captured | CUDA graph | kernels | kernel time | measured bench wall time |
|---|---|---:|---:|---:|---:|
| `decode_b1_nograph` | B=1, input=100, output=32; one decode | off | 385 | 1,930.696 us | 17,374.168 us median |
| `decode_b1_graph` | same | on | 374 | 1,966.550 us | 2,613.469 us median |
| `prefill_b4x4000_nograph` | B=4, input=4000, output=8; prefill, T=16,000 | off | 519 | 155,206.995 us | 161,125.245 us |
| `prefill_b4x4000_graph` | same | on | 519 | 154,730.349 us | 160,850.091 us |

The two prefill traces are effectively replicates: prefill is eager even when
decode CUDA graphs are enabled. Candidate-chain times agree within 0.5%.

Common nsys flags:

```text
profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
--cuda-graph-trace=node --capture-range=cudaProfilerApi
--capture-range-end=stop --force-overwrite=true --export=sqlite
```

## Launch/dispatch overhead

Two different measurements are reported and must not be added:

1. **Kernel launch API time** is CPU time inside `cudaLaunchKernel`,
   `cuLaunchKernel[Ex]`, etc. It can overlap GPU execution and includes nsys
   tracing overhead.
2. **Device idle between activities** is the single-stream interval not occupied
   by a kernel/memcpy/memset. It is an upper bound on dispatch overhead because
   Python/framework work and synchronization can also cause it.

| run | kernels inside 24 layers | layer kernel time | layer device-idle total | idle / layer | launch-API total in layers | launch API / layer |
|---|---:|---:|---:|---:|---:|---:|
| decode, graph off | 366 | 1,756.711 us | **16,546.060 us** | **689.419 us** | 1,886.614 us | 78.609 us |
| decode, graph on | 354 graph nodes | 1,788.565 us | **35.886 us** | **1.495 us** | 0 per-node | 0 |
| prefill, graph off | 416 | 154,752.241 us | **1,865.638 us** | **77.735 us** | 2,145.181 us | 89.383 us |
| prefill, graph setting on | 416 | 154,272.941 us | 2,158.855 us | 89.952 us | 2,155.709 us | 89.821 us |

The graph trace contains one `cudaGraphLaunch` API call measured at 682.631 us,
but node tracing itself is intrusive; it is not a production graph-launch
latency estimate. The robust observation is the collapse of within-layer gaps
from 689.4 us/layer to 1.5 us/layer.

**Consequence:** a decode fusion that only removes launches but saves no tensor
traffic has little real-serving headroom. Prefill remains eager, so launch and
dispatch reduction can still matter there.

## Exact repeating decoder-layer timeline

Layer 3 is a representative `conv + MoE` layer. Full demangled names are in
`representative_layer3_timeline.csv`; short names and semantic roles are shown
below. Negative sub-microsecond gaps are timestamp quantization/overlap and
should be read as zero.

### Decode: graph off versus real graph path

| order | kernel / role | no-graph duration | gap to next kernel | graph duration | graph gap |
|---:|---|---:|---:|---:|---:|
| 1 | `FusedAddRMSNormKernel` / operator norm | 2.528 us | 75.904 us | 2.400 us | 0.128 us |
| 2 | `nvjet...4x1...` / conv input projection | 8.960 | 22.496 | 9.280 | 0.064 |
| 3 | `vectorized_elementwise_kernel` / `B*x` | 0.992 | 26.944 | 1.120 | 0.096 |
| 4 | `unrolled_elementwise_kernel` / int64→int32 cache index | 1.152 | 21.216 | **absent** | — |
| 5 | `causal_conv1d_update_kernel` | 2.880 | 17.376 | 2.432 | 0.032 |
| 6 | `vectorized_elementwise_kernel` / `C*conv_out` | 1.088 | 35.937 | 1.088 | 0.128 |
| 7 | `nvjet...4x1...` / conv output projection | 5.600 | 21.856 | 5.568 | 0.256 |
| 8 | `FusedAddRMSNormKernel` / FFN norm | 1.920 | 32.608 | 1.952 | 0.064 |
| 9 | `nvjet...1x1...` / router | 4.800 | 38.400 | 4.960 | 0.096 |
| 10 | `topkGatingSigmoid` | 1.984 | **145.344** | 2.080 | 0.064 |
| 11 | `moe_align_block_size_small_batch_expert_kernel` | 4.384 | **79.584** | 4.416 | 0.064 |
| 12 | first `fused_moe_kernel` | 19.841 | 5.280 | 19.776 | ~0 |
| 13 | `act_and_mul_kernel` | 1.408 | **59.040** | 1.600 | ~0 |
| 14 | second `fused_moe_kernel` | 14.752 | **181.088** | 15.104 | 0.064 |
| 15 | `triton_poi_fused_copy__mul_sum_0` | 1.024 | **126.817** | 1.088 | 0.224 |

Without graphs, a 1–4 us kernel is routinely followed by 20–180 us of device
idle. With graphs the same sequence has 0–0.25 us gaps. The no-graph trace is
useful for attribution, but launch-only opportunities must be judged from the
graph column.

### Prefill, T=16,000

| order | kernel / role | duration | gap to next kernel |
|---:|---|---:|---:|
| 1 | `FusedAddRMSNormKernel` | 63.840 us | 4.928 us |
| 2 | conv input `nvjet` GEMM | 521.890 | 1.920 |
| 3 | `B*x` | 89.344 | 1.024 |
| 4 | BF16 direct copy, `[T,H]→[H,T]` | 221.121 | 41.696* |
| 5 | cache-index int cast | 1.248 | 23.552 |
| 6 | `causal_conv1d_fwd_kernel` | 57.920 | 1.216 |
| 7 | `C*conv_out` | 226.881 | 6.144 |
| 8 | conv output `nvjet` GEMM | 174.177 | ~0 |
| 9 | `FusedAddRMSNormKernel` | 60.896 | 1.312 |
| 10 | router GEMM | 20.256 | 1.120 |
| 11 | `topkGatingSigmoid` | 3.104 | 0.928 |
| 12 | `moe_align_block_size_kernel` | 15.008 | 0.928 |
| 13 | `count_and_sort_expert_tokens_kernel` | 23.616 | 1.056 |
| 14 | first `fused_moe_kernel` | 3,176.842 | 1.248 |
| 15 | `act_and_mul_kernel` | 155.072 | 1.152 |
| 16 | second `fused_moe_kernel` | 1,602.277 | 1.024 |
| 17 | `moe_sum_reduce_warp_per_token_vec_kernel` | 76.768 | ~0 |

\* The 41.696 us kernel-to-kernel gap contains a 16-byte D2D copy (1.024 us)
and a 4-byte H2D copy (0.736 us) used to rebuild `query_start_loc`, plus host
dispatch.

## Ranked opportunities in detail

### 1. Packed-QKV layout copies + Q/K RMSNorm + RoPE

The six attention layers currently issue:

```text
prefill: q layout copy -> q RMSNorm -> k layout copy -> k RMSNorm -> RoPE
decode:                 q RMSNorm ->                 k RMSNorm -> RoPE
```

Measured whole-forward chain:

| regime | calls in current chain | time | kernel-time share |
|---|---:|---:|---:|
| graph decode | 18 | 32.512 us | 1.653% |
| default prefill | 30 | 5,581.104 us | 3.607% |

Prefill breakdown is 469.856 us of Q copies, 3,706.668 us of Q norms,
130.624 us of K copies, 936.003 us of K norms, and 337.953 us of RoPE.

SGLang already ships `sgl_kernel.fused_qk_norm_rope`, which:

- operates in-place on packed `[T, (32+8+8)*64]` BF16 QKV;
- supports head dimension 64;
- combines both norms and RoPE in one CUDA kernel.

Evidence: `sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu:359-427`.
Qwen3-MoE already uses it at
`python/sglang/srt/models/qwen3_moe.py:559-585`; LFM does not.

The existing kernel requires int32 positions while LFM positions are int64.
With one cast hoisted outside the six layers, the **net** removals are 11 decode
and 23 prefill kernels. Extending the fused kernel to accept int64 positions
would make the gross removals 12 and 24.

Traffic arithmetic, including the one hoisted positions cast:

```text
Q+K elements/layer = T*(2048+512)
decode saved = 6 * 2 * T*(2048+512)*2 B - T*(8+4) B
             = 61,428 B at T=1
prefill saved = 6 * 4 * T*(2048+512)*2 B - T*(8+4) B
              = 1,965,888,000 B at T=16,000
```

This is the best first implementation: it reuses a tested CUDA primitive and
directly removes the attention-only portion of the 13 remaining plain RMSNorms.

### 2. ShortConv gate and layout fusion

Across 18 convolution layers, the default prefill path spends:

| standalone glue | calls | time | kernel-time share |
|---|---:|---:|---:|
| `B_gate*x` | 18 | 1,642.596 us | 1.062% |
| `[T,H]→[H,T]` materialization | 18 | 3,975.724 us | 2.570% |
| `C_gate*conv_out` | 18 | 4,080.077 us | 2.637% |
| **total** | **54** | **9,698.397 us** | **6.268%** |

Decode has no layout materialization, but the two gates are still 38.880 us,
36 kernels, and 1.977% of kernel time.

Let `S = T*2048*2` bytes be one BF16 activation:

```text
decode: 18 layers * 4S = 294,912 B at T=1
prefill: 18 layers * 6S = 7,077,888,000 B at T=16,000
```

The 4S decode saving is the eliminated write/read of `Bx` and `conv_out`.
Prefill adds the eliminated write/read of `Bx_t`, for 6S.

The full form requires extending `causal_conv1d_fwd/update` to consume
`B_gate`, `x`, and `C_gate` directly and, for prefill, accept token-major input
instead of forcing `stride(-1)==1`
(`causal_conv1d.py:16-74,77-129`). This is a real CUDA-kernel change.

A staged implementation is possible:

1. fused gated transpose `(B*x).T`: removes 18 kernels and 2,359,296,000 B on
   long prefill;
2. apply `C_gate` in the convolution output store: removes another 18 kernels
   and 2,359,296,000 B;
3. make the convolution consume token-major B/x directly: removes the layout
   copy and the remaining 2,359,296,000 B.

### 3. MoE sum + next residual-add RMSNorm

Every MoE layer ends in a top-4 reduction and is immediately followed by the
next layer's `FusedAddRMSNormKernel` (the final MoE layer is followed by the
embedding norm).

```text
moe_sum(intermediate[T,4,H]) -> ffn_output[T,H]
fused_add_rmsnorm(ffn_output, residual)
```

A single kernel can reduce the four expert outputs, add the residual, write the
updated residual, and produce the normalized next-layer input.

| regime | kernels removed | current two-kernel chain | kernel-time share |
|---|---:|---:|---:|
| graph decode | 22 | 75.424 us | 3.835% |
| default prefill | 22 | 3,050.282 us | 1.971% |

Traffic saved is the reduced FFN output write followed by the norm read:

```text
22 * 2 * T*2048*2 B
= 180,224 B decode
= 2,883,584,000 B prefill
```

This needs a new top-k=4 reduction+RMSNorm Triton/CUDA kernel. The measured
chain is a ceiling; both reduction and normalization arithmetic remain.

### 4. MoE activation in the down-GEMM input prologue

The repeating MoE sequence is:

```text
up fused_moe -> act_and_mul -> down fused_moe
```

`intermediate_cache2` has shape `[T, 4, 1792]`. A down-GEMM kernel that loads
the gate/up pair and computes `SiLU(gate)*up` before the dot product removes the
standalone activation and its output buffer.

| regime | kernels removed | activation time | kernel-time share |
|---|---:|---:|---:|
| graph decode | 22 | 32.089 us | 1.632% |
| default prefill | 22 | 3,410.028 us | 2.204% |

Traffic arithmetic:

```text
22 * 2 * T*4*1792*2 B
= 630,784 B decode
= 10,092,544,000 B prefill
```

This is a large traffic target, but it modifies the Triton GEMM main loop and
is higher risk than the Q/K or reduction+norm candidates. The activation
arithmetic is not eliminated.

### 5. RoPE + KV-cache store

The existing rotary kernel accepts `FusedSetKVBufferArg`
(`sgl-kernel/python/sgl_kernel/elementwise.py:241-353`), and several SGLang
models already use `create_fused_set_kv_buffer_arg`. LFM currently emits a
separate `store_kvcache` after every RoPE.

This removes six kernels and the separate reads of K and V:

```text
6 * T*(512+512)*2 B
= 12,288 B decode
= 196,608,000 B prefill
```

The measured standalone store is only 0.335% of graph-decode kernel time and
0.064% of prefill kernel time. It is a reasonable opportunistic call-site
cleanup, not a primary project.

## Investigated and rejected/deprioritized

### Top-k + MoE alignment: apparent large decode share, almost no fusion value

`topkGatingSigmoid` plus alignment occupies 147.905 us, or 7.521% of graph
decode kernel time. This initially looks like the top target. The timeline
rejects that interpretation:

- graph-node gaps are only 0.064–0.128 us;
- top-k IDs still have to be materialized for later MoE kernels;
- the only conservative traffic saving is one alignment read:
  `22*T*4*4 B = 352 B` in decode and 5,632,000 B in prefill;
- prefill chain share is only 0.663%.

Most of the duration is real selection/padding/sort work, not launch overhead.
A different small-M MoE algorithm may help, but merely merging adjacent
kernels has no evidence-backed 7.5% saving. **Rejected as a pure fusion.**

### Hoist ShortConv metadata: cheap cleanup, not a fusion target

The eager prefill path rebuilds the same `query_start_loc` and casts the same
cache indices in all 18 conv layers. Current cost is 18 cast kernels plus 36
tiny memcopies:

- GPU activity: 52.320 us in the no-graph prefill run;
- host API time: 512.011 us;
- net hoisting removes 17 cast kernels and 34 memcopies;
- traffic saved is only 1,156 B.

In graph decode, the 18 per-layer cast kernels are already absent. This could
be a small call-site cleanup (prefill host-side ceiling about 0.3%), but it is
not an HBM/kernel-fusion priority.

### The “61 unfused RMSNorms” are not 61 actionable norms after the patch

The patched graph-decode trace contains:

- **48 `FusedAddRMSNormKernel`** calls: residual addition is already fused;
- **13 plain `RMSNormKernel`** calls:
  - 12 are Q/K norms and are addressed by candidate 1;
  - one is the first layer's operator norm.

That first norm is 1.984 us in graph decode and about 41 us in prefill:
approximately 0.10% and 0.03% of kernel time. Fusing all 49 main-path norms
into downstream consumers is theoretically a 5.53% decode / 1.96% prefill
chain, but it requires custom norm+GEMM kernels across several backends. The
MoE normalized activation also has two consumers (router and expert GEMM), so
it cannot simply be folded into one consumer. **Rejected for near-term work.**

### Sum directly in the MoE down GEMM

An aggressive alternative is to atomically accumulate the four weighted expert
outputs in the down GEMM. It would remove the same 22 sum kernels and avoid
11,534,336,000 B of prefill intermediate write/read traffic
(`22*2*T*4*2048*2`). However:

- the standalone sum is only 1.183% of graph-decode and 1.075% of prefill
  kernel time;
- cross-expert accumulation requires atomics or a different schedule;
- it risks numerical and determinism changes.

The sum+next-RMSNorm fusion is a lower-risk endpoint. The atomic GEMM variant is
deprioritized.

### Rewriting causal convolution itself

The convolution kernels are already small: 46.912 us total in graph decode
(2.386%) and 1,080.130 us in prefill (0.698%). The prefill glue around the
convolution is 6.268%, nearly nine times the convolution itself. Rewriting only
the convolution is therefore the wrong target; any kernel change should absorb
the gates/layout.

## Artifacts

- Raw reports: `*.nsys-rep`
- SQLite exports: `*.sqlite`
- Nsight summary CSVs: `*_stats_cuda_{gpu_kern,api}_sum.csv`
- Full kernel timelines: `*_timeline.csv`
- Kernel/memcpy/memset timelines: `*_activities.csv`
- Role totals: `role_summary.csv`
- Per-layer spans and gaps: `layer_summary.csv`
- Launch summary: `launch_overhead_summary.csv`
- Candidate arithmetic: `candidate_summary.csv`
- Representative layers:
  - `representative_layer3_timeline.csv`
  - `representative_attention_layer2_timeline.csv`
- Reproduction metadata: `profile_manifest.json`
- Raw-report integrity: `SHA256SUMS`
- Analysis script: `scripts/lfm_fusion/nsys_analyze.py`

No file under `/home/t-jialianggu/work/sglang/` was modified for this work.
