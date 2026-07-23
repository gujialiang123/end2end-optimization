# Search space — v48 LFM2.5 serving-knob plateau study (clean, no warm start)

## Tuned knobs (4 serving-level, 192 combinations)
| knob | values |
|---|---|
| max_running_requests | 8, 16, 24, 32, 48, 64, 96, 128 |
| chunked_prefill_size | -1, 2048, 8192 |
| schedule_policy | lpm, fcfs |
| mem_fraction_static | 0.75, 0.80, 0.85, 0.90 |

Total = 8 × 3 × 2 × 4 = **192** possible configs (≥100 unique evaluations feasible).

## Fixed (NOT tuned)
- moe_runner_backend = **triton** (single fixed MoE path; no auto/flashinfer switching)
- attention_backend = **fa3**
- disable_cuda_graph = **false** (CUDA graph ALWAYS enabled; `--disable-cuda-graph` never passed)
- tensor_parallel_size = 1
- context_length = 73728, schedule_conservativeness = 1.0, max_prefill_tokens = 96000
- disable_radix_cache = false, reasoning_parser = qwen3, tool_call_parser = lfm2, trust_remote_code
- dtype/quantization/KV-cache/model/workload identical to v3

## NOT tuned (out of scope)
backend selection · CUDA graph · attention backend · TP/DP/EP/PP · dtype/quantization ·
speculative decoding · model/workload shape

## Sampler
`TPESampler(seed=20260722, n_startup_trials=20, multivariate=True)` — fresh study,
**no enqueue_trial, no warm start, no cookbook/best injection**. Trials 0–19 are
sampler-generated startup exploration; TPE takes over from trial 20.

## Objective
maximize R_concurrent_decode **request throughput** (req/s).

## Uniqueness / failures
100 **unique COMPLETE** configs required. Duplicate proposals are PRUNED and
re-sampled. Failed launches/OOM/bench failures are logged to `failures.csv`,
never assigned a fake low score, and never counted toward the 100.
