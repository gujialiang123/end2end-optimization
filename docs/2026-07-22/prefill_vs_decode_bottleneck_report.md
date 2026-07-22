# Prefill vs. Decode Bottleneck Analysis — Regimes, Method, and Results

**Model:** Qwen3-30B-A3B (bf16) · **HW:** H200 · **Engine:** sglang (Triton fused MoE)
**Author:** Jialiang · **Purpose:** answer Chendi's two questions — (1) which regimes did we test to conclude decode is a real bottleneck, and (2) reconcile with his nsys observation that in the *long-context* bench the batched prefill kernel dominates.

---

## TL;DR (direct answers)

1. **"Decode is the bottleneck" is regime-specific, not universal.** It is true for **serving-style regimes** (short input, long/streamed output, high concurrency — i.e. real agent/chat traffic), where decode is **83–99%** of end-to-end wall-clock. It is **not** true for **long-context / long-prefill regimes** (large prompt, small generation), where **prefill dominates**.

2. **Chendi's nsys observation is correct and fully consistent with our data.** In a long-context bench (large input, small output), the batched prefill GEMM *should* dominate — we measure prefill at **55–89%** of wall-clock in those regimes. There is no contradiction: our "decode-bound" statement was about serving/agent regimes with realistic in:out ratios and concurrency, **not** about long-context prefill benches.

The two claims describe **different regimes**. The whole point of the regime sweep was to show the bottleneck **moves** with the workload.

---

## 1. Regimes tested (complete list)

We use a fixed **regime suite** (`regime_scout/candidates/seed_*.yaml`) plus a real agent trace. Each is a `(input_len, output_len, concurrency)` point chosen to isolate a distinct operating point:

| Regime | input_len | output_len | concurrency | in:out | what it stresses |
|---|---:|---:|---:|---:|---|
| tiny_latency | 8 | 4 | 1 | 2:1 | pure launch/overhead floor |
| short_in_short_out | 128 | 32 | 16 | 4:1 | balanced small |
| sched_overhead_hiconc | 128 | 16 | 64 | 8:1 | scheduler tail @ high conc |
| **prefill_medium** | 4096 | 16 | 4 | 256:1 | **medium long-context prefill** |
| **prefill_long** | 16384 | 16 | 2 | 1024:1 | **extreme long-context prefill** |
| **decode_medium** | 128 | 512 | 16 | 1:4 | **decode-heavy serving** |
| **decode_heavy** | 128 | 1024 | 32 | 1:8 | **decode-heavy, high conc** |
| prefix_reuse / churn | shared-prefix | — | 16 | — | prefix-cache behavior |
| **agent_toolagent** | ~2555 (real) | ~220 (real) | 32 | ~12:1 | **real mooncake tool-agent trace** |

Plus a **bench_one_batch grid** for controlled composition/latency: `batch ∈ {1,32,64} × input ∈ {256,2048,4096} × output=32`.

---

## 2. Method

- **Wall-clock split (per regime):** real sglang server + `bench_serving`. `TTFT ≈ prefill time`, `E2E − TTFT ≈ decode time`. Also `bench_one_batch`, which reports `prefill_latency` and per-step `median_decode_latency` separately.
- **Within-step composition:** `torch` profiler traces (`--profile --profile-stage {prefill,decode}`), CUDA-graph disabled to expose real kernel time, categorized by kernel name (MoE GEMM / dense GEMM / attention / moe_align_sort / elementwise).
- All numbers are our own measurements on H200, not vendor claims.

---

## 3. Result A — Where does wall-clock time actually go? (per regime)

**Real server, `TTFT`(≈prefill) vs `E2E−TTFT`(≈decode):**

| Regime | TTFT (ms) | E2E (ms) | **prefill %** | **decode %** |
|---|---:|---:|---:|---:|
| decode_heavy (128/1024/32) | 58.5 | 8471 | **0.7%** | **99.3%** |
| decode_medium (128/512/16) | 55.6 | 3569 | 1.6% | 98.4% |
| short_in_short_out (128/32/16) | 96.6 | 576 | 16.8% | 83.2% |
| **agent_toolagent (real ~2555/~220)** | 399.6 | 2347 | **17.0%** | **83.0%** |
| sched_overhead (128/16/64) | 126.2 | 479 | 26.4% | 73.6% |
| **prefill_long (16384/16/2)** | 200.9 | 367.6 | **54.7%** | 45.3% |
| **prefill_medium (4096/16/4)** | 130.8 | 234.6 | **55.8%** | 44.2% |
| tiny_latency (8/4/1) | 48.2 | 55.6 | 86.8% | 13.2% |

**Controlled `bench_one_batch` (output=32), prefill % of total wall-clock:**

| batch | input | prefill % of wall-clock |
|---:|---:|---:|
| 1 | 256 | 20.2% |
| 1 | 4096 | 35.8% |
| 32 | 2048 | **76.7%** |
| 32 | 4096 | **85.7%** |
| 64 | 4096 | **89.4%** |

**Reading:** the bottleneck flips entirely with the in:out ratio and concurrency.
- **Decode-dominated** (83–99%): decode_heavy, decode_medium, short_in_short_out, **agent** — these are the *serving* regimes.
- **Prefill-dominated** (55–89%): prefill_long, prefill_medium, and any long-context bench with small output — **this is exactly Chendi's case**.

---

## 4. Result B — Within-step composition (why the two stages behave differently)

**DECODE step** (kernel-time breakdown):

| | MoE GEMM | dense GEMM | attention | moe_align/sort | elementwise |
|---|---:|---:|---:|---:|---:|
| b=1 short (in=16) | 33.6% | 32.4% | 21.4% | 8.6% | 3.6% |
| b=64 concurrent | 68.1% | 12.1% | 12.6% | 4.4% | 2.6% |

→ Decode is **memory-bound weight streaming** (MoE + dense = 65–80%). At b=1 each new token must re-read the full active-expert weights; this is bandwidth-bound, so faster GEMM math barely helps.

**PREFILL step** (b=1, in=10240):

| MoE GEMM | attention | dense GEMM | moe_align/sort | elementwise |
|---:|---:|---:|---:|---:|
| 41.0% | **38.8%** | 12.8% | 3.5% | 3.8% |

→ Prefill is **compute-bound**. MoE grouped-GEMM is the largest kernel, and **attention grows to 38.8%** (it is O(seq²) — at short context attention is only ~16–21%). This matches Chendi's nsys: batched prefill GEMM (MoE + attention) dominates the long-context bench.

---

## 5. Reconciliation with Chendi's nsys report

There is **no discrepancy**:

- Chendi's long-context bench = large input, small output ⇒ **prefill-dominated regime** (Result A: 55–89% prefill). His nsys correctly shows the **batched prefill kernel** on top. ✔
- Our "decode is the bottleneck" statement was scoped to **serving/agent regimes** (Result A: 83–99% decode). In those, wall-clock is dominated by thousands of memory-bound single-token decode steps. ✔

Same system, different regime. The regime sweep exists precisely to make this explicit: **there is no single global bottleneck — it moves with the workload.**

---

## 6. Consequence for optimization (what actually helped, per regime)

This regime split explains our end-to-end tuning results:

- **Prefill (compute-bound):** MoE kernel-config tuning translates to real e2e gains — **+34–43% prefill throughput / +17–25% E2E on prefill-heavy & agent regimes** (vs. the no-config default heuristic). This is the compute-bound stage where better GEMM tiling matters.
- **Decode (memory-bound):** config/kernel changes give **≈0** e2e — MoE is only ~41% of the step and the step is bandwidth-bound. The real decode-side levers are **speculative decoding** (+23–30% observed) and **serving concurrency**, not kernel math.

> Caveat on the +34–43% number: its baseline is the **default heuristic (config disabled)**, which illustrates the *value of having a tuned config at all*. Against **sglang's current shipped fallback config**, re-tuning adds only ~0% overall (one exception: b=1 long prefill, +17% prefill / +4.6% E2E), because the shipped fallback is already close to optimal. Both baselines are stated explicitly to avoid confusion.

---

## 7. Bottom line

1. **Decode is a real bottleneck — in serving/agent/decode-heavy regimes** (83–99% of wall-clock). We concluded this from decode_heavy, decode_medium, short_in_short_out, sched_overhead, and the real agent_toolagent trace.
2. **Prefill is the bottleneck in long-context regimes** (55–89% of wall-clock), which is **exactly what Chendi's nsys shows** — his observation is correct and consistent.
3. The optimization implication follows the same split: **tune the prefill GEMM for long-context/compute-bound work; use spec-decoding/serving for the memory-bound decode-heavy work.**

### Data / repro
- Server regime split: `results/2026-07-20_v43_server_e2e/`
- bench_one_batch grid + wall-clock: `results/2026-07-20_v42_kernel_e2e/`, `results/2026-07-21_v46_retune/ab/`
- Composition traces: `results/2026-07-20_v33_decode_audit/` (decode short), `v45_decode_audit/` (decode b=64), `v44_longprefill/` (prefill in=10240)
