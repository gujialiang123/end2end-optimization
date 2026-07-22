# Prefill vs. Decode Bottleneck Analysis — Regimes, Method, and Results
# Prefill 与 Decode 瓶颈分析 — Regime、方法与结果（中英双语）

**Model / 模型:** Qwen3-30B-A3B (bf16) · **HW / 硬件:** H200 · **Engine / 引擎:** sglang (Triton fused MoE)
**Author / 作者:** Jialiang
**Purpose / 目的:** answer Chendi's two questions — (1) which regimes did we test to conclude decode is a real bottleneck, and (2) reconcile with his nsys observation that in the *long-context* bench the batched prefill kernel dominates.
回答 Chendi 的两个问题 —（1）我们基于哪些 regime 得出"decode 是真正瓶颈"的结论；（2）与他 nsys 观察到的"长 context bench 里 batched prefill kernel 主导"如何对账。

---

## TL;DR (direct answers) / 直接回答

1. **"Decode is the bottleneck" is regime-specific, not universal.** It is true for **serving-style regimes** (short input, long/streamed output, high concurrency — real agent/chat traffic), where decode is **83–99%** of end-to-end wall-clock. It is **not** true for **long-context / long-prefill regimes** (large prompt, small generation), where **prefill dominates**.
   **"Decode 是瓶颈"是分 regime 的，不是普适的。** 它在 **serving 类 regime**（短输入、长/流式输出、高并发 —— 真实 agent/chat 流量）成立，decode 占端到端墙钟 **83–99%**；但在 **长 context / 长 prefill regime**（大 prompt、小生成）**不成立**，那里是 **prefill 主导**。

2. **Chendi's nsys observation is correct and fully consistent with our data.** In a long-context bench (large input, small output), the batched prefill GEMM *should* dominate — we measure prefill at **55–89%** of wall-clock in those regimes. No contradiction: our "decode-bound" statement was scoped to serving/agent regimes with realistic in:out ratios and concurrency, **not** long-context prefill benches.
   **Chendi 的 nsys 观察是正确的，与我们数据完全一致。** 长 context bench（大输入、小输出）里 batched prefill GEMM 本就应该主导 —— 我们实测这些 regime 的 prefill 占墙钟 **55–89%**。没有矛盾：我们说的"decode-bound"只针对真实 in:out 比和并发的 serving/agent regime，**不是**长 context prefill bench。

The two claims describe **different regimes**. The regime sweep exists to show the bottleneck **moves** with the workload.
这两个论断描述的是**不同 regime**。做 regime sweep 的意义正是展示瓶颈会**随 workload 移动**。

---

## 1. Regimes tested (complete list) / 测试的完整 regime 列表

We use a fixed **regime suite** (`regime_scout/candidates/seed_*.yaml`) plus a real agent trace. Each is an `(input_len, output_len, concurrency)` point isolating a distinct operating point.
我们使用固定的 **regime 套件**（`regime_scout/candidates/seed_*.yaml`）加一个真实 agent trace。每个是一个 `(输入长度, 输出长度, 并发)` 点，隔离一个不同的工况。

| Regime | input_len | output_len | concurrency | in:out | what it stresses / 压什么 |
|---|---:|---:|---:|---:|---|
| tiny_latency | 8 | 4 | 1 | 2:1 | pure launch/overhead floor / 纯启动开销下限 |
| short_in_short_out | 128 | 32 | 16 | 4:1 | balanced small / 均衡小负载 |
| sched_overhead_hiconc | 128 | 16 | 64 | 8:1 | scheduler tail @ high conc / 高并发调度尾延迟 |
| **prefill_medium** | 4096 | 16 | 4 | 256:1 | **medium long-context prefill / 中等长上下文 prefill** |
| **prefill_long** | 16384 | 16 | 2 | 1024:1 | **extreme long-context prefill / 极长上下文 prefill** |
| **decode_medium** | 128 | 512 | 16 | 1:4 | **decode-heavy serving / decode 重的服务** |
| **decode_heavy** | 128 | 1024 | 32 | 1:8 | **decode-heavy, high conc / decode 重 + 高并发** |
| prefix_reuse / churn | shared-prefix | — | 16 | — | prefix-cache behavior / 前缀缓存行为 |
| **agent_toolagent** | ~2555 (real) | ~220 (real) | 32 | ~12:1 | **real mooncake tool-agent trace / 真实 mooncake 工具 agent 轨迹** |

Plus a **bench_one_batch grid** for controlled composition/latency: `batch ∈ {1,32,64} × input ∈ {256,2048,4096} × output=32`.
另外用一个 **bench_one_batch 网格**做受控的组成/延迟测量：`batch ∈ {1,32,64} × 输入 ∈ {256,2048,4096} × 输出=32`。

---

## 2. Method / 方法

- **Wall-clock split (per regime):** real sglang server + `bench_serving`. `TTFT ≈ prefill time`, `E2E − TTFT ≈ decode time`. Also `bench_one_batch`, which reports `prefill_latency` and per-step `median_decode_latency` separately.
  **墙钟拆分（按 regime）：** 真实 sglang server + `bench_serving`。`TTFT ≈ prefill 时间`，`E2E − TTFT ≈ decode 时间`。也用 `bench_one_batch`，它分别报告 `prefill_latency` 和每步 `median_decode_latency`。
- **Within-step composition:** `torch` profiler traces (`--profile --profile-stage {prefill,decode}`), CUDA-graph disabled to expose real kernel time, categorized by kernel name (MoE GEMM / dense GEMM / attention / moe_align_sort / elementwise).
  **step 内组成：** torch profiler trace（`--profile --profile-stage {prefill,decode}`），关闭 CUDA-graph 以暴露真实 kernel 时间，按 kernel 名归类（MoE GEMM / dense GEMM / attention / moe_align_sort / elementwise）。
- All numbers are our own measurements on H200, not vendor claims.
  所有数字均为我们在 H200 上的实测，非厂商宣称。

---

## 3. Result A — Where does wall-clock time actually go? (per regime) / 墙钟时间到底花在哪（按 regime）

**Real server, `TTFT`(≈prefill) vs `E2E−TTFT`(≈decode) / 真实 server：**

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

**Controlled `bench_one_batch` (output=32), prefill % of total wall-clock / 受控网格，prefill 占总墙钟：**

| batch | input | prefill % of wall-clock / prefill 占墙钟 |
|---:|---:|---:|
| 1 | 256 | 20.2% |
| 1 | 4096 | 35.8% |
| 32 | 2048 | **76.7%** |
| 32 | 4096 | **85.7%** |
| 64 | 4096 | **89.4%** |

**Reading / 解读:** the bottleneck flips entirely with the in:out ratio and concurrency. 瓶颈随 in:out 比和并发彻底翻转。
- **Decode-dominated** (83–99%): decode_heavy, decode_medium, short_in_short_out, **agent** — the *serving* regimes. / **decode 主导**（83–99%）：这些是*服务*类 regime。
- **Prefill-dominated** (55–89%): prefill_long, prefill_medium, and any long-context bench with small output — **exactly Chendi's case**. / **prefill 主导**（55–89%）：长上下文 + 小输出的 bench —— **正是 Chendi 的场景**。

---

## 4. Result B — Within-step composition / step 内组成（为何两阶段行为不同）

**DECODE step** (kernel-time breakdown) / **DECODE step**（kernel 时间拆分）:

| | MoE GEMM | dense GEMM | attention | moe_align/sort | elementwise |
|---|---:|---:|---:|---:|---:|
| b=1 short (in=16) | 33.6% | 32.4% | 21.4% | 8.6% | 3.6% |
| b=64 concurrent | 68.1% | 12.1% | 12.6% | 4.4% | 2.6% |

→ Decode is **memory-bound weight streaming** (MoE + dense = 65–80%). At b=1 each new token re-reads the full active-expert weights; bandwidth-bound, so faster GEMM math barely helps.
→ Decode 是**内存带宽受限的权重流式读取**（MoE + dense = 65–80%）。b=1 时每个新 token 都要重新读一遍激活专家的全部权重；受带宽限制，所以把 GEMM 算得更快几乎没用。

**PREFILL step** (b=1, in=10240) / **PREFILL step**（b=1，输入=10240）:

| MoE GEMM | attention | dense GEMM | moe_align/sort | elementwise |
|---:|---:|---:|---:|---:|
| 41.0% | **38.8%** | 12.8% | 3.5% | 3.8% |

→ Prefill is **compute-bound**. MoE grouped-GEMM is the largest kernel, and **attention grows to 38.8%** (O(seq²) — at short context attention is only ~16–21%). This matches Chendi's nsys: the batched prefill GEMM (MoE + attention) dominates the long-context bench.
→ Prefill 是**计算受限**。MoE grouped-GEMM 是最大 kernel，且 **attention 涨到 38.8%**（O(seq²) —— 短上下文时 attention 只有约 16–21%）。这与 Chendi 的 nsys 吻合：batched prefill GEMM（MoE + attention）在长上下文 bench 里主导。

---

## 5. Reconciliation with Chendi's nsys report / 与 Chendi nsys 报告的对账

There is **no discrepancy** / **没有矛盾**:

- Chendi's long-context bench = large input, small output ⇒ **prefill-dominated regime** (Result A: 55–89% prefill). His nsys correctly shows the **batched prefill kernel** on top. ✔
  Chendi 的长上下文 bench = 大输入、小输出 ⇒ **prefill 主导 regime**（Result A：55–89% prefill）。他的 nsys 正确地显示 **batched prefill kernel** 在顶部。✔
- Our "decode is the bottleneck" statement was scoped to **serving/agent regimes** (Result A: 83–99% decode). There, wall-clock is dominated by thousands of memory-bound single-token decode steps. ✔
  我们"decode 是瓶颈"的说法只针对 **serving/agent regime**（Result A：83–99% decode）。那里墙钟被成千上万个内存受限的单 token decode step 主导。✔

Same system, different regime. The regime sweep exists precisely to make this explicit: **there is no single global bottleneck — it moves with the workload.**
同一系统，不同 regime。做 regime sweep 正是为了说清这点：**不存在唯一的全局瓶颈 —— 它随 workload 移动。**

---

## 6. Consequence for optimization / 对优化的含义（每个 regime 什么有效）

This regime split explains our end-to-end tuning results. 这个 regime 拆分解释了我们的端到端 tuning 结果。

- **Prefill (compute-bound):** MoE kernel-config tuning translates to real e2e gains — **+34–43% prefill throughput / +17–25% E2E on prefill-heavy & agent regimes** (vs. the no-config default heuristic). This is the compute-bound stage where better GEMM tiling matters.
  **Prefill（计算受限）：** MoE kernel-config tuning 转化为真实端到端收益 —— 在 prefill 重和 agent regime 上 **prefill 吞吐 +34–43% / E2E +17–25%**（对比无 config 的 default 启发式）。这是"更好的 GEMM 分块有意义"的计算受限阶段。
- **Decode (memory-bound):** config/kernel changes give **≈0** e2e — MoE is only ~41% of the step and the step is bandwidth-bound. The real decode-side levers are **speculative decoding** (+23–30% observed) and **serving concurrency**, not kernel math.
  **Decode（内存受限）：** config/kernel 改动端到端 **≈0** —— MoE 只占该 step 约 41%，且整个 step 受带宽限制。decode 侧真正的杠杆是**投机解码**（实测 +23–30%）和**服务并发**，不是 kernel 算力。

> **Caveat on the +34–43% number / 关于 +34–43% 的重要说明:** its baseline is the **default heuristic (config disabled)**, illustrating the *value of having a tuned config at all*. Against **sglang's current shipped fallback config**, re-tuning adds only ~0% overall (one exception: b=1 long prefill, +17% prefill / +4.6% E2E), because the shipped fallback is already near-optimal. Both baselines are stated explicitly to avoid confusion.
> 它的 baseline 是 **default 启发式（关闭 config）**，说明的是"有没有 tuned config"的价值。相对 **sglang 当前实际加载的 fallback config**，重新 tune 整体只 ~0%（唯一例外：b=1 长 prefill，prefill +17% / E2E +4.6%），因为出厂 fallback 已近最优。两个 baseline 都明确标出，避免混淆。

---

## 7. Bottom line / 结论

1. **Decode is a real bottleneck — in serving/agent/decode-heavy regimes** (83–99% of wall-clock). Concluded from decode_heavy, decode_medium, short_in_short_out, sched_overhead, and the real agent_toolagent trace.
   **Decode 在 serving/agent/decode 重的 regime 里是真正瓶颈**（占墙钟 83–99%）。依据：decode_heavy、decode_medium、short_in_short_out、sched_overhead 和真实 agent_toolagent trace。
2. **Prefill is the bottleneck in long-context regimes** (55–89% of wall-clock), which is **exactly what Chendi's nsys shows** — his observation is correct and consistent.
   **Prefill 在长上下文 regime 里是瓶颈**（占墙钟 55–89%），这**正是 Chendi 的 nsys 所示** —— 他的观察正确且一致。
3. The optimization implication follows the same split: **tune the prefill GEMM for long-context/compute-bound work; use spec-decoding/serving for the memory-bound decode-heavy work.**
   优化含义遵循同一拆分：**长上下文/计算受限的活 → 调 prefill GEMM；内存受限的 decode 重的活 → 用投机解码/服务层。**

### Data / repro / 数据与复现
- Server regime split / 服务端 regime 拆分: `results/2026-07-20_v43_server_e2e/`
- bench_one_batch grid + wall-clock / 网格与墙钟: `results/2026-07-20_v42_kernel_e2e/`, `results/2026-07-21_v46_retune/ab/`
- Composition traces / 组成 trace: `results/2026-07-20_v33_decode_audit/` (decode short), `v45_decode_audit/` (decode b=64), `v44_longprefill/` (prefill in=10240)
