# Kernel-Level 优化正面攻关日志（打赢 sglang 已 tuned 的 kernel）

**日期**：2026-07-20（autopilot 通宵）
**目标**：正面尝试在 kernel level 优化性能——不是融合 sglang 未优化的小算子（那些低垂果实大多已被摘），而是**对标 sglang 已 tuned 的核心 kernel（尤其 `fused_moe`）看能否再拿到提升**。
**模型/硬件**：Qwen3-30B-A3B（E=128, top-8, N=768, hidden=2048, bf16）· H200 · triton 3.5.1 · GPU 0-3。
**方法约束**：bf16、不改分布（不量化）；baseline 必须是 **sglang 真实 GPU 代码 + 其实际加载的 tuned config**（吸取上一轮 SwiGLU 教训）。

---

## 攻关计划（按优先级）
1. **建立真实 baseline**：sglang `fused_experts` 用实际 tuned config，跨 decode/prefill batch 计时。
2. **backend 横评**：triton vs cutlass vs deep_gemm vs flashinfer MoE，同一 shape，找是否已有更快的现成 kernel（"换更好的 kernel"也算 kernel-level 赢）。
3. **NCU 归因**：fused_moe 在 decode 的时间到底花在哪（padding 浪费？纯 weight 搬运？）→ 找可攻击点。
4. **尝试特化 kernel**：针对固定 shape 硬编码常量 / 改 tiling / persistent kernel，正面对标 tuned triton。
5. 诚实记录每一步：baseline 是什么、谁测的、赢了还是输了、为什么。

## 进行中 / 结论
（下面按时间追加）

### [1] sglang fused_moe 真实 baseline + 达到带宽（v27）✅
对标 sglang 生产 triton kernel（实际加载的 tuned config），测达到的 HBM 带宽利用率：

| batch | time(µs) | 触及专家数 | 权重读(GB) | 达到带宽 | **%HBM** | 判定 |
|---|---|---|---|---|---|---|
| 1 | 31.3 | 7.8 | 0.073 | 2.35 TB/s | **49%** | 延迟受限，有空间 |
| 8 | 129 | 50 | 0.48 | 3.69 | 77% | 接近屋顶 |
| 32 | 260 | 111 | 1.05 | 4.02 | **84%** | 近最优 |
| 64-256 | 299-327 | 128 | 1.21 | 3.7-4.0 | 74-84% | 近最优 |
| 1024 | 398 | 128 | 1.21 | 3.04 | 63% | 转 compute |
| 4096 | 878 | 128 | 1.21 | 1.38 | **29%** | compute-bound |

**结论（诚实且重要）**：
- b≥32 的 decode，sglang fused_moe 已达 **74-84% HBM**（近内存屋顶）→ **无损 kernel 空间很小**（<1.3×）。
- **唯一有明显空间的是 b1（单请求 decode）：仅 49% HBM**（读 8 个专家权重 73MB 本应 15µs、实测 31µs，~2× 空间）——skinny GEMM（M=1）没吃满带宽，是真实 kernel 低效。
- prefill 大 batch 是 compute-bound，属 config-tuning 地盘（已 +50%）。
→ **kernel 攻关目标锁定 b1-b8 的 small-M decode MoE。**

### [2] b1 诊断：config 救不了 + 时间拆解（v28 + profiler）✅
- **216 个 config 全扫，b1 最好 47.8% HBM**（tuned 也是 ~49%）→ **不是 config 问题，是 M=1 的根本 kernel 限制**。b8 已 76% 近最优。
- **b1 时间拆解**（profiler，总 30.6µs）：
  - `fused_moe_kernel`（两次 GEMM）**24.3µs / 79.4%**（M=1 时 GEMM 达 63% weight-BW）
  - `moe_align_block_size` 2.4µs / 7.9%
  - `act_and_mul` 1.65µs / 5.4%
  - `count_and_sort_expert_tokens` 1.1µs / 3.7%
  - topk `copy_mul_sum` 1.1µs / 3.6%
- **攻击点**：(a) GEMM 在 M=1 只有 63% BW（skinny GEMM 并行不足）→ split-K 提并行；(b) ~6µs 开销（align/sort/act/sum）可融合消除。理论上 31→~16-18µs（~1.8×），但仅限单请求 decode。

### [3] 正面 kernel 尝试：small-M 特化 MoE（进行中）
**尝试 1：朴素 split-tile GEMV 自定义 kernel（v29）—— 输了，诚实记录**
- 写了 small-M 特化 MoE（w1+SwiGLU 融合 + w2+加权求和 融合，消除 align/sort/act/sum 开销），正确性 OK（相对误差 0.037）。
- **公平对比（两边都 cudagraph，真实 serving 用）**：sglang **31.75µs** vs 自定义 **52.68µs** → **我慢 0.6×，sglang 赢**。
- （非 graph 下自定义快 4.4×，但无关——真实 serving 用 cudagraph，会消除 sglang 的多 kernel launch 开销。）
- **为什么输**：sglang fused_moe 用 `tl.dot`（tensor core）；我的朴素版用标量 `tl.sum` 做 GEMV，compute 效率低；且 cudagraph 把 sglang 那 6µs 开销也隐藏了。
- **教训**：即使在"49% HBM"的 b1 点，sglang 的 tuned tensor-core kernel 在 cudagraph 下也很难被朴素重写打赢。
