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

**尝试 2：tensor-core (tl.dot) + 跳过 align/sort + 融合 act/sum + tuned tiling（v30/v31）—— ★赢了！**
- 用 `tl.dot`（tensor core，M padding 到 16）匹配 sglang 的 GEMM 效率；**跳过 align/sort**（M 小无需分组）；**把 SwiGLU 激活融进 w1 kernel、加权求和融进 w2 kernel**（消除 act_and_mul + moe_sum 两个 kernel）；再对 tiling 做 sweep。
- **b1 结果（cudagraph 公平对比）**：sglang **32.0µs** vs 自定义 **26.1µs** → **1.23× 加速** ✅
  - 最优 config：w1(BN=64,BK=256,warps=4) + w2(BN=64,BK=128,warps=4)
- **正确性 + 精度（关键）**：多 seed 验证，自定义 vs fp32 真值相对误差 **3.7-5.2%**，而 **sglang vs fp32 是 10-14%** → **我们的 kernel 不但更快，还比 sglang 更准**（bf16 累加更干净）。
- **胜因**：省掉 align/sort/act/sum 4 个开销 kernel（~6µs）+ tuned tiling 让 GEMM 接近 sglang → 净赢 1.23×。
- **适用范围（诚实）**：当前仅 M=1（单请求 decode）；kernel 硬编码单 token。b≥8 时 sglang 已 76-84% HBM，此法（M padding 到 16）优势消失。**这是"单请求/超低延迟 decode"场景的真实 kernel-level 胜利。**

## 阶段结论（截至 v31）
1. **sglang fused_moe 整体已高度优化**：b≥32 decode 达 74-84% HBM（近内存屋顶），prefill compute-bound 处 config-tuning 已 +50%。**无损 kernel 空间总体很小。**
2. **但存在一个真实的 kernel-level 胜利点**：**M=1 单请求 decode**，sglang 只有 49% HBM + 6µs 路由/激活/求和开销。特化 kernel（tensor-core + 跳过 align/sort + 融合 act/sum + tuned tiling）拿到 **1.23× 且更准**。
3. **方法学教训贯穿**：必须对标 sglang 真实 GPU 代码 + cudagraph；朴素重写（v29 scalar）会输，tensor-core + 去开销 + 调 tiling（v31）才赢。

### [4] 胜利的适用范围：M 扫描 crossover（v32）
泛化 kernel 到多 token（pair→token 映射），扫 M：

| M | sglang(µs) | custom(µs) | 加速 | 触及专家 | 判定 |
|---|---|---|---|---|---|
| 1 | 31.7 | 26.7 | **1.19×** | 8 | ✅ 赢 |
| 2 | 49.6 | 49.4 | 1.00× | 16 | 打平 |
| 4 | 82.6 | 78.4 | 1.06× | 30 | 小赢 |
| 8 | 128 | 132 | 0.97× | 50 | ❌ 输 |
| 16 | 199 | 233 | 0.86× | 82 | ❌ 输 |

**为什么 M 大就输**：我们按 (token,expert) pair 逐个读专家权重；M 大时多个 token 复用同一专家，sglang 的**专家分组**（每个权重只读一次）更省。M=1 时 8 个 pair→8 个不同专家、无复用，我们的"省开销"净赢。
→ **胜利范围 = M≤4（单请求 / 低并发 decode）**。每 MoE 层省 ~5µs，48 层 → 单 token 省 ~240µs（端到端 TPOT 影响待集成测量）。

## 最终结论（明早看这里）
**问题：kernel level 能不能再优化 sglang 的性能？答案：能，但空间窄。**

1. **sglang 的 MoE kernel 整体已高度优化**：
   - decode b≥32：74-84% HBM（近内存屋顶），无损空间 <1.3×。
   - prefill：compute-bound，config-tuning 已 +50%。
   - 朴素重写（scalar GEMV）会输；tensor-core + 去开销 + 调 tiling 才可能赢。
2. **找到一个真实 kernel-level 胜利点**：**M=1 单请求 decode**。
   - sglang 在此只有 49% HBM + 6µs 路由/激活/求和开销。
   - 特化 kernel（tl.dot + 跳过 align/sort + 融合 act/sum + tuned tiling）：**1.19-1.23× 且比 sglang 更准**（vs fp32：我们 4-5% err，sglang 10-14%）。
   - 范围 M≤4；M≥8 sglang 专家分组反超。
3. **另一类真实空缺**（上一份报告）：shared-expert gate 在 sglang **CUDA 路径未融合**（融合版只有 CPU）→ 融合 kernel 2-3×（但算子小）。

### 后续要做的事（优先级排序）
1. **端到端集成**：把 M=1 特化 MoE kernel 挂进 sglang decode 路径，测单请求 TPOT 真实提升（预计 ~1.05-1.1× 端到端）。判断值不值得工程化。
2. **扩大胜利范围**：研究能否在 M=2-8 也赢（如：混合策略——小专家用 pair、大专家用 group；或 split-K 减少 M 大时的权重重读）。
3. **系统扫 sglang 的"CUDA 未融合"空缺**：像 shared-expert gate（CPU 有融合、CUDA 没有）这类，agent 可自动发现并补 CUDA 融合 kernel（单个收益小但可批量）。
4. **prefill kernel**：b4096 是 compute-bound，config-tuning 已 +50%；研究 kernel 层是否还有（如更好的 persistent GEMM）——但 sglang 已 tuned，预期难。
5. **量化方向**（若放开约束）：FP8/NVFP4 的 MoE kernel（那批 PR 的主战场）——但会改分布，需另测精度。

### 诚实的总体判断
- **kernel-level 无损优化 sglang 的空间总体很小**（它已优化得很好）；真实胜利集中在 **sglang 未覆盖/未融合的边角**（M=1 decode、CUDA 未融合的 gate）。
- 这**支持"autotuner/kernel agent"的定位是"自动发现并补 sglang 的覆盖空缺"**（config 未覆盖 shape、CUDA 未融合算子、小 M 特化路径），而非"重写 sglang 已 tuned 的核心 GEMM"（那个很难赢）。
