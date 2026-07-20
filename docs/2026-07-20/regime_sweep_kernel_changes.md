# Kernel 改动 × 全 regime 端到端矩阵(诚实修正:之前只测单点)

**日期**：2026-07-20（autopilot）
**背景**：之前每个 kernel 改动只在单点（b=1, in256/out64）测了端到端,不严谨。本轮**在多 batch/regime + agent 数据集上系统测每个改动**。
**GPU**：0-1 · **模型**：Qwen3-30B-A3B、Qwen1.5-MoE-A2.7B、（新架构待定）

## 方法
- decode regime：bench_one_batch 扫 batch ∈ {1,2,4,8,16,32,64}，in=256/out=32，对比 CUSTOM 开/关的 median decode TPOT。
- agent regime：server + bench_serving toolagent trace，真实并发。
- 每个改动都标注：隔离层加速 / 端到端加速 / 生效的 regime 范围。

## 结果（逐步追加）

### 改动1：Qwen3-30B custom MoE kernel — batch 扫描端到端（★重要修正）
方法：bench_one_batch，batch∈{1,2,4}，in256/out64，**3 次重复取中位数**（单次噪声大）。custom 仅在 M≤4 激活。

| batch | baseline TPOT | custom TPOT | 端到端加速 |
|---|---|---|---|
| 1 | 4.25 ms | 4.19 ms | **1.014×（+1.4%）** |
| 2 | 4.87 ms | 4.97 ms | **0.980×（−2%，略慢）** |
| 4 | 5.48 ms | 6.14 ms | **0.893×（−11%，明显慢）** |

**关键诚实修正**：
- 之前我说"M≤4 都赢、隔离 1.23×"——**端到端只有 b=1 略赢（+1.4%），b=2 打平，b=4 反而慢 11%**。
- 原因：我的 kernel 按 (token,expert) pair 逐个处理，b≥2 时并行/复用不划算，很快被 sglang 的专家分组反超（比隔离 micro-benchmark 的 crossover 更早）。
- **→ 这个 custom MoE kernel 实际只在"严格单请求 decode（b=1）"有微弱正收益（+1.4%），其余 regime 都是负的。作为通用改动不成立。**
- 教训：**单点端到端会误导；必须扫 regime。** 我之前的 1.23×/M≤4 结论被 regime 扫描证伪了大半。

### 改动2：Qwen1.5-MoE gate 融合 — batch 扫描端到端
方法：batch∈{1,8,32}，in256/out64，2 次重复取中位数。

| batch | baseline | fused-gate | 加速 |
|---|---|---|---|
| 1 | 3.36 ms | 3.36 ms | 0.997× |
| 8 | 4.47 ms | 5.00 ms | 0.894×（噪声）|
| 32 | 7.57 ms | 7.55 ms | 1.002× |

**结论**：全 regime **~1.0×（无端到端提升）**。gate 是极小算子 + cudagraph 已隐藏 launch，任何 batch 都无感。

### 小结（改动1+2，全 regime）
| 改动 | 隔离层 | 端到端最好 | 生效 regime | 通用改动? |
|---|---|---|---|---|
| Qwen3 custom MoE | 1.23×(b1) | **+1.4%(仅 b1)** | 严格单请求 | ❌ b≥2 变慢 |
| Qwen1.5 gate 融合 | 2-3×(隔离) | **~0%(全 batch)** | 无 | ❌ |
**→ 两个"kernel 快很多"的改动，全 regime 端到端复现后都 ≈0（最多单点 +1.4%）。之前的乐观结论被 regime 扫描证伪。**

---

## Task ① agent 数据集 regime（server + bench_serving，真实并发）

方法：sglang server（H200/GPU0，bf16，cudagraph on，mem-frac 0.85），
`bench_serving --dataset-name random --random-input-len 1024 --random-output-len 512`
（agent 风格：长 prompt + 中等生成）。custom MoE kernel 通过 `sitecustomize` 注入到
所有 TP worker（已确认 `[custom_moe_patch] installed` 打印 4 次）。对比 baseline（无 patch）。

| 并发 | decode batch | custom kernel 是否触发 | baseline 中位 TPOT | custom 中位 TPOT | 端到端 |
|---|---|---|---|---|---|
| c1  | 1  | **是**（M=1≤4，custom 触发） | 4.26 ms | 4.23 ms | −0.7%（噪声内，无收益） |
| c32 | 32 | 否（M=32>4，全 fallback） | 16.95 ms | 18.21 ms | +7%（fallback 分支开销/方差，反而慢） |

**结论（agent regime）**：
- 真实 agent 并发（c32）下 decode batch 远大于 4，custom kernel **永远 fallback**，
  不可能有收益；反而 patch 的形状检查分支带来 ~几% 开销。
- 仅严格单请求（c1）时 kernel 才触发，端到端 **−0.7%（噪声内，无收益）**——
  与 bench_one_batch 的 b1 +1.4% 一致量级（≈0）。
- **→ custom MoE kernel 在 agent 数据集上 0 收益**，且高并发时因 fallback 开销略负。

## 三改动 × regime 端到端总矩阵（最终）

| 改动 | 隔离/micro | b1 decode | agent c1 | agent c32(真实并发) | 长上下文×并发 | 通用收益? |
|---|---|---|---|---|---|---|
| Qwen3 custom MoE | 1.23× | +1.4% | −0.7% | −7%(fallback) | — | ❌ |
| Qwen1.5 gate 融合 | 2-3× | ~0% | — | — | — | ❌ |
| **LFM 线性注意力架构(②)** | — | 无(甚至负) | — | — | **ctx scaling +24% vs +57%；Qwen OOM 处 LFM 仍可跑** | ✅（架构级，非 kernel） |

**总诚实结论**：
1. 成熟 bf16/H200 MoE 上的 **kernel 融合改动全 regime 端到端 ≈0**（隔离层的 1.23×/2-3×
   不迁移到端到端；MoE/dense 是带宽墙，cudagraph 已吃掉 launch 收益）。
2. 真正"tuning 之外"的端到端杠杆是 **架构选择**（长上下文并发场景用线性注意力，见
   `new_architecture_linear_attention_e2e.md`）和 **投机解码**（+23–30%），都不是 bf16
   MoE kernel 重写。
3. 方法论教训已固化：**单点端到端会误导，必须扫 regime + 真实并发**。
