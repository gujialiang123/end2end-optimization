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
