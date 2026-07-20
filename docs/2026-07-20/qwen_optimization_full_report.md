# Qwen3-30B-A3B 优化全纪录：做了什么 · 结果 · 有无提升 · 分析

**模型**：Qwen3-30B-A3B-Instruct-2507（MoE，E=128，top-8，hidden=2048，moe_intermediate=768，无 shared expert）
**硬件**：H200（GPU0/1），bf16，sglang（editable @17f7a1da1，triton 3.5.1）
**日期**：2026-07-19 ~ 2026-07-20 · **所有数字均为本项目实测**（非 PR 自称），对标 sglang 真实 GPU 代码 + cudagraph
**报告目的**：给团队/Dey/Ofer 一个关于 Qwen 这条线的完整、诚实口径。

---

## 0. 一句话总结

在**成熟 bf16/H200 上的 Qwen3-30B**，我们验证的**可复现同模型端到端提升 = config 自动调优（prefill +35~54%）+ 开启投机解码（decode c32 +30.6%）**；**自己重写/融合 kernel 在全 regime 端到端 ≈0**（隔离层 1.23× 不迁移）。真正"tuning 之外"的空间在**算法层（spec decoding）**和**架构层（线性注意力，需换模型）**，不在重写 bf16 MoE kernel。

---

## 1. 我们在 Qwen 上做过的所有事情（含结果与判定）

| # | 做了什么 | 层次 | 隔离/micro | **端到端实测** | 有无提升 | 判定 |
|---|---|---|---|---|---|---|
| 1 | **config 自动调优**（重 tune MoE triton config vs 默认启发式） | 配置 | — | decode +13%、**prefill +35~54%** | ✅ 有 | **主力杠杆**（autotuning） |
| 2 | **自写 small-M(decode) MoE kernel**（跳过 align/sort、融合 act/sum、tensor-core dot） | kernel | b1 **1.23×** 且更准 | b1 +1.4% / b2 −2% / b4 −11%；agent c1 −0.7% / c32 −7% | ❌ ≈0，高并发负 | 通用改动不成立 |
| 3 | **shared-expert gate 融合**（linear+sigmoid+mul 三算子）| kernel | 隔离 2-3× | Qwen3 无 shared expert，**不适用**；换 Qwen1.5-MoE 测得全 batch ~1.0× | ❌ 0 | 对 Qwen3 不适用 |
| 4 | **投机解码（spec decoding）** | 算法 | — | decode **c1 +6.6% / c32 +30.6%**（exact，不改分布） | ✅ 有 | **最大可实现杠杆** |
| 5 | **decode step 组成审计**（哪块占时间） | 诊断 | — | MoE 41% + dense 32% + attn 16% = 89% memory-bound | 诊断 | 解释了为何 kernel 杠杆小 |
| 6 | **MoE HBM 带宽 vs batch** | 诊断 | b≥32 达 74–84% HBM | — | 诊断 | decode 已近内存屋顶 |
| 7 | **roofline 天花板** | 诊断 | decode 理论上界 ~1.85× | — | 诊断 | config 够不到的 memory 侧空间 |
| 8 | **线性注意力架构对比**（Qwen3 vs LFM2.5，长上下文 scaling） | 架构 | — | Qwen decode scaling +57% vs LFM +24%（bs=32, 512→8192）；Qwen bs=32×16k **OOM** | ✅（架构级，非同模型） | 选型洞察 |

---

## 2. 三张核心图（Dey 要的"tuning 以外还有多少空间"）

> 全部为 Qwen3-30B-A3B / decode / H200 / bf16 实测。文件在 `results/2026-07-20_v34_figures/`。

### 图1 — decode step 组成（`fig1_decode_composition.png`）
**这张图是什么**：把一步 decode 的 GPU kernel 时间按算子类型拆开的饼图/柱图。
**数据**：**MoE 41% + dense_gemm(qkv/o/lm_head) 32% + attention 16% = 89%**，其余 norm/act/sample/misc ≈ 11%。
**解析**：decode 前三大块**全是 memory-bound 的权重/KV 流式读取**（b1 下光 lm_head 单 token 就要读 vocab×hidden≈600MB 权重）。这是"为什么抠单个 kernel 的算力，端到端杠杆很小"的根因——整步本质是在**读权重**，不是在算。任何只提升算力/省 launch 的 kernel 改动，最多动 89% 里很小一角。

### 图2 — MoE 达到的 HBM 带宽 vs batch（`fig2_moe_bandwidth_vs_batch.png`）
**这张图是什么**：sglang fused_moe kernel 在不同 batch 下实际打满的 HBM 带宽百分比曲线。
**数据**：b≥32 达 **74–84% HBM**（近内存屋顶，无损 kernel 空间 <1.3×）；b=4096 掉到 **29%**（此时转 compute-bound，即 prefill 区）。
**解析**：**decode = memory-bound**（kernel 已近内存屋顶，config 和 kernel 都难再压）；**prefill = compute-bound**（另一套故事，config-tuning 已在这里拿到 +50%）。所以"还有多少空间"这个问题**必须按 regime 分开答**——decode 和 prefill 的瓶颈根本不同。

### 图3 — ★headroom BEYOND tuning（`fig3_headroom_beyond_tuning.png`，核心图）
**这张图是什么**：以 **best-tuned config 为 baseline（=1.0×）**，展示"在把 config 调到最优之后，别的手段还能再拿多少"的分组柱状图（decode，exact 方法）。
**数据**：

| 手段 | 单请求 c=1 | 并发 c=32 |
|---|---|---|
| best-tuned config（baseline） | 1.00× | 1.00× |
| + kernel 重写（实测 e2e） | **+1.5%** | —（未测/预期更低）|
| + spec decoding（实测 e2e，exact） | **+6.6%** | **+30.6%** |
| roofline 天花板（理论上界，exact） | 1.85× | 1.85× |

**解析（一句话）**：config 调到平台期后，decode 理论上还有 ~**1.85×** 空间（全在 memory 侧，config 够不到）；**spec decoding 已实测兑现 +6.6%(c1)/+30.6%(c32)，是目前最大可实现杠杆**（它一次验证多 token，同时摊薄 MoE+dense+lm_head+attn 全部 89% 的 memory 读取）；而**纯 kernel 重写只兑现 +1.5%**（因为 MoE 仅占 41%，且 sglang kernel 已近内存屋顶）。

---

## 3. 补充图 — 线性注意力如何"扩大" tuning 以外的空间（`results/2026-07-20_v39_ctxscan/ctx_scaling.png`）

**这张图是什么**：两张并排子图。左：Qwen3-30B（全注意力）vs LFM2.5-8B（混合线性注意力）**decode 每步延迟 随上下文长度**曲线（batch=32）；右：同数据**归一化到 ctx=512** 后的 scaling 因子。
**数据**：

| context | Qwen3(ms) | LFM(ms) | Qwen 归一 | LFM 归一 |
|---:|---:|---:|---:|---:|
| 512 | 8.42 | 5.44 | 1.00× | 1.00× |
| 2048 | 8.68 | 5.83 | 1.03× | 1.07× |
| 8192 | 13.25 | 6.74 | **1.57×** | **1.24×** |
| 16384 | **OOM** | (单发 prefill 亦 OOM) | — | — |

**解析**：全注意力每生成一个 token 要**读全部历史 KV cache**，上下文越长 decode 越慢、显存越涨 → Qwen 延迟 +57%、且 bs=32×16k **直接 OOM**。线性注意力把历史压进**固定大小 O(1) 状态**，decode 几乎不随上下文涨 → LFM 只 +24%、显存足迹小仍能跑。**这是 tuning 和 kernel 重写都够不到的架构级杠杆**——但注意它是"**换一类模型**"，不是把 Qwen 本身变快（Qwen3 与 LFM 是不同模型、质量不同）。

---

## 4. 关键分析与教训

### 4.1 为什么 kernel 重写在 Qwen 上端到端 ≈0
1. **decode 是带宽墙**：89% 时间在流式读权重/KV（图1），kernel 省的是算力/launch，不是读带宽。
2. **sglang kernel 已近内存屋顶**：b≥32 打满 74–84% HBM（图2），无损空间 <1.3×。
3. **cudagraph 已吃掉 launch 开销**：融合省的那点 launch，cudagraph 早已隐藏 → 端到端无感。
4. **MoE 只占 41%**：即便 MoE kernel 拿到隔离 1.23×，摊到整步也只剩个位数,且我的按-pair 处理在 b≥2 被 sglang 专家分组反超。

### 4.2 方法学教训（已固化）
- **单点端到端会误导**：自写 MoE kernel 隔离 1.23×、"M≤4 都赢"，一扫 regime 就被证伪（b4 −11%）。**必须扫 batch/context/并发 + 真实 server 负载**。
- **必须对标 sglang 真实 GPU 代码 + cudagraph**：早期用朴素 PyTorch baseline 得出过误导性"SwiGLU 加速"，已撤回。
- **所有数字自测**，不采信 PR 自称。

### 4.3 提升来源总分类（回答"是不是全靠 autotuning"）
| 来源 | 端到端 | 是 autotuning? | 同模型变快? |
|---|---|---|---|
| config 调优 | prefill +35~54% | ✅ | ✅ |
| spec decoding | decode c32 +30.6% | ❌（算法层） | ✅（开特性） |
| 线性注意力架构 | scaling +24 vs +57% | ❌ | ❌（换模型） |
| 重写 bf16 MoE kernel | ≈0 | — | ✅ 但无效 |

**结论**：同模型上可复现的端到端提升 = **config 调优 + 开启 spec decoding**；kernel 层证伪 ≈0；架构层是选型洞察。若把"开特性"也算进广义配置，则**目前所有同模型端到端提升都来自"配置/特性开关层"，没有一个来自我们自写 kernel**。

---

## 5. 对项目定位的启示
- 若 agent 目标 = **"自动把某模型在某机器上调到最优"** → 价值在**穷举 config + 特性开关空间**（config tuning + spec + chunk/并发），这本身是有价值的产品，且我们已有正面证据。
- 若 agent 目标 = **"发现 kernel 级新提升"** → 在成熟 bf16 模型上 payoff 很低（本报告证据）；kernel 空间集中在**新架构 / AMD / 量化 / sglang 未覆盖的边角**。

## 6. 相关文档与产物
- 本报告：`docs/2026-07-20/qwen_optimization_full_report.md`
- 图：`results/2026-07-20_v34_figures/fig1_3*.png`、`results/2026-07-20_v39_ctxscan/ctx_scaling.png`
- 全 regime 扫描 + 最终矩阵：`docs/2026-07-20/regime_sweep_kernel_changes.md`
- 新架构线性注意力：`docs/2026-07-20/new_architecture_linear_attention_e2e.md`
- 图说明（Dey）：`docs/2026-07-20/headroom_beyond_tuning_figures.md`
- kernel 攻坚全过程：`docs/2026-07-20/kernel_optimization_attempt_log.md`
- config-tuning 验证：`docs/2026-07-19/pr_validation_report.md`
- 自写 kernel / patch：`scripts/custom_moe_patch.py`、`scripts/serve_with_patch.py`
