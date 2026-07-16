# Mason 路线图落地矩阵：Qwen3-30B-A3B（MoE）逼近 roofline 的分层进度表

**目的**：按 Mason 的建议，把"从 config autotuning 榨干一切 → kernel autotuning 补 shape → 选定 kernel 手改/融合"这条递进路线，针对**唯一目标模型 Qwen3-30B-A3B-Instruct-2507（bf16, H200）**列成一张可执行矩阵。**已有的填数据、缺的留空**，一眼看清还差什么、下一步做什么。
**日期**：2026-07-16
**模型**：Qwen3-30B-A3B-Instruct-2507（128 experts，top-8，`moe_intermediate=768`，hidden=2048，48 层，32 Q-heads / 4 KV-heads，head_dim=128）
**硬件**：NVIDIA H200（132 SM，HBM3e 4.8 TB/s，bf16 TC 峰值 989.5 TFLOP/s）
**软件**：sglang（sglang-dev env）、triton **3.5.1**

> 图例：✅ 已有数据　⚠️ 部分/需补　⬜ 完全缺失（待做）

---

## Part 0：Mason 建议的五步框架（对照本项目）

| # | Mason 的要求 | 本项目对应 | 状态 |
|---|---|---|---|
| 1 | 把不同 regime 分类 | §1 regime 目录 | ⚠️ 有旧聚类，缺当前模型的统一清单 |
| 2 | 每个 regime 展示 config autotuning 的天花板 + 平台期曲线（几点即可看出 plateau），并指出 config 价值因 regime 而异 | §2 config 曲线族 | ⚠️ 只有 2 个 regime（toolagent/shared_prefix） |
| 3 | 重量级 kernel（MoE/attention）检查 kernel autotuning：Qwen 是否有 shape 未被覆盖、是否需要像链接的 PR 那样新 autotune | §3 shape 覆盖对照 | ⚠️ 已发现 H200 config 缺口，缺系统对照 |
| 4 | 用尽以上后，选 NCU 显示有明显 headroom 的**单个 kernel**（在某 regime），做 agent 辅助 kernel 重写：(a) 硬编码常量的底层 CUDA 重写；(b) 找周边低垂融合候选做 fused kernel（经典解 memory-bound，可能需 torch.compile） | §4 kernel 选择 + §5 重写 | ⬜ 未开始（NCU 选 kernel 的证据已齐） |
| 5 | 产出"分层逼近 roofline"总图：config autotune → kernel constexpr autotune → kernel 手改，各自是可测量的提升。量出手改相对 autotune 的额外提升 **X** | §6 总图 + X | ⬜ 未开始 |

**最终叙事**（Mason 原话）：先"here's how we get everything we can out of autotuning"，再"on a selected kernel in a selected regime, re-authoring further improved it by **X%**"。
- **若 X 很小** → autotuning 故事：重点转向发现 autotuning 覆盖空白（补 config，像链接的 PR）。
- **若 X 很大** → kernel-agent 故事叠加在 autotuning 之上，且这就是证明其价值的实证。

---

## Part 1：Regime 目录（Qwen3-30B-A3B，人造 + sglang dataset 全列）

### 1A. 人造合成 regime（4-regime，固定 batch/seqlen，用于 kernel profiling）
来源：`docs/2026-06-09/sglang_triton_4regime_profiling.md`

| Regime | batch | in_len | out_len | conc | 特征 | 主导 kernel(NCU) | gpu_util |
|---|---|---|---|---|---|---|---|
| `R_short_decode` | 8 | 100 | 256 | 1 | 极低专家利用（B1→8/128 专家各 1 token） | `fused_moe_kernel` | 8.5% |
| `R_medium_balanced` | 16 | 800 | 256 | 8 | 典型 B8，多数专家激活 | `fused_moe_kernel` | 11.6% |
| `R_long_prefill` | 4 | 4000 | 32 | 4 | prefill 主导，attention 可见 | `fused_moe_kernel` | 12.1% |
| `R_concurrent_decode` | 32 | 200 | 256 | 32 | 高并发 decode，MoE batch 行为 | `fused_moe_kernel` | 15.4% |

### 1B. 真实/dataset regime（用于 serving 端 config sweep）
| Regime 名 | 数据来源 | in:out 画像 | 说明 |
|---|---|---|---|
| `toolagent` | mooncake `toolagent` trace（FAST'25 公开） | in~2700 / out~207（≈13:1，prefill 主导） | 主线真实 agent 负载 |
| `shared_prefix` / `generated-shared-prefix` | sglang 合成，高前缀共享 | 长共享前缀 + 短续写 | radix cache 友好 |
| `random` | sglang 随机 in/out | 可调 | 通用基线 |
| `sharegpt` | ShareGPT 对话 | 中等 in/out | 对话画像 |

### 1C. 旧 regime 聚类（Qwen3-0.6B，非当前模型，仅供参考）
来源：`regime_scout/outputs/regime_map.md`（12 workload → 10 cluster，如 `R_scheduler_tail`/`R_scheduler_or_cuda_graph` 等）。**⚠️ 这是在 Qwen3-0.6B 上做的，未迁移到 30B-A3B，需重做或废弃。**

### 缺口
- ⬜ **统一的、当前模型（30B-A3B）的规范 regime 清单**：把 1A/1B 合并成 3–5 个带明确 (batch,in,out,conc,dtype) 的规范点，作为后续所有实验的固定坐标系。

---

## Part 2：每个 regime 的 config autotuning 天花板 + 平台期（Mason 第 2 步）

**Knobs 扫过**：`chunked-prefill-size ∈ {2048,4096,8192,16384}` × `max-running-requests(cap) ∈ {32,64,128,192,256}`（v8，Qwen 实测）。

### 2A. `toolagent`（真实，prefill 主导）— ✅ 有数据
| cap | req/s | out_tok/s | median TPOT(ms) | 备注 |
|---|---|---|---|---|
| 32 | 7.60 | 1413 | 20.9 | |
| 64 | 9.85 | 1832 | 31.7 | |
| **128** | **11.74** | **2181** | 52.7 | **拐点** |
| 192 | 11.63 | 2162 | 53.0 | +0%（平台）|
| 256 | 11.30 | 2100 | 56.0 | 略降 |
→ **plateau 在 cap=128**（扩到 192/256 ≤ ±1–2%）；chunked-prefill 影响 <10%。

### 2B. `shared_prefix`（合成，共享前缀）— ✅ 有数据
| cap | req/s | out_tok/s | median TPOT(ms) |
|---|---|---|---|
| 32 | 14.30 | 3661 | 7.8 |
| 64 | 19.11 | 4893 | 11.2 |
| **128** | **21.76** | **5569** | 19.4 |
| 192 | 21.99 | 5630 | 19.3 |
| 256 | 21.48 | 5499 | 19.5 |
→ 同样 **plateau≈cap=128**；但**绝对吞吐是 toolagent 的 ~2.6×**（前缀共享省 prefill）。**config 价值因 regime 而异的初步证据。**

### 2C. 其余 regime 的 config 曲线 — ⬜ 缺
| Regime | config 曲线 | plateau | config autotune 收益 |
|---|---|---|---|
| `toolagent` | ✅ | cap=128 | 32→128：+55% req/s |
| `shared_prefix` | ✅ | cap=128 | 32→128：+52% req/s |
| `random` | ⬜ | ? | ? |
| `sharegpt` | ⬜ | ? | ? |
| 4 个人造 regime（serving 形式） | ⬜ | ? | ? |

### 待做
- ⬜ 在 `random`/`sharegpt` + 人造 regime 上跑同样的 `chunked × cap` 扫参，产出**并排的曲线族**，量化"config autotune 收益 vs regime"（Mason 要的核心图之一）。**复用现成 harness，工作量小。**

---

## Part 3：重量级 kernel 的 shape 覆盖 + kernel autotuning 状态（Mason 第 3 步）

### 3A. Qwen3-30B-A3B 实际用到的关键 kernel shape
| kernel | 关键 shape | 说明 |
|---|---|---|
| MoE GEMM（`fused_moe`） | **E=128, N=768**（moe_intermediate），hidden=2048，top-8 | 每 token 选 8/128 专家 |
| Attention（fa3/cutlass Sm90） | 32 Q-heads, 4 KV-heads, head_dim=128, GQA | decode 主导热点之一 |
| dense GEMM（nvjet） | qkv_proj / o_proj / gate | cuBLAS，自带 autotune |

### 3B. sglang MoE tuned-config 覆盖对照（★关键发现）
sglang 按 `E={E},N={N},device_name={dev}` + **当前 triton 版本目录**查表。实测：

| 需要的 config | 当前 triton_3_5_1 目录 | 回退情况 |
|---|---|---|
| `E=128,N=768,H200`（bf16） | ❌ **不存在**（3_5_1 里 E=128 只有 B200/H100） | 回退到 `triton_3_2_0/E=128,N=768,H200.json`，打印 **"Performance might be sub-optimal!"** |
| `E=128,N=768,H200`（fp8_w8a8） | ❌ 不存在 | 回退到 triton_3_2_0 的 fp8 版 |

**含义（正好回答 Mason）**：
- Qwen3-30B 的 MoE shape 在 H200 上**有** tuned config，但只在**旧 triton（3.2.0）**目录；当前 triton **3.5.1 没有为 H200 重调过**。
- sglang 用回退逻辑加载 3.2.0 的 block-size 常量，**这些常量是为旧 triton 编译器调的，在 3.5.1 上未必最优** → 这正是 Mason 说的"specific shapes ... aren't already covered and some new kernel autotuning ... in order"。
- **可交付的 PR-式贡献**：为 triton 3.5.1 × H200 × E=128,N=768 重新跑 sglang 的 `benchmark/kernels/tuning_fused_moe_triton.py`，生成 tuned config，量测提升——这就是"补 autotuning 覆盖空白"的具体动作。

### 3C. kernel constexpr autotuning 状态
| kernel | 现状 | constexpr autotune 做了吗 |
|---|---|---|
| `fused_moe`（triton） | 用回退的 3.2.0 config（BLOCK_M/N/K, num_warps, num_stages） | ⬜ 未针对 3.5.1+H200 重调 |
| attention（fa3） | sglang 已选 fa3（v11-A2 证明是三种里最快） | ✅ 已是最优实现，⬜ 未探 constexpr |

### 待做
- ⬜ 系统 dump Qwen3 decode/prefill 触发的**全部 MoE/attention shape**，逐一对当前 triton 目录查表，产出**"shape × 是否有原生 config × 是否回退"**完整对照表。
- ⬜ 为 `E=128,N=768,H200,triton3.5.1` 跑 fused_moe 官方 tuning 脚本 → 得到 constexpr autotune 的**可测量提升**（Part 6 图的第 2 层）。

---

## Part 4：NCU headroom → 选定要手改的单个 kernel（Mason 第 4 步）

### 4A. per-kernel headroom（Qwen3，decode，NCU + roofline + GFLOP/s 实测）
| kernel | SM% | DRAM% | 占带宽屋顶 | **achieved 算力(占bf16峰值)** | headroom 判断 |
|---|---|---|---|---|---|
| `flash_attn`（fa3 Sm90） | 46.6 | 71.9 | 距屋顶 ~1.4× | 34–41% | 已较满，且实现最优 |
| **`fused_moe`（MoE GEMM）** | **16.1** | **75.3** | **距屋顶 ~1.3×** | **7.0–7.6%** | ★**最大 headroom**：算力极低、memory-bound、config 回退 |
| `nvjet`（dense gemm） | 5–10 | 49–65 | — | 5–9% | cuBLAS 自调，动不了 |

### 4B. 选择结论
- **头号候选 = `fused_moe`**，regime 选 **`R_concurrent_decode`（B32）或真实 `agent_decode_b32`**（decode、MoE 主导、gpu_util 最高、NCU 数据最全）。
- 佐证：MoE decode 搬:算 ≈ **103:1**（memory-bound 定量），MoE GEMM 仅 7% 峰值且**几乎不随 batch 改善**（7.0%→7.6%，b32→b64）→ 典型"搬专家权重主导"。
- 融合候选（周边低垂果实）：`fused_moe` 前后的 `topkGatingSoftmax`（router）、`count_and_sort_expert_tokens`（含 atomics 顺序瓶颈，`long_scoreboard`≈3182 warps/issue）、`moe_sum_reduce`、`act_and_mul` → **router→gather→GEMM→scatter→reduce 链是经典可融合序列**。

---

## Part 5：kernel 重写 / 融合（Mason 第 4 步两条路径）— ⬜ 未开始

| 路径 | 做法 | 状态 | 预期难度 |
|---|---|---|---|
| **(a) 硬编码常量的底层 CUDA 重写** | 对 `fused_moe` 把 E=128/N=768/top-8/hidden=2048 等**编译期常量写死**，去通用性换性能，agent 辅助生成 CUDA | ⬜ | ★★★★ |
| **(b) 融合周边 kernel** | 把 router→gather→MoE GEMM→scatter→`moe_sum`→`act_and_mul` 中低垂的几段**融合成一个 kernel**，减少中间结果反复读写 HBM（经典解 memory-bound）；可能需 torch.compile 替换算子序列 | ⬜ | ★★★☆ |

**执行方式**：agent 辅助 + researcher-in-the-loop。**先做 (b) 的最小融合**（如 act_and_mul + moe_sum，或 router+gather），风险低、直接打 memory-bound。

---

## Part 6：最终交付——分层逼近 roofline 总图 + 关键指标 X（Mason 第 5 步）— ⬜ 待填

对**选定 regime（如 R_concurrent_decode / agent_decode_b32）**，画 decode 整步或选定 kernel 逼近 roofline 的分层进度：

| 层级 | 手段 | 相对上一层的提升 | 达到的 %屋顶 | 状态 |
|---|---|---|---|---|
| 0. baseline | 默认 config（含 MoE config 回退） | — | fused_moe ~7% 峰值算力 / 距带宽屋顶 1.3× | ✅ 已测 |
| 1. config autotune | chunked × cap 调到 plateau（cap=128） | 32→128：吞吐 +52–55% | （serving 层，非 kernel roofline） | ✅ 已测（2 regime） |
| 2. kernel constexpr autotune | 为 triton3.5.1×H200×E=128,N=768 重跑 fused_moe tuning | **?**（待 §3 补） | ? | ⬜ |
| 3. kernel 手改/融合 | §5 的 (a)/(b) | **X = ?**（相对第 2 层的额外提升） | ? | ⬜ |

**关键产物 = X**（第 3 层相对第 2 层的额外提升）：
- X 小 → **autotuning 故事**（补 config 覆盖空白，走 PR 路线）。
- X 大 → **kernel-agent 故事**（手改 agent 有独立价值，且有实证）。

---

## 附：一页缺口清单（我还缺什么）

| 缺口 | 属于 Mason 第几步 | 工作量 | 依赖 |
|---|---|---|---|
| ⬜ 当前模型的统一 regime 清单（3–5 个规范点） | 1 | 小 | — |
| ⬜ `random`/`sharegpt`/人造 regime 的 config 扫参曲线 | 2 | 中 | 规范 regime 清单 |
| ⬜ 完整 shape × config 覆盖对照表 | 3 | 小 | — |
| ⬜ 为 triton3.5.1×H200×E=128,N=768 重跑 fused_moe autotune（constexpr 层提升） | 3→6 层2 | 中 | — |
| ⬜ fused_moe 手改/融合（(a) 硬编码 / (b) 融合链） | 4–5 | 大 | 上一项 |
| ⬜ 分层逼近 roofline 总图 + 量出 X | 6 | 中 | 以上全部 |

**建议起步顺序**：①统一 regime 清单 → ②补 config 曲线族（最快交付 Mason 第 2 步的图）→ ③shape 覆盖对照 + 重跑 fused_moe autotune（拿到第 2 层提升）→ ④fused_moe 融合（拿 X）→ ⑤拼总图。

---

## 数据出处
- regime：`docs/2026-06-09/sglang_triton_4regime_profiling.md`、`regime_scout/outputs/regime_map.md`
- config sweep：`results/consolidated_v8_tuning.csv`（v8）、`consolidated_v7_*.csv`
- NCU/roofline：`results/consolidated_v9_ncu.csv`、`docs/2026-07-14/opportunity_gap_comprehensive_analysis.md`（证据 B–F2）
- GFLOP/s：`results/2026-07-15_v18_gflops/gflops_accurate.json`
- 干预（可回收）：`docs/2026-07-15/v11_realize_gap_results.md`、`v12_ncu_spec_mechanism.md`
- shape 覆盖：sglang `fused_moe_triton/configs/triton_3_5_1/` vs `triton_3_2_0/`，loader `fused_moe_triton_config.py:61,80-112`
