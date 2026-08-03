# 交付物梳理：mentor 要什么，我们有什么，哪些扣得上

**日期**：2026-08-03
**用途**：把 Dey 和 Mason 的要求逐条列出，对照我们已有的实验结果，判断哪些能作为交付证据

---

## 第一部分：mentor 到底要什么

### Debadeepta Dey 的核心要求

原话：

> "Our aim is not to get torch compile working. It is to show that for different
> regimes we can genetically rewrite kernels to improve **beyond what the best
> auto tuning config provides**."

拆成三条要证明的：

| # | 要证明的 |
|---|---|
| D1 | 不同 regime 需要不同的 kernel specialization |
| D2 | **最佳 autotuning config 不是终点**（这条是核心） |
| D3 | 修改 kernel 的算法、结构或 fusion boundary 能继续带来提升 |

他的成功标准不是"系统有多少组件"，而是：

> 有没有一个**可信、可复现**的实验说明：autotuning 到这里就停了，
> 但 kernel rewrite 又向前推进了 X%。

**明确说了不要的**：把 torch.compile 搞通、被 side quest 分散、无止境改 slides。

### Mason Remy 的实验闭环

他给的是一条**证据链**，每一环都要有：

| # | 环节 |
|---|---|
| M1 | Categorize regimes |
| M2 | 对每个 regime，展示 SGLang config tuning 的收益和 **plateau** |
| M3 | 检查 heavyweight kernels 的**现有 kernel autotuning 是否覆盖真实 workload shapes** |
| M4 | 选一个 **NCU 显示有显著 headroom** 的 kernel |
| M5 | 做 low-level rewrite，**或**把目标 kernel 与周围 elementwise kernels 融合 |
| M6 | 展示 kernel editing 在 autotuning 之上的**额外**增益 |

他还强调：**要从真实端到端运行中的 shapes 和 kernel sequence 出发**，
不要平均对待孤立 microbenchmark 里的所有 shape——只有真实 workload 里出现的 shape 才重要。

---

## 第二部分：我们手上有什么（全部已核实，非记忆）

### 结果 A —— LFM2.5：config autotuning **零收益**

来源：`docs/2026-06-30/lfm2.5_conditional_autotuning.md`

| 臂 | R_concurrent_decode |
|---|---|
| **cookbook 默认**（3 次独立 server lifetime） | **23.74 ± 0.12 req/s**（stddev 0.5%） |
| Optuna 条件化搜索 "best"（25 trial / 288 组合） | 22.32 → **低 6%** |
| 手工修正（MoE backend 换回 triton） | 23.53 → **持平** |

报告结论原文：**"sglang 团队的开箱默认就是最优"**。

> **扣 D2 和 M2。** 而且是比"plateau"更强的形式：不是收益递减，是**完全推不动**。

**⚠ 弱点**：只有 25 次 TPE trial，而且报告自己写了 TPE 坏掉了
（前 7 个 trial 把 `triton MoE` 和差 batching 绑一起，之后 18 个再没试过好组合）。
审稿人会说「这不是 ceiling，是搜索失败」。**必须补网格穷举。**

### 结果 B —— LFM2.5：kernel 工作 +5.30% ~ +6.57%

来源：`docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md`

| regime | 七项全开 | p |
|---|---|---|
| A 低批 decode | **+6.57%** | 4.6e-14 |
| B 并发 decode | **+6.21%** | 2.4e-08 |
| C 长 prefill | **+5.30%** | 1.2e-05 |

七项明细：

| 项 | 类型 | 主要收益 regime |
|---|---|---|
| `conv` | **手写 Triton kernel** | 长 prefill +2.33% |
| `moesum` | **手写 Triton kernel** | 低批 decode +4.55% |
| `qkrope` | 调用点（融合 kernel） | 并发 decode +5.42% |
| `norm+scale` | 调用点 | decode +3.89% |
| `gate+idx` | 调用点 | **三 regime 全不显著**（诚实负面） |

> **扣 D1（四种不同形状的收益，只测一个 regime 一个都看不全）、D3、M5、M6。**

**关键**：serving 配置与结果 A 完全一致（`mem 0.85 / lpm / cap 32 / chunk -1`），
所以两者可以画在同一张图上。

**⚠ 弱点**：跑的时候树里**没有** tuned MoE config（已由实验 1 确认），
所以这是叠在"cookbook 默认"上，不是叠在"最优 kernel config"上。

### 结果 C —— LFM2.5：H200 MoE tuned config +23.3%

来源：`docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md`（PR #32687）

| regime | 基线 | 补 config 后 | 提升 | p |
|---|---|---|---|---|
| C 长 prefill (in=4000, out=32, conc=4) | 12.277 req/s | **15.142** | **+23.34%** | 1.3e-10 |

上游 #22791 给 LFM2 的 MoE shape 做过 tuning，覆盖 H100 / B200 / MI325X，**唯独没有 H200**。
`get_moe_configs` 的查找 key 含 `device_name`，版本回退只换 `triton_*` 目录、文件名不变，
所以 **H200 永远拿不到 H100 的 config**。

> **扣 M3**（现有 kernel autotuning 是否覆盖真实 workload shapes —— 答案是没覆盖）。

### 结果 D —— Gemma-3：RMSNorm dispatch 修复

来源：`docs/2026-07-28/three_fusion_cases.md`，PR #32670

| regime | 增量（对 #32383 后的等价基线） |
|---|---|
| 低批 decode | **+36.6%** |
| 并发 decode | **+24.5%** |
| 4K prefill | +7.3%（n.s.） |

> **扣 D3、M5、M6。已经上游 PR，是外部可验证的证据。**

### 结果 E —— Gemma-3：fused_qk_norm_rope 接线

来源：`docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md`

| regime | 对 main | **真实增量**（对含 #32670 的基线） | p |
|---|---|---|---|
| decode bs=1 | 1.387× | **1.005×** | 4e-05 |
| decode bs=32 | 1.385× | **1.004×** | 6e-05 |
| decode bs=64 | 1.338× | 1.007× | 0.073 **n.s.** |
| prefill heavy | 1.395× | **1.055×** | 0.008 |

> **这个案例的价值不在收益（+0.5~5.5%），在方法学**：
> 对 main 读数 1.39×，加消融臂后发现 **97% 属于另一个在飞的 PR**。
> 这是"如何不谎报"的教材。

### 结果 F —— OLMo-2：绕过自己的融合 kernel

来源：`docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md` §7.4，上游 issue #33415 / draft PR #33416

| 指标 | 结果 |
|---|---|
| **prefill 直测** | 87.80ms → 70.79ms = **1.24×** |
| prefill-heavy 吞吐 | **+17.61%**，p<0.001，7 次重复 |
| decode bs=1/32/64 | 1.00×，**全部 n.s.**（符合机制预期） |
| 数值 | **bit-identical**，max abs diff 0.0 |
| GSM8K 400 题配对 | 65.50%→65.25%，**McNemar p=1.000** |

> **扣 M1+M4+M5 的完整链条 —— 这是唯一一个"两个工具交叉才找到"的案例，详见第四部分。**

**⚠ 弱点**：**没有做过 config autotuning 基线**，所以单独拿它证明不了 D2。

### 结果 G —— 方法学产出（超出要求，是加分项）

| 产出 | 内容 |
|---|---|
| **同类优化强烈次可加** | 兑现率 **0.90 / 0.70 / 0.49**，随 regime 饱和度单调下降 → 各项分别测量之和会高估整个 stack，系统越饱和高估越严重 |
| **regime→backend 规则跨模型不可迁移** | 用错最差 **−34%** → backend 选择是避坑杠杆，不是提速杠杆 |
| **家族关注度预测指标** | 缺口大小不取决于架构新旧或模型大小，而是**该模型文件受过多少优化关注**；四个 Qwen 模型全<1%，三个非 Qwen 家族全>6% |

### 结果 H —— Agent loop（SLO-agent）

| 内容 | 状态 |
|---|---|
| `scan` 阶段 + `kernel_fusion_gap` mode | PR #9 |
| 回测 6 个历史案例 | **5/5 重现**（第 6 个已被上游修掉，正确报 N/A） |
| 语义等价闸门 | 6.62% 缺口 + 性能闸门全绿 → 仍被一票否决 |
| 案例知识库 | PR #30，6 个确认 + 5 个否决 |

> **扣 Mason 的 "agent assisted but researcher-in-the-loop"。**
> 但要诚实说明：scan 的规则是从已知案例反推写的，证明的是"方法可复用"，不是"能预测"。

---

## 第三部分：对照表 —— 哪条要求由哪个结果满足

| 要求 | 满足它的结果 | 状态 |
|---|---|---|
| **D1** 不同 regime 需要不同 specialization | B（四项收益形状完全不同）、G（次可加性随饱和度变化） | ✅ 强 |
| **D2** best autotuning 不是终点 | **A + B 组合**（同一基线） | ⚠️ **需补实验 2** |
| **D3** kernel 改动能继续提升 | B、D、F | ✅ 强 |
| **M1** Categorize regimes | A/B/C 三 regime 贯穿所有实验 | ✅ |
| **M2** config tuning 收益与 plateau | A（LFM2.5 零收益）+ Qwen 对照（4.75~8.86×） | ⚠️ 需补实验 2 |
| **M3** kernel autotuning 是否覆盖真实 shapes | **C**（H200 缺 config，+23.3%） | ✅ 完美扣题 |
| **M4** NCU 显示 headroom 的 kernel | 数据在 `results/2026-07-08_v5_ncu/` 等，**但没串进叙事** | ⚠️ 需补实验 4 |
| **M5** rewrite 或与 elementwise 融合 | B（2 个手写 Triton）、D、F | ✅ 强 |
| **M6** kernel editing 在 autotuning 之上的额外增益 | **A + C + B 三层** | ⚠️ **需补实验 3** |
| 真实 workload shapes | 全部实验用真实 serving harness | ✅ |
| Agent workflow demo | H | ✅ |
| 诚实的 limitations | E（消融纠正）、G、R01-R05 否决案例 | ✅ 强 |

---

## 第四部分：交付结构建议

### 主线（回答 D2 / M2 / M6）—— LFM2.5 四层图

```
Bar 1  sglang 裸默认
Bar 2  cookbook 默认            ← 25 trial Optuna 无法超越 = ceiling（结果 A）
Bar 3  + tuned MoE kernel config  +23.3%（结果 C）← kernel autotuning
Bar 4  + kernel rewrite/fusion    +X%（结果 B 重测）← 论点在这一格
```

**这张图直接回答**：kernel rewrite 在 best autotuning 之上还贡献了多少？

### 泛化证据（回答"不是只对一个模型有效"）

- **OLMo-2**（结果 F）：prefill 1.24×，扫描+profiling 交叉发现，bit-identical
- **Gemma-3**（结果 D）：+36.6%，已上游 PR #32670

**三个模型、三个家族（Liquid / AI2 / Google）都找到 kernel 级机会**，
本身就是"不是碰巧"的证据。

### 方法学（加分）

结果 G 的三条 + 结果 E 的消融纠正 + agent loop 回测。

---

## 第五部分：还缺什么

| # | 实验 | 补的是 | 耗时 |
|---|---|---|---|
| **2 ★** | **裁剪空间网格穷举**取代坏掉的 TPE | D2 / M2 的可信度 | 2–4h |
| **3** | 结果 B 在装了 tuned config 的基线上重测 | M6 的干净基线 | 3–4h |
| 4 | NCU headroom 串进叙事 | M4 | 1h |

**实验 2 最关键**。现在的 ceiling 建立在一次自己承认失败的搜索上。

**实验 3 的预期**：按次可加性（0.90/0.70/0.49），+6% 可能缩到 **+2~4%**。
这不是坏消息——**基线干净的 +2% 比基线脏的 +6% 强**，我们已经因为基线脏栽过两次
（#32383、#32670）。

---

## 第六部分：⚠️ 一个必须处理的内部矛盾

`docs/2026-06-25/autotuning_ceiling_report.md`（Qwen3-30B）原文：

> "**This challenges the original 'agent for kernel rewriting' motivation**:
> the gap between default and ceiling is *huge*... Agent value should be
> redirected toward *automating the search itself*, **not rewriting kernels**."

**这份和 kernel 成果一起交，等于在论证 Dey 的目标不成立。**

**根因是苹果比橘子**：

| | 含义 |
|---|---|
| Qwen 的 `8.86×` | **default → tuned**（众所周知的巨大空档） |
| LFM 的 `+6%` | **在已调优基线之上**的增量 |

两个数不可比，但柱状图一放读者一定会比。

**处理方式**：以 **LFM2.5 为主线**（那里 config autotuning 零收益，
"autotuning 是上限"是更强的形式），Qwen 那份降为
「不同模型的 ceiling 高度差异极大，所以必须逐模型判断」的对照证据。

---

## 第七部分：建议停掉的

按 Dey 的 "don't get distracted by side quests"：

- ❌ 通用 SGLang full-model FX importer
- ❌ 让所有模型支持 torch.compile
- ❌ SLO-agent 继续加功能（两个 PR 已够 demo agent loop）
- ❌ 扫更多新模型找新机会（6 个案例够了）
- ❌ 继续改 slides

---

## 附：所有结果的证据文件

| 结果 | 文档 | 原始数据 |
|---|---|---|
| A | `docs/2026-06-30/lfm2.5_conditional_autotuning.md` | `results/2026-06-30_lfm2.5/` |
| B | `docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md` | `results/lfm_fusion/` |
| C | `docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md` | PR #32687 |
| D | `docs/2026-07-28/three_fusion_cases.md` | PR #32670 |
| E | `docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md` | `results/fx_fusion/e2e_ab_gemma3_ablation.json` |
| F | 同上 §7.4 | `results/fx_fusion/e2e_ab_olmo2.json`、`gsm8k_paired_olmo2.json` |
| G | `docs/2026-07-27/...` §8、`docs/2026-07-28/cross_architecture_audit.md` | — |
| H | SLO-agent PR #9 / #30 | `results/slo_agent_run/` |
| Qwen 对照 | `docs/2026-06-25/autotuning_ceiling_report.md` | — |
