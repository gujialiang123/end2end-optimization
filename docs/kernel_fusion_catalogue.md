# Kernel fusion 机会全集：我们找到的每一个、怎么找的、拿到多少

**维护范围**：本项目至今全部 kernel fusion 相关的发现与验证（2026-07-27 ~ 2026-08-01）
**硬件**：1× NVIDIA H200 · BF16
**这份文档的定位**：**汇总索引**。每个案例给出「怎么发现的 / 改了什么 / 收益 / 证据在哪」，
细节指向各自的原始文档。

---

## 0. 一张表看全部

| # | 模型 | 已存在的原语 | 谁在用 | 谁漏了 | **端到端收益** | 发现方式 | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | LFM2.5 | `fused_add_rmsnorm` | llama / qwen2 / 几乎所有 | `Lfm2MoeDecoderLayer` | +2.35% decode | 静态扫描 1a | 已验证 |
| 2 | LFM2.5 | `fused_qk_norm_rope` | Qwen3-MoE | `Lfm2MoeAttention` | **+5.42%** 并发 decode | 静态扫描 1b | 已验证 |
| 3 | Gemma-3 | `gemma_rmsnorm` | **同文件 100 行之上的 `GemmaRMSNorm`** | `Gemma3RMSNorm.forward_cuda` | **2.13×**（对旧 main）<br>**+36.6%**（对 #32383 后的真实增量） | profiling + 读代码 | **PR #32670** |
| 4 | Gemma-3 | `gemma_rmsnorm`（rank 守卫） | 2-D 输入 | **4-D 的 q_norm/k_norm** | 含在 #3 的增量里 | profiling + 读代码 | 同上 |
| 5 | Gemma-3 | `fused_qk_norm_rope` | qwen3_moe / deepseek_v4 / mellum / interns1pro | `gemma3_causal.py` 从未提及 | **+0.5% ~ +1.1%** | **静态扫描 1b（自动）** | 已验证 |
| 6 | **OLMo-2** | **它自己的融合 norm 路径** | 自己的 capture-mode 分支 | **非 capture 路径显式调 `forward_native`** | **prefill 1.24× / +24%** | **扫描 + profiling 交叉** | 已验证 |

**一句话概括这 6 个案例**：

> 融合 kernel 已经写好、测好、编译进二进制里了。**某个模型的调用点没有调用它。**

六次都**没有发明任何新东西**，只是发现「这里本该调用那个」。

### 未兑现 / 被否决的（同样重要）

| 案例 | 结论 | 为什么记录它 |
|---|---|---|
| QK-norm 手工合并（per-head 权重） | **作废** | 需新写 kernel、T=4096 退化到 0.56×，被案例 5 全面取代 |
| Gemma-3 残差加法（第二个缺口） | −0.09% / +0.44%，**均不显著** | 大修落地后，同类剩余开销**没有可转换的余量**了 |
| Triton 3.6 MoE 调优 | **撤回** | 基线污染（见 `RETRACTION_triton36_baseline_contamination.md`） |
| OLMo-2 用 `fused_qk_norm_rope` | **不可用** | 跨 head 归一化 vs per-head kernel，**语义不等价** |
| EXAONE-4 用 `fused_qk_norm_rope` | **不可用** | `rope_type=llama3`，kernel 只有 default/YaRN |

---

## 1. 案例 1–4：LFM2.5 与 Gemma-3 RMSNorm

**完整技术细节见 `docs/2026-07-28/three_fusion_cases.md`。** 这里只留摘要。

### 案例 1：LFM2.5 residual 加法从未被融合

`lfm2_moe.py:433-456` 里 `residual = hidden_states` 覆盖了传入的 residual 参数，
导致 `operator_norm` 没收到 residual → 走非融合分支 → 加法变成独立 elementwise kernel。

**改法**：改用 deferred residual 惯例（`models/llama.py:304-316` 是标准写法）。
**收益**：每层省 2 个 kernel × 24 层 = 48 个，decode **+2.35%**。

### 案例 2：LFM2.5 QK-norm + RoPE 没有融合

`Lfm2MoeAttention` 分开调 q_norm / k_norm / rotary，而 Qwen3-MoE 已经在用 `fused_qk_norm_rope`。

**收益**：并发 decode **+5.42%**——那一轮单项最大赢家，纯调用点改动。

### 案例 3+4：Gemma-3 RMSNorm 整个掉进 eager ★

`Gemma3RMSNorm.forward_cuda` 直接 `return self.forward_native(x)`，
而**同一个文件、约 100 行之上**的 `GemmaRMSNorm` 正在调用 `gemma_rmsnorm`。
1 个 kernel 变成 6 个。

**两个只有实测才发现的坑**：
1. **dtype 不匹配静默返回 NaN**——`nn.Parameter(torch.zeros(dim))` 是 fp32，激活是 bf16，
   融合 kernel 不报错、直接给 NaN
2. **rank > 2 的输入**——`q_norm`/`k_norm` 收到 `[tokens, heads, head_dim]`，
   2-D 守卫把它们静默留在慢路径上，**这就是 1.56× 和 2.13× 的差距**

**收益**：2.13×（对当时的 main）。
**但上游 #32383 在我们提交前落地了 2-D 和 residual 那两半**，
诚实的数字是**对 main 等价基线的增量 +36.6% / +24.5% / +7.3%(n.s.)**，约为原数字的三分之一。
→ **PR #32670**

---

## 2. 案例 5：Gemma-3 漏用 `fused_qk_norm_rope`

**完整记录见 `docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md`。**

### 怎么发现的：静态扫描，全自动

```bash
grep -rl "fused_qk_norm_rope" python/sglang/srt/models/
#   qwen3_moe.py  deepseek_v4.py  mellum.py  interns1pro.py
grep -c "fused_qk_norm_rope" python/sglang/srt/models/gemma3_causal.py
#   0
```

脚本化后（`scripts/fx_fusion/scan_qknorm_rope_candidates.py`）扫全部 212 个模型文件：
**4 个已用，18 个有独立 q_norm/k_norm 调用且未融合 rope，16 个不需要新 kernel 变体**。

**这次是纯静态扫描发现的**——不需要 profiler、不需要人读代码。

### 改了什么

kernel 侧一处（`fused_qknorm_rope.cuh`），加编译期标志：

```cpp
template <int head_dim, bool interleave, bool yarn, bool add_one = false>
...
float w = device::cast<float>(wvec[i]);
if constexpr (add_one) {
  w += 1.0f;              // Gemma 的 (1+w)，在 fp32 域做
}
elements[i] *= rms_rcp * w;
```

模型侧接线，构造期解析一次能力，forward 里只是一个 bool 判断。

### 收益：+0.5% ~ +1.1%（不是 +38%）

| regime | main | 融合norm基线 | 全融合 | vs main | **真实增量** | p |
|---|---|---|---|---|---|---|
| decode bs=1 | 3.01ms | 2.18 | 2.17 | 1.387× | **1.005×** | 0.000 |
| decode bs=32 | 3.31 | 2.40 | 2.39 | 1.385× | **1.004×** | 0.000 |
| decode bs=64 | 3.72 | 2.80 | 2.78 | 1.338× | **1.007×** | 0.073 **n.s.** |
| prefill heavy | 3.32 | 2.51 | 2.38 | 1.395× | **1.055×** | 0.008 |

**「vs main」那一列不能用**：main 里还有 PR #32670 正在修的 rank 守卫缺口，
**其中约 97% 是那个 PR 的功劳**。

微基准是 1.5–2.3×，端到端只有 1%——26 层 × 每层省 3.5us ≈ 91us，
对 2.2ms 的 decode 步上限就 4%。**微基准不能外推。**

### 精度：不降反升

对 fp64 参考，**kernel 0.141% vs 模型现有路径 0.14–0.23%**，都在 bf16 ULP（0.39%）内。
融合后更准，因为它全程 fp32 寄存器，norm 和 rotate 之间不落 bf16。

GSM8K 400 题配对 McNemar：21.50% → 22.00%，**21 胜 19 负，p=0.875，无可检测变化**。

---

## 3. 案例 6：OLMo-2 绕过自己的融合 kernel ★ 收益最大

**完整记录见 `docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md` §7.4。**

### 怎么发现的：扫描 + profiling **交叉**

这是本项目唯一一个**两个工具各说对一半、合起来才定位**的案例。

- **静态扫描**说：OLMo-2 **不能**用 `fused_qk_norm_rope`
  （它的 `q_norm = RMSNorm(hidden_size)` 是跨 head 归一化，kernel 是 per-head，**语义不等价**）
- **Profiling** 说：OLMo-2 有 **97 次 eager-norm 调用 = 6.62% 的 decode kernel 时间**

两个都对。于是问题变成：**它到底为什么是 eager 的？**

### 问题代码（`olmo2.py:165-192`）

```python
if self.alt_stream is not None and get_is_capture_mode():
    q_by_last = q.reshape(-1, q_shape[-1])
    q_by_last = self.q_norm(q_by_last)        # ← 走融合 kernel
    ...
else:
    q = self.q_norm.forward_native(q)          # ← 显式绕过融合 kernel
    k = self.k_norm.forward_native(k)
```

- `q` 来自 `qkv.split(...)`，本来就是 2-D `[tokens, q_size]`
- `q_norm = RMSNorm(hidden_size)`，宽度正好对上
- **`self.q_norm(q)` 直接就能命中融合 kernel**，`forward_native` 是白丢

`get_is_capture_mode()` **只在 CUDA graph 捕获期为真**：

| 阶段 | 走哪条 | 后果 |
|---|---|---|
| decode（graph replay） | 捕获时用的融合路径 | 已经是快的 |
| **prefill**（从不被捕获） | **eager** | **一直在丢** |

### 改了什么：3 行

```python
q_shape, k_shape = q.shape, k.shape
q = self.q_norm(q.reshape(-1, q_shape[-1])).view(q_shape)
k = self.k_norm(k.reshape(-1, k_shape[-1])).view(k_shape)
```

### 收益：prefill 1.24× / +24%

| 闸门 | 结果 |
|---|---|
| **prefill 直接测量** | **87.80ms → 70.79ms = 1.24×** |
| prefill-heavy 总吞吐 | **+17.6%，p<0.001** |
| decode 各 regime | 1.00×，全部 n.s.——**符合机制预期** |
| GSM8K 配对 McNemar | 65.50% → 65.25%，**p=1.000 无变化**（8 胜 9 负） |
| greedy 生成（8 条 prompt） | 8/8 完全一致 |

**decode 无变化恰恰证明机制判断对了**：decode 本来就在跑融合路径。

**同家族对照**：OLMoE 直接写 `self.q_norm(q.contiguous())`，**没有这个 bug**——
profiling 里它是 1 次 eager-norm 调用，OLMo-2 是 97 次。**扫描正确区分了同家族两个模型。**

---

## 4. 发现方式的统计

把 6 个案例按「假设从哪来」分类：

| 发现方式 | 案例 | 占比 |
|---|---|---|
| 静态扫描（自动） | 1, 2, 5 | 3/6 |
| profiling + 人读代码 | 3, 4 | 2/6 |
| **扫描 + profiling 交叉** | 6 | 1/6 |

**FX / torch.compile 发现了几个？0 个。**
（FX 在本项目里的实际作用是**验证与量化**，不是发现。
详见 `docs/2026-07-31/two_fusion_cases_full_record.md`。）

### 预测指标（已在 10 个模型上验证）

**不是**架构新旧、**不是**模型大小，而是**该模型文件受过多少优化关注**，
这跟家族在框架用户群里的分量相关：

| 模型 | 家族 | eager-norm 调用 | 占 decode kernel 时间 |
|---|---|---|---|
| Gemma-3-1B | Google | 157 | **6.41%** |
| **OLMo-2-1B** | AI2 | **97** | **6.62%** |
| LFM2.5-8B | Liquid | — | **11.31%** |
| OLMoE-1B-7B | AI2 | 1 | 0.13% |
| Qwen3-0.6B | Qwen | 1 | 0.32% |
| Qwen3-Coder-Next | Qwen | — | 0.64% |
| Qwen3-30B | Qwen | — | 0.23% |
| Qwen3-32B | Qwen | — | 0.05% |

**四个 Qwen 模型（dense/MoE/linear-attention，0.6B–80B）全部低于 1%；
三个非 Qwen 家族全部高于 6%。** 架构类型、架构新旧、模型大小三个假设都被这张表排除
（Qwen3-0.6B 比 Gemma-3-1B **更小**却干净 81 倍）。

> **可行动形式：优先审计不属于框架主力用户群的家族。**

---

## 5. 方法论：踩过的坑（全部写进 `.github/skills/fusion-gap-hunting/SKILL.md` v3）

### 5.1 静态扫描的假阳性——每一条都是**静默错误**

| 假阳性 | 表象 | 真相 |
|---|---|---|
| `QuickGELU` | `forward_cuda` 转 native，`forward_hip` 调 `gelu_quick` | `gelu_quick` 只在 `elif _is_hip` 下 import，**CUDA build 里不存在** |
| **OLMo-2 / OLMoE** | 源码形态与 Gemma-3 一模一样 | **跨 head 归一化 vs per-head kernel，语义不等价** |
| **EXAONE-4** | head_dim=64 在支持范围内 | **`rope_type=llama3`，kernel 只有 default/YaRN** |
| `Ernie4_5_VLRotaryEmbedding` | kernel 在 `positions.ndim==2` 守卫后 | **另一条分支调了别的 kernel**，不是缺口 |
| Qwen3-0.6B | 从不提 `fused_qk_norm_rope` | 通过 helper 调用了 `fused_qknorm_warp` |

**这几类失败不会报错、不会跑挂，只会给出错的数。**
每一条都必须有对应判据写进筛选器。

### 5.2 测量方法的坑

1. **拿近似比近似，只能证明不一致，不能定位谁错。**
   两次误判（4% 精度损失、87% rope 不一致）都是这个原因。
   必须用 fp64 第三方参考 + **平均**相对误差（不是最大值，会被近零元素主导）。
2. **噪声基线必须真的能动。** greedy 解码换 seed 恒为 0，
   用它当分母任何差异都"显著"。同题两臂要用**配对 McNemar**。
3. **token 一致性不是合适的闸门**——改变舍入的融合本就不可能 bit-exact。
   要用任务指标（dense 模型）；MoE 模型连任务指标都要小心
   （top-k 路由是离散 argmax，bf16 扰动会让专家选择跳变）。
4. **基线要选「当天的 main 加上所有在飞的修复」。** 本项目**两次**踩到：
   #32383 让 2.13× 变成在 claim 别人的工作；
   案例 5 对 main 的 1.39× 里 97% 是 PR #32670 的。
   **消融臂要放在同一棵树里用环境变量切换**——建第二棵 worktree 会引入布局差异
   （强制 `.contiguous()` 改变了 eager 路径原本保留的 stride，attention backend 的 `view()` 直接拒绝）。
5. **微基准加速比不能外推成端到端。** 2.3× 的 preamble 换来 +1%。
6. **`--disable-cuda-graph` 会改变走哪条分支。**
   OLMo-2 的融合分支条件是 `get_is_capture_mode()`，
   关掉 graph 会强制走 eager 分支——**那 6.62% 对 prefill 成立，对 graphed decode 不成立**。
7. **先确认模型在没有你的改动时是否本来就挂。**
   OLMo-2 + fa3 + CUDA graph 在未打补丁的 main 上就报 `cudaErrorIllegalAddress`。

### 5.3 一个横向观察

6 个案例都是「框架有 kernel，调用点没接上」，但**表现形态分两类**：

| 形态 | 案例 | 隐蔽程度 |
|---|---|---|
| **从没接过** | 1, 2, 3, 5 | 静态扫描能看见 |
| **接了一半**（只在某条路径/某个形状上接） | 4（只接 2-D）, 6（只接 capture 路径） | **静态扫描看不见**，且 decode profile 可能是干净的 |

第二类更危险，因为**源码里确实出现了那个 kernel 的名字**，grep 会认为它已被使用。

---

## 6. 证据文件索引

| 案例 | 原始文档 | 数据 | 补丁 |
|---|---|---|---|
| 1, 2 | `docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md` | `results/lfm_fusion/` | — |
| 3, 4 | `docs/2026-07-28/three_fusion_cases.md` | — | PR #32670 |
| 5 | `docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md` | `results/fx_fusion/e2e_ab_gemma3_ablation.json`<br>`results/fx_fusion/gsm8k_paired.json`<br>`results/fx_fusion/accuracy_vs_fp64.csv` | `patches/gemma3_fused_qknorm_rope/` |
| 6 | 同上 §7.4 | `results/fx_fusion/e2e_ab_olmo2.json`<br>`results/fx_fusion/gsm8k_paired_olmo2.json` | `patches/olmo2_fused_qk_norm/` |
| 跨模型扫描 | 同上 §7 | `results/fx_fusion/model_pipeline_profiled.json`<br>`results/fx_fusion/qknorm_rope_candidates.json` | — |
| 撤回记录 | `docs/2026-07-29/RETRACTION_triton36_baseline_contamination.md` | — | — |
| 方法论 | `.github/skills/fusion-gap-hunting/SKILL.md` (v3) | — | — |

---

## 6b. 这套方法已接入 agent loop（2026-08-02）

以上 6 个案例的发现方式已经写成规则，接进了 SLO-agent 的 loop
（分支 `feat/kernel-fusion-gap-mode`）。

**back-test 结果：5/5 重现**（第 3 个已被上游 #32383 修掉，正确报 N/A），
包括最难的 OLMo-2（grep 看不见、decode profile 干净）。

新增了一种本文档此前没有系统扫过的形态：

> **`fused_add_rmsnorm` 从来不被任何模型文件写出名字**——它靠「调 norm 时传两个参数」触发。
> 按名字扫返回 137 个候选（无用）；改扫**调用约定**后收敛到 4 个 plain-add 候选，
> 其中 2 个是已知的 LFM2.5，2 个是全新的：`Exaone4DecoderLayer`、`JetNemotronDecoderLayer`。

**但新候选不等于新机会**：Exaone4 profiling 实测只占 **0.45% of decode kernel time**，
被闸门正确否决（需 ≥3%）。

自由模式与 loop 模式的完整对比见
`docs/2026-08-02/free_exploration_vs_agent_loop.md`。

---

## 7. 下一步

**优先级 1 — OLMo-2 提 PR**：prefill 1.24×、改动 3 行、GSM8K p=1.000 无变化。
性价比最高，且缺口形态（「接了一半」）是个未被上游注意的类别。

**优先级 2 — 把「接了一半」做成可扫描的**：
- 已有：`scan_fusion_gaps.py` 的 scan 1c（AST 分析守卫，抓形状分流）
- 缺：**按执行路径分流**的检测（`get_is_capture_mode()` 这类）——
  案例 6 是人读出来的，规则可以机械化：
  *一个模块内，某条分支调融合 kernel、另一条调 `forward_native`，且分支条件不是输入属性*

**优先级 3 — 剩余 16 个候选**：`qknorm_rope_candidates.json` 里还有 16 个模型
有独立 q_norm/k_norm 调用且未融合 rope，本次只验了 2 个。
每一个都要先过**语义等价闸门**（§5.1）才能测。
