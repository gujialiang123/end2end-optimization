# 三个 kernel 级案例的对照报告：重写 vs 补融合 vs 接线

**日期**：2026-07-28 · **硬件**：H200 · bf16 · **所有数字均为本项目实测**，非上游自称
**范围**：本报告只收 kernel 层的干预。config 调优、投机解码、backend 选择不在内。

---

## 0. 为什么把这三个放在一起

这三件事表面上都叫"优化 kernel"，但它们是**三种完全不同的干预**，而且结论几乎相反：

| | 案例 1 · Qwen3-30B | 案例 2 · LFM2.5-8B | 案例 3 · Gemma-3 |
|---|---|---|---|
| **干预方式** | 从零**重写**一个 kernel | **补上**框架没有的融合 | **接上**框架已有的 kernel |
| **谁写的 kernel** | 我写的 Triton | 我写的 Triton | **上游早就写好了** |
| **改动量** | ~400 行新 kernel | 7 个组件 | **一处 dispatch** |
| **隔离层收益** | 1.23× | 各 1.2–3× | ~6 kernel → 1 |
| **端到端** | **+1.17%**（且 b≥2 回归） | **+5.3% ~ +6.6%** | **+24.5% ~ +36.6%** |
| **判定** | ❌ 通用改动不成立 | ✅ 成立 | ✅ 成立，已提 PR |

**收益和工作量成反比。** 花最大力气自己写 kernel 的那个，端到端几乎为零还带负作用；花最小力气把已有 kernel 接上去的那个，收益最大。

这不是巧合，第 4 节给出机制解释。

---

## 案例 1：Qwen3-30B-A3B —— 自写 MoE kernel（batch==1）

### 改了什么

`scripts/custom_moe_patch.py`，monkeypatch `fused_experts_impl`，仅在 M≤4 + bf16 + gated-silu + 非量化 + shape 匹配时接管，否则回退原实现。

sglang 原路径是为**大 M 吞吐**设计的：

1. `moe_align_block_size` 把 token 按 expert 排序/分组成对齐 block
2. `fused_moe_kernel` 做 w1 grouped GEMM → 单独 `silu_and_mul` → w2 grouped GEMM
3. 再做 topk 加权求和

排序+分组的开销在大 M 下被摊薄，但在 decode（M=1~4）下几乎是纯开销。我改了 4 处：

1. **完全跳过 `moe_align_block_size`** —— 不排序不分组，改为按 (token, expert) pair 并行，共 `P = M × topk` 个 program（b=1 只有 8 个 pair）
2. **`_w1_act`：融合 w1 GEMM + SwiGLU** —— 用 tiled `tl.dot` 同时算 gate 和 up，kernel 内直接做 `silu(gate)*up`，少一次 [P×I] 的写回+读入
3. **`_w2_sum`：融合 w2 GEMM + 路由加权 + 缩放 + 规约** —— 用 `atomic_add` 就地累加 topk 个专家贡献
4. **fp32 累加** —— 数值比 sglang 的 bf16 路径更准

### 结果

隔离层 **1.23×**，而且更准。端到端（n=15 交错重复 + Welch t）：

| batch | baseline (ms) | ours (ms) | 变化 | \|t\| | 判定 |
|---:|---|---|---:|---:|---|
| **1** | 4.267 ± 0.025 | 4.217 ± 0.016 | **+1.17%** | 6.51 | 真信号 |
| 2 | — | — | **−4.3%** | 3.2 | **真回归** |
| 4 | — | — | **−11.7%** | 9.9 | **真回归** |

b=1 的 +1.17% 是**真的**（delta 是噪声带 ±0.30% 的约 4 倍），但绝对值只有 ~0.05ms / 4.27ms。

### 为什么 b=1 赢、b≥2 输

sglang 的 expert 分组能让**同一 expert 的多个 token 复用一次权重加载**——Wg/Wu/W2 每 expert-tile 只读一次，摊到多个 token。我的 per-pair 方案对每个 (token, expert) 都重新加载权重，显存流量随 M 线性上升，加上 `atomic_add` 竞争，很快被反超。

一句话：**我用"省掉排序开销"换掉了"权重复用"，这笔交易只在 M=1 划算。**

### 这个案例真正的价值

它是个**负面结果**，但它坐实了一条判据：

> **隔离层 1.23× 完全不能推断端到端。** 单点端到端同样会误导——"M≤4 都赢"一扫 regime 就被证伪。

以及一条 decode 的根因诊断：b=1 decode 步里 MoE 41% + dense 32% + attn 16% = **89% 是 memory-bound 的权重/KV 流式读取**（光 lm_head 单 token 就要读 ~600MB 权重）。整步本质在**读权重**，不在算。所以任何只提升算力的 kernel 改动，最多动 89% 里很小一角。

**这直接解释了为什么案例 2 和 3 能赢**：它们省的是 **HBM 往返次数**，不是算力。

---

## 案例 2：LFM2.5-8B-A1B —— 补上框架没有的融合

### 背景：先审计，别猜

算子级审计（`lf_audit.py`）对比 LFM2.5 和 Qwen：

| | LFM2.5 | Qwen（对照） |
|---|---:|---:|
| 未融合 RMSNorm | **61** | 1 |
| 独立 residual add | **48** | 0 |
| gating mul | **36** | 0 |

Qwen 几乎没有空缺——它是被反复优化过的主线模型。LFM2.5 是新架构，融合基础设施还没跟上。

### 改了什么（7 个组件）

| 组件 | 手法 |
|---|---|
| `norm` | deferred residual —— 把 `x = x + res; x = norm(x)` 换成 `fused_add_rmsnorm(x, res)`，两个 kernel 合一，residual 就地更新 |
| `scale` | 归一化里的缩放合进上一个 kernel |
| `conv` | **手写 Triton**：ShortConv 的 gate + transpose 合成一个 kernel（原本是纯数据搬运） |
| `gate` / `idx` | gating 乘法与索引 |
| `qkrope` | QK-norm + RoPE 融合 |
| `moesum` | **手写 Triton**：MoE 归约 + norm |

### 结果（各 6 次重复，精确 Welch t）

| regime | 单独最好 | **7 个全开** | p |
|---|---:|---:|---|
| A 低批 decode | +4.60% | **+6.57%** | 4.6e-14 |
| B 并发 decode | +6.01% | **+6.21%** | 2.4e-08 |
| C 长 prefill | +5.81% | **+5.30%** | 1.2e-05 |

GSM8K 无质量回归。**这是本项目第一个同模型、正向、全 regime 显著的内核级端到端结果。**

### 关键发现：次可加性

把各组件单独收益加起来 vs 实测全开：

| regime | 单独之和 | 实测全开 | **兑现率** |
|---|---:|---:|---:|
| C 长 prefill | 5.86% | 5.30% | 0.90 |
| A 低批 decode | 9.37% | 6.57% | 0.70 |
| B 并发 decode | 12.80% | 6.21% | **0.49** |

**同类优化不相加，而且兑现率跟踪 regime 的饱和度。** B（并发 decode）最饱和，所以兑现率最低——GPU 已经忙不过来，省下来的时间填不进去。

这条对 agent 设计很重要：**不能把候选优化的预估收益线性叠加**，否则在饱和 regime 上会高估一倍。

---

## 案例 3：Gemma-3 —— 接上框架已有的 kernel ★

### 改了什么

一行 dispatch。`layers/layernorm.py` 里：

```python
class Gemma3RMSNorm(CustomOp):
    def forward_cuda(self, x):
        return self.forward_native(x)      # ← 整个类掉进 eager PyTorch
```

而**同一个文件里往上 100 行**的 `GemmaRMSNorm`（gemma/gemma2 用的）早就在用融合 kernel `gemma_rmsnorm`。两个类的 `forward_native` 参考实现**逐字节相同**，语义完全一致——所以这不是精度取舍，就是漏接。

eager 路径是 `pow → mean → add → rsqrt → mul` 加上 fp32 上下转换，约 **6 个 kernel、7 次 HBM 往返**；融合路径是 1 个。gemma-3-1b 每次 forward 有 **157 次** norm 调用。

### 结果

原始测量（完全不融合 → 完全融合，真实源码补丁，8 次重复）：

| regime | 提升 |
|---|---:|
| A 低批 decode | **2.128×** |
| B 并发 decode | **1.996×** |
| C 长 prefill | 1.521× |

### 但这个数字现在不能这么报了

准备提 PR 期间，上游合入了 **#32383**，已经把 2-D 路径和 residual 路径接上了。所以 2.13× 里有一部分**已经是别人的功劳**。

剩下的真实缺口有两个：

1. **高维输入仍走 eager** —— #32383 的守卫是 `if x.dim() == 2`，但 `q_norm`/`k_norm` 的输入是 `[tokens, heads, head_dim]`，3-D，直接掉回 eager。**157 次里有 52 次**
2. **weight dtype 无条件透传** —— `gemma_rmsnorm(bf16_x, fp32_weight)` **静默返回 NaN**，不抛异常

重新用「等效于当前 main」的基线 A/B，测出真正属于这次改动的增量：

| regime | baseline(= main) | patched | **增量** | p |
|---|---:|---:|---:|---|
| A 低批 decode | 1.300 req/s | 1.776 | **+36.6%** | 2.4e-14 |
| B 并发 decode | 33.648 req/s | 41.891 | **+24.5%** | 1.2e-06 |
| C 长 prefill | 23.385 req/s | 25.085 | +7.3% | 0.053 **不显著** |

**PR**：https://github.com/sgl-project/sglang/pull/32670（draft）

### 精度

以 fp64 为真值：eager 路径的最大相对误差**恒等于 bf16 的理论下界**，fused 路径 97.8–98.4% 的元素与 eager 逐位相同，其余在 BF16 舍入量级。不是 bit-identical（融合 kernel 在激活 dtype 下施加 weight），但这是 #32383 的 2-D 路径**已经接受的同一取舍**，非本次引入。

---

## 4. 三个案例合起来说明了什么

### 4.1 收益与工作量成反比，这有机制解释

decode 步 89% 是 memory-bound。所以：

- **案例 1 省的是算力**（跳过排序）→ 动的是那 11%，且代价是更多 HBM 流量 → 端到端 ≈0，b≥2 转负
- **案例 2、3 省的是 HBM 往返次数**（多个 kernel 合一，中间结果不落显存）→ 直接动那 89% → 端到端显著

**判据**：在 memory-bound 的 decode 上，只有**减少显存往返**的改动才可能兑现到端到端；提升算力的改动几乎必然被淹没。

### 4.2 越"没人管过"的地方，收益越大

| 模型 | 成熟度 | kernel 空缺占比 | 端到端可拿 |
|---|---|---:|---:|
| Qwen3 | 主线，反复优化 | ~0.2% | ≈0 |
| LFM2.5 | 新架构 | 11.3% | +6.6% |
| Gemma-3 | 成熟但非主线 | **46.3%** | **+36.6%** |

⚠️ **但"成熟度"这个假设被我自己的数据推翻了**：最新的 Qwen3-Next 最干净（0.64%），成熟的 Gemma-3 最差（46.3%），而 IBM granite 只有 0.30%。真正的预测因子更接近**模型家族有没有人在主线上持续维护**，不是模型新旧。

### 4.3 最高杠杆的模式：框架已有 kernel，但调用点没接上

案例 3 是这类。它的特征是：

- 上游**已经写好并测试过**融合 kernel
- 某个模型的调用点因为守卫条件、显式 `forward_native` 调用等原因没接上
- 修复是**一处 dispatch**，不需要写任何 kernel
- 风险极低——参考实现逐字节相同就能证明不是精度取舍

这条 signature 我做成了可执行扫描器（`.github/skills/fusion-gap-hunting/`），**已经命中 4 次**：Gemma-3、OLMo-2、以及另外两个未兑现的。

**第 4 个案例 OLMo-2**（`models/olmo2.py:190`，`else` 分支显式调 `forward_native`）：

| regime | 增益 | p |
|---|---:|---|
| A 低批 decode | +0.45% | 2.7e-04 |
| B 并发 decode | +0.70% | 0.143 n.s. |
| **C 长 prefill** | **+14.51%** | **3.3e-05** |

### 4.4 一个测量方法的教训（OLMo-2 查出来的）

OLMo-2 的审计报 7.71%，但 decode 端到端只有 +0.45%。没糊弄过去，装了分支计数器：

```
[olmo2_branch] {'fused_capture': 384, 'eager_else': 16}
```

**CUDA graph 捕获时走的本来就是融合分支**——而 `lf_audit.py` 默认加 `--disable-cuda-graph`。我把只在关图时才存在的开销当成了真实缺口。开图重跑审计确认：decode 里那三项（3.88+2.17+1.66 = **正好 7.71%**）全部消失，prefill 纹丝不动（7.93→7.91）。

据此做了可证伪预测——缺口只在 prefill，那 prefill-heavy regime 应有真实增益——24 次重复验证：**+14.51%，p=3.3e-05** ✅

**教训**：审计给的 `% kernel time` 是**上界**，而且当缺口被 capture-mode 守卫保护时会系统性高估。查过了，11 个受审模型里只有 olmo2 和 qwen3_next 属这类。

---

## 5. 数据出处

| 案例 | 报告 | 原始数据 |
|---|---|---|
| 1 · Qwen | `docs/2026-07-20/qwen_optimization_full_report.md` §1.5 | `docs/2026-07-20/noise_verification_custom_moe_b1.md` |
| 2 · LFM | `docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md` | `results/lfm_fusion/processed/fusion_ab_all7.csv` |
| 3 · Gemma-3 | `docs/2026-07-28/PR_DRAFT_gemma3_rmsnorm_v2.md` | `results/lfm_fusion/processed/fusion_ab_incremental.csv` |
| 4 · OLMo-2 | 本报告 §4.3–4.4 | `results/lfm_fusion/processed/fusion_ab_olmo2.csv`、`olmo2_audit_cudagraph_effect.csv` |
| 全案例技术细节 | `docs/2026-07-28/three_fusion_cases.md` | — |
| 方法论沉淀 | `.github/skills/fusion-gap-hunting/SKILL.md` | — |
