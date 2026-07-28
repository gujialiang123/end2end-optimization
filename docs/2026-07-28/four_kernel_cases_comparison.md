# 四个 kernel 级案例的对照报告：调 config vs 重写 vs 补融合 vs 接线

**日期**：2026-07-28 · **硬件**：H200 · bf16 · **所有数字均为本项目实测**，非上游自称
**范围**：kernel 层的干预。serving 参数 autotuning 不在内（那条线是负面结果，见 §6）。

---

## 0. 为什么把这四个放在一起

这四件事都被我们叫过"kernel 优化"，但它们**动的东西完全不同**，结论也几乎相反：

| | 案例 0 · LFM2.5 | 案例 1 · Qwen3-30B | 案例 2 · LFM2.5 | 案例 3 · Gemma-3 |
|---|---|---|---|---|
| **动了什么** | kernel 的**配置** | **重写** kernel 逻辑 | **补上**缺的融合 | **接上**已有 kernel |
| kernel 源码 | **一行没动** | ~400 行新 Triton | 手写 + 已有原语 | **一行没写** |
| **端到端** | **+22.3%**（长 prefill） | **+1.17%**（b≥2 回归） | **+5.3~6.6%** | **+24.5~36.6%** |
| 零回归？ | ✅ 8/8 干净 | ❌ b≥2 真回归 | ✅ | ✅ |
| 可提 PR？ | ❌ 是数据文件不是代码 | ⚠️ 已开 #31836，前景差 | 🔶 只该提 `norm` | ✅ #32670 |

**共同点只有一个**：收益全都来自**"这个东西没人管过"**，而不是"我们比上游聪明"。

---

## 案例 0：LFM2.5 —— 这个 shape 从来没人调过 config

### 机制

SGLang 的 MoE config 是一张按 `E=专家数, N=中间维度, device_name=GPU` 命名的查找表。
LFM2.5-8B-A1B 需要的是 `E=32,N=1792,device_name=NVIDIA_H200.json` ——
**这个文件在仓库里不存在**。找不到就走一段只有两档的启发式：

```python
config = {BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8}
if M <= E:                       # tokens <= 专家数
    config = {BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 32, BLOCK_SIZE_K: 64, ...}
```

**整个 M 范围只有两档**，`num_warps`/`num_stages` 直接用 Triton 默认。
上游自己在日志里就打印 "Performance might be sub-optimal!" 并给出调优工具链接。

### 结果

| regime | global-best 单一配置 | regime-aware(naive) | **guarded（最终）** |
|---|---:|---:|---:|
| 低批 decode | 0.923× | 0.745× | **0.998×**（中性） |
| 并发 decode | 1.004× | 1.060× | **1.014×** |
| **长 prefill** | 0.796× | 1.170× | **1.223×** |

最终 arm 做了两批重复，都指向同一结论：

| 批次 | 长 prefill（patched vs baseline） | 判定 |
|---|---|---|
| §0，各 6 次 | 15.06 ± 0.08 vs 12.32 ± 0.03 req/s | 6/6 不重叠 |
| §7，各 8 次（**可部署结论**） | 14.63–15.19 vs 11.91–12.40 req/s | **8/8 分布完全不重叠**，1.221× |

**+22.3%，任何 regime 零回归。**

### 三个必须尊重的机制（这才是最可迁移的部分）

1. **单一 global 配置是有害的**（0.80–0.92×）——不能"调一个最好的配置全局用"
2. **naive per-regime 特化在 prefill 赢大、但 decode 亏 25%** ——必须 **guarded**：
   只在 oracle 证明有空间的地方特化，其余保持运行时默认
3. 三个纠错，每个都曾让结论完全错：
   - 我们在调**一个 server 从不执行的 kernel 变体**（expert bias）
   - **CUDA graph capture 会把 config 烘焙进去**，decode 是回放
   - **`M` 是 token 数，不是 `tokens × top_k`** ——profile key 错了 top_k 倍，
     真实空间被藏在错位的桶里。**只有 live trace 能暴露这个。**

### 诚实边界

baseline 是**很弱的启发式默认值**，不是认真调过的配置。所以正确表述是：

> **不是"我们把 kernel 优化快了 1.6×"，而是"这个 model/GPU 组合从来没人调过，补上调优值 1.6×"。**

Qwen 是有用的对照：它**有**真实 tuned config（只是 Triton 版本差一个小版本），
所以它的空间只有 **0.96–1.23×**。

> **有人调过的 shape 空间小；没人调过的 shape 空间大。这个对比本身就是结果。**

### 还有一条框架修正

原本叫 "regime-aware"，但实测下来 kernel 调优是 **shape 相关，不是 regime 相关**：

```
regime  →  M 分布  →  最优 kernel config
```

而运行时**本来就按 M 分派**。专门做的对照实验（§6b）测了"固定 M 时，
regime 的另一个属性（专家路由分布）会不会改变最优解"——**不会（可靠地不会）**。
所以站得住的说法是关于 **shape 特化**，regime 只在它决定 shape 的意义上成立。

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

### 七个组件逐个说明

| # | 组件 | 具体改了什么 | 性质 |
|---|---|---|---|
| 1 | **`norm`** | **deferred residual**：把 `x = x + res; x = norm(x)` 改成 `normed, res = norm(x, res)`。`RMSNorm(x, residual)` 的两参数形式执行 `residual += x` 并就地归一化，**独立的 add kernel 直接消失** | **用已有原语** |
| 2 | **`scale`** | 归一化里的缩放合进上一个 kernel，省一次全激活读写 | 用已有原语 |
| 3 | **`conv`** | ShortConv 的 **gate + transpose 合成一个 Triton kernel**。原本 `B*x` 和 `C*conv_out` 两次 gating 乘法加转置全是**纯数据搬运** | **手写 Triton** |
| 4 | **`gate`** | ShortConv 的 gating 乘法 | 手写 |
| 5 | **`idx`** | 缓存 int32 索引，避免每步重建 | 调用点 |
| 6 | **`qkrope`** | **QK-norm + RoPE 融合**，只动 24 层里的 3 个 attention 层 | **纯调用点改动** |
| 7 | **`moesum`** | **MoE 归约 + norm 合成一个 Triton kernel**（`no_combine` 出 partials，再一次性归约+归一化） | **手写 Triton** |

### 结果（各 6 次重复，精确 Welch t）

| regime | `qkrope` | `gate+idx` | `norm+scale+conv` | `moesum` | 六项 | **七项全开** |
|---|---:|---:|---:|---:|---:|---:|
| A 低批 decode | +0.93% | −0.00% (n.s.) | +3.89% | +4.55% | +4.60% | **+6.57%** |
| B 并发 decode | **+5.42%** | +0.65% (n.s.) | +3.65% | +3.08% | +6.01% | **+6.21%** |
| C 长 prefill | +1.99% | +0.40% (n.s.) | +3.47% | — | +5.81% | **+5.30%** |

七项全开 p = **4.6e-14 / 2.4e-08 / 1.2e-05**，GSM8K 无质量回归。
**这是本项目第一个同模型、正向、全 regime 显著的内核级端到端结果。**

`gate+idx` 三个 regime **全不显著**——诚实负面：机制在 kernel 级真实可测（1~2%），但没兑现到端到端。

### 四个组件的收益形状完全不同

- **`norm+scale`** 省的是**每 forward 固定数量**的 kernel 和全激活读写，与该 forward 做多少计算无关 → decode 每 forward 才 ~2 ms，占比大（+4.2%）；长 prefill ~157 ms，被稀释（+1.6%）
- **`conv`** 省的是**随 token 数增长**的流量，且要 T≥2048 才划算 → decode 够不到（精确中性，p=0.22/0.95），长 prefill 跑在 T=4000–16000（+2.33%）
- **`qkrope`** 省的是 3 个注意力层里的工作 → 并发 decode 最受益（+5.42%）
- **`moesum`** 省的是 launch + HBM 往返，**小 T 最赚** → 低批 decode 最受益（+4.55%）

**四种不同形状的收益。只测一个 regime 一个都看不全。**

### 关键发现：同类优化强烈次可加

| regime | 各项之和 | 一起测 | **兑现率** |
|---|---:|---:|---:|
| C 长 prefill | 5.86% | 5.30% | 0.90 |
| A 低批 decode | 9.37% | 6.57% | 0.70 |
| B 并发 decode | 12.80% | 6.21% | **0.49** |

并发 decode 上：`qkrope` 单独 +5.42%，再加单独值 +3.65% 的 `norm+scale+conv` 只多买到 **0.12 点**；再加单独值 +3.08% 的 `moesum` 又只多买到 **0.19 点**。三者都在消除**同一份"固定每-forward 开销"的余量**。

**兑现率精确跟踪 regime 饱和度**（0.90 / 0.70 / 0.49）。这与案例 0 的 waterfall 非叠加（serving 1.78× + kernel 1.22× → **1.70×** 而非 2.17×）是同一现象。两个独立研究都撞上，可以固化成规则：

> **消除同一"种类"成本的优化不会相加。报告各项分别测量之和会高估整个 stack，且系统越饱和高估越严重。**

**实践含义：最便宜的组件反而最有价值。** `qkrope` 是纯调用点改动，单独就拿下并发 decode 的大部分空间。

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

## 4. 四个案例合起来说明了什么

### 4.1 收益与"写了多少 kernel 代码"成反比

| 案例 | 写了多少 kernel 代码 | 端到端 |
|---|---|---:|
| 0 · config 调优 | **0 行**（只是数据文件） | **+22.3%** |
| 3 · 接线 | **0 行**（用上游的） | **+24.5~36.6%** |
| 2 · 补融合 | 部分手写 | +5.3~6.6% |
| 1 · 重写 kernel | ~400 行 | **+1.17%**，且 b≥2 回归 |

这有机制解释。decode 步 89% 是 memory-bound：

- **案例 1 省的是算力**（跳过排序）→ 动的是那 11%，且代价是更多 HBM 流量 → 端到端 ≈0，b≥2 转负
- **案例 2、3 省的是 HBM 往返次数**（多 kernel 合一，中间结果不落显存）→ 直接动那 89%
- **案例 0 连 kernel 都不换**，只是让已有 kernel 用对 tile 形状 → prefill（compute-bound）上收益最大

**判据**：在 memory-bound 的 decode 上，只有**减少显存往返**的改动才可能兑现；提升算力的改动几乎必然被淹没。而在 compute-bound 的 prefill 上，**tile 形状选对**比什么都重要。

### 4.2 四个案例是同一句话的四个版本

> **收益来自"这块地没人管过"，不是"我们比上游聪明"。**

- 案例 0：这个 **shape** 没人调过 config → +22.3%；Qwen 调过 → 只剩 0.96–1.23×
- 案例 2：这个 **模型文件** 没人补过融合 → 61 个未融合 norm；Qwen 只有 1 个
- 案例 3：这个 **调用点** 没人接上 → 46.32% 的 kernel 时间；修完 +36.6%
- 案例 1：这块地**有人精心管过**（sglang 的 MoE 大 M 路径） → 我重写只拿到 +1.17% 且 b≥2 回归

**这是给 agent 最有用的先验：先找无人区，不要正面挑战被优化过的热路径。**

### 4.3 跨架构证据

| 模型 | 家族 | kernel 空缺占比 |
|---|---|---:|
| Gemma-3-1B | Google | **46.32%** |
| OLMo-2-1B | AllenAI | **27.74%** |
| EXAONE-4.0 | LG | 15.66% |
| Phi-4-mini | Microsoft | 13.87% |
| LFM2.5-8B | Liquid | 11.31% |
| OLMoE-1B-7B | AllenAI | 4.70% |
| Qwen3-Coder-Next | Qwen | 0.64% |
| **Granite-3.1** | **IBM** | **0.30%** |
| Qwen3-30B-A3B | Qwen | 0.23% |
| Qwen3-32B | Qwen | 0.05% |

⚠️ **两条修正**（都写进了 `cross_architecture_audit.md` §7.1/§7.2）：
- "非 Qwen 就有空缺"**不成立**——IBM granite 只有 0.30%；同属 AllenAI 的 OLMoE(4.70%) 和 OLMo-2(27.74%) 差 6 倍。**关注度是按模型文件计的，不是按家族计的**
- 审计的 `% kernel time` 是**上界**，且对被 CUDA-graph capture 守卫保护的缺口会**系统性高估**（见 §5）

### 4.4 最高杠杆的模式：框架已有 kernel，但调用点没接上

案例 3 是这类。特征：上游**已经写好并测试过**融合 kernel，某个模型的调用点因守卫条件或显式 `forward_native` 调用没接上，修复是**一处 dispatch**，风险极低（参考实现逐字节相同就能证明不是精度取舍）。

这条 signature 做成了可执行扫描器（`.github/skills/fusion-gap-hunting/`），**已命中 4 次**。

**第 4 个命中 OLMo-2**（`models/olmo2.py:190`，`else` 分支显式调 `forward_native`）：

| regime | 增益 | p |
|---|---:|---|
| A 低批 decode | +0.45% | 2.7e-04 |
| B 并发 decode | +0.70% | 0.143 n.s. |
| **C 长 prefill** | **+14.51%** | **3.3e-05** |

---

## 5. 一个测量方法的教训（OLMo-2 查出来的）

OLMo-2 的审计报 7.71%，但 decode 端到端只有 +0.45%。装了分支计数器：

```
[olmo2_branch] {'fused_capture': 384, 'eager_else': 16}
```

`alt_stream` 在 CUDA 上恒非 None ——**CUDA graph 捕获时走的本来就是融合分支**，稳态 decode 回放的图里根本没有 eager kernel。而 `lf_audit.py` 默认加 `--disable-cuda-graph`。**我把只在关图时才存在的开销当成了真实缺口。**

开图重跑审计确认：decode 里那三项（3.88+2.17+1.66 = **正好 7.71%**）全部消失，prefill 纹丝不动（7.93→7.91）。

据此做可证伪预测——缺口只在 prefill——24 次重复验证：**+14.51%，p=3.3e-05** ✅

> 注意这和**案例 0 的第 2 条纠错是同一个根因**：CUDA graph 会把决策烘焙在 capture 时。一个影响 config 生效方式，一个影响审计归因。**凡是 decode 路径的结论，都要先问一句"CUDA graph 开着吗"。**

---

## 6. 不在本报告内：serving 参数 autotuning（负面结果）

LFM2.5 还有一条 **serving flag** autotuning 的线，容易和案例 0 混淆，但它**不是 kernel 层**：

- **2026-06-30 条件化搜索**：Optuna 25 trial 找到的 "best" = 22.32 req/s，**比基线 23.74 低 6%**
- 根因：TPE 在早期 7 个 trial 里把 `triton MoE` 和 `cap=8`、`disable-cuda-graph` 这些**糟糕的 batching 选项**绑在一起，据此判定整个 `triton` 子空间"差"，后续 18 个 trial 再没试过 `triton + 好 batching`
- **结论：cookbook 默认就是最优。** 这条线没东西可提 PR

**区分**：案例 0 调的是 **Triton kernel 的 tile 形状**（`BLOCK_SIZE_M/N/K` 等），这条调的是 **sglang server 启动 flag**。前者正面，后者负面。

---

## 7. 数据出处

| 案例 | 报告 | 原始数据 |
|---|---|---|
| 0 · LFM config | `docs/2026-07-27/regime_kernel_results.md` | `results/regime_kernel/` |
| 1 · Qwen 重写 | `docs/2026-07-20/qwen_optimization_full_report.md` §1.5 | `docs/2026-07-20/noise_verification_custom_moe_b1.md` |
| 2 · LFM 融合 | `docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md` | `results/lfm_fusion/processed/fusion_ab_all7.csv` |
| 3 · Gemma-3 | `docs/2026-07-28/PR_DRAFT_gemma3_rmsnorm_v2.md` | `results/lfm_fusion/processed/fusion_ab_incremental.csv` |
| 4 · OLMo-2 | 本报告 §4.4–5 | `results/lfm_fusion/processed/fusion_ab_olmo2.csv` |
| 跨架构审计 | `docs/2026-07-28/cross_architecture_audit.md` | `results/lfm_fusion/processed/cross_architecture_audit_summary.csv` |
| 方法论沉淀 | `.github/skills/fusion-gap-hunting/SKILL.md` | — |


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
