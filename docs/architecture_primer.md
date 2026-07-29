# 推理侧模型架构入门 —— 从 infra 视角

**给谁看**：做推理性能优化，但没系统学过模型架构的人。
**不讲什么**：反向传播、优化器、训练技巧。推理不需要。
**怎么用**：所有 shape 和数字都取自 `gemma-3-1b-it` 的真实 config，代码行号对应
`sglang` 仓库。每一节末尾标注了**我们自己在这一层做过什么优化**，可以直接对照实验记录。

---

## 0. 先建立一个总印象

一个大模型推理，就是**同一段代码循环 N 层**。gemma-3-1b 是 26 层。

```
输入 token id
    │
    ▼ embedding                把 id 变成向量
    │
    ▼ ┌──────────────────┐
      │  Decoder Layer     │  ← 循环 26 次，每层结构完全一样
      │  1. 归一化          │
      │  2. 注意力          │
      │  3. 归一化          │
      │  4. 前馈网络(MLP)   │
      └──────────────────┘
    │
    ▼ 最后归一化
    ▼ lm_head                 投影到词表，得到每个词的分数
    ▼ 采样                    选出下一个 token
```

**所以只要读懂一层，就读懂了整个模型。** 那一层在
`python/sglang/srt/models/gemma3_causal.py:373`，26 行。

---

## 1. gemma-3-1b 的真实尺寸

从 `config.json` 读出来的（后面所有计算都基于这些）：

| 参数 | 值 | 含义 |
|---|---|---|
| `hidden_size` | **1152** | 每个 token 的向量长度，贯穿全模型的"主干宽度" |
| `num_hidden_layers` | **26** | 上面那个 Decoder Layer 循环几次 |
| `num_attention_heads` | **4** | Q 切成几个头 |
| `num_key_value_heads` | **1** | K/V 切成几个头（比 Q 少 = GQA，见 §3.4）|
| `head_dim` | **256** | 每个头的向量长度 |
| `intermediate_size` | **6912** | MLP 中间层宽度（是 hidden 的 6 倍）|
| `vocab_size` | **262144** | 词表大小 |
| `sliding_window` | **512** | 部分层只看最近 512 个 token |

---

## 2. 一层里的四件事（对着代码读）

`gemma3_causal.py:373` 的 `Gemma3DecoderLayer.forward`：

```python
# 1. 归一化（顺带把上一层的 residual 加进来）
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

# 2. 注意力
hidden_states = self.self_attn(positions=..., hidden_states=hidden_states, ...)
hidden_states = self.post_attention_layernorm(hidden_states)

# 3. 归一化
hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)

# 4. 前馈网络
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states)
```

### 2.1 归一化（Normalization）

**做什么**：把每个 token 的向量重新缩放到统一尺度，防止数值飘掉。

RMSNorm 公式，对**每一行**独立做：

```
1. 平方均值   m   = (x₁² + ... + x_d²) / d
2. 开根号     rms = √(m + ε)          ← 这一行的"平均幅度"
3. 除以它     x / rms                 ← 现在 RMS = 1
4. 乘权重     × (1 + w)               ← w 是学出来的
```

★ **关键性质：只沿最后一维规约，行与行之间零交互。**
这条性质是我们那个 PR 成立的全部依据（见 §6.1）。

代码：`python/sglang/srt/layers/layernorm.py`，`Gemma3RMSNorm` 在 1051 行。

### 2.2 残差（Residual）

**做什么**：`输出 = 这一层算的 + 这一层的输入`。

为什么要有：26 层堆起来，信号会越传越弱。残差给了一条"高速公路"，让信息可以直接跨层传递。

★ **infra 关注点**：`x = x + residual` 和紧跟其后的 `norm(x)` 可以**融合成一个 kernel**
（`fused_add_rmsnorm`），省掉一次全激活的读写。这正是我们在 LFM2.5 上做的 `norm` 组件。

### 2.3 注意力（Attention）—— 见 §3

### 2.4 前馈网络（MLP）

**做什么**：每个 token 独立地过一个两层网络，做非线性变换。

```
[tokens, 1152] --gate_up_proj--> [tokens, 6912×2] --激活--> [tokens, 6912]
                                                   --down_proj--> [tokens, 1152]
```

先升维到 6912（6 倍），做非线性，再降回 1152。

**参数量最大的部分**：每层 `1152×6912×3 ≈ 24M` 参数，26 层就是 620M——
占 1B 模型的一大半。

★ **infra 关注点**：MoE（混合专家）就是把这个 MLP 换成"N 个专家，每个 token 只走 top-k 个"。
LFM2.5 是 32 专家 top-4，我们那个 MoE config PR 调的就是这里的 kernel。

---

## 3. 注意力：最复杂也最值得懂的部分

代码：`gemma3_causal.py:316` 的 `Gemma3Attention.forward`。

### 3.1 Q / K / V 是什么

每个 token 被投影成三个向量：

| | 全称 | 作用 | 类比 |
|---|---|---|---|
| **Q** | Query | 我想找什么 | 搜索框输入的关键词 |
| **K** | Key | 我是什么 | 每个网页的标题 |
| **V** | Value | 我的内容 | 网页正文 |

计算：

```
1. 相关度   scores = q · kᵀ / √d      ← Q 和每个 K 做内积
2. 归一化   p = softmax(scores)        ← 变成概率，加起来 = 1
3. 加权和   out = Σ p·v                ← 按概率把 V 混起来
```

一句话：**每个 token 用自己的 Q 去和所有 token 的 K 匹配，按匹配度把它们的 V 混合起来。**

### 3.2 多头（Multi-Head）

不是只做一次，而是把向量**切成几段各做各的**，最后拼起来。

```
[tokens, 1152] --qkv_proj--> q:[tokens, 1024]
                             --拆成4个头--> [tokens, 4, 256]
                                                    ↑    ↑
                                                  头数  head_dim
```

为什么：不同的头可以关注不同的东西（一个看语法、一个看指代……），比单头表达力强。

★ 这个**多出来的 head 维度**，就是我们那个 PR 的起因（见 §6.1）。

### 3.3 QK-Norm（新东西，Gemma-3/Qwen3/OLMo-2 才有）

在算内积**之前**，先把 Q 和 K 各自归一化：

```python
q = self.q_norm(q)   # gemma3_causal.py:259
k = self.k_norm(k)
```

**为什么需要**：softmax 对输入尺度极其敏感。实测（同样的相对关系，只放大幅度）：

| q/k 幅度 | 内积范围 | softmax 最大值 | 熵 |
|---|---|---|---|
| 正常 | [−2.6, 0.8] | 0.298 | 1.774 |
| 放大 3× | [−19.4, 13.8] | 0.959 | 0.176 |
| 放大 8× | [−173.7, 117.7] | **1.0000** | **0.000** |

幅度一大，softmax 就**饱和**——概率 1.0 全给一个位置，熵归零，梯度消失，还可能数值溢出。
QK-Norm 把 q、k 钉在 RMS=1，内积尺度就稳了。

早期模型靠 `1/√d` 缩放（config 里的 `query_pre_attn_scalar: 256`）就够；
模型越大越深不够了，才加这个。

### 3.4 GQA —— 直接决定你的显存

注意 gemma-3-1b：`num_attention_heads=4` 但 `num_key_value_heads=1`。

**Q 有 4 个头，K/V 只有 1 个头**，4 个 Q 头共享同一份 K/V。这叫 GQA（Grouped-Query Attention）。

为什么这么设计——算笔账：

```
KV cache 每 token 每层 = 2(K和V) × 1头 × 256 × 2字节(bf16) = 1024 B
全部 26 层                                                 = 26.0 KB / token

如果不用 GQA（kv_heads=4）                                  = 104.0 KB / token
                                                          -> GQA 省了 4×
```

★ **infra 关注点**：KV cache 是**显存的主要消耗者**，直接决定你能开多大 batch。
你调的 `--mem-fraction-static` 分配的就是它。

### 3.5 KV cache —— 理解 decode 的关键

生成第 100 个 token 时，前 99 个 token 的 K/V **不会变**。所以缓存起来，
每步只算新 token 的 K/V。

**这带来一个根本性的后果**：

| | prefill（处理 prompt） | decode（逐个生成） |
|---|---|---|
| 一次处理多少 token | 几百~几千 | **1** |
| 矩阵乘形状 | 大 × 大 | **1 × 大** |
| 瓶颈 | **算力**（compute-bound） | **显存带宽**（memory-bound） |
| 为什么 | 真在做大矩阵乘 | 为算 1 个 token 要把**全部权重读一遍** |

★★ **这是整个推理优化最重要的一条。** 两个阶段瓶颈完全不同，
**优化手段也完全相反**：

- decode 慢是因为**在搬数据**，不是在算 → 优化方向是**减少显存往返**（融合 kernel）
- prefill 慢是因为**在算** → 优化方向是**让矩阵乘更高效**（调 tile 形状）

我们所有实验都按 regime 分（A 低批 decode / B 并发 decode / C 长 prefill），原因就在这。

### 3.6 RoPE（旋转位置编码）

注意力本身**不知道词的顺序**（打乱输入结果一样）。RoPE 通过把 Q/K 向量按位置"旋转"
一个角度，把位置信息编码进去。

★ **infra 关注点**：长上下文的问题多半出在这。另外它紧跟在 QK-Norm 之后，
**两者可以融合**——我们在 LFM2.5 上做的 `qkrope` 组件就是这个，B regime 单独 +5.42%。

---

## 4. 一次 forward 的完整数据流（带真实 shape）

以 7 个 token 的 decode 为例：

```
input_ids                                [7]
    │
    ▼ embedding
hidden_states                            [7, 1152]
    │
    ├──────────── 循环 26 层 ────────────┐
    │                                     │
    ▼ input_layernorm                     │  [7, 1152]
    ▼ qkv_proj                            │  [7, 1024+256+256]
    ▼ split                               │  q[7,1024] k[7,256] v[7,256]
    ▼ unflatten + unsqueeze               │  q[1,7,4,256]  k[1,7,1,256]
    ▼ q_norm / k_norm   ★我们改的地方      │  同上
    ▼ rotary_emb (RoPE)                   │  同上
    ▼ attn (读写 KV cache)                 │  [7, 1024]
    ▼ o_proj                              │  [7, 1152]
    ▼ post_attention_layernorm            │  [7, 1152]
    ▼ + residual                          │  [7, 1152]
    ▼ pre_feedforward_layernorm           │  [7, 1152]
    ▼ mlp: gate_up_proj                   │  [7, 13824]
    ▼      激活                            │  [7, 6912]
    ▼      down_proj                      │  [7, 1152]
    ▼ post_feedforward_layernorm          │  [7, 1152]
    ▼ + residual                          │  [7, 1152]
    └─────────────────────────────────────┘
    │
    ▼ 最终 norm                            [7, 1152]
    ▼ lm_head                              [7, 262144]   ← 词表大小，这一步很贵
    ▼ 采样                                 [7]
```

**注意 lm_head**：`1152 × 262144 ≈ 302M` 参数。decode 每步为了 1 个 token
就要把这 302M（bf16 约 600MB）全读一遍——**这就是 decode 是 memory-bound 的直观体现**。

---

## 5. infra 必懂的五个概念（复习）

| 概念 | 一句话 | 为什么 infra 关心 |
|---|---|---|
| **KV cache** | 缓存历史 token 的 K/V | 显存主要消耗者，决定 batch 上限 |
| **prefill vs decode** | 一个算得多，一个搬得多 | **瓶颈相反，优化手段相反** |
| **GQA/MQA** | K/V 头数少于 Q | KV cache 直接除以这个倍数 |
| **MoE** | MLP 换成 N 个专家选 top-k | 参数多但计算少；kernel 调优的主战场 |
| **RoPE** | 用旋转编码位置 | 长上下文问题的源头 |

---

## 6. 对照：我们做过的优化落在哪一层

### 6.1 Gemma-3 的 q_norm/k_norm（PR #32670）

**位置**：§3.3，注意力里的 QK-Norm。

**问题**：融合 kernel `gemma_rmsnorm` 只接受 2-D 输入 `[N, D]`，
而 QK-Norm 的输入带 head 维度，是 4-D `[1, 7, 4, 256]`。
上游的守卫写的是 `if x.dim() == 2`，于是这条路掉回 eager PyTorch（约 6 个 kernel）。

**修复**：压平 → 融合 kernel → 还原。

```python
flat = x.reshape(-1, x.shape[-1]).contiguous()
return gemma_rmsnorm(flat, self.weight.data, self.eps).view_as(x)
```

**为什么无损**：RMSNorm 只沿最后一维规约（§2.1），行间零交互，
所以 `[1,7,4,256]` 和 `[28,256]` 对它是同一件事。实测逐位相同，差值 0.0。

**为什么这是"补漏"不是"发明"**：同一个文件里的通用 `RMSNorm` 类
（`layernorm.py:417`）早就有这段逻辑，注释直说
`sgl_kernel rmsnorm requires 2D input; reshape higher-rank tensors`。
`Gemma3RMSNorm` 是独立实现，没继承到。

**收益**：低批 decode +36.6%，并发 decode +24.5%。

### 6.2 LFM2.5 的 MoE config（PR #32687）

**位置**：§2.4，MLP 换成 MoE。

**问题**：这个 shape（E=32, N=1792）在 H200 上没有调优配置文件，
运行时走一段只有两档的启发式。

**修复**：补上配置文件。零代码改动。

**收益**：长 prefill +23.3%。decode 不变——因为小 M 桶写的就是默认值
（§3.5 解释了为什么 decode 的 M 很小）。

### 6.3 LFM2.5 的七个融合组件

| 组件 | 对应本文哪一节 |
|---|---|
| `norm` (deferred residual) | §2.2 残差 + §2.1 归一化 |
| `qkrope` (QK-norm + RoPE 融合) | §3.3 + §3.6 |
| `moesum` (MoE 归约 + norm) | §2.4 |
| `conv` (ShortConv gate+transpose) | LFM 特有的架构，不在标准 Transformer 里 |

---

## 7. 怎么继续学

**第一优先：读代码，不是读教程。**

```bash
# 一个完整的模型实现，160 行左右
python/sglang/srt/models/gemma3_causal.py

# 从这里开始读
#   :373  Gemma3DecoderLayer.forward   ← 先读这个，26 行
#   :316  Gemma3Attention.forward      ← 再读这个
#   :111  Gemma3MLP.forward            ← 最后这个
```

**第二：把代码和 profiler trace 对上。**

```bash
python scripts/lfm_fusion/lf_audit.py --model gemma3 --regime A_low_batch_decode --gpu 0
```

它会告诉你每类 kernel 占多少时间。**建立"这行代码 → 这几个 kernel"的映射**，
是 infra 直觉的来源。

**第三：换个模型再读一遍。** 读完 gemma3 再去读 `qwen3_moe.py`，
你会发现 90% 是一样的，剩下 10% 就是这个模型的特点。看三个模型之后基本就通了。

---

## 8. 术语速查

| 缩写 | 全称 | 一句话 |
|---|---|---|
| **RMSNorm** | Root Mean Square Norm | 除以均方根，比 LayerNorm 省一步减均值 |
| **QK-Norm** | — | 算注意力前先归一化 Q 和 K，防 softmax 饱和 |
| **GQA** | Grouped-Query Attention | 多个 Q 头共享一份 K/V，省 KV cache |
| **MQA** | Multi-Query Attention | GQA 的极端情况，K/V 只有 1 个头 |
| **MoE** | Mixture of Experts | MLP 换成 N 个专家，每 token 只走 top-k 个 |
| **RoPE** | Rotary Position Embedding | 用旋转把位置编码进 Q/K |
| **KV cache** | — | 缓存历史 token 的 K/V，避免重算 |
| **prefill** | — | 处理输入 prompt 的阶段，compute-bound |
| **decode** | — | 逐个生成 token 的阶段，memory-bound |
| **SwiGLU** | — | MLP 里常用的激活，`silu(gate) * up` |
| **top-k** | — | MoE 里每个 token 选几个专家 |
| **head_dim** | — | 每个注意力头的向量长度 |

---

**所有数字来源**：`/data/hf/models/gemma-3-1b-it/config.json`，
代码行号对应 `sgl-project/sglang` @ `main`（2026-07-29）。
softmax 饱和那组数据是本文档撰写时实测的。
