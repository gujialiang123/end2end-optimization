# 三个 kernel fusion 案例 —— 具体改了什么、为什么被漏掉、拿到多少

**日期**：2026-07-27 ~ 28 · **硬件**：1× NVIDIA H200 · BF16
**软件**：sglang 0.5.12.post1 @ `17f7a1da1` · torch 2.9.1+cu128 · Triton 3.5.1

这份文档只讲**三个同型案例**的技术细节。整体研究背景见
`docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md`，跨架构审计见
`docs/2026-07-28/cross_architecture_audit.md`，方法论沉淀见
`.github/skills/fusion-gap-hunting/SKILL.md`。

---

## 0. 为什么把这三个放在一起

它们**不是三个不同的优化**，而是**同一个模式的三次出现**：

> **融合 kernel 已经写好、测好、编译进二进制里了。某个模型的调用点没有调用它。**

三次都不需要发明任何新东西——只需要发现"这里本该调用那个"。

| # | 模型 | 已存在的原语 | 谁在用 | 谁漏了 | 端到端收益 |
|---|---|---|---|---|---|
| 1 | LFM2.5 | `fused_add_rmsnorm` | llama / qwen2 / 几乎所有模型 | `Lfm2MoeDecoderLayer` | +2.35%（decode） |
| 2 | LFM2.5 | `fused_qk_norm_rope` | Qwen3-MoE | `Lfm2MoeAttention` | **+5.42%**（并发 decode） |
| 3 | **Gemma-3** | `gemma_rmsnorm` | **同文件的 `GemmaRMSNorm`** | `Gemma3RMSNorm.forward_cuda` | **+112.8%** |

第三个最极端：**原语和漏用它的类在同一个文件里，相隔约 100 行**。

---

## 案例 1：LFM2.5 —— residual 加法从未被融合

### 背景：deferred residual 是什么

Transformer 的残差连接朴素写法是"归一化 → 算子 → 把原值加回来"，其中**加法是一个独立的 elementwise kernel**。

sglang 提供了 `fused_add_rmsnorm(x, residual, w, eps)`，它在**一个 kernel 内**完成 `residual += x` 和 `normalize(residual)`。用法是把残差当作"欠账"传给下一层，由下一层的 norm kernel 顺手结清——这叫 **deferred residual**，`models/llama.py:304-316` 就是标准写法。

### 问题代码

`sglang/srt/models/lfm2_moe.py:433-456`：

```python
def forward(self, layer_id, positions, hidden_states, residual, forward_batch, **kwargs):
    residual = hidden_states                    # ← 传进来的 residual 参数被直接覆盖
    normed = self.operator_norm(hidden_states)  # ← 没传 residual → 走非融合分支

    if self.is_attention_layer:
        hidden_states = self.self_attn(positions, normed, forward_batch)
    else:
        hidden_states = self.conv(normed, forward_batch)

    hidden_states = hidden_states + residual    # ← 单独一个 elementwise kernel
    hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
                    #                ↑ 又一个单独的 kernel
    return hidden_states, residual
```

**三个观察**：
1. 函数签名**收了 `residual` 参数**，第一行就把它覆盖——传进来的值从没被用过
2. `RMSNorm.forward_cuda(x, residual)` 本来就会走 `fused_add_rmsnorm`（`layers/layernorm.py:139-147`）
3. `Lfm2MoeModel.forward` **本来就在层间传递 residual**

→ **接线全都在，只是这一层没接上。**

### 修复

```python
if residual is None:                        # 只有第一层
    residual = hidden_states
    normed = self.operator_norm(hidden_states)
else:                                        # 其余层：走融合分支
    normed, residual = self.operator_norm(hidden_states, residual)

hidden_states = self.conv(normed, forward_batch) if not self.is_attention_layer \
                else self.self_attn(positions, normed, forward_batch)

hidden_states, residual = self.ffn_norm(hidden_states, residual)   # 融合
hidden_states = self.feed_forward(hidden_states)
return hidden_states, residual
```

### 数学等价性

写 `x` 为进入本层的激活：

```
原版：  a = op(rms(x));  h1 = a + x;  out = h1 + ffn(rms(h1))
新版：  rms(x, r) → r := x,      n := rms(x)          ← 加法在 norm kernel 内部
        a = op(n)
        rms(a, r) → r := a + x = h1,  n2 := rms(h1)
        返回 (ffn(n2), h1)
```

下一层拿到 `ffn(n2) + h1`，与原版的 `out` 是同一个值。

### 收益

**每层省 2 个 kernel × 24 层 = 48 个**。端到端 decode **+2.35%**。

代数等价但**非 bit-exact**（累加顺序不同）：原语单测 residual 差 **0.0**、归一化输出差约 **2 个 bf16 ulp**，且融合版**更准**（加法保持更高精度）。

---

## 案例 2：LFM2.5 —— QK-norm + RoPE 没有融合

### 问题代码

`lfm2_moe.py:236-263`：

```python
q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
q = q.reshape(T, num_q_heads, head_dim)
k = k.reshape(T, num_kv_heads, head_dim)
q = self.q_layernorm(q.reshape(-1, head_dim)).reshape(...)   # 独立 RMSNorm
k = self.k_layernorm(k.reshape(-1, head_dim)).reshape(...)   # 独立 RMSNorm
q, k = self.rotary_emb(positions, q, k)                       # 独立 RoPE
```

而 `sgl_kernel.fused_qk_norm_rope`（`sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu`）把**两个 head-wise RMSNorm 和 RoPE 合并成一个 in-place CUDA kernel**，**Qwen3-MoE 已经在调用**（`models/qwen3_moe.py:559-585`）。

**实测现有链路**：decode 18 次调用 32.5 µs（**1.65%** kernel 时间）；prefill 30 次调用 5581 µs（**3.61%**）。

### 修复

```python
if qkv.dtype == torch.bfloat16 and self.head_dim == 64:
    pos = positions.view(-1).to(dtype=torch.int32, device=qkv.device).contiguous()
    fused_qk_norm_rope(
        qkv, self.num_local_q_heads, self.num_local_kv_heads, self.num_local_kv_heads,
        self.head_dim, self.q_layernorm.variance_epsilon,
        self.q_layernorm.weight, self.k_layernorm.weight,
        self._lfm_rope_theta, self.rotary_emb.is_neox_style, pos, 1.0, 0, 0, 1.0)
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
```

不满足条件时**回退原实现**。LFM2.5 的 `rope_type` 是 `default` 且无 `rope_scaling`，所以 yarn 参数退化为恒等 `(1.0, 0, 0, 1.0)`。

（注：kernel 实际支持 head_dim **64/128/256**，见 `fused_qknorm_rope_kernel.cu:293/313/333`。）

### 收益

**并发 decode +5.42%**——LFM2.5 那一轮里**单项最大的赢家**，而且是纯调用点改动。

---

## 案例 3：Gemma-3 —— 整个 RMSNorm 掉进 eager PyTorch ★

### 问题代码

`sglang/srt/layers/layernorm.py`，**相隔约 100 行**的两个类：

```python
class GemmaRMSNorm(MultiPlatformOp):          # ~line 402（gemma / gemma2 用）
    def _forward_impl(self, x, residual=None, ...):
        if residual is not None:
            gemma_fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        return gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)   # ← 融合 CUDA kernel

class Gemma3RMSNorm(MultiPlatformOp):         # ~line 505（gemma3 用）
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward_cpu(self, x):
        if _is_cpu_amx_available and x.stride(-1) == 1:
            return torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, self.weight, self.eps)  # ← CPU 有融合
        return self.forward_native(x)
    def forward_cuda(self, x):
        return self.forward_native(x)          # ← CUDA 掉进 eager
    def forward_npu(self, x):
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)            # ← NPU 有融合
```

### 代价：1 个 kernel 变 6 个

PyTorch 逐算子执行，中间结果都要写回显存再读出来：

| 步骤 | 显存往返 |
|---|---|
| `x.float()` | 读 x，写 fp32 副本 |
| `.pow(2)` | 读，写 |
| `.mean(-1)` | 读，写 |
| `+ eps` | 读，写 |
| `rsqrt` | 读，写 |
| `x * ...` | 读，写 |
| `* (1.0+w)` | 读，写 |
| `.type_as(x)` | 读，写 |

Gemma-3-1B：**26 层 × 6 个 norm**（input / post_attention / pre_feedforward /
post_feedforward / q_norm / k_norm）= **157 次归一化/forward × ~6 kernel ≈ 940 次 kernel 启动，只为做归一化**。

实测占 **decode CUDA kernel 时间的 15.98%**（还只统计了 mean/rsqrt/pow 三步）。

### 为什么确定是 bug 而不是刻意取舍

**两个类的数值语义逐字相同**：

```python
# GemmaRMSNorm.forward_native            # Gemma3RMSNorm.forward_native
x = x.float()                            output = self._norm(x.float())
variance = x.pow(2).mean(-1, keepdim=True)
x = x * torch.rsqrt(variance + eps)
x = x * (1.0 + self.weight.float())      output = output * (1.0 + self.weight.float())
x = x.to(orig_dtype)                     return output.type_as(x)
```

同样的 fp32 上转、同样的 `(1.0 + weight)`、同样最后才转回。**但一个走融合 kernel，一个不走。** 加上 CPU 和 NPU 都有融合路径——**唯独 CUDA 掉队**。

另外：**上游 main（`a82ead53b`，比我们本地新）仍然是这样**，且 `Gemma3RMSNorm` **没有任何单元测试**——这解释了为什么一直没被发现。

### 修复

```python
def forward_cuda(self, x):
    if not _gemma3_rmsnorm_fused_available(x, self.weight):
        return self.forward_native(x)
    if x.dim() == 2:
        return gemma_rmsnorm(x.contiguous(), self._fused_weight(x.dtype), self.eps)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    return gemma_rmsnorm(flat, self._fused_weight(x.dtype), self.eps).view_as(x)
```

### 两个只有实测才发现的坑

**坑 1：dtype 不匹配会静默出 NaN。**
`self.weight` 由 `nn.Parameter(torch.zeros(dim))` 创建，是 **fp32**；激活是 bf16。融合 kernel 要求两者同类型，**传错不报错，直接算出 NaN**。我第一次测就撞上，差点误判"融合 kernel 语义不对"。所以按 module 缓存转型，且**用 dtype 相等而非 `hasattr` 做守卫**。

**坑 2：`q_norm`/`k_norm` 是 3-D。**
第一版加了"只处理 2-D"的保护，结果这两个（输入 `[token数, 头数, 头维度]`）被挡回 eager——**每层 6 个 norm 里漏了 2 个**。

这个漏洞是**打完补丁后重新跑审计**才发现的（残留 52 次 = 正好 2.00/层）：

| 版本 | eager norm 调用 | decode kernel 时间 | 端到端 |
|---|---:|---:|---:|
| 原版 | **157** | 3.81 ms | 1.00× |
| 补丁 v1（rank-2 守卫） | 52（2.00/层） | 2.40 ms | 1.56× |
| **补丁 v2（含 3-D）** | **0** | **1.89 ms（−50.4%）** | **2.13×** |

RMSNorm 只沿最后一维归约，所以摊平成 2-D 再还原是**精确**的。

**坑 3（工程性）**：`MultiPlatformOp` 在 `__init__` 里绑定 `_forward_method`，只替换类方法对已构造的模块无效，**必须同时 patch 构造函数**（这只影响 monkeypatch 做法；正式源码补丁无此问题）。

### 收益（PR 级验证，真实源码补丁，8 次重复）

| regime | baseline | patched | **加速** | p |
|---|---:|---:|---:|---:|
| A 低批 decode | 0.839 req/s | 1.784 req/s | **2.128×** | 3.5e-22 |
| B 并发 decode | 21.671 req/s | 43.247 req/s | **1.996×** | 4.2e-18 |
| C 长 prefill | 17.156 req/s | 26.088 req/s | **1.521×** | 4.5e-15 |

GSM8K 1319 题 × 3：0.2260 → 0.2213（二项误差 ±2.2 点内）。
数值验证 120 组合零失败，最差相对偏差 9.3e-3。

**诚实边界**：gemma-3-1b 只有 1B，decode 严重 launch-bound，**这个 ~2× 应读作上界**；更大的 Gemma-3 每 forward 计算量大得多，能把固定开销藏起来更多。更大的 checkpoint 在 HF 上是 gated，未能验证。

---

## 修完之后还剩什么（gemma3 的下一步）

打上 norm 补丁后重新审计 gemma-3-1b（低批 decode，kernel 总时间已从 3.81 → 1.84 ms）：

| 剩余空缺 | 次数 | 每层 | 占比 |
|---|---:|---:|---:|
| **独立 residual add** | 52 | **2.00** | **3.00%** |
| layout copy | 6 | 0.23 | 0.49% |

**这正是案例 1 的同型。** `gemma3_causal.py:295-316` 的结构是：

```python
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states      # ← 独立 add
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)   # ← 紧接着的 norm
```

`add 然后 norm` 正好是 `gemma_fused_add_rmsnorm` 的语义——而这个原语**已经存在**
（我的扫描器在 CUDA 符号表里列出了它）。

### 我把它实现并测了 —— 端到端**没有收益**

只融合第一个 add（第二个的后继 norm 是**下一层**的 `input_layernorm`，要跨层
边界改返回签名，刻意不动）。6 次重复，与 `gemma_norm` 直接对比：

| regime | `gemma_norm` | `gemma_norm+residual` | **residual 的边际贡献** | p |
|---|---:|---:|---:|---:|
| A 低批 decode | 2.1226× | 2.1207× | **−0.09%** | 0.311 |
| B 并发 decode | 1.9607× | 1.9693× | **+0.44%** | 0.454 |

**两个 regime 都不显著。** GSM8K 0.2247，与 baseline 一致。

**为什么值得记下来**：审计说这个空缺占 **3.00% 的 kernel 时间**，实测端到端
**0**。这是**次可加性规律的第三次独立印证**——norm 修复已经把主导性的固定开销
拿走了，剩下的同类开销就不再转化。

**两条可操作的教训**：
1. **审计的"% of kernel time"是上界，不是预期收益。** 尤其在一个大修复之后，
   剩余同类空缺的转化率会显著下降。
2. **"不把它塞进 PR"这个决定被数据证实是对的** —— 它会让 PR 的 diff 变大、
   review 变难，却换不到任何端到端收益。

（作为对照，`gemma_norm` 单独的数字在两次独立测量中稳定复现：
A 2.128×/2.123×，B 1.996×/1.961×。）

---

## 对比：手写 kernel vs 补漏用的原语

同一轮工作里我还手写了两个 Triton kernel（ShortConv 的 gate+transpose、MoE 归约+norm），投入远大于上面三个案例：

| 路径 | 投入 | 端到端产出 |
|---|---|---|
| 手写 2 个 Triton kernel（含 tile 扫描、正确性门禁、形状门控） | 大部分时间 | **~6%**（LFM2.5 七项合计 +6.57%） |
| 案例 3（补一处漏用，约 10 行） | 很小 | **2.13×** |

> **杠杆在"找对地方"，不在"写得多精巧"。** 这是这轮工作最该被记住的一句话。

---

## 产物索引

| 内容 | 位置 |
|---|---|
| 案例 1、2 的实现 | `scripts/lfm_fusion/lfm_fusion_patch.py`（`norm` / `qkrope` 组件） |
| 案例 3 的实现（monkeypatch，用于 A/B） | `scripts/lfm_fusion/gemma_fusion_patch.py` |
| 案例 3 的**正式源码补丁 + 单元测试** | `results/lfm_fusion/pr_gemma3/0001-*.patch` |
| PR 草稿 | `docs/2026-07-28/PR_DRAFT_gemma3_rmsnorm.md` |
| 数值验证脚本 | `scripts/lfm_fusion/pr_verify_gemma3.py` |
| 端到端 A/B | `scripts/lfm_fusion/lf_e2e.py` + `lf_analyze.py` |
| 算子级审计 | `scripts/lfm_fusion/lf_audit.py` |
| 原始数据 | `results/lfm_fusion/{audit,e2e,processed,correctness,pr_gemma3}/` |
