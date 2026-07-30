# 融合机会的发现—验证—尝试全流程（结合模型运行流程讲）

**日期**：2026-07-30
**写给**：要设计 agent 自动找 fusion 机会的人
**特点**：所有步骤都用**真实案例现场跑过**，包括一次「找到 2 个候选、全是误报」的完整演示。

---

## 0. 先纠正一个我之前的分类错误

我在上一份文档里把 **LFM2.5 的 MoE config** 放进了 fusion 案例的对比表。**这是错的。**

| | 改的是什么 | 属于 |
|---|---|---|
| **fusion** | 把 N 个 kernel 合成 1 个，减少显存往返 | kernel **结构** |
| **config tuning** | 同一个 kernel，换 tile 形状/warps/stages | kernel **参数** |

MoE config 那个案例（+23.3%）**一个 kernel 都没合并**，只是让已有的 kernel 用对了 tile 尺寸。
它和 fusion 是**两条独立的优化线**，不该放在同一张表里比较。

**修正后的结论**：如果只讨论 fusion，FX 和 profiling 的覆盖面差距没有我之前说的那么大。
真正的区别在别处（见 §4）。

---

## 1. 先看融合机会长在模型的什么位置

一层 Transformer 的执行流（gemma-3-1b，真实 shape）：

```
hidden_states [7, 1152]
   │
   ▼ ① input_layernorm ────────── RMSNorm：pow→mean→add→rsqrt→mul  ← 融合点
   ▼   qkv_proj                    大矩阵乘（GEMM，融合价值低）
   ▼   split + unflatten           纯 view，不产生 kernel
   ▼ ② q_norm / k_norm ────────── 又一次 RMSNorm                   ← 融合点（我们的 PR）
   ▼   rotary_emb ──────────────── RoPE：多个逐元素算子             ← 融合点（qkrope）
   ▼   attn                        注意力（已有高度优化的 kernel）
   ▼   o_proj                      GEMM
   ▼ ③ post_attention_layernorm ── RMSNorm                          ← 融合点
   ▼ ④ + residual ──────────────── 单独一个 add kernel              ← 融合点（deferred residual）
   ▼ ⑤ pre_feedforward_layernorm ─ RMSNorm                          ← 融合点
   ▼   mlp: gate_up_proj           GEMM
   ▼      act_fn ───────────────── SwiGLU：silu + mul               ← 融合点
   ▼      down_proj                GEMM
   ▼ ⑥ post_feedforward_layernorm  RMSNorm                          ← 融合点
   ▼ ⑦ + residual                                                   ← 融合点
```

**规律**：融合机会几乎全在 **GEMM 之间的「胶水算子」**上——
归一化、激活、残差、位置编码。这些的共同特征是：

| 特征 | 后果 |
|---|---|
| **逐元素**（每个数独立处理） | 算术强度极低，全是在搬数据 |
| **算得少读得多** | memory-bound，瓶颈是显存带宽不是算力 |
| **一个接一个** | 中间结果写回显存又立刻读出来，纯浪费 |

而 GEMM 本身**不是**融合目标——它是 compute-bound，cuBLAS/CUTLASS 已经优化到极致。

★ **给 agent 的第一条先验：只在 GEMM 之间的逐元素算子链上找机会，别碰 GEMM。**

---

## 2. 融合到底省了什么（用数字说）

同一个 RMSNorm，`[4096, 1152]` bf16，实测：

| 执行方式 | kernel 数 | GPU 时间 |
|---|---:|---:|
| eager（每个算子一个 kernel） | **7** | 65.5 us |
| Inductor 自动融合 | 1 | 10.1 us |
| 手写融合 kernel | 1 | **6.8 us** |

**7 个 kernel 变 1 个 → 6.5 倍。** 省的不是计算，是**6 次显存往返**：

```
eager:  读x→算pow→写   读→算mean→写   读→算rsqrt→写   ...   (7 次读写)
fused:  读x→ 全部在寄存器里算完 →写                          (1 次读写)
```

---

## 3. 四个阶段，逐个说清楚

### 阶段 1：静态扫描（秒级，不占 GPU，**故意允许误报**）

**目的**：用最低成本列出所有「结构上可能没融合」的地方。

**两种扫法：**

**(a) 签名扫描**——我们命中率最高的一条：
> **框架里已经有融合 kernel，但某个调用点没接上。**

具体做法是对比「供给侧」和「需求侧」：

```bash
# 供给侧：sgl_kernel 里有哪些融合 kernel
python -c "import sgl_kernel; print([o for o in dir(sgl_kernel) if 'rmsnorm' in o])"
# -> ['fused_add_rmsnorm', 'gemma_fused_add_rmsnorm', 'gemma_rmsnorm', 'rmsnorm']

# 需求侧：哪些 forward_cuda 直接回退到 native（说明没用上融合 kernel）
grep -rn "def forward_cuda" -A 3 layers/*.py | grep -B1 "return self.forward_native"
```

**(b) FX 图扫描**——找展开的逐元素算子链：

在 post-grad 图上匹配这种形态（实测导出的真实样子）：
```
convert_element_type → pow → mean → add → rsqrt → mul → add → mul → convert_element_type
```
连续的逐元素算子，且中间结果 `num_users=1`（没别人用）→ 候选。

**这一步的定位是高召回、低精度。误报是预期内的，交给阶段 2 过滤。**

---

### 阶段 2：执行确认 ★★ 最关键，也最容易被跳过

**目的**：回答一个问题——**这个候选在真实运行中，真的会被执行、且真的重要吗？**

**这一步我用现场演示说明为什么必要。** 刚才阶段 1 在 sglang 上扫出 2 个候选：

#### 候选 A：`RMSNormWithoutScale`（layernorm.py:1194）

```python
def forward_cuda(self, x):
    return self.forward_native(x)      # 看起来是个完美的融合空缺
```

**阶段 2 检查：有模型在用它吗？**
```bash
grep -rn "RMSNormWithoutScale" python/sglang/srt/models/*.py
# 空 —— 没有任何模型使用
```
**→ 死代码。修了也是零收益。淘汰。**

#### 候选 B：`QuickGELU`（activation.py:259）

```python
def forward_cuda(self, x):
    return self.forward_native(x)      # x * sigmoid(1.702*x)，也没融合
```

**阶段 2 检查（一）：有模型在用吗？**
```bash
grep -rln "QuickGELU" python/sglang/srt/models/*.py
# clip.py, cohere2_vision.py, ernie45_vl.py, kimi_vl.py  ← 有，4 个视觉模型
```
有人用，继续查。

**阶段 2 检查（二）：有对应的融合 kernel 吗？**
```bash
python -c "import sgl_kernel; print([o for o in dir(sgl_kernel) if 'gelu' in o.lower()])"
# ['gelu_and_mul', 'gelu_tanh_and_mul']
```
**只有 `gelu_and_mul`（gelu 后再乘，用于 gated MLP），
没有独立的 `quick_gelu`。而 QuickGELU 是 `x*sigmoid(1.702x)`，是另一个函数。**
**→ 没有现成 kernel 可接。淘汰**（要做就得自己写 kernel，成本完全不同）。

**演示结论：阶段 1 找到 2 个，阶段 2 全部否掉，理由还不一样。**
没有阶段 2，agent 会报告两个毫无价值的"机会"。

#### 阶段 2 还要做的第三类检查：这条路径真的执行吗

这是我们**吃过大亏**的一条。OLMo-2 的代码：

```python
if self.alt_stream is not None and get_is_capture_mode():
    q = self.q_norm(q.reshape(-1, q_shape[-1]))   # 融合
else:
    q = self.q_norm.forward_native(q)             # eager ← 扫描会命中这里
```

静态扫描（和 FX trace）会报告 `else` 分支是个融合机会。**但它在生产中几乎不执行**：

装一个分支计数器实测：
```
[olmo2_branch] {'fused_capture': 384, 'eager_else': 16}
```
**CUDA graph 捕获时走的就是融合路径**，稳态 decode 回放的图里根本没有 eager kernel。

我们当初没做这个检查，导致 profiling 审计**高估了 10 倍**（报 7.71%，实测 +0.45%）。

**阶段 2 的通用形式：别信代码怎么写的，去查实际执行了什么。**

| 想确认的事 | 怎么查 |
|---|---|
| 这个分支走了吗 | 装计数器 / 看 CUDA graph capture 日志 |
| 实际编译了哪个 kernel | 翻 Triton 缓存的 `.ttgir`（我们靠这个发现 BK=32 从未编译） |
| 实际加载了哪个 config | 断言而非假设（已固化成 `assert_clean_baseline()`） |
| 这个层被调用几次 | monkeypatch 一个计数器跑一遍 |

---

### 阶段 3：profiling 定量（分钟级，占 GPU）

**目的**：给通过阶段 2 的候选**排优先级**——它占多少 kernel 时间？

```bash
python scripts/lfm_fusion/lf_audit.py --model gemma3 --regime A_low_batch_decode --gpu 4
```

输出形如：
```
=== gemma3 / A_low_batch_decode / decode ===
  norm            46.32%   ← 这就是收益上界
  dense_gemm      31.20%
  attention        9.66%
```

**两个必须遵守的纪律：**

1. **必须在生产配置下测**（CUDA graph 开着）。
   我们关掉它做归因，结果测了个不存在的开销。
2. **`% kernel time` 是上界，不是可实现收益。**
   即使全部消除，端到端也拿不到那么多——因为还有 CPU 开销、内存带宽、其他瓶颈。

---

### 阶段 4：端到端 A/B（唯一算数的证据）

前三个阶段都只是**筛选**。真正的收益只有这里能证明。

**必须做到的四件事：**

| | 为什么 |
|---|---|
| **多次重复 + 统计检验** | 单次运行的差异可能纯是噪声 |
| **顺序对照** | LFM 那次 decode 初测 −0.37%(p=4.9e-04) 像真回归，**把顺序调换后符号翻转**，证明是位置效应 |
| **多个 regime 都测** | 我们的组件里，`conv` 在 decode 精确中性、在长 prefill +2.33%。只测一个看不全 |
| **组合要整体测** | **次可加性**：各项之和 12.80%，一起测只有 6.21%，兑现率 **0.49** |

★ 次可加性这条特别重要：**不能把候选的预估收益线性相加**，
越饱和的 regime 高估越严重（我们实测兑现率 0.90 / 0.70 / 0.49，精确跟踪 regime 饱和度）。

---

## 4. 那 FX 和 profiling 到底怎么分工

修正掉 MoE config 那个错误分类之后，真实的区别是：

| | FX graph | profiling |
|---|---|---|
| 回答 | 「**结构上**这里没融合」 | 「这里**实际**花了多少时间」 |
| 成本 | 秒级 / CPU | 10+ 分钟 / 独占 GPU |
| 覆盖 | 能看到**没被执行到的路径** | 只能看到**这次跑到的** |
| 精确性 | 数据依赖精确（`num_users` 免费给出融合安全性） | 有测量噪声 |
| 失败模式 | **只记录 trace 那刻走的分支** | **只对测量时的配置成立** |
| 能否排优先级 | ❌ | ✅ |

**两者的失败模式我们都踩过，而且是同一个坑的两面**：

- FX：trace 时 `capture_mode=False` → 报告一个不执行的机会
- profiling：测量时 `--disable-cuda-graph` → 测到一个不存在的开销

**所以阶段 2 不是可选项，它是把这两个失败模式都堵上的那道闸门。**

**分工建议**：
```
FX      → 阶段 1 的扫描器之一（廉价、可扫 100 个模型、能看到冷门路径）
签名扫描 → 阶段 1 的另一个扫描器（我们命中率最高的）
profiling → 阶段 3（排序，且必须在生产配置下）
A/B      → 阶段 4（唯一算数）
```

---

## 5. 完整走一遍：Gemma-3 案例（我们真实做过的）

| 阶段 | 做了什么 | 结果 |
|---|---|---|
| **1 扫描** | 发现 `Gemma3RMSNorm.forward_cuda` 有 `if x.dim()==2` 守卫，高维掉回 native | 候选 ✅ |
| **2 确认** | ① 有对应 kernel？`gemma_rmsnorm` 存在 ✅<br>② 有模型用？gemma-3 主力路径 ✅<br>③ 真的执行？`q_norm`/`k_norm` 每 forward 调 52 次 ✅ | 通过 |
| **3 定量** | 审计：norm 占 decode kernel 时间 **46.32%** | 高优先级 |
| **4 A/B** | 真实源码补丁，6 次重复，Welch t | **+36.6% / +24.5%**，p=2.4e-14 / 1.2e-06<br>prefill +7.3% **标注不显著** |

**额外做的两件事**（PR 级别才需要）：

- **数值验证**：20 个 shape/dtype 组合，含非连续输入和 residual 路径
- **变异测试**：故意破坏修复，确认测试会失败（去掉 dtype 守卫 → 68 个失败；去掉 shape 还原 → 28 个失败）
  → **一个不会失败的测试等于没测**

---

## 6. 反例：通过了前三阶段但阶段 4 挂掉

`LFM2.5 gate+idx` 组件：

| 阶段 | 结果 |
|---|---|
| 1 扫描 | ✅ 结构上确实没融合 |
| 2 确认 | ✅ 真的在执行 |
| 3 定量 | ✅ kernel 级可测到 1~2% |
| **4 A/B** | ❌ **三个 regime 全不显著** |

**机制真实存在、kernel 级能测到，但没兑现到端到端。**

这就是为什么阶段 4 不能省——前三个阶段全通过，仍然可能是零。

---

## 7. 给 agent 的检查清单

```
阶段1 静态扫描
  □ 枚举框架已有的融合 kernel（供给侧）
  □ 找 forward_cuda 回退到 native 的调用点（需求侧）
  □ FX 图上找连续逐元素算子链（num_users==1）
  □ 只看 GEMM 之间的胶水算子，跳过 GEMM 本身

阶段2 执行确认 ★
  □ 有模型真的用这个层吗？        （否 → 死代码，淘汰）
  □ 有对应的融合 kernel 吗？      （否 → 要自己写，成本级别不同）
  □ 这个分支在生产配置下执行吗？   （装计数器，别读代码猜）
  □ 每次 forward 调用几次？       （次数太少 → 收益封顶）

阶段3 profiling
  □ 在生产配置下测（CUDA graph 开着！）
  □ 记住 % kernel time 是上界

阶段4 端到端 A/B
  □ 多次重复 + 统计检验
  □ 顺序对照（arm 顺序调换，看符号会不会翻）
  □ 多个 regime
  □ 组合整体测（次可加性，别线性相加）
  □ 数值等价验证
  □ 不显著就如实标注，别塞进 headline
```

---

## 8. 一句话总结

> **发现很便宜，验证很贵，而收益只有验证能证明。**
> **agent 的价值不在于「找到更多候选」，而在于「用最低成本把假候选淘汰掉」——
> 因为真正贵的是阶段 4，每个假候选都要花掉一次完整的 A/B。**

我们现场演示的那次扫描就是证据：**2 个候选，0 个为真**。
如果直接进阶段 4，就是两次白跑的 GPU 实验。
