# Gemma-3 fused QK-norm + RoPE：从发现到验证的完整实验记录

**日期**：2026-07-31 · **硬件**：H200（GPU 0–3） · **基线 commit**：sglang `89f4a80c1f`（当天 main）
**读者**：Chendi

本文档记录一次完整的 kernel fusion 优化：怎么找到机会、怎么建基线、kernel 具体改了什么、
性能提升多少、以及**过程中三次被自己的测量方法误导并纠正**。

---

## 0. 摘要

| 项 | 结论 |
|---|---|
| 原计划 | 验证前一天的「手工合并 q/k norm」方案 |
| **实际做的** | 侦察时发现 sglang **自带** `fused_qk_norm_rope`（连 RoPE 一起融），**原方案作废**，见 §1.0 |
| 机会 | Gemma-3 把 q_norm / k_norm / RoPE 当三个算子调用，而这个 kernel 一次融完 |
| 怎么发现的 | **静态扫描**（scan 1b：模型没调用同僚在用的原语），非 profiling、非人工读代码 |
| kernel 改动 | 加编译期 `JIT_ADD_ONE` 标志，在 fp32 域做 Gemma 的 `(1+w)` |
| 微基准 | 对 PR #32670 基线 **1.5–2.3×**，全 token 范围无退化 |
| 精度 | 对 fp64 参考 **0.141%**，**优于**模型现有路径（0.14–0.23%），均在 bf16 ULP（0.39%）内 |
| GSM8K | 21.50% → 22.00%，**McNemar p=0.875，无可检测变化**（21 胜 19 负） |
| **端到端（诚实数字）** | **+0.5% 到 +1.1%**（对融合-norm 基线，7 次重复） |
| 端到端（对 main） | +24.8% 到 +37.9%——**不能用，97% 是 PR #32670 的成果** |
| **附带找到并修掉的第二个缺口** | **OLMo-2 prefill：87.8ms → 70.8ms = 1.24× / +24%，GSM8K p=1.000 无变化** |

**必须先说的一件事**：本工作最初测出的端到端数字是 **1.34–1.39×**（吞吐 +24.8% 到 +37.9%）。
**这个数字不能用**，因为基线是未修的 main，里面还含着 PR #32670 正在修的 rank 守卫缺口。
消融实验证实：那个提升的**约 97% 来自 rank 守卫修复**，
rope 融合本身的增量是 **+0.5% 到 +1.1%**。详见 §6。

**收益最大的其实不是主线**：跨模型扫描时在 OLMo-2 上发现了另一个缺口——
它的 `_apply_qk_norm` 在非 CUDA-graph-capture 路径上**显式调用 `forward_native`**，
绕过了自己本来就能命中的融合 kernel。因为 prefill 从不被 graph 捕获，
**整个 prefill 阶段一直在丢**。修掉后 prefill **1.24×**，且生成 **8/8 完全一致**。详见 §7.4。

---

## 1. 机会是怎么发现的

### 1.0 先回答原问题：那个「手工合并 q/k norm」的方案怎么样了

出发点本来是验证前一天发现的 **QK-norm 手工合并**——把 `q_norm(q)` 和 `k_norm(k)`
两次调用合成一次 `[tokens, heads, head_dim]` 的 norm 配 per-head 权重。
侦察阶段查到 sglang **已经自带** `fused_qk_norm_rope`，
连 RoPE 一起融，于是转向了它。**这个转向让原方案作废，理由是三条硬的**：

| | 手工合并 q/k norm | `fused_qk_norm_rope` |
|---|---|---|
| 融了什么 | 2 个 norm | **2 个 norm + RoPE** |
| 微基准（小 T） | 1.94× | **2.3×** |
| **大 T（4096）** | **0.56×，反而更慢** | 全范围加速，无退化 |
| 落地成本 | `sgl_kernel.gemma_rmsnorm` 的 weight 是 1-D，**表达不了 per-head 权重 → 必须新写 kernel** | 已存在，只需加一个编译期标志 |
| 端到端验证 | 从未做过 | 见 §6 |

原方案的数据（`results/fx_fusion/qknorm_merge.csv`）仍然有效，
它证明的是**「slice 阻断了横向融合」**这个机制；
但作为一个可落地的优化，它被一个已经存在的、更强的 kernel 取代了。

> **结论：原方案不再推进。** 它需要新写 kernel、在大 batch 上会退化，
> 而 sglang 自带的 kernel 在所有维度上都更好。

### 1.1 方法：scan 1b（模型没调用同僚在用的原语）

来自 `.github/skills/fusion-gap-hunting/SKILL.md`。规则是纯机械的：

> 框架积累融合 kernel，每个新模型文件必须**主动 opt-in**。没有任何机制强制它这么做，
> 也不会因此失败——模型仍然**正确**，只是更慢。所以这类缺口对测试不可见，只在 profile 里现形。

具体到本例：

```bash
grep -rl "fused_qk_norm_rope" python/sglang/srt/models/
#   qwen3_moe.py  deepseek_v4.py  mellum.py  interns1pro.py
grep -c "fused_qk_norm_rope" python/sglang/srt/models/gemma3_causal.py
#   0
```

而 gemma3 的注意力前导是（`gemma3_causal.py:254-261`）：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = self.q_norm(...)      # 算子 1
k = self.k_norm(...)      # 算子 2
q, k = apply_rotary_pos_emb(q, k, cos, sin)   # 算子 3
```

**三次独立调用，可以是一次。**

脚本化后（`scripts/fx_fusion/scan_qknorm_rope_candidates.py`）在全 212 个模型文件上跑：

```
已调用 fused_qk_norm_rope 的模型          : 4
有独立 q_norm/k_norm 调用且未融合 rope 的 : 18
其中不需要新 kernel 变体的                : 16
```

### 1.2 兼容性闸门（做之前必须过）

| 约束 | kernel 要求 | Gemma-3 | 结论 |
|---|---|---|---|
| head_dim | 64 / 128 / 256 | 256 | ✓ |
| dtype | bf16 only | bf16 | ✓ |
| rope 风格 | NeoX 或 GPT-J interleave | NeoX | ✓ |
| rope base | 运行时参数 | 局部 1e4 / 全局 1e6 交替 | ✓ 可逐层传 |
| norm 语义 | `x * rms * w` | `x * rms * (1+w)` | ✗ **需要改 kernel** |

---

## 2. Kernel 改了什么

### 2.1 问题：Gemma 的 `(1+w)`

kernel 里是一行（`fused_qknorm_rope.cuh:147-150`）：

```cpp
wvec.load((isQ ? q_weight : k_weight) + offsetThread - offsetWarp);
for (int i = 0; i < numElemsPerThread; i++) {
  elements[i] *= rms_rcp * device::cast<float>(wvec[i]);
}
```

**第一次尝试（错的）**：在 host 侧把 `1+w` 预算成 bf16 传进去。

### 2.2 改动：编译期 `JIT_ADD_ONE`

```cpp
template <int head_dim, bool interleave, bool yarn, bool add_one = false>
...
for (int i = 0; i < numElemsPerThread; i++) {
  float w = device::cast<float>(wvec[i]);
  if constexpr (add_one) {
    w += 1.0f;              // 在 fp32 域做，不落 bf16
  }
  elements[i] *= rms_rcp * w;
}
```

Python 侧把 `add_one` 加进 JIT 缓存键和 `-DJIT_ADD_ONE` 编译宏，
所以两个变体是**两份独立编译产物**，零运行时分支开销。

补丁：`patches/gemma3_fused_qknorm_rope/02_model_and_kernel.patch`

### 2.3 模型接入

`gemma3_causal.py`：构造期解析一次能力，forward 里只是一个 bool 判断。

```python
self.use_fused_qk_norm_rope = False
if _is_cuda and self.rotary_dim == self.head_dim:
    self.use_fused_qk_norm_rope = can_use_fused_qk_norm_rope(
        self.head_dim, is_neox=True, dtype=torch.bfloat16, add_one=True)
```

**两个踩到的坑**：

1. **`positions` 不是 1-D**。模型在 `gemma3_causal.py:703` 做了
   `einops.rearrange(positions, "s -> 1 s")` 以配合 `Gemma3RotaryEmbedding`，
   而 kernel 要求 `[num_tokens]`。需 `positions.reshape(-1)`。
2. **CUDA 路径走的是 `forward_native`，不是 `forward_cuda`**。
   `forward()` 里只有 CPU+AMX 才走 `forward_cpu`，其余一律 `forward_native`——
   命名有误导性，改错地方会完全不生效。

---

## 3. 三次测量方法上的错误（以及怎么发现的）

这一节是本文档最有价值的部分。三次都是**测量方法错**，不是代码错，
且每一次的第一版结论都是错的。

### 3.1 错误一：把 bf16 量化误判为 kernel 缺陷

**现象**：预算 `1+w` 后，最大相对误差 3.94%。
**当时的推断**：bf16 在 1.0 附近 ULP 是 2⁻⁷，折叠导致精度损失 → 于是加了 `add_one`。
**结果**：加完之后误差**纹丝不动**，还是 3.94%。假设被证伪。

**真相**（`scripts/fx_fusion/locate_fused_qknorm_error.py`）：

```
sglang Gemma3RMSNorm vs fp64          :   1.86%
kernel norm (rope-free 尾部) vs fp64  :   1.86%     ← 完全一样
一个 bf16 ULP 在该量级                :   0.39%
```

**两条路径精度完全相同**，1.86% ≈ 5 个 ULP，是 bf16 输出的固有量化。
3.94% 是**对 bf16 参考取最大相对误差**造成的——被某个接近零的元素主导。

**教训**：拿一个近似去对另一个近似，只能告诉你它们不一致，**不能告诉你谁错**。
必须有 fp64 第三方。

改成对 fp64 取**平均**相对误差后：

| tokens | bf16 ULP | sglang 现有 | add_one kernel | 结论 |
|---|---|---|---|---|
| 1 | 0.39% | 0.14% | **0.14%** | 无精度代价 |
| 128 | 0.39% | 0.20% | **0.14%** | 无精度代价 |
| 2048 | 0.39% | 0.20% | **0.14%** | 无精度代价 |

**融合后反而更准**：kernel 全程 fp32 寄存器，norm 和 rotate 之间不落 bf16。
`add_one` 最终保留了，但理由是调用更干净（不用维护 host 侧权重 buffer），
**不是**精度需要。

### 3.2 错误二：对着错误的参考验证

第一版等价性验证用 `get_rope` 做参考。但 Gemma-3 的 CUDA 路径**不用它**——
它用 `Gemma3RotaryEmbedding` 预算的 cos/sin + `apply_rotary_pos_emb`，
而且先 transpose 到 `[b, h, s, d]`。三处可能不一致：布局、`attention_scaling`、
cos/sin 的 base 来源。

改用模型自己的路径后，报出：**局部层 match，全局层 87% 不一致**。
看起来像 kernel 在 base=1e6 时有 bug。

**又是比较方法的问题**。分别对 fp64 打分：

```
base=1e6, T=64
  模型路径 vs fp64 :   0.194%
  kernel   vs fp64 :   0.141%
```

两条都对。87% 是因为**两个 bf16 近似互相直接相减**，且相对量取自 rotate 后接近零的元素。

### 3.3 错误三：噪声基线恒为零

GSM8K 闸门设计了一个"噪声基线臂"——同一棵树换 seed 再跑一次。
它报 **0.00 pts**，于是 `+0.50 pts` 被判为 "IMPROVED beyond noise"。

**这个噪声基线测不到任何东西**：greedy 解码是确定性的，换 seed 什么都不会变，
所以它**必然**是 0。用它当分母会让**任何**差异都显著。

正确做法：两臂答的是**同一批题**，应该用配对检验。

```
n = 400
  both correct        : 67
  both wrong          : 293
  baseline only (输)  : 19
  fused only    (赢)  : 21

  精确 McNemar p = 0.875   →  无可检测的精度变化
```

**40/400 的答案变了，净效果是抛硬币**——正是 bf16 级扰动应有的样子。
结论应表述为「精度不变」，不是「精度提升」。

---

## 4. 正确性闸门

| 闸门 | 结果 | 判读 |
|---|---|---|
| 逐元素 vs fp64 | kernel 0.141% / 模型 0.14–0.23% | 融合不是精度取舍 |
| 对模型真实 rope 路径 | 全部 OK，kernel 更接近 fp64 | 无约定不匹配 |
| greedy 生成 token 一致性 | **6/8** | 不是 bit-exact，符合预期 |
| GSM8K 配对 McNemar | p=0.875 | 无可检测变化 |

**为什么 token 一致性不是合适的闸门**：融合路径把 norm+rotate 全放在 fp32 寄存器，
模型现有路径在两者之间落一次 bf16。**两者本就不可能 bit-identical**。
6/8 只是复述了「算术变了」这个已知事实。Gemma-3 是 dense 模型，
没有 routed expert 会因 bf16 扰动而离散跳变，所以任务指标（GSM8K）是可用的闸门。

---

## 5. 微基准

`scripts/fx_fusion/verify_add_one_kernel.py`，H200，bf16，head_dim=256，4 q-head + 1 kv-head。

**三臂**，因为只报「对 main」会把 PR #32670 的功劳算进来：

| tokens | main（rank 守卫未修） | **#32670 基线** | fused | vs main | **vs #32670** |
|---|---|---|---|---|---|
| 1 | 26.03us | 3.72 | 2.50 | 10.26× | **1.48×** |
| 8 | 30.13 | 6.10 | 2.61 | 11.43× | **2.34×** |
| 32 | 32.05 | 6.32 | 2.78 | 11.38× | **2.27×** |
| 128 | 34.14 | 7.18 | 3.15 | 10.83× | **2.28×** |
| 512 | 39.56 | 8.91 | 4.28 | 9.06× | **2.08×** |
| 2048 | 66.13 | 17.99 | 9.50 | 6.91× | **1.89×** |

**该引用的是 1.5–2.3×，不是 7–11×**。且**全 token 范围都是加速**——
不像手工合并 q/k norm 那个方案在 T=4096 会退化到 0.56×。

---

## 6. 端到端

（结果见 §6.1 表格，由 `scripts/fx_fusion/e2e_ab_gemma3.py` 产出）

### 6.0 为什么需要消融臂

第一版 A/B 只有两臂（main vs fused），结果是 decode **1.34–1.39×**、吞吐 **+24.8% 到 +37.9%**，
四个 regime 全部 p<0.001。

**这个数字不能用。** 基线是未修的 main，它的 4-D q/k norm 因 `if x.dim() == 2` 守卫
掉回 eager（每个 norm 10 个 kernel）。PR #32670 正在修这个。
只报「对 main」等于把那个 PR 的成果算进本次改动。

**这正是 skill 里 PR READINESS 一节写过的教训**——上一次是 #32383 落地后，
我们 `2.13×` 的标题数字悄悄变成了在 claim 别人的工作，诚实的数字是**增量**（`+36.6%`），
大约是原来的三分之一。

### 6.1 消融设计

第一次尝试是建第三棵 worktree，只打 rank 守卫修复。**它崩了**：

```
RuntimeError: view size is not compatible with input tensor's size and stride
```

根因很微妙：main 的 eager 路径对**非连续**输入做逐元素运算，
PyTorch **保留了输入的 stride 模式**，于是后面 `permute(0,2,1,3)` 转回来正好连续。
我强制 `.contiguous()` 后再 reshape，permute 回去反而**非**连续，
attention backend 的 `o.view(...)` 拒绝。

**改进的设计**：把消融臂放进**同一棵树**，用环境变量切换：

```python
SGLANG_GEMMA3_NO_FUSED_ROPE=1   # 融合 norm，rope 独立启动 = #32670 等价
（不设）                          # 全融合
```

两臂共享同一份布局处理代码，A/B 只隔离 rope 融合本身，
不会混入两棵树之间的其他差异。

### 6.2 结果

`results/fx_fusion/e2e_ab_gemma3_ablation.json`，7 次重复，Welch t 检验：

| regime | main | 融合norm<br>(#32670 等价) | 全融合 | vs main | **vs 融合norm<br>（真实增量）** | p |
|---|---|---|---|---|---|---|
| decode bs=1 | 3.010 ms | 2.180 | 2.170 | 1.387× / +37.9% | **1.005× / +0.63%** | 0.000 |
| decode bs=32 | 3.310 | 2.400 | 2.390 | 1.385× / +37.5% | **1.004× / +0.77%** | 0.000 |
| decode bs=64 | 3.720 | 2.800 | 2.780 | 1.338× / +28.7% | **1.007× / +0.49%** | 0.073 **n.s.** |
| prefill heavy | 3.320 | 2.510 | 2.380 | 1.395× / +24.8% | **1.055× / +1.09%** | 0.008 |

> **那 1.34–1.39× 里约 97% 来自 rank 守卫修复，rope 融合本身只有 0.5–1.1%。**

三点判读：

1. **该报的数字是 +0.5% 到 +1.1%**，不是 +25% 到 +38%。
   后者绝大部分是 PR #32670 的成果，只是因为它还没落地才出现在我的基线里。
2. **bs=64 那一档 p=0.073，不显著**，按规矩它不算结果，必须带着判定一起写出来，
   不能进标题数字。
3. **prefill 那一档增量最大（+1.09%）**，符合机制预期：
   preamble 的固定开销在长序列上摊得更薄，但 kernel launch 数的节省是常数级的，
   在 prefill 这种大 batch × 长序列的场景里绝对节省更多。

**为什么微基准 2.3× 只换来端到端 1%**：preamble 只是 decode 步的一小片。
26 层 × 每层省约 3.5us ≈ 91us，而一个 decode 步是 2.2ms——约 4%，
再考虑 kernel 之间的重叠，实测 0.5–1.1% 是合理的。
**微基准的加速比不能外推成端到端**，这是本项目第四次撞上同一条教训。

---

## 7. 跨模型扫描

对 6 个模型跑完整流程（`scripts/fx_fusion/scan_models_pipeline.py`）。

### 7.1 静态可用性

| 模型 | 家族 | head_dim | 可用？ | 原因 |
|---|---|---|---|---|
| gemma3_causal | Google | 256 | **YES** | — |
| qwen3 (dense) | Qwen | 128 | **YES** | — |
| olmo2 | AI2 | 128 | no | **q_norm 跨 head 归一化，语义不等价** |
| olmoe | AI2 | 128 | no | 同上 |
| exaone4 | LG | 64 | no | **rope_type=llama3，kernel 不支持** |
| qwen3_moe | Qwen | 128 | no | 已经在用 |

### 7.2 两个假阳性（筛选器的价值在这里）

**假阳性 A —— OLMo-2 / OLMoE**：源码看起来一模一样（`q_norm` + `k_norm` + `rotary_emb` 分开调），
config 里 head_dim=128 也在支持范围。但：

```python
# olmo2.py:118-122
self.k_norm = RMSNorm(self.total_num_kv_heads * self.head_dim, ...)
self.q_norm = RMSNorm(self.config.hidden_size, ...)      # ← 整个 q 投影一起归一化
```

**它是跨 head 归一化，kernel 是 per-head**。数学不等价，接进去会**静默产生错误结果**。
config.json 里看不出来，只有读模型源码的构造参数才知道。
筛选器已加此检查（`norm_scope()`）。

**假阳性 B —— EXAONE-4**：`rope_scaling.rope_type = "llama3"`，
kernel 只实现了 default 和 YaRN。频率会算错，同样是静默错误。

> **这两个假阳性是本次扫描最有价值的产出**：静态扫描的召回很高但精度不足，
> 而这一类失败**不会报错、不会跑挂**，只会给出错的数。
> 每一条都必须有对应的判据写进筛选器。

### 7.3 Profiling 验证「家族关注度」预测

`--disable-cuda-graph` 下的 decode profile，统计 eager-norm 签名调用：

| 模型 | 家族 | eager-norm 调用 | 占 kernel 时间 |
|---|---|---|---|
| **gemma3** | Google | **157** | **6.41%** |
| **olmo2** | AI2 | **97** | **6.62%** |
| olmoe | AI2 (MoE) | 1 | 0.13% |
| qwen3 | Qwen | 1 | 0.32% |

skill 里的预测指标（**不是**架构新旧或模型大小，而是**该模型文件受到过多少优化关注，
这跟家族在框架用户群里的分量相关**）在这批样本上继续成立：
两个非 Qwen 家族都在 6% 以上，Qwen 系两个都低于 0.35%。

**但 OLMo-2 那 6.62% 修不了**——它的 norm 语义根本不匹配这个 kernel。
**「有缺口」和「能用现成 kernel 补」是两件事。**

### 7.4 第二个可修复的缺口：OLMo-2 的 prefill（本次实际修掉了）

7.2 说 OLMo-2 不能用 `fused_qk_norm_rope`（跨 head 归一化，语义不等价）。
但 7.3 那 6.62% 是真的，所以问题变成：**它到底为什么是 eager 的？**

读代码（`olmo2.py:165-192`）：

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
- **所以 `self.q_norm(q)` 直接就能命中融合 kernel**，`forward_native` 是白丢

而 `get_is_capture_mode()` 只在 CUDA graph **捕获期**为真。也就是说：

| 阶段 | 走哪条 | 后果 |
|---|---|---|
| decode（graph 已捕获后 replay） | 捕获时用的是融合路径 | 已经是快的 |
| **prefill**（从不被 graph 捕获） | **eager** | **一直在丢** |
| decode（`--disable-cuda-graph`） | eager | 丢 |

**这解释了为什么 7.3 profile 出 6.62%**——那次跑的是 `--disable-cuda-graph`。

**修法**（`patches/olmo2_fused_qk_norm/`）：else 分支改成走融合 kernel，形状本来就对。

**验证**：

| 闸门 | 结果 |
|---|---|
| greedy 生成一致性 | **8/8 完全相同**（不同于 Gemma-3 的 6/8） |
| **GSM8K 配对 McNemar** | 65.50% → 65.25%，**p=1.000，无可检测变化**（8 胜 9 负） |
| decode 各 regime | 1.00×，全部 n.s.——**符合预期**，decode 本来就走融合路径 |
| **prefill 直接测量** | **87.80ms → 70.79ms = 1.24× / +24%** |
| prefill-heavy 总吞吐 | **+17.6%，p<0.001** |

**为什么这次是 8/8 而 Gemma-3 是 6/8**：这里融合路径和 eager 路径都是标准 RMSNorm，
数值上等价；Gemma-3 那次是把 norm+rope 合成一个 kernel，中间少了一次 bf16 舍入，
**本就不可能 bit-identical**。

（注意 8/8 是那 8 条短 prompt；GSM8K 400 题里仍有 17 题答案变了——
融合 kernel 和 eager 的规约顺序不同，仍非 bit-exact，只是短序列上撞不出来。
**这也说明 8 条 prompt 的 token 一致性是弱证据，任务指标才是闸门。**）

**同家族对照**：OLMoE 直接写 `self.q_norm(q.contiguous())`，**没有这个 bug**——
profiling 里它是 1 次 eager-norm 调用，OLMo-2 是 97 次。**扫描正确区分了同家族的两个模型。**

> **这个案例是本次扫描最好的产出**：静态扫描把 OLMo-2 标成「不能用那个 kernel」是对的，
> profiling 把它标成「有 6.62% 缺口」也是对的，**两个结论都对，但要合起来才知道该修什么**。
> 而且它的收益（prefill +24%）比 Gemma-3 的 rope 融合（+1%）大一个数量级。

### 7.5 一个改变全局判读的发现

```python
# server_args.py:1891
enable_fused_qk_norm_rope: A[bool, "...", NS("exec.kernel")] = False
```

**这个 flag 默认关闭。** 也就是说，那 4 个「已接入」的模型**默认也没在用**。

所以本工作的准确表述不是「gemma-3 漏了一个大家都在用的 kernel」，而是：

> sglang 有这个 kernel，把它挂在一个默认关闭的 flag 后面，接了 4 个模型；
> **gemma-3 连这个 flag 都够不着**。

给上游提 PR 时应当**沿用同一个 flag**，而不是无条件启用——
本文的测量等价于「flag 打开 + 已接线」对「flag 打开 + 未接线」，正是该测的对比。

---


## 8. 结论与启示

### 8.1 关于这两个优化

**Gemma-3（rope 融合）**
- 由**静态扫描自动发现**，不需要 profiler、不需要人读代码
- kernel 改动很小（一个编译期标志），**精度不降反升**
- 微基准 1.5–2.3×（对 #32670 基线）
- **端到端只有 +0.5% 到 +1.1%**，其中 bs=64 一档不显著（p=0.073）
- 对 main 的 1.34–1.39× 里 **约 97% 是 rank 守卫修复的功劳**，那是 PR #32670 的
- **值不值得提 PR**：+1% 且无精度代价、代码改动小，值得；但标题绝不能写 1.39×

**OLMo-2（prefill 的 norm 融合）—— 收益大一个数量级**
- 由**扫描 + profiling 交叉**发现：扫描说「不能用那个 kernel」，profiling 说「有 6.62% 缺口」，
  两个都对，合起来才定位到真正该改的地方
- 修法是三行：把 `forward_native` 换成正常调用，形状本来就对
- **prefill 1.24× / +24%**，decode 不变（本来就走融合路径），**GSM8K p=1.000 无变化**
- 更值得提 PR

**一个横向观察**：两个缺口都是「框架有 kernel，调用点没接上」，
但表现形态完全不同——Gemma-3 是**从没接过**，OLMo-2 是**接了一半**
（只在 graph 捕获路径上接）。后者更隐蔽，因为 decode 的 profile 看起来是干净的。

### 8.2 关于方法（更重要）

1. **拿近似比近似，只能证明不一致，不能定位谁错。** 三次误判里有两次是这个原因。
   任何数值验证都要有一个更高精度的第三方参考。
2. **相对误差要看平均，不要看最大**，除非你确认最大值不是被接近零的元素主导。
3. **噪声基线必须真的能动。** greedy 解码下换 seed 的"噪声"恒为 0，
   拿它当分母会让任何差异都显著。同题两臂要用**配对检验**。
4. **每个"这里没融合"的候选都要先过语义等价闸门。**
   OLMo-2 和 Gemma-3 的源码形态一模一样，一个能用一个不能，
   差别只在构造函数的一个参数里。
5. **基线要选「当天的 main 加上所有在飞的修复」**，不是「我开始时的 main」。
   否则会把别人的成果算进自己的数字——这是本项目第二次踩到。
6. **微基准的加速比不能外推成端到端。** preamble 的 2.3× 换来端到端 +0.5% 到 +1.1%，
   因为 preamble 只占 decode 步的一小片（26 层 × 3.5us ≈ 91us，对 2.2ms 的 decode 步）。
7. **不显著就是不显著。** bs=64 那一档 p=0.073，必须带判定写出来并排除在标题数字之外。
8. **一个缺口在哪个阶段暴露，取决于哪条路径被 CUDA graph 捕获。**
   OLMo-2 的 decode profile 是干净的，缺口全在 prefill——
   因为它的融合分支条件是 `get_is_capture_mode()`。
   只看 decode 的 profile 会完全错过它。
9. **别用 `--disable-cuda-graph` 的 profile 直接推断线上行为。**
   它是看清单个 kernel 的标准手段，但对 OLMo-2 这种按 capture-mode 分支的代码，
   它**改变了走哪条路**。6.62% 是真的，但它在 decode 上不成立。

---

## 9. 复现

```bash
ENV=~/.conda/envs/gemma-sglang; CU13=$ENV/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU13 PATH=$CU13/bin:$ENV/bin:$PATH LD_LIBRARY_PATH=$CU13/lib
export PYTHONPATH=/tmp/sglang_fqr/python CUDA_VISIBLE_DEVICES=0

# 1. 静态扫描：谁能用但没用
python scripts/fx_fusion/scan_qknorm_rope_candidates.py --src /tmp/sglang_fqr \
    --out results/fx_fusion/qknorm_rope_candidates.json

# 2. 跨模型可用性 + profiling（--profile 需要 GPU）
python scripts/fx_fusion/scan_models_pipeline.py --src /tmp/sglang_fqr_base --gpu 1 \
    --models gemma3_causal=/data/hf/models/gemma-3-1b-it \
             olmo2=/home/t-jialianggu/models/OLMo-2-0425-1B-Instruct \
    --profile --out results/fx_fusion/model_pipeline_profiled.json

# 3. 精度：对 fp64，不是对 bf16
python scripts/fx_fusion/accuracy_vs_fp64.py --out results/fx_fusion/accuracy_vs_fp64.csv
python scripts/fx_fusion/verify_against_model_path.py \
    --out results/fx_fusion/model_path_equivalence.json

# 4. 微基准三臂
python scripts/fx_fusion/verify_add_one_kernel.py --out results/fx_fusion/add_one_kernel.csv

# 5. 任务指标 + 配对检验
python scripts/fx_fusion/gsm8k_accuracy_gate.py \
    --baseline-tree /tmp/sglang_fqr_base --patched-tree /tmp/sglang_fqr -n 400
python scripts/fx_fusion/gsm8k_paired_test.py --out results/fx_fusion/gsm8k_paired.json

# 6. 端到端，带消融臂
python scripts/fx_fusion/e2e_ab_gemma3.py \
    --baseline-tree /tmp/sglang_fqr_base --patched-tree /tmp/sglang_fqr \
    --ablate --gpu 0 --reps 7 --out results/fx_fusion/e2e_ab_gemma3_ablation.json
```

**环境注意**：`import sglang` 需要 `CUDA_HOME` 指向
`site-packages/nvidia/cu13`，否则 deep_gemm 报 AssertionError。
用 `sgl.Engine` 的脚本必须有 `if __name__ == "__main__":` 守卫——
sglang 用 multiprocessing spawn 拉起 scheduler，子进程会重新 import 该文件，
没有守卫会再建一个 Engine 然后死掉，父进程只报
"scheduler died during initialization"，看不出真正原因。

```bash
# 7. OLMo-2 的第二个缺口（注意：fa3 + CUDA graph 在这个模型上本身就崩，
#    与本改动无关，用 triton backend）
export SGLANG_AB_BACKEND=triton
python scripts/fx_fusion/verify_generation_identical.py \
    --baseline-tree /tmp/sglang_fqr_base --patched-tree /tmp/sglang_olmo \
    --model /home/t-jialianggu/models/OLMo-2-0425-1B-Instruct --gpu 2
python scripts/fx_fusion/e2e_ab_gemma3.py \
    --baseline-tree /tmp/sglang_fqr_base --patched-tree /tmp/sglang_olmo \
    --model /home/t-jialianggu/models/OLMo-2-0425-1B-Instruct \
    --gpu 1 --reps 7 --out results/fx_fusion/e2e_ab_olmo2.json
```

**已知的既有问题（非本改动引入）**：OLMo-2 + fa3 + CUDA graph 在未打补丁的 main 上
就报 `cudaErrorIllegalAddress`。已用未打补丁的树复现确认。
本文所有 OLMo-2 数据都用 triton backend。
