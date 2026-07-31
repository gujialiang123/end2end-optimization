# 用 torch.compile / FX 自动发现 kernel fusion 机会 —— 实验结果

**日期**：2026-07-31（GPU 4/6）
**动机**：把优化方法从「CUDA + profiling + 人工读代码」迁移到**硬件无关**的路线上，
使其能用于 MAIA 等非 NVIDIA 后端（那边的部署也基于 torch.compile）。
**核心问题**：能否用 FX graph 代替人工发现融合机会？自动化能替代多少？

**所有数字均为本次实测**，原始数据在 `results/fx_fusion/`。

---

## 0. 四条结论

1. **可行性验证通过**：写了一个硬件无关的 FX 扫描器，
   它**独立重现了我们此前人工发现、并已验证 +36.6% 的 Gemma-3 案例**。
2. **torch.compile 拿到了大部分收益**：相比 eager，RMSNorm **11.8×**、SwiGLU **3.9×**（中位）。
3. **反直觉且重要**：**Inductor 生成的 kernel 中位比 sglang 手写的还快**
   （手写/compile 中位 0.80× 和 0.92×；21 个 RMSNorm 配置里手写只赢 1 个）。
4. **找到一个 Inductor 融不掉的真实模式**：QK-norm 的两个切片各自 norm 会生成 2 个 kernel，
   合并后在 decode 场景（小 token）可达 **1.96×**。这是自动化流程找到的新机会。

**对 MAIA 的含义：如果部署走 torch.compile，大部分融合收益是自动拿到的；
人工的价值应该转向「找编译器融不掉的模式」，而这件事可以被自动化检测。**

---

## 1. 工具：硬件无关的 FX 融合扫描器

`scripts/fx_fusion/fx_fusion_scanner.py`

**设计原则：不含任何 CUDA 假设。** 它读的是 `torch.compile` 产生的 post-grad FX 图，
那是任何走 Dynamo/AOTAutograd 的后端都有的 IR。**不看 kernel、不看设备计时**，
所以同一个 pass 可以跑在 CPU、CUDA 或厂商加速器上。

**找什么**：最长的「逐元素/规约算子」链，且**每个中间结果只有一个消费者**。

为什么单消费者是安全条件：如果中间结果有两个消费者，融合掉它就得重算或落盘，收益消失。
`num_users` 是 FX 节点自带的，**这个安全性检查是免费的**。

自测（一个标准 RMSNorm）：
```
by_signature: {"pow->mean->add->rsqrt->mul->mul->convert_element_type": 1}
```
完整识别出 7 个算子的链。

---

## 2. 可信度验证：自动化能否重现人工发现 ★

**这是整套方法最关键的一次检验**——用一个**已知答案**的案例。

在真实 `gemma-3-1b`（HF 模型，2 层，8 tokens）上跑扫描器：

```
graphs=1  chains=37  bytes_saved=10.4MB  wall=50.3s
  x13   convert_element_type->add
  x9    pow->mean->add->rsqrt->mul->mul->convert_element_type
  x4    mul->add
  x3    pow->mean->add->rsqrt->mul->mul->convert_element_type->add
  ...
```

把 RMSNorm 链按**归约维度**分类：

| 归约维度 | 条数 | 对应什么 |
|---:|---:|---|
| 1152 | 9 | `hidden_size` → input/post-attn layernorm |
| **256** | **4** | **`head_dim` → q_norm / k_norm** ★ |

**每层 2 条 dim=256 的链，正是 `q_norm` 和 `k_norm`** ——
也就是我们花了很久人工读代码才找到、并已端到端验证 **+36.6%** 的那个案例。

> **扫描器在没有任何先验知识的情况下，独立发现了它。**

这条验证的意义：证明「FX 自动发现」不是纸上谈兵，
而且我们**有 ground truth 可以衡量它**——这是别的团队多半没有的条件。

---

## 3. 跨模型扫描（5 个家族）

| 模型 | 层数 | 链数 | 估计可省 | 主要签名 |
|---|---|---:|---:|---|
| gemma3-1b | 2/26 | 37 | 10.4 MB | RMSNorm ×13 |
| olmoe-1b-7b | 2/16 | 26 | 9.2 MB | RMSNorm ×9 |
| granite-3.3-2b | 2/40 | 18 | 6.9 MB | RMSNorm ×5, SiLU ×2 |
| exaone4-1.2b | 2/30 | 18 | 5.3 MB | RMSNorm ×5 |
| olmo2-1b | 2/16 | 17 | 9.7 MB | RMSNorm ×5 |

（phi4-mini 与 falcon-h1 因 transformers 版本不兼容未跑通 —— 如实记录。）

**跨模型重复出现的签名**（按覆盖模型数）：

| 出现于 | 签名 | 是什么 |
|---:|---|---|
| 4 个模型 | `neg->exp->add->div->convert_element_type->mul` | **SiLU/SwiGLU**（`1/(1+e^-x)` 的展开） |
| 4 个模型 | `sin->mul->...` / `cos->mul->...` | **RoPE** |
| 3 个模型 | `pow->mean->add->rsqrt->mul->convert_element_type->mul` | **RMSNorm** |
| 4 个模型 | `mul->add` | 各种缩放/偏置 |

**这四类覆盖了几乎全部机会**，且都在 GEMM 之间的「胶水算子」上 —— 与我们此前的人工结论一致。

---

## 4. 三种 kernel 来源的正面对比 ★★

对同一算子，比较三条路径的**纯 GPU 时间**（用 profiler 排除 Python 开销）：

原始数据：`results/fx_fusion/kernel_source_comparison.csv`（33 个配置）

| 算子 | compile / eager（中位） | **手写 / compile（中位）** | 手写更快的配置 |
|---|---:|---:|---:|
| RMSNorm | **11.83×** | **0.80×** | **1 / 21** |
| SwiGLU | **3.94×** | **0.92×** | **0 / 12** |

**两个结论：**

**(a) torch.compile 相对 eager 的收益巨大**（11.8× / 3.9×）——
自动融合确实拿到了绝大部分空间。

**(b) 手写 kernel 并不总是更快，中位反而更慢。**

按归约维度扫描（T=4096 固定）：

| H | compile | 手写 | 手写/compile | 判定 |
|---:|---:|---:|---:|---|
| 256 | 3.7us | 6.7us | 0.56× | Inductor 更快 |
| 512 | 3.9us | 6.8us | 0.57× | Inductor 更快 |
| **1152** | 9.9us | 6.8us | **1.45×** | **手写更快** |
| 2048 | 10.5us | 9.5us | 1.10× | 持平 |
| 4096 | 20.3us | 22.5us | 0.90× | Inductor 更快 |
| 8192 | 33.3us | 49.1us | 0.68× | Inductor 更快 |
| 16384 | 64.9us | 83.7us | 0.78× | Inductor 更快 |

**手写 kernel 只在 H≈1152–2048 的窄窗口内占优，两端都是 Inductor 更快。**

> 注意 H=1152 恰好是 gemma-3 的 `hidden_size` —— 手写 kernel 是针对常见 shape 调过的，
> 出了那个窗口就不再有优势。

### 一次差点报错的数据（方法学记录）

第一次跑这个 sweep 得到「手写/compile 中位 **7.27×**」，且 `compile/eager = 1.00×`。
**这是假的** —— 日志里有：
```
[1/8] last reason: tensor 'x' size mismatch at index 0. expected 512, actual 4096
```
Dynamo 因反复 recompile 触及 cache 上限，**直接放弃编译退回 eager**，
所以「compile」臂测的其实是 eager。

修法：每个 shape 前 `torch._dynamo.reset()`，让它独立编译。修正后中位从 7.27× 变成 **0.80×** ——
**结论完全反转**。

**教训：`compile/eager ≈ 1.00` 是「编译没生效」的信号，必须当作错误来查。**

---

## 5. 阶段 2 实证：静态可融合 ≠ 实际未融合 ★

我们的四阶段漏斗里，阶段 2 是「执行确认」。这里给出它的直接证据。

把 FX 找到的链数，和 Inductor **实际生成的 kernel 数**对比：

| 测试形态 | FX 找到的链 | 实际生成 kernel | 结论 |
|---|---:|---:|---|
| 单个 RMSNorm | 1 | **1** | 已被融合 |
| norm → matmul → norm | 2 | 4（含 matmul） | 已被融合 |
| **residual add + norm** | 1 | **1** | **已被融合** |
| 多消费者中间结果 | — | **1** | 仍被融合（比预期强） |
| 两个独立 norm | — | **1** | **横向融合了** |

**关键发现：我们此前人工做的那些融合，Inductor 全都自动做到了** ——
包括案例 3（residual + norm），那正是我们在 LFM2.5 上手工实现的 deferred residual。

> **所以「FX 报告一条未融合的链」不等于「有机会」**。
> 必须再问一句：编译器是不是已经融了？这就是阶段 2 的价值，
> 而且它可以被自动检查（数生成的 kernel 数）。

---

## 6. 新发现：Inductor 融不掉的一个真实模式 ★★

在探测「什么融不掉」时，找到一个：

```
QK-norm: 同一个 qkv 张量的两个切片各自 norm
  -> triton_per_fused_..._slice_0
  -> triton_per_fused_..._slice_1        ← 2 个 kernel
```

对比：**两个完全独立的 norm 反而能横向融成 1 个 kernel**。
**切片阻止了融合** —— 这是编译器的一个真实局限。

而这正是**真实模型的形态**（gemma-3、qwen3、olmo-2 的 QK-norm 都从 qkv 切片而来）。

### 语义等价的合并方式

第一版测试里 q/k 用了同一个权重，那不是真实情况。真实模型里 q_norm 和 k_norm
**权重不同**，不能直接当成一个大 batch。

但可以做等价变换：两者都沿 `head_dim` 规约，所以把张量看成
`[tokens, heads, head_dim]`、权重按 head 展开成 `[heads, head_dim]`，
**一次 norm 即可，且逐元素等价**：

```python
wcat = torch.cat([wq.repeat(QH,1), wk.repeat(KH,1)], 0)   # [QH+KH, HD]
o = v.reshape(-1, QH+KH, HD).float()
out = (o * torch.rsqrt(o.pow(2).mean(-1,keepdim=True)+1e-6) * wcat.float()).type_as(v)
```

### 量化结果（`head_dim=256`，4 个 q 头 + 1 个 kv 头，bf16）

原始数据：`results/fx_fusion/qknorm_merge.csv`
复现：`python scripts/fx_fusion/verify_qknorm_merge.py`

| tokens | 切片版（2 kernel） | 合并版（1 kernel） | 加速 | 数值等价 |
|---:|---:|---:|---:|:--:|
| 8 | 2.03 us | 1.02 us | **1.98×** | ✅ |
| 32 | 2.19 us | 1.09 us | **2.00×** | ✅ |
| 64 | 2.30 us | 1.27 us | **1.81×** | ✅ |
| 128 | 2.58 us | 1.52 us | **1.70×** | ✅ |
| 512 | 3.76 us | 2.78 us | 1.35× | ✅ |
| 2048 | 8.60 us | 7.46 us | 1.15× | ✅ |
| 4096 | 7.66 us | 13.62 us | **0.56×**（更慢） | ✅ |

**7 个 token 数全部数值等价**（`allclose`，非假设）。
**6/7 更快，最高 2.00×；但 4096 tokens 时反而慢 44%。**

**规律**：小 token 数（= **decode**）收益最大，大 token 数（= **prefill**）会变慢 ——
所以这应该是一个 **shape 相关的分派**，不是无条件替换。

诚实边界：这是 micro-benchmark，**还没有做端到端 A/B**（四阶段漏斗的阶段 4）。
真实收益取决于 QK-norm 在整个 forward 里的占比。

---

### 这个模式有多普遍

不是孤例。sglang 里 **21 个模型文件**都是这个形态（`self.q_norm(q)` 和 `self.k_norm(k)` 分两次调用）：

```
grep -rln "self.q_norm(" python/sglang/srt/models/*.py | wc -l   ->  21
```

gemma-3、qwen3、olmo-2 等主流模型都在内。

### 用 sglang 真实融合 kernel 复测

不只是 Inductor 生成的 kernel 有这个现象，**手写 kernel 路径同样存在**：

原始数据：`results/fx_fusion/qknorm_sglang_kernel.csv`

| tokens | 两次 `gemma_rmsnorm` | 一次调用 | 加速 |
|---:|---:|---:|---:|
| **1** | 2.90 us | 1.44 us | **2.02×** |
| 8 | 2.95 us | 1.51 us | **1.96×** |
| 32 | 3.18 us | 1.71 us | **1.86×** |
| 128 | 3.79 us | 2.33 us | 1.62× |
| 512 | 6.29 us | 4.78 us | 1.31× |
| 2048 | 15.82 us | 14.25 us | 1.11× |

**T=1（最典型的 decode）加速 2.02×。**

### ★ 两条路线在这里分叉，这点对 MAIA 论证最关键

真实模型里 q_norm 和 k_norm 的**权重是不同的参数**，所以不能直接拼起来调一次。

| 路线 | 能否表达这个融合 | 为什么 |
|---|---|---|
| **torch.compile / FX** | ✅ **今天就能做** | per-head 权重广播即可，**已验证 7/7 数值等价** |
| **sglang 手写 kernel** | ❌ **需要新 kernel** | `gemma_rmsnorm(input, weight, ...)` 的 `weight` 是单个 1-D 张量，不支持 per-head |

```python
# torch.compile 路线：直接可写，无需新 kernel
wcat = torch.cat([wq.repeat(QH,1), wk.repeat(KH,1)], 0)     # [heads, head_dim]
o = v.reshape(-1, QH+KH, HD).float()
out = (o * torch.rsqrt(o.pow(2).mean(-1,keepdim=True)+1e-6) * wcat.float()).type_as(v)
```

**这正是编译器路线优于手写 kernel 路线的一个具体实例**：
手写 kernel 的接口是固定的（一个 weight 向量），
而编译器从图出发，能为任意权重布局现场生成代码。

**对 MAIA 的直接含义**：这类"手写 kernel 接口表达不了、但编译器能表达"的融合，
在 torch.compile 部署上是**免费的**，不需要为每种权重布局写一个 kernel 变体。

---

## 6b. FX 的排序能替代 profiling 的排序吗？（部分能，但要选对指标）

我们有 11 个模型的人工 profiling 审计基线，其中 **7 个**也做了 FX 扫描，
可以直接检验：**FX 给出的排序，和 profiling 实测的机会大小，一致吗？**

原始数据：`results/fx_fusion/fx_vs_audit_correlation.csv`

| model | FX 链数/层 | FX MB/层 | 审计 removable% |
|---|---:|---:|---:|
| gemma3-1b | 18.5 | 5.21 | **37.06%** |
| olmo2-1b | 8.5 | 4.83 | 14.71% |
| lfm25-8b-a1b | 4.0 | 2.92 | 4.06% |
| exaone4-1.2b | 9.0 | 2.67 | 3.54% |
| qwen3-0.6b | 9.5 | 1.92 | 0.46% |
| olmoe-1b-7b | 13.0 | 4.61 | 0.43% |
| granite-3.3-2b | 9.0 | 3.43 | 0.23% |

**Spearman 秩相关（n=7）：**

| FX 指标 | 与审计 removable% |
|---|---:|
| **链数 / 层** | **−0.11**（无关） |
| **估计字节数 / 层** | **+0.46**（弱正相关） |

### 结论：方向清楚，但强度不足以单独使用

**「找到几条链」完全不预测真实机会**（ρ=−0.11）。
olmoe 有 13 条/层（第二多），审计只有 0.43%；granite 9 条/层，0.23%。
**链多不代表热。**

**「估计字节数」明显更好（+0.46），但也只是弱正相关。**

> 注：n=5 时这个值是 +0.70，加到 n=7 后降到 +0.46。
> **样本增加后相关性减弱，说明原来的 +0.70 有偶然성** —— 如实记录，不取更好看的那个。

反例很清楚：**olmoe 的 FX 字节数排第 3（4.61 MB/层），审计却排倒数第 2（0.43%）**。
原因是 MoE 模型的 FX 图里有大量专家相关的逐元素算子，但它们**在实际执行中占比很低**
（每个 token 只走 top-k 个专家）。**静态图看不出这一点。**

### 给 agent 的直接建议

1. **阶段 1 排序要用字节数，不要用链数**（−0.11 vs +0.46）
2. **但 FX 排序不能单独用来决定优先级** —— ρ=+0.46 意味着仍会严重排错
3. **所以阶段 3（profiling）不能省**，FX 只负责把候选池从"所有代码"缩小到"结构上可疑的地方"

这恰好支持我们的四阶段设计：**FX 做廉价的高召回筛选，profiling 做昂贵但准确的排序。**

---

## 7. 这套方法对 MAIA / 非 CUDA 后端的价值

| 步骤 | 是否依赖 CUDA | 能否迁移 |
|---|---|---|
| FX 图导出 | ❌ 纯 PyTorch | ✅ |
| 链检测 + 单消费者安全性 | ❌ 纯图分析 | ✅ |
| 「编译器是否已融合」检查 | 数生成的 kernel 数 | ✅（换后端计数方式） |
| 性能量化 | 需要该后端的 profiler | ⚠️ 需适配 |
| 端到端 A/B | 需要该后端的 runtime | ⚠️ 需适配 |

**前三步完全可迁移**，而它们正是「发现」环节的全部。
后两步是「验证」环节，任何后端都得有自己的一套。

**具体建议**：
1. 在 MAIA 上跑同一个扫描器，得到候选列表 —— **零改动**
2. 对比 MAIA 后端实际生成的 kernel 数，找出「编译器融不掉的」
3. 那些就是值得手写 kernel 的地方 —— **而不是把所有融合都手写一遍**

---

## 8. 复现

```bash
cd scripts/fx_fusion

# 自测（几秒）
python fx_fusion_scanner.py --selftest

# 扫一个真实模型
python fx_scan_models.py --model gemma3-1b --layers 2 --tokens 8 --out /tmp/g3.json

# 三种 kernel 来源对比（注意每个 shape 前要 dynamo.reset）
# 见 results/fx_fusion/kernel_source_comparison.csv
```

**环境**：H200，torch 2.11.0+cu130，Triton 3.6.0，bf16。
（RMSNorm/SwiGLU 手写 kernel 来自 `sgl_kernel`。）

## 9. 产物

| 文件 | 内容 |
|---|---|
| `scripts/fx_fusion/fx_fusion_scanner.py` | 硬件无关的 FX 融合链扫描器 |
| `scripts/fx_fusion/fx_scan_models.py` | 在真实 HF 模型上运行扫描 |
| `results/fx_fusion/model_scans/*.json` | 5 个模型的完整扫描结果 |
| `results/fx_fusion/kernel_source_comparison.csv` | 33 个配置的三方对比 |

---

## 10. 还没做的（诚实的 backlog）

- **QK-norm 合并的端到端验证**：目前只有 micro-benchmark 的上界，
  还没按四阶段漏斗做完整 A/B。这是下一步最该做的。
- **phi4-mini / falcon-h1 扫描失败**：transformers 版本不兼容，未解决。
- **召回率量化**：还没把扫描结果和 11 模型人工审计逐一对比。
- **MAIA 上的实际验证**：本文只论证了可迁移性，没有真机数据。
