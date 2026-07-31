# Final project：可迁移的 kernel fusion 自动发现

**日期**：2026-07-31 · **硬件**：H200 ×2（GPU 4/6）
**一句话**：把「找 kernel 融合机会」从 CUDA + profiling + 人工读代码，
迁移到**基于 torch.compile / FX 的硬件无关流程**，并用已验证的案例证明它有效。

**为什么这个方向**：MAIA 的部署基于 torch.compile。
任何绑定 CUDA profiler 的方法都无法迁移；而 FX 图分析是纯 PyTorch 层的，**换后端零改动**。

---

## 1. 交付物

| 产物 | 说明 |
|---|---|
| `scripts/fx_fusion/fx_fusion_scanner.py` | 硬件无关的融合链扫描器（不含任何 CUDA 假设） |
| `scripts/fx_fusion/fx_scan_models.py` | 在真实 HF 模型上运行扫描 |
| `scripts/fx_fusion/verify_qknorm_merge.py` | 新发现的可复现验证脚本 |
| `results/fx_fusion/` | 5 个模型扫描 + 33 配置对比 + 2 组验证数据 |

---

## 2. 四个结果

### 结果一：自动化能重现人工发现（可信度验证）★

**方法学上这一步最重要**：用一个**已知答案**的案例检验自动化。

在真实 `gemma-3-1b` 上跑扫描器，无任何先验知识。按归约维度分类结果：

| 归约维度 | 条数 | 是什么 |
|---:|---:|---|
| 1152 | 9 | `hidden_size` → 常规 layernorm |
| **256** | **4** | **`head_dim` → q_norm / k_norm** |

每层 2 条 dim=256 的链 —— **正是我们此前人工找到、并已端到端验证 +36.6% 的那个案例**。

> 扫描器独立发现了它。这证明 FX 路线不是纸上谈兵。

### 结果二：torch.compile 拿到了大部分收益

33 个配置的三方对比（纯 GPU 时间，profiler 测量）：

| 算子 | compile / eager | **手写 / compile** | 手写更快的配置 |
|---|---:|---:|---:|
| RMSNorm | **11.83×** | 0.80× | **1 / 21** |
| SwiGLU | **3.94×** | 0.92× | **0 / 12** |

**两个结论**：
- 自动融合相对 eager 收益巨大（11.8× / 3.9×）
- **手写 kernel 中位反而更慢**，只在 H≈1152–2048 的窄窗口内占优（那恰是它被调优的 shape）

**对 MAIA 的含义**：走 torch.compile 部署，**大部分融合收益是自动拿到的**。
人工的价值应转向「找编译器融不掉的」，而不是把所有融合都手写一遍。

### 结果三：静态可融合 ≠ 实际有机会（阶段 2 的实证）

把 FX 找到的链数与 Inductor **实际生成的 kernel 数**对比：

| 形态 | FX 链 | 实际 kernel | 结论 |
|---|---:|---:|---|
| 单个 RMSNorm | 1 | 1 | 已融合 |
| **residual add + norm** | 1 | **1** | **已融合** |
| 多消费者中间结果 | — | 1 | 已融合（比预期强） |
| 两个独立 norm | — | **1** | **横向融合了** |

**我们此前人工做的融合，Inductor 全都自动做到了** —— 包括 LFM2.5 的 deferred residual。

> 所以「FX 报告一条未融合的链」**不等于**「有机会」。
> 必须再问：编译器是不是已经融了？**而这可以自动检查（数 kernel 数）。**

### 结果四：找到一个编译器也融不掉的新机会 ★★

探测「什么融不掉」时发现：

```
QK-norm: 同一 qkv 的两个切片各自 norm
  -> triton_per_fused_..._slice_0
  -> triton_per_fused_..._slice_1      ← 2 个 kernel
```

对照：**两个完全独立的 norm 反而能横向融成 1 个**。**是切片阻止了融合。**

**等价重写**（两者都沿 `head_dim` 规约，可合并为 `[tokens, heads, head_dim]` 上的一次 norm）：

| tokens | 切片版 | 合并版 | 加速 | 数值等价 |
|---:|---:|---:|---:|:--:|
| 8 | 2.03 us | 1.02 us | **1.98×** | ✅ |
| 32 | 2.19 us | 1.09 us | **2.00×** | ✅ |
| 128 | 2.58 us | 1.52 us | 1.70× | ✅ |
| 512 | 3.76 us | 2.78 us | 1.35× | ✅ |
| 4096 | 7.66 us | 13.62 us | **0.56×** | ✅ |

**7/7 数值等价**（`allclose` 实测），**6/7 更快，最高 2.00×**；
但 4096 tokens 时反而慢 —— 所以应该是 **shape 相关的分派**，不是无条件替换。

**普遍性**：sglang 里 **21 个模型文件**都是这个形态（gemma-3、qwen3、olmo-2 等）。

用 sglang 真实融合 kernel 复测，同样存在：**T=1 时 2.02×**。

---

## 3. 最有说服力的一点：编译器能表达而手写 kernel 不能

真实模型里 q_norm 和 k_norm 的**权重是不同的参数**，不能直接拼起来。

| 路线 | 能否表达 | 原因 |
|---|---|---|
| **torch.compile / FX** | ✅ **今天就能** | per-head 权重广播，已验证等价 |
| **手写 kernel** | ❌ **要写新 kernel** | `gemma_rmsnorm(input, weight,...)` 的 weight 是单个 1-D 张量 |

```python
# 编译器路线：直接可写
wcat = torch.cat([wq.repeat(QH,1), wk.repeat(KH,1)], 0)
o = v.reshape(-1, QH+KH, HD).float()
out = (o * torch.rsqrt(o.pow(2).mean(-1,keepdim=True)+1e-6) * wcat.float()).type_as(v)
```

> **kernel 的签名在编写时就固定了；编译器则为图里出现的任意布局现场生成代码。**
> 这类融合在 torch.compile 部署上是免费的，不需要为每种权重布局维护一个 kernel 变体。

---

## 4. 方法：四阶段漏斗

```
阶段1 静态扫描 (秒级, CPU)     FX 找链 + 签名扫描        高召回, 允许误报
   ↓
阶段2 执行确认 ★               编译器融了吗? 代码活着吗?   ← 最易被跳过
   ↓
阶段3 profiling (分钟级, GPU)  占多少 kernel 时间         排优先级, 是上界
   ↓
阶段4 端到端 A/B               多次重复 + 统计检验         唯一算数的证据
```

**阶段 2 是我们用两次失败换来的**：
- OLMo-2：审计报 7.71%，实测 +0.45%（profiling 关了 CUDA graph）
- Triton 3.6：得出错误结论（基线加载了我们自己的 config）

**通用形式：别信声明，去查执行产物。**

---

## 5. 迁移到 MAIA 的路径

| 步骤 | 依赖 CUDA？ | 迁移成本 |
|---|---|---|
| FX 图导出 | ❌ 纯 PyTorch | **零** |
| 链检测 + 单消费者安全性 | ❌ 纯图分析 | **零** |
| 「编译器已融合吗」检查 | 数 kernel 数 | 换计数方式 |
| 性能量化 | 需要后端 profiler | 需适配 |
| 端到端 A/B | 需要后端 runtime | 需适配 |

**前三步 —— 也就是「发现」的全部 —— 完全可迁移。**

**建议**：
1. 在 MAIA 上跑同一个扫描器拿候选（零改动）
2. 对比 MAIA 后端实际生成的 kernel，找出**它融不掉的**
3. 只为那些写手写 kernel

---

## 6. 诚实的边界

- **QK-norm 只有 micro-benchmark**，还没做端到端 A/B（阶段 4）。真实收益取决于它在 forward 里的占比。
- **phi4-mini / falcon-h1 扫描失败**（transformers 版本不兼容），未解决。
- **排序指标已验证**（n=7）：FX「链数」与人工审计**无关**（ρ=−0.11），
  「估计字节数」+0.46（弱正相关）。**排序要用字节数，但 FX 不能单独定优先级** ——
  这支持四阶段设计里 profiling 不可省。（n=5 时是 +0.70，加样本后降到 +0.46，如实记录。）
- **无 MAIA 真机数据**：本文只论证可迁移性。

### 一次差点报错的数据（方法学记录）

三方对比第一次跑出「手写快 7.27×」，且 `compile/eager = 1.00`。
**那是假的** —— Dynamo 因反复 recompile 触及 cache 上限，静默退回 eager，
所以「compile」臂测的其实是 eager。每个 shape 前 `dynamo.reset()` 后，
中位从 7.27× 变成 **0.80×**，**结论完全反转**。

> **`compile/eager ≈ 1.00` 是「编译没生效」的信号，必须当错误查。**

---

## 7. 复现

```bash
cd scripts/fx_fusion
python fx_fusion_scanner.py --selftest                                    # 几秒
python fx_scan_models.py --model gemma3-1b --layers 2 --tokens 8 --out /tmp/g3.json
python verify_qknorm_merge.py --out /tmp/qk.csv                           # 新发现的验证
```

**环境**：H200，torch 2.11.0+cu130，Triton 3.6.0，bf16。
