# 跨架构 fusion 空缺审计 —— 我的假设被自己的数据推翻了，修正后的结论更有用

**日期**：2026-07-27 · **硬件**：1× H200 · BF16 · sglang 0.5.12.post1 @ `17f7a1da1`
**脚本**：`scripts/lfm_fusion/lf_audit.py` · **数据**：`results/lfm_fusion/audit/`

---

## 0. 一句话结论

把算子级审计扩展到 **5 个架构**后：

1. **我原来的假设"融合空缺是架构成熟度的函数"被推翻了**。真正的预测因子不是架构新旧，而是**这个模型文件受到过多少优化关注**——而它跟随的是**模型家族在框架用户群里的地位**。
2. 途中在 **Gemma-3** 上发现了一个远大于 LFM2.5 的空缺：它的 RMSNorm 在 CUDA 上跑 **eager PyTorch**。修复是**一行 fall-through**，端到端 **2.07× / 1.75× / 1.57×**。

---

## 1. 假设与测试设计

上一轮在 LFM2.5 上的结论是：v33 那句"sglang 热路径已全部融合"只在 Qwen 上成立，**覆盖空缺是"架构成熟度"的属性**。但那是**单模型观察**，不足以称为规律。

测试集沿两个轴展开（dense/MoE × 成熟/新）：

| 模型 | 架构 | 层数 | 上游成熟度 | 预测 |
|---|---|---:|---|---|
| Qwen3-0.6B | dense，llama 式 | 28 | 非常成熟 | 干净 |
| Qwen3-30B-A3B | MoE + 全注意力 | 48 | 成熟 | 干净（已验证） |
| Gemma-3-1B | dense + 滑窗注意力 | 26 | 中等 | ? |
| Qwen3-Coder-Next | MoE(512E) + GDN 线性注意力 | 48 | **新** | **有空缺**（按假设） |
| LFM2.5-8B-A1B | MoE + gated short conv | 24 | 新 | 有空缺（已验证） |

**关键改进**：审计新增**按层归一化**（`calls_per_layer`）。空缺是结构性的，层数多的模型自然发出更多杂散 kernel，**假设是关于每层速率的**。

> **未能纳入的**：`gemma-4-26B-A4B`（这版 sglang 只支持到 gemma3n，无法加载）。
> `Qwen3-Coder-Next` 已用 **TP2（GPU 4+5）** 补测 —— 它是本次的**关键判决实验**。

---

## 2. 审计结果

### 2.1 结构性空缺（低批 decode，每层 kernel 启动次数 + 占 kernel 时间比例）

| 模型 | 家族 | 成熟度 | 层数 | norm/层 | add/层 | gmul/层 | eager/层 | **空缺合计** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Gemma-3-1B** | Google | 成熟 | 26 | 0.00 | 2.00 | 12.08 | **6.04** | **46.32%** |
| **LFM2.5-8B-A1B** | Liquid | 新 | 24 | 2.54 | 2.00 | 1.50 | 0.00 | **11.31%** |
| Qwen3-Coder-Next | Qwen | **新** | 48 | 0.52 | 1.00 | 1.25 | 0.00 | **0.64%** |
| Qwen3-0.6B | Qwen | 非常成熟 | 28 | 0.04 | 0.00 | 0.00 | 0.00 | **0.57%** |
| Qwen3-30B-A3B | Qwen | 成熟 | 48 | 0.02 | 0.00 | 0.00 | 0.00 | **0.23%** |

图：`results/lfm_fusion/plots/cross_architecture_gaps.png`

### 2.2 eager norm 分解占 CUDA kernel 时间的比例

| 模型 | 低批 decode | decode（长 prefill 负载） | prefill（短） | prefill（T=16000） |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen3-30B-A3B | 0.00% | 0.00% | 0.00% | 0.00% |
| LFM2.5-8B-A1B | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen3-Coder-Next | 0.00% | — | 0.00% | — |
| **Gemma-3-1B** | **15.98%** | **16.35%** | **19.31%** | **11.07%** |

**信号极其干净**：一个模型 11–19%，其余全部**精确 0.00%**，调用次数 0。

> 注：这 11–19% **只统计了 mean/rsqrt/pow 三步**，不含乘法和 fp32 上下转型，所以真实成本更高。

### 2.3 假设的判决：**被推翻**

我的假设是"架构越新 → 空缺越多"。数据说不是：

- **Qwen3-Coder-Next 是这批里最新的架构之一**（GDN 线性注意力 + 512 专家），却**几乎干净（0.64%）**
- **Gemma-3 是成熟、广泛使用的模型**，却是**最差的（46.32%）**
- Gemma-3 的空缺比 Qwen3-Next **大 72 倍**，尽管它更"老"

**真正的分界线是模型家族**：

| | 空缺合计 |
|---|---|
| **三个 Qwen 模型**（dense / MoE / 线性注意力，横跨成熟度） | 0.23% · 0.57% · 0.64% |
| **两个非 Qwen 模型** | 11.31% · 46.32% |

Qwen 家族在 **dense、MoE、线性注意力三种架构上、横跨"非常成熟"到"新"**，全部干净。这排除了"架构类型"和"架构年龄"两个解释。

> **修正后的结论**：预测因子不是架构新旧，而是**这个模型文件受到过多少优化关注**——而它跟随的是**模型家族在框架用户群里的地位**，不是架构的新颖度。SGLang 被大量用于 Qwen，Qwen 团队和 SGLang 团队直接优化它们；Gemma-3 和 LFM2.5 是"支持"但被跑得少的路径。

这比原假设**更有操作性**：它说的是"**去查那些不是框架主力用户的模型家族**"，而不是"去查新架构"。

---

## 3. Gemma-3 的空缺：一行 fall-through

### 3.1 现象

审计里 gemma3 的 top kernel 呈现这个模式（各 157 次）：

```
reduce_kernel<MeanOps>      n=157     ← mean(x²)
CUDAFunctorOnSelf_add       n=314     ← +eps
rsqrt_kernel_cuda           n=157     ← rsqrt
pow_tensor_scalar           n=157     ← x²
BinaryFunctor (mul)         n=131     ← 乘 weight
direct_copy_kernel          n=317     ← fp32 上下转型
```

这是 **RMSNorm 被展开成原始 PyTorch 算子**：`x * rsqrt(mean(x²) + eps)`。

`26 层 × 6 个 norm（input / post_attention / pre_feedforward / post_feedforward / q_norm / k_norm）= 156 ≈ 157` ✓

### 3.2 根因

`sglang/srt/layers/layernorm.py` 里**相隔约 100 行**定义了两个 Gemma norm 类：

```python
class GemmaRMSNorm(MultiPlatformOp):          # ~line 402
    def _forward_impl(self, x, residual=None, ...):
        if residual is not None:
            gemma_fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        return gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)   # ← 融合 CUDA kernel

class Gemma3RMSNorm(MultiPlatformOp):         # ~line 505
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward_cpu(self, x):
        if _is_cpu_amx_available and x.stride(-1) == 1:
            return torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, self.weight, self.eps)   # ← CPU 有融合
        return self.forward_native(x)
    def forward_cuda(self, x):
        return self.forward_native(x)          # ← CUDA 直接掉进 eager
    def forward_npu(self, x):
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)   # ← NPU 有融合
```

**CPU 有融合 kernel，NPU 有融合 kernel，唯独 CUDA 掉进 eager。** 而 `sgl_kernel` 里 `gemma_rmsnorm` 和 `gemma_fused_add_rmsnorm` 都是**预编译好的**。

> 这是我们那条**静态 signature**（"枚举代码库已有的融合原语，找出没调用它们的调用点"）的**第三个独立实例，也是最大的一个**。而且这次融合原语不只是"在代码库某处"，而是**在同一个文件里、相隔 100 行**。

### 3.3 修复

`scripts/lfm_fusion/gemma_fusion_patch.py`，`GEMMA_FUSION_PATCH=norm` opt-in：

```python
def _patched_gemma3_forward_cuda(self, x):
    if x.dtype not in (torch.bfloat16, torch.float16):
        return self.forward_native(x)
    if x.dim() == 2:
        return gemma_rmsnorm(x.contiguous(), _fused_weight(self, x), self.eps)
    if x.dim() > 2 and x.shape[-1] == self.weight.numel():
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        return gemma_rmsnorm(flat, _fused_weight(self, x), self.eps).view_as(x)
    return self.forward_native(x)
```

**两个只有靠实测才发现的细节**：

1. **dtype 陷阱**：`Gemma3RMSNorm.weight` 由 `nn.Parameter(torch.zeros(dim))` 创建，是 **fp32**，而激活是 bf16。给融合 kernel 传 fp32 weight 会**静默产生 NaN**（我第一次测试就撞上了）。所以按 module 缓存一次转型，且**用 dtype 相等而非 `hasattr` 做守卫**，防止模块被重新转型后返回陈旧缓存。
2. **3-D 陷阱**：`q_norm`/`k_norm` 的输入是 `[tokens, heads, head_dim]`。最初的 rank-2 保护把它们挡回了 eager —— **每层 6 个 norm 里有 2 个没修到**。是**打上补丁后重新审计**才发现的（残留 52 次 = 2.00/层）。RMSNorm 只沿最后一维归约，所以摊平成 2-D 再还原是精确的。

`MultiPlatformOp` 在 `__init__` 里绑定 `_forward_method`，所以除了替换类方法还必须**同时 patch 构造函数**，否则已构造的模块不会生效。

### 3.4 机械闭环验证

打上补丁后**重新跑审计**：

| 版本 | eager norm 调用 | 融合 norm 调用 | decode kernel 总时间 |
|---|---:|---:|---:|
| baseline | **157** | 0 | 3.81 ms |
| 补丁 v1（rank-2 守卫） | 52（2.00/层） | 105 | 2.40 ms（−37%） |
| **补丁 v2（含 3-D）** | **0** | **157（6.04/层）** | **1.89 ms（−50.4%）** |

审计 signature 从 157 归零 —— **完全闭环**。

---

## 4. 端到端结果

`lf_e2e.py`，6 次重复/臂，精确 Student-t 的 Welch 检验。两臂 resolved 配置逐项核验一致（`moe_backend=auto`、`attn=fa3`、`cuda_graph=True`），唯一差异是 norm 实现。

| regime | baseline req/s | patched req/s | **加速** | p |
|---|---:|---:|---:|---:|
| A 低批 decode | 0.808 ± 0.006 | 1.669 ± 0.044 | **2.065× (+106.5%)** | 1.4e-07 |
| B 并发 decode | 20.320 ± 0.541 | 35.648 ± 2.143 | **1.754× (+75.4%)** | 1.6e-05 |
| C 长 prefill | 16.718 ± 0.197 | 26.200 ± 1.016 | **1.567× (+56.7%)** | 5.2e-06 |

**质量门禁**（GSM8K 全量 1319 题 × 3 次）：baseline **0.2233**、patched **0.2210**。差 0.23 个点，而 n=1319、p≈0.22 的二项抽样误差是 **±2.2 点** → 在噪声内。

**为什么这么大**：gemma-3-1b 只有 1B 参数，decode 阶段**严重 launch-bound**。157 个 norm × ~6 kernel ≈ **每 forward 940 个 kernel 只为做归一化**；换成融合版是 157 个，**净减约 780 次 kernel 启动**。模型越小、batch 越小，这个比例越大——这解释了 2.07× → 1.75× → 1.57× 的排序。

---

## 5. 与 LFM2.5 工作的对比

| | LFM2.5（7 个组件） | **Gemma-3（1 个组件）** |
|---|---|---|
| 改动量 | 2 个手写 Triton kernel + 5 处调用点 | **1 处调用点（约 10 行）** |
| 端到端 | +4.7 ~ +6.6% | **+56.7 ~ +106.5%** |
| 空缺性质 | 新算子周围的胶水 | **整个 norm 实现掉进 eager** |
| 上游是否算 bug | 否（只是没人填） | **是**（CPU/NPU 有融合，CUDA 掉队） |

> **一个诚实的自我评价**：我在 LFM2.5 上花了大部分精力手写 Triton kernel，拿到约 6%。而**扩展审计到第四个架构、用一行 fall-through 修复，拿到 2×**。这印证了本项目反复得到的教训——**杠杆在"找对地方"，不在"写得多精巧"**。

---

## 6. 诚实范围与缺口

- `Qwen3-Coder-Next` 用 **TP2** 测量，而其余四个是 TP1。TP2 会引入 allreduce kernel 并改变时间占比。**空缺计数是结构性的（每层次数），不受此影响**，但百分比口径与其余模型不完全可比 —— 结论依赖的是计数（eager/层 = 0），这一点是稳的。
- `gemma-4-26B-A4B` 这版 sglang 不支持，无法测。
- 样本 **5 个架构、3 个家族**、1 张卡。"家族决定空缺"有 5 个点支持，但**只有 2 个非 Qwen 家族** —— 这是最弱的一环，应该再加几个非 Qwen 模型。
- Gemma-3 的修复**非 bit-exact**（1–2 个 bf16 ulp），因为 eager 路径全程 fp32 而融合 kernel 在激活 dtype 下做 weight 乘法。**这正是 sglang 已经为 gemma/gemma2 接受的同一权衡**。质量结论依赖任务指标，不是 token 一致性。
- gemma-3-1b 只有 1B，**这个量级的收益不会等比例出现在大模型上**（大模型每 forward 的计算多得多，固定 launch 开销占比小）。gemma-3 的更大版本值得单独测。

---

## 7. 结论

1. **原假设被推翻**。"架构越新空缺越多"不成立：最新的 Qwen3-Next 几乎干净（0.64%），成熟的 Gemma-3 最差（46.32%）。**真正的分界是模型家族**——三个 Qwen 模型横跨 dense/MoE/线性注意力、横跨成熟度全部干净，两个非 Qwen 模型都有空缺。预测因子是"**这个模型文件受到过多少优化关注**"，跟随的是家族在用户群里的地位。这比原假设更有操作性：**去查非主力家族的模型**，而不是查新架构。
2. **静态 signature 再次奏效，且是最高杠杆的工具**："枚举已有融合原语 → 找没调用的调用点"，**不需要 profiling**，这一条已经找到了本项目最大的三个赢家（LFM2.5 的 `norm`、`qkrope`，和 Gemma-3 的这个）。
3. **Gemma-3 的 CUDA fall-through 是真正的上游 bug**，值得提 issue/PR：CPU 和 NPU 都有融合路径，CUDA 独缺，而所需 kernel 已经预编译在 `sgl_kernel` 里。

## 8. 复现

```bash
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$ENVDIR PATH=$ENVDIR/bin:$PATH HF_HOME=$PWD/.hf_cache

# 审计任一模型
python scripts/lfm_fusion/lf_audit.py --model gemma3 --regime A_low_batch_decode --gpu 5
python scripts/lfm_fusion/lf_audit.py --model qwen06  --regime A_low_batch_decode --gpu 5

# 打补丁后重新审计（机械闭环）
GEMMA_FUSION_PATCH=norm \
PYTHONPATH=$PWD/scripts/lfm_fusion/gm_inject:$PWD/scripts/lfm_fusion \
python scripts/lfm_fusion/lf_audit.py --model gemma3 --regime A_low_batch_decode \
    --gpu 5 --tag _patched

# 端到端 + 质量门禁
python scripts/lfm_fusion/lf_e2e.py --model gemma3 --regime A_low_batch_decode \
    --gpu 4 --arms baseline,gemma_norm --reps 6 --tag _gemma2
python scripts/lfm_fusion/lf_analyze.py --runset gemma3_gemma2 --out fusion_ab_gemma2.csv
python scripts/lfm_fusion/lf_correctness.py accuracy --arm gemma_norm --model gemma3 \
    --gpu 5 --num-questions 1319 --reps 3

# 部署
GEMMA_FUSION_PATCH=norm \
PYTHONPATH=$PWD/scripts/lfm_fusion/gm_inject:$PWD/scripts/lfm_fusion \
python -m sglang.launch_server --model-path /data/hf/models/gemma-3-1b-it ...
```
