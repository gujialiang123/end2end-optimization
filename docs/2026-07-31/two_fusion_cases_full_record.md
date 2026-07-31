# 两个 kernel fusion case 的完整记录：怎么发现的、怎么验证的、以及什么能自动化

**日期**：2026-07-31 · **硬件**：H200（GPU 4） · **代码基线**：sglang main @ `08af5aea57` 之后
**读者**：Chendi

---

## 0. 先说结论（包含一条自我更正）

这份文档记录两个真实的 kernel fusion case，以及一个方法学问题的诚实答案：
**能不能用 torch.compile / FX graph 代替「profiling + 人工读代码」来发现融合机会？**

| | Case 1：rank 守卫导致的 dispatch gap | Case 2：slice 阻断的横向融合 |
|---|---|---|
| 现象 | 融合 kernel 存在，但 4-D 输入进不去，掉回 eager（10 个 kernel） | q_norm / k_norm 本可合成 1 个 kernel，实际是 2 个 |
| **实际怎么发现的** | **profiling + 读代码** | **读代码（先验知识）** |
| FX 的作用 | **验证 + 量化**（不是发现） | **验证 + 量化**（不是发现） |
| 收益 | eager 下 10 kernel → 1 kernel | 小 batch 1.94×，大 batch 会退化 |
| 状态 | 已提 PR #32670 | 仅 microbenchmark，**未做端到端** |

**必须更正的一条**：我此前在 `FINAL_PROJECT_portable_fusion_discovery.md` §2「结果一」里写
「自动化能重现人工发现」。**这句话不成立，予以撤回**，理由见 §3.3。
两个 case 的假设都来自人读代码，FX 是测量仪器而非发现机制。

**但这不等于「读代码更有用」**。两个 case 的信号都是**图上可机读的**，
我已经把检测规则写成了具体条件（§5），其中**规则 A 的雏形已跑通并通过正/负对照**。
当前结论应表述为：
*现有 FX 工具没能发现它们；信号可机械化，规则 A 已验证可行但仍需人工指定探测目标，规则 B 尚未实现。*
考虑到 sglang 有 212 个模型文件（其中 39 个用了 q_norm 这一模式），人工读代码不 scale，
把这两条规则做成检测器才是有价值的方向。

---

## 1. 共同背景：sglang 的算子如何 dispatch

sglang 的归一化层继承 `MultiPlatformOp`，按硬件选择实现：

```python
# python/sglang/srt/layers/utils/multi_platform.py:82
def forward(self, *args, **kwargs):
    return self._forward_method(*args, **kwargs)   # forward_cuda / forward_native / ...
```

`Gemma3RMSNorm` 的 CUDA 实现（`python/sglang/srt/layers/layernorm.py`）：

```python
def forward_cuda(self, x, residual=None):
    if residual is not None:
        gemma_fused_add_rmsnorm(x, residual, self.weight.data, self.eps)
        return x, residual
    if x.dim() == 2:                                          # ← 本文的主角
        return gemma_rmsnorm(x, self.weight.data, self.eps)   # 融合 kernel
    return self.forward_native(x)                             # 掉回 eager
```

而 `forward_native` 就是纯 PyTorch 表达式：

```python
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

**这道 `if x.dim() == 2` 是 Case 1 的全部根因。**

历史（`git log -S`）：

| 版本 | `forward_cuda` | 说明 |
|---|---|---|
| #32383 之前 | `return self.forward_native(x)` | 压根没接融合 kernel |
| #32383（`08af5aea57`，别人的 PR） | 加了 `dim()==2` 守卫 | 接上了，但只对 2-D 开门 |
| PR #32670（本工作） | 压平后所有 rank 都能进 | — |

---

## 2. Case 1：rank 守卫导致的 dispatch gap

### 2.1 问题

Gemma-3 每层 6 个 RMSNorm，收到的张量 rank 不同。
`gemma3_causal.py:254-261`：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

q = q.unflatten(-1, (self.num_heads, self.head_dim)).unsqueeze(0)  # -> [1, s, h, 256]
q = self.q_norm(q)                                                 # 4-D 输入！
k = k.unflatten(-1, (self.num_kv_heads, self.head_dim)).unsqueeze(0)
k = self.k_norm(k)
```

`unflatten` 把 `[s, h*hd]` 变成 `[s, h, hd]`，再 `unsqueeze(0)` 变成 **4-D** `[1, s, h, 256]`。
于是 `x.dim() == 2` 为假 → 走 `forward_native`。

融合 kernel `sgl_kernel.gemma_rmsnorm` 的签名只接受 2-D `[num_tokens, hidden_size]`，
**这就是那道守卫存在的原因**——不是编译器融不动，是手写 kernel 的接口收不下高维输入。

### 2.2 发现路径（诚实版）

1. 跑 profiler，看到 Gemma-3 注意力前有一串碎 kernel
2. 去 `layernorm.py` 读源码，看到 `if x.dim() == 2`
3. 回到 `gemma3_causal.py` 确认 q_norm 收到的确实是 4-D

**发现靠的是 profiler + 人眼。** 这一步没有用到 FX。

### 2.3 FX 上的信号（本次新做的实验）

问题：这个 bug 在 FX 图上留下痕迹了吗？如果有，就说明它**原则上可被自动检测**，
且检测过程不需要 CUDA profiler（可迁移到 MAIA）。

实验：同一个模块、同一份权重，只改输入 rank，用 `torch._dynamo.explain` 抓图。

```python
m = Gemma3RMSNorm(1152).cuda()
for x in (torch.randn(64, 1152, dtype=torch.bfloat16),          # 2-D
          torch.randn(1, 64, 4, 1152, dtype=torch.bfloat16)):   # 4-D
    torch._dynamo.reset()
    e = torch._dynamo.explain(lambda t: m(t))(x)
    print(e.graphs[0].code)
```

**图 A（2-D 输入）** —— 只有 1 个计算节点，是个不透明的自定义 op：

```python
def forward(self, L_t_, G_m_parameters_weight_):
    _get_data_attr = torch._C._autograd._get_data_attr(g_m_parameters_weight_)
    out = torch.empty_like(l_t_)
    get_device_capability = torch.cuda.get_device_capability(0)
    gemma_rmsnorm_default = torch.ops.sgl_kernel.gemma_rmsnorm.default(
        out, l_t_, _get_data_attr, 1e-06, True)
    return (out,)
```

**图 B（4-D 输入）** —— 自定义 op 消失，摊开成 10 个逐元素/规约算子：

```python
def forward(self, L_t_, G_m_parameters_weight_):
    float_1  = l_t_.float()
    pow_1    = float_1.pow(2)
    mean     = pow_1.mean(-1, keepdim = True)
    add      = mean.add(1e-06)
    rsqrt    = torch.rsqrt(add)
    output   = float_1.mul(rsqrt)
    float_2  = g_m_parameters_weight_.float()
    add_1    = float_2.add(1.0)
    output_1 = output.mul(add_1)
    output_2 = output_1.type_as(l_t_)
    return (output_2,)
```

| | 2-D | 4-D |
|---|---|---|
| 计算节点数 | 1 | 10 |
| 有 `sgl_kernel.*` 自定义 op | 有 | **无** |
| 有 `pow`/`mean`/`rsqrt` 等 eager 标记 | 无 | **有** |

`if x.dim() == 2` 是 Python 层分支，Dynamo 追踪时会**求值并特化**，
所以图里不会出现 `if`，而是直接表现为**两种完全不同的图形态**。
**这就是那行守卫留下的指纹。**

### 2.4 kernel 计数（把损失落到实处）

```
eager   2-D : 1  kernel   ← 走 sgl_kernel.gemma_rmsnorm
eager   4-D : 10 kernel   ← 掉回 forward_native
compile 4-D : 1  kernel   ← Inductor 自己把链融回去了
```

10 个节点 → 10 个 kernel，与图 B 完全对应。

**一个重要且反直觉的结论**：真实损失发生在 **eager 模式**。
sglang 默认不开 `--enable-torch-compile`，线上跑的就是那 10 个 kernel。
一旦开了 torch.compile，Inductor 反而把这个 bug **掩盖**了（10 → 1）。

> 所以 torch.compile 在这里的角色是 **X 光片，不是药**：
> 用它的追踪结果做静态体检，而修复落在 eager 执行路径上。
> 这也正是该方法可迁移的原因——它只读图，不读时间、不依赖 CUDA profiler。

### 2.5 这个实验的局限（必须声明）

有三件事是**我手工指定的**，不是自动的：

1. **探哪个模块**——我直接挑了 `Gemma3RMSNorm`，因为已知它有问题
2. **试哪些 rank**——2-D / 4-D 是手挑的
3. 两次探测的 dim 都用 1152（受控演示），**不是真实 q_norm 的 head_dim=256**

所以本实验证明的是「**信号存在且可判定**」，**不是**「能自动发现」。

一个意外的对照组增强了可信度：我最初用的是旧版 sglang 代码树，
那里 `forward_cuda` 直接 `return self.forward_native(x)`，
结果**两个 rank 都展开**、检测不出不对称——**没有 kernel 就没有 gap**，
恰好符合检测逻辑的预期。

---

## 3. Case 2：slice 阻断的横向融合（QK-norm 合并）

### 3.1 问题

回看 `gemma3_causal.py:254-261`：qkv 是**一次**融合投影出来的，
然后被 `split` 切成 q / k / v，再**分别**调用 `self.q_norm(q)` 和 `self.k_norm(k)`。

实测：**两个从同一张量切片出来的 norm → Inductor 出 2 个 kernel；
两个互相独立的 norm → Inductor 融成 1 个 kernel。**

→ **是 slice 阻止了横向（lateral）融合**，不是「两个 reduction 本质上融不了」。

### 3.2 改法与数据

两个 norm 都沿同一个 `head_dim=256` 规约，因此可以 view 成 `[tokens, heads, 256]`
做**一次** norm，配一个 **per-head 权重**：

```python
def merged(v, W):
    o = v.reshape(-1, QH + KH, HD).float()
    return (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()).type_as(v)
# W = cat([wq.repeat(QH,1), wk.repeat(KH,1)])  -> [QH+KH, HD]，与原语义精确等价
```

实测（bf16，head_dim=256，4 q-head + 1 kv-head，H200）：

| tokens | 切片版 (us) | 合并版 (us) | 加速 | 数值等价 |
|---|---|---|---|---|
| 8 | 1.989 | 1.023 | **1.94×** | ✓ |
| 32 | 2.209 | 1.141 | **1.94×** | ✓ |
| 64 | 2.336 | 1.267 | 1.84× | ✓ |
| 128 | 2.588 | 1.500 | 1.73× | ✓ |
| 512 | 3.807 | 2.787 | 1.37× | ✓ |
| 2048 | 8.612 | 7.461 | 1.15× | ✓ |
| 4096 | 7.668 | 13.619 | **0.56×（更慢）** | ✓ |

7/7 数值等价（`allclose`, atol=rtol=2e-2）。
收益集中在小 token 数——**正好是 decode 阶段**，serving 的主战场。
T=4096 会退化，真要落地**必须加形状守卫**。

### 3.3 为什么这个 case 对「编译器 vs 手写 kernel」有意义

合并需要 **per-head 权重**，而 `sgl_kernel.gemma_rmsnorm(input, weight, eps)`
的 weight 是**单个 1-D 张量**，表达不了。

- 走**手写 kernel** 路线：必须**新写一个 kernel**
- 走 **torch.compile** 路线：**三行就写完了**（见上面的 `merged`）

这是编译器路线相对手写 kernel 路线的一个具体优势案例。

### 3.4 发现路径（诚实版）与两条更正

**实际路径**：

1. 做 PR #32670 时读过注意力代码，**已知** q_norm / k_norm 是分开调用、qkv 先投影再切片
2. **基于这个先验**去做定向实验：编译两种写法，数 Inductor 出多少 kernel
3. 得到「切片版 2 个 / 独立版 1 个」，反推出 slice 是屏障
4. 再量化加速比、验证数值等价

**更正一：FX 扫描器没有发现它，而且结构上发现不了。** 证据在扫描器自己的代码里：

```python
# scripts/fx_fusion/fx_fusion_scanner.py:48-52
# Ops that only relabel memory. They neither cost nor block fusion, so a chain
# may pass straight through them.
_VIEWLIKE = {..., "slice", "select", ...}
```

扫描器**明确假设 slice 不阻止融合**，走链时直接穿过去——
而本 case 的全部内容恰恰是「slice 阻止了横向融合」。
此外扫描器只沿 `num_users==1` 找**纵向线性链**，
对「两条并列兄弟链能否合并」**没有任何代码**（grep `horizontal|lateral|sibling` 全空）。

**更正二：`results/fx_fusion/qknorm_sglang_kernel.csv` 那张表不能当结论用。**
它测的是"调 1 次 vs 调 2 次 `sgl_kernel.gemma_rmsnorm`"（2.02× → 1.11×），
**没有等价性验证列**——因为现有 kernel 表达不了 per-head 权重，
所以那不是一个合法实现，只能视作 **launch 开销的上界估计**。

---

## 4. 方法学结论：FX 能做什么、不能做什么

### 4.1 FX 真实做到的

- **把假设变成硬证据**：图 A/B 的形态差异、10→1 的 kernel 计数，都是可复核的客观量
- **硬件无关**：全程只读追踪出来的图，不需要 nsys / CUDA profiler。
  扫描器部分已验证可在 `--device cpu` 下运行
- **可量化**：给出了收益规模和**退化边界**（T=4096 反而变慢）

### 4.2 FX 没做到的

- **两个 case 的假设都来自人读代码**，FX 只承担验证
- 扫 HF 模型这条路**没有区分度**：HF 实现里根本没有自定义 op，
  所有 RMSNorm 都长成图 B 那样。`results/fx_fusion/model_scans/gemma3.json` 里
  13 条 RMSNorm 链中，dim=1152（9 条）和 dim=256（4 条）**签名完全相同**，
  当初分出「dim=256 是 bug」是我凭先验手工点的，扫描器无此能力

### 4.3 核心教训

**必须扫框架自己的模块，而不是 HF 参考实现。**
信号来自「**同一模块内融合与否的不对称性**」，
HF 那边没有对照物（全是展开的），也就没有信号。

**两类 gap 需要两种工具**：

| gap 类型 | 特征 | 检测方式 |
|---|---|---|
| 调用点没接上（Case 1） | 框架**有** kernel，某些形状进不去 | 跨 rank 追踪同一模块，看形态是否不对称 |
| 编译器融不掉（Case 2） | 图结构（slice）阻断了本可发生的融合 | 兄弟链 + 共同 producer + 落在不同 kernel |

---

## 5. 可机械化的检测规则（尚未实现）

这是把「验证」升级成「发现」的关键，两条规则都只依赖图上直接可读的信息。

### 规则 A —— dispatch gap 检测（**雏形已跑通**）

> 枚举框架内所有 `MultiPlatformOp` 子类。对每个模块，用 rank 1~4 的输入各追踪一次。
> 若**某些 rank 的图含自定义 op（不透明）、另一些 rank 的图是展开链**
> → 报告为 dispatch gap。

- 判据分两级：
  - **正证据**——图中是否出现**非 PyTorch 命名空间**的注册算子
    （`namespace` 不属于 `aten/prims/prim/...`），即一个手写 kernel。有则判为 fused，
    并**报出 kernel 名**
  - **辅助启发式**——是否出现 ≥2 个 `pow/mean/rsqrt/var/erf` 等 eager 标记。
    这只能说明「没看见 eager 数学」，是弱证据，仅在无正证据时使用
- 若**全部** rank 都展开 → 框架根本没有该 kernel，**不算 gap**

实现：`scripts/fx_fusion/fx_dispatch_gap_detector.py`。已跑出一组**正/负对照**：

**阳性（sglang main，有融合 kernel + `dim()==2` 守卫）** →
`results/fx_fusion/dispatch_gap_gemma3rmsnorm.json`

```
rank=2 shape=[64, 1152]        FUSED  (registered kernel)
    kernel: ['sgl_kernel.gemma_rmsnorm.default']
rank=3 shape=[1, 64, 1152]     EAGER  (expanded math)   markers: ['mean','pow','rsqrt']
rank=4 shape=[1, 64, 4, 1152]  EAGER  (expanded math)   markers: ['mean','pow','rsqrt']

DISPATCH GAP FOUND
  fused at: ['[64, 1152]']
  eager at: ['[1, 64, 1152]', '[1, 64, 4, 1152]']
```

**阴性对照（旧代码树，`forward_cuda` 直接转 `forward_native`，无融合路径）** →
`results/fx_fusion/dispatch_gap_negative_control.json`

```
rank=2/3/4  全部 EAGER (expanded math)，无任何注册 kernel
no gap: all shapes trace the same way
```

阴性对照很关键：它证明检测器报的是**「有 kernel 却进不去」**，
而不是简单地「看到展开链就报警」——后者会对任何 eager 实现全部误报。
注意 rank=3 同样命中，与 `gemma3_causal.py:291` 那条 `forward_native` 路径一致。

**仍待完成**：目前模块和 rank 列表仍是**手工指定**的。
要成为真正的「发现」，需要
(a) 自动枚举框架内全部 `MultiPlatformOp` 子类；
(b) 给真实模型挂 forward hook 记录每个模块**实际收到的形状**，而非人为猜测 rank。

### 规则 B —— 横向融合候选检测

> 在 post-grad 图里找 ≥2 条**签名相同**的链，其输入均可回溯到
> **同一 producer 的不同 slice**，且 Inductor 最终把它们放进了**不同的输出 kernel**
> → 报告为横向融合候选。

- 三个条件都在图上直接可读，不需要知道这是什么模型
- 实现的第一步是**把 `slice` 从 `_VIEWLIKE` 里移除**，反过来标记为「融合屏障」
- **验收标准**：在**不告知这是 gemma-3** 的前提下，能把 q/k norm 那一对报出来。
  做到这一点，才算真正「用 FX 发现了」融合机会

---

## 6. 复现方式

环境：`~/.conda/envs/gemma-sglang`（torch 2.11.0+cu130 / triton 3.6.0），
sglang 源码树 `/tmp/sglang_lfm`（main）。
注意 CUDA toolchain 在 `site-packages/nvidia/cu13/{bin,include,lib}`，需设 `CUDA_HOME`，
否则 `import sglang` 会因 deep_gemm 报 AssertionError。

```bash
ENV=~/.conda/envs/gemma-sglang; CU13=$ENV/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU13 PATH=$CU13/bin:$ENV/bin:$PATH LD_LIBRARY_PATH=$CU13/lib
export PYTHONPATH=/tmp/sglang_lfm/python CUDA_VISIBLE_DEVICES=4

# Case 1 —— dispatch gap 检测（阳性）
python scripts/fx_fusion/fx_dispatch_gap_detector.py \
    --out results/fx_fusion/dispatch_gap_gemma3rmsnorm.json

# Case 1 —— 阴性对照（旧代码树，无融合路径，应报 no gap）
#   注意换用 sglang-dev 环境，且 CUDA_HOME 指向该 env 本身
CUDA_HOME=~/.conda/envs/sglang-dev ~/.conda/envs/sglang-dev/bin/python \
    scripts/fx_fusion/fx_dispatch_gap_detector.py \
    --sglang-src /home/t-jialianggu/work/sglang/python \
    --out results/fx_fusion/dispatch_gap_negative_control.json

# Case 2 —— 完整验证（含数值等价检查）
python scripts/fx_fusion/verify_qknorm_merge.py --out results/fx_fusion/qknorm_merge.csv
```

§2.3 的两张完整 FX 图与 §2.4 的 kernel 计数可直接粘贴代码片段运行。

### 踩过的测量陷阱（复现时务必注意）

1. **Dynamo recompile cache 超限会静默退回 eager**，导致 `compile/eager ≈ 1.00` 的假信号。
   每换一个 shape 前必须 `torch._dynamo.reset()`。
   修正前后中位数从 7.27× 变成 0.80×，**结论完全反转**
2. **profiler 上下文紧邻计时区会让后续测量归零**，两者需分离
3. **函数内定义的闭包会被 Dynamo 反复重编译**，被测函数必须定义在模块级
4. `e.device_type.name` 在某些 torch 版本不可靠，应用 `"CUDA" in str(e.device_type)`
5. wall-clock 会被 Python guard 开销污染，必须用 profiler 只统计 GPU 时间
6. **要验证的是 main 分支的代码**：旧代码树 `forward_cuda` 无融合路径，会得到相反结论

---

## 7. 未完成的工作

- [ ] **完成规则 A**：雏形已跑通并有正/负对照（§5），
      但模块与 rank 仍是手工指定；需自动枚举全部 `MultiPlatformOp` 子类 + hook 真实形状
- [ ] **实现规则 B**：兄弟链检测；验收标准见 §5
- [ ] **Case 2 端到端 A/B**：目前只有 microbenchmark，缺 sglang 端到端吞吐/延迟数据
- [ ] 修改 `FINAL_PROJECT_portable_fusion_discovery.md` §2「结果一」的表述（本文 §0 已撤回）
- [ ] Case 2 落地需加形状守卫（T≥4096 会退化）
- [ ] 两个模型扫描失败待修：phi4-mini（transformers 版本）、falcon-h1（权限）
