# LFM2.5 / SGLang 内核优化全纪录（2026-07-26 ~ 07-27）

**作者**：本轮会话 · **仓库**：`gujialiang123/end2end-optimization` @ `main`
**硬件**：1× NVIDIA H200（TP1）· **精度**：BF16
**软件**：sglang 0.5.12.post1 @ `17f7a1da1` · torch 2.9.1+cu128 · Triton 3.5.1 · CUDA 12.8 · driver 580.105.08
**模型**：LFM2.5-8B-A1B（主）· Qwen3-30B-A3B-Instruct-2507（对照）

> 本文是给团队/汇报用的**自包含总结**。想看某条线的完整证据，见文末 §10 的文档地图。
> 所有数字均为**本轮实测**，不引用任何 PR 自称的数据。

---

## 0. 三句话总结

1. **拿到了什么**：在 LFM2.5-8B-A1B 上，通过 7 个内核层改动，端到端请求吞吐 **低批 decode +6.57% · 并发 decode +6.21% · 长 prefill +5.30%**（各 6 次重复，精确 Welch t，p = 4.6e-14 / 2.4e-08 / 1.2e-05），GSM8K 无质量回归。**这是本项目第一个同模型、正向、全 regime 显著的内核级端到端结果。**
2. **为什么以前没有**：项目此前的结论"成熟 bf16 MoE 上内核层不转化为端到端收益"是**在 Qwen 一个模型上**得出的。该结论对 Qwen 成立，但**覆盖空缺是"架构成熟度"的函数**——新架构在其新算子**周围的调用点胶水**上带着几个百分点。
3. **最可迁移的产出**（比那 6% 更重要）：**消除同一"种类"成本的优化不会相加**，兑现率随 regime 饱和度单调下降（0.90 / 0.70 / 0.49）；以及 **regime→backend 规则跨模型不可迁移**（用错最差 −34%）。

---

## 1. 这两天做了什么（时间线）

| # | 工作 | 性质 | 结果 |
|---|---|---|---|
| 1 | K1 跨模型验证：Qwen 的 MoE backend 对比 | 交接文档 §8.1 指定 | **负面但重要**：规则不可迁移 |
| 2 | LFM2.5 算子级审计（+ Qwen 对照） | 新方法 | 找到 3 类结构性空缺 |
| 3 | 第一轮修复：`norm` + `scale` | 调用点改动 | decode +3.8~4.0% |
| 4 | nsys 时间线分析（子 agent） | 深度 profiling | 5 个候选排序 + 2 个否决 |
| 5 | FX/Inductor 图挖掘（子 agent） | 编译器视角 | 独立验证 + 机制修正 |
| 6 | 第二轮：`conv` Triton kernel | **手写内核** | 长 prefill +2.33% |
| 7 | 第三轮：`qkrope` / `gate` / `idx` | 调用点改动 | 并发 decode +5.42% |
| 8 | 第四轮：`moesum` Triton kernel（子 agent） | **手写内核** | 低批 decode +4.55% |
| 9 | 组合验证 + 次可加性分析 | 方法学 | **最有价值的产出** |

共 **10 个 commit**、约 **300 次 benchmark 运行**、**0 次静默失败**（失败全部显式记录）。

---

## 2. 线一：K1 跨模型 —— regime→backend 规则**不可迁移**（负面结果）

### 2.1 背景
上一轮在 LFM2.5 上发现：SGLang 的 MoE runner backend（`triton` / `triton_kernel` / `flashinfer_cutlass`）**排序随 regime 翻转**。而 backend 是**启动时定死、全程不变**的，所以"按 regime 选 backend"是运行时真正缺失的能力。交接文档 §8.1 要求在 Qwen 上验证是否为跨模型规律。

### 2.2 做法
同协议：3 regime × 4 backend × 5 重复，serving 参数冻结，只改 `--moe-runner-backend`。共 60 次运行，**0 失败**。

### 2.3 结果（ratio vs `auto`）

| backend | A 低批 decode<br>LFM / Qwen | B 并发 decode<br>LFM / Qwen | C 长 prefill<br>LFM / Qwen |
|---|---:|---:|---:|
| `triton` | 0.999 / 1.001 | 1.006 / **1.033** | 1.004 / 0.987 |
| **`triton_kernel`** | **0.650 / 0.641** | 0.966 / 1.008 | 0.996 / **0.647** |
| **`flashinfer_cutlass`** | 0.965 / 0.934 | **1.017 / 1.047** | **0.664** / **1.027** |

**两个 regime 两模型一致**：`triton_kernel` 在低批 decode 都是灾难（0.650/0.641）；`cutlass` 在并发 decode 都最好（1.017/1.047）。

**长 prefill 完全反转** —— 断崖不是变小，而是**换到了另一个 backend**：
- LFM2.5 上 `cutlass` **最差**（0.664×），`triton_kernel` 无害（0.996×）
- Qwen 上 `cutlass` **最好**（1.027×），`triton_kernel` 才是灾难（0.647×）

### 2.4 结论
**可迁移的结论不是"regime → backend"。** 把 LFM 的长 prefill 规则用到 Qwen 会放弃它最好的 backend；把 Qwen 的规则用到 LFM **−34%**。

> **静态 regime→backend 查找表不只是不完整，而是有害的。** 这反过来是"必须按每个部署实测（即需要 agent）"的最直接论据。

不对称性保持并略微扩大：最好 +4.7%，最差 −36% → **backend 选择是避坑杠杆，不是提速杠杆。**

---

## 3. 线二：LFM2.5 算子级审计 —— 发现空缺

### 3.1 为什么要重做审计
项目此前的 v33 审计结论是：**"对 Qwen3-30B，sglang 热路径已全部 CUDA 融合，没有可补的空缺"**。这个结论此后成为降低 kernel 融合优先级的依据。

但它是**在一个模型上**得出的。而 v33 自己那张"CPU 有融合、CUDA 没有"的表里，三个算子各自服务**不同架构**——这暗示覆盖空缺是**架构**的属性。LFM2.5 是本机上最新的架构（24 层里 **18 层是 gated short conv**，只有 6 层全注意力），从没被算子级审计过。

### 3.2 方法
复用 v33 方法：`bench_one_batch --profile` + **关闭 CUDA graph**（让每个算子单独现形）→ 按 kernel 名分桶。**新增**：不只统计时间，而是数**"融合实现根本不会执行的 kernel"**的个数，并拿 Qwen 做对照。

脚本：`scripts/lfm_fusion/lf_audit.py`

### 3.3 结果（每次 forward 的 kernel 启动次数）

| 模型 | 未融合 RMSNorm | 独立 residual add | gating mul | layout copy |
|---|---:|---:|---:|---:|
| **LFM2.5** | **61** | **48** | **36** | 22–53 |
| Qwen3-30B（对照） | **1** | **0** | **0** | 4–52 |

**计数是结构性的，不是约数**：
- `48 = 2 个 residual add × 24 层`
- `36 = 2 个 gating mul × 18 个 conv 层`

**对照组是决定性的**：Qwen 一整个 forward 只有 1 个未融合 norm、0 个独立 add。**这不是 sglang 的通病，是这个模型的实现漏了。**

### 3.4 顺带得到的架构对比（长 prefill，同一 workload）

| bucket | LFM2.5 | Qwen3-30B |
|---|---:|---:|
| MoE | 70.8% | 54.2% |
| **注意力** | **2.8%** | **21.6%** |
| short conv | 0.7% | — |
| dense GEMM | 12.5% | 16.1% |
| **norm + elementwise** | **12.8%** | **5.6%** |

LFM2.5 的架构确实兑现了承诺：注意力 + 替代它的 conv 一共 **3.5%**，而 Qwen 是 **21.6%**。但它把这个结构性优势的一部分**以 12.8% 的未融合胶水交还回去**（Qwen 只有 5.6%）。

> **空缺不在新算子里**——`causal_conv1d` 本身只占 0.7%，很快。**空缺在上游还没来得及融合的调用点胶水上。**

---

## 4. 七个内核改动：具体改了什么

所有改动通过 `LFM_FUSION_PATCH` 环境变量 **opt-in**，不设变量时走的是**逐字未改动的 sglang 原路径**，所以 A/B 的 baseline 是真 baseline。

注入方式：模型类被 model registry **懒加载**，`sitecustomize` 执行时 `lfm2_moe` 还没导入，用定时器打 patch 是**竞态**。改用 `sys.meta_path` finder，在该模块 exec 完成的**瞬间**打补丁（`lf_inject/sitecustomize.py`）。

---

### G1 `norm` —— residual 加法从未被融合

**问题**（`sglang/srt/models/lfm2_moe.py:433-456`）：

```python
def forward(self, layer_id, positions, hidden_states, residual, forward_batch, **kwargs):
    residual = hidden_states                    # ← 传进来的 residual 参数被直接覆盖
    normed = self.operator_norm(hidden_states)  # ← 没传 residual → 走非融合分支

    hidden_states = self.conv(normed, forward_batch)

    hidden_states = hidden_states + residual    # ← 单独一个 elementwise kernel
    hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
                    #                ↑ 又一个单独的 kernel
    return hidden_states, residual
```

三个关键观察：
1. 函数签名**收了 `residual` 参数**，第一行就把它覆盖掉——传进来的值从没被用过
2. `RMSNorm.forward_cuda(x, residual)` **本来就会走 `fused_add_rmsnorm`**（`layers/layernorm.py:139-147`），一趟做完加法和归一化
3. `Lfm2MoeModel.forward` **本来就在层间传递 residual**

→ **接线全都在，只是这一层没接上。**

**修复**：改成 llama / qwen2 / 所有正常模型都在用的 **deferred-residual** 写法（`models/llama.py:304-316`）：

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

**数学等价**（写 `x` 为进入本层的激活）：

```
原版：  a = op(rms(x));  h1 = a + x;  out = h1 + ffn(rms(h1))
新版：  rms(x, r) → r := x,      n := rms(x)          ← 加法在 norm kernel 内部完成
        a = op(n)
        rms(a, r) → r := a + x = h1,  n2 := rms(h1)
        返回 (ffn(n2), h1)
```
下一层拿到 `ffn(n2) + h1`，与原版的 `out` 是同一个值。**残差不在本层结清，而是当作"欠账"传给下一层，由下一层的 norm kernel 顺手结清。**

**效果**：每层省 2 个 kernel × 24 层 = **48 个**。

---

### G2 `scale` —— 每次 forward 有 22 个 kernel 在"乘以 1"

**问题**（`lfm2_moe.py:156-169`）：

```python
def forward(self, hidden_states):
    router_logits, _ = self.gate(hidden_states)
    topk_output = self.topk(hidden_states, router_logits)
    final_hidden_states = self.experts(hidden_states, topk_output)
    return final_hidden_states * self.routed_scaling_factor   # ← 这里
```

而 LFM2.5 的 `config.json` 里 `"routed_scaling_factor": 1.0`。

→ 每次 forward **22 个 GPU kernel** 在把整个 `[T, 2048]` 激活张量逐元素乘以 1：读一遍、乘 1、写一遍，什么都没发生。

**修复**（3 行）：
```python
if self.routed_scaling_factor == 1.0:
    return final_hidden_states
return final_hidden_states * self.routed_scaling_factor
```

**这是 bit-exact 的**——有限的 bf16 数乘 1.0 就是它自己。

> 注：代码里那个 factor 不放进 `FusedMoE` 而手动乘是有正当理由的（放进去会引入与 HuggingFace 的数值差异），所以 factor ≠ 1 时那个乘法该留着。只跳过 = 1 的情况。

---

### G3 `conv` —— **手写 Triton kernel ①**，ShortConv 的胶水不合并

**问题**（`lfm2_moe.py:321-377`）：

```python
proj, _ = self.in_proj(hidden_states)        # GEMM -> [T, 3H]
B_gate, C_gate, x = proj.chunk(3, dim=-1)    # 3 个 strided 视图
Bx = B_gate * x                              # elementwise -> [T, H]
Bx_t = Bx.transpose(0, 1).contiguous()       # 物化 -> [H, T]
conv_out = causal_conv1d_fn(Bx_t, ...).transpose(0, 1)   # 视图, [T, H]
output, _ = self.out_proj(C_gate * conv_out) # elementwise，读的是转置视图
```

`causal_conv1d_fn` 是**不透明的外部 CUDA 算子**，要求 `[dim, seqlen]` 布局且 `x.stride(-1) == 1`（`causal_conv1d.py:59-60`）。所以布局转换**躲不掉，只能被吸收**进相邻的 elementwise 工作里。

**关键诊断——问题不是流量，是访问不合并**：

```
18 个 conv 层 × 500 MB 流量 = 8.79 GB  用了 10.3 ms  →  0.83 TB/s
H200 HBM 峰值 ~4.8 TB/s                              →  仅 17% 峰值
```

`Bx.transpose(0,1).contiguous()` 和 `C_gate * conv_out` 里的转置读都是跨步访问，每次取回的 cache line 大部分被丢弃。

**修复**（`lf_triton_shortconv.py`）：conv **两侧各一个 tiled Triton kernel**，把 chunk + gating mul + transpose 折叠进一趟，转置用 `tl.trans` 在寄存器/共享内存里完成，不发跨步全局访问。

```python
@triton.jit
def _fused_gate_transpose_kernel(proj_ptr, out_ptr, T, H, ...):
    # 沿 H 合并读入 [BLOCK_T, BLOCK_H] tile
    b = tl.load(proj_ptr + base, mask=mask)                    # B_gate
    x = tl.load(proj_ptr + base + 2*H*stride_h, mask=mask)     # x
    bx = (b.to(tl.float32) * x.to(tl.float32)).to(b.dtype)
    # 转置写出：[BLOCK_H, BLOCK_T]，沿 T 合并
    tl.store(out_ptr + out_off, tl.trans(bx), mask=...)
```

**隔离结果**（correctness 门禁先于计时）：

| T | input side | output side | 带宽 | 每 forward 省 |
|---:|---:|---:|---|---:|
| 1024 | 0.94× | 0.71× | — | −0.22 ms |
| 2048 | 1.29× | 0.93× | 0.9 → 0.7 TB/s | +0.27 ms |
| 4096 | 2.24× | 1.76× | 0.9 → 1.3 TB/s | +1.47 ms |
| **16000** | **5.93×** | **4.33×** | **0.98 → 3.46 TB/s** | **+7.86 ms** |

带宽从 **17% 提到 ~72% 峰值**。**每个测试形状都 bit-exact**（max|diff| = 0.0）。

**形状门控**：融合 kernel 有 ~30 µs 的地板（Triton 的 Python launch 路径），T < 2048 时打不过原生 elementwise。低于 `CONV_FUSION_MIN_TOKENS` 走原路径。tile 尺寸来自**实测扫描**（`lf_tune_shortconv.py`，每形状 32 组配置，先验正确性再计时），不是猜的。

decode 路径**根本不转置**（`causal_conv1d_update` 直接吃 `[T,H]`），所以这个组件**结构上就是 prefill-only**。

---

### G4 `qkrope` —— 融合原语早已存在，这个模型没调用

**问题**（`lfm2_moe.py:236-263`）：

```python
q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
q = q.reshape(T, num_q_heads, head_dim)
k = k.reshape(T, num_kv_heads, head_dim)
q = self.q_layernorm(q.reshape(-1, head_dim)).reshape(...)   # 独立 RMSNorm
k = self.k_layernorm(k.reshape(-1, head_dim)).reshape(...)   # 独立 RMSNorm
q, k = self.rotary_emb(positions, q, k)                       # 独立 RoPE
```

而 `sgl_kernel.fused_qk_norm_rope` **早就存在**（`sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu`），把两个 head-wise RMSNorm 和 RoPE 合并成**一个 in-place CUDA kernel**，**Qwen3-MoE 已经在调用它**（`models/qwen3_moe.py:559-585`）。LFM2.5 没有。

**测得的现有链路**：decode 18 次调用 32.5 µs（**1.65%** kernel 时间）；prefill 30 次调用 5581 µs（**3.61%**）。

**修复**：在 packed QKV 上直接调融合 kernel。LFM2.5 `head_dim = 2048/32 = 64`（该 kernel 支持），`rope_type` 为 `default` 无 `rope_scaling` → yarn 参数退化为恒等 `(1.0, 0, 0, 1.0)`：

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
不满足条件时**回退原实现**（保留 fallback）。

**这是 G1 的同一模式，也是第三个实例：融合原语已存在，调用点没用。**

---

### G5 `idx` —— 18 个 kernel 只为搬 12 字节

`req_pool_indices.to(torch.int32)` 在**每一个 conv 层**里重算一遍（18 次/forward）。这个 kernel 只搬 **12 字节**，是纯 launch 开销，但占**低批 decode kernel 时间 ~1.3%**。

**修复**：按 forward 缓存，用**源张量的 identity 作 key**，保证不会返回陈旧缓存：

```python
cached = getattr(forward_batch, "_lfm_int32_idx", None)
if cached is not None and cached[0] is req_pool_indices:
    return cached[1]
out = req_pool_indices.to(torch.int32)
forward_batch._lfm_int32_idx = (req_pool_indices, out)
```

---

### G6 `gate` —— strided rows 让向量化失效

decode 路径的 `B_gate * x` 读的是 `proj` 的**跨步行**。这**是合并访问**，但仍只跑到 54% 峰值——因为跨步的**行**让 PyTorch 的 `TensorIterator` 无法向量化，退化成标量 `elementwise_kernel` 而不是 `vectorized_elementwise_kernel<8>`（由 trace 里的 kernel 名确认）。改用一个直接读 `proj` 的 Triton kernel 绕开。

---

### G7 `moesum` —— **手写 Triton kernel ②**，消除 MoE 归约的 HBM 往返

**问题**：MoE 的 top-k 归约把 `[T, H]` 写回 HBM，紧接着**下一层**的 `fused_add_rmsnorm` 又把它读回来。两者都是**行方向**的操作。

**修复**（`lf_triton_moesum.py`）：让 `FusedMoE` 返回 4 个加权专家输出，一个 kernel 做完**归约 + 残差加 + RMSNorm**：每行加载 top-k 分量 → 求和 → 加残差 → 算 RMS → 乘权重 → 同时写出归一化输出和更新后的残差。

**隔离结果**：

| T | stock | fused | 加速 | 带宽 | bit-exact |
|---:|---:|---:|---:|---:|---|
| 1 | 90.5 µs | 36.8 µs | **2.46×** | 0.8 GB/s | ✅ |
| 8 | 95.6 | 35.7 | **2.68×** | 6.1 | ✅ |
| 32 | 94.8 | 36.0 | **2.64×** | 23.9 | ✅ |
| 128 | 24.9 | 33.8 | 0.74× | 101 | ✅ |
| 1024 | 24.8 | 34.3 | 0.72× | 798 | ✅ |
| 4096 | 42.2 | 36.9 | 1.14× | 2961 | ✅ |
| 16000 | 145.7 | 111.8 | 1.30× | 3821 | 4.9e-4 |

residual 输出全程 bit-exact；归一化输出到 T=4096 精确，T=16000 差 4.9e-4。

> **这与 G3 的形状依赖正好相反。** G3 在 T<2048 无用，G7 在**小 T 才是赢面**——因为省的是 **launch + 一次 HBM 往返**，T=1 时那几乎就是全部成本。两个 kernel 形状依赖相反，合起来覆盖了整个范围。

双侧门控：`T <= 32 或 T >= 4096`。

---

## 5. 两个子 agent 的深度调查

### 5.1 nsys 时间线（`results/lfm_fusion/nsys/FINDINGS.md`）

4 次捕获，用 `--capture-range=cudaProfilerApi` 只追踪测量区间。

**最重要的两个基础事实**：

| 场景 | 每层 device idle | 每层 launch API |
|---|---:|---:|
| decode，CUDA graph **关** | **689 µs** | 78.6 µs |
| decode，CUDA graph **开** | **1.5 µs** | 0 |
| prefill（graph 设置无关） | 77.7–90.0 µs | 89.4 µs |

→ **CUDA graph 开启时 launch 开销根本不是 decode 的问题**（1.5 µs/层）。而 **prefill 即使开了 decode graph 也是 eager 的**（两条 prefill trace 一致到 0.5%），所以 prefill 的 launch 开销是真实的。这直接决定了哪些优化值得做。

排序 5 个候选并**带数字否决 2 个**（topk+alignment、atomic GEMM reduction）。

### 5.2 FX / Inductor 图挖掘（`results/lfm_fusion/fx/FINDINGS.md`）

**独立验证**：把**未修改的**模块交给 Inductor，它自己推导出
`triton_poi_fused_causal_conv1d_fwd_clone_mul_split_transpose_0` —— 两次跨步加载 + 一次乘法 + 一次转置存储，零中间存储。**结构上就是我手写的那个 kernel。** 说明手写设计正是编译器会选的设计。

**Inductor 对 ShortConv 其余部分 `found 0 possible fusions`**：`causal_conv1d_fwd/_update` 是 `ExternKernelSchedulerNode`，**硬屏障**（`cannot fuse op1 with op3: no shared data`）→ **融合只能发生在 conv 两侧**，这正是两个 kernel 的形状。

**机制修正（对我自己文档的纠错）**：不是一个效应而是**两个**。transpose 和转置读确实不合并（14% / 21% 峰值），但 `B_gate*x` 是**合并的**却仍只有 54%——原因是 strided rows 让 `TensorIterator` 退化成标量 kernel。

**身份确认**：`triton_poi_fused_copy__mul_sum_0` = `moe_sum_reduce_torch_compile`，**已是流量最优，不是空缺**。

---

## 6. 正确性验证

### 6.1 一个结构性发现：token-identity 对这个模型不可用

`scale` / `conv` / `moesum` 的 residual 部分是 bit-exact 的。但 `norm` 和 `qkrope` **代数等价而非 bit-exact**（`fused_add_rmsnorm` 累加顺序与精度不同）。

三层证据：
1. **原语单测**：`fused_add_rmsnorm` vs 手动 `add` + `rmsnorm` —— residual 差 **0.0**，归一化输出差 0.03（量级 4.34），约 **2 个 bf16 ulp**，且**融合版更准**（加法保持在更高精度）
2. **重构单测**：6 层代数替身栈跑两遍 —— 相对偏差 1.1%，符合 bf16 累加漂移，无结构性错误
3. **整模型**：12 个 prompt 的 next-token 分布 —— top-1 11/12 一致，但 KL 最高到 0.99

第 3 层看着吓人，直到机制被指出：**LFM2.5 走 top-4/32 专家路由，专家选择是离散 argmax。** bf16 级扰动偶尔会翻转选中哪个专家，输出就不连续地变了。

> **所以 token-identity 对这个模型是结构性不可用的门禁**——任何数值上非恒等的改动都会触发它。必须改用任务指标。

### 6.2 用 bit-exact 的对照臂免费标定噪声底

GSM8K 全量 1319 题，贪心解码：

| 臂 | 各次结果 | 均值 |
|---|---|---:|
| baseline | 0.348 / 0.349 / 0.344 | 0.3470 |
| **`scale`（可证 bit-exact）** | 0.338 / 0.339 / 0.340 | **0.3390** |
| `norm` | 0.362 / 0.368 / 0.361 | 0.3637 |
| `norm+scale` | 0.359 / 0.359 / 0.359 | 0.3590 |
| **`conv`（bit-exact）** | 0.342 / 0.350 | **0.3460** |
| `qkrope` | 0.352 / 0.346 | 0.3490 |
| **`moesum`（bit-exact）** | 0.343 / 0.347 | **0.3450** |
| 全部七项 | 0.371 / 0.364 / 0.370 | 0.3683 |

注意 `scale` 臂：它**数学上必然等于 baseline**，却读数低 **0.8 点**。这不是 bug，是它**免费帮我标定出了 harness 的系统噪声**（`--parallel 32` 让 batch 组成在不同 server 实例间不同，而 batch 相关的 reduction 会改变贪心输出）。

三个噪声度量：between-arm 系统噪声 ≥ 0.8 点；within-arm 跨度 0.0–0.8 点；n=1319, p≈0.35 的二项抽样误差 ±2.6 点。**全部 8 个臂跨度 2.5 点，在三个度量下都在噪声内。**

> **口径：未检测到质量回归。** 不是"质量提升"——这个实验分辨不了这么小的差异，而那个 bit-exact 的臂就是证据。

---

## 7. 端到端结果

`lf_e2e.py` 复用 canonical serving harness，只变 `LFM_FUSION_PATCH`；模型、serving 参数、backend、CUDA graph 设置完全一致。**server log 会被检查 patch 生效标记**——否则一个静默失效的 patch 会被当成"与 baseline 相同"记录下来。

6 次重复/臂，Welch t + **精确 Student-t 尾**。

### 7.1 单项与组合

| regime | `qkrope` | `gate+idx` | `norm+scale+conv` | `moesum` | 六项 | **七项全开** |
|---|---:|---:|---:|---:|---:|---:|
| A 低批 decode | +0.93% | −0.00% (n.s.) | +3.89% | +4.55% | +4.60% | **+6.57%** |
| B 并发 decode | **+5.42%** | +0.65% (n.s.) | +3.65% | +3.08% | +6.01% | **+6.21%** |
| C 长 prefill | +1.99% | +0.40% (n.s.) | +3.47% | — | +5.81% | **+5.30%** |

七项全开的 p = **4.6e-14 / 2.4e-08 / 1.2e-05**。

### 7.2 组件按机制互补

- **`norm+scale`** 消除的是**每 forward 固定数量**的 kernel 和全激活读写，与该 forward 做多少计算无关 → decode 每 forward 才 ~2 ms，占比大（+4.2%）；长 prefill ~157 ms，被稀释（+1.6%）
- **`conv`** 消除的是**随 token 数增长**的流量，且要 T≥2048 才划算 → decode 够不到（精确中性，p=0.22/0.95），长 prefill 跑在 T=4000–16000（+2.33%）
- **`qkrope`** 消除的是 6 个注意力层里的工作 → 并发 decode 最受益（+5.42%）
- **`moesum`** 消除的是 launch + HBM 往返，**小 T 最赚** → 低批 decode 最受益（+4.55%）

**四种不同形状的收益。只测一个 regime 一个都看不全。**

`gate+idx` **三个 regime 全不显著**——诚实负面：机制在 kernel 级真实可测（1~2%），但**没能兑现到端到端**。

---

## 8. ★ 最有价值的产出：同类优化强烈次可加

| regime | 各项之和 | 一起测 | 兑现率 |
|---|---:|---:|---:|
| C 长 prefill | 5.86% | **5.30%** | 0.90 |
| A 低批 decode | 9.37% | **6.57%** | 0.70 |
| B 并发 decode | 12.80% | **6.21%** | **0.49** |

并发 decode 上：`qkrope` 单独 +5.42%，再加单独值 +3.65% 的 `norm+scale+conv` 只多买到 **0.12 点**；再加单独值 +3.08% 的 `moesum` 又只多买到 **0.19 点**。三者都在消除**同一份"固定每-forward 开销"的余量**，消完之后别的东西成为瓶颈。

**兑现率的排序精确跟踪 regime 的饱和程度**：长 prefill 每 forward 工作最多、最能把开销藏起来，损失最小（0.90）；并发 decode 最饱和，损失最大（0.49，不到一半）。

这与 regime-kernel 研究的 waterfall 非叠加（serving 1.78× + kernel 1.22× → **1.70×** 而非 2.17×）是同一现象。**两个独立研究都撞上，可以固化成规则：**

> **消除同一"种类"成本的优化不会相加。报告各项分别测量之和会高估整个 stack，且系统越饱和高估越严重。任何会真实部署的组合都必须按组合测量。**

实践含义：**最便宜的组件反而最有价值**。`qkrope` 是纯调用点改动，单独就拿下并发 decode 的大部分空间。

---

## 9. 结论、诚实范围与自我纠错

### 9.1 对项目既有结论的修正

之前的立场"成熟 bf16 MoE 上 kernel 层不转化为端到端收益"**仍然成立**，但补上边界条件：

> **覆盖空缺是"架构成熟度"的函数，不是 sglang 的属性。** 上游优化过的模型族（Qwen3-30B：1 个未融合 norm、0 个独立 add）在融合层没剩空间。新加入的架构（LFM2.5：61 + 48 + 36 + 一条未融合的 QK-norm+RoPE 链 + 一次 MoE 归约往返）带着 **6.6%** 的纯开销——**不在它的新算子里**（`causal_conv1d` 只占 0.7%，很快），**在周围的调用点胶水上**。

最锋利的版本：**两个最大的赢家都是"sglang 已有融合原语、这个模型的调用点没用"**（`fused_add_rmsnorm`、`fused_qk_norm_rope`），加上一个乘以 1.0 和一个冗余 `.to(int32)`。两个真正需要写 kernel 的，都是相邻行方向工作的机械融合，而且 **Inductor 自己就能推导出其中一个**。**全程没有发明任何新东西。**

由此得到**第二条可机械检查的 signature**：**枚举代码库里已有的融合原语，检查哪些模型的调用点没用它们**——纯静态、不需要 profiling，这一条就找到了最大的两个赢家。

### 9.2 诚实范围
- 绝对值 ~5–6.6%，**单模型单卡**
- `norm` / `qkrope` 非 bit-identical，质量结论依赖噪声底 0.8 点的任务指标
- 大部分收益来自**补漏用的融合原语**，不是新 kernel
- 一个组件（`gate+idx`）是**实测负面**
- §8 表明这个 stack **不会**交付各部分之和

### 9.3 过程中发现并修正的自身错误
1. **统计方法**：原用正态近似算 p，在 n=6 下 anti-conservative。改用精确 Student-t 后**无结论翻转**，但文档里"全部 p<0.005"是错的（`qkrope` 在 C 实为 0.018），已改为逐格 p 值
2. **数据丢失 bug**：`lf_bench_shortconv.py --tokens` 做局部扫描会**静默覆盖**完整曲线，已加 `--out` 并恢复数据
3. **正确性门禁选错**：最初用 token-identity，被专家路由的离散性证伪，改用 GSM8K
4. **一次实验失败被正确捕获**：regime A 的一臂 `rc=-9`（两个 server 争资源），harness 记为 `launch_failed` 而非静默丢弃，已单独重测

---

## 10. 产物地图

### 文档
| 文件 | 内容 |
|---|---|
| **`docs/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md`** | 本文（总结） |
| `docs/lfm_fusion_results.md` | fusion 线主报告（577 行，含全部方法学细节） |
| `docs/regime_kernel_results.md` §11c | K1 跨模型 backend |
| `results/lfm_fusion/nsys/FINDINGS.md` | nsys agent 完整证据 |
| `results/lfm_fusion/fx/FINDINGS.md` | FX/Inductor agent 完整证据（637 行） |
| `results/lfm_fusion/moesum/FINDINGS.md` | moesum agent 完整证据 |
| `HANDOFF_regime_kernel.md` | 新会话交接（含新踩的坑） |

### 脚本 `scripts/lfm_fusion/`
| 文件 | 作用 |
|---|---|
| `lf_audit.py` | 算子级审计 + fusion gap 签名检测 |
| `lfm_fusion_patch.py` | **七个组件的实现**，`LFM_FUSION_PATCH` opt-in |
| `lf_inject/sitecustomize.py` | `sys.meta_path` finder 注入（避免懒加载竞态） |
| `lf_triton_shortconv.py` | **手写 Triton kernel ①**（gate+transpose 双侧） |
| `lf_triton_moesum.py` | **手写 Triton kernel ②**（MoE 归约+norm） |
| `lf_tune_shortconv.py` / `lf_bench_*.py` | tile 扫描 / 正确性门禁微基准 |
| `lf_e2e.py` | 端到端 A/B（含 patch 生效校验） |
| `lf_correctness.py` | logprob 门禁 + GSM8K 质量门禁 |
| `lf_analyze.py` | 精确 Welch t，按 (runset, regime) 隔离 |
| `lf_plots.py` | 4 张图 |
| `nsys_*.py` / `fx_*.py` | 两个 agent 的分析脚本 |

### 图
`results/lfm_fusion/plots/`：`fusion_gaps_by_model` · `fusion_final_stack`（含次可加性）· `shortconv_crossover` · `fusion_e2e_by_regime`

### 复现
```bash
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$ENVDIR PATH=$ENVDIR/bin:$PATH HF_HOME=$PWD/.hf_cache

# 审计（LFM2.5 + Qwen 对照）
./scripts/lfm_fusion/run_audit_all.sh 5

# 端到端（七项全开）
python scripts/lfm_fusion/lf_e2e.py --regime A_low_batch_decode --gpu 4 \
    --arms baseline,all7 --reps 6
python scripts/lfm_fusion/lf_analyze.py --runset lfm25_all7 --out fusion_ab_all7.csv

# 质量门禁
python scripts/lfm_fusion/lf_correctness.py accuracy --arm all7 --gpu 5 \
    --num-questions 1319 --reps 3

# 部署
LFM_FUSION_PATCH=norm,scale,conv,gate,idx,qkrope,moesum \
PYTHONPATH=scripts/lfm_fusion/lf_inject:scripts/lfm_fusion \
python -m sglang.launch_server --model-path /data/hf/LFM2.5-8B-A1B ...
```

---

## 11. 建议的下一步

1. **把审计跑到其他新架构上**（~15 分钟/模型，**最便宜且最有价值**）。"架构成熟度决定 fusion 空缺"目前仍是**单模型观察**；在第二、第三个新架构上复现才能变成规律——而这正是把它做成 agent 检查的前提。
2. **把两条 signature 做成 agent 的机械检查**：(a) 随层数线性增长的 kernel 计数，Qwen 作为"干净"的对照；(b) **枚举已有融合原语、找没调用的模型**——纯静态，不需要 profiling。
3. **上游那两个 bit-exact 的修复**（乘以 1.0、`.to(int32)` 提升），review 成本几乎为零。
4. **去掉 ShortConv 的形状门控**：FX 研究实测 **GPU 侧 crossover 在 T=512 以下**，T≈2048 的门控只是因为 Triton 的 Python launch 把 wall time 钉在 ~19–30 µs。用 CUDA graph 捕获或预编译 launcher 就能让它全程可用。
5. **gating 融进 `causal_conv1d_update`**（decode 1.8–1.9%）—— 需要新 tensor 参数 + schema 变更 + `sgl-kernel` 重编译（`activation` 在到 C++ 前塌成 bool）。**唯一剩下的非调用点改动**，优先级最低。
