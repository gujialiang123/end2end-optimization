# 在 OLMo-2 和 Falcon-H1 上复现整套发现流程

**日期**：2026-08-04 · **GPU**：H200 #1/#3/#4 · **对照组**：Qwen3-30B
**方法**：`docs/2026-08-04/METHODOLOGY_three_layer_optimization.md`
**目标**：不是复现某个数字，而是检验**这套发现流程本身**能不能在没人看过的模型上再产出一次价值

---

## 0. 三行结论

1. **olmo2**：审计信号极干净，但**只有一个根因，而且是我们自己已经报过的**。
   完整审计**没有**找到已知之外的机会 —— 这是个阴性结果，如实记录。
2. **falconh1**（此前从未审计过）：找到**两个新东西**，其中一个是流程原本假设不存在的层。
3. **流程本身踩出一个上游 bug**：`bench_one_batch` 在任何 hybrid-mamba 模型上都会崩。

---

## 1. olmo2：一个根因，已知

### 1.1 计数信号非常干净

三个 regime，对照组 Qwen3-30B（框架优化得最充分的模型）：

| gap | olmo2（16 层） | qwen（48 层） |
|---|---|---|
| `gating_mul` | **64**（4.00/层） | 0 |
| `eager_norm_decomp` | **32**（2.00/层） | 0 |
| `eager_norm_pow` | **32**（2.00/层） | 0 |
| `eager_norm_rsqrt` | **32**（2.00/层） | 0 |
| `residual_add` | **32**（2.00/层） | 0 |
| `layout_copy` | 36（2.25/层） | 4（0.08/层） |

**每个计数都能被 16 整除，对照组全部为 0。** 按方法论的两条判定信号，这是最强的形态。

### 1.2 ★ 但 CUDA graph 对照证明这不是六个发现

`lf_audit` 默认关 CUDA graph（让每个算子现形），而 `olmo2_A_low_batch_decode_cg`
是开着跑的。两次运行**只差 capture mode**：

| gap | graph OFF | graph ON |
|---|---:|---:|
| `eager_norm_decomp` | 32 | **0** |
| `eager_norm_pow` | 32 | **0** |
| `eager_norm_rsqrt` | 32 | **0** |
| `gating_mul` | 64 | **0** |
| `layout_copy` | 36 | 5 |
| **`residual_add`** | **32** | **32** ← 只有它活下来 |
| `unfused_rmsnorm` | 33 | **65**（= 33 + 32） |
| **total kernel** | **1744 us** | **1435 us** → **17.7%** |

`unfused_rmsnorm` 从 33 涨到 65，正好 +32 —— q_norm/k_norm 改走了融合 kernel。
**所有消失的桶同源。**

### 1.3 根因：`_apply_qk_norm` 的 fall-through

`python/sglang/srt/models/olmo2.py:164-191`：

```python
if self.alt_stream is not None and get_is_capture_mode():
    ...  # self.q_norm(...) → forward_cuda → 融合 RMSNorm kernel
else:
    q = self.q_norm.forward_native(q)   # ← 显式走 eager 分解
    k = self.k_norm.forward_native(k)
```

`forward_native` 每次调用产生约 7 个 kernel（cast→pow→mean→rsqrt→mul→cast→mul），
2 次/层 × 16 层 = 32 次调用 ≈ 224 次 kernel launch，替代本可以是 32 次融合调用。

**这个 gap 已经在 `docs/kernel_fusion_catalogue.md` 里，也已经开了
上游 issue #33415 + draft PR #33416。**

> **所以 olmo2 那个 27.74% 的 headline headroom 是一个根因，不是一片富矿。**
> **完整审计没有找到第二个同量级的机会。**

### 1.4 唯一独立的 gap：`residual_add`，而且不能照抄 LFM2.5 的修法

它在 capture 模式下**活下来**（32 → 32），所以是真正独立的第二个 gap。

但**修法不能照搬**。olmo2 是 **norm-after** 架构
（`python/sglang/srt/models/olmo2.py:302-319`）：

```python
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = hidden_states + residual        # norm(x) + residual
```

而 sglang 的 `fused_add_rmsnorm` 算的是 **`norm(x + residual)`**
（`layers/layernorm.py:139-147`）—— **数学上不是同一个操作。**

LFM2.5 的 G1 `norm` 修法（改用 deferred-residual 约定）在这里**直接不成立**。
要吃掉这 1.95–2.53% 得写一个新的 `rmsnorm_then_add` kernel。

> ⚠️ **审计工具自己对这个桶的注释是错的**：
> `"standalone residual addition; absorbed by fused_add_rmsnorm"` ——
> 在 norm-after 架构上照着做会得到错的数值。**工具给候选，不给判定。**

---

## 2. falconh1：审不了 → 先修工具 → 找到两个新东西

### 2.1 ★ 上游 bug：`bench_one_batch` 在 hybrid-mamba 上必崩

第一次审计直接失败：

```
prefill 正常（0.78 s, 20421 token/s）
decode → AssertionError: Mamba selective_state_update backend not initialized.
         Call initialize_mamba_selective_state_update_backend() first.
```

查到根因：`initialize_mamba_selective_state_update_backend` **只在
`managers/scheduler.py:501` 被调用**，而 `bench_one_batch` 不构造 Scheduler。

> **影响面**：`python -m sglang.bench_one_batch --model-path <任何 mamba hybrid>`
> 都会 prefill 成功、decode 崩溃。Falcon-H1、以及推测 Nemotron-H / Zamba 等同类。
> **这是可以直接提给上游的 bug。**

绕过：`scripts/lfm_fusion/mamba_inject/sitecustomize.py` 在首次使用时懒初始化。
**这是我们工具的 workaround，不是建议的修法** —— 真正的修法属于上游。

### 2.2 gap 审计：比预期干净得多

| regime | stage | all_gaps | removable |
|---|---|---:|---:|
| A 低批 decode | decode | 2.83% | 1.34% |
| B 并发 decode | decode | 3.36% | 1.92% |
| C 长 prefill | decode | 5.00% | 2.72% |
| A/B/C | **prefill** | **6.38–7.60%** | **4.25–5.11%** |

唯一像样的是 prefill 的 `layout_copy`：**96 次 = 4.00/层**，占 4.31–4.66%。

**定位方法**（trace 关联，不靠猜）：96 个 kernel 事件 → `aten::copy_` →
全部 96 个的父节点是 `aten::clone` → 再上层是 `aten::contiguous`。

对应源码是**两处**：
- `mamba/causal_conv1d.py:60` —— `if x.stride(-1) != 1: x = x.contiguous()`。
  输入是 `hidden_states_B_C_p.transpose(0, 1)`（`mamba.py:501`），
  **transpose 之后 stride 必然不为 1，所以这个 copy 每层必触发。**
- `mamba/ops/ssd_combined.py:63/65/69` —— B、C、x 各一次 `.contiguous()`。

**和 LFM2.5 的 G3 `conv` 是同一形态**：为了迁就 conv 的 layout 而 transpose，
下游 kernel 再各自把它 materialise 回来。

### 2.3 ★★ 真正的新发现：SSD kernel 的 tile 是硬编码的 16

看 falconh1 prefill 的 kernel 构成（C 长 prefill，252 ms）：

| kernel | 次数 | 时间 | 占比 |
|---|---:|---:|---:|
| `_chunk_state_fwd_kernel` | 24 | 64.3 ms | **25.5%** |
| `_chunk_scan_fwd_kernel` | 24 | 51.3 ms | **20.3%** |
| `_state_passing_fwd_kernel` | 24 | 33.2 ms | **13.1%** |
| （三者合计） | | **148.8 ms** | **59.0%** |
| `direct_copy`（§2.2 的 contiguous） | 96 | 11.7 ms | 4.6% |

这三个 Triton kernel：

```python
# ssd_chunk_state.py:154-156, ssd_chunk_scan.py:95-97
BLOCK_SIZE_M: tl.constexpr = 16,
BLOCK_SIZE_N: tl.constexpr = 16,
BLOCK_SIZE_K: tl.constexpr = 16,
# ssd_state_passing.py:57
BLOCK_SIZE: tl.constexpr = 16,
```

- **没有 `@triton.autotune`**
- **调用点一个 BLOCK_SIZE 都不传**（实测 `grep -c` = 0）

→ **在 H200 上永远跑 16×16×16，与 shape 无关。**

> ★ **这在方法论文档里是个盲点。** 那份文档说 L2 需要「config 驱动的热 kernel」，
> 并断言 dense 模型上这个机制不存在、L2 不适用。
> **falconh1 证明还有第三种形态：热 kernel 的 tile 是硬编码 constexpr 默认值。**
> 它和 LFM2.5 缺 MoE config 是**同一类问题**（没人为这张卡调过），
> 只是到达方式不同 —— **不需要有 MoE 也能有 L2。**

量化实验见 §3。

---

## 3. SSD tile 扫描

<!-- FILL -->

---

## 4. 复现

```bash
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python
cd $REPO

# 审计（falconh1 必须带 mamba shim，否则 decode 崩）
$PY scripts/lfm_fusion/lf_audit.py --model olmo2 --regime A_low_batch_decode --gpu 1
MAMBA_SSU_AUTOINIT=triton PYTHONPATH=$REPO/scripts/lfm_fusion/mamba_inject \
    $PY scripts/lfm_fusion/lf_audit.py --model falconh1 --regime C_long_prefill --gpu 3
$PY scripts/lfm_fusion/lf_audit.py --model qwen --regime A_low_batch_decode --gpu 4   # 对照组

# 跨模型对照表 + CUDA graph 对照
$PY scripts/lfm_fusion/gap_table_2026_08_04.py

# SSD tile 扫描
$PY scripts/lfm_fusion/fh_sweep_ssd_tiles.py --gpu 3 --reps 3
```

| 产物 | 路径 |
|---|---|
| 逐模型审计 | `results/lfm_fusion/audit/{olmo2,falconh1,qwen}_*/audit.json` |
| 跨模型对照表 | `results/lfm_fusion/processed/gap_table_2026_08_04.json` |
| SSD tile 扫描 | `results/lfm_fusion/processed/falconh1_ssd_tile_sweep_*.json` |
| mamba 初始化 shim | `scripts/lfm_fusion/mamba_inject/sitecustomize.py` |
| SSD tile 注入 shim | `scripts/lfm_fusion/ssd_inject/sitecustomize.py` |
| 日志 | `logs/2026-08-04/audit_*.log`、`fh_ssd_sweep.log` |

---

## 5. 对方法论的三条修正

1. **§0 的目标写窄了**（已改）：原文把成功标准定为「L3 增量那一行横着看是否平」，
   那是验证已有结论；实际目标是**复现发现流程本身**。
   在窄框架下 `olmoe`（headroom 4.70%）是好选择因为它能填格子，
   **在实际目标下它是差选择，正因为 headroom 只有 4.70%。**

2. **§1 说「L2 需要 config 驱动的热 kernel，dense 模型上不存在」—— 不完整。**
   第三种形态：**热 kernel 的 tile 是硬编码 constexpr**（falconh1 的 SSD kernel）。
   判定问题应该从「有没有 JSON config」改成
   **「最热的 kernel 的 tile 参数是谁定的、有没有人为这张卡定过」**。

3. **审计工具的桶注释会误导。** `residual_add` 的注释说
   "absorbed by fused_add_rmsnorm"，在 norm-after 架构上是错的。
   **注释是按 LFM2.5 写的，跨模型时必须回到源码验证。**

---

## 6. 诚实结论

**这轮最有价值的产出是一个阴性结果和一个盲点。**

- 阴性：**olmo2 上完整审计没有找到已知之外的机会。**
  这恰恰是流程可信度的证据 —— 它不会为了产出而虚报 gap。
  27.74% 的 headline 数字经不起「同不同源」这一问。
- 盲点：**L2 的适用性判据不对。** 我们一直按「有没有 MoE config」判断，
  而 falconh1 说明「硬编码 tile 的热 Triton kernel」是同一类机会。

两者都不是靠跑更多 GPU 小时得到的，是靠**对照组**（Qwen）和
**天然对照**（CUDA graph 开关）得到的。