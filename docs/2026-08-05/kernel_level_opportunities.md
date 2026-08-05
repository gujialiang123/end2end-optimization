# kernel 级机会搜索：OLMo-2 与 Falcon-H1

**日期**：2026-08-05 · **GPU**：H200 #0–3 · **对照组**：Qwen3-30B
**问题**：除了 SSD tile 那个 config 级发现，**kernel 级（L3）还有没有值得做的改动？**

---

## 0. 结论

**有两个，都确认了。** 另外**推翻了我自己前一天的一个结论**，并**证伪了一个候选**。

| 项 | 模型 | 类型 | 结果 |
|---|---|---|---|
| **`foldmul`** | falconh1 | **权重折叠**（kernel 直接消失） | ✅ **+3.75% / +2.55% / +1.62%**，三 regime 全正 |
| **`normadd`** | olmo2 | **新写的 Triton kernel** | ✅ **+3.04% / +2.00%**（p=6.8e-08 / 7.2e-34） |
| `qknorm` | olmo2 | 调用点改动 | ⚠️ decode +0.90% / +0.69%；长 prefill 不可采信 |
| `convtriton` | falconh1 | 调用点改动 | ❌ **+0.55%，证伪** |
| SSD tile | falconh1 | config 级（L2） | ✅ +27.63%（8-04 已做） |

**两个 kernel 级改动合计**：falconh1 上 +1.6~3.8%，olmo2 上 +2.0~3.0%。

---

## 1. ★ 我前一天的结论错了：只看 decode 会漏掉主要 gap

8-04 我报告「olmo2 的六个 gap 桶收敛到一个已知根因，完整审计没找到别的」。
**收敛是真的，但我是只用 decode 阶段得出的，而 CUDA graph 只捕获 decode。**

开着 CUDA graph 重审（= 真实部署状态），prefill 全额付账：

| regime | prefill 总时间 | **qk-norm 簇** | residual_add |
|---|---:|---:|---:|
| A 低批 decode | 2189 us | **640 us（29.2%）** | 49 us（2.2%） |
| B 并发 decode | 31077 us | **6699 us（21.6%）** | 609 us（2.0%） |
| C 长 prefill | 81681 us | **16463 us（20.2%）** | 1487 us（1.8%） |

`gating_mul` / `layout_copy` / 三个 `eager_norm_*` 全是同一个 `forward_native`
分解的不同侧面（各 32–33 次 = 2/层）。

> **教训：CUDA graph 会掩盖 decode 上的 gap，但 prefill 永远不被捕获。
> 只审计 decode 会把一个 20–29% 的 prefill gap 报成「已解决」。**

---

## 2. `normadd`：本轮唯一确认的 kernel 级收益

### 2.1 为什么需要新写 kernel

olmo2 是 **norm-after**（`models/olmo2.py:302-319`）：

```python
hidden = self.post_attention_layernorm(hidden)
hidden = hidden + residual          # norm(x) + residual
```

而 sglang 现成的 `fused_add_rmsnorm` 算的是 **`norm(x + residual)`** ——
**数学上不是同一个函数**，所以 LFM2.5 上那套「抄现成融合原语」的做法在这里不适用。

`scripts/lfm_fusion/ol_triton_normadd.py`：一次读入 x、算 RMS、乘权重、加残差、写一次输出，
替代「RMSNorm kernel + 独立 elementwise add」两次 launch 和一次多余的全量读写。

**正确性：所有测试形状 bit-identical 100%**（T=4000 时唯一差异是单个元素的 bf16 舍入，
mean 6.6e-08）。

### 2.2 结果

| regime | baseline | patched | 变化 | p | 正/逆序 |
|---|---:|---:|---:|---|---|
| **B 并发 decode** | 53.401 | **55.024** | **+3.04%** | 6.8e-08 | +3.43 / +2.65 |
| **A 低批 decode** | 2.555 | **2.606** | **+2.00%** | 7.2e-34 | +1.95 / +2.06 |
| C 长 prefill ×10 | 82.869 | 83.838 | +1.17% | 0.31 **n.s.** | +0.54 / +1.83 |

### 2.3 ★ 第一版是 −3.5%，原因是我漏了形状门控

首次测量长 prefill 得到 **−3.5%**。microbench 直接给出原因（H=2048, bf16）：

| T | stock (norm+add) | 我的 kernel | 比值 |
|---:|---:|---:|---:|
| 1–2048 | 0.022 ms | 0.029 ms | **0.74–0.77×（stock 赢）** |
| 8192 | 0.059 ms | 0.047 ms | 1.25× |
| 16000 | 0.099 ms | 0.069 ms | 1.44× |

**kernel 每行一个 program，所以 launch 宽度 = token 数**；几千行以下 GPU 填不满，
省下的一次 launch 抵不过损失的并行度。

加上 `MIN_TOKENS=4096` 门控后，长 prefill 从 −3.5% 变成 **+1.17%（n.s.）**，
decode 不受影响（decode 恒在门控之下，走 stock 路）。

> ⚠️ **这是重复犯错**。2026-07 LFM2.5 那批手写 kernel 就总结过
> 「手写 kernel 必须有形状门控」，我还是先发了没门控的版本。记在这里而不是悄悄修掉。

---

## 3. `qknorm`：decode 上小幅正向，长 prefill 不可采信

改动很小：`_apply_qk_norm` 在非 capture 模式下不再绕过 dispatch 直接调
`forward_native`（一次 RMSNorm 被拆成约 7 个 eager kernel），改走正常的
`self.q_norm(...)`。

| regime | 变化 | p | 正/逆序 |
|---|---:|---|---|
| B 并发 decode | +0.90% | 4.6e-02 | +1.66 / +0.15 |
| A 低批 decode | +0.69% | 3.9e-18 | +0.58 / +0.80 |
| C 长 prefill ×10 | −2.42% | 2.4e-02 | **−0.01 / −4.83** |

**decode 上小是预期内的**：decode 走 capture 快路，本来就没有这个 gap。

**长 prefill 那格不可引用**，理由有三：
1. 两个顺序差 4.8 个百分点
2. 四个 server lifetime 的 baseline 跨 81–90 req/s（**10%**），而待测效应是 2%
3. microbench 显示替换路径在**所有**形状上快 3.4–9.6×，且两臂的 CUDA graph 使用完全相同
   （各 28 次 graph decode、33 次 prefill）——**找不到能解释这个回归的机制**

要解决需要每臂多个 lifetime。**如实记为未解决，不当作负面结论。**

---

## 3b. ★ `foldmul`：把四个常数乘数折进权重，kernel 直接消失

审计里 falconh1 的 `gating_mul`（修完 tile 后 **4.11%**）是每层四次**整张量标量乘**
（`models/falcon_h1.py:334-355`）：

```python
self_attention(hidden_states * attention_in_multiplier)
attention_hidden_states * attn_out_multiplier
mamba(hidden_states * ssm_in_multiplier)
mamba_hidden_states * ssm_out_multiplier
```

24 层 = **96 次 kernel launch**。

**关键观察**：四个乘数全是 config 常数，且每个都紧挨着一个线性层。
所以 `(x · a) @ W ≡ x @ (a · W)` —— **把常数折进相邻权重，这些 kernel 就彻底不存在了**，
而不是被融合进别的东西：

| 乘数 | 折进 |
|---|---|
| `attention_in` | `qkv_proj.weight` |
| `attn_out` | `o_proj.weight` |
| `ssm_in` | `mamba.in_proj.weight` |
| `ssm_out` | `mamba.out_proj.weight` |

四个投影在 Falcon-H1 上都无 bias（`attention_bias` / `mamba_proj_bias` /
`projectors_bias` 全 false），代码里**断言**了这一点而不是假设。

### 结果（三 regime 全正、全显著、正逆序一致）

| regime | baseline | foldmul | 变化 | p | 正/逆序 |
|---|---:|---:|---:|---|---|
| **A 低批 decode** | 1.091 | **1.132** | **+3.75%** | 1.4e-18 | +3.67 / +3.83 |
| **B 并发 decode** | 17.637 | **18.086** | **+2.55%** | 1.7e-07 | +1.82 / +3.29 |
| **C 长 prefill** | 9.591 | **9.747** | **+1.62%** | 4.3e-15 | +2.05 / +1.19 |

延迟：TTFT p50 −1.20%（p=4.7e-13）、E2E mean −1.65%（p=3.0e-19）。

正确性 4/5 逐 token 相同——第 5 个不同是因为**折叠改变了浮点乘法的顺序**，
预期内，所以这一项**不像 olmo2 的 kernel 那样声称 bit-exact**。

### 两个先踩到的坑

1. **把乘数设成 1.0 没用** —— `x * 1.0` 照样启动 kernel、照样全量读写。
   必须写一个**不含这些乘法**的 forward，而不是替换常数。
2. **不能在 patch 里重新 import `falcon_h1`** —— 会拿到还在初始化中的
   `sys.modules` 条目并抛 AttributeError。要用 import hook 传进来的 module 对象。

---

## 4. `convtriton`：证伪

falconh1 的 prefill 里有 96 次 `direct_copy`（4 次/层），来自
`causal_conv1d.py:60` 对 transposed view 的 `.contiguous()`。

**修法是现成的**：Triton 版 `causal_conv1d_fn` 直接读 stride
（`causal_conv1d_triton.py:441-449`，甚至有专门的 `is_channel_last` 分支），
且树里已有两个模型（granitemoehybrid、nemotron_h）设了 `use_triton_causal_conv=True`。

**实测 +0.55%，不值得。** copy 是真的、Triton 路确实能省掉它，
但 **CUDA 版 conv 本身快到足以抵消这个节省**。

> **这类结果必须记录。**「审计发现的 gap」不等于「端到端收益」——
> 中间隔着「替代实现是否同样快」这一步。

---

## 5. ★ Amdahl：gap 的占比会随基线变好而上升

8-04 修好 SSD tile 后重审 falconh1：

| | 修 tile 前 | 修 tile 后 | 绝对时间 |
|---|---:|---:|---|
| prefill 总 kernel 时间 | 252.5 ms | **152.7 ms（−39.5%）** | — |
| `layout_copy` | 4.66% | **7.72%** | 11779 us → 11786 us（**没变**） |
| `gating_mul` | 2.47% | **4.11%** | 6250 us → 6274 us（**没变**） |

**绝对时间纹丝不动，占比几乎翻倍。**

这正是我在 LFM2.5 那轮亲手总结过的效应，而我在 8-04 说「4.3–4.7%，太小不值得」时
**用的是一个已经过时的基线**。

> **规则：每修完一层，剩余 gap 的占比必须重算，不能沿用旧基线上的百分比。**

---

## 6. 一个测量陷阱：1B 模型上 `R_long_prefill` 不可用

| 模型 | `R_long_prefill` 窗口 |
|---|---:|
| LFM2.5 (8B) | 0.31 s |
| **olmo2 (1B)** | **43–56 ms** |

窗口 50 毫秒级时，两个顺序给出的符号相反（−3.1% / +8.7%），标准差是均值的 12%。

新增 `R_long_prefill_x10`（同形状，40 个 prompt 而非 4）。
**刻意做成新 workload 而不是改旧定义**，这样此前所有格子仍然可比。

---

## 7. 复现

```bash
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization; cd $REPO
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python

# 审计（olmo2 必须开 CUDA graph，否则漏掉 prefill gap）
$PY scripts/lfm_fusion/lf_audit.py --model olmo2 --regime C_long_prefill --gpu 1 --cuda-graph --tag _cg

# falconh1 在修好 tile 的基线上重审
MAMBA_SSU_AUTOINIT=triton SSD_TILES="chunk_state:64,64,64;chunk_scan:64,64,64" \
PYTHONPATH=$REPO/scripts/lfm_fusion/fh_inject \
  $PY scripts/lfm_fusion/lf_audit.py --model falconh1 --regime C_long_prefill --gpu 0 --tag _tiled

# 单项 e2e（每项独立，绝不攒批）
$PY scripts/lfm_fusion/lf_e2e.py --model olmo2 --regime OL_concurrent_decode \
    --gpu 2 --arms baseline,normadd --reps 10 --warmup 4 --tag _x --correctness-nogate
```

| 产物 | 路径 |
|---|---|
| 新 kernel | `scripts/lfm_fusion/ol_triton_normadd.py` |
| olmo2 注入补丁 | `scripts/lfm_fusion/olmo2_fusion_patch.py` + `ol_inject/` |
| falconh1 注入补丁 | `scripts/lfm_fusion/falcon_fusion_patch.py` + `fh_inject/` |
| 审计数据 | `results/lfm_fusion/audit/{olmo2_*_cg,falconh1_*_tiled}/` |
| e2e 数据 | `results/lfm_fusion/e2e/olmo2_ol{,y}_*/` |
| 统计输出 | `logs/2026-08-05/olmo2_final.txt` |

---

## 8. 对「profiling + 审计能不能找到 kernel 级机会」的回答

**能，但这轮的产出比 LFM2.5 那次（七项 ~7%）少，而且原因是可解释的。**

| | LFM2.5 | olmo2 | falconh1 |
|---|---|---|---|
| 层数 | 24 | 16 | 24 |
| 特殊结构 | 18 层 gated short conv | 无（标准 dense） | mamba2 hybrid |
| 找到并确认的项 | 7 | **1**（+1 未解决） | **1**（+1 证伪） |
| kernel 级收益 | ~6.5% | **+2.0~3.0%** | **+1.6~3.8%** |
| 另有 config 级 | +23.3%（MoE config） | 无 | **+27.6%（SSD tile）** |

**olmo2 是标准 dense 架构，可融合的点本来就少。** LFM2.5 的七项里有四项
（`conv`/`gate`/`idx`/`moesum`）依赖它特有的 gated short conv 和 MoE 结构。

**这轮真正的价值不在数字大小，而在三件事**：

1. **方法能纠正自己**：CUDA graph 对照揭穿了「六个 gap」实为一个；
   开着 graph 重审又揭穿了我「已解决」的误判。
2. **方法会给出阴性结果**：`convtriton` 被证伪，没有为了产出而美化。
3. **失败模式是可诊断的**：normadd 首测 −3.5%，microbench 一次就定位到形状门控，
   修完变成正的。**这说明流程不是碰运气，是有因果链的。**

---

## 9. 仍然开放

1. **qknorm 的长 prefill 格**（§3）——需要每臂 3–4 个 server lifetime 才能分辨
2. **`normadd` 的 `MIN_TOKENS` 只粗调过一次**（4096），没有扫过
3. ~~falconh1 的 `gating_mul`~~ —— **已完成，见 §3b（+1.6~3.8%）**
4. **falconh1 的 `layout_copy`（7.72%）仍未解决** —— `convtriton` 那条路被证伪（§4），
   但 copy 本身还在。真正的修法可能是让上游的投影直接产出 conv 需要的布局，
   而不是先 transpose 再 materialise —— 那要改 sglang 的 mamba mixer，成本更高。
