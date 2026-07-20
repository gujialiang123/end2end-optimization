# Kernel-Level 性能提升证据（能否靠"改 kernel 代码"拿到加速？）

**日期**：2026-07-20
**目标（纠正后）**：证明**在 kernel 层面（真正改/写 kernel 代码，而非挑 config 参数）能拿到性能提升**。这对应 Mason 说的"changing the kernel code"层、以及那个关键指标 **X**（kernel 手改相对 autotuning 的额外提升）。
**约束**：bf16（不碰量化）· NVIDIA H200 · triton 3.5.1。

> ⚠️ 与前一份报告的区别：`docs/2026-07-19/pr_validation_report.md` 验证的是 **config-tuning 类**（#27112/#20565/#18969）——那只是给现有 kernel 挑 constexpr 参数，属于 **autotuning 故事（小 X）**，**不是** kernel-level 改动。本报告回答的是"**kernel 代码本身改动能否加速**（大 X）"。

---

## 0. TL;DR

- **能，但要挑对目标 —— 且必须对标 sglang 的真实 GPU 代码（不是朴素 PyTorch）。**
- **有效证据（gate 融合 / #22325）**：我们写的融合 triton kernel，**对标 sglang GPU 上真正跑的 3 算子代码**（`F.sigmoid(gate(x))*shared_out`，核实 sglang 的 CUDA 路径确实没融合，融合版只有 CPU 实现）→ 实测 **2.0–3.1×**，profiler 确认 kernel 数 3→1，数值正确。**这是对 sglang 现状的真实 kernel 层改进。**
- **无效证据（已撤回，SwiGLU）**：sglang GPU 上早已用 `silu_and_mul` CUDA 融合 kernel；我最初拿朴素 PyTorch 当 baseline 得到的 1.4–1.66× **不成立**（见 §2b）。
- **方法学教训**：kernel 层"有没有提升空间"必须**对标框架的真实 kernel**，否则会得出误导性结论。所有数字均为**我们实测**，非 PR 自称。

### 数据来源与 baseline 声明（回答"baseline 是什么、谁测的"）
| 项 | baseline | 是否=sglang 真实 GPU 代码 | 结果来源 | 结论 |
|---|---|---|---|---|
| gate 融合（#22325）| `F.sigmoid(Linear(x))*shared_out`（3 算子）| ✅ 是（sglang CUDA 路径未融合；融合版仅 CPU）| **我们实测** | **有效，2–3×** |
| SwiGLU 融合 | `F.silu(gate)*up`（朴素 PyTorch）| ❌ 否（sglang 已有 `silu_and_mul` CUDA 融合）| 我们实测 | **无效，撤回** |

---

## 1. 先分清：哪些 PR 是真正的 "kernel-code 改动"

| PR | 内容 | 类型 | bf16 | 能否复现 |
|---|---|---|---|---|
| **#22325** | fuse **linear+sigmoid+mul** in shared_experts | **KERNEL 融合** | ✅ | ✅ **已复现（本报告）** |
| **#26727** | Qwen shared-expert **四算子融合** | **KERNEL 融合** | ✅ | 同族（可扩展）|
| #31370 | fold padded-topk_ids fill into fused shared-experts append+remap | KERNEL 融合 | ✅ | 较难（需 MoE dispatch 上下文）|
| #31552 | Marlin MoE occupancy-aware launch **specialization** | KERNEL 特化 | ❌ Marlin/量化 | 否 |
| #27211 | fused GEMM + DeepEP combine (CuteDSL) | KERNEL 融合 | ❌ FP8/DeepEP | 否 |
| #28666 | fuse shared_expert_gate GEMV into MoE append | KERNEL 融合 | ✅ | AMD/HIP only |
| #31470/#31408 | FlashInfer MegaMOE（新 kernel）| KERNEL 新写 | ❌ NVFP4/MXFP8 | 否 |
| #27112/#20565/#18969 | add tuned fused_moe **config** | CONFIG（非 kernel）| ✅ | 是（但属 autotuning 故事）|

**结论**：bf16/NVIDIA 上能干净复现的 kernel-code 改动 = **#22325 / #26727（shared-expert 融合）**。我们选 #22325 实际实现并验证。

---

## 2. 复现 #22325：写一个融合 kernel，实测加速 ✅

### 2.1 融合对象
Qwen2-MoE / Qwen3.5 的 shared-expert gate epilogue：
```
g   = sigmoid( x @ w_gate )     # x:[M,H]  w_gate:[H,1] -> [M,1]  (GEMV)
out = g * shared_out            # 每行一个标量，广播乘到 [M,H]
```
- **未融合**：`matmul` + `sigmoid` + `mul` = **3 个 CUDA kernel、3 次 launch**。
- **融合（我们写的）**：一个 triton kernel 内完成 GEMV 归约 + sigmoid + 广播乘 = **1 次 launch**。
- 代码：`scripts/run_v25_kernel_fusion.py`（`fused_gate_kernel`）。

### 2.1b baseline = sglang 的真实 GPU 代码（关键，已核实）
这一点最重要——我们的 baseline **不是随手写的朴素版本，而是 sglang 在 GPU 上真正跑的代码**：
- sglang `models/qwen2_moe.py::_forward_shared_experts`（GPU/else 分支）：
  ```python
  shared_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
  ```
  正是 `linear → sigmoid → mul` 三个独立算子。
- sglang **只有 CPU/Intel-AMX** 分支才调融合的 `torch.ops.sgl_kernel.fused_linear_sigmoid_mul`；核实 `sgl-kernel/csrc/` 后确认该融合 kernel **只有 CPU 实现（`torch::kCPU`），没有 CUDA 版**。
- **→ 所以 sglang 在 GPU 上确实没有融合这几个算子，我们的融合 kernel 是对 sglang 现状的真实改进，不是打稻草人。** 这也正是 #22325/#28666（把 shared_expert_gate 融进去）这一类 PR 想补的 CUDA 空缺。

### 2.2 profiler 确认 kernel 数 3→1
```
UNFUSED: 3 kernels  → nvjet_tst_...(GEMV) + sigmoid(vectorized_elementwise) + elementwise(mul)
FUSED:   1 kernel   → fused_gate_kernel
```

### 2.3 实测加速（H200，bf16，hidden=2048，300 iters，flush L2，数值校验通过）
| batch | 未融合 (µs) | 融合 (µs) | **加速** | 数值 |
|---|---|---|---|---|
| 1 | 15.07 | 6.91 | **2.18×** | OK |
| 8 | 14.72 | 6.82 | **2.16×** | OK |
| 32 | 15.07 | 7.30 | **2.07×** | OK |
| 64 | 16.19 | 7.33 | **2.21×** | OK |
| 128 | 16.48 | 7.78 | **2.12×** | OK |
| 256 | 16.13 | 7.84 | **2.06×** | OK |
| 512 | 17.63 | 8.80 | **2.00×** | OK |
| 1024 | 29.70 | 9.63 | **3.08×** | OK |
| 4096 | 40.00 | 22.59 | **1.77×** | OK |

### 2.4 结论
- **kernel-code 改动确实带来加速**：对标 sglang 真实 GPU 代码，融合 kernel 在 decode~prefill 全程 **1.8–3.1×**（对该 gate epilogue），数值正确、launch 数 3→1。
- 机制 = **省 kernel launch 开销**（小 batch/decode 尤其明显，这几个算子极小、几乎全是 launch 开销）。
- **定位（诚实）**：这是"kernel 层对 **sglang 未优化部分** 有真实提升"的证据；但该算子小、净收益有限（见 §2.5）。它证明**手法可行**，还**不等于**证明了对 sglang 已 tuned 核心 kernel 的"大 X"（见 §3）。

### 2.5 端到端语境（诚实标注）
- 该 gate epilogue 在 shared-expert 路径里占比：**decode ~27% / prefill ~9%**（见 `run_v24_shared_expert_fusion.py`）。所以对 shared-expert 块的净收益约 **decode ~15% / prefill 很小**。
- 但作为**方法学证据**，它证明了"agent 辅助写融合 kernel"这条路能真实拿到 kernel 层加速；同样的手法可推广到更大的融合目标（#26727 四算子、MoE 激活链等）。

---

## 2b. ⚠️ 撤回：SwiGLU 激活融合证据无效（sglang GPU 已融合）

**诚实更正**：我最初写了一个 SwiGLU（`silu(gate)*up`）融合 kernel，测得 1.4–1.66× —— 但那个 **baseline 是朴素 PyTorch `F.silu(gate)*up`（2 个算子），不是 sglang 的实际代码**。核实后发现：
- sglang `layers/activation.py` 的 `SiluAndMul.forward_cuda` 直接调 `sgl_kernel` 的 **`silu_and_mul` CUDA 融合 kernel** —— **sglang 在 GPU 上早就把 silu+mul 融合了**。
- 所以我的 1.4–1.66× 是打赢了朴素 PyTorch，**不是打赢 sglang 现状**。→ **此证据作废**，不能作为"kernel 层还能提升"的依据。
- （脚本 `run_v26_swiglu_fusion.py` 保留仅作反面教材：说明选错 baseline 会得出误导性结论。）

---

## 3. 为什么这回答了"agent 有没有意义"（诚实版）

- **有效但要清醒**：对标 sglang 真实 GPU 代码，shared-expert gate 确实**没融合**（融合版只有 CPU），我们的 kernel 拿到 **2–3×** —— 这是**真的 kernel 层改进**，证明"手法可行"。
- **但要承认 sglang 已经优化了大部分热点**：`silu_and_mul`、`fused_moe`、attention 等**都已是融合/tuned 的 CUDA kernel**。所以"随手找个没融合的算子融一下"这种低垂果实**大多已被 sglang 摘走**（我们的 SwiGLU 尝试就撞上了这个）。
- **真正的"大 X"仍未证明**：要拿到有分量的 kernel 提升，需要**打赢 sglang 现有的 tuned kernel**（例如 `fused_moe` 只有 7% 峰值算力，但它已是 tuned 的）——这比"融合未融合的小算子"难得多，是尚未解决的核心问题。
- **对 agent 的诚实结论**：本报告证明了 agent 辅助写融合 kernel **能对 sglang 未优化的部分拿到真实加速**（gate 融合 2–3×，虽小但真实）；能否对 sglang **已优化的核心 kernel** 再拿到提升（大 X），仍需后续验证。

---

## 4. 下一步（把 kernel-level 证据做强、并回答"大 X"）
1. **接入真实 sglang 前向**：把 gate 融合 kernel 挂进 sglang 的 `_forward_shared_experts`（GPU 分支），测端到端 decode TPOT 提升（把微基准 2–3× 折算成端到端 %，确认不是无关紧要的小数）。
2. **正面挑战 sglang 已 tuned 的核心 kernel**：选 `fused_moe`（NCU 显示 7% 峰值算力，但已 tuned）——**目标是 beat 它，而不是融合旁边的小算子**。这才是 Mason 的"大 X"主战场，也是唯一能证明"kernel agent 有独立价值"的硬骨头。
3. **系统扫描 sglang 未融合的 GPU 算子**：像 gate 这样"CPU 有融合、CUDA 没有"的缺口可能还有——这是 agent 能自动发现并补的低垂果实（虽单个收益小，但可批量化）。

## 5. 产物
- **有效**：`scripts/run_v25_kernel_fusion.py` + `results/2026-07-20_v25_kernel_fusion/shared_expert_gate_fusion.json`（gate 融合，对标 sglang 真实 GPU）
- **已撤回**：`scripts/run_v26_swiglu_fusion.py` + `swiglu_activation_fusion.json`（baseline 选错，仅留作反面教材）
- 机会分解：`scripts/run_v24_shared_expert_fusion.py`
- sglang 源码依据：`models/qwen2_moe.py:234-252`（GPU 未融合 gate）、`layers/activation.py:73-77`（silu_and_mul 已融合）、`sgl-kernel/csrc/cpu/gemm.cpp:725`（fused_linear_sigmoid_mul 仅 CPU）
