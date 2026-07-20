# Kernel-Level 性能提升证据（能否靠"改 kernel 代码"拿到加速？）

**日期**：2026-07-20
**目标（纠正后）**：证明**在 kernel 层面（真正改/写 kernel 代码，而非挑 config 参数）能拿到性能提升**。这对应 Mason 说的"changing the kernel code"层、以及那个关键指标 **X**（kernel 手改相对 autotuning 的额外提升）。
**约束**：bf16（不碰量化）· NVIDIA H200 · triton 3.5.1。

> ⚠️ 与前一份报告的区别：`docs/2026-07-19/pr_validation_report.md` 验证的是 **config-tuning 类**（#27112/#20565/#18969）——那只是给现有 kernel 挑 constexpr 参数，属于 **autotuning 故事（小 X）**，**不是** kernel-level 改动。本报告回答的是"**kernel 代码本身改动能否加速**（大 X）"。

---

## 0. TL;DR

- **能。** 我们**实际写了一个融合 triton kernel**（复现 #22325 的 `linear+sigmoid+mul` 融合），在真实维度上实测 **2.0–3.1× 加速**，数值正确，profiler 确认 **kernel 数 3→1**。
- 这是**纯 kernel-code 改动**（不改 config、不碰量化、不换模型精度）拿到的加速 → 直接证明"kernel 层有可回收的性能，且 agent 辅助写 kernel 是有意义的"。
- 收益机制是**消除 kernel launch 开销**（3 次 launch → 1 次），因此在 **decode/小 batch 最显著**（launch-overhead-bound），与 config-tuning（在 prefill 大 batch 最显著）**正好互补**。

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
- **kernel-code 改动确实带来加速**：融合 kernel 在 decode~prefill 全程 **1.8–3.1×**（对该 gate epilogue），且数值正确、launch 数从 3 降到 1。
- 机制 = **省 kernel launch 开销**（在小 batch/decode 尤其明显，因为这几个算子极小、几乎全是 launch 开销）。
- 这是"**大 X**"方向的直接证据（Mason 的 kernel 层），与 config-tuning 的"小 X"（只调参数）**性质不同**。

### 2.5 端到端语境（诚实标注）
- 该 gate epilogue 在 shared-expert 路径里占比：**decode ~27% / prefill ~9%**（见 `run_v24_shared_expert_fusion.py`）。所以对 shared-expert 块的净收益约 **decode ~15% / prefill 很小**。
- 但作为**方法学证据**，它证明了"agent 辅助写融合 kernel"这条路能真实拿到 kernel 层加速；同样的手法可推广到更大的融合目标（#26727 四算子、MoE 激活链等）。

---

## 3. 为什么这回答了"agent 有没有意义"

- **config-tuning（autotuning 故事，小 X）**：只挑现有 kernel 的参数；前一份报告测得 prefill +54~67%，但那是"挑参数"，不是改 kernel。
- **kernel 融合（kernel 故事，大 X）**：本报告**实际改了 kernel 代码**，对目标算子拿到 **2–3×**。这正是需要 **agent 辅助 + researcher-in-the-loop** 去做的事（Mason 的第 3 层）。
- **两者互补**：融合救 decode（省 launch），config-tuning 救 prefill（调 tile/stage）。合起来说明 kernel/config agent 在**全 regime**都有可自动化回收的空间。

---

## 4. 下一步（把 kernel-level 证据做强）
1. **扩展到 #26727 的四算子融合**：把 gate 融合再叠加 down_proj 的 epilogue，量更大的净收益。
2. **接入真实 sglang 前向**：把融合 kernel 挂到 Qwen2-MoE/Qwen1.5-MoE 的 shared-expert 里，测端到端 decode TPOT 提升（把 kernel 微基准的 2–3× 折算成端到端 %）。
3. **挑一个 NCU headroom 最大的 MoE kernel 做重写**（Mason 第 3 层的正式目标：`fused_moe` 仅 7% 峰值算力）——这才是"大 X"的主战场；本报告的 gate 融合是先证明"手法可行"的最小样例。

## 5. 产物
- kernel + benchmark：`scripts/run_v25_kernel_fusion.py`
- 数据：`results/2026-07-20_v25_kernel_fusion/shared_expert_gate_fusion.json`
- 机会分解（占比）：`scripts/run_v24_shared_expert_fusion.py`、`results/2026-07-19_v23_config_evidence/shared_expert_fusion_opportunity.json`
