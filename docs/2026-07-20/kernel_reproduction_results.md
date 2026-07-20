# 复现 kernel-level 提升 —— 真实实测证据（在真实模型上端到端）

**日期**：2026-07-20 · 目的:按要求**真复现**那些 kernel PR 的技术,在真实模型上端到端量,得到我们自己的实测数字。

## 复现清单与结果

### 复现 1:small-M MoE 特化 kernel（Qwen3-30B-A3B，对应 MoE Align&Sort / 路由融合类）
- 手法:tensor-core + 跳过 align/sort + 融合 act/sum + tuned tiling（v30/v31）。
- **隔离层:1.23×（且比 sglang 更准）；端到端(插进 sglang decode)：+1.5% TPOT**（4.24→4.18ms，正确性 3.95% rel err）。

### 复现 2:shared-expert gate 融合（Qwen1.5-MoE-A2.7B，对应 PR #22325 `fused_linear_sigmoid_mul`）
- **这是 sglang 真实的 CUDA 空缺**:qwen2_moe 在 GPU 上跑 `F.sigmoid(gate(x))*shared_out` 三个未融合 kernel（只有 Intel AMX 有融合版）。
- 手法:写融合 triton kernel（linear+sigmoid+mul → 1 launch），monkeypatch 进 `Qwen2MoeSparseMoeBlock._forward_shared_experts`。
- **正确性 ✅**:就地对比 sglang 未融合版,max 1.4% rel err（bf16 级别）。
- **端到端**:baseline **3.34ms** → 融合 **3.36ms** = **~0%（无提升,甚至略慢）**。
  - 原因:gate 是极小算子（Linear 2048→1）+ **cudagraph 已经隐藏了那 2 个被省掉的 launch**,所以 GPU 时间线上几乎没变化。

## 真实证据的结论（诚实）
1. **两个 kernel 融合技术都成功复现、数值正确,但端到端 ~0–1.5%**（Qwen3 MoE 1.5% / Qwen1.5 gate 0%）。
2. **别人 PR 报的大数字（Align&Sort 3–10×、Kimi +162%）是"组件级 micro-benchmark"或"AMD/新架构/量化"**,不是成熟 bf16/H200 路径上的端到端。我们端到端复现后,增益回落到个位数%甚至 0。
3. **物理原因**:decode 是 memory-bound（我们审计:MoE+dense+attn=89% 全是权重/KV 读取）;融合小算子省的是 launch 开销,而 **cudagraph 本来就把 launch 开销藏掉了** → 端到端无感。
4. **组件级提升要转化为端到端,前提是该组件占比大**。例:MoE align/sort 3× 很硬,但审计里 align+sort 只占 decode ~4% → 3× → 端到端上限 ~2.7%。

## 那"换模型 kernel 还有空间吗"的最终答案
- **有,但要看在哪**：成熟 bf16/H200/标准 MoE（Qwen3-30B）几乎没有端到端空间（我们两次复现证实）;
- **真正有端到端空间的是**：① **新架构**（线性注意力/GDN/indexer,如 Qwen3-Next、DeepSeek-V3.2 indexer —— 别人 PR：Indexer Prologue Fusion 12→4 kernel、bs=1 **+8%**）;② **AMD/新硬件**;③ **量化路径**。
- **但即便如此,端到端最大杠杆仍是 spec decoding（+23~30%），不是 kernel 融合**。

## 产物
- 复现脚本:`scripts/qwen15_gate_patch.py`（gate 融合 kernel+patch）、`run_e2e_qwen15*.py`；`scripts/custom_moe_patch.py`（Qwen3 MoE）
- 数据:本文档 + `kernel_optimization_attempt_log.md`（§7 端到端）
