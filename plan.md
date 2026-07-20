# Plan / 项目状态（2026-07-20 更新）

## 当前主线：证明"kernel/config agent"有意义（能拿到 sglang 之外的性能提升）

### 复现 kernel PR 技术（2026-07-20，真实模型端到端实测）
- **复现1 small-M MoE kernel（Qwen3-30B）**：隔离 1.23× → e2e **+1.5%**。
- **复现2 shared-expert gate 融合（Qwen1.5-MoE-A2.7B，填 sglang CUDA 空缺 fused_linear_sigmoid_mul）**：正确（1.4% rel err）但 e2e **~0%**（3.34→3.36ms；gate 太小 + cudagraph 隐藏 launch）。
- **结论**：成熟 bf16/H200 标准 MoE 的 kernel 端到端空间 ~0–1.5%；别人报的大数字（Align&Sort 3–10×、Kimi +162%）是组件级/AMD/新架构/量化。真实 e2e 空间在**新架构（线性注意力/GDN/indexer，bs=1 +8%）**或 **spec decoding（+23–30%）**。
- 文档：`docs/2026-07-20/kernel_reproduction_results.md`、`kernel_headroom_other_models_pr_evidence.md`。
- 脚本：`scripts/qwen15_gate_patch.py`、`custom_moe_patch.py`、`run_e2e_*.py`。
- **待用户定**：是否转向新架构（线性注意力/GDN 小模型）复现那个正向的 bs=1 +8% kernel 证据。


### 最新（2026-07-20）：回答 Dey "tuning 以外还有多少空间" + decode 审计
- **decode-step 审计（v33，实测）**：Qwen3-30B b1 = MoE 41% + dense_gemm(qkv/o/lm_head) 32% + attn 16% = **89% memory-bound 权重/KV 读取**。
- **图（v34，`results/2026-07-20_v34_figures/`）**：Fig1 组成 / Fig2 MoE 带宽 vs batch（decode memory-bound→prefill compute-bound，**headroom 须按 regime 分开画**）/ **Fig3 headroom bars**：baseline→+kernel(+1.5%)→+spec(+6.6%c1/+30.6%c32)→roofline 1.85×。
- **spec e2e（A1b 实测）**：c1 +6.2%，c32 +23.4%。
- **CUDA 未融合空缺**：核实对 Qwen3-30B **不适用**（服务 qwen2_moe/GDN 等其他架构）；v25 已写好 fused_linear_sigmoid_mul 的 CUDA 替代（低 e2e 杠杆）。
- 说明文档：`docs/2026-07-20/headroom_beyond_tuning_figures.md`、`kernel_optimization_attempt_log.md`（§7 端到端、§8 审计、§9 CUDA 空缺）。


### 已完成的证据（按可信度）
1. **✅ Kernel-level 胜利(隔离) + 端到端现实检验**：M=1 单请求 decode MoE 自定义 kernel 隔离 **1.23× 且更准**；但**插进 sglang decode 路径测端到端只有 ~1.5% TPOT**（4.24→4.18ms，正确性 max 3.95% rel err）。→ **kernel 微基准加速不等比例转化到端到端**。
   - 文档：`docs/2026-07-20/kernel_optimization_attempt_log.md`（§7 端到端）
   - 脚本：`scripts/run_v27-v32*.py`、`scripts/custom_moe_patch.py`（sglang 集成 patch）、`scripts/run_e2e_*.py`
2. **✅ Kernel 融合空缺（shared-expert gate）**：sglang CUDA 路径未融合 `linear+sigmoid+mul`（融合版只有 CPU）→ 融合 CUDA kernel 2-3×（算子小）。
   - 文档：`docs/2026-07-20/kernel_level_improvement_evidence.md`
3. **✅ Config-tuning（autotuning 故事，非 kernel）**：未覆盖 shape 重 tune vs 默认启发式，prefill +54~67%（Qwen3-30B + DeepSeek-V2-Lite 两模型）。
   - 文档：`docs/2026-07-19/pr_validation_report.md`

### 核心诚实判断
sglang 的 MoE kernel 整体已高度优化（decode b≥32 达 74-84% HBM，prefill config-tuning 已 +50%）。**无损 kernel 空间集中在 sglang 未覆盖/未融合的边角**（M=1 decode、CUDA 未融合算子），而非重写已 tuned 的核心 GEMM。**而且即使拿到隔离层 1.23×，端到端也只剩 ~1.5%（kernel 重写对端到端杠杆很小）。**
→ **agent 定位 = "自动发现并补 sglang 的覆盖空缺"**；追端到端加速应优先算法层（spec −23%）/serving 层（并发 2.5×）/config（prefill +50%），而非重写核心 kernel。

## 下一步（优先级）
1. **端到端集成**：把 M=1 特化 MoE kernel 挂进 sglang decode 路径，测单请求 TPOT 真实提升（估 ~1.1×）。
2. **补 CUDA 融合空缺**（agent 可批量）：`fused_linear_sigmoid_mul`、`fused_gdn_gating`、`fused_rmsnorm_gated`（均 CPU-only，缺 CUDA）。
3. **扩大 M=1 胜利范围**（研究性，payoff 不确定）：M=2-8 的混合分组策略。
4. **Chendi 广度实验**（另一条线，待新机器）：20 model × regime × config × 3 repeat 的 AutoTuner 普适性 spreadsheet。设计稿：`docs/2026-07-17/breadth_autotuner_experiment_design.md`。
5. **Mason 深度线**：`docs/2026-07-16/mason_roadmap_qwen3_moe_matrix.md`（regime → config ceiling → kernel autotune → rewrite）。

## 关键方法学教训
- 对比 kernel 性能**必须对标 sglang 真实 GPU 代码 + cudagraph**，不能用朴素 PyTorch baseline（否则得出误导性"加速"，如已撤回的 SwiGLU）。
- 所有数字都是**我们实测**，非 PR 自称。
