# Plan / 项目状态（2026-07-20 更新）

## 当前主线：证明"kernel/config agent"有意义（能拿到 sglang 之外的性能提升）

### 已完成的证据（按可信度）
1. **✅ Kernel-level 胜利（M=1 单请求 decode MoE）**：自定义 triton kernel **1.23× 且比 sglang 更准**（对 fp32：我们 4-5% err vs sglang 10-14%）。做法：tensor-core `tl.dot` + 跳过 align/sort + 融合 act/sum + tuned tiling。范围 M≤4；M≥8 sglang 反超。
   - 文档：`docs/2026-07-20/kernel_optimization_attempt_log.md`
   - 脚本：`scripts/run_v27-v32*.py`
2. **✅ Kernel 融合空缺（shared-expert gate）**：sglang CUDA 路径未融合 `linear+sigmoid+mul`（融合版只有 CPU）→ 融合 CUDA kernel 2-3×（算子小）。
   - 文档：`docs/2026-07-20/kernel_level_improvement_evidence.md`
3. **✅ Config-tuning（autotuning 故事，非 kernel）**：未覆盖 shape 重 tune vs 默认启发式，prefill +54~67%（Qwen3-30B + DeepSeek-V2-Lite 两模型）。
   - 文档：`docs/2026-07-19/pr_validation_report.md`

### 核心诚实判断
sglang 的 MoE kernel 整体已高度优化（decode b≥32 达 74-84% HBM，prefill config-tuning 已 +50%）。**无损 kernel 空间集中在 sglang 未覆盖/未融合的边角**（M=1 decode、CUDA 未融合算子），而非重写已 tuned 的核心 GEMM。
→ **agent 定位 = "自动发现并补 sglang 的覆盖空缺"**。

## 下一步（优先级）
1. **端到端集成**：把 M=1 特化 MoE kernel 挂进 sglang decode 路径，测单请求 TPOT 真实提升（估 ~1.1×）。
2. **补 CUDA 融合空缺**（agent 可批量）：`fused_linear_sigmoid_mul`、`fused_gdn_gating`、`fused_rmsnorm_gated`（均 CPU-only，缺 CUDA）。
3. **扩大 M=1 胜利范围**（研究性，payoff 不确定）：M=2-8 的混合分组策略。
4. **Chendi 广度实验**（另一条线，待新机器）：20 model × regime × config × 3 repeat 的 AutoTuner 普适性 spreadsheet。设计稿：`docs/2026-07-17/breadth_autotuner_experiment_design.md`。
5. **Mason 深度线**：`docs/2026-07-16/mason_roadmap_qwen3_moe_matrix.md`（regime → config ceiling → kernel autotune → rewrite）。

## 关键方法学教训
- 对比 kernel 性能**必须对标 sglang 真实 GPU 代码 + cudagraph**，不能用朴素 PyTorch baseline（否则得出误导性"加速"，如已撤回的 SwiGLU）。
- 所有数字都是**我们实测**，非 PR 自称。
