# Plan / 项目状态（2026-07-20 更新）

## ★最新汇总文档：`docs/2026-07-20/qwen_optimization_full_report.md`（Qwen 全纪录，含 4 图 + §1.5 kernel 重写 + §1.6 kernel-config 调优）

### 2026-07-20 晚：噪声验证（Chendi 要求）+ kernel 细节入报告
- **v41 噪声验证**：把 custom MoE kernel 的 b1 "+1.4%" 用 **n=15 交错重复 + Welch t 检验**验证 → **+1.17%，|t|=6.51，真信号（非波动）**；b2 −4.3%(|t|=3.2)、b4 −11.7%(|t|=9.9) 是**真回归**。文档 `docs/2026-07-20/noise_verification_custom_moe_b1.md`（带误差棒图）、脚本 `scripts/run_v41_noise_verify.py`。
- **报告补充**：§1.5 写清 custom kernel 具体改了什么（去 align/sort + 融合 w1+SwiGLU / w2+加权求和 + fp32 累加）；§1.6 写清 kernel-config 调优（Triton `fused_moe_kernel` meta 参数：decode +13%、prefill +35~54% kernel 时间，U 形；本质是 autotuning，且是隔离时间非 e2e）。
- **教训固化**：信号 vs 噪声必须多次重复 + t 检验（连 3 次中位数都可能误判方向）。


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

## 2026-07-20 autopilot 全 regime 复现结果（已推 main: ad5207f/6ab9d7e/5f04253）
**① kernel 改动全 regime 端到端**（含 agent 数据集，server+bench_serving 真实并发）：
- Qwen3 custom MoE：b1 **+1.4%**、b2 −2%、b4 −11%；agent c1(触发) −0.7%(噪声)、c32(全 fallback) −7%。→ **通用收益 ≈0，高并发反而负**。
- Qwen1.5 gate 融合：全 batch **~1.0×**。
- **教训固化：单点端到端会误导，必须扫 regime + 真实并发。** 之前的 1.23×/M≤4 结论被证伪。
- 文档：`docs/2026-07-20/regime_sweep_kernel_changes.md`（含最终 3改动×regime 矩阵）。

**② 新架构端到端 —— 唯一真实正结果 ✅**：线性注意力(LFM2.5-8B-A1B hybrid) vs 全注意力(Qwen3-30B)：
- decode 随上下文 scaling：**LFM +24% vs Qwen +57%**（bs=32, 512→8192）；bs=64：+16% vs +47%。
- Qwen 在 bs=32×16k **OOM**，LFM 的 O(1) 递归状态 KV 足迹仍可跑。
- 线性注意力把 O(context) KV cache 换成 O(1) 状态 = **架构级端到端杠杆，tuning/kernel 重写都够不到**。
- 图：`results/2026-07-20_v39_ctxscan/ctx_scaling.png`；文档：`docs/2026-07-20/new_architecture_linear_attention_e2e.md`。

**总诚实结论**：成熟 bf16/H200 MoE 的 **kernel 融合全 regime 端到端 ≈0**；真正"tuning 之外"的端到端提升 = **架构选择（长上下文并发用线性注意力）+ 投机解码（+23–30%）**，都不是 bf16 MoE kernel 重写。

## 下一步（优先级）
1. **（②延伸）served 分页 prefill 下量化 LFM vs Qwen 长上下文优势**：bench_one_batch 单发 prefill 在 16k 两模型都 OOM；server 路径能把 LFM 推得更远（deferred）。
2. **投机解码作为主端到端杠杆**：在 tuned baseline 上叠 spec 的 A/B（+23–30% 已有 A1b 数据，可做成 stacked headroom 图）。
3. **补 CUDA 融合空缺**（agent 可批量，低 e2e 杠杆但补覆盖）：`fused_linear_sigmoid_mul`、`fused_rmsnorm_gated`（`fused_gdn_gating` 已有 triton CUDA，非空缺）。
4. **Chendi 广度实验**（待新机器）：20 model × regime × config × 3 repeat AutoTuner 普适性。设计稿：`docs/2026-07-17/breadth_autotuner_experiment_design.md`。
5. **Mason 深度线**：`docs/2026-07-16/mason_roadmap_qwen3_moe_matrix.md`。

## 关键方法学教训
- 对比 kernel 性能**必须对标 sglang 真实 GPU 代码 + cudagraph**，不能用朴素 PyTorch baseline（否则得出误导性"加速"，如已撤回的 SwiGLU）。
- 所有数字都是**我们实测**，非 PR 自称。
