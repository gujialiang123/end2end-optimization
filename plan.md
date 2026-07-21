# Plan / 项目状态（2026-07-20 更新）

## ★最新汇总文档：`docs/2026-07-20/qwen_optimization_full_report.md`（Qwen 全纪录，含 4 图 + §1.5 kernel 重写 + §1.6 kernel-config 调优 + §1.7 kernel-config e2e + §1.8 server/agent e2e）

### 2026-07-20 深夜：kernel-config tuning 的端到端验证（v42 + v43）
- **v42（bench_one_batch，全 regime，n=3）**：tuned config vs default 启发式 —— **prefill +34~43% e2e**（M≥2048）、**decode ≈0**；总 e2e（prefill+decode，out=32）+15~33%（依 in:out 比）。
- **v43（真实 server + bench_serving，全人造 regime + mooncake agent 数据集）**：**agent_toolagent E2E +17.5%/TTFT +27%/TPOT +14%**；prefill_medium/long E2E +23~25%/TTFT +24~34%；decode-heavy ≈0；短序列/高并发噪声内。
- **结论**：kernel-config tuning 端到端收益集中在 **prefill-heavy + 真实 agent 负载**（E2E +17~25%），decode 无实质收益。补上了之前"只有隔离 kernel µs"的 e2e gap。
- **Triton 版本**：tuning 基于 3.5.1，但 sglang config 目录无我们 shape → 回退加载 3.2.0 config（版本错配）；迁移须放进 `triton_3_5_1/`。
- 脚本 `run_v42_kernel_e2e.py`、`run_v43_server_e2e.py`；数据 `results/2026-07-20_v42_kernel_e2e/`、`v43_server_e2e/`。

### 2026-07-21：v45 — server 级 ours vs fallback，全 regime + agent 数据集（补 v44 的最后一块）✅
- **v45（真实 server + bench_serving，全 8 regime + mooncake toolagent agent 数据集，n=3，median + Welch t）**：v44 是 bench_one_batch，本节升级到真实 server 排队场景 + agent 负载。对比同 v44：**ours(3.6.0 重 tune) vs sglang 实际加载的 fallback(triton_3_2_0)**。
  - **结论：全 regime 端到端 ≈0。** 8 regime × 4 指标(TTFT/TPOT/E2E/out_tput)全在 ±2% 内，无一有意义加速；唯一显著的是 decode_heavy ~+1%（幅度微小）。**agent_toolagent 平手**（median TTFT −5% p=0.18、E2E −1% p=0.31）——注意 ours agent r0 是冷启动离群(TTFT 250ms vs 稳态 52ms)，用 median 正确滤掉。
  - **三层测量一致**：隔离 kernel（§1.6.5）+0.6% · bench_one_batch（v44）≈0 · server+agent（v45）≈0。**"重 tune 已被 fallback 覆盖的 shape = 端到端无用功"这个结论现在铁证如山，连真实 agent 负载也证实。**
  - **harness 修复**：sglang v0.5.12.post1 的 `bench_serving` mooncake 路径有 bug（两处把 list-of-dict 当 DatasetRow 访问 `.prompt`/`.prompt_len`），已修（`patches/sglang_bench_serving_mooncake_v0.5.12.post1.patch`）。
  - 文档 `docs/2026-07-20/kernel_config_server_ours_vs_fallback_e2e.md`；脚本 `scripts/run_v45_server_ours_vs_fallback.py`、`analyze_v45_server_ab.py`；raw `results/2026-07-21_v45_server_ours_vs_fallback/`（server_ab.jsonl 48 行 + 每 regime jsonl + 分析表）。

### 2026-07-20 深夜：迁移新机器 aifx-clou000001（8×H200, triton 3.6.0）+ config-tuning 第 3 层 e2e（v44）✅
- **v44 — re-tune vs fallback 的端到端 A/B（补上 §1.6.3 断言缺的 e2e 证据，与上面 v42/v43 互补）**：v42/v43 测的是 **default 启发式 vs tuned/fallback**（有 config 的价值，prefill +34~43%）；**v44 测的是 fallback vs 我们重新 tune** —— 在本机 triton 3.6.0 上重新 tune 未覆盖 shape 的 fused_moe config（18 桶全 tune，2149s），放进 `triton_3_6_0/` 让 sglang 优先加载，`bench_one_batch` A/B = **ours(3.6.0 重 tune) vs sglang 实际加载的 fallback(triton_3_2_0)**。
  - **结论：无端到端收益（≈0）。** 7 cell（decode b=1/8/32 各 n=8 + prefill in=512/1024/2048/4096 各 n=3）无一显著加速；唯二显著的是**小回归**（decode b1 −1.64%、prefill 1024 −2.75%），其余全在噪声内。**用真实 e2e 证实了 §1.6.3 那句"重 tune vs fallback 仅 +0.6%（隔离）"在端到端层面 = 0。**
  - **三层完整故事**：default 启发式 →(+34~43%)→ fallback config →(≈0)→ ours 重 tune。**config-tuning 的全部 e2e 收益来自"别掉进 default 启发式"，而非"按 triton 版本重 tune 已被 fallback 覆盖的 shape"。**
  - **噪声教训再现**：n=3 时 decode b8 显示 −8.84%（吓人），n=8 后塌成 +0.91%(p=0.93) 纯噪声 → 信号 vs 噪声必须多重复 + t 检验。
  - 文档 `docs/2026-07-20/kernel_config_retune_vs_fallback_e2e.md`；脚本 `scripts/run_v44_e2e_config_ab.py`、`analyze_v44_config_ab.py`；raw `results/2026-07-20_v44_retune_e2e_ab/`（config + tune.log + e2e_ab.jsonl 72 行 + 分析表）。
- **新机器环境**：triton 3.6.0 / torch 2.11 / CUDA 13.0；坑：`kernels==0.12.3`（sglang 未 pin 上界，pip 拉 0.16 破 transformers 5.6.0）、`CUDA_HOME=$CONDA_PREFIX`（conda 装 cuda-13.0 toolkit 提供 nvcc，deep_gemm import 期 JIT 建 _C）。

### 2026-07-20 晚：噪声验证（Chendi 要求）+ kernel 细节入报告
- **v41 噪声验证**：把 custom MoE kernel 的 b1 "+1.4%" 用 **n=15 交错重复 + Welch t 检验**验证 → **+1.17%，|t|=6.51，真信号（非波动）**；b2 −4.3%(|t|=3.2)、b4 −11.7%(|t|=9.9) 是**真回归**。文档 `docs/2026-07-20/noise_verification_custom_moe_b1.md`（带误差棒图）、脚本 `scripts/run_v41_noise_verify.py`。
- **报告补充**：§1.5 写清 custom kernel 具体改了什么（去 align/sort + 融合 w1+SwiGLU / w2+加权求和 + fp32 累加）；§1.6 写清 kernel-config 调优（Triton `fused_moe_kernel` meta 参数：decode +13%、prefill +35~54% kernel 时间，U 形；本质是 autotuning，且是隔离时间非 e2e）。
- **教训固化**：信号 vs 噪声必须多次重复 + t 检验（连 3 次中位数都可能误判方向）。


### 2026-07-21：PR 复现线开启 — #31558 (FLA l2norm 按 token 数重编译) ✅ 复现成功
- **背景**：用户给出一份"复现 SGLang 最新 PR (A=v0.5.15.post1, B=+单个 PR patch) 证明 agent 价值"的方案。从最易验证的 Qwen VLM + #31558 开始。
- **模型/环境**：`Qwen/Qwen3.6-35B-A3B-FP8`（hybrid linear-attn + VLM，走 FLA l2norm 路径）；新建 env `sglang-v515`（v0.5.15.post1, transformers 5.12.1, kernels 0.14.1, flashinfer 0.6.12），保留 sglang-dev 不动。
- **patch**：#31558 只改 3 行（l2norm kernel 的 `T` 从 `tl.constexpr` 改成 `do_not_specialize` runtime 标量）。PR 的 main 路径在 v0.5.15 不存在（后续重构），手动移植到 `srt/layers/attention/fla/l2norm.py`（逐行等价）。
- **机制（微基准 v46，铁证）**：baseline 对 10 个不同 token 数 **编译 10 个 kernel variant**；patched **编译 0 个**（复用 1 个）。完全复现上游 "N cubin → 1 cubin"。
- **端到端（v47，真冷 TRITON_CACHE_DIR）**：VLM 冷启动 + 8 个不同图片分辨率，patched 把首轮 8 分辨率总 TTFT **−13.7%**（4.005→3.454s，**Welch t=20.9, p≪0.001**）；每个新分辨率 baseline 稳定多付 **~70ms** l2norm 编译停顿（小分辨率上 +25~33% TTFT）。稳态/固定分辨率控制两 arm 相同（无回归）。
- **关键 confound**：Triton 磁盘缓存 `~/.triton/cache` 会掩盖效应，必须清 `TRITON_CACHE_DIR` 才测得真实编译代价（方案已预警）。
- 文档 `docs/2026-07-21/pr31558_fla_l2norm_recompile_repro.md`；脚本 `run_v46_*`、`run_v47_*`；patch `patches/l2norm_v0.5.15.post1_*.py`；raw `results/2026-07-21_v46_*`、`v47_*`。


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
