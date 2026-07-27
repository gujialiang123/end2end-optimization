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


### 2026-07-21：PR #31438 复现（VLM 多模态预处理并行化）✅ 正向 + 语义精确
- 第三个复现（同 Qwen3.6-35B VLM）。把图片 I/O + HF processor 从 tokenizer event loop 移到独立 worker 池（patched 默认 2 processor + 16 I/O worker）。
- **结果（暖态 burst A/B）**：默认 2 worker 图片 burst 请求吞吐 **+14.5%(c8) / +8.5%(c16)**，p99 TTFT 改善；**correctness 闸门通过**（baseline==patched==patched4w 贪心输出逐字一致）。
- **4 worker 不再提升**（此 workload 2 已够，与 PR 选默认 2 一致）；比上游 +80% 温和是因随机小图解码便宜、预处理非主瓶颈——机制真实，e2e 幅度依 workload。
- **移植**：5 纯 Python 文件，cherry-pick 解 2 冲突（保留 v515 的 SGL_USE_CUDA_IPC/get_global_server_args；丢弃 PR 附带的会崩的 BOS 块）。executor.py 零外部依赖。
- 文档 `docs/2026-07-21/pr31438_mm_preproc_parallel_repro.md`；脚本 `run_v49_pr31438_mm_preproc_ab.py`；raw `results/2026-07-21_v49_pr31438_mm_preproc/`；patch `patches/pr31438/`。

### 2026-07-21：PR #29007 复现（DeepSeek-V4 MoE TP allreduce NCCL 对称内存）✅ 强正结果
- **候选一 headline**（纯文本、语义不变、两边同配置）。模型 `sgl-project/DeepSeek-V4-Flash-FP8`（294GB, FP8, TP8, attention_backend=dsv4），两边都 `--enable-symm-mem`，唯一差异 = PR 是否把 MoE 输出分配进对称内存池（让 TP allreduce 走低延迟 NCCL 对称路径）。
- **结果（全 cell 一致正向，gain=改善）**：c1(4096/1024) **TPOT +9.2% / E2E +10.6% / 吞吐 +10.6%**；c8 +6~7%；c16 +5.3~6.4%。**复现且略超上游**（上游 −6.58% E2E / +7.05% tput）。n=2 两次高度一致。
- **移植**：PR 基于更新 main，cherry-pick 到 v0.5.15 解 2 处冲突（dp_attention 保留 v515 额外字段 + PR 默认值；deep_gemm 保留 v515 的 post_reorder_triton_kernel 只加 symmetric wrap）。5 个纯 Python 文件 baseline/patched 快照存 `patches/pr29007/`。
- **环境修复（关键）**：`--enable-symm-mem` 的 nccl_allocator JIT 编译需 CUDA+NCCL 头/库：`CPATH`+=targets/include & nccl/include，`LIBRARY_PATH`+=stubs:/usr/lib:nccl/lib，`ln -sf libnccl.so.2 libnccl.so`；清 /tmp/symm_allocator 缓存。
- 文档 `docs/2026-07-21/pr29007_dsv4_symm_mem_allreduce_repro.md`；脚本 `run_v48_dsv4_pr29007_ab.py`；raw `results/2026-07-21_v48_dsv4_pr29007/`。

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

## 2026-07-24/25 serving-ceiling campaign（全网格 · 双模型 · 六 regime）
**目标**：量化"仅靠 serving 配置能关闭多少端到端 gap",作为 profiling/kernel 线的交接依据。

**设计**（`docs/2026-07-24/qwen_serving_ceiling_methodology.md`）：
- 空间 = 4 个 serving knob 全网格 **192 配置/模型**(cap 8..128 × chunk{−1,2048,8192} × policy{lpm,fcfs} × mem{.75..*.90}),Qwen3-30B-A3B + LFM2.5-8B-A1B,单 H200 TP1 BF16。
- **backend/CUDA graph 冻结并逐配置从 server log 核验**(fa3 / auto / capture 完成),无 warm start,cookbook = 网格内 config 74。
- **六 regime 统一用一个流式客户端** `sglang.bench_serving --output-details`;一次 server 启动跑完六个 → transfer matrix 全是实测而非外推。
- sqlite 工作队列 → 多 GPU 并行 + 断点续跑 + 失败分类(基础设施故障重试一次,绝不假打分)。

**关键方法学发现(本轮最重要产出)**:
- 短 workload **未达稳态**会产生完全错误的结论。`R_long_prefill` 每次仅跑 0.33s/4 请求,rep0→rep4 漂移 **+36.5%**;同一配置单次 coverage 与 5-rep validation 最坏差 **5.2×**。丢弃 rep0 不足以解决。
- 修复 = 按**实测漂移**分配预热轮次(long_prefill 4 / medium·concurrent 2 / short·shared_prefix 1 / tool_agent 0)。
- 修复后漂移 → **≤1.2%**,5-rep 95% CI **≤2.6%**,且预热后的 1-rep coverage 与 5-rep validation 一致到 **0.978–1.000**。
- 两个未预热版本完整保留为证据:`results/2026-07-24_serving_ceiling{,_validation}_nowarmup/`。
- **教训固化:benchmark 必须先证明处于稳态,再谈配置差异;测量窗口 <1s 的 workload 尤其危险。**

**结果**(未预热版初步,预热版重跑中):
- cookbook 已匹配的 regime(medium_balanced / long_prefill)**天花板 ≈1%**,98% 配置被 cookbook 支配。
- 错配的 regime 有真实断崖:shared_prefix **+78.6%**(LFM)/**+27.7%**(Qwen),但 TPOT p95 **−21.5%/−31.7%** → 是 **TRADE-OFF 不是白拿**。
- 诚实负结果:LFM tool_agent 吞吐赢家仅 +0.5%,却让 TPOT p95 恶化 **82.6%**。
- **下行远大于上行**:最差配置在饱和 regime 掉 **60–72%**(cap=8 饿死批处理),而最好只涨 0.5–2.9%。
- **Transfer matrix 是最强证据**:long_prefill 赢家迁到 concurrent_decode 只有 **0.36×**;off-diagonal 几乎没有 >1.00×。
- 结论口径(见 `qwen_serving_ceiling_slide_claims.md` 的 14 条 claim + 禁用措辞表):
  "serving tuning 消除 workload 特定的配置断崖并在延迟/吞吐前沿上选点,但不提供通用配置,也不能把端到端前沿整体外推" → 交接 profiling。

**产物**:`results/2026-07-24_serving_ceiling/`(summary/pareto/5 个 transfer matrix/18 图 PNG+SVG/逐请求 parquet)、
`docs/2026-07-24/{serving_tuning_data_audit,qwen_serving_ceiling_methodology,qwen_serving_ceiling_results,qwen_serving_ceiling_slide_claims}.md`、
六页 slide 草稿 `performance_gap_slides_1to6_draft.pptx`、脚本 `scripts/{serving_ceiling_lib,run_serving_ceiling_campaign,run_serving_ceiling_validation,analyze_serving_ceiling,render_serving_ceiling_figures,update_performance_gap_slides}.py`。

## 2026-07-26 alternative-objective study（换 tuning 目标会怎样?）
**问题**:如果 autotuner 的目标不是 request throughput,而是 TTFT / TPOT / E2E / 平衡分,会选出不同的 config 吗?

**方法**:**不重跑网格、不新开 Optuna**。warmed 192 网格已把每个 config × 每个 workload 测完,换目标只是换"选谁"。离线重选 + 只补跑缺失验证(58 config × 6 workload × 5 rep = 1740 run,112,520 逐请求记录,零失败)。8 个策略:纯单指标 p95/p50、SLO 约束吞吐(1/3/5%)、约束 TTFT/TPOT/E2E、maximin、几何均值、strict/noise-tolerant 全指标、Pareto knee。

**答案**:
- **不同目标选不同 config**:12 单元平均 **4.8 个不同**(最多 7)。
- **吞吐赢家从来不是 TPOT 赢家(0/12)**,与 TTFT/E2E 赢家仅 3/12 重合。TPOT 优先反复选 `cap=8`,用 **−45%~−64% 吞吐**换 +24%~+47% TPOT —— 吞吐优先失效模式的镜像。
- **maximin 显著优于吞吐优先**:全指标胜 **8 vs 4**,回归 **2 vs 5**。→ **优化"最差指标"比优化"最好指标"稳健得多**(已存 memory)。
- 10/12 单元存在验证过的全指标赢家;lfm25 short-decode 与 qwen tool-agent 只有权衡。

**诚实负面结果**:
- **吞吐优先在 5/12 单元验证时回归** = 普通 **selection-on-noise**(192 含噪点取 argmax 系统性高估),集中在真实差异仅 ~1% 的饱和 regime。已用 cookbook anchor 排除时间窗漂移(new/old 0.995–1.043)。→ **教训:绝不能只凭单次网格搜索发布 serving tuning 结果。**
- coverage 级 12/12 有"全指标赢家",验证后剩 10/12 且来自**不同**策略(比值 1.001 是噪声)。
- 旧 62-config 验证集**大部分不可复用**(76 选中里仅 14 个有 5 rep),因为它选自修预热缺陷**之前**的网格。→ **选择集不能跨测量协议变更迁移。**

**主结论未变、反被强化**:换目标让你**沿着**前沿移动(且无需重搜),但**移不动前沿**;大幅收益仍只在 long-prefill / shared-prefix 两个断崖 regime。

**产物**:`results/2026-07-26_alternative_objectives/`(audit/plan/validated/comparison matrix/knobs/17 图/parquet/reproduce.sh)、`docs/2026-07-26/{alternative_objective_validation_audit,alternative_serving_objectives}.md`、脚本 `scripts/{analyze_alternative_serving_objectives,run_alternative_objective_validation,finalize_alternative_objectives,render_alternative_objective_figures}.py`。

## 2026-07-26/27 Regime-aware Kernel Specialization（P0 完成）
**问题**:regime 不同 → kernel workload 不同 → 最优 kernel config 是否也不同?能否自动化选择/验证/部署?

**切入点(审计发现)**:两个模型在这台 H200 上都跑在**未调优的 MoE Triton config**(LFM2.5 `E=32,N=1792` 无配置文件;Qwen `E=128,N=768` 回退 triton 3.2.0)。而 SGLang 的 MoE config **本身就是 `M → config` 映射 + 最近邻查找** —— regime-aware specialization 是运行时已有、但没人填过的机制。不写新 CUDA kernel;`override_config`(micro)/`SGLANG_MOE_CONFIG_DIR`(E2E)开关,默认路径不变。

**最终 E2E**(LFM2.5,serving 参数冻结,只换 kernel profile,6 次重复):
| regime | global-best | regime-aware(朴素) | **guarded(M 修正)** |
|---|---:|---:|---:|
| 低批 decode | 0.923× | 0.745× | **0.998×** 中性 |
| 并发 decode | 1.004× | 1.060× | **1.014×** |
| 长 prefill | **0.796×** | 1.170× | **1.223×**(6/6 不重叠) |

**四个 RQ 全部有答案**:①最优 config 随 regime 系统移动(BLOCK_M 16→128,算术强度 0.083→170.7 FLOP/byte);②迁移最差 **0.123×(慢 8 倍)**;③单一 global profile **有害**(大 M 0.39–0.62×),3 个 profile 恢复 73–100% oracle;④agent 对不同诊断走不同动作序列,48 候选超过 240 候选穷举。

**三个方法学发现(最有价值)**:
1. **必须跑服务器真实执行的 kernel 变体**。LFM2.5 有 `use_expert_bias`;调 no-bias 变体得到的 1.067× 是**服务器从不执行的代码路径**的假象,部署损失 25%。真实 with-bias 空间 1.007×。
2. **`M` 是 token 数,不是 `tokens × top_k`**(`M = min(num_tokens, CHUNK_SIZE)`)。我的 profile key 大了 4 倍 → 配置错位到相邻桶,**掩盖了真实空间**。只有 trace 活服务器才暴露。修正后:M≤32 无空间,**M≥64 有 1.39–1.64×**,crossover 在 M≈64。
3. **CUDA graph 重放 decode**,config 在 capture 时烘焙 → 稳态 decode **零次**配置查找;regime A 的 MoE 调用其实是 prompt prefill(实测 M=101–125)。
   附:并发 decode 的双峰纯粹是 **Triton JIT 重编译**,warm cache 下 8/8 干净 → 部署新 config 必须预热 cache。

**产物**:`scripts/regime_kernel/`(9 脚本)· `configs/regime_kernel/profiles/`(可直接 `SGLANG_MOE_CONFIG_DIR` 部署)· `results/regime_kernel/`(raw+processed+traces+9 图)· `docs/regime_kernel_{status,experiment_plan,results}.md`。所有性能数字均经 correctness 门禁,~9000 配置零正确性失败。

**P1 待办**:`_down` 伴随配置(受 `BLOCK_SIZE_M` 必须相等约束)· cookbook→serving→kernel 完整 waterfall · CUDA graph 下的运行时 bucket dispatch · 提高 `cuda_graph_max_bs` 让 decode 也能进入有空间的 M 区间 · 第二个模型族。

### 2026-07-27 补充:regime-kernel 的两次修正 + K1
- **修正1**:`M` 是 token 数不是 `tokens×top_k`;profile key 曾大 4 倍,掩盖了真实空间。修正后 crossover 在 **M≈64**(≤32 无空间,≥64 有 1.39–1.64×)。
- **修正2**:routing 交叉验证证明 **routing-specific 调优是拟合噪声**(4 个 M 里 3 个,skewed 调的配置在 skewed 上反输给 uniform 调的)。→ **config 调优是 shape-dependent,不是 regime-dependent**;而运行时本来就按 M 分发。
- **K1(新增,最强结果)**:换 **kernel 实现**(MoE runner backend)才是真正 regime-dependent —— 排序随 regime 翻转(cutlass 并发 decode 1.017× / 长 prefill 0.664×;triton_kernel 低批 decode 0.650×)。且 **backend 是启动时定死、全程不变**,所以"按 regime 选 backend"是真正缺失的能力。但不对称:选对最多 +1.7%,选错损失 34–35% → **避坑杠杆**。
- **baseline 局限**:LFM2.5 的默认值是两分支启发式(无 num_warps/stages),诚实表述是"这个 shape 从没调过",不是"我们优化了 kernel"。Qwen 有真实调优配置,空间仅 0.96–1.23×,这个对比本身是结果。
- **交接文档**:`HANDOFF_regime_kernel.md`(新会话从这里开始)。
