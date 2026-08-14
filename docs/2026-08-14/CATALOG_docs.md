# Documentation catalog / 文档目录总览

EN: This catalog indexes every Markdown document under `docs/` plus the repository-root Markdown files, ordered chronologically where dates exist. Dated directories are experiment records; named directories are living documentation. Each entry preserves the file's own authoritative title and adds a one-line bilingual gloss grounded in that title or summary.

中文：本目录索引 `docs/` 下的全部 Markdown 文档以及仓库根目录 Markdown 文件；有日期的文档按时间顺序排列。日期目录是实验记录，命名目录是持续维护的常驻文档。每一项都保留文件自身的权威标题，并给出基于标题或摘要的一句中英文说明。

## Chronological index / 时间线索引

### 2026-06-01

**EN:** The project began by defining and benchmarking regime-aware SGLang performance on H200 across Qwen dense and MoE models.  
**中文：**这一天启动了 H200 上面向 regime 的 SGLang 性能研究，对 Qwen dense 与 MoE 模型做基准与会议说明。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-01/meeting_brief_2026_06_01.md | SGLang regime-aware performance study — meeting brief | Meeting brief for Jialiang Gu's 2026-06-01 two-round SGLang regime experiment on 8× H200 with single-GPU runs. | 2026-06-01 的会议简报，概述 Jialiang Gu 在 8×H200 上以单 GPU 运行的两轮 SGLang regime 实验。 |
| docs/2026-06-01/regime_benchmark_experiment.md | Regime benchmark experiment — Qwen3-0.6B vs Qwen3-30B-A3B (MoE) on H200 | Stage-A benchmark report comparing Qwen3-0.6B and Qwen3-30B-A3B MoE regimes on H200 using the aggregation harness. | Stage-A 基准报告，用聚合脚本比较 H200 上 Qwen3-0.6B 与 Qwen3-30B-A3B MoE 的 regime 表现。 |

### 2026-06-03

**EN:** Work moved from coarse regime benchmarking to kernel inventory for the Qwen3 MoE mixed-length R7 regime.  
**中文：**这一天从整体 regime 基准推进到 Qwen3 MoE 混合长度 R7 regime 的 kernel 清单梳理。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-03/kernel_inventory_R7_qwen3_moe.md | Kernel Inventory — Qwen3-30B-A3B MoE on H200, R7 mixed-lengths regime | End-to-end profiling inventory of kernels for Qwen3-30B-A3B MoE on H200 in the R7 mixed-length regime. | Qwen3-30B-A3B MoE 在 H200 的 R7 混合长度 regime 下的端到端 kernel profiling 清单。 |

### 2026-06-04

**EN:** The focus was backend and kernel implementation choices, comparing MoE dispatch paths and Triton rewrite feasibility.  
**中文：**这一天聚焦 MoE backend 与 kernel 实现路径，比较 dispatch 决策并评估 Triton 重写可行性。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-04/moe_backend_decision_trees.md | MoE Backend Decision Trees — vLLM vs sglang deep dive | Comprehensive decision-tree enumeration of every MoE backend dispatch branch in vLLM and sglang with code-snippet and file-line evidence. | 对 vLLM 与 sglang 的每条 MoE backend dispatch 分支做完整决策树枚举，并附代码片段与文件行号证据。 |
| docs/2026-06-04/triton_rewrite_investigation.md | Triton rewrite investigation — Is rewriting Triton kernels (to CUDA / Gluon / CUTLASS-DSL) a viable contribution to sgla | Investigation of whether rewriting Triton kernels into CUDA, Gluon, or CUTLASS-DSL is a viable contribution path for sglang. | 调研把 Triton kernel 重写成 CUDA、Gluon 或 CUTLASS-DSL 是否能成为对 sglang 有价值的贡献方向。 |

### 2026-06-08

**EN:** A dense profiling and correction day clarified vLLM versus sglang MoE behavior, autotune and cudagraph interactions, and available profiling tools.  
**中文：**这一天集中做 profiling 与纠错，厘清 vLLM 和 sglang 的 MoE 差异、autotune/cudagraph 交互以及工具能力。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-08/agent_profiling_capability_audit.md | Agent profiling 能力盘点 — 现状 / 缺口 / 需要 mentor 协助的部分 | Audit of the agent profiling stack, noting that the NCU gap was resolved by sudo NOPASSWD and a new `ncu-microarch` skill. | 盘点 agent profiling 能力，记录 NCU 权限缺口已通过 sudo NOPASSWD 和新的 `ncu-microarch` skill 解锁。 |
| docs/2026-06-08/buga_fix_validation.md | Bug A Fix 验证 — autotune 跑了但 e2e 没改善 | Validation showing the 9 ms gap was outside MoE, in attention, rmsnorm, sampling, or IPC, so MoE tuning could not improve it. | 验证 Bug A：9 ms 差距不在 MoE，而在 attention、rmsnorm、sample 或 IPC，因此 autotune 无法改善端到端。 |
| docs/2026-06-08/fix1_invalidated.md | Fix 1 验证 — 失败,原 ROOT_CAUSE 分析错了 | Failed Fix 1 validation showing the original root-cause analysis was wrong and the 9× gap did not improve. | Fix 1 失败验证：原 ROOT_CAUSE 分析错误，9× 差距没有改善且略微变差。 |
| docs/2026-06-08/nsys_2x2_validation_and_nsys_usage.md | 2×2 nsys 验证: max(CPU, GPU) 假说 + nsys 用法证据 | 2×2 nsys validation of the wall-time model `wall ≈ max(CPU work, GPU work)`, showing both dimensions must be reduced together. | 用 2×2 nsys 实验证明 `wall ≈ max(CPU work, GPU work)`，只有同时降低 CPU 和 GPU 两侧才会改善。 |
| docs/2026-06-08/nsys_deep_dive_and_proton.md | nsys 能拿到什么 + proton 是什么 — 二级 profiling 工具盘点 | Tooling note concluding that nsys gives panoramic profiling while proton provides targeted attribution, and both belong in the toolbox. | 工具盘点：nsys 负责全景调查，proton 负责定向归因，两者互补且都应纳入工具箱。 |
| docs/2026-06-08/server_lifecycle_and_sm100_tuning.md | LLM Server 全生命周期时间事件表 + SM100 flashinfer tuning 调研 | Lifecycle timing note showing server startup is roughly 15–30 seconds and mostly dominated by weight loading, plus SM100 FlashInfer tuning research. | 记录 LLM server 启动全生命周期约 15–30 秒且主要由权重加载主导，并调研 SM100 FlashInfer tuning。 |
| docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md | sglang vs vLLM:FlashInfer CUTLASS MoE 性能差异根因分析 | Root-cause analysis explaining why the same `flashinfer.fused_moe.cutlass_fused_moe` kernel differs in performance between sglang and vLLM. | 根因分析：解释同一 `flashinfer.fused_moe.cutlass_fused_moe` kernel 在 sglang 与 vLLM 中为何性能不同。 |
| docs/2026-06-08/triton_vs_cutlass_moe_kernel_source_comparison.md | sglang vs vLLM — MoE Kernel 源码对比(SM90 BF16) | Source-level comparison of Triton MoE and FlashInfer CUTLASS MoE kernels on SM90 BF16. | 对 SM90 BF16 下 Triton MoE kernel 与 FlashInfer CUTLASS MoE kernel 的关键源码做对比。 |
| docs/2026-06-08/vllm_2x2_autotune_cudagraph_matrix.md | vLLM 2×2 矩阵: autotune × cudagraph 实测,翻盘所有之前推论 | 2×2 vLLM experiment showing autotune or cudagraph alone is almost useless, overturning earlier inferences. | vLLM 的 autotune×cudagraph 2×2 实测：单开任一项几乎无用，推翻此前推论。 |
| docs/2026-06-08/vllm_autotune_e2e_impact.md | vLLM autotune 影响 e2e 实测 — 2.5-3.4× 真的 | Direct end-to-end measurement showing vLLM without autotune becomes as slow as sglang, confirming a real 2.5–3.4× autotune impact. | 端到端实测显示关闭 autotune 后 vLLM 与 sglang 一样慢，证明 2.5–3.4× 收益真实存在。 |
| docs/2026-06-08/vllm_autotune_vs_sglang_correction.md | vLLM 的 flashinfer autotune 机制 — 跟 sglang 的关键差异 | Correction note identifying vLLM's explicit startup `with autotune(): _dummy_run()` as the key FlashInfer autotune difference from sglang. | 纠错说明：vLLM 启动时显式执行 `with autotune(): _dummy_run()`，这是与 sglang 的关键差异。 |

### 2026-06-09

**EN:** Follow-up profiling established the CUTLASS-versus-Triton e2e result, swept sglang Triton regimes, and documented the skill architecture.  
**中文：**这一天补充 CUTLASS 与 Triton 的端到端结论，完成 sglang Triton 四 regime profiling，并整理 skill 架构。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-09/cutlass_vs_triton_e2e_investigation.md | Why is flashinfer CUTLASS MoE no faster than Triton MoE at e2e? | Investigation of why hand-tuned FlashInfer CUTLASS MoE does not beat Triton codegen end-to-end. | 调查为什么理论上更强的 FlashInfer CUTLASS MoE 在端到端上没有明显跑赢 Triton codegen。 |
| docs/2026-06-09/cutlass_vs_triton_e2e_investigation.zh.md | 为什么 vLLM/sglang 上 flashinfer CUTLASS MoE 没比 Triton 快？ | Chinese version of the CUTLASS-versus-Triton e2e investigation explaining why CUTLASS MoE did not outperform Triton. | CUTLASS 与 Triton 端到端调查的中文版本，解释 flashinfer CUTLASS MoE 没比 Triton 快的原因。 |
| docs/2026-06-09/sglang_triton_4regime_profiling.md | sglang Triton MoE — 4-regime nsys + ncu profiling sweep | Completed four-regime sweep of sglang Triton MoE with nsys traces and NCU full-set profiling. | 完成 sglang Triton MoE 四个 regime 的 nsys 和 NCU 全量 profiling sweep。 |
| docs/2026-06-09/sglang_triton_4regime_profiling.zh.md | sglang Triton MoE — 4-regime nsys + ncu profiling sweep | Chinese report of the completed four-regime nsys and NCU profiling sweep for sglang Triton MoE. | 中文报告：四个 regime 的 nsys 与 NCU profiling 已全部完成并生成 unified profile。 |
| docs/2026-06-09/skill_architecture.md | Skill architecture — how the 14 skills fit together (2026-06-09) | Architecture note describing how the 14 skills fit together and noting the remaining limitation that NCU can only profile its own process. | skill 架构说明，梳理 14 个 skills 如何配合，并记录 NCU 仍只能 profile 自己启动进程的限制。 |

### 2026-06-11

**EN:** Harness v1 results and meeting-prep notes reframed the recent sglang optimization findings.  
**中文：**这一天用 harness v1 的四路 bench 结果和会议草稿重述近期 sglang 优化发现。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-11/harness_v1_4way_findings.md | 2026-06-11 harness v1 — 4-way bench results | Harness v1 four-way benchmark report, later updated after discovering an equivalent existing point in the experiments. | harness v1 的四路 bench 结果报告，并记录实验后发现已有相同对照点而需要修订。 |
| docs/2026-06-11/ofer_meeting_findings_draft.md | Ofer 会议草稿：sglang 推理优化项目 —— 近期发现与问题定义 | Draft Ofer meeting document updated after harness v1, revising the TL;DR and adding a new point from the 4-way bench. | Ofer 会议草稿，基于 harness v1 四路 bench 修订 TL;DR 并新增发现点。 |

### 2026-06-25

**EN:** The project evaluated framework-level autotuning limits with Optuna, including honest results, search-space design, and rejected FP8 config copying.  
**中文：**这一天评估 sglang 框架级 autotuning 的上限，覆盖 Optuna 结果、搜索空间和 FP8 配置复制的负结果。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-25/autotuning_ceiling_report.md | Framework-Level Autotuning Ceiling Report — sglang on H200 | Report answering Debadeepta's 6/24 feedback by quantifying the framework-level autotuning ceiling for sglang on H200. | 回应 Debadeepta 6/24 反馈的报告，量化 H200 上 sglang 框架级 autotuning 的上限。 |
| docs/2026-06-25/autotuning_honest_results.md | Framework-Level Autotuning Study — Honest Results | Honest Optuna study of sglang framework-level autotuning across four regimes on H200. | sglang × Optuna × 四个 regime × H200 的框架级 autotuning 诚实结果。 |
| docs/2026-06-25/fp8_config_copy_experiment.md | FP8 Tuned Config Copy Experiment — Negative Result | Negative experiment rejecting the hypothesis that copying vLLM's tuned FP8 config into sglang improves performance. | 负结果实验：拒绝“复制 vLLM tuned FP8 config 到 sglang 能改善性能”的假设。 |
| docs/2026-06-25/sglang_autotuning_search_space.md | sglang Framework-Level Autotuning — Search Space Design | Search-space design note for quantifying sglang framework-level autotuning after Debadeepta's feedback. | 框架级 autotuning 搜索空间设计，服务于 Debadeepta 反馈后的量化实验。 |

### 2026-06-29

**EN:** Work validated the universal autotuned configuration through profiling and generated cookbook deployment baselines.  
**中文：**这一天用 profiling 验证 universal autotuned config 的瓶颈，并生成 cookbook 部署 baseline。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-29/profiling_validation_of_universal_config.md | Profiling Validation of the Autotuned Config — what's the bottleneck? | Follow-up profiling report on the H200 sglang Optuna universal config to identify the remaining bottleneck. | H200 上 sglang Optuna universal config 的 follow-up profiling，定位剩余瓶颈。 |
| docs/2026-06-29/profiling_validation_of_universal_config.zh.md | Profile 验证：autotuned universal config 到底在哪个瓶颈上？ | Chinese version of the profiling validation for the autotuned universal config and its bottleneck. | 中文版 profiling 验证：说明 autotuned universal config 到底卡在哪个瓶颈。 |

### 2026-06-30

**EN:** The line expanded to cookbook baseline analysis and conditional autotuning for LFM2.5.  
**中文：**这一天扩展到 cookbook baseline 分析，并开始 LFM2.5 的条件化搜索空间调参。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-06-30/cookbook_baseline_analysis.md | sglang Cookbook 部署 baseline 分析 | Input-file analysis report for sglang cookbook deployment baseline commands. | sglang Cookbook 部署 baseline 的输入文件分析报告。 |
| docs/2026-06-30/lfm2.5_conditional_autotuning.md | LFM2.5-8B-A1B 条件化搜索空间自动调参实验报告（2026-06-30） | Experiment report on conditional search-space autotuning for LFM2.5-8B-A1B. | LFM2.5-8B-A1B 条件化搜索空间自动调参的实验报告。 |

### 2026-07-02

**EN:** The LFM2.5 effort added MFU and long-context regimes, while a design doc proposed an LLM-plus-autotuner loop.  
**中文：**这一天 LFM2.5 实验引入 MFU 与长上下文 regime，并提出 LLM+Autotuner 迭代 pipeline。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-02/lfm2.5_v3_mfu_longctx.md | LFM2.5-8B-A1B v3 实验报告：MFU 引入 + 长上下文 regime + TPE 失效修复（2026-07-02） | LFM2.5 v3 report adding MFU, long-context regimes, and a fix for TPE failure. | LFM2.5 v3 实验报告：引入 MFU、长上下文 regime，并修复 TPE 失效。 |
| docs/2026-07-02/llm_autotuner_pipeline_design.md | LLM + Autotuner: 迭代式 sglang 推理配置优化 pipeline | Design document and project proposal for an iterative LLM plus autotuner pipeline for sglang inference configuration. | LLM + Autotuner 迭代式 sglang 推理配置优化 pipeline 的设计文档与项目提案。 |

### 2026-07-08

**EN:** Profiling reached hardware-counter depth by running NCU on real sglang kernels.  
**中文：**这一天用 NCU 对真实 sglang kernel 做硬件计数器级 profiling。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-08/v6_ncu_sglang_experiment_report.md | v6 实验报告：用 NCU 对真实 sglang kernel 做硬件计数器级 profiling | v6 experiment report using NCU to profile real sglang kernels at hardware-counter level during 2026-07-08 to 07-09. | v6 实验报告：在 2026-07-08 至 07-09 用 NCU 对真实 sglang kernel 做硬件计数器级 profiling。 |

### 2026-07-09

**EN:** Real and agent workloads were characterized, swept, and tuned through serving-level knob experiments.  
**中文：**这一天对真实/agent workload 做画像、配置 sweep，并调节 chunked-prefill 与 max-running-requests 等 serving knob。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-09/v7_agentic_workload_characterization.md | v7 实验报告：用真实 / agent 负载给 regime 做画像（bench_serving） | v7 report characterizing regimes with real and agent workloads using `bench_serving`. | v7 实验报告：用 `bench_serving` 在真实与 agent 负载上给 regime 做画像。 |
| docs/2026-07-09/v7_config_sweep_on_agentic.md | v7 config sweep：在真实 agent 负载上对比 tuned config | v7 config sweep comparing tuned configurations on a real agent workload. | v7 config sweep：在真实 agent 负载上对比 tuned config 的表现。 |
| docs/2026-07-09/v8_tuning_on_real_workload.md | v8 实验报告：真实负载上的 knob tuning（chunked-prefill × max-running-requests） | v8 report tuning serving knobs, especially chunked-prefill and max-running-requests, on real workloads. | v8 实验报告：在真实负载上调 chunked-prefill 与 max-running-requests 等 serving knob。 |

### 2026-07-10

**EN:** The team answered concurrency and headroom questions with NCU evidence that tuning alone could not hit hardware ceilings.  
**中文：**这一天回应 concurrency 与 TBT headroom 追问，用 NCU 证据说明单靠 tuning 不足以达到硬件上限。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-10/discussion_prep_concurrency_and_headroom.md | 讨论准备：concurrency / serving command / TBT headroom（回应 Dey&Chendi 追问） | Discussion-prep note answering Dey and Chendi on concurrency, serving command, and TBT headroom using same-workload cap evidence. | 讨论准备文档：用同一负载不同 server cap 下并发几乎不变的证据回应 Dey 与 Chendi。 |
| docs/2026-07-10/reply_to_dey_tbt_headroom.md | Reply to Dey — "How much can TBT still improve at best config?" | Reply explaining remaining TBT headroom at the best config using kernel-level SM idle and NCU No Eligible evidence. | 给 Dey 的回复：用 kernel 级 SM idle 与 NCU No Eligible 证据说明 best config 下 TBT 还能改善多少。 |
| docs/2026-07-10/v9_ncu_hardware_ceiling_evidence.md | v9 实验报告：证明"单靠 tuning 不足以达到硬件上限"（真实负载 + NCU） | v9 experiment report proving on real workloads and NCU data that tuning alone is insufficient to reach the hardware ceiling. | v9 实验报告：通过真实负载和 NCU 数据证明“单靠 tuning 不足以达到硬件上限”。 |
| docs/2026-07-10/v9b_walltime_and_stall_analysis.md | v9b/v9c 补充：真实 workload 的时间占比 + decode 的空闲/等待分析 | v9b/v9c supplement answering where wall time goes and how much idle time lies behind the 45.8% average. | v9b/v9c 补充：回答真实 workload 的时间去哪了，以及 45.8% 平均值背后有多少 decode 空闲/等待。 |

### 2026-07-14

**EN:** The opportunity-gap story was consolidated with offered-load sweeps and roofline framing for reporting.  
**中文：**这一天把 opportunity gap 的证据链串起来，并用 offered-load sweep 与 roofline 语言补齐汇报材料。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-14/opportunity_gap_comprehensive_analysis.md | Performance Opportunity Gap 综合分析报告 | Comprehensive analysis chaining v6–v12 evidence into a logical case that the optimization opportunity gap is visible and recoverable. | 综合分析报告：把 v6 到 v12 的证据串成“优化机会 gap 看得见且摸得着”的逻辑链。 |
| docs/2026-07-14/v10_loadsweep_and_roofline.md | v10：Offered-load sweep + 标准 Roofline 重述（为 Ofer/Li 汇报补齐） | v10 report adding offered-load sweep and standard roofline restatement for Ofer/Li reporting. | v10 报告：补充 offered-load sweep 和标准 roofline 重述，用于 Ofer/Li 汇报。 |

### 2026-07-15

**EN:** A major analysis day tied intervention experiments, scheduler mechanics, MoE routing, decode kernels, and dynamic top-k accuracy tradeoffs into the opportunity-gap narrative.  
**中文：**这一天集中推进干预实验、scheduler 机制、MoE routing、decode kernel 与动态 top-k 精度权衡，形成完整 gap 叙事。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-15/discussion_log_2026-07-15.md | 2026-07-15 讨论纪要：从"gap 看得见"到"gap 摸得着"再到"kernel 层根因" | Full discussion log recording the chain from visible gap, to recoverable gap, to scheduler and MoE kernel root causes. | 讨论纪要：完整记录从“gap 看得见”到“gap 摸得着”再到 kernel 层根因的对话、数据与问答。 |
| docs/2026-07-15/dynamic_topk_and_benchmark_plan.md | 动态 topk 可行计划：实现难度 + 轻量级精度 benchmark | Feasibility plan for dynamic or fixed top-k changes plus the lightweight accuracy benchmark needed because the change is lossy. | 动态 top-k 可行计划：评估实现难度，并设计因有损而必需的轻量级精度 benchmark。 |
| docs/2026-07-15/experiment_ideas_realizing_the_gap.md | 实验设计：证明 opportunity gap "摸得着"（非 config 手段真能吃回一部分） | Experiment design for proving that non-config interventions can recover part of the previously visible opportunity gap. | 实验设计：证明除 config 以外的手段确实能吃回一部分 opportunity gap。 |
| docs/2026-07-15/moe_routing_optimization_survey.md | MoE Routing 优化研究现状调研（回应"router 是否最优 + 改 route 换效率"） | Literature survey on whether routers are optimal and whether altered routing can trade model quality for inference efficiency. | MoE routing 调研：回答 router 是否最优，以及是否能通过改变 route 排布以部分精度换效率。 |
| docs/2026-07-15/reply_to_chendi_decode_analysis.md | Reply to Chendi — decode wall proportion + NCU decode analysis (interim) | Interim reply showing decode accounts for 88–96% of end-to-end wall time for the agent workload and analyzing it with NCU. | 给 Chendi 的阶段性回复：说明 agent workload 中 decode 占端到端 wall time 的 88–96%，并给出 NCU 分析。 |
| docs/2026-07-15/reply_to_dey_decode_kernels.md | Slack reply to Dey — decode is the bottleneck; which kernels to tune (2026-07-15) | Slack reply tracing agent-tuned knobs to the best config, decode bottleneck identification, and the specific hot kernels to tune. | 给 Dey 的 Slack 回复：从 agent tuned knobs 到 best config，再到 decode 瓶颈与应调的热 kernel。 |
| docs/2026-07-15/reply_to_dey_progress_update.md | Slack reply to Dey — progress update (2026-07-15) | Progress update to Dey emphasizing tuning on real sglang regimes rather than only hand-made regimes. | 给 Dey 的进展更新：强调已在真实 sglang regimes 上 tuning，而不只是手工构造 regime。 |
| docs/2026-07-15/scheduler_mechanism_and_agent_aware_idea.md | sglang 请求调度机制报告 + agent-aware 调度 idea 评估 | Report explaining how sglang receives, queues, batches, and processes requests, and evaluating an agent-aware scheduler idea. | sglang 请求调度机制报告：说明接收、排队、组批和处理流程，并评估 agent-aware scheduling 想法。 |
| docs/2026-07-15/triton_moe_kernel_analysis.md | Triton MoE Kernel 分析：为什么 SM 利用率低 + 优化空间 | Analysis of sglang's Triton `fused_moe_kernel` using NCU decode evidence of low SM utilization, high DRAM, and No Eligible stalls. | 分析 sglang 的 Triton `fused_moe_kernel`，结合 NCU decode 中 SM 低、DRAM 高、No Eligible 高的证据找优化空间。 |
| docs/2026-07-15/v11_realize_gap_results.md | v11：干预实验——证明 opportunity gap "摸得着"（首批结果） | v11 intervention results proving that part of the opportunity gap is recoverable. | v11 干预实验首批结果：证明 opportunity gap 不只是上界，而是“摸得着”。 |
| docs/2026-07-15/v12_ncu_spec_mechanism.md | v12：NCU 测 spec decoding 的 SM 空转 —— 机制发现（反直觉但重要） | v12 NCU study of SM idleness under speculative decoding, documenting a counterintuitive but important mechanism. | v12 NCU 实验：测 spec decoding 的 SM 空转并发现反直觉但重要的机制。 |
| docs/2026-07-15/v13_router_analysis.md | v13：Router 行为分析（Qwen3-30B-A3B，agent 输入）—— 量化"批内 expert 聚集" idea 的空间 | v13 router-behavior analysis for Qwen3-30B-A3B on agent inputs, quantifying space for within-batch expert consolidation. | v13 router 行为分析：在 Qwen3-30B-A3B agent 输入上量化“批内 expert 聚集”的空间。 |
| docs/2026-07-15/v14_consolidation_tradeoff.md | v14：批内 expert 聚集的搬运节省 vs 精度代价（模拟） | v14 simulation of the tradeoff between data-movement savings and accuracy cost from within-batch expert consolidation. | v14 模拟：评估批内 expert 聚集带来的搬运节省与精度代价。 |
| docs/2026-07-15/v15_perplexity_tradeoff.md | v15：expert 缩减的真实 perplexity 代价 + 最终权衡曲线 | v15 measurement of real perplexity cost from expert reduction and the final tradeoff curve. | v15 实测 expert 缩减的真实 perplexity 代价，并给出最终权衡曲线。 |
| docs/2026-07-15/v16_router_distribution.md | v16：Router 分布详细分析（Qwen3-30B-A3B，agent 输入）—— 带分布图 | v16 detailed router-distribution analysis for Qwen3-30B-A3B on agent inputs, including distribution plots. | v16 详细 router 分布分析：针对 Qwen3-30B-A3B agent 输入并附分布图。 |
| docs/2026-07-15/v17_gsm8k_topk_results.md | v17：GSM8K 真实精度 × 时间分布 vs top-k（丢专家的下游任务成本曲线） | v17 GSM8K results measuring real accuracy and time distribution versus top-k as the downstream cost curve for dropping experts. | v17 GSM8K 结果：测真实精度和时间分布随 top-k 的变化，形成丢专家的下游任务成本曲线。 |
| docs/2026-07-15/v18_dynamic_topk_results.md | v18：动态 top-k（置信度自适应）vs 固定 top-k — GSM8K 精度 × 平均激活专家 | v18 comparison of confidence-adaptive dynamic top-k versus fixed top-k on GSM8K accuracy and average active experts. | v18 动态 top-k 对比固定 top-k：评估 GSM8K 精度与平均激活 expert 数。 |
| docs/2026-07-15/v19_partC_decode_potential.md | v19 Part C — Decode optimization potential & gaps (NCU 11-metric analysis) | v19 Part C NCU 11-metric analysis of decode optimization potential and remaining gaps. | v19 Part C：用 NCU 11 个指标分析 decode 优化潜力与 gap。 |

### 2026-07-16

**EN:** Mason's staged roadmap was translated into an executable matrix for Qwen3-30B-A3B roofline work.  
**中文：**这一天把 Mason 建议的分层路线落成 Qwen3-30B-A3B 逼近 roofline 的执行矩阵。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-16/mason_roadmap_qwen3_moe_matrix.md | Mason 路线图落地矩阵：Qwen3-30B-A3B（MoE）逼近 roofline 的分层进度表 | Executable matrix applying Mason's staged path from config autotuning to kernel autotuning and manual fusion for Qwen3-30B-A3B. | 将 Mason 的“config autotuning → kernel autotuning → 手改/融合”路线落到 Qwen3-30B-A3B 的可执行矩阵。 |

### 2026-07-17

**EN:** A broad AutoTuner generality experiment was designed alongside the Mason depth line.  
**中文：**这一天设计了 AutoTuner 普适性广度实验，并保留 Mason 深度线。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-17/breadth_autotuner_experiment_design.md | 实验设计：AutoTuner 普适性广度实验（20 model × regime × dataset × config）+ Mason 深度线 | Review-ready design for a broad AutoTuner generality experiment over models, regimes, datasets, and configs plus Mason's depth line. | 待 review 的 AutoTuner 普适性广度实验设计，覆盖 model×regime×dataset×config，并保留 Mason 深度线。 |

### 2026-07-19

**EN:** Chendi's SGLang MoE/kernel PR list was validated on H200 with bf16 evidence.  
**中文：**这一天对 Chendi 的 sglang MoE/kernel PR 清单做 H200 bf16 验证。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-19/pr_validation_report.md | SGLang MoE/Kernel PR 验证报告（Chendi PR 清单 · bf16 · H200） | Validation report for Chendi's SGLang MoE/kernel PR list, with config-tuning PRs validated on two models and shared-expert fusion left optional. | Chendi PR 清单的验证报告：config-tuning 类已在两个模型上验证，shared-expert 融合机会作为可选后续。 |

### 2026-07-20

**EN:** The kernel-level campaign reconciled configuration retuning, fallback baselines, PR evidence, custom MoE noise, and end-to-end headroom.  
**中文：**这一天围绕 kernel-level 优化集中处理 config retune、fallback baseline、PR 证据、custom MoE 噪声和端到端 headroom。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-20/conversation_context_snapshot.md | 对话上下文快照 — 2026-07-20 晚（kernel tuning e2e gap + 迁移到新机器） | Context snapshot noting the missing gap that kernel-config tuning had only isolated kernel timing and still needed e2e A/B validation after migration. | 对话上下文快照：指出 kernel-config tuning 只有隔离 kernel 时间，迁移新机器后首先要补端到端 A/B。 |
| docs/2026-07-20/headroom_beyond_tuning_figures.md | "Tuning 以外还有多少空间" — Qwen3-30B-A3B 图表说明（回答 Dey） | Figure explanation for Dey quantifying remaining space beyond tuning on Qwen3-30B-A3B decode with H200 bf16 measurements. | 回答 Dey 的图表说明：用 H200 bf16 上 Qwen3-30B-A3B decode 实测量化 tuning 之外还有多少空间。 |
| docs/2026-07-20/kernel_config_retune_vs_fallback_e2e.md | v44 — Kernel-config tuning, layer 3: re-tune vs the fallback sglang actually loads (e2e) | v44 e2e comparison showing retuning the kernel config for the current Triton version buys nothing over the fallback sglang loads. | v44 端到端对比：为当前 Triton 版本重新 tuning 相对 sglang 实际加载的 fallback 没有收益。 |
| docs/2026-07-20/kernel_config_server_ours_vs_fallback_e2e.md | v45 — Server-level e2e A/B: ours (re-tuned) vs fallback, all regimes + agent dataset | v45 server-level e2e A/B measuring the marginal value of retuning versus fallback across regimes and the agent dataset. | v45 server 级端到端 A/B：在全 regime 与 agent dataset 上比较 ours retuned 与 fallback 的边际价值。 |
| docs/2026-07-20/kernel_headroom_other_models_pr_evidence.md | 换模型/换场景,kernel level 还有空间吗?—— 用别人的 sglang PR 作证据 | Evidence note answering whether kernel-level headroom remains for other models or scenarios using merged sglang PRs and official blog evidence. | 用已合并 sglang PR 与官方博客作证据，回答换模型/场景后 kernel 层是否仍有空间。 |
| docs/2026-07-20/kernel_level_improvement_evidence.md | Kernel-Level 性能提升证据（能否靠"改 kernel 代码"拿到加速？） | Evidence report on whether changing kernel code can produce real performance improvements. | kernel-level 性能提升证据报告：回答是否能靠“改 kernel 代码”拿到加速。 |
| docs/2026-07-20/kernel_optimization_attempt_log.md | Kernel-Level 优化正面攻关日志（打赢 sglang 已 tuned 的 kernel） | Overnight attack log for trying to beat sglang's already-tuned kernel at the kernel level. | kernel-level 正面攻关日志：记录通宵尝试打赢 sglang 已 tuned kernel 的过程。 |
| docs/2026-07-20/kernel_reproduction_results.md | 复现 kernel-level 提升 —— 真实实测证据（在真实模型上端到端） | End-to-end reproduction evidence for kernel PR techniques on real models, producing the project's own measured numbers. | 在真实模型上端到端复现 kernel PR 技术，得到项目自己的实测 kernel-level 提升证据。 |
| docs/2026-07-20/new_architecture_linear_attention_e2e.md | New-architecture end-to-end headroom: linear attention decouples decode cost from context | Autopilot-session report measuring end-to-end headroom from linear attention, where decode cost is decoupled from context length. | 新架构端到端 headroom 报告：linear attention 让 decode cost 与上下文长度解耦。 |
| docs/2026-07-20/noise_verification_custom_moe_b1.md | 噪声验证：custom MoE kernel 的 b=1 "+1.4%" 是真信号还是波动？（Chendi 要求） | Chendi-requested noise verification for whether the custom MoE kernel's b=1 +1.4% on Qwen3-30B-A3B/H200/bf16 is real. | Chendi 要求的噪声验证：确认 custom MoE kernel 在 b=1 的 +1.4% 是真信号还是波动。 |
| docs/2026-07-20/qwen_optimization_full_report.md | Qwen3-30B-A3B 优化全纪录：做了什么 · 结果 · 有无提升 · 分析 | Full optimization record for Qwen3-30B-A3B-Instruct-2507, covering actions, results, gains or non-gains, and analysis. | Qwen3-30B-A3B-Instruct-2507 优化全纪录，汇总做了什么、结果、有无提升和分析。 |
| docs/2026-07-20/regime_sweep_kernel_changes.md | Kernel 改动 × 全 regime 端到端矩阵(诚实修正:之前只测单点) | Honest correction matrix measuring kernel changes across all regimes end-to-end instead of only the earlier single point. | 诚实修正：把 kernel 改动放到全 regime 端到端矩阵中重测，而不是只看单点。 |

### 2026-07-21

**EN:** Recent upstream SGLang PRs were reproduced as stable-versus-patched A/Bs for real end-to-end impact.  
**中文：**这一天把近期上游 SGLang PR 复现为 stable 与单 PR patch 的 A/B，验证真实端到端影响。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-21/pr29007_dsv4_symm_mem_allreduce_repro.md | PR #29007 reproduction — MoE TP allreduce via NCCL symmetric memory (DeepSeek-V4) | Reproduction of PR #29007 using DeepSeek-V4 TP8 with DSV4 attention and NCCL symmetric-memory allreduce. | PR #29007 复现：在 DeepSeek-V4 TP8、DSV4 attention 下验证 NCCL symmetric memory 的 MoE TP allreduce。 |
| docs/2026-07-21/pr31438_mm_preproc_parallel_repro.md | PR #31438 reproduction — parallelize VLM multimodal preprocessing | Positive, bit-exact reproduction of PR #31438, moving image I/O and HF-processor work off the serial path. | PR #31438 复现：正向且 bit-exact，验证并行化 VLM multimodal preprocessing 的收益。 |
| docs/2026-07-21/pr31558_fla_l2norm_recompile_repro.md | PR #31558 reproduction — avoid FLA l2-norm recompilation by token count | Positive reproduction of PR #31558 showing the exact mechanism and significant e2e gains from avoiding FLA l2-norm recompilation by token count. | PR #31558 复现：机制吻合且端到端显著改善，避免 FLA l2-norm 按 token 数重复编译。 |
| docs/2026-07-21/pr_reproduction_session_summary.md | Upstream-PR reproduction line — session summary (2026-07-21) | Session summary for reproducing upstream SGLang PRs as fair stable-release versus single-patch A/Bs on 8×H200. | 上游 PR 复现线的会话总结：在 8×H200 上做 stable release 与单 PR patch 的公平 A/B。 |

### 2026-07-22

**EN:** Roofline analysis separated decode-heavy and prefill-heavy behavior, while LFM2.5 serving autotuning was checked for plateau behavior.  
**中文：**这一天用 roofline 分析区分 decode-heavy 与 prefill-heavy 瓶颈，并检查 LFM2.5 serving autotuning 的平台期。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-22/lfm25_serving_autotuning_plateau.md | LFM2.5 Serving-Knob Autotuning: A Clean Plateau Study (no warm start) | Clean no-warm-start plateau study of LFM2.5 serving-knob autotuning. | LFM2.5 serving knob autotuning 的干净平台期研究，不使用 warm start。 |
| docs/2026-07-22/ncu_real_regime_both_stages_roofline.md | NCU roofline — real decode-heavy & prefill-heavy regimes, both-stage hot kernels | NCU roofline capture of hot kernels in both stages for real decode-heavy and prefill-heavy regimes with CUDA graph disabled. | 在关闭 CUDA graph 后，对真实 decode-heavy 与 prefill-heavy regimes 的两阶段热 kernel 做 NCU roofline。 |
| docs/2026-07-22/ncu_roofline_fused_moe_analysis.md | NCU roofline analysis — fused_moe kernel, decode (memory-bound) vs prefill (compute-bound) | Roofline analysis of Qwen3-30B-A3B fused_moe, contrasting memory-bound decode with compute-bound prefill. | Qwen3-30B-A3B fused_moe 的 roofline 分析：对比 memory-bound decode 与 compute-bound prefill。 |
| docs/2026-07-22/prefill_vs_decode_bottleneck_report.md | Prefill vs. Decode Bottleneck Analysis — Regimes, Method, and Results | Bottleneck report for Qwen3-30B-A3B bf16 on H200 with sglang Triton fused MoE, covering regimes, method, and results. | Qwen3-30B-A3B bf16/H200/sglang Triton fused MoE 的 prefill 与 decode 瓶颈分析。 |

### 2026-07-23

**EN:** The high-concurrency TTFT workload and serving-level slide data were audited for correctness.  
**中文：**这一天审计高并发 TTFT 负载与 serving-level tuning 幻灯片数据来源。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-23/high_concurrency_ttft_workload_audit.md | High-concurrency TTFT rerun — recovered workload audit | Audit of the recovered high-concurrency TTFT rerun using a streaming `/generate` client. | 对使用 streaming `/generate` client 的高并发 TTFT 重跑与恢复 workload 做审计。 |
| docs/2026-07-23/serving_tuning_slide_verified_data.md | Serving-level tuning slide — verified data & source audit | Source audit correcting that remembered Qwen long-input numbers were actually LFM2.5-8B-A1B data. | serving-level tuning slide 的数据源审计，纠正记忆中的 Qwen 长输入数字实际来自 LFM2.5-8B-A1B。 |

### 2026-07-24

**EN:** A serving-ceiling campaign for Qwen and LFM2.5 was documented, audited, and translated into slide-claim provenance.  
**中文：**这一天记录并审计 Qwen 与 LFM2.5 的 serving-ceiling campaign，并为幻灯片 claim 建立来源说明。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-24/qwen_serving_ceiling_methodology.md | Qwen + LFM2.5 serving-ceiling campaign — methodology | Methodology for the `results/2026-07-24_serving_ceiling/` campaign on Qwen and LFM2.5. | Qwen + LFM2.5 serving-ceiling campaign 的方法文档，对应 `results/2026-07-24_serving_ceiling/`。 |
| docs/2026-07-24/qwen_serving_ceiling_results.md | Qwen + LFM2.5 serving-ceiling campaign — results | Results report for the Qwen plus LFM2.5 serving-ceiling campaign. | Qwen + LFM2.5 serving-ceiling campaign 的结果报告。 |
| docs/2026-07-24/qwen_serving_ceiling_slide_claims.md | Slide claims — provenance and limits | Provenance document listing each slide claim's wording, source fields, calculation, limits, and evidence status. | 幻灯片 claim 溯源文档：逐条记录措辞、来源字段、计算方式、限制以及直接证据/推断状态。 |
| docs/2026-07-24/serving_tuning_data_audit.md | Serving-Tuning Data Audit (Phase 0) | Phase-0 data audit of serving-tuning results at commit `915f636`. | serving-tuning 数据 Phase 0 审计，审计时仓库 commit 为 `915f636`。 |

### 2026-07-26

**EN:** Alternative serving objectives and the regime-aware kernel specialization plan/status were audited before execution.  
**中文：**这一天审计替代 serving objective，并形成 regime-aware kernel specialization 的计划与仓库状态。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-26/alternative_objective_validation_audit.md | Alternative-objective validation audit (Phases 1–3) | Audit covering phases 1–3 of validation for alternative serving objectives. | 替代 serving objective 验证的 Phase 1–3 审计。 |
| docs/2026-07-26/alternative_serving_objectives.md | Does the serving objective change which configuration wins? | Study asking whether changing the serving objective changes the winning configuration. | 研究 serving objective 改变时，胜出的配置是否也会改变。 |
| docs/2026-07-26/regime_kernel_experiment_plan.md | Regime-aware Kernel Specialization — experiment plan | One-to-two-week single-H200 plan for regime-aware kernel specialization without new CUDA kernels or serving-runtime changes. | Regime-aware kernel specialization 的 1–2 周单 H200 实验计划，不新增 CUDA kernel 或 serving runtime 改动。 |
| docs/2026-07-26/regime_kernel_status.md | Regime-aware Kernel Specialization — repository status (Step 1–2) | Read-only repository status for steps 1–2 of regime-aware kernel specialization before running experiments. | regime-aware kernel specialization Step 1–2 的只读仓库状态审计，尚未运行实验。 |

### 2026-07-27

**EN:** LFM2.5 kernel optimization and fusion gaps were audited, patched, and reported with end-to-end results.  
**中文：**这一天推进 LFM2.5 kernel 优化与 fusion gap，完成审计、patch 和端到端结果报告。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md | LFM2.5 / SGLang 内核优化全纪录（2026-07-26 ~ 07-27） | Full record of LFM2.5/SGLang kernel optimization over 2026-07-26 to 07-27. | 2026-07-26 至 07-27 的 LFM2.5/SGLang 内核优化全纪录。 |
| docs/2026-07-27/lfm_fusion_results.md | LFM2.5 kernel-fusion gaps — audit, patch and end-to-end result | Audit, patch, and end-to-end result for LFM2.5 kernel-fusion gaps, with Qwen3-30B-A3B as control. | LFM2.5 kernel-fusion gap 的审计、patch 与端到端结果，并以 Qwen3-30B-A3B 作对照。 |
| docs/2026-07-27/regime_kernel_results.md | Regime-aware Kernel Specialization — results | Results for regime-aware kernel specialization linked back to the 2026-07-26 status and plan. | regime-aware kernel specialization 的结果报告，关联 2026-07-26 的 status 与 plan。 |

### 2026-07-28

**EN:** Kernel-fusion evidence matured into PR drafts, cross-architecture auditing, and comparisons of config tuning, rewrites, fusion, and wiring fixes.  
**中文：**这一天把 kernel-fusion 证据沉淀为 PR 草稿、跨架构审计，以及 config tuning/重写/补融合/接线的对照。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-28/PR_DRAFT_gemma3_rmsnorm_v1_superseded.md | PR 草稿 — `fix(gemma3): dispatch Gemma3RMSNorm to the fused CUDA kernel` | Superseded PR draft for dispatching Gemma3RMSNorm to the fused CUDA kernel, later partly covered by upstream #32383. | 已作废的 Gemma3RMSNorm fused CUDA dispatch PR 草稿，收益的一部分已被上游 #32383 覆盖。 |
| docs/2026-07-28/PR_DRAFT_gemma3_rmsnorm_v2.md | PR 草稿 v2 — `fix(gemma3): fuse high-rank RMSNorm and harden mixed-dtype weights` | Draft PR v2 for sglang #32670, fusing high-rank Gemma3 RMSNorm and hardening mixed-dtype weights. | sglang #32670 的 v2 PR 草稿：融合 Gemma3 high-rank RMSNorm 并加固 mixed-dtype weights。 |
| docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md | PR 草稿 — LFM2.5 的 H200 MoE tuned config（补 #22791 漏掉的那块） | Draft PR for adding LFM2.5 H200 MoE tuned config to cover the gap left by #22791. | LFM2.5 H200 MoE tuned config 的 PR 草稿，用于补上 #22791 漏掉的部分。 |
| docs/2026-07-28/cross_architecture_audit.md | 跨架构 fusion 空缺审计 —— 我的假设被自己的数据推翻了，修正后的结论更有用 | Cross-architecture fusion-gap audit whose data overturned the original hypothesis and produced a corrected conclusion. | 跨架构 fusion 空缺审计：原假设被数据推翻，并形成更有用的修正结论。 |
| docs/2026-07-28/four_kernel_cases_comparison.md | 四个 kernel 级案例的对照报告：调 config vs 重写 vs 补融合 vs 接线 | Comparison report of four kernel-level cases—config tuning, rewrite, missing fusion, and wiring—with project-measured H200 bf16 numbers. | 四个 kernel 级案例对照：调 config、重写、补融合、接线，数字均为本项目 H200 bf16 实测。 |
| docs/2026-07-28/three_fusion_cases.md | Kernel fusion 案例全集 —— 改了什么、为什么被漏掉、拿到多少（含未兑现的） | Complete set of kernel-fusion cases from 2026-07-27 to 07-28, covering what changed, why it was missed, and realized or unrealized gain. | Kernel fusion 案例全集：说明改了什么、为什么被漏掉、拿到多少收益，并包含未兑现案例。 |

### 2026-07-29

**EN:** Triton 3.6 retuning conclusions were first recorded and then formally retracted because the baseline was contaminated.  
**中文：**这一天记录 Triton 3.6 重扫结果，并因 baseline 污染正式撤回主要结论。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-29/RETRACTION_triton36_baseline_contamination.md | 撤回：Triton 3.6 的实验因基线污染而无效 | Retraction noting that the main conclusion of `triton_36_retune_findings.md` is invalid because of baseline contamination. | 撤回声明：因 baseline 污染，`triton_36_retune_findings.md` 的主要结论无效。 |
| docs/2026-07-29/triton_36_retune_findings.md | Triton 3.6 重扫：编译器升级吃掉了我们 tuning 的全部收益 | Superseded findings claiming Triton 3.6 compiler upgrade erased tuning gains, later invalidated because the default baseline loaded the tuned config. | 已失效的 Triton 3.6 重扫发现：曾认为编译器升级吃掉 tuning 收益，但默认基线实际加载了 tuned config。 |

### 2026-07-30

**EN:** The project formalized how to discover fusion opportunities, contrasting FX graphs, profiling, and SGLang model/backend dispatch.  
**中文：**这一天把 fusion 机会发现流程体系化，对比 FX、profiling，并梳理 SGLang 模型接入与 backend dispatch。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-30/fusion_discovery_walkthrough.md | 融合机会的发现—验证—尝试全流程（结合模型运行流程讲） | Walkthrough of the full discover–validate–try process for fusion opportunities, tied to the model execution flow. | 结合模型运行流程讲解 fusion 机会从发现到验证再到尝试的全流程。 |
| docs/2026-07-30/fx_vs_profiling_for_fusion_discovery.md | FX graph 还是 profiling？—— 给 agent 设计融合机会发现流程 | Agent workflow design comparing FX graph and profiling for fusion discovery, with a same-day correction of a classification error. | 为 agent 设计 fusion 机会发现流程，比较 FX graph 与 profiling，并记录同日纠正的分类错误。 |
| docs/2026-07-30/sglang_model_onboarding_and_backend_dispatch.md | SGLang 如何接入模型、如何选 backend —— 以及和 torch.compile/FX 体系的接口在哪 | Explanation of how SGLang onboards models, selects backends, and interfaces with torch.compile/FX systems. | 说明 SGLang 如何接入模型、如何选择 backend，以及与 torch.compile/FX 体系的接口位置。 |

### 2026-07-31

**EN:** Portable fusion discovery was tested with FX-based results, Gemma-3 QK-norm/RoPE fusion, and two full fusion-case records.  
**中文：**这一天验证可迁移 kernel fusion 自动发现，包括 FX 结果、Gemma-3 QK-norm/RoPE fusion 和两个完整案例。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-07-31/FINAL_PROJECT_portable_fusion_discovery.md | Final project：可迁移的 kernel fusion 自动发现 | Final-project report for portable automatic kernel-fusion discovery on H200 GPUs. | final project 报告：在 H200 上做可迁移的 kernel fusion 自动发现。 |
| docs/2026-07-31/fx_based_fusion_discovery_results.md | 用 torch.compile / FX 自动发现 kernel fusion 机会 —— 实验结果 | Experiment results for using torch.compile/FX to automatically discover kernel-fusion opportunities. | 使用 torch.compile/FX 自动发现 kernel fusion 机会的实验结果。 |
| docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md | Gemma-3 fused QK-norm + RoPE：从发现到验证的完整实验记录 | Full record from discovery to validation of fused QK-norm plus RoPE for Gemma-3. | Gemma-3 fused QK-norm + RoPE 从发现到验证的完整实验记录。 |
| docs/2026-07-31/two_fusion_cases_full_record.md | 两个 kernel fusion case 的完整记录：怎么发现的、怎么验证的、以及什么能自动化 | Full record of two kernel-fusion cases, including discovery, validation, and what can be automated. | 两个 kernel fusion case 的完整记录：说明怎么发现、怎么验证，以及哪些环节可自动化。 |

### 2026-08-02

**EN:** Free exploration and agent-loop workflows were compared on the same fusion problem.  
**中文：**这一天在同一个问题上比较自由探索与 agent loop 两种工作模式。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-08-02/free_exploration_vs_agent_loop.md | 自由探索 vs Agent loop：同一个问题，两种模式的实测对比 | Measurement-based comparison of free exploration versus an agent loop on the same problem using H200 and sglang baseline context. | 自由探索与 agent loop 在同一问题上的实测对比，基于 H200 与 sglang baseline。 |

### 2026-08-03

**EN:** The LFM2.5 final-case narrative was consolidated with handoffs, ablation setup, evidence chains, and mentor-deliverable mapping.  
**中文：**这一天集中整理 LFM2.5 final case，包括交接、消融矩阵、证据链与 mentor 交付物匹配。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-08-03/HANDOFF_fill_ablation_matrix.md | 交接：补全 LFM2.5 消融矩阵的空格 | Handoff for filling the remaining cells of the LFM2.5 ablation matrix. | 交接文档：说明如何补全 LFM2.5 消融矩阵的剩余空格。 |
| docs/2026-08-03/HANDOFF_kernel_fusion_agent_loop.md | 交接文档：kernel fusion gap 的自由探索 → agent loop 化 | Handoff for converting kernel-fusion-gap free exploration into an agent-loop workflow. | 交接文档：把 kernel fusion gap 的自由探索流程改造成 agent loop。 |
| docs/2026-08-03/HANDOFF_lfm25_layered_experiment.md | 交接：LFM2.5 分层实验（autotuning ceiling vs kernel rewrite） | Handoff for continuing the LFM2.5 layered experiment comparing autotuning ceiling with kernel rewrite. | 交接文档：继续 LFM2.5 分层实验，对比 autotuning ceiling 与 kernel rewrite。 |
| docs/2026-08-03/LFM25_FINAL_CASE_full_record.md | LFM2.5-8B-A1B 端到端优化全记录 —— 候选 final case | Candidate final-case full record for end-to-end optimization of LFM2.5-8B-A1B. | LFM2.5-8B-A1B 端到端优化全记录，作为候选 final case。 |
| docs/2026-08-03/LFM25_ablation_matrix_EN.md | LFM2.5-8B-A1B — Optimization Ablation Matrix | English ablation matrix for LFM2.5-8B-A1B optimization on one H200 with BF16 and TP1. | LFM2.5-8B-A1B 优化消融矩阵英文版，设定为 1×H200、BF16、TP1。 |
| docs/2026-08-03/deliverables_vs_mentor_requirements.md | 交付物梳理：mentor 要什么，我们有什么，哪些扣得上 | Deliverables mapping document comparing mentor requirements with existing artifacts and matches. | 交付物梳理：对照 mentor 要求、已有材料以及哪些内容能扣上要求。 |
| docs/2026-08-03/evidence_chain_how_gaps_were_found.md | 证据链：这几个 kernel fusion 机会到底是怎么被找到的 | Evidence-chain document explaining how the kernel-fusion opportunities were actually found. | 证据链文档：说明这些 kernel fusion 机会到底是如何被找到的。 |
| docs/2026-08-03/exp3_kernel_on_tuned_baseline.md | 实验 3：把 kernel 增量重测在装了 tuned MoE config 的干净基线上 | Experiment 3 retesting kernel increments on a clean baseline that includes the tuned MoE config for LFM2.5. | 实验 3：在装了 tuned MoE config 的干净 baseline 上重测 LFM2.5 kernel 增量。 |
| docs/2026-08-03/serving_ceiling_is_regime_specific.md | regime C 的 autotuning ceiling 不是 cookbook —— 以及这对交付叙事意味着什么 | Reinterpretation and validation experiment showing regime C's autotuning ceiling is not the cookbook baseline and affects deliverable framing. | 重新解读并验证：regime C 的 autotuning ceiling 不是 cookbook baseline，并说明这对交付叙事的影响。 |

### 2026-08-04

**EN:** The LFM2.5 methodology and ablation matrix were completed, and the workflow was replicated on OLMo-2 and Falcon-H1.  
**中文：**这一天完成 LFM2.5 方法论与消融矩阵，并在 OLMo-2 与 Falcon-H1 上复现发现流程。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-08-04/METHODOLOGY_three_layer_optimization.md | 三级优化方法论：在新模型上复现 LFM2.5 结果的操作手册 | Operational manual for reproducing LFM2.5's three-layer optimization results on a new model. | 三级优化方法论操作手册：指导如何在新模型上复现 LFM2.5 结果。 |
| docs/2026-08-04/ablation_matrix_complete.md | 补全 LFM2.5 消融矩阵：6 regime × 8 列，48 格全测 | Completed LFM2.5 ablation matrix covering 6 regimes by 8 columns, all 48 cells measured. | 完整 LFM2.5 消融矩阵：6 个 regime × 8 列，48 格全部实测。 |
| docs/2026-08-04/agent_workflow_evidence_chain.md | Agent 工作流证据链：LFM2.5 的 kernel 机会是怎么被找到、验证、修好的 | Evidence chain for the agent workflow that found, validated, and fixed LFM2.5 kernel opportunities. | Agent 工作流证据链：说明 LFM2.5 kernel 机会如何被找到、验证并修好。 |
| docs/2026-08-04/pipeline_replication_olmo2_falconh1.md | 在 OLMo-2 和 Falcon-H1 上复现整套发现流程 | Replication of the full discovery pipeline on OLMo-2 and Falcon-H1 with Qwen3-30B as control. | 在 OLMo-2 与 Falcon-H1 上复现整套发现流程，并以 Qwen3-30B 作对照。 |

### 2026-08-05

**EN:** Kernel-level opportunity search expanded to OLMo-2 and Falcon-H1 with Qwen as control.  
**中文：**这一天把 kernel 级机会搜索扩展到 OLMo-2 与 Falcon-H1，并保留 Qwen 对照组。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-08-05/kernel_level_opportunities.md | kernel 级机会搜索：OLMo-2 与 Falcon-H1 | Kernel-level opportunity search for OLMo-2 and Falcon-H1 on H200 GPUs with Qwen3-30B as control. | OLMo-2 与 Falcon-H1 的 kernel 级机会搜索，使用 H200 并以 Qwen3-30B 作对照。 |

### 2026-08-07

**EN:** External validity of L3 improvements was tested on real workloads through load sweeps and dataset expansion.  
**中文：**这一天用负载扫描和数据集扩展验证 L3 在真实 workload 上的外部有效性。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-08-07/REAL_WORKLOAD_EXTERNAL_VALIDITY.md | 真实 workload 上 L3 的外部有效性：负载扫描 + 数据集扩展 | External-validity study of L3 improvements on real LFM2.5 workloads using load sweeps and dataset expansion. | 真实 workload 上 L3 的外部有效性研究：对 LFM2.5 做负载扫描与数据集扩展。 |

### 2026-08-10

**EN:** The final dated experiment measured L2, L3, and combined L2+L3 ablations on a new real/agentic workload.  
**中文：**最后一个日期实验在新的真实/agentic workload 上测 L2、L3 与 L2+L3 消融。

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/2026-08-10/RT_L2L3_ablation.md | 新真实/agentic workload 上的 L2 / L3 / L2+L3 消融 | Nightly LFM2.5 BF16 TP1 ablation measuring L2, L3, and L2+L3 on a new real/agentic workload. | 夜间 LFM2.5 BF16 TP1 消融实验：在新真实/agentic workload 上测 L2、L3 与 L2+L3。 |

## Living documents / 常驻文档

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| docs/architecture/two-stage-overview.md | Two-Stage Architecture | Living architecture document for the two-agent Problem-Setter/Problem-Solver design, replacing the earlier three-stage design. | 常驻架构文档：说明 Problem-Setter/Problem-Solver 两阶段设计，并取代早期三阶段方案。 |
| docs/development/developer-guide.md | Developer Guide — SGLang End-to-End Optimization Agent | Long-form developer and design guide for the SGLang end-to-end optimization agent. | SGLang 端到端优化 agent 的长篇开发者与设计指南。 |
| docs/development/history.md | Project history — how we got here | Timeline explaining the project's evolution from a single-agent fantasy to the current two-stage architecture. | 项目历史时间线：解释从 single-agent 设想到当前两阶段架构的演进。 |
| docs/development/log-layout.md | Log layout — Stage 1 RegimeScout | Reference for Stage 1 RegimeScout log files, raw results, and per-workload artifacts. | Stage 1 RegimeScout 的日志、raw results 与每个 workload 产物布局说明。 |
| docs/development/restructure-notes.md | Restructure Status | Working note tracking the in-progress two-stage refactor, current status, pending work, and converged design. | 两阶段重构的工作笔记，跟踪当前状态、待办事项与收敛后的设计。 |
| docs/idea-pool/schema.md | Idea Pool — Schema | Schema for the bidirectional idea channel between Problem-Setter and Problem-Solver agents. | Problem-Setter 与 Problem-Solver agent 之间双向 idea channel 的 schema。 |
| docs/paper_audit/REPOSITORY_EVIDENCE_AUDIT_ZH.md | 仓库证据审计报告（面向论文写作）· REPOSITORY EVIDENCE AUDIT | Read-only repository evidence audit for paper writing, with traceable and strength-qualified materials. | 面向论文写作的只读仓库证据审计，强调可追溯、可核验并区分证据强弱。 |
| docs/problem-package/schema.md | Problem Package — Schema | Single source of truth for the problem-package data contract between Problem-Setter and Problem-Solver fleets. | Problem-Setter 与 Problem-Solver fleet 之间 problem package 数据契约的单一事实来源。 |
| docs/reports/2026-05-28-progress.md | Progress Report — End-to-End SGLang Optimization Agent | Progress report showing the agent system found, packaged, and fixed a Qwen3-30B-A3B MoE TTFT regression in about two days. | 进展报告：说明 agent 系统约两天内发现、打包并修复 Qwen3-30B-A3B MoE 的 TTFT 回归。 |
| docs/research/regime-search-extensions.md | Regime Search & Input Generation — Strengthening Strategy | Research and engineering review of current Stage-A input construction limits and proposals for strengthening regime search. | Stage-A 输入构造与 regime search 的研究/工程评审，列出当前限制和增强方案。 |
| docs/skills/README.md | Skills — design principles & catalog | Design-principles and catalog document defining project skills as reusable procedural knowledge units. | skills 设计原则与目录，定义项目中的 skill 是可复用的流程知识单元。 |
| docs/README.md | docs/ | Documentation index explaining dated experiment directories, living named directories, and methodology placement conventions. | docs 目录说明：解释日期实验目录、命名常驻目录以及 methodology 应放到 skills 的约定。 |
| docs/architecture_primer.md | 推理侧模型架构入门 —— 从 infra 视角 | Infrastructure-oriented primer on inference-side model architecture using real Gemma-3 config shapes and project optimization links. | 面向 infra 的推理侧模型架构入门，用 Gemma-3 真实 config shape 并关联本项目优化记录。 |
| docs/kernel_fusion_catalogue.md | Kernel fusion 机会全集：我们找到的每一个、怎么找的、拿到多少 | Maintained index of all kernel-fusion discoveries and validations, with discovery method, change, gain, and evidence links. | kernel fusion 机会总索引：维护所有发现与验证，列出发现方式、改动、收益和证据位置。 |

## Repo-root documents / 仓库根文档

| File | Title (as written) | What it is (EN) | 说明 (中文) |
|---|---|---|---|
| README.md | EndtoEnd Optimization Agent for SGLang | Top-level bilingual README for the two-stage SGLang optimization agent that discovers poor regimes and fixes them through solver agents. | 仓库顶层双语 README，介绍两阶段 SGLang 优化 agent 如何发现差 regime 并由 solver agent 修复。 |
| HANDOFF_regime_kernel.md | Regime-aware Kernel Specialization — 交接文档 | Handoff document for regime-aware kernel specialization, giving current status, conclusions, and optional remaining work. | regime-aware kernel specialization 交接文档，说明当前状态、核心结论与可选后续工作。 |
| context.md | Copilot CLI Session | Exported Copilot CLI session log and context snapshot for the long-running project conversation. | Copilot CLI 会话导出与上下文快照，记录长期项目对话。 |
| plan.md | Plan / 项目状态（2026-07-28 更新） | Project status plan updated 2026-07-28, summarizing Gemma-3 PR validation, tests, evidence artifacts, and skill extraction. | 2026-07-28 更新的项目状态计划，汇总 Gemma-3 PR 验证、测试、证据产物与 skill 沉淀。 |
| sglang_cookbook_deployment_baselines.md | SGLang Cookbook Deployment Baselines | Generated baseline index of normalized `sglang serve` deployment commands from official cookbook config snippets. | 从官方 cookbook config snippets 生成的 `sglang serve` 部署命令 baseline 索引。 |
