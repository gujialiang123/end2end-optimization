# Script Catalog / 脚本目录

This catalog indexes every `.py` and `.sh` file under `scripts/` from `docs/2026-08-14/scripts.tsv`, grouped by functional area; each entry gives a one-line English description based on the script docstring or leading comment plus a Chinese translation.

本目录索引 `docs/2026-08-14/scripts.tsv` 中 `scripts/` 目录下的每个 `.py` 与 `.sh` 文件，并按功能领域分组；每条记录都给出基于脚本文档字符串或开头注释的一句英文说明及中文翻译。

Total scripts cataloged / 收录脚本总数：**204**.

## Import-time patch shims / 导入时补丁垫片

Small sitecustomize hooks that install runtime patches before application code imports target modules. 这些小型 sitecustomize 钩子会在应用代码导入目标模块前安装运行时补丁。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/_siteinject/sitecustomize.py` | Import-time shim that installs custom_moe_patch when CUSTOM_MOE_INJECT=1. | 导入时垫片：当 CUSTOM_MOE_INJECT=1 时把 scripts 加入路径并安装 custom_moe_patch。 |

## Report, spreadsheet, and slide builders / 报告、表格与幻灯片生成

Scripts that convert experiment outputs into spreadsheets, reports, figures, or slide drafts. 这些脚本把实验输出转换成表格、报告、图或幻灯片草稿。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/add_mfu_retro.py` | Retro-annotate all existing bench summary.json files with MFU/MBU fields. | 为现有 bench summary.json 追补 MFU/MBU 字段。 |
| `scripts/build_chendi_v4_report.py` | Build spreadsheet per Chendi's spec (2026-07-07): | 按 Chendi 2026-07-07 规格生成电子表格报告。 |
| `scripts/build_config_spreadsheet.py` | Build consolidated config spreadsheet across v2 and v3 experiments. | 汇总 v2 与 v3 实验配置并生成统一配置表。 |
| `scripts/build_config_xlsx.py` | Convert consolidated_config_spreadsheet.csv → xlsx with formatting. | 将 consolidated_config_spreadsheet.csv 转成带格式的 xlsx。 |
| `scripts/build_v4_spreadsheet.py` | Build v4 cross-model spreadsheet. | 生成 v4 跨模型电子表格。 |
| `scripts/build_v5_kernel_report.py` | Aggregate v5 kineto kernel-level data into a shareable spreadsheet. | 把 v5 kineto kernel 级数据聚合为可分享的表格。 |
| `scripts/build_v5b_ncu_report.py` | Aggregate v5b NCU kernel-level metrics into a clean spreadsheet. | 把 v5b NCU kernel 级指标聚合为整洁表格。 |
| `scripts/build_v6_sglang_ncu_report.py` | Build unified NCU kernel report combining:   - June 9 data: Qwen3-30B-A3B (bf16), 1 config (cookbook), 4 regimes | 合并 June 9 Qwen3-30B-A3B 与后续数据生成统一 NCU kernel 报告。 |
| `scripts/update_performance_gap_slides.py` | Phase-10: build the first six slides of the performance-gap deck as a NEW draft. | Phase 10：把 performance-gap deck 前六页生成为新的草稿。 |

## Serving-ceiling campaigns and benchmark orchestration / Serving ceiling 活动与基准编排

Launch, run, parse, validate, and visualize serving-ceiling and benchmark campaigns. 这些脚本负责启动、运行、解析、验证和可视化 serving-ceiling 与基准活动。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/analyze_serving_ceiling.py` | Analysis for the 2026-07-24 serving-ceiling campaign. | 分析 2026-07-24 serving-ceiling 活动结果。 |
| `scripts/finalize_alternative_objectives.py` | Phase-5: validated selection for the alternative-objective study. | Phase 5：为替代目标研究做已验证的配置选择。 |
| `scripts/generate_seed_suite.py` | Materialize regime_scout/seed_suite.yaml → one workload YAML per seed. | 把 regime_scout/seed_suite.yaml 实例化为每个 seed 一个 workload YAML。 |
| `scripts/launch_server.py` | Launch a SGLang server from a YAML config. | 按 YAML 配置启动 SGLang server。 |
| `scripts/lfm25_serving_ceiling_per_regime.py` | What the serving-knob ceiling actually is on LFM2.5, per regime. | 逐 regime 衡量 LFM2.5 的 serving knob ceiling。 |
| `scripts/logging_setup.py` | Logging helper: file + stdout simultaneously, with structured fields. | 配置同时输出到文件和 stdout、带结构化字段的日志 helper。 |
| `scripts/parse_metrics.py` | Normalize one sglang.bench_serving --output-file jsonl into a stable metrics schema. | 把一个 sglang.bench_serving 输出 jsonl 规范化为稳定 metrics schema。 |
| `scripts/render_alternative_objective_figures.py` | Phase-6 figures for the alternative-objective study (PNG + SVG). | 为替代目标研究生成 Phase 6 PNG 与 SVG 图。 |
| `scripts/render_serving_ceiling_figures.py` | Slide-ready figures for the 2026-07-24 serving-ceiling campaign (Phase 9). | 为 2026-07-24 serving-ceiling 活动生成 Phase 9 投影片可用图。 |
| `scripts/run_alternative_objective_validation.py` | Phase-4: run ONLY the missing targeted validations for alternative objectives. | Phase 4：只运行替代目标缺失的 targeted validations。 |
| `scripts/run_benchmark.py` | Run one sglang.bench_serving invocation against an already-running server. | 对已运行的 server 发起一次 sglang.bench_serving 调用。 |
| `scripts/run_configs_with_gpu_profile.py` | Run selected configs with real GPU sampling + TTFT/TPOT capture from sglang. | 用真实 GPU 采样和 sglang TTFT/TPOT 捕获运行选定 configs。 |
| `scripts/run_experiment.py` | Run ONE workload end-to-end: launch server → wait → benchmark → parse → cleanup. | 端到端运行一个 workload：启动 server、等待、benchmark、parse、cleanup。 |
| `scripts/run_regime_suite.py` | Run a list of workload YAMLs under one fixed server config and collect metrics. | 在固定 server config 下运行一组 workload YAML 并收集 metrics。 |
| `scripts/run_serving_ceiling_campaign.py` | 2026-07-24 Qwen/LFM serving-ceiling campaign runner. | 运行 2026-07-24 Qwen/LFM serving-ceiling campaign。 |
| `scripts/run_serving_ceiling_validation.py` | Phase-4 validation pass: re-run selected configurations with N repetitions. | Phase 4 validation：以 N 次重复重跑选定配置。 |
| `scripts/serving_ceiling_lib.py` | Shared library for the 2026-07-24 Qwen/LFM serving-ceiling campaign. | 提供 2026-07-24 Qwen/LFM serving-ceiling campaign 的共享库。 |
| `scripts/trace_characterize.py` | Characterise the Mooncake replay traces before spending any GPU time. | 在消耗 GPU 时间前刻画 Mooncake replay traces。 |
| `scripts/wait_ready.py` | Poll a SGLang server until it answers /health, /v1/models, or accepts TCP. | 轮询 SGLang server，直到 /health、/v1/models 或 TCP 连接可用。 |

## NCU / Nsight / roofline tooling / NCU、Nsight 与 roofline 工具

Profiling, conversion, and analysis utilities for Nsight Compute, Nsight Systems, and roofline metrics. 这些工具用于 Nsight Compute、Nsight Systems 和 roofline 指标的采集、转换与分析。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/analyze_nsys_universal_config.py` | Analyze nsys profile of universal-config sglang run. | 分析 universal-config sglang 运行产生的 nsys profile。 |
| `scripts/bench_ncu_all_regimes.sh` | Run all 4 regimes through bench_ncu_one_regime.sh sequentially. | 顺序调用 bench_ncu_one_regime.sh 覆盖全部 4 个 regime。 |
| `scripts/bench_ncu_one_regime.sh` | Per-regime NCU runner — uses sglang.bench_one_batch with --profile-activities CUDA_PROFILER | 按单个 regime 运行 NCU，并用 CUDA_PROFILER 包裹 sglang.bench_one_batch。 |
| `scripts/compute_v10_roofline.py` | v10 roofline: decode end-to-end floor + per-kernel bandwidth roofline. | 计算 v10 decode 端到端下限与逐 kernel 带宽 roofline。 |
| `scripts/compute_v18_gflops.py` | v18: compute achieved GFLOP/s per kernel from NCU FLOP counters. | 用 NCU FLOP 计数器计算 v18 每个 kernel 的实际 GFLOP/s。 |
| `scripts/generate_ncu_reports.py` | Generate human-readable markdown NCU report per regime from ncu_summary.json. | 根据 ncu_summary.json 为每个 regime 生成可读 Markdown NCU 报告。 |
| `scripts/ncu_csv_to_summary.py` | Convert one regime's ncu_raw.csv (from bench_ncu_one_regime.sh) into the ncu_summary.json shape expected by profile-summary-unified. | 把单个 regime 的 ncu_raw.csv 转成 profile-summary-unified 期望的 ncu_summary.json 形状。 |
| `scripts/ncu_csv_wide_to_summary.py` | Convert NCU 'wide' CSV (one row per kernel, many metric columns) to ncu_summary.json shape. Used when CSV is from `ncu --import ... --csv --page raw` | 把 NCU wide CSV 转为 ncu_summary.json，适用于 ncu --import --csv --page raw 输出。 |
| `scripts/ncu_discover_kernels.py` | Discover which kernels are hot in sglang decode + prefill on a given model. | 发现给定模型在 sglang decode 与 prefill 中的热点 kernel。 |
| `scripts/ncu_one_regime.sh` | Per-regime NCU runner — launches sglang under ncu wrap, runs ONE regime, kills. | 按单个 regime 运行 NCU：在 ncu 包裹下启动 sglang、跑一个 regime、再杀掉。 |
| `scripts/ncu_run_one_regime_workload.py` | Run one regime workload against the local sglang server. Used inside ncu_one_regime.sh. | 对本地 sglang server 运行一个 regime workload，供 ncu_one_regime.sh 调用。 |
| `scripts/nsys_on_universal_config.sh` | Wrap sglang server under nsys, run a single regime workload, kill, export sqlite. | 在 nsys 下包裹 sglang server，跑一个 regime workload，停止后导出 sqlite。 |
| `scripts/parse_v18_ncu_long.py` | Parse NCU long-format ncu_raw.csv (Metric Name/Value) -> accurate achieved GFLOP/s. | 解析 NCU long-format ncu_raw.csv 为准确的 achieved GFLOP/s。 |
| `scripts/parse_v19b_ncu.py` | Parse v19b NCU CSVs into a per-regime, per-kernel-family summary of the 11 metrics, and compute decode roofline signals (DRAM %, occupancy limiter, bytes). | 解析 v19b NCU CSV，按 regime 与 kernel family 汇总 11 项指标并计算 decode roofline 信号。 |
| `scripts/run_v12_ncu_spec.sh` | v12: NCU on ngram-spec vs baseline decode -> measure how much SM idle (No-Eligible) spec reclaims | v12 NCU：比较 ngram-spec 与 baseline decode，以量化 spec 回收的 No-Eligible/SM idle。 |
| `scripts/run_v19b_ncu_decode.py` | v19 Part B: NCU decode-kernel profiling with Chendi's EXACT metric list. | v19 Part B 使用 Chendi 指定指标列表对 decode kernel 做 NCU profile。 |
| `scripts/run_v50_ncu_moe_microbench.py` | v50 NCU microbench: isolate the sglang Triton fused_moe grouped-GEMM kernel at a single token count M so Nsight Compute captures ONE clean launch for roofline. | v50 NCU microbench 在单一 token count M 隔离 sglang Triton fused_moe grouped-GEMM kernel 以捕获干净 roofline。 |
| `scripts/run_v50_ncu_roofline.sh` | v50 NCU roofline capture of the sglang Triton fused_moe grouped-GEMM kernel at | v50 对 sglang Triton fused_moe grouped-GEMM kernel 做 NCU roofline capture。 |
| `scripts/run_v5_ncu_profile.py` | v5 NCU + kineto profiling pipeline. | v5 运行 NCU + kineto profiling pipeline。 |
| `scripts/run_v5b_ncu.py` | v5b: NCU profiling for top hot kernels only. | v5b 只对最热 kernels 运行 NCU profiling。 |
| `scripts/run_v6_ncu_sglang.py` | v6 NCU on REAL sglang kernels via sglang.bench_one_batch. | v6 通过 sglang.bench_one_batch 对真实 sglang kernels 做 NCU。 |
| `scripts/run_v9_ncu_realworkload.py` | v9: NCU kernel profiling on REAL-workload representative points, with the v8-tuned BEST config, to show that tuning alone cannot reach the hardware limit. | v9 对真实 workload 代表点做 NCU kernel profiling，证明 tuning alone 达不到硬件上限。 |
| `scripts/run_v9d_nsys_serveridle.sh` | v9d: measure REAL server idle with nsys timeline (delay/duration aligns capture to bench). | v9d 用 nsys timeline 测量真实 server idle，按 bench 对齐 delay/duration。 |
| `scripts/unify_sweep.py` | Per-regime unifier: run profile-summary-unified for each of the 4 regimes. Idempotent — re-run anytime to refresh unified.json after ncu finishes. | 按 regime 调用 profile-summary-unified 刷新每个 regime 的 unified.json，且可重复运行。 |

## MoE routing, kernel tuning, and E2E experiments / MoE 路由、kernel 调优与端到端实验

Top-level MoE, routing, custom-kernel, and versioned end-to-end experiment scripts. 这些顶层脚本覆盖 MoE、路由、自定义 kernel 与版本化端到端实验。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/analyze_alternative_serving_objectives.py` | Alternative-objective re-analysis of the completed serving-ceiling campaign. | 对已完成的 serving-ceiling 活动按替代目标重新分析。 |
| `scripts/analyze_moe_bucket_usage.py` | Which MoE config bucket does each forward actually land in? | 统计每次 forward 实际落入哪个 MoE 配置桶。 |
| `scripts/analyze_v44_config_ab.py` | v42 analysis: compare baseline vs ours e2e A/B, per cell, with Welch t-test. | 对 v42 基线与 ours 的端到端 A/B 逐单元做 Welch t 检验。 |
| `scripts/analyze_v45_server_ab.py` | v45 analysis: server-level ours vs fallback, per regime, Welch t-test. | 对 v45 server 级 ours 与 fallback 按 regime 做 Welch t 检验。 |
| `scripts/autotune_two_regimes.sh` | Drive 2 Optuna studies sequentially on one GPU. | 在一张 GPU 上顺序驱动两个 Optuna study。 |
| `scripts/bench_moe_3way.py` | MoE GEMM 3-way performance benchmark on H200 bf16. v2 — fixed API. | 在 H200 bf16 上做三路 MoE GEMM 性能基准。 |
| `scripts/check_moe_down_config.py` | Can an up-projection and a down-projection MoE config be used together? | 检查 up-projection 与 down-projection MoE 配置能否搭配使用。 |
| `scripts/custom_moe_patch.py` | Monkeypatch sglang's TritonRunner.run to use a custom small-M (decode) MoE kernel when M<=SMALL_M_MAX and the case is bf16/gated/no-quant/shape-matched. Falls back to | Monkeypatch sglang TritonRunner.run，在小 M decode 的 bf16/gated/no-quant/shape-matched 场景使用自定义 MoE kernel，否则回退。 |
| `scripts/plot_v16.py` | Plot v16 router distributions from raw.npz (CPU only, no GPU). | 在 CPU 上从 raw.npz 绘制 v16 router 分布。 |
| `scripts/qwen15_gate_patch.py` | Patch Qwen2MoeSparseMoeBlock._forward_shared_experts to FUSE the shared-expert gate (linear + sigmoid + mul) into ONE triton kernel on CUDA — filling sglang's CUDA gap | 为 Qwen2MoeSparseMoeBlock._forward_shared_experts 融合 shared-expert gate 的 linear+sigmoid+mul CUDA Triton kernel。 |
| `scripts/run_e2e_bench.py` | E2E launcher: install custom MoE patch, then run sglang.bench_one_batch. Toggle CUSTOM_MOE=1 to use the custom small-M decode kernel (captured into cudagraph). | 安装 custom MoE patch 后运行 sglang.bench_one_batch，可用 CUSTOM_MOE=1 启用 custom small-M decode kernel。 |
| `scripts/run_e2e_correctness.py` | Installs custom_moe_patch and runs sglang.bench_one_batch correctness_test on Qwen3-30B-A3B. | 安装 custom_moe_patch 并在 Qwen3-30B-A3B 上运行 sglang.bench_one_batch correctness_test。 |
| `scripts/run_e2e_qwen15.py` | Installs qwen15_gate_patch and runs sglang.bench_one_batch on Qwen1.5-MoE-A2.7B-Chat. | 安装 qwen15_gate_patch 并在 Qwen1.5-MoE-A2.7B-Chat 上运行 bench_one_batch。 |
| `scripts/run_e2e_qwen15_verify.py` | Installs qwen15_gate_patch and runs a short no-CUDA-graph Qwen1.5-MoE verification benchmark. | 安装 qwen15_gate_patch 并运行短输出、禁用 CUDA graph 的 Qwen1.5-MoE 验证 benchmark。 |
| `scripts/run_e2e_sweep.py` | Installs custom_moe_patch and sweeps sglang.bench_one_batch batch sizes for a Qwen MoE model. | 安装 custom_moe_patch 并对 Qwen MoE 模型扫描 bench_one_batch batch size。 |
| `scripts/run_e2e_verify.py` | Installs custom_moe_patch and runs a short no-CUDA-graph Qwen3-30B-A3B verification benchmark. | 安装 custom_moe_patch 并运行短输出、禁用 CUDA graph 的 Qwen3-30B-A3B 验证 benchmark。 |
| `scripts/run_v10_load_sweep.py` | v10: offered-load sweep — how much does higher concurrency recover? | v10 offered-load sweep：评估更高 concurrency 能恢复多少性能。 |
| `scripts/run_v11b2_multistream.sh` | v11-B2: multi-stream utilization sweep — prove serving idle is recoverable by | v11-B2 multi-stream utilization sweep：证明 serving idle 可被恢复。 |
| `scripts/run_v13_router_analysis.py` | v13: analyze MoE router behavior on agentic input (Qwen3-30B-A3B). | v13 分析 Qwen3-30B-A3B 在 agentic input 上的 MoE router 行为。 |
| `scripts/run_v14_consolidation.py` | v14: simulate batch-level expert consolidation -> transfer saving vs cost. | v14 模拟 batch 级 expert consolidation，估计节省与代价。 |
| `scripts/run_v14b_consolidation_batch.py` | v14b: consolidation tradeoff AS A FUNCTION OF BATCH SIZE. | v14b 分析 consolidation tradeoff 如何随 batch size 变化。 |
| `scripts/run_v15_perplexity.py` | v15: REAL perplexity cost of reducing active experts (top-k reduction). | v15 测量减少 active experts/top-k reduction 的真实 perplexity 代价。 |
| `scripts/run_v16_router_dist.py` | v16: DETAILED router distribution analysis (Qwen3-30B-A3B, agent input). | v16 对 Qwen3-30B-A3B agent input 做详细 router 分布分析。 |
| `scripts/run_v17_gsm8k_topk.py` | v17: GSM8K accuracy + timing vs active experts (top-k reduction). | v17 测量 GSM8K accuracy 与 timing 随 active experts/top-k reduction 的变化。 |
| `scripts/run_v18_dynamic_topk.py` | v18: DYNAMIC top-k (confidence-adaptive) vs fixed top-k on GSM8K. | v18 在 GSM8K 上比较 confidence-adaptive dynamic top-k 与 fixed top-k。 |
| `scripts/run_v19_wall_sweep.sh` | v19 Part A: prefill vs decode wall proportion across concurrency, agent (toolagent) workload. | v19 Part A 扫描 toolagent workload 下 concurrency 对 prefill/decode wall 占比的影响。 |
| `scripts/run_v23_config_evidence.py` | v23: Evidence for config-tuning PRs — our re-tuned fused_moe config vs sglang's stale fallback config, on the exact Qwen3-30B-A3B shape (E=128,N=768,H200). | v23 为 config-tuning PR 生成证据：在 Qwen3-30B-A3B 真实 shape 上比较重调 fused_moe config 与 sglang stale fallback。 |
| `scripts/run_v23_generic.py` | v23c: model-agnostic config-tuning benchmark. Compares default heuristic vs a tuned fused_moe config on the real triton kernel, for any MoE model shape. | v23c 泛模型 config-tuning benchmark：在真实 Triton kernel 上比较默认 heuristic 与 tuned fused_moe config。 |
| `scripts/run_v24_shared_expert_fusion.py` | v24: shared-expert fusion opportunity (reproduces the SPIRIT of #22325 / #26727 via torch.compile). Qwen2-MoE-style shared-expert path (Qwen1.5-MoE-A2.7B dims): | v24 用 torch.compile 复现 shared-expert fusion 机会，模拟 Qwen2-MoE shared-expert 路径。 |
| `scripts/run_v25_kernel_fusion.py` | v25: KERNEL-LEVEL improvement evidence — reproduce #22325 (fuse linear+sigmoid+mul in shared_experts) by actually WRITING a fused triton kernel and measuring its speedup | v25 编写 fused Triton kernel 复现 linear+sigmoid+mul shared_experts fusion 并测加速。 |
| `scripts/run_v26_swiglu_fusion.py` | v26: second kernel-level fusion (bandwidth-saving, persists at large batch). Fuses the SwiGLU activation silu(gate)*up over the shared MLP intermediate [M, N=5632]. | v26 融合 shared MLP intermediate 上的 SwiGLU silu(gate)*up，验证大 batch 下仍节省带宽。 |
| `scripts/run_v27_moe_baseline.py` | v27: sglang fused_moe real baseline + achieved HBM bandwidth (decode/prefill). | v27 测量 sglang fused_moe 真实 baseline 与 decode/prefill achieved HBM bandwidth。 |
| `scripts/run_v28_b1_diagnose.py` | v28: diagnose b1 decode MoE — can the existing triton kernel hit higher bandwidth with a different (parallelism-heavy) config? Sweep BLOCK_N / GROUP / warps / stages at | v28 诊断 b1 decode MoE，通过扫描 BLOCK_N/GROUP/warps/stages 判断现有 Triton kernel 是否能提高带宽。 |
| `scripts/run_v29_custom_moe.py` | v29: custom small-M decode MoE kernel attempt (bf16). Target: beat sglang's fused_moe at M=1 (which is only 49% HBM / 24us GEMM + 6us overhead). | v29 尝试 bf16 small-M decode 自定义 MoE kernel，目标超过 sglang fused_moe 的 M=1 表现。 |
| `scripts/run_v30_custom_moe_tldot.py` | v30: small-M decode MoE with tensor cores (tl.dot), skipping align/sort and fusing act+sum. Target sglang's b1 31.8us (24us GEMM + 6us overhead). Idea: match the GEMM with | v30 用 tl.dot/tensor cores 实现 small-M decode MoE，跳过 align/sort 并融合 act+sum。 |
| `scripts/run_v31_tune_custom.py` | v31: tune the custom small-M MoE (v30) tiling to try to actually beat sglang b1 (31.8us). Sweeps BN/BK/num_warps for both kernels; reports best vs sglang. | v31 扫描 BN/BK/num_warps 调优 v30 small-M MoE kernel 并与 sglang b1 对比。 |
| `scripts/run_v32_scan_M.py` | v32: generalize the winning small-M MoE kernel to M>1 (token = pair//topk) and find the crossover batch where sglang (which groups by expert, reading each weight once) | v32 将获胜 small-M MoE kernel 推广到 M>1，并寻找与 sglang 分组按 expert 读取权重策略的交叉 batch。 |
| `scripts/run_v34_figures.py` | v34: build the 'headroom beyond tuning' figures for Qwen3-30B-A3B (decode). All numbers measured in this project (cited in the doc). Saves PNGs. | v34 生成 Qwen3-30B-A3B decode 的“tuning 之外 headroom”图并保存 PNG。 |
| `scripts/run_v41_noise_verify.py` | Chendi verification: is the custom-MoE-kernel b1 +1.4% e2e real or noise? Runs MANY interleaved separate bench_one_batch launches (cudagraph ON, matching | Chendi 验证：用多次交错独立 bench_one_batch 检验 custom-MoE-kernel b1 +1.4% e2e 是否只是噪声。 |
| `scripts/run_v42_kernel_e2e.py` | v42: END-TO-END validation of MoE kernel-config tuning across ALL regimes. Answers Chendi/user: the §1.6 kernel-level +35-54% (tuned config vs default | v42 对所有 regimes 做 MoE kernel-config tuning 的端到端验证。 |
| `scripts/run_v43_server_e2e.py` | v43: server-level END-TO-END A/B (default heuristic vs tuned MoE config) across our artificial regimes + the sglang agent dataset (mooncake toolagent). | v43 在人工 regimes 与 sglang agent dataset 上做 server 级 default heuristic vs tuned MoE config A/B。 |
| `scripts/run_v44_e2e_config_ab.py` | v42: End-to-end A/B of fused_moe kernel-config tuning (the gap left by §1.6). | v44/v42 对 fused_moe kernel-config tuning 剩余 gap 做端到端 A/B。 |
| `scripts/run_v45_server_ours_vs_fallback.py` | v45: server-level END-TO-END A/B — OURS (re-tuned on triton 3.6.0) vs the FALLBACK config sglang actually loads — across all regimes + the agent dataset. | v45 在所有 regimes 与 agent dataset 上做 server 级 ours vs fallback A/B。 |
| `scripts/run_v46_ab.py` | v46 A/B: OURS (retuned for triton 3.5.1) vs FALLBACK (sglang现状, triton_3_2_0). | v46 比较为 Triton 3.5.1 重调的 ours 与 sglang 当前 Triton 3.2.0 fallback。 |
| `scripts/run_v4_decode_sweep.py` | v4 decode-stress sweep: 3 models × 3 configs × 5 regimes = 45 combos. | v4 decode-stress sweep：3 个模型 × 3 个配置 × 5 个 regime 共 45 组。 |
| `scripts/run_v4_lfm25_addon.py` | Add-on to v4 sweep: run only LFM2.5-8B-A1B (same 3 configs × 5 regimes). | v4 sweep 补充：只运行 LFM2.5-8B-A1B 的 3 configs × 5 regimes。 |
| `scripts/run_v51_high_conc_ttft.py` | v51 orchestrator — high-concurrency TTFT rerun (2 models x 3 configs x 2 regimes). | v51 编排高并发 TTFT 重跑：2 models × 3 configs × 2 regimes。 |
| `scripts/run_v7_agentic_bench.py` | v7: characterize REAL / agentic workloads via sglang.bench_serving. | v7 用 sglang.bench_serving 刻画真实/agentic workloads。 |
| `scripts/run_v7_config_sweep.py` | v7 config sweep: compare tuned configs on realistic agent workloads. | v7 config sweep：在真实 agent workloads 上比较 tuned configs。 |
| `scripts/run_v8_tuning_sweep.py` | v8: knob tuning sweep on REAL agent workloads (one model per GPU). | v8 在真实 agent workloads 上做 knob tuning sweep，每个模型一张 GPU。 |
| `scripts/serve_with_patch.py` | Launch sglang server with custom_moe_patch installed before model load. Usage: python scripts/serve_with_patch.py -- <sglang.launch_server args> | 在模型加载前安装 custom_moe_patch 并启动 sglang server。 |
| `scripts/v51_stream_bench.py` | v51 — high-concurrency TTFT rerun for the v4 slide data points. | v51 为 v4 投影片数据点做高并发 TTFT stream benchmark 重跑。 |

## Case-specific PR A/B and plateau campaigns / 特定 PR A/B 与 plateau 活动

Focused validation campaigns for individual upstream PRs, regressions, baselines, and plateau studies. 这些脚本针对单个上游 PR、回归、baseline 与 plateau 研究做专项验证。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/run_v46_l2norm_recompile_microbench.py` | v46 microbench: does PR #31558 (do_not_specialize=["T"]) actually remove per-token-count recompilation of the FLA l2norm Triton kernel? | v46 microbench 检查 PR #31558 的 do_not_specialize=["T"] 是否消除 FLA l2norm 按 token count 重编译。 |
| `scripts/run_v47_pr31558_server_ab.py` | v47: server-level e2e A/B for PR #31558 (avoid FLA l2-norm recompile by token count) on Qwen3.6-35B-A3B-FP8 (hybrid linear-attention VLM). | v47 在 Qwen3.6-35B-A3B-FP8 上做 PR #31558 避免 FLA l2-norm 重编译的 server 级 A/B。 |
| `scripts/run_v48_baseline.py` | v48 cookbook baseline reference — measured SEPARATELY, never enqueued into Optuna. | v48 cookbook baseline reference，单独测量且不进入 Optuna 队列。 |
| `scripts/run_v48_dsv4_pr29007_ab.py` | v48: DeepSeek-V4-Flash-FP8 e2e A/B for PR #29007 (MoE TP allreduce via NCCL symmetric memory / in-pool output allocation) on 8×H200 TP8. | v48 在 8×H200 TP8 上对 DeepSeek-V4-Flash-FP8 做 PR #29007 MoE TP allreduce server A/B。 |
| `scripts/run_v48_lfm25_plateau.py` | v48 — LFM2.5 serving-knob autotuning plateau study (clean, no warm start). | v48 对 LFM2.5 serving-knob autotuning plateau 做干净无 warm-start 研究。 |
| `scripts/run_v48_plots.py` | v48 plots + plateau analysis. Reads per_trial_log.csv, baseline_reference.json, best_validated.json. Emits 3 figures (png+svg) and prints plateau statistics that | v48 读取 trial/baseline/best validation 结果，生成 PNG+SVG 图并输出 plateau 统计。 |
| `scripts/run_v48_validate.py` | v48 post-search validation — re-run top configs + cookbook, interleaved x5. | v48 搜索后验证：交错重复 5 次重跑 top configs 与 cookbook。 |
| `scripts/run_v49_pr31438_mm_preproc_ab.py` | v49: PR #31438 (parallelize multimodal preprocessing) e2e A/B on Qwen3.6-35B VLM. | v49 在 Qwen3.6-35B VLM 上做 PR #31438 并行化多模态预处理的端到端 A/B。 |

## FX graph fusion discovery / FX 图融合发现

Torch FX and torch.compile scanners, feasibility checks, and accuracy gates for fusion opportunities. 这些脚本用 Torch FX 与 torch.compile 扫描融合机会，并做可行性与准确率门禁。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/fx_fusion/accuracy_vs_fp64.py` | Is the fused kernel as accurate as the path it replaces? | 检验 fused kernel 相比被替代路径是否保持 FP64 精度。 |
| `scripts/fx_fusion/e2e_ab_gemma3.py` | End-to-end A/B of the fused qk-norm + rope path on Gemma-3. | 在 Gemma-3 上对 fused QK-norm + RoPE 路径做端到端 A/B。 |
| `scripts/fx_fusion/fx_dispatch_gap_detector.py` | Detect dispatch gaps by comparing a module's FX graph across input shapes. | 比较不同输入形状下模块 FX 图以发现 dispatch gap。 |
| `scripts/fx_fusion/fx_fusion_scanner.py` | Find fusable elementwise chains in a torch.compile FX graph. | 在 torch.compile FX 图中寻找可融合的 elementwise 链。 |
| `scripts/fx_fusion/fx_scan_models.py` | Run the FX fusion scanner over real HF models. | 在真实 HF 模型上运行 FX fusion scanner。 |
| `scripts/fx_fusion/gemma3_fused_qknorm_rope_feasibility.py` | Can Gemma-3 use the fused QK-norm + RoPE kernel sglang already ships? | 检查 Gemma-3 是否能调用 sglang 已有的 fused QK-norm + RoPE kernel。 |
| `scripts/fx_fusion/gsm8k_accuracy_gate.py` | Does the fused path cost any task accuracy? | 检查 fused path 是否影响 GSM8K 任务准确率。 |
| `scripts/fx_fusion/gsm8k_paired_test.py` | Paired significance test for the GSM8K arms. | 对 GSM8K 实验臂做配对显著性检验。 |
| `scripts/fx_fusion/locate_fused_qknorm_error.py` | Where does the 3.94% actually come from? | 定位 fused QK-norm 3.94% 误差的来源。 |
| `scripts/fx_fusion/scan_models_pipeline.py` | Run the fusion-gap pipeline over several models and report what it finds. | 在多个模型上运行 fusion-gap pipeline 并汇报发现。 |
| `scripts/fx_fusion/scan_qknorm_rope_candidates.py` | Which models could call fused_qk_norm_rope but do not? | 找出本可调用 fused_qk_norm_rope 但尚未调用的模型。 |
| `scripts/fx_fusion/verify_add_one_kernel.py` | Does the add_one kernel variant remove the accuracy cost of folding (1 + w)? | 验证 add_one kernel 变体是否消除折叠 (1 + w) 的精度损失。 |
| `scripts/fx_fusion/verify_against_model_path.py` | [b, h, s, d] -> [b, s, h, d], then flatten to the kernel's layout | 把张量从 [b,h,s,d] 转为 [b,s,h,d] 并展平到 kernel 需要的布局以对照模型路径。 |
| `scripts/fx_fusion/verify_generation_identical.py` | Does the fused path change what the model generates? | 验证 fused path 是否改变模型生成结果。 |
| `scripts/fx_fusion/verify_qknorm_merge.py` | Validate the QK-norm merge opportunity found by FX scanning. | 验证 FX 扫描发现的 QK-norm 合并机会。 |

## LFM2.5 / cross-model kernel fusion / LFM2.5 与跨模型 kernel 融合

The LFM2.5 and cross-model fusion harness, patches, injections, microbenchmarks, audits, and RT workload analysis. 这一组覆盖 LFM2.5 及跨模型融合 harness、补丁、注入、微基准、审计和 RT workload 分析。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/lfm_fusion/exp3_analyze.py` | Analyse the 2x2 layered experiment on LFM2.5 regime C. | 分析 LFM2.5 regime C 的 2x2 layered 实验。 |
| `scripts/lfm_fusion/exp3_latency.py` | Latency for the cells where throughput is the wrong yardstick. | 为吞吐量不是合适指标的实验单元计算 latency。 |
| `scripts/lfm_fusion/exp3_layered.sh` | Experiment 3: does the kernel rewrite still pay once the baseline already has | 实验 3：评估 baseline 已有优化后 kernel rewrite 是否仍有收益。 |
| `scripts/lfm_fusion/exp3_layered_cfgonly.sh` | Experiment 3: does the kernel rewrite still pay once the baseline already has | 实验 3 配置版：评估 baseline 已有优化后 kernel rewrite 是否仍有收益。 |
| `scripts/lfm_fusion/exp3_littles_law.py` | Is the client the bottleneck, or is the server? | 用 Little 定律判断瓶颈在 client 还是 server。 |
| `scripts/lfm_fusion/exp3_moesum_marginal.py` | Marginal contribution of `moesum`, at both baseline levels, on regime C. | 在 regime C 上衡量 moesum 在两个 baseline level 下的边际贡献。 |
| `scripts/lfm_fusion/falcon_fusion_patch.py` | Opt-in kernel patches for Falcon-H1 (and any mamba2 hybrid). | 为 Falcon-H1 及 mamba2 hybrid 提供可选 kernel patch。 |
| `scripts/lfm_fusion/fh_inject/sitecustomize.py` | Both Falcon-H1 shims in one module. | 在一个模块中导入并安装两个 Falcon-H1 shim。 |
| `scripts/lfm_fusion/fh_sweep_ssd_tiles.py` | Sweep the mamba SSD kernels' tile sizes on Falcon-H1 prefill. | 在 Falcon-H1 prefill 上扫描 mamba SSD kernel 的 tile size。 |
| `scripts/lfm_fusion/fx_bench_elementwise_paths.py` | Decompose the ShortConv glue into its three elementwise passes and compare each against a fully contiguous baseline of identical HBM traffic. | 分解 ShortConv glue 的三个 elementwise pass，并与相同 HBM 流量的连续 baseline 比较。 |
| `scripts/lfm_fusion/fx_bench_fusions.py` | Microbenchmarks for the ShortConv fusion candidates found by the FX/Inductor graph study. | 对 FX/Inductor 图研究发现的 ShortConv fusion candidate 做 microbenchmark。 |
| `scripts/lfm_fusion/fx_common.py` | Shared helpers for the LFM2.5 FX / dynamo graph-export study. | 提供 LFM2.5 FX / dynamo 图导出研究的共享 helper。 |
| `scripts/lfm_fusion/fx_export_graphs.py` | Export FX / dynamo / AOT graphs and Inductor output code for the LFM2.5 ShortConv module and a full decoder layer. | 导出 LFM2.5 ShortConv 模块与完整 decoder layer 的 FX/dynamo/AOT 图和 Inductor 代码。 |
| `scripts/lfm_fusion/fx_verify_fusion.py` | Correctness verification for the ShortConv fusion candidates. | 验证 ShortConv fusion candidate 的正确性。 |
| `scripts/lfm_fusion/gap_table_2026_08_04.py` | Cross-model gap table for the 2026-08-04 campaign, against the Qwen control. | 以 Qwen control 为基准生成 2026-08-04 活动的跨模型 gap 表。 |
| `scripts/lfm_fusion/gemma_fusion_patch.py` | Opt-in fusion patch for SGLang's Gemma-3 implementation. | 为 SGLang Gemma-3 实现提供可选 fusion patch。 |
| `scripts/lfm_fusion/gm_inject/sitecustomize.py` | Applies the Gemma-3 fusion patch at import time (loaded via PYTHONPATH). | 通过 PYTHONPATH 加载，在导入时应用 Gemma-3 fusion patch。 |
| `scripts/lfm_fusion/lf_analyze.py` | Aggregate the LFM2.5 fusion A/B into tidy CSVs plus a Welch t-test. | 把 LFM2.5 fusion A/B 聚合成整洁 CSV 并做 Welch t 检验。 |
| `scripts/lfm_fusion/lf_audit.py` | Operator-level audit of LFM2.5 (and Qwen as control) across the three regimes. | 跨三个 regime 对 LFM2.5 与 Qwen control 做 operator 级审计。 |
| `scripts/lfm_fusion/lf_bench_moesum.py` | Correctness-gated microbenchmark for MoE sum + residual RMSNorm. | 对 MoE sum + residual RMSNorm fused kernel 做带正确性门禁的 microbenchmark。 |
| `scripts/lfm_fusion/lf_bench_shortconv.py` | Correctness-gated microbenchmark for the fused ShortConv glue kernels. | 对 fused ShortConv glue kernels 做带正确性门禁的 microbenchmark。 |
| `scripts/lfm_fusion/lf_correctness.py` | Numeric correctness gate for the LFM2.5 fusion patch. | 为 LFM2.5 fusion patch 做数值正确性门禁。 |
| `scripts/lfm_fusion/lf_e2e.py` | End-to-end A/B for the LFM2.5 fusion patch, with a correctness gate. | 对 LFM2.5 fusion patch 做带正确性门禁的端到端 A/B。 |
| `scripts/lfm_fusion/lf_inject/sitecustomize.py` | Applies the LFM2.5 fusion patches at import time (loaded via PYTHONPATH). | 通过 PYTHONPATH 加载，在导入时应用 LFM2.5 fusion patches。 |
| `scripts/lfm_fusion/lf_lib.py` | Shared definitions for the LFM2.5 kernel-fusion / rewrite study. | 提供 LFM2.5 kernel fusion / rewrite study 的共享定义。 |
| `scripts/lfm_fusion/lf_plot_crossarch.py` | Cross-architecture audit figure: the gap tracks model family, not novelty. | 绘制跨架构审计图，说明 gap 跟随模型家族而非新颖度。 |
| `scripts/lfm_fusion/lf_plots.py` | Two figures for the LFM2.5 fusion study. | 为 LFM2.5 fusion study 生成两张图。 |
| `scripts/lfm_fusion/lf_precision_analysis.py` | Precision analysis: is a fused-kernel swap lossy, or just a different rounding? | 分析 fused-kernel swap 是有损还是仅改变舍入。 |
| `scripts/lfm_fusion/lf_triton_moesum.py` | Fused top-k MoE reduction, residual add, and RMSNorm for LFM2.5. | 实现 LFM2.5 的 top-k MoE reduction、residual add 与 RMSNorm fused Triton kernel。 |
| `scripts/lfm_fusion/lf_triton_shortconv.py` | Fused Triton kernels for the LFM2.5 gated short-convolution path. | 实现 LFM2.5 gated short-convolution 路径的 fused Triton kernels。 |
| `scripts/lfm_fusion/lf_tune_shortconv.py` | Tile-size sweep for the fused ShortConv kernels. | 为 fused ShortConv kernels 扫描 tile size。 |
| `scripts/lfm_fusion/lfm_fusion_patch.py` | Opt-in fusion patches for SGLang's LFM2.5 (Lfm2Moe) implementation. | 为 SGLang LFM2.5 Lfm2Moe 实现提供可选 fusion patch。 |
| `scripts/lfm_fusion/mamba_inject/sitecustomize.py` | Make bench_one_batch usable on hybrid-mamba models. | 让 bench_one_batch 可用于 hybrid-mamba 模型。 |
| `scripts/lfm_fusion/nsys_analyze.py` | Derive LFM2.5 kernel timelines and fusion ceilings from nsys SQLite exports. | 从 nsys SQLite export 推导 LFM2.5 kernel timeline 与 fusion ceiling。 |
| `scripts/lfm_fusion/ol_inject/sitecustomize.py` | Applies the OLMo-2 fusion patches at import time (loaded via PYTHONPATH). | 通过 PYTHONPATH 加载，在导入时应用 OLMo-2 fusion patches。 |
| `scripts/lfm_fusion/ol_triton_normadd.py` | Fused RMSNorm-then-add for OLMo-2. | 实现 OLMo-2 的 RMSNorm-then-add fused kernel。 |
| `scripts/lfm_fusion/olmo2_fusion_patch.py` | Opt-in fusion patches for SGLang's OLMo-2 implementation. | 为 SGLang OLMo-2 实现提供可选 fusion patches。 |
| `scripts/lfm_fusion/pr_verify_gemma3.py` | PR-grade numeric verification of the Gemma3RMSNorm CUDA fix. | 对 Gemma3RMSNorm CUDA 修复做 PR 级数值验证。 |
| `scripts/lfm_fusion/rt_l2l3_consolidate.py` | Consolidate the L2/L3/L2+L3 ablation on the real/agentic RT_ workloads. | 汇总真实/agentic RT workloads 上 L2/L3/L2+L3 消融结果。 |
| `scripts/lfm_fusion/rt_l2l3_matrix.sh` | Run the L2/L3/L2+L3 ablation (the 2x2 {MoE config off/on} x {L3 off/on}, | 运行 2x2 {MoE config off/on} × {L3 off/on} 的 L2/L3/L2+L3 消融。 |
| `scripts/lfm_fusion/rt_load_curve.py` | Turn the Tool-Agent arrival-load sweep into a saturation curve. | 把 Tool-Agent 到达负载 sweep 转成饱和曲线。 |
| `scripts/lfm_fusion/rt_plot_load_curve.py` | Plot the Tool-Agent arrival-load sweep. | 绘制 Tool-Agent 到达负载 sweep。 |
| `scripts/lfm_fusion/rt_plot_workload_matrix.py` | Plot L3's effect across every real / agentic workload, both metrics. | 绘制 L3 对每个 real/agentic workload 两类指标的影响。 |
| `scripts/lfm_fusion/rt_workload_matrix.py` | Compare the L3 effect across every real / agentic workload measured. | 比较已测每个 real/agentic workload 中 L3 的效果。 |
| `scripts/lfm_fusion/run_audit_all.sh` | Operator-level fusion-gap audit: LFM2.5 across 3 regimes + Qwen3-30B control. | 执行 operator 级 fusion-gap 审计：LFM2.5 三个 regime 加 Qwen3-30B control。 |
| `scripts/lfm_fusion/ssd_inject/sitecustomize.py` | Override the mamba SSD kernels' hardcoded tile sizes at launch time. | 在 launch time 覆盖 mamba SSD kernels 的硬编码 tile size。 |

## Regime-aware kernel specialization / Regime 感知 kernel 专门化

Closed-loop, profile-building, backend-comparison, routing-control, and plotting scripts for regime-aware kernel specialization. 这些脚本支持 regime 感知 kernel 专门化的闭环、profile 构建、backend 对比、routing control 与绘图。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/regime_kernel/rk_agent.py` | Minimal closed-loop kernel-specialization agent (RQ4). | 提供 RQ4 最小闭环 kernel-specialization agent。 |
| `scripts/regime_kernel/rk_backend_table.py` | Build the K1 backend-comparison table from the raw per-run JSON. | 从原始 per-run JSON 构建 K1 backend 对比表。 |
| `scripts/regime_kernel/rk_backends.py` | K1: does the best MoE kernel IMPLEMENTATION differ by regime? | 研究最佳 MoE kernel 实现是否随 regime 改变。 |
| `scripts/regime_kernel/rk_build_config.py` | Build a tuned MoE config from sweep results, using the guarded policy. | 用 guarded policy 从 sweep results 构建 tuned MoE config。 |
| `scripts/regime_kernel/rk_campaign.py` | P0 driver: tuning sweep, transfer matrix and routing control. | P0 driver：执行 tuning sweep、transfer matrix 与 routing control。 |
| `scripts/regime_kernel/rk_e2e.py` | End-to-end stage: swap MoE kernel profiles with serving knobs frozen. | 在 serving knobs 固定时替换 MoE kernel profiles 做端到端阶段实验。 |
| `scripts/regime_kernel/rk_guarded_profile.py` | Build a guardrailed regime-aware profile. | 构建带 guardrail 的 regime-aware profile。 |
| `scripts/regime_kernel/rk_lib.py` | Regime-aware kernel specialization — shared harness. | 提供 regime-aware kernel specialization 共享 harness。 |
| `scripts/regime_kernel/rk_microbench.py` | Fused-MoE kernel microbenchmark + correctness engine. | 提供 fused-MoE kernel microbenchmark 与 correctness engine。 |
| `scripts/regime_kernel/rk_plots.py` | Figures for the regime-aware kernel specialization study. | 为 regime-aware kernel specialization study 生成图。 |
| `scripts/regime_kernel/rk_process.py` | Turn raw regime-kernel results into the tidy tables the plots consume. | 把 raw regime-kernel results 转为绘图所需的 tidy tables。 |
| `scripts/regime_kernel/rk_profiles.py` | Build kernel profiles and compare selection strategies (RQ2 / RQ3). | 构建 kernel profiles 并比较 RQ2/RQ3 的选择策略。 |
| `scripts/regime_kernel/rk_routing_cross.py` | Does the regime change the optimal kernel config BEYOND the shape it implies? | 检验 regime 是否在 shape 之外改变 optimal kernel config。 |
| `scripts/regime_kernel/rk_trace/sitecustomize.py` | Opt-in MoE kernel trace (loaded as sitecustomize when on PYTHONPATH). | 通过 PYTHONPATH 加载的可选 MoE kernel trace sitecustomize。 |
| `scripts/regime_kernel/run_qwen_backends.sh` | HANDOFF §8.1 -- Qwen backend comparison (K1 cross-model validation). | 运行 handoff §8.1 的 Qwen backend comparison 以做 K1 跨模型验证。 |

## Regime study and hardware-view profiling / Regime study 与 hardware-view profiling

Regime-study aggregation, hardware-view collection, GPU sampling, and MoE optimization-level runs. 这些脚本负责 regime-study 聚合、hardware-view 采集、GPU 采样与 MoE 优化等级运行。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/regime_study/aggregate.py` | Aggregate regime-study runs across models + repeats into CSV + Markdown reports. | 把跨模型和重复的 regime-study runs 聚合成 CSV 与 Markdown 报告。 |
| `scripts/regime_study/aggregate_hw_view.py` | Aggregate hardware_view.json files across (model, regime) into one CSV + Markdown table. | 聚合各 model/regime 的 hardware_view.json 为 CSV 与 Markdown 表。 |
| `scripts/regime_study/aggregate_moe_opt_levels.py` | Aggregate MoE optimization-level study (configs/moe_variants/C*.yaml). | 聚合 configs/moe_variants/C*.yaml 的 MoE optimization-level study。 |
| `scripts/regime_study/detect_silent_noop.py` | Detect silent-no-op cases where a sglang server arg flag was ignored. | 检测 sglang server arg flag 被忽略的 silent-no-op 情况。 |
| `scripts/regime_study/gpu_sampler.py` | nvidia-smi periodic sampler. Append-only CSV. | 周期性调用 nvidia-smi 并追加写 CSV。 |
| `scripts/regime_study/run_hw_view.py` | Hardware-view run: one regime + nvidia-smi sampler + torch profile + /get_server_info. | 运行一个 regime 的 hardware view，包括 nvidia-smi sampler、torch profile 与 /get_server_info。 |
| `scripts/regime_study/run_moe_opt_levels.sh` | Run hardware-view profile on 7 MoE optimization-knob variants × R8 regime. | 对 7 个 MoE optimization-knob variants × R8 regime 运行 hardware-view profile。 |
| `scripts/regime_study/run_moe_real_kernel_swap.sh` | Run hardware-view profile on the 2 REAL kernel-swap MoE variants × R8. | 对 2 个真实 kernel-swap MoE variants × R8 运行 hardware-view profile。 |

## Stage-1 triage and config-agent workflow / Stage 1 分诊与 config-agent 流程

Early-stage suspiciousness scoring, regime clustering, problem-package selection, and minimal config-agent code. 这些脚本处理早期可疑度打分、regime 聚类、problem package 选择和最小 config-agent。

| Script | What it does (EN) | 说明 (中文) |
|---|---|---|
| `scripts/archive/run_stage1_v0.2.py` | Stage 1 end-to-end orchestrator. | Stage 1 端到端编排器。 |
| `scripts/archive/score_suspicion_v1.py` | Stage 1: score suspiciousness of each workload run. | Stage 1：为每个 workload run 计算可疑度分数。 |
| `scripts/cluster_regimes.py` | Stage 1: cluster suspicious cases into regimes and emit a human-readable map. | Stage 1：把可疑案例聚成 regime 并输出可读映射。 |
| `scripts/select_cases_for_stage2.py` | Stage 1: pick top suspicious cases for Stage 2/3 handoff. | Stage 1：选择最可疑案例移交 Stage 2/3。 |
| `scripts/select_problems.py` | Stage 1: assemble v1 problem packages from suspicious cases. | Stage 1：从可疑案例组装 v1 problem packages。 |
| `scripts/solver/config_agent.py` | Minimal Stage B config-agent. | 提供最小 Stage B config-agent。 |
| `scripts/utils.py` | Shared utilities for Stage 1 scripts. | 提供 Stage 1 脚本共享工具。 |
