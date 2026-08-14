# Reusable skills catalog / 可复用技能目录

These are reusable, self-contained methodology skills under `.github/skills/`. Each `SKILL.md` defines a contract — inputs → procedure → outputs — so an agent can invoke the same method on a new model, workload, server configuration, or profiling artifact. This catalog treats the skills as a core deliverable: the reusable methodology, not just the one-time benchmark results.

这些技能位于 `.github/skills/`，是可复用、自包含的方法论模块。每个 `SKILL.md` 都定义了一个契约：输入 → 执行流程 → 输出，使 AI agent 能把同一方法迁移到新的模型、负载、服务器配置或 profiling 产物上。本目录强调：核心交付物是可复用的方法论，而不只是一次性的实验结果。

## Skills at a glance / 速览

| Skill | One-line purpose (EN) | 一句话用途 (中文) | When to use / 何时使用 |
|---|---|---|---|
| `regime-bench-harness` | Deterministically launch a server from a bench spec, run all regimes, quality-gate, and clean up. | 从 bench spec 确定性启动服务、跑全 regimes、做质量门禁并清理。 | Use for real end-to-end experiments from `bench-spec.yaml`; preferred over ad-hoc launch + bench scripts. / 用于基于 `bench-spec.yaml` 的正式端到端实验，替代手写启动和压测流程。 |
| `e2e-bench-runner` | Run repeated multi-regime benchmarks against an already-running OpenAI-compatible or sglang-native server. | 对已运行的 OpenAI-compatible 或 sglang-native 服务执行多 regime、多重复 benchmark。 | Use for A/B or N-way config comparisons, after non-trivial config/source changes, and before declaring a fix or regression. / 用于 A/B 或多配置比较、重要配置或源码变更后、宣布修复或回归前。 |
| `regime-sweep-runner` | Deprecated matrix runner that compared configs × regimes by calling `e2e-bench-runner`. | 已废弃的 configs × regimes 矩阵 runner，内部调用 `e2e-bench-runner`。 | Do not use in new work; use `regime-bench-harness` and shell loops over bench specs instead. / 新工作不要调用；改用 `regime-bench-harness` 和 bench spec 循环。 |
| `nsys-capture` | Capture an arbitrary action with Nsight Systems and immediately export SQLite for downstream SQL analysis. | 用 Nsight Systems 捕获任意动作，并立即导出 SQLite 供后续 SQL 分析。 | Use when a bench gap is unexplained, when kernel/timeline evidence is needed, or before/after a hot-kernel patch. / 当 benchmark 差距无法由日志解释、需要 kernel/timeline 证据、或热 kernel 补丁前后对比时使用。 |
| `nsys-timeline-sql` | Reduce an nsys SQLite trace into GPU active/idle, top kernels, idle gaps, launch counts, memcpy, and queryable recipes. | 将 nsys SQLite trace 归约为 GPU 活跃/空闲、热点 kernel、空洞、launch 次数、memcpy 和可查询 recipes。 | Use after `nsys-capture` succeeds, or to diff baseline vs patched timelines. / 在 `nsys-capture` 成功后，或对比 baseline 与 patched timeline 时使用。 |
| `ncu-microarch` | Profile a narrowed hot kernel with Nsight Compute and summarize microarchitectural bottlenecks. | 用 Nsight Compute 针对已缩小范围的热点 kernel 采集微架构瓶颈。 | Use after `nsys-timeline-sql` identifies a hot kernel and before claiming it is memory-, compute-, occupancy-, or tensor-core-bound. / 在 `nsys-timeline-sql` 定位热点 kernel 后、声明其受内存/计算/占用率/Tensor Core 限制前使用。 |
| `pytorch-profiling` | Capture and reduce an sglang Torch profiler trace into phase, top-kernel, MoE, and CUDA-graph summaries. | 捕获并归约 sglang Torch profiler trace，输出阶段、热点 kernel、MoE 和 CUDA graph 摘要。 | Use when L1–L3 evidence is insufficient, config sweeps fail, or a kernel agent lacks `profile_summary.json`. / 当 L1–L3 证据不足、配置 sweep 无效、或 kernel agent 缺少 `profile_summary.json` 时使用。 |
| `profile-summary-unified` | Merge bench, nsys, torch profile, and framework profile outputs into one evidence-attributed JSON. | 将 bench、nsys、torch profile 和框架 profile 结果合并为带 evidence attribution 的统一 JSON。 | Use at the end of an investigation before writing a conclusion or handoff, when at least two profile sources exist. / 在调查结束、写结论或 handoff 前，且至少有两个 profiling 来源时使用。 |
| `fusion-gap-hunting` | Find cases where a model runs unfused code even though the framework already has a fused primitive. | 找出模型没有调用框架已有 fused primitive、仍跑 unfused 路径的情况。 | Use before writing a new kernel, for non-primary model families, new architectures, or profiles with large elementwise/other/norm buckets. / 写新 kernel 前、非主流模型族/新架构、或 profile 中 elementwise/other/norm 占比高时使用。 |
| `cross-regime-anomaly` | Rank interesting cross-regime matrix patterns such as winner inversions, regime-dependent gaps, and unreliable cells. | 对跨 regime 矩阵中的 winner inversion、regime-dependent gap、不可靠 cell 等异常排序。 | Use after a matrix has at least 2 configs × 2 regimes and no specific hypothesis yet. / 当矩阵至少有 2 配置 × 2 regime，且尚无明确假设时使用。 |
| `boundary-expansion` | Generate neighboring workload YAMLs around a suspicious workload axis to expose nonlinear regime boundaries. | 围绕可疑负载轴生成邻近 workload YAML，用于暴露非线性 regime 边界。 | Use in Stage 1 second wave after suspicious/lonely seed runs, capacity flags, KV pressure, or failures. / 在 Stage 1 第二波中，seed 可疑/孤立、容量标志、KV 压力或失败后使用。 |
| `minimal-repro-shrink` | Shrink workload axes until a suspicious symptom disappears, preserving the smallest reproducer. | 逐步缩小 workload 轴直到症状消失，保留最小可复现负载。 | Use before handing Stage 1 cases to Stage 2 or when Stage 2 needs a faster repro. / 在 Stage 1 案例交给 Stage 2 前，或 Stage 2 需要更快 repro 时使用。 |
| `server-log-mining` | Parse `server.log` into structured features such as CUDA graph range, capacity, KV pressure, and retract events. | 将 `server.log` 解析为 CUDA graph 范围、容量、KV 压力、retract 等结构化特征。 | Use immediately after every benchmark run and before scoring or diagnosis. / 每次 benchmark 后、评分或诊断前立即使用。 |
| `failure-classification` | Classify failures and near-failures into typed labels that downstream logic can branch on. | 将失败和 near-failure 分类为下游逻辑可分支处理的类型标签。 | Use after metrics parsing and `server-log-mining` for every run, including passed runs. / 每次 run 在 metrics 解析和 `server-log-mining` 后使用，包括通过的 run。 |
| `noise-aware-scoring` | Calibrate metric noise from repeated baseline runs and provide adjusted thresholds. | 通过重复 baseline run 校准指标噪声，并提供调整后的阈值。 | Use once per model/server/hardware scouting tuple and again after fleet drift or every 20 experiments. / 每个模型/服务配置/硬件组合 scouting 开始时使用，并在环境漂移或每 20 次实验后重跑。 |
| `suspicion-scoring` | Combine log features, classifications, noise baselines, and local nonlinearity into auditable suspicion scores. | 将日志特征、失败分类、噪声基线和局部非线性组合为可审计的可疑分数。 | Use after every benchmark wave, because neighbor population changes can alter scores. / 每一波 benchmark 后使用，因为新增邻居会改变评分。 |
| `three-layer-optimization-campaign` | Run a serving-config / kernel-config / kernel-rewrite factorial campaign to test layered optimization claims. | 执行 serving-config / kernel-config / kernel-rewrite 的阶乘实验，检验分层优化结论。 | Use when testing whether a result generalizes to another model or whether kernel rewriting still pays after tuning ceilings. / 当需要验证单模型结论能否泛化，或 kernel rewrite 在调优 ceiling 后是否仍有效时使用。 |
| `handoff-prompt-template` | Provide a structured, falsifiable markdown handoff from analysis agent to coding agent. | 提供从分析 agent 到编码 agent 的结构化、可证伪 markdown handoff。 | Use when analysis has a concrete file:line change, evidence chain, acceptance test, and bounded scope for a coding agent. / 当分析已有具体 file:line 改动、证据链、验收测试和范围边界，需交给 coding agent 时使用。 |

## Skill details / 技能详情

### regime-bench-harness

- **Purpose / 目的:** EN: Run a complete regime-based end-to-end benchmark from one `bench-spec.yaml`, managing server lifecycle, health checks, deterministic spec hashing, quality gate, summary writing, and cleanup. 中文：从单个 `bench-spec.yaml` 执行完整 regime-based 端到端 benchmark，负责服务生命周期、健康检查、确定性 `spec_hash`、质量门禁、summary 写入和清理。
- **Inputs → Outputs / 输入 → 输出:** EN: Input `--spec` under `bench-specs/` and `--out-dir`; outputs schema-v1 `summary.json`, `server.log`, `per_run/*.json`, `regimes_resolved.yaml`, and `server_config_used.yaml`. 中文：输入 `bench-specs/` 下的 `--spec` 和 `--out-dir`；输出 schema-v1 `summary.json`、`server.log`、`per_run/*.json`、`regimes_resolved.yaml` 和 `server_config_used.yaml`。
- **Wraps / 封装:** EN: `harness/run_bench.py`; wraps `e2e-bench-runner` plus server launch/wait/kill and quality gate. 中文：封装 `harness/run_bench.py`；在 `e2e-bench-runner` 外增加服务启动/等待/强制清理和质量门禁。
- **When to invoke / 何时调用:** EN: When benchmarking a specific model/engine/backend/dtype config, replaying a past `spec_hash`, or doing N-way comparisons via N bench specs. 中文：当需要 benchmark 某个模型/engine/backend/dtype 配置、复现实验 `spec_hash`，或用多个 bench spec 做 N-way 比较时调用。

### e2e-bench-runner

- **Purpose / 目的:** EN: Run repeated multi-regime end-to-end benchmarks against an already-running server, dropping cold run 1 and reporting throughput, latency percentiles, completion rate, and run-to-run stddev. 中文：对已运行服务执行多 regime、重复端到端 benchmark，丢弃冷启动第 1 次，并报告吞吐、延迟分位、完成率和 run-to-run 标准差。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs include `url`, `backend`, `tag`, regimes or `regimes_file`, `num_runs`, and `out_dir`; outputs `bench_summary.json` and raw `per_run/<regime>_run<N>.json`. 中文：输入包括 `url`、`backend`、`tag`、regimes 或 `regimes_file`、`num_runs`、`out_dir`；输出 `bench_summary.json` 和原始 `per_run/<regime>_run<N>.json`。
- **Wraps / 封装:** EN: `.github/skills/e2e-bench-runner/impl/run_bench.py`; drives deterministic prompt generation and request execution against OpenAI-compatible / sglang endpoints. 中文：封装 `.github/skills/e2e-bench-runner/impl/run_bench.py`；以确定性 prompt 生成和请求执行驱动 OpenAI-compatible / sglang endpoint。
- **When to invoke / 何时调用:** EN: For first-touch config comparisons, after any non-trivial server config change, after applying a fix, and before declaring success or regression. 中文：用于首次配置比较、任何重要服务配置变更后、应用修复后，以及宣布成功或回归前。

### regime-sweep-runner

- **Purpose / 目的:** EN: Deprecated archival skill for sweeping an existing set of servers across configs × regimes and producing `regime_sweep_summary.json`. 中文：已废弃的归档技能，用于对已存在的服务器执行 configs × regimes sweep 并生成 `regime_sweep_summary.json`。
- **Inputs → Outputs / 输入 → 输出:** EN: Historical inputs were `configs_file`, `regimes_file`, `num_runs`, and `out_dir`; output was a flattened matrix `regime_sweep_summary.json` plus per-config bench artifacts. Current front matter declares no active inputs or outputs. 中文：历史输入为 `configs_file`、`regimes_file`、`num_runs`、`out_dir`；输出为扁平矩阵 `regime_sweep_summary.json` 和各配置 bench 产物。当前 front matter 声明无活跃输入/输出。
- **Wraps / 封装:** EN: `.github/skills/regime-sweep-runner/impl/sweep.py`, which serially called `e2e-bench-runner`. 中文：封装 `.github/skills/regime-sweep-runner/impl/sweep.py`，其串行调用 `e2e-bench-runner`。
- **When to invoke / 何时调用:** EN: Do not invoke in new work; `SKILL.md` says to use `regime-bench-harness` instead. 中文：新工作不要调用；`SKILL.md` 明确要求改用 `regime-bench-harness`。

### nsys-capture

- **Purpose / 目的:** EN: Wrap a benchmark, curl, sleep, or other target command with `nsys profile`, then immediately export `.nsys-rep` to SQLite so downstream skills can query raw timeline data. 中文：用 `nsys profile` 包裹 benchmark、curl、sleep 或其他目标命令，并立即将 `.nsys-rep` 导出为 SQLite，供下游技能查询原始 timeline 数据。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `target_cmd`, `duration_s`, `gpu_id`, `out_dir`, and optional `extra_nsys`; outputs `nsys_capture.json`, `profile.nsys-rep`, and `profile.sqlite`. 中文：输入为 `target_cmd`、`duration_s`、`gpu_id`、`out_dir` 和可选 `extra_nsys`；输出 `nsys_capture.json`、`profile.nsys-rep`、`profile.sqlite`。
- **Wraps / 封装:** EN: `.github/skills/nsys-capture/impl/run_capture.py`; drives Nsight Systems `nsys profile` and `nsys export --type sqlite`. 中文：封装 `.github/skills/nsys-capture/impl/run_capture.py`；驱动 Nsight Systems 的 `nsys profile` 和 `nsys export --type sqlite`。
- **When to invoke / 何时调用:** EN: When a benchmark gap is ≥20% and not explained by `server-log-mining`, when GPU active/idle, per-kernel cost, CPU launch count, stream parallelism, or cudagraph behavior is needed, or before/after hot-kernel patches. 中文：当 benchmark 差距 ≥20% 且 `server-log-mining` 无法解释，或需要 GPU 活跃/空闲、kernel 成本、CPU launch 次数、stream 并行、cudagraph 行为，或热 kernel 补丁前后对比时调用。

### nsys-timeline-sql

- **Purpose / 目的:** EN: Turn an nsys-exported SQLite database into a compact `timeline_summary.json` with GPU utilization, streams, top kernels, largest idle gaps, CUDA API launches, graph/eager ratio, and memcpy stats. 中文：将 nsys 导出的 SQLite 数据库转换为紧凑的 `timeline_summary.json`，包含 GPU 利用率、stream、热点 kernel、最大 idle gap、CUDA API launch、graph/eager 比例和 memcpy 统计。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `sqlite_path`, `out_dir`, optional `stream_id`, `top_n`, and optional `window_ns`; outputs `timeline_summary.json` and `recipes_used.json`, with `diff` and read-only `query` modes. 中文：输入为 `sqlite_path`、`out_dir`、可选 `stream_id`、`top_n`、可选 `window_ns`；输出 `timeline_summary.json` 和 `recipes_used.json`，并支持 `diff` 与只读 `query` 模式。
- **Wraps / 封装:** EN: `.github/skills/nsys-timeline-sql/impl/summarize.py` plus SQL recipes under `recipes/`. 中文：封装 `.github/skills/nsys-timeline-sql/impl/summarize.py` 以及 `recipes/` 下的 SQL recipes。
- **When to invoke / 何时调用:** EN: After `nsys-capture` produces an ok SQLite trace; also for baseline-vs-patched timeline diffs or focused read-only SQL questions. 中文：在 `nsys-capture` 成功生成 SQLite trace 后调用；也用于 baseline 与 patched timeline diff，或有聚焦问题时执行只读 SQL。

### ncu-microarch

- **Purpose / 目的:** EN: Answer why a named hot kernel is slow by collecting curated Nsight Compute metrics and deriving verdicts such as compute-bound, memory-bound, latency-bound, low occupancy, or tensor-core idle. 中文：通过采集精选 Nsight Compute 指标，解释某个已命名热点 kernel 为什么慢，并给出 compute-bound、memory-bound、latency-bound、low occupancy、tensor-core idle 等判断。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `target_cmd`, required `kernel_regex`, `launch_count`, optional `metrics_file`, `gpu_id`, and `out_dir`; outputs `ncu_summary.json` and raw `ncu_raw.csv`. 中文：输入为 `target_cmd`、必需 `kernel_regex`、`launch_count`、可选 `metrics_file`、`gpu_id`、`out_dir`；输出 `ncu_summary.json` 和原始 `ncu_raw.csv`。
- **Wraps / 封装:** EN: `.github/skills/ncu-microarch/impl/run_ncu.py`; drives `sudo -n ncu` with `--kernel-name regex:<regex>`, metric sets, CSV export, and summary reduction. 中文：封装 `.github/skills/ncu-microarch/impl/run_ncu.py`；驱动 `sudo -n ncu`、`--kernel-name regex:<regex>`、metric set、CSV 导出和摘要归约。
- **When to invoke / 何时调用:** EN: After `nsys-timeline-sql` identifies a hot kernel taking >15% of GPU active time and before making microarchitectural or source-level optimization claims. 中文：在 `nsys-timeline-sql` 找到占 GPU active time >15% 的热点 kernel 后、提出微架构或源码优化判断前调用。

### pytorch-profiling

- **Purpose / 目的:** EN: Capture an sglang Torch profiler trace for a target workload and reduce it into top kernels, phase breakdown, MoE dispatch/routing overhead, and CUDA graph fallback signals. 中文：为目标 workload 捕获 sglang Torch profiler trace，并归约为热点 kernel、阶段 breakdown、MoE dispatch/routing overhead、CUDA graph fallback 信号。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `candidate_config`, `workload`, and `profile_num_steps`; outputs structured `profile_summary.json` and optional raw `raw_trace/*.json.gz` or trace files. 中文：输入为 `candidate_config`、`workload`、`profile_num_steps`；输出结构化 `profile_summary.json` 和可选原始 `raw_trace/*.json.gz` 或 trace 文件。
- **Wraps / 封装:** EN: `.github/skills/pytorch-profiling/impl/run_profile.py` and `impl/parse_trace.py`; uses sglang launch parity, `SGLANG_TORCH_PROFILER_DIR`, and `sglang.bench_serving --profile`. 中文：封装 `.github/skills/pytorch-profiling/impl/run_profile.py` 与 `impl/parse_trace.py`；使用 sglang 启动配置一致性、`SGLANG_TORCH_PROFILER_DIR` 和 `sglang.bench_serving --profile`。
- **When to invoke / 何时调用:** EN: Stage A when server-log/failure/neighbor evidence is insufficient; Stage B after config sweeps fail; mandatory before a kernel agent proposes source-level changes without an existing profile. 中文：Stage A 中日志/失败/邻居证据不足时；Stage B 中配置 sweep 失败后；kernel agent 在无现成 profile 时提出源码改动前必须调用。

### profile-summary-unified

- **Purpose / 目的:** EN: Produce one canonical `profile_unified.json` that merges profiling artifacts and records an `evidence_chain` for field-level skill attribution. 中文：生成一个规范化 `profile_unified.json`，合并多种 profiling 产物，并通过 `evidence_chain` 记录字段级 skill attribution。
- **Inputs → Outputs / 输入 → 输出:** EN: Optional inputs include `bench_summary`, `timeline_summary`, `torch_profile_text`, `sglang_profile`, plus `subject`, `workload`, and `out`; output is `profile_unified.json` following `schema/profile_unified.schema.json`. 中文：可选输入包括 `bench_summary`、`timeline_summary`、`torch_profile_text`、`sglang_profile`，以及 `subject`、`workload`、`out`；输出符合 `schema/profile_unified.schema.json` 的 `profile_unified.json`。
- **Wraps / 封装:** EN: `.github/skills/profile-summary-unified/impl/unify.py` with adapters for e2e, nsys timeline, torch profiler text, sglang profile, and future NCU sources. 中文：封装 `.github/skills/profile-summary-unified/impl/unify.py`，包含 e2e、nsys timeline、torch profiler text、sglang profile 和未来 NCU 来源的 adapter。
- **When to invoke / 何时调用:** EN: At the end of an investigation, before conclusions or handoff, when at least two different profiling skill outputs should be merged and cited. 中文：在调查结束、写结论或 handoff 前，且至少有两个不同 profiling 技能输出需要合并和引用时调用。

### fusion-gap-hunting

- **Purpose / 目的:** EN: Find high-leverage missed fusions: model paths that run eager/unfused operators even though the framework already ships a fused primitive, then confirm by FX sweep or operator audit before fixing. 中文：寻找高杠杆 missed fusion：模型路径仍跑 eager/unfused operator，但框架已有 fused primitive；随后用 FX sweep 或 operator audit 确认再修复。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `framework_src`, `model`, and `gpu`; outputs `fusion_gap_candidates.json`, `gap.json`, `audit.json`, and a confirmed/refuted verdict with deciding numbers. 中文：输入为 `framework_src`、`model`、`gpu`；输出 `fusion_gap_candidates.json`、`gap.json`、`audit.json`，以及带决定性数字的 confirmed/refuted verdict。
- **Wraps / 封装:** EN: `.github/skills/fusion-gap-hunting/impl/scan_fusion_gaps.py`; also uses static AST/grep-style scans, `scripts/fx_fusion/fx_dispatch_gap_detector.py`, and `scripts/lfm_fusion/lf_audit.py`. 中文：封装 `.github/skills/fusion-gap-hunting/impl/scan_fusion_gaps.py`；还使用静态 AST/grep 风格扫描、`scripts/fx_fusion/fx_dispatch_gap_detector.py` 和 `scripts/lfm_fusion/lf_audit.py`。
- **When to invoke / 何时调用:** EN: Before writing a new kernel, for a model family not central to the framework, for newly added architectures, or when operator profiles show significant `elementwise`, `other`, or `norm` buckets. 中文：在写新 kernel 前、处理框架非核心模型族、新增架构，或 operator profile 显示 `elementwise`、`other`、`norm` bucket 占比较高时调用。

### cross-regime-anomaly

- **Purpose / 目的:** EN: Surface and rank optimization opportunities from a regime sweep matrix: winner inversions, large uniform gaps, reliability flags, failed cells, regime-dependent gaps, and outlier regimes. 中文：从 regime sweep 矩阵中发现并排序优化机会：winner inversion、大型一致 gap、可靠性告警、失败 cell、regime-dependent gap、outlier regime。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `sweep_file`, `top_n`, and `min_gap_pct`; output is `anomaly_report.json` with ranked findings, evidence, hypothesis seeds, and recommended next skill. 中文：输入为 `sweep_file`、`top_n`、`min_gap_pct`；输出 `anomaly_report.json`，包含排序 findings、证据、hypothesis seed 和推荐 next skill。
- **Wraps / 封装:** EN: `.github/skills/cross-regime-anomaly/impl/find.py`, a rule-based detector set. 中文：封装 `.github/skills/cross-regime-anomaly/impl/find.py`，使用规则化 detector 集合。
- **When to invoke / 何时调用:** EN: After `regime-sweep-runner` or equivalent matrix output has at least 2 configs × 2 regimes, especially when no specific hypothesis exists or when comparing baseline vs patched configs. 中文：在 `regime-sweep-runner` 或等价矩阵输出至少有 2 配置 × 2 regime 后调用，尤其是尚无明确假设或比较 baseline 与 patched 配置时。

### boundary-expansion

- **Purpose / 目的:** EN: Adaptively generate neighbor workload YAMLs along an implicated axis so local nonlinearity and boundary cliffs can be measured instead of inferred from isolated points. 中文：沿被怀疑的轴自适应生成邻近 workload YAML，使局部非线性和边界 cliff 能被实测，而不是从孤立点推断。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `parent_workload`, `axis`, and `search_space`; outputs one neighbor workload YAML per selected axis value and optional `summary_json`. 中文：输入为 `parent_workload`、`axis`、`search_space`；输出每个选中轴值对应的邻居 workload YAML，以及可选 `summary_json`。
- **Wraps / 封装:** EN: `.github/skills/boundary-expansion/impl/expand.py`; supports `bracket`, `geometric`, `downward`, and `upward` strategies. 中文：封装 `.github/skills/boundary-expansion/impl/expand.py`；支持 `bracket`、`geometric`、`downward`、`upward` 策略。
- **When to invoke / 何时调用:** EN: In Stage 1 second-wave exploration after seed runs show concurrency caps, too-small CUDA graph capture, high token usage, failures needing downward shrink, or lonely regime-hint clusters. 中文：在 Stage 1 第二波探索中，当 seed run 显示 concurrency cap、CUDA graph capture 太小、高 token usage、需要向下缩小的失败、或孤立 regime-hint cluster 时调用。

### minimal-repro-shrink

- **Purpose / 目的:** EN: Reduce a suspicious workload along cost axes until the symptom disappears, keeping the smallest workload that still reproduces the suspicious metric and using the shrink path as diagnostic evidence. 中文：沿成本轴缩小可疑 workload 直到症状消失，保留仍能复现可疑指标的最小 workload，并把 shrink 路径作为诊断证据。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are case workload YAML, target symptom, and baseline config; planned outputs are shrunk workload YAML and `shrink_log.jsonl` with provenance. 中文：输入为 case workload YAML、目标症状和 baseline config；计划输出为 shrunk workload YAML 和带 provenance 的 `shrink_log.jsonl`。
- **Wraps / 封装:** EN: No implementation in v0.3; `SKILL.md` specifies a planned sequential-halving procedure, with dependencies on `server-log-mining` and `failure-classification`. 中文：v0.3 中暂无实现；`SKILL.md` 规定了计划中的顺序 halving 流程，并依赖 `server-log-mining` 和 `failure-classification`。
- **When to invoke / 何时调用:** EN: Before handing a suspicious Stage 1 case to Stage 2, or when Stage 2 needs a faster reproducer for one-knob-at-a-time experiments. 中文：在把 Stage 1 可疑案例交给 Stage 2 前，或 Stage 2 需要更快的复现实例来做单旋钮实验时调用。

### server-log-mining

- **Purpose / 目的:** EN: Extract structured server-side evidence from sglang `server.log`, including CUDA graph configured/captured ranges, request caps, KV pressure, retracts, OOM/crash events, and derived booleans. 中文：从 sglang `server.log` 提取结构化服务端证据，包括 CUDA graph 配置/捕获范围、请求上限、KV 压力、retract、OOM/crash 事件和派生布尔值。
- **Inputs → Outputs / 输入 → 输出:** EN: Input `server_log`; output `server_features.json` with parsed fields, warnings, and errors. 中文：输入 `server_log`；输出 `server_features.json`，包含解析字段、warnings 和 errors。
- **Wraps / 封装:** EN: `.github/skills/server-log-mining/impl/parse_server_log.py`, a regex-based best-effort parser. 中文：封装 `.github/skills/server-log-mining/impl/parse_server_log.py`，即基于正则的 best-effort parser。
- **When to invoke / 何时调用:** EN: Immediately after every benchmark run, before scoring, case selection, diagnosis, or fix-attempt evaluation. 中文：每次 benchmark run 后立即调用，在评分、case 选择、诊断或修复尝试评估之前。

### failure-classification

- **Purpose / 目的:** EN: Assign every run, including passed runs, a typed failure or near-failure category so downstream scoring and agents can branch deterministically. 中文：为每个 run（包括通过的 run）分配失败或 near-failure 类型，使下游评分和 agent 能确定性分支处理。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are parsed `metrics`, `server_features`, and optional `bench_log_text`; output is `classification.json` with enum label and evidence. 中文：输入为解析后的 `metrics`、`server_features` 和可选 `bench_log_text`；输出含枚举标签和证据的 `classification.json`。
- **Wraps / 封装:** EN: `.github/skills/failure-classification/impl/classify.py`; pure Python with no I/O beyond reading input JSON/text. 中文：封装 `.github/skills/failure-classification/impl/classify.py`；纯 Python，除读取输入 JSON/text 外无额外 I/O。
- **When to invoke / 何时调用:** EN: After metrics parsing and `server-log-mining` for every run, regardless of pass/fail, to detect `clean_pass`, near-failure, partial success, OOM, crash, timeout, parse error, or unknown failure. 中文：每个 run 在 metrics 解析和 `server-log-mining` 后调用，无论通过/失败，用于检测 `clean_pass`、near-failure、partial success、OOM、crash、timeout、parse error 或 unknown failure。

### noise-aware-scoring

- **Purpose / 目的:** EN: Empirically calibrate per-metric coefficient of variation from repeated baseline runs and expose noise-adjusted decision thresholds. 中文：通过重复 baseline run 实证校准每个指标的 coefficient of variation，并提供 noise-adjusted 决策阈值。
- **Inputs → Outputs / 输入 → 输出:** EN: Input list of normalized baseline metrics JSONs or a calibration config/workload/repeats run; outputs `noise_baseline.json` and `adjusted_threshold(metric_name, base_threshold)`. 中文：输入为标准化 baseline metrics JSON 列表，或校准用 config/workload/repeats；输出 `noise_baseline.json` 和 `adjusted_threshold(metric_name, base_threshold)`。
- **Wraps / 封装:** EN: `.github/skills/noise-aware-scoring/impl/calibrate_noise.py` and `impl/threshold.py`; the calibration path spins the server and repeats the same `bench_serving` invocation. 中文：封装 `.github/skills/noise-aware-scoring/impl/calibrate_noise.py` 与 `impl/threshold.py`；校准路径会启动服务并重复相同 `bench_serving` 调用。
- **When to invoke / 何时调用:** EN: Once per `(model, server config, hardware)` scouting session, after large fleet changes, or every 20 experiments to detect drift. 中文：每个 `(model, server config, hardware)` scouting session 开始时调用一次，在大规模环境变化后或每 20 次实验后重跑以检测漂移。

### suspicion-scoring

- **Purpose / 目的:** EN: Compose server-log signals, failure classification, noise-aware thresholds, and local nonlinearity into one auditable suspicion score per workload run. 中文：将 server-log 信号、失败分类、noise-aware 阈值和局部非线性组合成每个 workload run 的可审计可疑分数。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `raw_results.jsonl`, per-run `server_features.json`, per-run `classifications.json`, and optional `noise_baseline.json`; output is `suspicious_cases.jsonl`. 中文：输入为 `raw_results.jsonl`、每个 run 的 `server_features.json`、每个 run 的 `classifications.json`、可选 `noise_baseline.json`；输出 `suspicious_cases.jsonl`。
- **Wraps / 封装:** EN: `.github/skills/suspicion-scoring/impl/score.py`, importing upstream skill outputs and `noise-aware-scoring` thresholds. 中文：封装 `.github/skills/suspicion-scoring/impl/score.py`，导入上游技能输出和 `noise-aware-scoring` 阈值。
- **When to invoke / 何时调用:** EN: After every wave of `run_regime_suite.py` or equivalent benchmark waves, because newly added neighbors can change local-nonlinearity scores. 中文：在每一波 `run_regime_suite.py` 或等价 benchmark wave 后调用，因为新增邻居会改变 local-nonlinearity 分数。

### three-layer-optimization-campaign

- **Purpose / 目的:** EN: Test whether a layered optimization claim generalizes by running serving-config (L1), kernel-config (L2), and kernel-rewrite (L3) factorial evidence across regimes and baselines. 中文：通过跨 regime 和 baseline 的 serving-config (L1)、kernel-config (L2)、kernel-rewrite (L3) 阶乘证据，检验分层优化结论是否泛化。
- **Inputs → Outputs / 输入 → 输出:** EN: Inputs are `model`, optional `control_model`, and `gpus`; outputs include `ceiling_per_regime.json`, `audit.json`, `exp3_layered_*_summary.json`, and deliverable `matrix.md`. 中文：输入为 `model`、可选 `control_model` 和 `gpus`；输出包括 `ceiling_per_regime.json`、`audit.json`、`exp3_layered_*_summary.json` 和交付表 `matrix.md`。
- **Wraps / 封装:** EN: A multi-stage methodology using `scripts/run_serving_ceiling_campaign.py`, `scripts/lfm_fusion/lf_audit.py`, optional `sglang_agent_kernel_lab.cli scan`, `scripts/lfm_fusion/exp3_layered.sh`, and related LFM fusion harnesses; depends on `fusion-gap-hunting`, `e2e-bench-runner`, `noise-aware-scoring`, and `regime-bench-harness`. 中文：封装多阶段方法，使用 `scripts/run_serving_ceiling_campaign.py`、`scripts/lfm_fusion/lf_audit.py`、可选 `sglang_agent_kernel_lab.cli scan`、`scripts/lfm_fusion/exp3_layered.sh` 和相关 LFM fusion harness；依赖 `fusion-gap-hunting`、`e2e-bench-runner`、`noise-aware-scoring`、`regime-bench-harness`。
- **When to invoke / 何时调用:** EN: When the question is whether a single-model result generalizes, whether kernel rewriting still pays after tuning ceilings, or what headroom a new model family has. 中文：当问题是单模型结果是否泛化、kernel rewriting 在 tuning ceiling 后是否仍有效，或新模型族还有多少优化空间时调用。

### handoff-prompt-template

- **Purpose / 目的:** EN: Provide a self-contained markdown contract that lets an analysis agent pass one concrete, falsifiable, minimal-scope code-change task to a coding agent without narrative loss or scope creep. 中文：提供自包含 markdown 契约，使分析 agent 能把一个具体、可证伪、最小范围的代码修改任务交给 coding agent，避免叙事丢失和范围膨胀。
- **Inputs → Outputs / 输入 → 输出:** EN: No runtime inputs; copy and fill the template. Output is one `handoff.md` per code-change task with required sections: problem, evidence chain, hypothesis, suggested change, acceptance test, risks, and what not to do. 中文：无运行时输入；复制并填写模板。输出为每个代码修改任务一个 `handoff.md`，包含问题、证据链、假设、建议改动、验收测试、风险和禁止事项等必需章节。
- **Wraps / 封装:** EN: No implementation; copy `.github/skills/handoff-prompt-template/template/handoff.md`. 中文：无实现脚本；复制 `.github/skills/handoff-prompt-template/template/handoff.md`。
- **When to invoke / 何时调用:** EN: When analysis has a specific file and line range, proposed patch or pseudocode, mechanical acceptance test, falsification path, and a next actor that should edit code without re-reading all original profiles. 中文：当分析已有具体文件和行范围、建议 patch 或伪代码、机械验收测试、证伪路径，并且下一位执行者应在不重读全部原始 profile 的情况下修改代码时调用。
