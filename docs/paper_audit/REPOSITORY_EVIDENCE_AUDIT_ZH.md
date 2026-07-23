# 仓库证据审计报告（面向论文写作）· REPOSITORY EVIDENCE AUDIT

> 语言：中文（术语保留英文）。本报告是对 `end2end-optimization` 仓库的一次**只读**证据审计，目标是为一篇"Agent 辅助 LLM inference 性能定位、诊断、决策与反证"论文，建立**可追溯、可核验、诚实区分证据强弱**的素材。
> 配套机器可读产物：`evidence_catalog.jsonl`、`diagnosis_catalog.yaml`、`manual_validation_queue.yaml`、`tool_inventory.yaml`、`paper_claim_matrix.yaml`（同目录）。

---

# 0. 审计信息与范围

- **commit SHA**：`100ba9f6434c92779fbe7431161f820045cd5c54`
- **分支**：`main`
- **git status**：仅 `result.jsonl` 有未提交改动（工作区实验副产物），其余干净
- **审计日期**：2026-07-23（UTC）
- **审计方法**：4 个并行只读 explore agent 分区扫描（early June / mid v7–v22 / PR-repro v44–v50 / infra）+ 人工核验关键原始 JSON/CSV。**未运行任何 GPU / NCU / nsys / server / autotuning。**
- **扫描目录**：`README.md` `context.md` `plan.md` `docs/**`（含 2026-06-01 至 2026-07-22 全部按日期目录 + architecture/problem-package/reports/research/development/idea-pool/skills）`experiments/**` `results/**`（68 个结果目录）`regimes/**` `regime_scout/**` `bench-specs/**` `configs/**` `harness/**` `scripts/**` `stages/**` `.github/skills/**`（17 个 skill）`patches/**` `result.jsonl` `sglang_cookbook_deployment_baselines.{json,md}`
- **未纳入 / 缺失范围**：
  - 部分 `logs/**`、`.ncu-rep`/`.nsys-rep`/`.sqlite`/`.trace.json.gz` 大文件已被 `.gitignore` 排除（可重生成，不在 git）
  - `v18_gflops`、`v22_teacher_forced` 仅定位到原始文件，**未找到对应文字报告**
  - `v7_v4_decode_sweep`、`v5_smoke`、`v5_ncu`、`gpu_profiled` 仅有 raw，无独立叙事文档
  - `archive/**` 仅作历史设计追溯，**不计入当前系统能力**
- **报告限制**：本审计以"追溯到原始数据"为准；凡只有 markdown 结论而未定位原始数据的，均显式标注"仅有文字报告，原始数据未定位"。

---

# 1. 执行摘要

## 1.1 系统真实实现程度

本仓库是一个**两阶段（Stage A 出题人 / Stage B 做题人）**的 SGLang 性能诊断与修复框架，围绕**磁盘上冻结的 problem package**做数据契约（`docs/problem-package/schema.md`，`README.md:6-20`）。真实实现的可运行部分：

- **确定性 benchmark harness**（`harness/run_bench.py` 等）：spec→launch server→bench→quality gate→schema-v1 `summary.json`，**纯确定性、无 LLM**（`harness/README.md:18-46`）。
- **Stage A 规则化 regime 探索**：seed 生成→regime sweep→suspicion 打分→聚类→选 case（`scripts/run_regime_suite.py`、`score_suspicion.py`、`cluster_regimes.py`、`select_cases_for_stage2.py`）。
- **Stage B config-agent**：`scripts/solver/config_agent.py` —— **固定的单-knob 扫描循环 + 接受判据**，跑 target+neighbors+controls，写 `decision.json` 与 `solution.md`。
- **一批 profiling/tuning skill**（17 个 `.github/skills/*`）：NCU、nsys、pytorch profile、server-log-mining、noise-aware-scoring、boundary-expansion 等，均为**确定性/规则化**工具。
- **Optuna autotuning harness**（`harness/autotune*.py`）。
- **大量一次性实验脚本** `scripts/run_v4_*.py … run_v50_*.py`（论文证据脚本，非核心基础设施）。

## 1.2 哪些是 deterministic pipeline vs 真正 LLM/Agent decision（关键诚实判断）

| 类别 | 内容 | 证据 |
|---|---|---|
| **Deterministic pipeline** | harness 全链路（spec/lifecycle/executor/quality/output）；所有 profiling skill 的 impl；bench 指标解析 | `harness/README.md:18-46`；`.github/skills/*/impl` |
| **Rule-based** | Stage A suspicion 打分、cross-regime-anomaly、failure-classification；**Stage B config_agent.py 的单-knob 扫描 + 接受判据是固定循环，不是 LLM** | `scripts/solver/config_agent.py:93-165,373-493`；`.github/skills/cross-regime-anomaly/SKILL.md:156-163` |
| **真正 LLM/Agent decision** | 仅 Stage A 的 **LLM triage 模式**（决定扩哪个 workload、选哪个 axis/strategy、何时停）——`stages/problem-setter/policies/llm_agent.md`。**本仓库绝大多数实验结论由人工假设 / 上游 PR / 规则脚本驱动，而非 LLM 自主发现。** |

> **对论文最重要的一条诚实边界**：本仓库当前**没有**证据表明"LLM Agent 自主发现了新的 inference 加速"。config_agent 是规则化 solver；v7–v50 的假设多来自人工/会议/上游 PR。可诚实主张的是"**结构化 workflow + 规则/LLM 选证据 + 确定性工具**能定位根因、路由干预、并**反证虚假优化**"。

## 1.3 目前最强的 5–8 条实验结论

1. **错误 baseline 制造虚假优化（最强反证案例）**：2026-06-25 审计发现早期"autotuning 5–9×"完全是 **cudagraph-OFF baseline** 造成的假象；对真实 default（cudagraph ON），Optuna best 仅 **0.95–1.05×（噪声内）**。原始表：`docs/2026-06-25/autotuning_honest_results.md:85-103`。
2. **MoE P001 admission-capacity mismatch，config-agent 单-knob 修复 +92.6%**：R_scheduler_tail 的 `ttft_p95` 2282→168.7ms，提升 `max-running-requests`（→64/96/128），带 neighbors+controls+decision。`experiments/problems_moe/P001/{solution.md,attempts/*/decision.json}`。这是**最完整的因果链 + intervention** 证据（D2–D3）。
3. **microbenchmark 加速不迁移到端到端（已 t 检验）**：自写 M=1 MoE kernel 隔离 1.23×，插回 sglang 端到端仅 **+1.17%（n=15, Welch |t|=6.51）**；b≥2 是真回归（−4.3%/−11.7%）。`results/2026-07-20_v41_noise/summary.json`。
4. **prefill/decode 瓶颈随 regime 迁移**：真实 server 拆分显示 decode-heavy/agent regime decode 占墙钟 **83–99%**，长上下文 prefill regime prefill 占 **55–89%**；NCU roofline 佐证 decode DRAM 87.9–89.8%、prefill compute 64.5–67.4%。`results/2026-07-20_v43_server_e2e/`、`results/2026-07-22_v50_ncu_roofline/roofline_summary.csv`、`docs/2026-07-22/prefill_vs_decode_bottleneck_report.md`。
5. **config-tuning 的端到端收益来自"别掉进 default 启发式"，而非重 tune 已覆盖 shape**：default→tuned +34~43% prefill（`v42/v43`），但 ours-retune vs 框架实际 fallback **≈0**（`v44/v45`，三层测量一致）。
6. **serving-knob autotuning 存在 plateau（无 warm-start 干净复现）**：LFM2.5 100-trial，TPE 第 7 个 config 即达最优区，最后 20 个提升 **0%**，validated best 仅比 cookbook **+0.4%（CI 重叠）**。`results/2026-07-22_lfm25_plateau_100/`。
7. **上游 PR 端到端复现（3 个正结果，明确非自主发现）**：#31558 l2norm 冷缓存重编译（首轮 8 分辨率 TTFT **−13.7%**）、#29007 dsv4 symm-mem allreduce（TPOT/E2E **+9~10%**，TP8）、#31438 多模态并行预处理（吞吐 **+14.5%**，bit-exact）。`results/2026-07-21_v47/v48/v49_*`。
8. **架构选择是 tuning/kernel 够不到的端到端杠杆**：线性注意力 LFM2.5 vs 全注意力 Qwen3，decode 随上下文 scaling **+24% vs +57%**（bs=32, 512→8192），Qwen 在 bs=32×16k OOM。`results/2026-07-20_v39_ctxscan/`。

## 1.4 目前最可信的论文故事

**"Evidence-guided diagnosis & falsification of LLM-inference optimizations across workload regimes."**（详见 §13 故事线 A/B/C 对比）——核心不是"Agent 发现加速"，而是"在确定性 benchmark/profiling 工具之上，用结构化流程 + 规则/LLM 选证据，定位根因、路由干预、并**拒绝**由错误 baseline / 局部 benchmark / cold-start / cache / 统计噪声造成的虚假结论"。仓库里有**大量成对的"初看是加速→核验为伪影/噪声"案例**，这正是稀缺且有价值的论文素材。

## 1.5 当前最大证据缺口

1. **Agent attribution 弱**：绝大多数"下一步做什么"是人工决定；LLM triage 模式无端到端 run-trace 记录。无法诚实写"Agent-selected experiment"除非补记录。
2. **因果验证多停在 D1（profile/log）**：许多"根因"只有 profile 占比高，缺 intervention+revert+control 闭环（MoE P001、config admission 是少数例外）。
3. **统计强度不均**：早期实验多为单次或 n=3；只有 v41/v44/v45/v42/v43/v48 做了重复 + t 检验。
4. **baseline 强度前后不一致**：早期用 default 启发式/cudagraph-off 弱 baseline，后期才用框架实际 fallback。跨实验直接比较不成立。
5. **workload 覆盖偏窄**：主力是 Qwen3-30B-A3B、LFM2.5-8B-A1B；跨模型/跨硬件矩阵（Chendi 广度实验）尚未执行。

---

# 2. 仓库结构与端到端数据流

## 2.1 设计声称的流程（design docs）

```
model/hardware/config/workload
  → [Stage A] regime exploration (seed → sweep → suspicion → cluster → select)
  → benchmark (harness/run_bench.py, deterministic)
  → logs/profile (server.log mining, NCU, nsys, pytorch profile)
  → problem package (frozen: target/neighbors/controls/hypothesis/acceptance)
  → [Stage B] diagnosis + intervention (config-agent single-knob; 其他 solver 仅文档)
  → verification (target + neighbors + controls, decision.json)
  → report (solution.md / rejection.md)
```

对应文件/脚本/skill/数据产物（来自 infra 审计）：

| 阶段 | 文件 / 脚本 / skill | 数据产物 |
|---|---|---|
| Input | `bench-specs/*.yaml`、`configs/*.yaml`、`regimes/*.yaml`；`harness/spec.py` | resolved config + `spec_hash` |
| Regime 探索 | `scripts/run_regime_suite.py`、`generate_seed_suite.py`、`score_suspicion.py`、`cluster_regimes.py`、`select_cases_for_stage2.py`；`stages/problem-setter/**` | seed suite、suspicion 分、regime 聚类 |
| Benchmark | `harness/run_bench.py`+`lifecycle.py`+`executor.py`+`quality.py`+`output.py`；skill `e2e-bench-runner`、`regime-bench-harness` | `summary.json`、`server.log`、`bench_summary.json`、`per_run/*` |
| Logs/Profile | `harness/env_snapshot.py`；skill `server-log-mining`、`nsys-capture`、`nsys-timeline-sql`、`ncu-microarch`、`pytorch-profiling`、`profile-summary-unified` | `server_features.json`、`ncu_raw.csv`、`timeline_summary.json`、`profile_unified.json` |
| Problem package | `docs/problem-package/schema.md`；Stage A 产出 | `experiments/problems{,_moe}/PNNN/*` |
| Diagnosis/Intervention | `scripts/solver/config_agent.py`；skill `failure-classification`、`suspicion-scoring`、`cross-regime-anomaly`、`boundary-expansion`、`noise-aware-scoring` | `classification.json`、`attempts/*/candidate_config.yaml` |
| Verification | `config_agent.py:194-215`（target+neighbors+controls） | `attempts/*/verification/*`、`decision.json` |
| Report | `config_agent.py:300-370` | `solution.md` / `rejection.md` |

## 2.2 设计 vs 实际实现的差距（infra 审计确认）

- **Stage B 只有 config-agent 实现**；scheduler/kernel/workload-shape solver **仅文档**（`stages/problem-solver/README.md` 提及，代码不存在）。
- **harness 文档声称支持 vLLM（v1.1）**，代码仍是 v1 限制（`harness/spec.py:49-50`、`run_bench.py:136-143`）。
- **`regime-sweep-runner` skill 已被 `regime-bench-harness` 取代**（deprecated / 重叠）。
- **"LLM agent"仅存在于 Stage A triage 文档**（`policies/llm_agent.md`），且它只决策、不亲手算指标；**无落盘的 LLM 决策 run-trace**。
- 绝大多数论文级实验（v4–v50）是**独立一次性脚本**，不走 Stage A/B problem-package 主链路——即"证据"与"框架主链路"部分脱节。

---

# 3. Input 与问题表示清单

| Input 类别 | 路径 | 格式 | 谁生成 | 谁消费 | 完整性 | 可用于未来 controller |
|---|---|---|---|---|---|---|
| model inputs | `configs/*.yaml`（model-path）；`/data/hf/*` | yaml | 人工 | harness/server | 完整 | ✅（需抽象成 typed field） |
| hardware inputs | `harness/env_snapshot.py` 探测（nvidia-smi/nvcc/git） | json | 确定性工具 | summary.json | 完整 | ✅ |
| software/env | `environment.json`（v48）、各 doc 头部、`harness/env_snapshot.py` | json/md | 工具+人工 | 报告 | 部分（早期实验 commit 常为 null） | ⚠️ 需统一 schema |
| server configs | `configs/base.yaml`、`configs/moe_qwen3_30b.yaml`、`sglang_cookbook_deployment_baselines.{json,md}` | yaml/json | 人工/cookbook | harness | 完整 | ✅ canonical input |
| workload/regime | `regimes/*.yaml`、`regime_scout/candidates/seed_*.yaml`、problem package `workload.yaml` | yaml | 人工/seed 生成 | harness/bench | 完整 | ✅ canonical input |
| benchmark spec | `bench-specs/*.yaml`；schema `harness/spec.py` | yaml | 人工 | run_bench | 完整 | ✅（typed，含 hash） |
| problem package | `experiments/problems{,_moe}/PNNN/` | 目录+json/yaml/md | Stage A | Stage B/人 | 完整（P001 两例） | ✅ = incident schema 雏形 |
| acceptance criteria | `PNNN/acceptance_criteria.json` | json | Stage A | config_agent | 完整 | ✅ |
| neighbors/controls | `PNNN/neighbors/*`、`controls/*` | yaml+json | Stage A/boundary-expansion | verification | 完整 | ✅ 关键（反 false-accept） |
| profile/log inputs | `results/*/**/{ncu_raw.csv,timeline_summary.json,server.log,*.trace.json.gz}` | csv/json/log | profiling skill | 分析脚本/人 | 部分（大文件 gitignore） | ⚠️ 需索引 |
| patch/PR inputs | `patches/*.patch`、`patches/l2norm_*.py`、`scripts/*_patch.py` | diff/py | 人工移植上游 PR | server/microbench | 完整 | ✅（intervention 动作） |

---

# 4. 方法与工作流清单

| 方法 | 目的 | 输入 | 输出 | 入口 | 自动化 | Agent 决策? | 成熟度 | 用于哪些实验 |
|---|---|---|---|---|---|---|---|---|
| Stage A problem-setter（规则） | 发现服务差的 regime | config+seed | problem package | `scripts/run_regime_suite.py`+`select_cases_for_stage2.py` | 是 | 规则（非LLM） | 已用 | P001（dense/MoE） |
| Stage A LLM triage | 决定扩哪个 workload/停 | regime 结果 | 下一步动作 | `stages/problem-setter/policies/llm_agent.md` | 部分 | **LLM** | 文档+部分 | 未见落盘 run-trace |
| Stage B config-agent（规则） | 单-knob 修复 | problem package | attempts+solution.md | `scripts/solver/config_agent.py` | 是 | 规则（非LLM） | 已用 | MoE P001（+92.6%） |
| regime sweep | 扫 workload 网格 | workload dir | per-regime metrics | `scripts/run_regime_suite.py` | 是 | 否 | 已用 | v7/v8/v10 |
| boundary expansion | 生成 neighbor | target workload | neighbor yaml | skill `boundary-expansion` | 是 | 规则 | 已用 | P001 neighbors |
| failure classification | 分类失败 | metrics+features | class 标签 | skill `failure-classification` | 是 | 规则 | 已用 | Stage A |
| suspicion / noise-aware scoring | 排序可疑 regime + 噪声校准 | 多 run metrics | 分数 | skill `suspicion-scoring`/`noise-aware-scoring` | 是 | 规则 | 已用 | Stage A |
| server-log mining | server.log→结构化特征 | server.log | `server_features.json` | skill `server-log-mining` | 是 | 规则 | 已用 | P001, v-系列 |
| pytorch profiling | decode/prefill 组成 | 运行中 server | trace→组成% | skill `pytorch-profiling`；`run_v33/v44/v45` | 是 | 否 | 已用 | v33/v44/v45(组成) |
| NCU microarch | 屋顶线/占用/带宽 | bench_one_batch | ncu_raw.csv | skill `ncu-microarch`；`run_v6/v9/v19b/v50` | 是 | 否 | 已用 | v6/v9/v19b/v50 |
| Nsight Systems | 时间线/server idle | 运行中 server | timeline+idle% | skill `nsys-capture/-timeline-sql`；`run_v9d` | 是 | 否 | 已用 | v9d(idle 81-86%) |
| roofline | 理论上界 | ncu 数据 | 上界× | `compute_v10_roofline.py`；v50 | 是 | 否 | 已用 | v9/v10/v19b/v50 |
| kernel autotuning | 补 fused_moe config | model shape | tuned JSON | `benchmark/kernels/fused_moe_triton/tuning_*` | 是 | 否 | 已用 | v23/v44/v46(retune) |
| server config tuning (Optuna) | 服务参数搜索 | template spec | study.db+best | `harness/autotune*.py`；`run_v48` | 是 | 否（TPE） | 已用 | 6/25,6/30,7/02,v48 |
| speculative decoding | 算法层加速 | model+draft | TPOT | `run_v11a1*` | 是 | 否 | 已用 | v11(ngram +23%) |
| kernel patching | 填 sglang 空缺 | sglang 源码 | patch+microbench | `custom_moe_patch.py`、`qwen15_gate_patch.py` | 半 | 否 | 已用 | v27-v41 |
| PR reproduction | 复现上游 PR e2e | PR diff | A/B | `run_v46-v49_*` | 半 | 否（上游） | 已用 | v46-v49 |
| repeated/interleaved A/B | 信号 vs 噪声 | 两 arm | median+t | `run_v41/v44/v45/v48` | 是 | 否 | 已用 | v41/v44/v45/v48 |
| target/neighbor/control verify | 减少 false-accept | 候选 config | decision.json | `config_agent.py` | 是 | 规则 | 已用 | P001 |
| stop/plateau 判断 | 及时停止 | 收敛曲线 | plateau_stats | `run_v48_plots.py` | 是 | 否 | 已用 | v48 |

---

# 5. Tools、scripts 与 skills 总账

> 完整机器可读版见 `tool_inventory.yaml`。此处给核心表 + 整理建议。

| Tool/Skill | 路径 | 功能 | 输入 | 输出 | 副作用 | 当前调用方 | 可 typed tool | 状态 |
|---|---|---|---|---|---|---|---|---|
| run_bench harness | `harness/run_bench.py` | 确定性 e2e bench | bench-spec | summary.json | 起/杀 server | autotune/用户 | ✅ | implemented_and_used |
| spec | `harness/spec.py` | 校验+hash | spec/config/regime | resolved+hash | 读文件 | run_bench | ✅ | implemented_and_used |
| lifecycle | `harness/lifecycle.py` | server 生命周期 | config | pid/log | 起/杀 server | run_bench | ✅ | implemented_and_used |
| quality gate | `harness/quality.py` | 可靠性闸门 | per_run | gate dict | 无 | run_bench | ✅ | implemented_and_used |
| env_snapshot | `harness/env_snapshot.py` | 环境快照 | gpu/env | env dict | shell 探测 | run_bench | ✅ | implemented_and_used |
| mfu | `harness/mfu.py` | MFU/MBU | summary+hw | 增强 summary | 可改文件 | run_bench | ✅ | partially_implemented |
| config_agent | `scripts/solver/config_agent.py` | 单-knob solver（规则） | problem package | attempts+solution | 起/杀 server | 用户 | ✅ | implemented_and_used |
| autotune v1/v2/v3 | `harness/autotune*.py` | Optuna 调参 | template spec | study.db+best | 起/杀 server | 实验者 | ✅ | implemented_and_used |
| e2e-bench-runner | `.github/skills/e2e-bench-runner/impl/run_bench.py` | 多 regime bench（非流式） | url+regimes | bench_summary.json | 发请求 | executor | ✅ | implemented_and_used |
| server-log-mining | `.github/skills/server-log-mining/**` | log→特征 | server.log | features json | 无 | Stage A | ✅ | implemented_and_used |
| ncu-microarch / nsys-* / pytorch-profiling / profile-summary-unified | `.github/skills/*` | profiling | 运行中 server | csv/json | profiler | v-脚本 | ✅ | implemented_and_used |
| boundary-expansion / failure-classification / suspicion / noise-aware / cross-regime-anomaly | `.github/skills/*` | Stage A 诊断（规则） | metrics/features | 分数/标签/neighbor | 无 | Stage A | ✅ | implemented_and_used |
| regime-sweep-runner | `.github/skills/regime-sweep-runner/**` | 旧 sweep | — | — | — | — | — | **deprecated/重叠** |
| handoff-prompt-template | `.github/skills/handoff-prompt-template/**` | prompt 模板 | — | — | — | — | — | documentation_only |
| run_vNN_*.py 家族 | `scripts/run_v4_*…run_v50_*` | 一次性论文实验 | 各异 | 各异 | 起 server/profile | 人工 | 部分 | 多为 one-off |
| analyze/build/compute/plot_v*.py | `scripts/*` | 分析/画图 | raw | 表/图 | 无 | 人工 | 部分 | one-off/半复用 |
| archive/* | `scripts/archive/*` | 历史 Stage1 原型 | — | — | — | — | — | deprecated |

**整理建议（非破坏性）**：
- **canonical 可复用工具**（应封装成 typed tool adapter）：`harness/{run_bench,spec,output,quality,env_snapshot,mfu}.py`、`config_agent.py`、全部 `.github/skills/*`。
- **重复/一次性**：`run_v*.py`、`analyze_v*/build_v*/compute_v*/plot_v*`——保留为 historical，但抽取公共能力（起 server A/B、bench_serving 解析、median+t 检验、median 冷启动过滤）为共享库。
- **明确废弃**：`regime-sweep-runner` skill、`scripts/archive/*`。

---

# 6. 全部实验与数据目录

> 逐条 evidence record 见 `evidence_catalog.jsonl`。此处按时间+研究问题给叙事总账，并标注**证据范围 / 强度 / 正负 / 是否端到端 / 是否重复 / 是否适合论文**。

## 6.1 早期（2026-06）— baseline / autotuning / profiling 奠基

| exp | 研究问题 | baseline→candidate | 结果 | 范围/强度 | 论文用途 |
|---|---|---|---|---|---|
| **06-09 cutlass_investigation** | CUTLASS vs Triton e2e 为何差距小 | vLLM Triton vs CUTLASS（3 regime） | CUTLASS kernel 快（31.6% vs 46.8% self CUDA）但 e2e 被非-MoE/routing 掩盖 → **measurement-artifact** | kernel-e2e+profile / E2 | ✅ "isolated→e2e 不迁移"早例 |
| **06-09 sglang_triton_sweep** | 4-regime kernel 画像 | 默认 Triton 扫 4 regime（非 A/B） | 同 kernel decode memory-bound、prefill compute-lean | profile / E2 | ✅ 瓶颈迁移基础 |
| **06-11 harness-v1** | flashinfer_cutlass allowlist patch | triton-bf16 vs cutlass-bf16 patched（4 regime,3run） | cutlass +数×（0.10→0.83…2.94→13.86）；fp8-triton 更慢 | one-batch-e2e / E2 | ⚠️ **baseline 后被 6/25 证伪为 cudagraph-off 假象** |
| **06-25 autotuning** ★ | autotuning 天花板 | 6/11(cg off) vs today default(cg on) vs Optuna best | **"5–9×"是 baseline 伪影；对真实 default 仅 0.95–1.05×（噪声内）** | config-only / **E3** | ★★★ **最强"错误 baseline→虚假优化"案例** |
| **06-30 lfm2.5** | 条件搜索空间 autotune | true-default vs cookbook vs Optuna-v2 | 条件搜索初期**负**；default 已好；手动 triton swap 才回到 23.5 | config-only / E2 | ✅ "搜索偏置/负结果" |
| **07-02 lfm2.5_v3** | 修 v2 + MFU + 长上下文 | baseline vs v3 Optuna best（warm-start,8 regime） | 长上下文 chunk +36~121%；但 **warm-start 有偏**（后由 v48 干净替代） | config-only / E2 | ⚠️ 有偏，v48 替代 |
| 07-07 gpu_profiled / v4_decode_sweep | config×regime 画像 | cookbook vs v3 trials | 主要采 HW trace，非决定性 A/B | profile/server-e2e / E1 | 背景 |
| 07-08 v5_smoke/v5_ncu | LFM2.5 冒烟+kineto | cookbook vs big_batch | 冒烟通过；kineto top kernels | server-e2e/profile / E1 | 背景 |
| **07-08 v5b_ncu** | transformers 路径 NCU | — | **崩溃/无效**（transformers 无法代表 sglang kernel） | 无效 / E0 | ⚠️ 反面教材（工具误用） |
| **07-08 v6_ncu** ★ | 真 sglang kernel NCU | cookbook（batch 相关） | decode DRAM~65%、SM~9-10% → **memory-bound**，纠正 v5b | kernel-e2e profile / E2 | ✅ decode memory-bound 铁证 |

## 6.2 中期（2026-07-09 ~ 16）— 瓶颈定位 / 算法层 / MoE 内部

| exp | 研究问题 | 结果 | 范围/强度 | 论文用途 |
|---|---|---|---|---|
| **v7_agentic** | 真实 agent 负载形状 | toolagent in:out~13:1、decode 占 64-80%；shared_prefix TTFT 被排队主导 | server-e2e/trace / E2 | ✅ 真实负载画像 |
| **v7_config_sweep** ★ | 合成调优 config 是否迁移 | **合成"prefill 赢家"(flashinfer_cutlass+fcfs) 在真实负载最差** | server-e2e/trace / E2 | ★ "config 迁移失败" |
| **v8_tuning** | 真实负载最优 knob | `max-running-requests` 主导；**修正 v7 的 chunked8192** | server-e2e/trace / E2 | ✅（含 v7↔v8 冲突） |
| **v9_ncu_realworkload** | 调优后是否到硬件顶 | 占用仅 12-25%，roofline headroom 2.2-2.4× | profile / E2 | ✅ 硬件天花板 |
| **v9b_stalls / v9c_split** | 时间去哪 | No-Eligible 50-90%；prefill:decode~42:58 | profile+one-batch / E2 | ✅ kernel idle vs server idle 分离 |
| **v9d_nsys** ★ | server idle 量化 | **GPU busy 仅 14-19%，server idle 81-86%** | server-e2e/trace / E2 | ★ serving 层大发现 |
| **v10_load_sweep** | 加载能回收多少 idle | load 8→256 吞吐 3.3-4.4×（伤 TPOT）；roofline 1.87-2.37× | server-e2e+roofline / E2 | ✅ |
| **v11b2_multistream** | 多流回收 idle | util 13%→32%、吞吐 1087→8062 tok/s | server-e2e / E2 | ✅ |
| **v11a2_backend** | 换 backend 降 TPOT | fa3 8.71ms vs triton 10.27ms；flashinfer JIT 失败 → **fa3 已最优** | kernel-e2e / E2 | ✅ 负结果 |
| **v11a1_spec** | EAGLE3 spec | **更慢**（TPOT 52→186ms，accept 1.28）→ draft 不匹配 | server-e2e / E2 | ✅ 负结果 |
| **v11a1b_ngram** ★ | n-gram spec | **TPOT 18.64→14.27ms（+23%），accept 2.08** | server-e2e / E2 | ✅ 算法层正杠杆 |
| **v12_ncu_spec** ★ | spec 是否降 SM idle | No-Eligible 77.5%→78.0%**不变** → 机制是"少 forward pass"非"填 SM" | profile / E2 | ★ **机制纠正（v11↔v12 冲突）** |
| v13_router / v16_router_dist | router 聚类空间 | Gini 0.646；top25% expert 得 73%；均值 25.5/128 expert 零选 | profile/model-internal / E2 | ⚠️ 模型内部，非 sglang e2e |
| v14/v14b consolidation | 专家合并省传输 | 大 batch 仅省 3.8-19.5%；小 batch 22.4% | 仿真/model-internal / E1 | ⚠️ 仿真 |
| v15_ppl | 减 expert 的 PPL 代价 | top6 +5.5%、top5 +7.6%、top4 +23% | model-internal / E2 | ⚠️ 精度代理 |
| **v17_gsm8k_topk** ★ | top-k 真实精度 | acc 83.5→80(top5)→75(top4)；**"speedup"因输出变长而误导** | model-internal / E2 | ★ "误导性 speedup" |
| v18_dynamic_topk | 自适应 top-k | 增益小/混合 | model-internal / E1 | 弱 |
| v18_gflops | GFLOPS/MFU 估计 | 仅原始文件，**报告未定位** | — / E1 | 缺 |
| **v19_wall_sweep** | decode 占墙钟 | decode 88-96%（conc 1-64） | server-e2e / E2 | ✅ |
| **v19b_ncu_decode** | decode kernel 分解 | FlashAttn+fused_moe~65%；fused_moe L2 hit 12.2%→41.5%(batch 32→128)；TBT roofline 1.8-1.9× | profile / E2 | ✅ |
| v21_k_vs_length | 精度 vs 固定 k | k6/k8 安全区；k10/k12 OOD | model-internal / E1 | ⚠️ OOD caveat |
| v22_teacher_forced | teacher-forced 逐题 | 仅 raw，**无报告** | model-internal / E1 | 缺 |

## 6.3 后期（2026-07-19 ~ 22）— kernel/config e2e 现实检验 + PR 复现（本 session 主线）

| exp | 研究问题 | 结果 | 范围/强度 | 论文用途 |
|---|---|---|---|---|
| **v23 config_evidence** | tuned vs default 启发式（kernel µs） | decode +1~13%、prefill +35~54%（Qwen）/+47~67%（DeepSeek）；ours vs fallback 仅 +0.6% | microbench / E2 | ✅ config 价值来源 |
| **v27-v32 custom MoE** | M=1 MoE kernel | 隔离 1.23× 且更准 | microbench / E2 | ✅ |
| **v33 decode audit** | decode 组成 | MoE 41%+dense 32%+attn 16%=89% memory-bound | profile / E2 | ✅ |
| **v39 ctxscan** ★ | 线性 vs 全注意力 scaling | LFM +24% vs Qwen +57%（bs32,512→8192）；Qwen 16k OOM | one-batch-e2e / E2 | ★ 架构杠杆 |
| **v41 noise** ★ | custom kernel e2e 真伪 | **b1 +1.17%（n=15,|t|=6.51 真信号）；b2/b4 真回归** | one-batch-e2e / **E3** | ★★ micro→e2e 不迁移+噪声方法学 |
| **v42 kernel_e2e** | default vs tuned e2e | prefill +34~43%（M≥2048）；decode ≈0 | one-batch-e2e / E3 | ✅ |
| **v43 server_e2e** ★ | 全 regime+agent server A/B | agent E2E +17.5%、prefill +23~25%、decode ≈0 | server-e2e/trace / E3 | ★ regime 依赖 |
| **v44 retune_e2e_ab** | fallback vs ours-retune（bench_one_batch） | **≈0**（唯二小回归） | one-batch-e2e / E3 | ✅ 负结果 |
| **v45 server_ours_vs_fallback** ★ | 同上 server+agent | **全 regime ≈0**（±2%内）；三层测量一致 | server-e2e/trace / E3 | ★ "重 tune 已覆盖 shape=无用功" |
| **v46 l2norm microbench** ★ | PR#31558 重编译 | baseline 编译 10 kernel，patched **0** | microbench / E2 | ✅（上游 PR） |
| **v47 pr31558 server** ★ | #31558 冷缓存 e2e | 首轮 8 分辨率 TTFT **4.005→3.454s(−13.7%)，t=20.9** | server-e2e cold / **E3** | ★ 上游 PR e2e 正结果 |
| **v48 dsv4 pr29007** ★ | #29007 symm-mem allreduce | c1 TPOT 6.92→6.33ms(+9.2%)、E2E +10.6%（TP8，持续 c8/c16） | server-e2e / E3 | ★ 上游 PR，多卡 |
| **v49 pr31438 mm_preproc** ★ | #31438 并行预处理 | 吞吐 8.81→10.09(+14.5%)、**bit-exact**；4 worker 反而差 | server-e2e / E3 | ★ 上游 PR，bit-exact |
| **v50 ncu_roofline** | fused_moe roofline | decode DRAM 87.9-89.8%、prefill compute 64.5-67.4% | profile / E2 | ✅ 佐证瓶颈迁移 |
| **v48 lfm25_plateau_100** ★ | 无 warm-start autotuning plateau | 第 7 config 达最优区；末 20 提升 0%；best +0.4%（CI 重叠） | server-e2e / E3 | ★ plateau/stopping |

---

# 7. 诊断与因果链总账

> 逐条见 `diagnosis_catalog.yaml`。诊断强度 D0（仅假设）/ D1（profile/log/代码）/ D2（有 intervention）/ D3（intervention+revert/control）。

| Diag ID | 症状 | 根因层 | 机制 | 支持证据 | 反证 | Intervention | 验证状态 | 等级 | 论文用途 |
|---|---|---|---|---|---|---|---|---|---|
| D01 admission-capacity mismatch | R_scheduler_tail ttft_p95 2282ms | scheduler/config | max-running-requests 太低→排队 | P001 classification+3 attempts | neighbors 小改、controls 无回归 | 提 cap→64/96/128 | **已执行+neighbors/controls** | **D2-D3** | ★★★ incident 范例 |
| D02 wrong-baseline false speedup | autotuning "5–9×" | measurement | baseline 用了 cudagraph-off | 6/25 三 baseline 对照表 | 对真实 default 仅 0.95-1.05× | 换真实 default baseline | **已执行（重测）** | **D2** | ★★★ 反证核心 |
| D03 micro→e2e non-transfer | 隔离 1.23× 但 e2e 微弱 | gpu_kernel | MoE 占 decode 仅 41%+cudagraph 隐藏 launch | v41 n=15 t 检验 | b≥2 真回归 | 插回 sglang decode | **已执行+t 检验** | **D2** | ★★ |
| D04 prefill/decode bottleneck shift | "瓶颈是 decode" 争议 | workload | in:out 比 + 并发决定 | v43 server 拆分、v50 roofline | 长上下文 prefill 55-89% | 按 regime 拆分测量 | **已执行（多 regime）** | **D2** | ★★ |
| D05 retune-covered-shape no-op | 重 tune 期望提升 | configuration | fallback config 已近最优 | v44/v45 三层 ≈0 | 唯二小回归 | ours→放进版本目录 A/B | **已执行+t 检验** | **D2** | ★ |
| D06 serving idle dominates | 单流 GPU busy 仅 14-19% | scheduler/workload | 请求间隔+排队，SM 空转 | v9d nsys idle 81-86%、v11b2 多流回收 | 多流 util 13→32% | 加并发/多流 | **已执行（load sweep）** | **D2** | ★ |
| D07 spec mechanism misattribution | "spec 填 SM idle" | algorithm | 实为"少 forward pass" | v12 No-Eligible 不变 | v11 措辞被纠正 | NCU 对照 | **已执行（profile 对照）** | **D2** | ★ 机制纠正 |
| D08 synthetic-config non-transfer | 合成最优在真实负载差 | workload/config | regime 分布不同 | v7_config_sweep | — | 真实 trace 重测 | **已执行** | **D2** | ★ |
| D09 l2norm token-count recompile | 每新 token 数编译新 kernel | runtime_jit_cache | `T: tl.constexpr` 触发特化 | v46 编译 10→0；v47 冷缓存 TTFT −13.7% | 稳态/固定形状控制 ≈0 | do_not_specialize patch | **已执行+revert-like control** | **D3** | ★★（上游 PR） |
| D10 mm serial preprocessing | VLM 突发吞吐受限 | cpu_preprocessing | tokenizer 事件循环串行 | v49 +14.5% bit-exact | 4 worker 反而差 | 并行 worker pool patch | **已执行+bit-exact+control** | **D3** | ★★（上游 PR） |
| D11 MoE TP allreduce alloc | dsv4 TPOT 偏高 | communication | MoE 输出分配非对称内存 | v48 TPOT +9.2% | — | symm-mem 分配 patch | **已执行** | **D2** | ★（上游 PR） |
| D12 transformers-path NCU invalid | LFM2.5 NCU 崩溃 | measurement | transformers 无法代表 sglang kernel | v5b 崩溃、v6 纠正 | v6 真 sglang 有效 | 换 bench_one_batch+NCU | **已执行（纠正）** | **D2** | ✅ 工具误用反面 |

> **关键纪律**：不能因 profile 占比高就判为端到端根因。D03/D04/D07 都是"profile 占比高 → intervention 后 e2e 未按朴素预期变化"的例子——这正是论文的反证价值。

---

# 8. Agent 贡献账本

> 核心诚实判断：**本仓库的"Agent"绝大多数是规则化 pipeline 或人工/上游 PR，不是 LLM 自主发现。** config_agent.py 是固定循环（`scripts/solver/config_agent.py:93-165`）。

| 结果 | Hypothesis | Experiment Selection | Tool Execution | Candidate | Verification | Final Decision | 合理表述 |
|---|---|---|---|---|---|---|---|
| MoE P001 +92.6% | 规则分类(concurrency_capped) | 规则 Stage A | 确定性 harness | 规则 config_agent 单-knob | target+neighbors+controls | config_agent decision.json | **Rule-discovered + Deterministically-verified** |
| 6/25 wrong-baseline 反证 | 人工/会议质疑 | 人工 | 确定性 harness | 人工换 baseline | 重测 | 人工 | **Human-designed**（工具执行） |
| v41 micro→e2e | 人工假设 | 人工 | custom_moe_patch+bench | 人工 kernel | n=15+t 检验 | 人工 | **Human-designed, Deterministically-verified** |
| v43 regime shift | 人工/Dey 提问 | 人工 | server+bench_serving | — | 多 regime | 人工 | **Human-designed** |
| v45 retune no-op | 人工 | 人工 | autotune+server A/B | 人工 retune | t 检验 | 人工 | **Human-designed** |
| v47/v48/v49 PR 复现 | **上游 PR** | 人工选 PR | 移植+server A/B | **上游 patch** | bit-exact/control | 人工 | **Upstream-reproduced** |
| v48 plateau | 人工/Chendi 规格 | 人工定 search space | Optuna TPE | TPE 采样 | 交错 x5+t | plateau 分析 | **Deterministically-verified**（TPE 非 LLM） |
| Stage A LLM triage | **LLM** | **LLM** | 确定性 skill | — | — | — | **Agent-selected (仅此项，且无落盘 trace)** |

**当前不能诚实称为 "Agent-discovered" 的**：几乎全部性能结论。可诚实称 **Rule-discovered**（P001）、**Deterministically-verified**（v41/v48）、**Upstream-reproduced**（v47-v49）、**Human-designed**（其余）。**唯一** LLM 决策点是 Stage A triage，但缺 run-trace 支撑论文级 attribution。

---

# 9. Paper Claim–Evidence Matrix

> 逐条见 `paper_claim_matrix.yaml`。此处给核心矩阵。

| Claim | 直接证据 | 强度 | Caveat | 缺失实验 | 推荐图表 | 可否写入 |
|---|---|---|---|---|---|---|
| C1 瓶颈随 prefill/decode 和 regime 迁移 | v43/v50/v33/v19；`prefill_vs_decode_bottleneck_report.md` | E3(server)+profile | 单模型为主 | 跨模型再验 | 每 regime prefill/decode 墙钟堆叠图 | **可** |
| C2 microbench 提升不必然转化 server | v41(t 检验)、cutlass 06-09、v42 decode≈0 | E3 | 特定 kernel/规模 | 更多 kernel 类型 | isolated× vs e2e% 散点 | **可** |
| C3 错误 baseline 制造虚假优化 | 6/25(5-9×→1.0×)、06-11↔06-25 | E3 | 需讲清 cudagraph | — | 三 baseline 对照柱状 | **可（强）** |
| C4 cold-start/cache/样本/噪声造成误诊 | v47(冷缓存)、v44(n=3 −8.84%→n=8 +0.91%)、v41 | E3 | 需清 TRITON_CACHE | — | 冷/热 + n 曲线 | **可（强）** |
| C5 结构化 workflow 帮选下一步证据 | Stage A/B pipeline、P001 | E2（规则，非LLM） | **非 LLM，须写"structured/rule-based"** | LLM triage run-trace | pipeline 图 | **可但须限定措辞** |
| C6 target/neighbor/control 减少 false-accept | P001 verification | E2 | 单例 | 更多 problem | verification 表 | **可** |
| C7 performance search 有 plateau，停止是对的 | v48 plateau_stats | E3 | 单 regime/模型 | 跨 regime plateau | 收敛+plateau 图 | **可** |
| C8 incident 应路由到不同层次 | D01-D12 覆盖 8 个根因层 | 混合 D1-D3 | 部分仅 D1 | 补 intervention | 根因层 × incident 矩阵 | **可** |
| C9 Agent 实际贡献边界 | §8 账本 | — | **不能称 Agent-discovered** | LLM 决策记录 | 贡献账本表 | **可（诚实版）** |

**最强 5–8 条可写 claim（最诚实措辞）**：
1. **"Wrong-baseline artifacts silently manufacture large speedups"**（6/25：cudagraph-off baseline 把真实 1.0× 变成"5–9×"）。Reviewer 质疑点：是否 cherry-pick → 用三 baseline 全 4 regime 对照表回应。
2. **"Isolated-kernel speedups do not transfer to serving"**（v41：1.23×→+1.17%，t 检验；cutlass 06-09）。质疑点：kernel 是否代表性 → 多 kernel + roofline。
3. **"Cold-start / JIT-cache confounds require cache-controlled A/B"**（v47 冷缓存 −13.7% vs 稳态 ≈0；v44 n=3→n=8 翻盘）。质疑点：是否人为制造冷缓存 → 说明真实首次部署即冷。
4. **"The bottleneck moves across regimes; there is no single global bottleneck"**（v43/v50）。质疑点：模型单一 → 补跨模型。
5. **"Serving-knob autotuning plateaus quickly without warm start; stopping is a valid decision"**（v48：第 7 config、末 20 提升 0%）。质疑点：search space 太小 → 说明 4 knob 是标准 serving knob。
6. **"Target+neighbor+control verification catches false-accept"**（P001）。质疑点：单例 → 补 problem。
7. **"Re-tuning an already-covered config shape is a no-op end-to-end"**（v44/v45 三层一致）。质疑点：是否 tune 得不好 → 展示 tune 覆盖 18 桶。
8. **"Upstream PR gains are regime/condition-specific"**（v47-v49：冷缓存/多卡/突发才显）。诚实标注 **Upstream-reproduced**。

**不能写 / 须强 caveat 的**：
- ❌ "Our Agent autonomously discovered kernel optimizations." → 无证据；custom kernel 是人工写、e2e≈0。
- ❌ "Autotuning gives 5–9× / +50%." → 前者是 baseline 伪影；+34~43% 是 vs default 启发式（非框架 fallback），且 decode≈0。
- ❌ 把 v47-v49 写成自主发现 → 是上游 PR 复现。
- ❌ 把 config_agent / Stage A 规则打分写成 "LLM Agent"。
- ⚠️ MoE 内部 topk/ppl/router（v13-v22）多为**模型内部仿真/精度代理，非 sglang 端到端**，且 v17 speedup 误导——写入须严格标注。

---

# 10. 可构建的 Incident Benchmark 候选

> 用途分类：diagnosis / intervention / false-positive-rejection / stopping-abstention / case-study。逐条隐藏文件建议见文末。

| # | Incident | 可见症状 | 隐藏 ground truth | 根因层 | valid intervention | target/neighbor/control | 需 live GPU | 标签 | 用途 | answer leakage 风险 |
|---|---|---|---|---|---|---|---|---|---|---|
| I01 | scheduler_tail ttft_p95 2282ms | 尾延迟爆炸 | max-running-requests 太低 | scheduler/config | 提 cap | ✅ 齐全（P001） | 否（可 replay 数据）| **Gold** | diagnosis+intervention | 高（solution.md 需隐藏） |
| I02 | autotuning "5–9×" | 看似巨大加速 | baseline 是 cudagraph-off | measurement | 换真实 default baseline | 三 baseline 对照 | 否 | **Gold** | **false-positive-rejection** | 高（honest_results 需隐藏） |
| I03 | custom kernel 1.23× | 隔离大加速 | e2e 仅 +1.17%、b≥2 回归 | gpu_kernel | 全 regime e2e + t 检验 | b1/b2/b4 | 否（有 raw）| **Gold** | false-positive-rejection | 中 |
| I04 | "spec 填 SM idle" | TPOT −23% + idle 假设 | 机制是少 forward pass | algorithm | NCU No-Eligible 对照 | baseline/ngram | 否 | **Silver** | diagnosis（机制纠正） | 中 |
| I05 | retune 期望提升 | 应更快 | fallback 已近最优 | configuration | ours vs fallback A/B | 三层 | 否 | **Gold** | false-positive-rejection | 中 |
| I06 | 合成最优在真实负载差 | config 应通用 | regime 分布不同 | workload | 真实 trace 重测 | 合成 vs trace | 否 | **Silver** | diagnosis | 中 |
| I07 | l2norm 每分辨率停顿 | 新分辨率首 token 慢 70ms | token-count 触发重编译 | runtime_jit_cache | do_not_specialize + 冷缓存 A/B | 冷/热、固定形状 control | 是（VLM）| **Gold** | diagnosis+intervention | 高（PR 已知）|
| I08 | VLM 突发吞吐受限 | 高并发吞吐低 | 串行预处理 | cpu_preprocessing | 并行 worker | 2w/4w、bit-exact | 是（VLM）| **Silver** | intervention | 高（PR 已知）|
| I09 | 单流 GPU 14-19% busy | GPU 看似空闲 | serving idle 非 kernel | scheduler/workload | 加并发/多流 | 1/2/4/8 流 | 是 | **Silver** | diagnosis | 中 |
| I10 | top-k 减少"提速" | tok/s 上升 | 输出变长伪影+精度降 | measurement/algorithm | 固定输出长度重测+gsm8k | k4-k8 | 否（有 raw）| **Silver** | false-positive-rejection | 中 |
| I11 | transformers-NCU 崩溃 | 无法 profile | 路径无法代表 sglang | measurement | 换 bench_one_batch+NCU | v5b/v6 | 是 | **Case-study** | tool-misuse | 低 |
| I12 | LFM2.5 serving 调参 | 想找最优 config | 存在 plateau | configuration | 100-trial + stopping | 全 config | 是 | **Gold** | **stopping/abstention** | 中（best_validated 需隐藏）|

---

# 11. 人工验证优先队列

> 完整版含 pass/fail 与命令模板见 `manual_validation_queue.yaml`。优先级 = paper value × information gain ÷ experiment cost。

| 优先级 | 诊断 | 当前证据 | 缺失 | 最小验证实验 | Target | Neighbor | Control | Revert | pass/fail | 成本 | 论文价值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1** | D02 wrong-baseline | 已有三 baseline 表 | 单独 cudagraph on/off 干净 A/B（同 config） | 复跑 default cg-on vs cg-off，n≥5 | cg-off 显著慢 | 各 regime | cg-on 稳定 | 关 cg | **cg-off/cg-on 差 >5×且 p<0.01 → pass** | 低（1 model,4 regime）| ★★★ |
| **P2** | D01 admission-capacity | P001 已有 intervention+neighbors/controls | revert 明确记录 | 复跑 cap=32(default) vs 64，加 revert | ttft_p95 恢复到 2282 | neighbors 小改 | controls 无回归 | cap 调回 32 | **cap64 −>90%、revert 复原 → pass** | 低 | ★★★ |
| **P3** | D09 l2norm recompile | v46/v47 已强 | 更多分辨率 + 明确 revert arm | 冷缓存 baseline vs patch vs revert，≥8 分辨率 | 首轮 TTFT −>10% | 固定分辨率 | 稳态两 arm 相同 | 还原 constexpr | **冷 −>10% p<0.01 且固定形状≈0 → pass** | 中（VLM,冷缓存）| ★★★ |
| **P4** | D03 micro→e2e | v41 已 t 检验 | 更多 kernel 类别（attn/dense） | 对 1-2 个别 kernel 重复 isolated+e2e | e2e < isolated | b1/b2/b4 | — | 关 patch | **isolated>10% 但 e2e<3% → pass** | 中 | ★★ |
| **P5** | D05 retune no-op | v44/v45 三层 | — | 已足够；可加 1 个新 shape | e2e ≈0 | 全 regime | fallback | 移除 ours config | **|Δ|<2% 全 regime → pass** | 中 | ★★ |
| **P6** | D07 spec mechanism | v12 profile 对照 | intervention（改 batch/并发再测 idle） | 不同并发下 No-Eligible + TPOT | TPOT 降但 idle 不降 | baseline/ngram | — | 关 spec | **TPOT<baseline 且 No-Eligible 不变 → pass** | 中 | ★ |
| **P7** | D06 serving idle | v9d/v11b2 | 干净 revert（降并发复原 idle）| 并发 1→8→1 idle 曲线 | idle 随并发降 | — | — | 降回并发 | **idle 单调随并发降且可复原 → pass** | 中 | ★ |
| **P8** | D10 mm preproc | v49 bit-exact | — | 已足；可补 revert arm | 吞吐 +>10% | 2w/4w | bit-exact | 还原串行 | **+>10% 且 bit-exact → pass** | 中（VLM）| ★★ |

**建议先做 P1+P2**（成本最低、论文价值最高、可 replay 数据或轻量 GPU）。命令模板：P2 可复用 `scripts/solver/config_agent.py --problem experiments/problems_moe/P001`；P1/P5 复用 `scripts/run_v44_e2e_config_ab.py`、`run_v45_server_ours_vs_fallback.py`；P3 复用 `scripts/run_v47_pr31558_server_ab.py`（需清 `TRITON_CACHE_DIR`）。**本审计不实际运行。**

---

# 12. 仓库整理与统一接口建议（非破坏性）

- **canonical input**：`configs/*.yaml`、`regimes/*.yaml`、`regime_scout/candidates/seed_*.yaml`、`bench-specs/*.yaml`、`sglang_cookbook_deployment_baselines.json`。
- **canonical raw data**：`results/*/**/{summary.json,bench_summary.json,*.jsonl,ncu_raw.csv,server.log}`、problem package。
- **derived summary**：`results/*/summary.json`、`*_summary.json`、`plateau_stats.json`、各 `analysis*.txt`。
- **narrative report**：`docs/YYYY-MM-DD/*.md`、`plan.md`。
- **适合 tool adapter**：`harness/*`、`config_agent.py`、`.github/skills/*`（已是 typed 契约）。
- **保留为 historical one-off**：`scripts/run_v*.py`、`analyze/build/compute/plot_v*.py`、`scripts/archive/*`。

**统一 experiment manifest 建议**（新增，不改现有）：一个 `experiments/manifest.jsonl`，每行 = {exp_id, date, model, hw, sw, workload, baseline, candidate, script, raw_paths[], summary_path, doc_path, evidence_scope, evidence_strength, conclusion}。本审计的 `evidence_catalog.jsonl` 即其雏形。

**problem package → incident schema 扩展建议**：在现有 `problem.json` 上加 `incident` 层：`{visible_symptom, hidden_ground_truth, root_cause_layer, allowed_actions[], required_evidence[], misleading_evidence[], valid_intervention, gold_label, hidden_files[]}`（§10 已给 12 个候选的字段）。

**为 controller 提供的接口对象**（建议 typed）：
- `Incident`（= 扩展 problem package）
- `Action`（起 server / bench / profile / patch / config-set，映射到现有 skill/script）
- `Observation`（summary.json / features / ncu_raw / trace）
- `Diagnosis`（root_cause_layer + strength）
- `Transition`（action→observation→diagnosis 更新）
- `Budget`（GPU-min / trial 数）
- `RunTrace`（**当前缺失，最需补**：每步 who-decided-what，用于 Agent attribution）

> 本任务**不做**大规模移动/重构，仅给计划。

---

# 13. 论文故事建议

| 故事线 | 证据匹配度 | novelty | 需新增实验 | Agent attribution 难度 | reviewer 风险 | 适合论文类型 | 优先级 |
|---|---|---|---|---|---|---|---|
| **A. Evidence-guided diagnosis & falsification** | **高**（6/25、v41、v44/v45、v47、v48、P001、v7↔v8、v11↔v12 大量成对反证） | 中-高（"反证/拒绝虚假优化"少见） | 少（补 P1/P2 revert）| **低**（可诚实写 rule/deterministic/human-in-loop）| 低 | measurement/benchmark/systems | ★★★ **推荐** |
| **B. LLM-inference optimization transfer-failure characterization** | 高（micro→e2e、config 迁移、retune no-op、上游 PR 条件性）| 中 | 中（跨模型/kernel）| 低 | 中（需说清 baseline）| systems/empirical study | ★★ |
| **C. Regime-aware optimization routing** | 中（瓶颈迁移、根因层路由 D01-D12）| 中 | 中-高（需真正 router/controller）| 中（若声称自动路由需 LLM trace）| 中 | systems | ★★ |
| **D. Agent-as-controller for perf debugging** | **低**（无 LLM 决策 run-trace，config_agent 是规则）| 高（若做出来）| **多**（要真做 controller + trace）| **高** | 高（易被质疑 attribution）| agent/systems | ★（远期，先补 RunTrace）|

**推荐**：以 **A** 为主线（证据最扎实、attribution 最诚实），**B** 作为支撑章节。**D** 是仓库的长期愿景，但当前证据不足以支撑"Agent 自主"叙事，需先补 §12 的 RunTrace 与 LLM triage 记录。

---

# 14. 未解决问题与下一步

| 类别 | 内容 |
|---|---|
| **数据缺失** | v18_gflops / v22_teacher_forced 无文字报告；v4_decode_sweep/v5 系列无叙事 doc；大 profile 文件已 gitignore（需重生成才能复核） |
| **原始结果与报告不一致（冲突记录）** | (1) 06-11 "5-9×" ↔ 06-25 "1.0×"（baseline 口径，06-25 更可信）；(2) v7 chunked8192 ↔ v8 max-running-requests（v8 修正 v7）；(3) v11 "spec 填 SM" ↔ v12 "少 forward pass"（v12 机制更准）；(4) v17 "speedup" ↔ 输出变长伪影；(5) 06-30 v2 负 ↔ 07-02 v3 正（warm-start 差异，但 v3 有偏，v48 干净替代）。**均已在 §6/§7 标注，需人工最终确认口径。** |
| **Agent attribution 缺失** | 无 LLM triage run-trace；无法诚实写 "Agent-selected experiment"；须补 RunTrace |
| **统计强度不足** | 早期 v-系列多单次/n=3；建议对进入论文的每条补 n≥5 + t 检验（v41/v44/v45/v48 已是范例） |
| **baseline 不够强** | 早期用 default 启发式/cudagraph-off；论文须统一用"框架实际 fallback / cudagraph-on"作 baseline，并明说 |
| **workload 覆盖不足** | 主力 2 模型；Chendi 广度实验（20 model × regime × config）设计稿在 `docs/2026-07-17/`，未执行 |
| **工程缺口** | Stage B 仅 config-agent；scheduler/kernel/workload solver 仅文档；harness vLLM 支持文档超前于代码 |
| **实习结束前可完成** | P1/P2 人工验证（低成本高价值）；补 evidence manifest；把 A 故事线的 5-8 图整理成稿 |
| **不建议继续投入** | 在成熟 bf16/H200 上重写核心 MoE kernel 追 e2e（v41/v44/v45 已证 ≈0）；MoE 内部 topk/ppl 追"提速"（v17 已证误导） |

---

## 附：自检结论（对应任务八）
1. 主报告主要数字均给了原始路径（§6/§7 表 + `evidence_catalog.jsonl`）✅
2. 已严格区分 micro/kernel-e2e/one-batch/server-e2e（每条标 evidence_scope）✅
3. v47-v49 明确标 **Upstream-reproduced**，未写成自主发现 ✅
4. 明确 config_agent/Stage A 打分是 **规则**，非 LLM ✅
5. 指出早期弱 baseline 并给出框架实际 fallback 口径 ✅
6. 报告了负结果（v11a2/v11a1/v44/v45/06-30）与无显著差异（6/25）✅
7. 记录了 5 组相互矛盾报告（§14）✅
8. 标注了仅 profile 支持的结论（D04/D07 等，evidence_scope=PROFILE_ONLY）✅
9. §11 给了人工可执行因果验证方案 + pass/fail ✅
10. JSONL/YAML 产物已生成（见同目录，随附解析自检）✅
11. **未删除/移动/改写任何原始实验文件** ✅
