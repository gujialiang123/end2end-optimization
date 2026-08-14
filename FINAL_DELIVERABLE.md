<!--
FINAL DELIVERABLE — top-level capstone and index.
最终交付物 —— 顶层总纲与索引。
Written 2026-08-14. Bilingual (English first, 中文 second) throughout.
-->

# FINAL DELIVERABLE — End-to-End Kernel-Specialization Study for SGLang
# 最终交付物 —— SGLang 端到端 Kernel 专化研究

**Date / 日期**: 2026-08-14
**Author / 作者**: Jialiang Gu (@gujialiang123)
**Hardware / 硬件**: 8× NVIDIA H200 (143 GB), single-GPU per run / 单卡每次运行
**Stack / 软件栈**: SGLang `0.5.12.post1 @ 17f7a1da1` · Triton 3.5.1 · bf16 · TP1
**Env / 环境**: `conda activate sglang-dev`

> **EN** — This is the single entry point to the whole project. It states the thesis, the
> evidence, the chronological arc, the upstreamed PRs, and how to reproduce; then it points to
> three exhaustive machine-readable catalogs that describe **every** document, script, and skill
> in the repository. Read this file top-to-bottom, then dive into the catalogs or the dated docs.
>
> **中文** —— 这是整个项目的唯一入口。它给出论点、证据、时间线脉络、已上游的 PR、以及如何复现；
> 然后指向三份详尽的、机器可读的编目，逐一描述仓库里**每一个**文档、脚本和 skill。
> 建议先从上到下读完本文件，再进入编目或按日期的实验记录。

---

## 0. Index of this deliverable / 本交付物的索引

| # | Section / 章节 | EN | 中文 |
|---|---|---|---|
| 1 | The thesis | What the mentors asked and what we proved | mentor 的要求与我们证明了什么 |
| 2 | Headline results | The 8 core results + 3 August extensions | 8 个核心结果 + 8 月三条扩展 |
| 3 | The flagship figure | LFM2.5 four-bar stack | LFM2.5 四层柱状图 |
| 4 | Chronological arc | June → August narrative | 六月到八月的叙事脉络 |
| 5 | Upstream PRs | External, verifiable evidence | 外部可验证证据 |
| 6 | Detailed catalogs | Every doc / script / skill | 每个文档/脚本/skill |
| 7 | Reproduce | Commands + environment | 命令与环境 |
| 8 | Open items | What a successor should do next | 接手者下一步 |
| 9 | Repo map | Top-level directory guide | 顶层目录指南 |

**Detailed catalogs (start here for file-level detail) / 详尽编目（文件级细节从这里开始）:**
- [`docs/2026-08-14/CATALOG_docs.md`](docs/2026-08-14/CATALOG_docs.md) — every doc, chronological / 每篇文档，按时间
- [`docs/2026-08-14/CATALOG_scripts.md`](docs/2026-08-14/CATALOG_scripts.md) — every script, grouped by function / 每个脚本，按功能
- [`docs/2026-08-14/CATALOG_skills.md`](docs/2026-08-14/CATALOG_skills.md) — every reusable skill / 每个可复用 skill

---

## 1. The thesis / 论点

**EN** — Two mentors framed the goal.

- **Debadeepta Dey**: *"…show that for different regimes we can genetically rewrite kernels to
  improve **beyond what the best auto-tuning config provides**."* → prove (D1) different regimes need
  different specialization, (D2) **the best autotuning config is not the ceiling**, (D3) editing the
  kernel algorithm / structure / fusion boundary keeps paying.
- **Mason Remy**: an evidence chain — (M1) categorize regimes, (M2) show config-tuning's gain **and its
  plateau**, (M3) check whether existing kernel autotuning covers **real** workload shapes, (M4) pick a
  kernel NCU shows headroom on, (M5) rewrite it or fuse it with neighboring elementwise kernels, (M6)
  show the kernel edit's **extra** gain **on top of** autotuning.

**The answer we deliver**: on LFM2.5, once serving config **and** MoE kernel config are both tuned to
their measured ceiling, a kernel rewrite still adds **+9.73 % (p = 9.5e-19)** on long-prefill and
**+6–8 %** across four regimes — and it does so **orthogonally** to the two layers below it. The same
kernel work reproduces on three more model families and holds across nine real/agentic workloads.

**中文** —— 两位 mentor 定了目标。

- **Debadeepta Dey**：*"…证明不同 regime 下我们能重写 kernel，取得**超越最佳 autotuning config**的提升。"*
  → 要证明 (D1) 不同 regime 需要不同专化，(D2) **最佳 autotuning config 不是上限**，(D3) 改 kernel
  的算法/结构/融合边界能继续带来提升。
- **Mason Remy**：一条证据链 —— (M1) 给 regime 分类，(M2) 展示 config tuning 的收益**及其 plateau**，
  (M3) 检查现有 kernel autotuning 是否覆盖**真实** workload shape，(M4) 选一个 NCU 显示有 headroom 的
  kernel，(M5) 重写它或与周围 elementwise kernel 融合，(M6) 展示 kernel 改动**在 autotuning 之上**的**额外**增益。

**我们交付的答案**：在 LFM2.5 上，当 serving config **和** MoE kernel config 都被调到实测上限之后，
kernel rewrite 在长 prefill 上仍贡献 **+9.73 %（p = 9.5e-19）**、在四个 regime 上 **+6–8 %**，而且
与下面两层**正交**。同一份 kernel 工作在另外三个模型家族上复现，并在九个真实/agentic workload 上成立。

> Full requirement-by-requirement mapping / 逐条对照：
> [`docs/2026-08-03/deliverables_vs_mentor_requirements.md`](docs/2026-08-03/deliverables_vs_mentor_requirements.md)

---

## 2. Headline results / 核心结果

**EN** — Eight core results (A–H), then three August extensions. Every row points to its evidence doc.

**中文** —— 八个核心结果（A–H），加上八月三条扩展。每行都指向证据文档。

| ID | Result / 结果 | Number / 数字 | Requirement / 扣题 | Evidence / 证据 |
|---|---|---|---|---|
| **A** | LFM2.5 serving-config autotuning has **zero** headroom / serving config 调优**零收益** | Optuna 25-trial "best" = **−6 %** vs cookbook; grid confirms | D2, M2 (ceiling) | `docs/2026-06-30/`, `docs/2026-07-22/` grid |
| **B** | LFM2.5 kernel rewrite (7 components) / kernel 重写（7 项） | **+6.2–6.6 %** on A/B/C (dirty baseline) | D1, D3, M5, M6 | `docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md` |
| **C** | LFM2.5 H200 MoE tuned config (missing upstream) / H200 缺 MoE config | **+23.34 %** long prefill (p=1.3e-10) | M3 | `docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md` (PR #32687) |
| **D** | Gemma-3 RMSNorm dispatch fix / RMSNorm 派发修复 | **+36.6 %** low-batch decode | D3, M5, M6 | `docs/2026-07-28/three_fusion_cases.md` (PR #32670) |
| **E** | Gemma-3 fused_qk_norm_rope wiring / 接线 | **+0.5–5.5 %** real increment; 97 % was another PR | methodology (honesty) | `docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md` |
| **F** | OLMo-2 bypass its own fused kernel / 绕过自融合 kernel | prefill **1.24×**, **+17.61 %**, bit-identical | M1+M4+M5 chain | same doc §7.4 (issue #33415 / PR #33416) |
| **G** | Methodology findings / 方法学结论 | sub-additivity **0.90/0.70/0.49**; backend rule non-transferable **−34 %** worst; "family-attention" predictor | bonus | `docs/2026-07-28/cross_architecture_audit.md` |
| **H** | SLO-agent loop / agent 闭环 | **5/5** historical cases reproduced | agent-in-the-loop | SLO-agent PR #9 / #30, `results/slo_agent_run/` |
| **★1** | **48-cell L1×L2×L3 factorial** (08-04) / 48 格三层全因子 | L3 **6.2–8.4 %**, orthogonal to L1/L2 on A/B/C/D | D2, M6 (clean) | `docs/2026-08-04/ablation_matrix_complete.md` |
| **★2** | **Cross-model replication** (08-04) / 跨模型复现 | Falcon-H1 SSD tile 16→64 = **+27.63 %** long prefill, token-identical; OLMo-2 audit clean | generalization | `docs/2026-08-04/pipeline_replication_olmo2_falconh1.md` |
| **★3** | **Real-workload external validity** (08-07) / 真实 workload 外部有效性 | L3 latency **−6.3…−21.8 %** across 9 workloads; throughput converts at saturation | external validity | `docs/2026-08-07/REAL_WORKLOAD_EXTERNAL_VALIDITY.md` |
| **★4** | **L2/L3/L2+L3 ablation on real workloads** (08-10) / 真实 workload 上的 L2/L3 消融 | L2 **not transferable** (ShareGPT ≈0, Mooncake TTFT −27…−48 %); load conversion x1→x4 **+9.15 %** req/s | M3 (bucket dependence) | `docs/2026-08-10/RT_L2L3_ablation.md` |

---

## 3. The flagship figure / 旗舰图：LFM2.5 four-bar stack

**EN** — This single stacked figure answers D2 / M2 / M6 directly. Same model, same serving knobs
(`mem 0.85 / lpm / cap 32 / chunk −1`), regime C (long prefill, in≈4000), counterbalanced n=16/arm.

**中文** —— 这一张柱状图直接回答 D2 / M2 / M6。同模型、同 serving 旋钮，regime C（长 prefill），双向对照 n=16/臂。

| Bar | Layer / 层 | req/s | vs Bar 2 | Meaning / 含义 |
|---|---|---:|---:|---|
| 1 | sglang bare default / 裸默认 | — | — | out-of-box |
| 2 | cookbook default / cookbook 默认 | 12.119 | 1.000× | **25-trial Optuna cannot beat this = the ceiling (A)** / Optuna 打不过 = 上限 |
| 3 | + tuned MoE kernel config / 调优 MoE config | 14.939 | 1.233× | kernel **autotuning** (C, PR #32687) |
| 4 | + kernel rewrite (7 components) / kernel 重写 | **16.392** | **1.352×** | **the argument lives here** / 论点在这一格 |

**Bar 3 → Bar 4 = +9.73 %, p = 9.5e-19.** Kernel rewrite pays **on top of** the best autotuning —
and the increment is **super-additive** here (realization 1.14), decomposed in
[`docs/2026-08-03/exp3_kernel_on_tuned_baseline.md`](docs/2026-08-03/exp3_kernel_on_tuned_baseline.md).
The 08-04 factorial then shows this is not a one-regime fluke: across A/B/C/D the kernel increment is a
near-constant 6.2–8.4 % regardless of the two layers below.

**Bar 3 → Bar 4 = +9.73 %，p = 9.5e-19。** kernel 重写在最佳 autotuning **之上**仍然有效，且此处**超可加**
（兑现率 1.14）；08-04 的全因子进一步证明这不是单 regime 的偶然：A/B/C/D 上 kernel 增量恒为 6.2–8.4 %，
不受下面两层影响。

---

## 4. Chronological arc / 时间线脉络 (June → August)

**EN** — The project ran in five phases. Each links to the dated docs; the full per-doc catalog is in
[`CATALOG_docs.md`](docs/2026-08-14/CATALOG_docs.md).

**中文** —— 项目分五个阶段推进。每段链接到按日期的文档；逐篇编目见 `CATALOG_docs.md`。

### Phase 1 — Regime discovery & the agent system (Jun 01 → Jun 25) / 阶段一：regime 发现与 agent 系统
**EN** — Built the Stage-A/Stage-B "problem-setter / problem-solver" harness; benchmarked Qwen3-0.6B and
Qwen3-30B-A3B across regimes; mined kernel inventories; mapped MoE backend dispatch in vLLM vs sglang.
Found Qwen's default→tuned gap is **huge** (up to 8.86×) — which later had to be reconciled against LFM.
**中文** —— 搭建 Stage-A/Stage-B「出题人/做题人」框架；对 Qwen3-0.6B 和 Qwen3-30B-A3B 跨 regime 基准；
盘点 kernel；梳理 vLLM 与 sglang 的 MoE backend 派发。发现 Qwen 的 default→tuned 空档**极大**（达 8.86×）——
后来必须与 LFM 的口径调和。
`docs/2026-06-01/` … `docs/2026-06-25/`

### Phase 2 — The autotuning-ceiling puzzle (Jun 25 → Jun 30) / 阶段二：autotuning 上限之谜
**EN** — Qwen said "autotuning is everything; don't rewrite kernels." LFM2.5 said the opposite:
config autotuning yields **zero**. The resolution (apples vs oranges: Qwen number is default→tuned, LFM is
increment-on-tuned) reframed LFM2.5 as the main line.
**中文** —— Qwen 的结论是「autotuning 就是一切，别改 kernel」，LFM2.5 却相反：config 调优**零收益**。
调和之道（苹果比橘子：Qwen 是 default→tuned，LFM 是在已调优基线上的增量）把 LFM2.5 定为主线。
`docs/2026-06-25/autotuning_ceiling_report.md`, `docs/2026-06-30/`

### Phase 3 — Kernel rewrites & upstream PRs (Jul 08 → Jul 28) / 阶段三：kernel 重写与上游 PR
**EN** — NCU/nsys profiling → seven LFM2.5 kernel components (two hand-written Triton kernels + five
call-site fusions) → +6 %. Cross-architecture audit over 11 models. Two upstream PRs: LFM2.5 H200 MoE
config (#32687) and Gemma-3 RMSNorm (#32670).
**中文** —— NCU/nsys profiling → LFM2.5 七项 kernel 组件（两个手写 Triton kernel + 五处调用点融合）→ +6 %。
11 模型跨架构审计。两个上游 PR：LFM2.5 H200 MoE config（#32687）和 Gemma-3 RMSNorm（#32670）。
`docs/2026-07-08/` … `docs/2026-07-28/`

### Phase 4 — FX fusion discovery & honesty corrections (Jul 29 → Jul 31) / 阶段四：FX 融合发现与诚实纠正
**EN** — Portable FX-graph fusion scanner (hardware-independent, MAIA-relevant): reproduced our manual
+36.6 % case automatically; found torch.compile already does most hand fusions; isolated one gap even the
compiler misses (QK-norm slice). Two retractions documented (Triton 3.6 baseline contamination; the
Gemma wiring case where 97 % of the apparent win belonged to another in-flight PR).
**中文** —— 可移植的 FX-graph 融合扫描器（硬件无关，与 MAIA 相关）：自动复现了我们手工发现的 +36.6 % 案例；
发现 torch.compile 已自动完成大多数手工融合；找到一个连编译器也融不掉的机会（QK-norm 切片）。
记录了两次撤回（Triton 3.6 基线污染；Gemma 接线案例中 97 % 的表观收益其实属于另一个在飞 PR）。
`docs/2026-07-29/` … `docs/2026-07-31/`

### Phase 5 — Three-layer factorial, cross-model & real workloads (Aug 02 → Aug 10) / 阶段五：三层全因子、跨模型与真实 workload
**EN** — The capstone experiments: the 48-cell L1×L2×L3 factorial (08-04); replication on OLMo-2 and
Falcon-H1, the latter giving a fresh **+27.63 %** SSD-kernel win; the nine-workload real/agentic external
validity study (08-07); and the L2/L3/L2+L3 ablation on those workloads (08-10) that pins down L2's
non-transferability. A reusable **methodology** doc and campaign skill fell out of this phase.
**中文** —— 收官实验：48 格 L1×L2×L3 全因子（08-04）；在 OLMo-2 与 Falcon-H1 上复现，后者给出全新的
**+27.63 %** SSD-kernel 收益；九 workload 真实/agentic 外部有效性研究（08-07）；以及这些 workload 上的
L2/L3/L2+L3 消融（08-10），坐实 L2 不可迁移。本阶段还沉淀出一份可复用的**方法论**文档与 campaign skill。
`docs/2026-08-02/` … `docs/2026-08-10/`, `docs/2026-08-04/METHODOLOGY_three_layer_optimization.md`

---

## 5. Upstream PRs / 上游 PR (external, verifiable / 外部可验证)

| PR / Issue | Model | What / 内容 | Result / 结果 | State |
|---|---|---|---|---|
| [sglang#32687](https://github.com/sgl-project/sglang/pull/32687) | LFM2.5 | H200 MoE tuned config (the missing device) / 补 H200 MoE config | long prefill **+23.3 %** | draft |
| [sglang#32670](https://github.com/sgl-project/sglang/pull/32670) | Gemma-3 | RMSNorm high-rank + dtype dispatch fix | low-batch decode **+36.6 %** | draft |
| [sglang#33416](https://github.com/sgl-project/sglang/pull/33416) (issue [#33415](https://github.com/sgl-project/sglang/issues/33415)) | OLMo-2 | bypass model's own fused QK-norm kernel | prefill **1.24×**, token-identical | draft |
| SLO-agent #9 / #30 | — | agent `scan` + `kernel_fusion_gap` mode + case KB | 5/5 backtest / 6 confirmed + 5 refuted | internal |

---

## 6. Detailed catalogs / 详尽编目

**EN** — For file-level detail (what every single document, script, and skill does), read the three
machine-readable catalogs. They are exhaustive: one row per file, bilingual.

**中文** —— 文件级细节（每一个文档、脚本、skill 具体干啥）见三份机器可读编目。它们是详尽的：每个文件一行，双语。

| Catalog / 编目 | Covers / 覆盖 | Count / 数量 |
|---|---|---|
| [`docs/2026-08-14/CATALOG_docs.md`](docs/2026-08-14/CATALOG_docs.md) | every doc under `docs/` + root `.md`, chronological / 所有文档 + 根 .md，按时间 | ~144 |
| [`docs/2026-08-14/CATALOG_scripts.md`](docs/2026-08-14/CATALOG_scripts.md) | every `.py`/`.sh` under `scripts/`, grouped by function / 所有脚本，按功能 | 204 |
| [`docs/2026-08-14/CATALOG_skills.md`](docs/2026-08-14/CATALOG_skills.md) | every reusable skill under `.github/skills/` / 所有可复用 skill | 18 |

> The raw extracted descriptions used to build these catalogs are kept alongside as `*.tsv`
> (`docs_dated.tsv`, `scripts.tsv`, `skills.tsv`) so the catalogs can be regenerated.
> 用于生成编目的原始提取描述以 `*.tsv` 形式保留，便于重新生成。

---

## 7. Reproduce / 复现

**EN** — Everything runs from the pinned stack; never upgrade the runtime to get a feature.

**中文** —— 全部基于固定的软件栈运行；**不要**为了拿某个功能而升级 runtime。

```bash
conda activate sglang-dev                       # env / 环境
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python

# --- The flagship 3-layer factorial on a regime (Bar 2→3→4) ---
GPU=4 REGIME=C_long_prefill bash scripts/lfm_fusion/exp3_layered.sh
python scripts/lfm_fusion/exp3_analyze.py --regime C_long_prefill

# --- L1 serving ceiling (192-point exhaustive grid, per model) ---
python scripts/run_serving_ceiling_campaign.py --init --models lfm25
python scripts/run_serving_ceiling_campaign.py --gpu 4 --worker w4

# --- L3 audit → implement → verify on a NEW model ---
python scripts/lfm_fusion/lf_audit.py --model <M> --regime C_long_prefill --gpu <G>

# --- Real/agentic workload L2/L3 ablation (this month) ---
bash scripts/lfm_fusion/rt_l2l3_matrix.sh 0 52200 RT_tool_agent_x1 RT_tool_agent_x4
python scripts/lfm_fusion/rt_l2l3_consolidate.py
```

**The reusable operating manual for a new model / 在新模型上复用的操作手册**:
[`docs/2026-08-04/METHODOLOGY_three_layer_optimization.md`](docs/2026-08-04/METHODOLOGY_three_layer_optimization.md)
(six phases, ~20–25 GPU-h, ~6–8 h wall-clock on 4 GPUs / 六阶段，约 20–25 GPU 小时，四卡约 6–8 小时挂钟).

---

## 8. Open items / 待办 (for a successor / 给接手者)

**EN**
1. **Full 2³ (incl. L1) on the new real workloads** — needs a 192-point grid **per workload** (~6 h each);
   L1 config does not transfer, so it must be re-searched.
2. **Re-tune L2 for ShareGPT** — its prefill lands outside the Mooncake-tuned M-buckets; the current L2 is a
   *transferred* config there, not a ceiling (see 08-10).
3. **Agentic SWE / OpenHands (multi-turn)** — needs backporting `AgenticTraceDataset` + a frozen-replay mode
   so baseline/candidate see identical per-turn prompts.
4. **NCU headroom → narrative** — the microarch data in `results/2026-07-08_v5_ncu/` etc. is not yet wired
   into the M4 story.

**中文**
1. **在新真实 workload 上做完整 2³（含 L1）**——每个 workload 需 192 点网格（各约 6 h）；L1 config 不可迁移，必须重搜。
2. **给 ShareGPT 重扫 L2**——它的 prefill 落在 Mooncake 调优 M-bucket 之外，当前 L2 在那里是*迁移*配置而非上限（见 08-10）。
3. **Agentic SWE / OpenHands（多轮）**——需 backport `AgenticTraceDataset` + frozen-replay，使 baseline/candidate 每轮 prompt 一致。
4. **NCU headroom 串进叙事**——`results/2026-07-08_v5_ncu/` 等的微架构数据尚未接入 M4 故事。

---

## 9. Repo map / 顶层目录指南

| Path / 路径 | EN | 中文 |
|---|---|---|
| `docs/YYYY-MM-DD/` | dated experiment records (primary) | 按日期的实验记录（主） |
| `docs/<named>/` | living docs (architecture, research, reports…) | 常驻文档（架构、调研、报告…） |
| `.github/skills/` | reusable methodology skills | 可复用方法论 skill |
| `scripts/` | all harnesses & analysis (204 files) | 全部 harness 与分析脚本（204 个） |
| `scripts/lfm_fusion/` | LFM2.5 + cross-model kernel A/B harness | LFM2.5 及跨模型 kernel A/B harness |
| `scripts/fx_fusion/` | portable FX-graph fusion discovery | 可移植 FX-graph 融合发现 |
| `results/` | raw run outputs (large; jsonl gitignored) | 原始运行输出（大；jsonl 已 gitignore） |
| `configs/` | serving + MoE config profiles | serving 与 MoE config 配置 |
| `patches/` | upstream PR patch worktrees | 上游 PR patch 工作树 |
| `experiments/`, `stages/`, `regimes/` | Stage-A/B agent system | Stage-A/B agent 系统 |
| `README.md` | agent-system quick start | agent 系统快速上手 |
| `plan.md`, `HANDOFF_regime_kernel.md` | project status + kernel-line handoff | 项目状态 + kernel 线交接 |

---

*Generated 2026-08-14 as the departing-engineer handoff. All numbers are from committed results under
`results/`; all claims trace to a dated doc under `docs/`. / 本文件为离任交接，所有数字来自 `results/`
下已提交的结果，所有论断可追溯到 `docs/` 下的按日期文档。*
