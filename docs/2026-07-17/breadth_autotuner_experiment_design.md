# 实验设计：AutoTuner 普适性广度实验（20 model × regime × dataset × config）+ Mason 深度线

**状态**：设计稿，待 review 后执行
**日期**：2026-07-17
**作者**：@gujialiang123（结合 Chendi + Mason 的要求）
**目标机器**：新的 8×GPU 机器（待配置；harness 设计为可移植，clone 仓库即可跑）

---

## 0. 一页总览：这个实验要回答什么

**Chendi（广度）**：现在 AutoTuner 只在 **1 个 model（LFM2.5）、只在 long-context regime** 证明了提升（R_prompt_8k~50k：+36%~+121%；R_concurrent_decode 打平）。这是单点证据。**需要用 ~20 个 model × 多 regime × 多 dataset，系统测出"调 serving 参数到底对谁、在什么情况下有提升"**，先拿全量数据，再判断 AutoTuner 的普适性。

**Mason（深度）**：在 config 调完之后，对 **Qwen3-30B MoE 单点**继续往 kernel 层钻：per-regime config 天花板 → kernel shape 覆盖 → 选 kernel 手改/融合 → 产出"分层逼近 roofline"图，量出手改相对 autotune 的额外提升 **X**。

**两者的衔接**：广度实验的**每个 (model, regime) 的 config sweep 曲线**，正是 Mason 第 2 步要的"config autotuning 天花板 + plateau，且价值因 regime 而异"。所以**一套 harness 同时喂两条线**：横向铺 20 model（Chendi），纵向在 Qwen3-30B 这一行接 NCU/kernel（Mason）。

**硬约束（Chendi 明确）**：所有参数**必须保持输出分布不变**——只调影响"速度"的 serving 旋钮，**禁止 quantization / 改 dtype / 任何改变数值输出的参数**。baseline 以 **sglang cookbook** 官方"已验证最佳启动参数"为准。

---

## 1. Model 清单（★待你 review）

**原则**：H200 单机（≤8 GPU）可跑；覆盖 MoE 与 dense、覆盖不同规模、覆盖不同架构（标准 attn / GQA / 混合 mamba）；优先 cookbook 里有官方 baseline 的；bf16（或模型原生非量化 dtype）。

> 图例：📦本地已有　⬇️需下载　🏭cookbook 有官方 baseline

### 1A. MoE 模型（项目重点，占多数）
| # | Model | 规模(总/激活) | 架构 | TP(H200) | 来源 | 备注 |
|---|---|---|---|---|---|---|
| 1 | Qwen3-30B-A3B-Instruct-2507 | 30B/3B | MoE, E=128 top-8 | 1 | 📦 | Mason 深度线主角 |
| 2 | LFM2.5-8B-A1B | 8B/1B | MoE+mamba 混合 | 1 | 📦🏭 | 已证明提升的基准 |
| 3 | gemma-4-26B-A4B-it | 26B/4B | MoE | 1–2 | 📦 | |
| 4 | Qwen3-235B-A22B（若显存允许）| 235B/22B | MoE | 4–8 | ⬇️ | 大 MoE 上界 |
| 5 | Mixtral-8x7B-Instruct | 47B/13B | MoE E=8 top-2 | 2 | ⬇️ | 经典粗粒度 MoE 对照 |
| 6 | Mixtral-8x22B-Instruct | 141B/39B | MoE E=8 top-2 | 4–8 | ⬇️ | |
| 7 | DeepSeek-V2-Lite | 16B/2.4B | MoE 细粒度 | 1 | ⬇️ | 细粒度 MoE |
| 8 | GLM-5.2（cookbook）| — | MoE | 依 cookbook | ⬇️🏭 | cookbook H200 FP8 verified（我们跑 bf16 变体） |
| 9 | MiniMax-M3（cookbook）| — | MoE | tp=8 | ⬇️🏭 | cookbook verified |
| 10 | Laguna-M.1（cookbook）| — | — | tp=8 | ⬇️🏭 | cookbook verified |
| 11 | OLMoE-1B-7B | 7B/1B | MoE | 1 | ⬇️ | 小 MoE |
| 12 | Phi-3.5-MoE-instruct | 42B/6.6B | MoE | 2 | ⬇️ | |

### 1B. Dense 模型（对照组，看"MoE vs dense 对调参的敏感度差异"）
| # | Model | 规模 | 架构 | TP | 来源 | 备注 |
|---|---|---|---|---|---|---|
| 13 | Qwen3-0.6B | 0.6B | dense | 1 | 📦 | 小 dense 快速基准 |
| 14 | gemma-3-1b-it | 1B | dense | 1 | 📦 | |
| 15 | Llama-3.1-8B-Instruct | 8B | dense GQA | 1 | ⬇️ | 最常用 dense 基准 |
| 16 | Qwen3-8B | 8B | dense | 1 | ⬇️ | |
| 17 | Llama-3.3-70B-Instruct | 70B | dense GQA | 2–4 | ⬇️ | 大 dense |
| 18 | Qwen3-32B | 32B | dense | 1–2 | ⬇️ | 与 30B-A3B MoE 同级对照 |
| 19 | Mistral-Small-24B | 24B | dense | 1 | ⬇️ | |
| 20 | gemma-2-27B-it | 27B | dense | 1–2 | ⬇️ | |

**待你确认**：(1) 是否有你指定必须包含的 model？(2) cookbook 里的 GLM/MiniMax/Laguna 名称/权重是否可下载？(3) 235B 这种超大 MoE 是否纳入（很吃显存 + 时间）？(4) 20 个是硬性数量还是"~20 左右"？

---

## 2. Regime 清单（统一规范，当前所有 model 通用）

**regime = 固定的工作负载画像**（决定 prefill/decode 比例、batch、并发）。合并历史定义为一套规范坐标。

| Regime ID | in_len | out_len | 并发 | 画像 | 主要压力 |
|---|---|---|---|---|---|
| `R_decode_heavy` | 200 | 256 | 32 | 短输入长输出高并发 | decode / MoE 搬运 |
| `R_balanced` | 800 | 256 | 8 | 均衡 | 混合 |
| `R_prefill_heavy` | 4000 | 32 | 4 | 长输入短输出 | prefill / attention |
| `R_prompt_long` | {8k,16k,32k,50k} | 64 | 4 | 超长上下文（AutoTuner 已知获益区）| prefill 分块 |
| `R_agent`（真实）| ~2700 | ~207 | 真实到达 | mooncake toolagent trace | 真实 agent |

> long-context 是 AutoTuner 目前唯一见效的 regime，必须包含以验证"是否只有 long-context 获益"。

**待确认**：`R_prompt_long` 的 4 个长度是全跑还是选 2 个（16k/50k）以省时间？

---

## 3. Dataset 清单

| Dataset | 来源 | 对应 regime | 说明 |
|---|---|---|---|
| `toolagent` | mooncake FAST'25 公开 trace | R_agent | 真实 agent |
| `shared_prefix` / `generated-shared-prefix` | sglang 合成 | 各 regime | 高前缀共享，radix cache 友好 |
| `random` | sglang 随机 in/out | 各 regime | 通用基线 |
| `sharegpt` | ShareGPT 对话 | R_balanced | 对话画像 |

**同一 regime 可跨 dataset 跑**，看 dataset 分布对"调参增益"的影响（Chendi 明确要"不同 dataset"）。

---

## 4. 参数网格（serving 旋钮，★保证不改分布）

### 4A. 允许调的参数（复用 `harness/autotune_v3_lfm.py` 搜索空间，**已天然排除 quantization**）
| 参数 | 候选值 | 是否改分布 |
|---|---|---|
| `mem-fraction-static` | {0.75, 0.85, 0.90} | ❌ 不改 |
| `max-running-requests`（cap）| {8, 16, 32, 64, 128, 192, 256} | ❌ |
| `chunked-prefill-size` | {-1, 2048, 8192, 16384} | ❌ |
| `schedule-policy` | {lpm, fcfs} | ❌（Chendi 明确允许改 schedule）|
| `attention-backend` | {fa3, flashinfer, triton} | ❌（数值等价，实现不同）|
| `disable-cuda-graph` | {True, False} | ❌ |
| `moe-runner-backend`（MoE 模型）| {auto, triton, cutlass} | ❌ |

### 4B. 禁止的参数（会改分布 → 精度不可比）
🚫 quantization（fp8/awq/nvfp4…）、🚫 改 dtype、🚫 kv-cache 量化、🚫 speculative decoding（虽 exact，但引入 draft 变量，本轮排除）、🚫 采样参数（固定 greedy 或固定 seed+温度）。

### 4C. 两种执行模式（都要，回答不同问题）
1. **网格 sweep（给 Chendi 的完整 spreadsheet）**：对关键旋钮做**结构化网格**（如 cap × chunked × schedule），每格 3 repeat。产出可画曲线/看 plateau 的密集点。
2. **Optuna 搜索（AutoTuner 本体）**：用 v3 的 TPE 在完整空间搜 best config，warm-start 防冷启动偏差。产出"AutoTuner 找到的 best vs baseline"的增益。

> Mason 第 2 步（per-regime 天花板 + plateau）用模式 1 的网格数据；Chendi 的"AutoTuner 有没有用"用模式 2 的 best-vs-baseline。

---

## 5. Baseline 定义（cookbook 为准）

| 情况 | baseline 取法 |
|---|---|
| model 在 cookbook 里（LFM/GLM/MiniMax/Laguna）| 用 cookbook 官方 verified config（去掉 quantization flag，改 bf16 变体）|
| model 不在 cookbook（Qwen/Llama/Mixtral 等）| 用 **sglang 默认启动参数**，明确标注"non-cookbook default" |
| 已知 cookbook 与 true-default 差异 | LFM 上实测 <1%（parser flag 不影响吞吐），沿用 |

每个 model 的 baseline 也跑 **3 repeat**，作为增益计算的分母。

---

## 6. Harness 架构（可移植，20 model × 8 GPU 并行 × 3 repeat）

### 6A. 设计原则
- **配置驱动**：一个 `configs/breadth/models.yaml`（model→路径/tp/是否cookbook）+ `regimes.yaml` + `param_grid.yaml`，改清单不改代码。
- **可移植**：路径用相对/env 变量（`$MODEL_ROOT`、`$RESULT_ROOT`），clone 到新机器只需设 2 个 env + 填 models.yaml。
- **GPU 调度器**：一个轻量 job queue，把 (model, config, repeat) 任务按 tp 需求打包到 8 张卡上并行（tp=1 的模型可 8 路并行；tp=4 的占半机）。
- **崩溃可续跑**：每个 (model, regime, dataset, config, repeat) 是独立 job，落地 `bench_summary.json`；已完成的 job 跳过（幂等）。
- **原始 log 全留**（符合你偏好）：server 日志、bench 原始输出、resolved config 全部落盘，便于事后补指标。

### 6B. 目录结构（拟）
```
harness/breadth/
  run_breadth.py            # 主入口：读 yaml，生成 job 列表，调度到 GPU
  gpu_scheduler.py          # 8-GPU job 打包 + 并发控制
  run_one_job.py            # 起 server → warmup → bench_serving → 收指标 → kill
  aggregate_to_spreadsheet.py  # 所有 bench_summary.json → 完整 xlsx/csv
configs/breadth/
  models.yaml  regimes.yaml  datasets.yaml  param_grid.yaml
results/2026-07-XX_breadth/
  <model>/<regime>/<dataset>/<config_hash>/repeat_{1,2,3}/bench_summary.json
  spreadsheet.xlsx
```

### 6C. 单 job 流程
起 sglang server（指定 config）→ 等就绪 → warmup N 请求 → `sglang.bench_serving`（指定 dataset/regime，固定随机 seed）→ 采集指标 → 记录 → kill server → 下一个 job。

### 6D. 规模估算（供排期）
- 任务数 ≈ 20 model × 5 regime × 2 dataset × (网格 ~12 config + baseline) × 3 repeat ≈ **数千 job**。
- 需要**分批 + 幂等续跑**；先做 pilot（下节）确认单 job 时长，再估总墙钟。
- **建议先跑一个"精简网格"**（每 model 选 3–4 个代表 config）确认全流程，再决定是否铺满全网格。

---

## 7. 采集的性能指标（每个 config × repeat）

| 类别 | 指标 |
|---|---|
| 吞吐 | req/s、output tok/s、total tok/s |
| 延迟-TTFT | median / p90 / p99 TTFT (ms) |
| 延迟-TPOT | median / p99 TPOT（time-per-output-token）(ms) |
| 延迟-E2E | median / p99 端到端 (ms) |
| 其它 | 实测并发、完成请求数、GPU 显存占用、server 启动耗时 |
| 元信息 | model、regime、dataset、完整 config、repeat、spec_hash、sglang/triton 版本、GPU 型号 |

3 repeat → 汇总为 **mean ± std**，并标注 CV（变异系数）以判断噪声。

---

## 8. Spreadsheet schema（最终产出）

一行 = 一个 (model, regime, dataset, config)（3 repeat 已聚合）。列：

```
model | arch(MoE/dense) | params | is_cookbook_baseline | regime | dataset |
config_id | mem_frac | max_running | chunked_prefill | schedule | attn_backend | cuda_graph | moe_backend |
req_s_mean | req_s_std | out_tok_s_mean | ttft_p50 | ttft_p99 | tpot_p50 | tpot_p99 | e2e_p50 | e2e_p99 |
vs_baseline_req_s_pct | vs_baseline_tpot_pct | is_significant(>3σ) | notes
```

额外 sheet：
- **Summary**：每个 model 的 best-config 增益 vs baseline（AutoTuner 视角）。
- **Per-regime plateau**：每 (model, regime) 的 config 曲线关键点（Mason 视角）。
- **Winners**：增益 >5% 且显著的 (model, regime, dataset) 列表。

---

## 9. 分析口径

### 9A. Chendi 广度分析（AutoTuner 普不普适）
1. **谁获益**：列出所有"AutoTuner best 显著优于 baseline"的 (model, regime, dataset)，增益幅度。
2. **规律**：增益是否**只集中在 long-context**？MoE vs dense 谁更敏感？大模型 vs 小模型？dataset 分布的影响？
3. **反例**：哪些 model/regime **完全没有**提升（baseline 已最优）→ 说明 AutoTuner 的适用边界。
4. **结论形态**："AutoTuner 在 {满足条件 X} 的场景平均提升 Y%，在 {条件 Z} 无效"。

### 9B. Mason 深度分析（Qwen3-30B 单点，逼近 roofline）
1. **per-regime config 天花板 + plateau**（用 §4 模式 1 网格）：画每个 regime 的 config sweep 曲线，标出 plateau，突出"config 价值因 regime 而异"。
2. **kernel shape 覆盖**：已发现 triton 3.5.1 无 H200 E=128,N=768 config → 重跑 fused_moe autotune，量 **constexpr 层提升**。
3. **kernel 手改/融合**：选 fused_moe（7% 峰值），做 (a) 硬编码常量 CUDA 重写 / (b) 融合周边（router→gather→GEMM→scatter→moe_sum）→ 量 **X**。
4. **总图**：baseline → config autotune → kernel constexpr autotune → kernel 手改，四层各自的可测量提升 + 距 roofline 的位置。
5. **X 的判定**：X 小 → autotuning story（补 config 覆盖，走 PR）；X 大 → kernel-agent story（有实证价值）。

---

## 10. 执行阶段（建议顺序）

| 阶段 | 内容 | 依赖 | 产出 |
|---|---|---|---|
| **P0 设计定稿** | review 本文档，敲定 model 清单/regime/网格规模 | 你 review | 定稿 |
| **P1 harness 搭建** | 写 `run_breadth.py` + gpu_scheduler + aggregate，本地 pilot（当前机器，2–3 个本地 model × 精简网格）跑通全流程 | P0 | 可移植 harness + pilot 数据 |
| **P2 新机器部署** | clone 仓库到 8-GPU 机器，配 env + 下载 model + 填 models.yaml | 新机器就绪 | 环境就绪 |
| **P3 广度全量** | 20 model × regime × dataset × 网格 × 3 repeat，分批幂等跑 | P1,P2 | 完整 spreadsheet（Chendi 交付）|
| **P4 广度分析** | 按 §9A 分析，出"AutoTuner 普适性"报告 | P3 | Chendi 汇报 |
| **P5 Mason 深度** | Qwen3-30B 接 NCU/kernel：shape autotune + fused_moe 手改 → X + roofline 图 | P3(拿到 Qwen config 曲线) | Mason 汇报 |

> P1 可**立即在当前 H200 机器**开始（不阻塞新机器）；P5 与 P3/P4 可并行。

---

## 11. 风险 / 待决

| 项 | 说明 |
|---|---|
| model 下载量大 | 15+ model 需下载，数 TB；需确认新机器存储 + 带宽 |
| 超大 MoE（235B/8x22B）| 吃满 8 卡 + 单 job 很慢；可能降级为"可选" |
| cookbook model 可得性 | GLM/MiniMax/Laguna 的权重是否公开可下载需核实 |
| 总墙钟 | 数千 job；P1 pilot 后才能准确估算；可能需要"精简网格"策略 |
| 分布不变的验证 | 建议对每个 model 抽 1–2 config 做 top-1 token 一致性 sanity check，确认 serving 旋钮确实不改输出 |
| bench 噪声 | 3 repeat + CV 标注；关键结论要求 >3σ |

---

## 附：复用的现有资产
- AutoTuner：`harness/autotune_v3_lfm.py`（搜索空间、warm-start、Optuna，需泛化为 model-agnostic）
- cookbook baseline：`docs/2026-06-30/cookbook_baseline_analysis.md` + `sglang_cookbook_deployment_baselines.json`（77 config）
- config sweep 先例：`docs/2026-07-09/v8_tuning_on_real_workload.md`、`results/consolidated_v8_tuning.csv`
- LFM 已证明的提升：`docs/2026-07-02/lfm2.5_v3_mfu_longctx.md`（long-ctx +36~121%）
- Mason 深度线矩阵：`docs/2026-07-16/mason_roadmap_qwen3_moe_matrix.md`
- regime/NCU/roofline/GFLOP/s：`docs/2026-07-14/opportunity_gap_comprehensive_analysis.md`、`results/2026-07-15_v18_gflops/`
