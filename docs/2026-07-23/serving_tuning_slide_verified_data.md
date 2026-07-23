# Serving-level tuning slide — verified data & source audit

> 目的：为修订 "serving-level tuning" 幻灯片，**先定位并核验**所需的两组数据（高并发 admission-capacity + 长输入 chunked-prefill），全部追溯到仓库原始 CSV/JSON。**所有数字均已从 raw 文件复算，未使用记忆值。**
> 审计 commit：`100ba9f`（main）· 日期：2026-07-23 · 环境只读，未改动任何原始实验文件。

---

## ⚠️ 最重要的核验结论（必须先看）

**用户记忆中的 "Qwen 长输入 8K/16K/32K/50K ≈ 1.36/1.55/1.47/2.21×" 实际上是 LFM2.5-8B-A1B 的数据，不是 Qwen。**

- 仓库中**唯一**包含 `R_prompt_8k/16k/32k/50k` 长输入 regime 的调优实验是 **2026-07-02 的 v3 Optuna study，模型为 LFM2.5-8B-A1B**（`results/2026-07-02_lfm2.5_v3/`，`baseline-cookbook/server_config_used.yaml` 明确 `model-path: /data/hf/LFM2.5-8B-A1B`）。
- `consolidated_config_spreadsheet.csv` 全部行是 v2/v3（均为 LFM2.5），**没有任何 Qwen 的 8K–50K 长输入行**。
- 因此：**不存在可核验的 Qwen 8K–50K 长输入调优曲线。** 按 prompt 要求"If those Qwen raw files cannot be found or cannot be validated, do not invent the curve" —— 右图应使用 **LFM2.5 长输入曲线**（并明确标注模型），或退回 LFM2.5 真实负载对比。

**第二个关键 caveat**：LFM2.5 长输入的获胜 config 是**多 knob**（chunked-prefill=8192 **且** schedule=fcfs **且** mem_fraction=0.75 **且** attention=fa3），相对 cookbook（chunk=-1, lpm, mem=0.85, auto）**这四项全都改了**。→ 必须标注为 **"long-context tuned configuration"**，不能声称收益全部来自 chunked prefill。

---

## 1. 高并发 decode（admission capacity）— 已核验 ✅

- **源文件**：`results/consolidated_v4_by_model_config.csv`
- **baseline**：`cookbook_baseline`（cap=32, chunk=-1, lpm, mem=0.85, moe/attn=auto, cudagraph on）
- **candidate**：`big_batch_cap128`（cap=128, chunk=2048, **fcfs**, mem=0.90, cudagraph on）→ **多 knob**（不止 cap）
- **指标**：request throughput（`req_per_s`），speedup = candidate / baseline

| 模型 | 场景 | baseline req/s | candidate req/s | **speedup** |
|---|---|---:|---:|---:|
| Qwen3-30B-A3B-bf16 | C64 / O512 | 7.410 | 11.663 | **1.57×** |
| Qwen3-30B-A3B-bf16 | C128 / O256 | 14.884 | 36.363 | **2.44×** |
| LFM2.5-8B-A1B | C64 / O512 | 12.306 | 17.332 | **1.41×** |
| LFM2.5-8B-A1B | C128 / O256 | 23.742 | 50.459 | **2.13×** |

→ 与用户提供值完全一致（1.57/2.44/1.41/2.13）。candidate 标签应为 **"High-concurrency configuration"**（cap + schedule + chunk + mem 均改），不要只写 "Cap-128"。

---

## 2. 长输入 / prefill-heavy（chunked prefill）— 已核验 ✅（但模型 = LFM2.5）

- **源文件**：`results/consolidated_config_spreadsheet.csv`（列 `R_prompt_*__speedup` / `__req_per_s`）+ `results/2026-07-02_lfm2.5_v3/`
- **模型**：**LFM2.5-8B-A1B**（hybrid linear-attn + MoE）· 单卡 **H200**（GPU6, SM90）· **bf16**
- **软件**：sglang commit `27f2b6f9b5`（dirty；version string 报 `0.0.0` 不可靠）· flashinfer 0.6.3 · torch 2.9.1 · **triton 3.5.1** · CUDA 12.8 · driver 580.105.08
- **baseline**：`v3-baseline-cookbook`（cap=32, chunk=-1, lpm, mem=0.85, moe/attn=auto, cudagraph on）
- **tuned**：`v3-trial-0028`（cap=64, **chunk=8192**, **fcfs**, **mem=0.75**, **attn=fa3**, cudagraph on）→ **long-context tuned configuration（多 knob）**
- **指标**：request throughput（`req_per_s`），speedup = tuned / cookbook
- **重复**：bench `num_runs=3`，丢弃 run0，2 次取中位数；trial 为单次 TPE 评估

| regime | prompt_words | out | conc | n | cookbook req/s | tuned req/s | **speedup** |
|---|---:|---:|---:|---:|---:|---:|---:|
| R_prompt_8k_c4_out128 | 8000 | 128 | 4 | 8 | 5.3204 | 7.2844 | **1.37×** |
| R_prompt_16k_c2_out128 | 16000 | 128 | 2 | 4 | 2.6978 | 4.0833 | **1.51×** |
| R_prompt_32k_c1_out128 | 32000 | 128 | 1 | 2 | 1.6330 | 2.4420 | **1.50×** |
| R_prompt_50k_c1_out64 | 50000 | 64 | 1 | 2 | 1.2750 | 2.8554 | **2.24×** |

（备选 tuned 配置 `v3-trial-0016`：cap=32, chunk=8192, fcfs, mem=0.75, attn=fa3 → 8k 1.36× / 16k 1.60× / 32k 1.48× / 50k 2.23×，与记忆值更接近。两者都是多 knob。）

→ 与用户记忆的 1.36/1.55/1.47/2.21 量级一致，**但模型是 LFM2.5 而非 Qwen**，且是多 knob 调优。

### 与 6/25 R_long_prefill Optuna study 的区别（避免混淆）
6/25 的 `R_long_prefill`（`docs/2026-06-25/...`）报告仅 ~1.05×，其获胜 config `chunked-prefill-size=-1`。本节是**另一个** 2026-07-02 的 8K/16K/32K/50K 专门实验（chunk=8192）。两者不是同一回事。

---

## 3. 真实负载 chunked-prefill 对比（跨模型差异）— 已核验 ✅

- **源文件**：`results/consolidated_v7_config_sweep.csv`
- **candidate**：LFM2.5 用 `chunked8192`，Qwen 用 `tuned_chunked`；baseline 为各自 cookbook/baseline_triton
- **指标**：`req_s`（request throughput）与 `median_ttft_ms`

| 模型 | 数据集 | throughput Δ | median TTFT Δ |
|---|---|---:|---:|
| **LFM2.5** | shared_prefix | **+28.6%**（14.09→18.12） | **−53.3%**（2758→1287ms） |
| **LFM2.5** | tool-agent | **+6.6%**（16.13→17.20） | **−86.8%**（1583→209ms） |
| **Qwen** | shared_prefix | **−2.9%** | +1.3% |
| **Qwen** | tool-agent | **+0.7%** | −9.8% |

→ 与用户提供值完全一致。**同一个 chunked-prefill candidate 在不同模型/regime 表现迥异** —— LFM2.5 上大幅正向，Qwen 上 shared_prefix 吞吐反而 −2.9%。**chunking 不是普适胜利。**

---

## 4. Speedup 计算公式

- 高并发 & 长输入：`speedup = req_per_s(tuned) / req_per_s(cookbook_baseline)`（同模型、同 regime、同 H200、bf16）。
- 真实负载 Δ：`throughput_delta = req_s(candidate)/req_s(baseline) − 1`；`ttft_delta = median_ttft(candidate)/median_ttft(baseline) − 1`。
- 所有对比均为**同模型、同硬件、同 dtype**内部比较，未跨环境混比。

---

## 5. 未核验 / 缺失的实验

| 项 | 状态 |
|---|---|
| **Qwen 8K/16K/32K/50K 长输入调优曲线** | ❌ **不存在**。仓库无 Qwen 长输入 `R_prompt_*` 调优行；记忆中的曲线实为 LFM2.5。 |
| v3 长输入的逐 knob 消融（chunk 单独 vs 全 config） | ❌ 未做。无法把长输入收益归因到 chunked-prefill 单一 knob。 |
| v3 sglang 精确 version | ⚠️ version string 报 `0.0.0`，仅有 git commit `27f2b6f9b5`。 |
| 长输入重复次数 | ⚠️ 每 cell 仅 2 次有效 run（num_runs=3 丢 run0），无 t 检验。 |

---

## 6. 结论：长输入收益是"单纯 chunked prefill" 还是"多 knob 调优配置"？

**是多 knob 的 long-context tuned configuration。** LFM2.5 v3-trial-0028 相对 cookbook 同时改了 4 项：`chunked-prefill-size -1→8192`、`schedule-policy lpm→fcfs`、`mem-fraction-static 0.85→0.75`、`attention-backend auto→fa3`。因此幻灯片右图必须：

- 标注模型为 **LFM2.5-8B-A1B**（不是 Qwen）；
- 标签用 **"long-context tuned configuration"**，并在注解列出改动的 knob；
- 附小字说明"同一 chunked candidate 在 Qwen shared_prefix 上吞吐 −2.9%，chunking 非普适胜利"（来自 §3 v7 数据）。

---

## 7. 源文件清单（供 footer 引用）

- `results/consolidated_v4_by_model_config.csv`（高并发 c64/c128）
- `results/consolidated_config_spreadsheet.csv`（LFM2.5 长输入 8K–50K speedup）
- `results/2026-07-02_lfm2.5_v3/{baseline-cookbook,optuna-v3-R_concurrent_decode/trial_0028}/`（config + raw per_run）
- `results/consolidated_v7_config_sweep.csv`（真实负载 shared_prefix / tool-agent）
- 核验数据机读版：`docs/2026-07-23/serving_tuning_slide_verified_data.json`
