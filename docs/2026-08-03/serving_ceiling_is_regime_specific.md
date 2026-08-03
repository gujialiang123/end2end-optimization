# regime C 的 autotuning ceiling 不是 cookbook —— 以及这对交付叙事意味着什么

**日期**：2026-08-03 · **性质**：既有数据的重新解读 + 一个新的验证实验
**关联**：`docs/2026-08-03/exp3_kernel_on_tuned_baseline.md`（同日的分层实验）

---

## 0. 一句话

我们一直把 **sglang cookbook 默认配置**当作 "best autotuning config" 来定义 Bar 2。
**在四个 workload 上这是对的（穷举只能多买 0.3%~1.8%），但在长 prefill 上是错的：
穷举能多买 +56.9%。**

交付图原计划用长 prefill 做主线。**按这个数据，那张图会被一击击穿**，
而换成 decode regime 之后，同一个论点反而更强。

**好消息**：在真正的 ceiling 配置上重测，kernel rewrite 依然值 **+6.26%（p=3.0e-08）**
（§5），和在 cookbook 基线上的 +6.18% 几乎相同。**论点本身站得住，
站不住的是我们对 Bar 2 的定义。**

---

## 1. 这个数据一直都在，只是没人把它和分层故事对上

「LFM2.5 上 autotuning 找不到提升」这个结论来自两份报告，**它们都只测了
`R_concurrent_decode`**：

| 报告 | 方法 | workload |
|---|---|---|
| `docs/2026-06-30/lfm2.5_conditional_autotuning.md` | 25 次条件化 Optuna | R_concurrent_decode |
| `docs/2026-07-22/lfm25_serving_autotuning_plateau.md` | 100 次 TPE，无热启动 | R_concurrent_decode |

而 `results/2026-07-24_serving_ceiling/` 那次 campaign **对 6 个 workload 各跑了全部
192 个 serving 配置**（`n_configs_evaluated=192`），并把 top 35 用 **n=5** 重测
（`results/2026-07-24_serving_ceiling_validation/`）。

这份数据里的长 prefill 那一行，从来没有和分层故事放在一起看过。

---

## 2. 各 regime 的真实 ceiling（n=5 验证集）

```
workload              cookbook  ceiling    gain   ceiling config
R_short_decode          1.681    1.688    +0.4%   cap8_chunk-1_fcfs_mem0.85
tool_agent              5.264    5.280    +0.3%   cap128_chunk8192_lpm_mem0.75
R_concurrent_decode    21.990   22.233    +1.1%   cap64_chunk8192_fcfs_mem0.75
R_medium_balanced       7.108    7.235    +1.8%   cap8_chunk2048_fcfs_mem0.9
R_long_prefill         12.604   19.781   +56.9%   cap8_chunk2048_fcfs_mem0.9
shared_prefix          14.080   27.262   +93.6%   cap96_chunk2048_lpm_mem0.9
```

生成脚本：`scripts/lfm25_serving_ceiling_per_regime.py`
产物：`results/2026-07-24_serving_ceiling_validation/analysis/lfm25/ceiling_per_regime.json`

**四个 workload 上「cookbook 就是上限」成立，而且是穷举意义上的成立**——
这比 25 次或 100 次 TPE 强得多，因为空间只有 192 个点而我们把有希望的都测了。

---

## 3. 长 prefill 上是哪个旋钮

`chunked_prefill_size`，而且**三档完全不重叠**：

| `chunked_prefill_size` | 验证的配置数 | req/s 范围 |
|---|---:|---|
| **2048** | 6 | **17.551 – 19.781** |
| 8192 | 16 | 13.549 – 15.456 |
| **−1（cookbook）** | 13 | **12.169 – 13.042** |

**cookbook 直接关掉了分块预填充。** 在 4000 token 的 prefill 上这要付 57% 的代价。

**而且这不是延迟换吞吐**：最佳配置同时把 TTFT p95 从 **208.5 ms 降到 94.0 ms**、
E2E p95 改善 37%，唯一退化的是 TPOT p95（+9%）——而这个 workload 只产出 32 个 token。
campaign 把它归为 `TRADE-OFF` 只是因为「有任一指标退化」，标签低估了它。

---

## 4. 这对交付图意味着什么

### 4.1 原计划（长 prefill 主线）会被击穿

```
Bar 2  cookbook            12.119  ← 我们称之为 "autotuning ceiling"
Bar 3  + tuned MoE config  14.939
Bar 4  + kernel rewrite    16.392  ← 号称在 ceiling 之上 +9.73%
```

**但真实的 serving ceiling 是 19.78 req/s。** 整个 Bar 4 还在 ceiling 底下。
审稿人只要读过 7/24 那份 campaign 就能一句话推翻它。

**这和 6/25 Qwen 报告里的问题是同一个**（default→tuned 空档巨大，
于是"改 kernel"显得没必要），只不过这次发生在我们自己选的主线 regime 上。

### 4.2 换成 decode regime，同一个论点强得多

| regime | 穷举 serving ceiling | kernel rewrite | 倍数 |
|---|---:|---:|---:|
| **A 低批 decode** | **+0.4%** | **+6.70%**（今日实测，n=16 双向） | **16×** |
| **B 并发 decode** | **+1.1%** | **+6.21%**（7/27） | **5.6×** |
| C 长 prefill | +56.9% | **+6.26%**（在 ceiling 配置之上实测，§5） | 0.11× |

**两边的基线逐字相同**：7/24 campaign 的 cookbook 配置是
`cap32_chunk-1_pollpm_mem0.85`，`lf_e2e.py` 里 regime A/B 的 serving knobs 也是
`cap=32, chunk=-1, policy="lpm", mem=0.85`。**可以直接拼成一张图，不需要新实验。**

而且交叉验证很干净：7/24 验证集测得 cookbook 在 R_short_decode 上是 **1.681 req/s**，
今天的实验测得 **1.686 req/s**——**差 0.3%**，跨 10 天、不同脚本。

**这是 Debadeepta 那句话可能拿到的最强形式**：
> 不是「25 次搜索没找到」，而是「**192 个配置全测了，最多多买 0.4%；
> 而 kernel rewrite 买到 6.70%，p=2e-41**」。

### 4.3 长 prefill 的正确做法

不是放弃它，而是把 Bar 2 换成真正的 ceiling 配置
（`cap=8, chunk=2048, policy=fcfs, mem=0.9`），在**那个**基线上重测 L2 和 L3。
见 §5。

---

## 5. 验证实验：在 L1 调优后的配置上重测 L3

**状态：部分完成（2026-08-03 23:27 用户叫停，改做别的实验）。**
无 config 的两格（正序+逆序）已跑完，**结论已经可以下**；
装 tuned MoE config 的两格没跑完，数据已删除。

### 5.1 做法

在 `lf_e2e.py` 里新增 regime `C_long_prefill_tuned`，serving knobs 换成 §2 的
ceiling 配置，其余（模型、backend、CUDA graph、workload）完全不变：

```python
"C_long_prefill_tuned": dict(workload="R_long_prefill",
                             cap=8, chunk=2048, policy="fcfs", mem=0.9),
```

### 5.2 ★ 先踩了一个坑：warmup 表是按 cookbook 配置标定的

第一版用默认的 4 次 warmup、8 次计分重复，结果是：

```
nocfg_rev   all7 22.590 ± 1.399   baseline 22.593 ± 0.859   → +0.0%（"收益归零"）
cfg_fwd     baseline 19.358 ± 5.641                          → 29% 标准差
```

看起来像是「kernel 收益在调优配置上消失了」。但逐次数据说不是：

```
cfg_fwd baseline:  [5.59, 20.97, 21.12, 20.81, 20.84, 20.86, 21.12, 23.57]
                    ↑ 单次灾难性离群
nocfg_fwd baseline:[20.90, 20.04, 21.64, 22.87, 22.17, 22.43, 22.81, 23.32]
                    ↑↑ 前两次仍在爬升
```

**`serving_ceiling_lib.WARMUP_RUNS` 的 4 次是按 cookbook 旋钮标定的；
换一组 serving 配置就是换一个 steady state。** 已加 `--warmup` 覆盖参数。

重跑用 **12 次 warmup + 30 次计分重复**（单次仅 0.2 s，成本几乎全在 server 启动）。

### 5.3 结果（无 tuned MoE config，正逆序合并 n=60/臂）

| | req/s | |
|---|---:|---|
| baseline | 21.530 ± 1.370 | |
| all7 | 22.879 ± 1.070 | |
| **kernel 增量** | **+6.26%** | **t=5.96, p=3.0e-08** |

分顺序看：

| 顺序 | baseline → all7 | 变化 | p |
|---|---|---:|---|
| 正序 | 22.462 → 23.358 | +3.99% | 8.3e-04 |
| 逆序 | 20.599 → 22.400 | **+8.75%** | 4.4e-09 |

**位置效应在这个配置下非常大**（baseline 正序 22.462 vs 逆序 20.599，差 9%），
所以单一顺序的数在这里毫无意义——这再次证明 2×2 设计是必需的，不是讲究。

### 5.4 结论

**kernel rewrite 的收益在真正的 serving ceiling 之上依然存在：+6.26%，p=3.0e-08。**

而且值得注意：它和在 cookbook 基线上的 +6.18% **几乎一样**。这说明七项改动消除的开销
和 `chunked_prefill_size` 消除的开销**基本正交**——尽管 `chunk=2048` 会让
`moesum` 的形状守卫（`prefill>=4096`）永不触发，`conv` 的守卫（`T>=2048`）也卡在边界上。

**所以 §4.3 担心的「守卫在调优配置上失效导致收益归零」没有发生**，
第一版看到的"归零"纯粹是 warmup 不足的假象。

### 5.5 还没做完的

- **装 tuned MoE config 的两格**（Bar 3 在 L1-tuned 之上）。
  按 regime C 在 cookbook 基线上的结果（config 值 +23%、kernel 增量随之升到 +9.73%），
  预期这里也会有正收益，但 `chunk=2048` 把每次 forward 的 M 压到 2048，
  落在 config 文件的 2048 桶而不是 4096 桶，**收益大小无法从已有数据外推**。
- 命令（约 40 分钟/2 格）：

```bash
SUITE=l1_ REGIME=C_long_prefill_tuned GPU=4 REPS=30 WARMUP=12 PORT=52145 \
    bash scripts/lfm_fusion/exp3_layered.sh
```

> `WARMUP` 已支持透传（默认不设，保持三个 cookbook regime 用标定好的表）。
> **非 cookbook 的 serving 配置必须显式设 `WARMUP`**，否则会得到 §5.2 那种欠 warmup 的数据。
> 上面这条命令会把四格都跑一遍；只补 config 两格的话，把 `run nocfg_*` 两行注释掉。

---

## 6. 复现

```bash
PY=~/.conda/envs/sglang-dev/bin/python
$PY scripts/lfm25_serving_ceiling_per_regime.py       # §2 §3 的表
```

| 产物 | 路径 |
|---|---|
| 各 regime ceiling | `results/2026-07-24_serving_ceiling_validation/analysis/lfm25/ceiling_per_regime.json` |
| §5 的两格原始数据 | `results/lfm_fusion/e2e/lfm25_exp3_l1_C_nocfg_{fwd,rev}/` |
| §5 的合并统计 | `results/lfm_fusion/e2e/exp3_l1tuned_nocfg_partial.json` |
| §5.2 欠 warmup 的第一版（留档） | `results/lfm_fusion/e2e/lfm25_exp3_l1v0_C_*/` |
