# 真实 workload 上 L3 的外部有效性：负载扫描 + 数据集扩展

**日期**：2026-08-07 · **模型**：LFM2.5-8B-A1B · **GPU**：H200 #3/#4
**sglang**：`0.5.12.post1 @ 17f7a1da1`（与既有 factorial 实验完全一致，未升级）

---

## 0. 要解决的问题

既有矩阵里有一行容易被误读：

> synthetic regime 上 +6~8%，唯一的真实 workload（Tool-Agent）吞吐只有 +0.4%
> → 「真实部署里价值有限」

**这个解读是错的**，但只靠文字解释说服力不够。本轮补两条互补的实验：

1. **同一条真实 trace 做到达率扫描** —— 证明收益从 latency 转化为 throughput
2. **增加不同类型的真实 workload** —— 证明不是只对 Mooncake Tool-Agent 成立

---

## 1. ★ 结论

**在 9 个真实/agentic workload 上，L3 让端到端延迟一致下降 6.3%–21.8%，全部统计显著。
吞吐增益只在客户端提供足够负载、请求开始排队时才出现。**

| workload | 类型 | 吞吐 | TTFT p50 | E2E mean | p(thr) | 正/逆序 |
|---|---|---:|---:|---:|---|---|
| Tool-Agent 1.0× | 真实 trace，**到达受限** | +0.50% | −8.42% | −6.26% | 9.4e-08 | +0.49/+0.51 |
| Tool-Agent 2.0× | 真实 trace | +0.31% | −6.74% | −7.99% | 1.1e-07 | +0.34/+0.28 |
| Tool-Agent 3.0× | 真实 trace，接近拐点 | +0.92% | −8.08% | −11.60% | 6.2e-18 | +0.92/+0.91 |
| **Tool-Agent 4.0×** | 真实 trace，**已饱和** | **+2.79%** | **−24.54%** | **−21.80%** | 7.1e-06 | +2.96/+2.63 |
| Conversation 2.0× | 真实 trace，生成密集 | +0.13% | −6.07% | −7.58% | 2.7e-03 | +0.14/+0.13 |
| Conversation 4.0× | 真实 trace，生成密集 | +1.26% | −8.15% | −9.60% | 4.2e-22 | +1.24/+1.28 |
| Mooncake arxiv 2.0× | Tool-Agent 的姊妹 trace | +0.86% | −7.70% | −8.25% | 5.5e-12 | +0.81/+0.91 |
| ShareGPT 8 req/s | **真实 prompt 文本** | +0.83% | −9.06% | −7.56% | 4.1e-15 | +0.82/+0.83 |
| ShareGPT 16 req/s | **真实 prompt 文本** | +1.61% | −19.69% | −8.91% | 2.2e-21 | +1.60/+1.61 |

**转化规律在每个数据集内部都独立成立**：
Tool-Agent 1×→4× 是 +0.50%→+2.79%，Conversation 2×→4× 是 +0.13%→+1.26%，
ShareGPT 8→16 req/s 是 +0.83%→+1.61%。

---

## 2. 负载扫描：吞吐扁平是负载假象，不是收益消失

Mooncake replay 按 trace 里的时间戳发请求（`--mooncake-slowdown-factor` 缩放时间轴，
**因子越小到达越快**；该模式下 `--request-rate` 被忽略，见 `bench_serving.py:3389`）。

| 负载 | offered | achieved | 在飞 | **峰值排队** | 吞吐增益 | TTFT p50 | E2E mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0× | 5.6 | 5.26 | 4.71 | 12 | +0.50% | −8.42% | −6.26% |
| 1.33× | 7.1 | 6.77 | 6.02 | 12 | +0.26% | −7.46% | −6.64% |
| 2.0× | 10.7 | 9.90 | 9.87 | 13 | +0.31% | −6.74% | −7.99% |
| 3.0× | 16.8 | 15.64 | 20.28 | 19 | +0.92% | −8.08% | −11.60% |
| **4.0×** | 22.2 | **18.52** | **55.50** | **85** | **+2.79%** | **−24.54%** | **−21.80%** |

**三条独立信号一致指向 4× 才是饱和点**：

1. achieved 到 3× 都紧跟 offered，4× 明显掉队（18.52 vs 22.2）
2. 在飞请求从 4.71 涨到 55.50
3. **服务器自己记录的排队深度**在 2× 前恒为 12–13，4× 突增到 85

> **机制**：固定的每请求节省，在到达受限的客户端下无处可去 ——
> 没有排队时，把请求处理得更快并不能退回更多请求，因为 trace 根本没给更多。
> 一旦开始排队，同样的节省就显现为吞吐；**而延迟收益还会变大**（−24.5%），
> 因为此时被消除的还包括排队延迟。

图：`results/2026-08-07_real_trace_study/toolagent_load_curve.png`

---

## 3. 数据集扩展：三种不同的「真实」

必须区分三者，不能混为一谈：

| 数据集 | 真实的是什么 | 不真实的是什么 |
|---|---|---|
| Mooncake Tool-Agent / Conversation | **到达时刻、token 数、prefix 复用** | prompt 文本（为隐私匿名化为 hash） |
| ShareGPT | **prompt 文本和长度分布** | 到达模式（我们指定的 Poisson） |
| Shared Prefix（既有） | 无 | 全部（synthetic 压力对照） |

### 3.1 ⚠️ 一个必须记录的负面发现：generic mooncake 不构成新数据集

计划把 `mooncake`（arxiv trace）当作第三个真实数据集。**实测它是 Tool-Agent 的姊妹 trace**：

```
记录数 23608 vs 23608     跨度 3537s vs 3537s
hash_ids p50 13 vs 13     output_length p50 30 vs 30
hash_id 词表重叠 86.4%    逐条完全相同的仅 4/23608
```

**记录本身不同，但分布几乎一致。** 把它当独立数据集会**虚增多样性**。
本报告保留它的数字（+0.86%，与 Tool-Agent 2× 的 +0.31% 同量级），
但**明确标为姊妹 trace，不计入 workload 多样性论据**。

### 3.2 真正的多样性来自 Conversation 和 ShareGPT

- **Conversation**：`output_length` p50 = **350**，而 Tool-Agent 是 **30** —— 生成量差 12 倍，
  是生成密集而非 prefill 密集的负载。
- **ShareGPT**：唯一有真实 prompt 文本的，补上 Mooncake 只有流量形状的短板。

---

## 4. 方法：核实过而非假设的三件事

### 4.1 pinned checkout 支持什么（Phase 0，未占 GPU）

计划引用的是 sglang **main** 的源码布局，**我们的树不一样**：

| 计划假设 | `17f7a1da1` 实际 |
|---|---|
| `python/sglang/benchmark/datasets/mooncake.py` | ❌ 不存在（main 的重构），功能都在单体 `bench_serving.py` |
| 四个 mooncake workload | ✅ 全部可用 |
| `--mooncake-slowdown-factor` | ✅ 真实 CLI 参数 |
| ShareGPT | ✅ 已支持 |
| `AgenticTraceDataset` | ❌ **完全不存在**，需要 backport |

**因此本轮不做 Agentic SWE** —— 它需要把 main 的 loader backport 进 pinned 树并改造
frozen replay，成本远高于其余三个，而三个已足以立住 external validity。
**没有为拿数据集而升级 runtime**，这是硬约束。

### 4.2 trace 够长，所以不需要 tiler

计划担心 x4 窗口太短（9.5s），设计了「deterministic trace tiler + hash_ids remap」。
**实测不需要**：

```
toolagent trace = 23608 条记录 / 3537 秒
我们之前只用了前 200 条（36 秒）
```

直接取更多 unique records 即可，**每档窗口都 ≥ 36 秒**：

| 负载 | slowdown | num-prompts | 窗口 |
|---|---|---|---|
| 1× | 1.0 | 200 | 36 s |
| 1.33× | 0.75 | 400 | 56 s |
| 2× | 0.5 | 400 | 38 s |
| 3× | 0.33 | 800 | 48 s |
| 4× | 0.25 | 800 | 36 s |

**好处不只是省事**：不重放同一批 hash_ids，就不会人为制造跨轮 prefix-cache 命中。

### 4.3 client cap 不是隐藏上限

把 client concurrency 从 64 提到 128，并在 1× 做对照：

```
cap=128: 5.268 req/s     cap=64: 5.261 req/s     差 0.13%
```

1× 不受影响，且 4× 峰值在飞 55.5 < 128，**cap 全程不 binding**。

### 4.4 顺带修掉一个 harness bug

ShareGPT 首次运行全部失败：`PermissionError: /data/hf/hub/datasets--anon8231489123--...`。

根因：`~/.bashrc` 设了 `HF_HUB_CACHE=/data/hf/hub`（共享只读），
**它的优先级高于 `HF_HOME`**，而 harness 只设了后者。
数据集其实已经在我们自己的 `.hf_cache` 里。

`serving_ceiling_lib.run_workload` 现在同时设置两者。
**影响面**：任何需要客户端拉取 HF 数据集的 workload 都会踩到。

---

## 5. 复现

```bash
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization; cd $REPO
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python

# Phase 0：trace 刻画（无 GPU）
$PY scripts/trace_characterize.py

# 负载扫描（每档双向）
for reg in RT_tool_agent_x1 RT_tool_agent_x133 RT_tool_agent_x2 \
           RT_tool_agent_x3 RT_tool_agent_x4; do
  for c in "fwd baseline,all7" "rev all7,baseline"; do set -- $c
    $PY scripts/lfm_fusion/lf_e2e.py --model lfm25 --regime $reg --gpu 3 \
        --port 52310 --arms "$2" --reps 8 --warmup 0 --tag "_rt_$1" \
        --correctness-nogate
  done
done

# 分析与作图
$PY scripts/lfm_fusion/rt_load_curve.py
$PY scripts/lfm_fusion/rt_plot_load_curve.py
$PY scripts/lfm_fusion/rt_workload_matrix.py
```

| 产物 | 路径 |
|---|---|
| trace 刻画 | `results/2026-08-07_real_trace_study/trace_characterization.json` |
| 负载扫描 | `.../toolagent_load_sweep.{json,csv}` |
| 负载曲线图 | `.../toolagent_load_curve.{png,svg}` |
| 跨数据集矩阵 | `.../real_workload_ablation.{json,csv}` |
| 逐次原始数据 | `results/lfm_fusion/e2e/lfm25_rt_{fwd,rev}/RT_*/` |

---

## 6. 给 mentor 的一页话

**旧表述（容易被误读）**：

> synthetic 上 +6~8%，真实 Tool-Agent trace 上只有 +0.4%。

**新表述（有数据支撑）**：

> 在 9 个真实与 agentic workload 上 —— 涵盖两条真实到达 trace、真实 prompt 文本、
> 以及从到达受限到饱和的负载区间 —— **L3 一致降低每请求延迟 6.3%–21.8%**。
> **当负载接近饱和，同一份节省显现为服务器吞吐**：Tool-Agent 上从 +0.50%（1×）
> 升到 **+2.79%（4×）**，Conversation 从 +0.13% 升到 +1.26%，
> ShareGPT 从 +0.83% 升到 +1.61%。
>
> Tool-Agent 1× 上吞吐扁平**不是收益消失，是该负载下服务器 93% 时间在空等 trace**
> —— 服务器自己记录的排队深度为 12，而 4× 时是 85。

---

## 7. 仍然开放

1. **Agentic SWE / OpenHands（多轮 session）** —— 需要 backport `AgenticTraceDataset`
   并实现 frozen replay（否则 baseline 和 L3 的后续轮次输入会分叉，
   性能差异就不再是干净的 kernel 因果效应）。本轮未做。
2. **完整 2³ factorial 在新 workload 上** —— 加 L1/L2 前必须先查
   `#new-token` 直方图确认 M-bucket 分布，否则 L2 只能标为 transferred 而非 best。
3. ~~ShareGPT rate8 逆序~~ —— **已完成并计入**（+0.82/+0.83）。
