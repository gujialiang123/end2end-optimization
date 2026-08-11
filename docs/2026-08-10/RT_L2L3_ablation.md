# 新真实/agentic workload 上的 L2 / L3 / L2+L3 消融

**日期**：2026-08-10（夜间，02:40 收尾）· **模型**：LFM2.5-8B-A1B · TP1 · bf16
**GPU**：H200 #0–#3（四卡并行）· **sglang**：`0.5.12.post1 @ 17f7a1da1`（与既有 factorial 完全一致，未升级）
**任务**：在 2026-08-07 外部有效性研究新增的真实/agentic workload 上，补跑 **L2 / L3 / L2+L3** 三臂消融（相对 S0）。

---

## 0. 为什么是 L2/L3/L2+L3，而不是完整 2³

8-07 的报告只在新 workload 上测了 **S0 vs L3**。这轮把缺的 **L2** 和 **L2+L3** 补上，
形成固定 serving 下的 **2² 消融**（`{MoE config 关/开} × {kernel rewrite 关/开}`）。

**刻意跳过 L1**，原因（见 `docs/2026-08-04/METHODOLOGY_three_layer_optimization.md`）：

| 层 | 迁移到新 workload 的成本 | 本轮 |
|---|---|---|
| **L1** serving config | 每个 workload 必须**重跑 192 全网格穷举**（~6h/workload），最优 config 与输入形状绑定 | ❌ 不做 |
| **L2** MoE kernel config | 机制可复用（JSON 查表 + 环境变量），但 **config 是否有效取决于新 workload 的 M-bucket 分布** | ✅ |
| **L3** kernel rewrite | 补丁通用，只需重跑正确性 gate + 对比测量，接近免费 | ✅ |

所有 workload 都用 **cookbook serving 旋钮**（`cap=32, chunk=-1, lpm, mem=0.85`），
两臂共享同一 `SGLANG_MOE_CONFIG_DIR`，所以 config 是基线属性、不是臂间差异。
每格 `{正序, 逆序}` 双向对照，合并 **n=16/臂**。驱动：`scripts/lfm_fusion/exp3_layered.sh`。

映射：`S0 = nocfg/baseline`，`L2 = cfg/baseline`，`L3 = nocfg/all7`，`L2+L3 = cfg/all7`。

---

## 1. ★ 两个结论

**(1) L3 在全部 9 个真实 workload 上复现**，S0→L3 的吞吐/延迟与 8-07 独立测量一致
（如 tool_agent x1 L3 吞吐 +0.47% vs 8-07 的 +0.50%）。

**(2) L2 不能跨 workload 迁移，而 L3 可以。** 这是本轮最有价值的发现：
L2（为 Mooncake 大 prefill 的 M≥4000 桶调的 MoE config）在 **Mooncake 家族**上砍 TTFT 27–48%，
但在 **ShareGPT**（真实 prompt，长度分布不同）上**零效果甚至变差**（TTFT +1%～+18%，不显著或更差）。
**同一份 L2 config，命中桶就赢、不命中就废** —— 正是 8-04 §4.2 预警的 M-bucket 依赖，第一次在跨 workload 上被直接测到。

---

## 2. 完整矩阵（相对 S0，orders pooled，n=16/臂）

粗体为统计显著（p<0.05）。延迟为负=更好。

| Workload | arm | throughput | TTFT p50 | E2E mean |
|---|---|---:|---:|---:|
| Tool-Agent 1.0× | L2 | +0.11% n.s. | **−28.77%** | **−12.16%** |
| | L3 | **+0.47%** | **−6.10%** | **−7.21%** |
| | L2+L3 | **+0.53%** | **−35.71%** | **−17.56%** |
| Tool-Agent 2.0× | L2 | **+0.27%** | **−29.35%** | **−12.10%** |
| | L3 | **+0.38%** | **−8.02%** | **−8.41%** |
| | L2+L3 | **+0.60%** | **−34.78%** | **−19.10%** |
| Tool-Agent 3.0× | L2 | **+1.49%** | **−29.25%** | **−22.99%** |
| | L3 | **+0.85%** | **−6.79%** | **−10.94%** |
| | L2+L3 | **+2.46%** | **−36.36%** | **−30.77%** |
| **Tool-Agent 4.0×**（饱和） | L2 | **+5.23%** | **−48.46%** | **−42.28%** |
| | L3 | **+3.07%** | **−25.58%** | **−22.18%** |
| | **L2+L3** | **+9.15%** | **−57.96%** | **−55.86%** |
| Conversation 2.0× | L2 | +0.09% | **−27.21%** | **−7.07%** |
| | L3 | +0.11% | **−6.94%** | **−8.03%** |
| | L2+L3 | **+0.31%** | **−32.67%** | **−15.13%** |
| Conversation 4.0× | L2 | **+0.84%** | **−26.52%** | **−15.22%** |
| | L3 | **+1.30%** | **−6.97%** | **−10.08%** |
| | L2+L3 | **+2.12%** | **−30.18%** | **−22.29%** |
| Mooncake arxiv 2.0× | L2 | **+0.22%** | **−29.03%** | **−12.32%** |
| | L3 | **+0.82%** | **−7.62%** | **−7.96%** |
| | L2+L3 | **+1.21%** | **−36.03%** | **−19.80%** |
| ShareGPT 8 req/s | L2 | +0.04% n.s. | +1.23% n.s. | −0.11% n.s. |
| | L3 | **+0.82%** | **−6.70%** | **−7.40%** |
| | L2+L3 | **+0.88%** | **−8.20%** | **−8.08%** |
| ShareGPT 16 req/s | L2 | **−0.48%** | +18.42% n.s. | +4.83% n.s. |
| | L3 | **+1.38%** | −12.89% n.s. | **−7.64%** |
| | L2+L3 | **+1.36%** | **−14.98%** | **−8.10%** |

数据：`results/2026-08-10_rt_l2l3/rt_l2l3_ablation.{json,csv,md}`。

---

## 3. L2 的 workload 依赖（本轮头条）

只看 **L2 单独的 TTFT p50**，按数据集家族分开就一目了然：

| Workload | L2 TTFT p50 | p | 命中调优桶？ |
|---|---:|---|---|
| Tool-Agent 1×/2×/3×/4× | −28.8 / −29.4 / −29.3 / **−48.5%** | <1e-3 | ✅ 大 prefill 落 M≥4000 |
| Conversation 2×/4× | −27.2 / −26.5% | <1e-25 | ✅ |
| Mooncake arxiv 2× | −29.0% | 5e-37 | ✅ |
| **ShareGPT 8 / 16 req/s** | **+1.2% / +18.4%** | 0.66 / 0.074 | ❌ **未命中** |

**机制**：cookbook serving 用 `chunk=-1`（不分块预填充），所以 Mooncake 家族的整段 prompt
一次进 prefill → batch 的 M 落在为它调优的 M≥4000 桶 → L2 大幅削 TTFT。
ShareGPT 的真实 prompt 长度分布不同，M 落到 config 未覆盖/未调优的桶 → L2 无效甚至反噬。

> **交付含义**：报告里必须把 L2 标为 **transferred config（迁移配置）**，不是 per-workload optimum。
> 「L2 在真实 workload 上也有效」这句话**只对 prefill 形状接近调优分布的 workload 成立**。
> L3 没有这个限制 —— 它在 9 个 workload 上一致有效。

---

## 4. 负载依赖的转化（Tool-Agent x1→x4）

同一条真实 trace，随到达率上升，**L2 和 L3 的延迟收益逐步转化为吞吐**：

| 负载 | L2+L3 throughput | L2+L3 E2E mean |
|---|---:|---:|
| 1.0×（到达受限） | +0.53% | −17.56% |
| 2.0× | +0.60% | −19.10% |
| 3.0×（近拐点） | +2.46% | −30.77% |
| **4.0×（饱和）** | **+9.15%** | **−55.86%** |

低负载下吞吐扁平不是收益消失，是服务器在空等 trace；一旦排队，节省显现为吞吐。
这条曲线同时覆盖 L2 和 L3，比 8-07 只有 L3 的版本更完整。

---

## 5. L2 与 L3 近似可加

吞吐维度，预测 `(1+L2)(1+L3)−1` vs 实测 L2+L3：

| Workload | L2 | L3 | 预测 | 实测 |
|---|---:|---:|---:|---:|
| Tool-Agent 3× | +1.49% | +0.85% | +2.36% | +2.46% |
| **Tool-Agent 4×** | +5.23% | +3.07% | +8.46% | **+9.15%** |
| Mooncake arxiv 2× | +0.22% | +0.82% | +1.04% | +1.21% |
| ShareGPT 8 | +0.04% | +0.82% | +0.86% | +0.88% |

预测与实测差不到 1 个点；饱和点（4×）略**超**可加，是排队缓解的二阶效应。
说明两层削的是不同的成本（L2 削 prefill kernel、L3 削 elementwise/per-forward），互不侵蚀。

---

## 6. 方法与踩坑

- **每格双向对照**，合并 n=16；harness 顺序执行臂有位置效应（8-03 曾因此差点报出反号）。
- **两次 `launch_failed` 单臂重跑**：tool_agent_x4 的 cfg_fwd/baseline、sharegpt_rate16 的 nocfg_fwd/baseline
  各有一次 server 启动失败（已知的泄漏 server / 端口占用问题），用 GPU 空档单独补跑该格，
  最终全部 36 格 × 9 workload 均为干净的 16 ok samples。
- **GPU3 空档补跑** tool_agent x2/x3，把 tool_agent 负载扫描补成 x1/x2/x3/x4 完整四点。
- 吞吐在到达受限 workload 上是错口径，全程同时报 TTFT/E2E。

---

## 7. 复现

```bash
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization; cd $REPO
# 每个 workload 跑 4 格（S0/L2/L3/L2+L3 x 双向），四卡并行
bash scripts/lfm_fusion/rt_l2l3_matrix.sh 0 52200 RT_tool_agent_x1 RT_tool_agent_x4
bash scripts/lfm_fusion/rt_l2l3_matrix.sh 1 52210 RT_conversation_x2 RT_conversation_x4
bash scripts/lfm_fusion/rt_l2l3_matrix.sh 2 52220 RT_mooncake_generic_x2 RT_sharegpt_rate16
bash scripts/lfm_fusion/rt_l2l3_matrix.sh 3 52230 RT_sharegpt_rate8 RT_tool_agent_x2 RT_tool_agent_x3
# 汇总成矩阵
python scripts/lfm_fusion/rt_l2l3_consolidate.py
```

| 产物 | 路径 |
|---|---|
| 消融矩阵（JSON/CSV/MD） | `results/2026-08-10_rt_l2l3/rt_l2l3_ablation.{json,csv,md}` |
| 每 workload 吞吐汇总 | `results/lfm_fusion/e2e/exp3_layered_RT_*_summary.json` |
| 逐格原始 | `results/lfm_fusion/e2e/lfm25_exp3_RT_{nocfg,cfg}_{fwd,rev}/RT_*/` |
| 运行日志 | `results/2026-08-10_rt_l2l3/*.log` |

---

## 8. 仍然开放

1. **完整 2³（含 L1）** —— 需为每个新 workload 重跑 192 网格（~6h/workload），本轮刻意不做。
2. **给 ShareGPT 重扫 L2** —— 若要 ShareGPT 上真实的 L2 ceiling，需按其 prefill M 分布重扫 tile；
   本轮明确保留其为 transferred config 的负面结果（这本身是证据）。
3. **Agentic SWE / OpenHands** —— 仍需 backport `AgenticTraceDataset` + frozen replay（延续 8-07 的待办）。
