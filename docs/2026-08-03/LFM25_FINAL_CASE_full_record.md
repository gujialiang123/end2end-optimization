# LFM2.5-8B-A1B 端到端优化全记录 —— 候选 final case

**写于**：2026-08-03
**目的**：把 LFM2.5 上做过的**三个优化层次**（serving config tuning → kernel config tuning → kernel rewrite/fusion）
整理成一份自洽的记录，逐层说明**具体改了什么代码、为什么这么改、在哪个 regime 有效、提升多少**。

**这份文档只写 LFM2.5。** 其他模型（Gemma-3、OLMo-2、Qwen3-30B）只在需要对照时出现。

> ⚠️ **读之前必须知道的一件事**：三层**都各自测过、都有显著提升**，但 L2 和 L3
> 长期只是从同一个 cookbook 基线出发的**两条平行分支**。
> **【2026-08-03 更新】L2+L3 的串联已经补测**（§9.5）：L3 叠在 L2 之上仍有
> **+9.73%（p=9.5e-19）**，整栈 **+35.25%**。剩下的缺口是 L1 也串进来（§10 实验 5）。

---

## 0. 摘要表

| 层 | 改的是什么 | 主收益 regime | 提升 | 统计 | 状态 |
|---|---|---|---|---|---|
| **L0** 基线 | sglang cookbook 默认 | — | — | — | 参照点 |
| **L1a** serving config tuning<br>**（断崖 regime，2/6）** | 4 个 serving 旋钮 | 长 prefill / shared-prefix | **+56.9% / +93.6%** | 192 全网格 + 5 rep 验证 | ✅ 有结果。**延迟同时改善**（TTFT p95 −54.9% / −94.8%），TPOT 退 4~9% |
| **L1b** serving config tuning<br>**（plateau regime，4/6）** | 同上 | 短 decode / 并发 decode / medium / tool-agent | **+0.3% ~ +1.8%** | 同上 + 100 trial 无热启动收敛研究 | ✅ **plateau（实质零收益）** |
| **L2** kernel config tuning | fused-MoE Triton kernel 的 tile 参数 | 长 prefill | **+22.1% ~ +23.3%** | 8/8 分布不重叠, p=1.3e-10 | ✅ 有结果 |
| **L3** kernel rewrite/fusion | 7 处代码（含 4 个手写 Triton kernel） | 三个 regime 全部 | **+5.30% ~ +6.57%** | p=4.6e-14 / 2.4e-08 / 1.2e-05 | ✅ 有结果 |
| **L2+L3 串联** | L2 在下、L3 叠在其上 | 长 prefill | **+9.73%**（L3 在 L2 之上）<br>整栈 **+35.25%** | counterbalanced n=16/臂, p=9.5e-19 | ✅ **2026-08-03 补测，见 §9.5** |

> **L1 为什么占两行**：它**不是两个实验**，是**同一个实验（192 全网格 × 6 regime）产出的两种相反结论**。
> 4/6 regime 上 serving 旋钮已经到顶（+0.3~1.8%），2/6 上还有巨大空间（+56.9% / +93.6%）。
> **把它们平均成一个数字会同时抹掉两个结论**，所以分开列。
>
> ⚠️ **2026-08-03 修正**：本表原先给 L1a 标了「TRADE-OFF」。**那是错的，已撤回。**
> 该标签引用的 −221% TPOT 来自 **n=1 coverage pass 的另一个配置**，
> 而那个配置**没通过 n=5 验证**。验证后的 ceiling 上，长 prefill 的 TTFT p95 从 208.5ms
> **降到 94.0ms**、shared-prefix 从 **7450ms 降到 389ms**、tool-agent 三项指标全部改善。
> 详见 `LFM25_ablation_matrix_EN.md` §1.1。

---

## 1. 为什么 LFM2.5 适合做 final case

1. **三个层次都有实测数据**，而且都在同一个模型、同一张卡、同一个 sglang commit 上
2. **三个层次共用同一套 workload 定义**（`scripts/serving_ceiling_lib.py` 的 `WORKLOADS`），可以直接拼图
3. **两种结论都有**：serving tuning 在 3/6 regime 是 plateau（证明 autotuning 有上限），在 2/6 regime 有巨大收益（证明不能一概而论）
4. **有真手写 kernel**（4 个 Triton kernel），不只是调参
5. **有诚实负面**（`gate+idx` 三 regime 全不显著）
6. **有方法学产出**（次可加性规律、regime→backend 规则不可迁移）

### 模型本身的特点（决定了机会在哪）

LFM2.5-8B-A1B 是**混合架构**：24 层里 **18 层是 gated short convolution**，只有 6 层是全注意力。MoE 是 top-4/32。

`docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md` §3.4，长 prefill 下的 kernel 时间构成：

| bucket | LFM2.5 | Qwen3-30B（对照） |
|---|---:|---:|
| MoE | 70.8% | 54.2% |
| **注意力** | **2.8%** | **21.6%** |
| short conv | 0.7% | — |
| dense GEMM | 12.5% | 16.1% |
| **norm + elementwise** | **12.8%** | **5.6%** |

**这张表决定了一切**：
- 新架构确实兑现了承诺——注意力 + 替代它的 conv 一共只有 **3.5%**（Qwen 是 21.6%）
- 但它把这个优势的一部分**以 12.8% 的未融合胶水交还回去**（Qwen 只有 5.6%）
- **空缺不在新算子里**（`causal_conv1d` 本身只占 0.7%，很快），**在周围的调用点胶水上**

---

## 2. 固定实验框架

所有实验共用：

```
模型      /data/hf/LFM2.5-8B-A1B   (bf16, TP=1)
硬件      1× NVIDIA H200, driver 580.105.08
软件      sglang 0.5.12.post1 @ 17f7a1da1
          torch 2.9.1+cu128, Triton 3.5.1, CUDA 12.8
conda     sglang-dev
```

### Regime（workload）定义 —— `scripts/serving_ceiling_lib.py`

| 代号 | 名称 | 参数 | cookbook 基线 |
|---|---|---|---|
| A | `R_short_decode` | in=100, out=256, conc=1, n=1 | 1.679 req/s |
| — | `R_medium_balanced` | — | 7.126 req/s |
| C | `R_long_prefill` | **in=4000, out=32, conc=4, n=4** | 12.87 req/s ¹ |
| B | `R_concurrent_decode` | in=200, out=256, conc=32 | 21.98 req/s |
| — | `shared_prefix` | — | 14.19 req/s |
| — | `tool_agent` | mooncake 真实 trace | 5.25 req/s |

¹ 07-24 campaign 测得 12.87；kernel 实验（07-26/07-27）测得 **12.25–12.28**。
两者相差约 5%，原因是 warm-up 预算不同（`R_long_prefill` 每次只跑 ~0.3 s，首尾漂移可达 36.5%，
07-24 campaign 给了 4 次 warm-up）。**跨 campaign 不能直接比绝对值，只能比 ratio。**

### cookbook 基线配置

**完整启动命令**（`scripts/serving_ceiling_lib.py:202-211` 构造，L2/L3 实验共用）：

```bash
python -m sglang.launch_server \
    --model-path /data/hf/LFM2.5-8B-A1B \
    --served-model-name lfm2.5-8b-a1b \
    --host 127.0.0.1 --port <PORT> \
    --tensor-parallel-size 1 \
    --context-length 8192 \
    --schedule-conservativeness 1.0 \
    --trust-remote-code \
    --moe-runner-backend auto \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --chunked-prefill-size -1 \
    --schedule-policy lpm \
    --max-prefill-tokens 16384
```

**★ CUDA graph 是开启的。** 这一点从**真实 server log 逐条核实**过，不是从代码推断的
（`results/lfm_fusion/e2e/lfm25_exp3_cfg_fwd/C_long_prefill/server_baseline.log`）：

```
disable_cuda_graph        = False
disable_cuda_graph_padding= False
cuda_graph_max_bs         = 256
Capture cuda graph begin. ...
Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32]
Capture cuda graph end. Time elapsed: 1.91 s.
```

捕获的 batch size 到 32 为止，正好等于 `max_running_requests`。
**所以 decode 路径全程走 graph 重放，baseline 和 all7 两臂都是。**

其余相关的 resolved server args（同一份 log）：

| 参数 | 值 | 含义 |
|---|---|---|
| `disable_cuda_graph` | **False** | CUDA graph 开 |
| `enable_torch_compile` | **False** | 没有用 torch.compile |
| `enable_piecewise_cuda_graph` | **False** | — |
| `disable_radix_cache` | **False** | radix cache 开 |
| `disable_overlap_schedule` | **False** | overlap 调度开 |
| `enable_fused_qk_norm_rope` | **False** | ★ 上游现在有这个 server flag，**默认关**，见下 |
| `attention_backend` | `fa3` | |
| `moe_runner_backend` | `auto` | |
| `dtype` / `kv_cache_dtype` | `auto` / `auto` | bf16 |
| `page_size` | 1 | |
| `speculative_algorithm` | `None` | 无投机解码 |
| `quantization` | `None` | 无量化 |

> **`enable_fused_qk_norm_rope=False` 值得注意**：上游后来加了这个 server 级开关。
> 我们的 G4 是在**模型调用点**接上融合 kernel，与这个 flag 是两条路径。
> baseline 和 all7 两臂这个 flag 都是 False，所以 A/B 是干净的——
> 但**上游可能已有另一种方式解决 G4**，交付时应主动说明。

**A/B 的干净性**：`LFM_FUSION_PATCH` 未设置时走的是**逐字未改动的 sglang 原路径**，
同一棵树、同一份 server 参数、同一个 commit（`17f7a1da1a`）。
server log 会被检查 patch 生效标记，否则静默失效的 patch 会被误记为"与 baseline 相同"。

---

## 3. L1 —— Serving Config Tuning

### 3.1 做了什么

调 4 个 serving 旋钮，**不碰任何 kernel**：

| 旋钮 | 取值 |
|---|---|
| `max_running_requests` | 8, 16, 24, 32, 48, 64, 96, 128 |
| `chunked_prefill_size` | -1, 2048, 8192 |
| `schedule_policy` | lpm, fcfs |
| `mem_fraction_static` | 0.75, 0.80, 0.85, 0.90 |

→ 192 个组合。

做了**两个独立研究**，互相印证：

**研究 A：全网格穷举**（`results/2026-07-24_serving_ceiling/`）
- 192 配置 × 6 regime × 2 模型 = 384 任务，**0 未解决失败**
- 2304 次 per-run 测量，148,992 条 per-request 记录
- 再做 5 重复验证 pass（62 配置 × 5 rep）

**研究 B：Optuna TPE 无热启动收敛研究**（`results/2026-07-22_lfm25_plateau_100/`）
- 100 个唯一完成 trial，**0 失败**，26 个重复被剪枝
- 关键设计：**没有任何 `enqueue_trial` 热启动**（此前 v3 研究因为塞了 4 个 cookbook 等价配置而作废）
- MoE backend 固定 triton、attention 固定 fa3、CUDA graph 恒开——**只调 serving 旋钮**，不混淆 kernel 路径

### 3.2 结果 —— 分裂成两种截然不同的结论

**研究 A（全网格，6 个 regime）：**

| regime | 最优旋钮 | 吞吐提升 | 最差配置 | 分类 |
|---|---|---:|---:|:--:|
| short decode | cap96 · chunk8192 · fcfs · mem0.90 | +0.4% | −1.9% | WIN |
| medium balanced | cap16 · chunk2048 · lpm · mem0.75 | +2.6% | −7.0% | REGRESSION |
| **long prefill** | **cap24 · chunk2048 · fcfs · mem0.75** | **+77.5%** | −19.2% | WIN |
| concurrent decode | cap48 · chunk−1 · fcfs · mem0.75 | +1.6% | **−64.9%** | WIN |
| **shared-prefix** | cap96 · chunk2048 · lpm · mem0.75 | **+94.1%** | −53.9% | TRADE-OFF |
| tool-agent | cap48 · chunk2048 · lpm · mem0.80 | +0.4% | −1.1% | REGRESSION |

5 重复验证 pass 定量收缩但**定性结论完全一致**：

| regime | 验证后提升 | 验证 pass 的最优旋钮 | 分类 |
|---|---:|---|:--:|
| short decode | +0.4% | — | WIN |
| medium balanced | +1.8% | — | REGRESSION |
| **long prefill** | **+56.9%** | **cap8 · chunk2048 · fcfs · mem0.90** | TRADE-OFF |
| concurrent decode | +1.1% | cap64 · chunk8192 · fcfs · mem0.75 | FLAT |
| **shared-prefix** | **+93.6%** | — | TRADE-OFF |
| tool-agent | +0.3% | — | WIN |

> ⚠️ **两个 pass 的最优旋钮不同，别混用。**
> coverage pass（1 rep，192 配置）的长 prefill 赢家是 `cap24 · chunk2048 · fcfs · mem0.75`（+77.5%）；
> **验证 pass（5 rep，CI-backed，35 配置）的赢家是 `cap8 · chunk2048 · fcfs · mem0.90`**
> （12.604 → **19.781 req/s**，+56.94%，ci95 ±0.295）。
> **要串联进 waterfall 的必须用验证 pass 的那个**——它是有重复、有 CI 的。
> 逐 regime 的完整数据：`results/2026-07-24_serving_ceiling_validation/analysis/lfm25/ceiling_per_regime.json`
>
> 两个 pass 唯一共同的结构性结论是 `chunked_prefill_size=2048` + `fcfs`；
> `cap` 和 `mem` 在两次之间就翻转了，**这本身又是一条"单次排名不可信"的证据**（与 §3.2 研究 B 的发现一致）。

**研究 B（TPE 收敛，只做 concurrent decode）：**

- cookbook 基线（5 次独立测量，从未进入 Optuna）：**19.49 ± 0.59 req/s**
- best-so-far 在 **第 7 个配置**就到达最终最优的 1% 以内
- best-through 10/20/50/75/100 = 19.89 / 19.89 / 19.89 / 19.98 / 19.98 req/s
- **最后 20 个配置对 best-so-far 的改进 = 0.0%**
- 事后 5 重复交错验证：最优 trial_41 = 19.80 vs cookbook 19.72 → **+0.4%，CI 重叠**
- **排名不稳定**：原始单次排名 `[50,30,7,41,26]` ≠ 验证排名 `[41,26,7,30,50]`，
  单次"最优"的 trial_50 在复测中**掉到最后一名**

### 3.3 这一层的三条结论

**① 3/6 regime 是真 plateau。**
`short decode`、`concurrent decode`、`tool agent` 的天花板只有 **+0.2% ~ +1.6%**。
concurrent decode 上 **80% 的配置被 cookbook 支配**。两个独立方法（穷举 + TPE）得到同一结论。

**② 2/6 regime 有真断崖，而且是"容量断崖"。**
`long_prefill` (+77.5%) 和 `shared_prefix` (+94.1%) 的赢家**都**把 `max_running_requests` 提到 cookbook 的 32 以上并开启 chunking。
**这是多旋钮联合效应，绝不能归因于 chunked prefill 单独一项。**

**③ 下行风险比上行收益大一个数量级。**

| regime | 最好 | 最坏 |
|---|---:|---:|
| concurrent decode | +1.6% | **−64.9%** |
| shared-prefix | +94.1% | −53.9% |

主要断崖驱动因素是 `max_running_requests = 8`（batching 被饿死）。
**→ serving 旋钮是"避坑杠杆"，不是"提速杠杆"。**

**④ 单目标优化会毁掉别的指标。**
`tool_agent`（唯一的真实 trace）上，吞吐赢家只买到 +0.4%（噪声内），
却把 **TPOT p95 恶化了 221%**。

---

## 4. L2 —— Kernel Config Tuning（fused-MoE Triton kernel）

### 4.1 做了什么

**不写任何 kernel 源码**，只改 sglang 那个 `fused_moe_kernel` 的**配置**：

```
BLOCK_SIZE_M / BLOCK_SIZE_N / BLOCK_SIZE_K / GROUP_SIZE_M / num_warps / num_stages
```

LFM2.5 的 MoE shape 是 **`E=32, N=1792`**。sglang 上游 PR #22791 已经为 LFM2 的 MoE 做过这件事，
覆盖 **H100 / B200 / MI325X —— 唯独没有 H200**。所以在 H200 上，较大的 prefill shape 全部落到
**两档启发式**（上游自己在日志里打 `Performance might be sub-optimal!`）。

工作内容：
- 每个 token-count 桶扫 **468–894 个候选**（`warmup=25, iters=100, repeats=5`）
- 桶集合与上游 H100/B200 文件**完全对齐**（19 个桶）。原研究只有 14 个，
  缺 `24, 48, 96, 1536, 3072` —— 这 5 个是为这个 PR 新扫的
- **每个候选都先过正确性门禁再计时**：与默认 kernel 输出比对（BF16 容差 + NaN/Inf 检查），
  失败直接丢弃不计时。**~9000 个 benchmark 配置，0 次正确性失败**

### 4.2 三次迭代才得到正确结果 —— 过程本身是产出

| 迭代 | 策略 | 低批 decode | 并发 decode | 长 prefill |
|---|---|---:|---:|---:|
| 1 | naive 每 regime 特化（在无 bias 变体上 tune） | **0.745×** | 1.007× | 1.183× |
| 2 | 在 server 真正执行的 with-bias 变体上重 tune | 0.879× | **1.061×** | 1.188× |
| 3 | **guarded**（只在 oracle 证明有 headroom 处特化） | **1.0015×** | 1.005× | **1.221×** |

**三个必须修正的错误**（最可迁移的发现）：

1. **我们在 tune 一个 server 从不执行的 kernel 变体**（expert bias 的有无）
2. **CUDA graph 捕获会重放 decode**，config 在**捕获时**就被烘焙进去了 → decode 路径事后改 config 无效
3. **`M` 是 token 数，不是 `tokens × top_k`** —— profile 的键错了一个 `top_k` 因子，
   真实 headroom 被藏在错位的桶后面。**只有活体 trace 能暴露这一点。**

### 4.3 最终结果

**guarded 策略**（`M ≤ 32` 全部与默认逐字段相同 → decode 行为不变；只特化 prefill 桶）：

| regime | 默认 (req/s) | guarded | 重复次数 |
|---|---:|---:|---:|
| A 低批 decode | 1.686 ± 0.004 | **1.0015×**（中性） | 5 |
| B 并发 decode | 21.997 ± 0.075 | 1.005×（中位 1.003×） | 8 |
| **C 长 prefill** | 12.254 ± 0.106 | **1.221×**（中位 1.223×） | 8 |

**长 prefill 8/8 次重复落在 14.63–15.19 req/s，基线是 11.91–12.40 —— 分布完全不重叠。**

后续的 PR 草稿（`docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md`）用更干净的协议复测：

| regime | default | 本 PR config | 变化 | p |
|---|---:|---:|---:|---|
| **C 长 prefill** | 12.277 req/s | **15.142** | **+23.34%** | 1.3e-10 |
| A 低批 decode | 1.6847 | 1.6825 | −0.13% | 0.079 **中性** |

microbenchmark 层面，特化桶的加速：

| M | 加速 |
|---:|---:|
| 1536 | **1.562×** |
| 3072 | **1.626×** |

### 4.4 一个方法学插曲：decode 那栏做了顺序对照

第一次测 decode 得到 **−0.37%, p=4.9e-04** —— 统计显著的小回归。
但 `rk_e2e.py` 是**顺序执行** arm（先 8 次 default，再 8 次 candidate），所以把顺序反过来重跑：

| 顺序 | ratio |
|---|---:|
| default 先 | 0.9963（**−0.37%**）|
| candidate 先 | 1.0012（**+0.12%**）|

**符号翻转，且两次都是"先跑的那个更快"** → 这是位置效应，不是 config 效应。
counterbalanced 合并（每臂 n=16）后：**−0.13%, p=0.079，不显著。**

### 4.5 一个负面发现：regime→backend 规则跨模型不可迁移

同协议在 LFM2.5 和 Qwen3-30B 上跑 3 regime × 4 backend × 5 重复（60 次运行，0 失败）：

| backend | A 低批 decode<br>LFM / Qwen | B 并发 decode<br>LFM / Qwen | C 长 prefill<br>LFM / Qwen |
|---|---:|---:|---:|
| `triton` | 0.999 / 1.001 | 1.006 / **1.033** | 1.004 / 0.987 |
| `triton_kernel` | **0.650 / 0.641** | 0.966 / 1.008 | 0.996 / **0.647** |
| `flashinfer_cutlass` | 0.965 / 0.934 | **1.017 / 1.047** | **0.664** / **1.027** |

**长 prefill 完全反转**：LFM 上 `cutlass` 最差（0.664×），Qwen 上 `cutlass` 最好（1.027×）。
把 Qwen 的规则用到 LFM → **−34%**。

> **静态 regime→backend 查找表不只是不完整，而是有害的。**
> 这是"必须按每个部署实测（即需要 agent）"的最直接论据。

---

## 5. L3 —— Kernel Rewrite / Fusion（7 处改动）

### 5.1 怎么找到这些机会的

**方法：数"融合实现根本不会执行的 kernel"的个数，并拿 Qwen 做对照。**

`bench_one_batch --profile` + **关闭 CUDA graph**（让每个算子单独现形）→ 按 kernel 名分桶。
脚本 `scripts/lfm_fusion/lf_audit.py`。

每次 forward 的 kernel 启动次数：

| 模型 | 未融合 RMSNorm | 独立 residual add | gating mul | layout copy |
|---|---:|---:|---:|---:|
| **LFM2.5** | **61** | **48** | **36** | 22–53 |
| Qwen3-30B（对照） | **1** | **0** | **0** | 4–52 |

**计数是结构性的，不是约数**：
- `48 = 2 个 residual add × 24 层`
- `36 = 2 个 gating mul × 18 个 conv 层`

**对照组是决定性的**：Qwen 一整个 forward 只有 1 个未融合 norm、0 个独立 add。
→ **这不是 sglang 的通病，是这个模型文件的实现漏了。**

之后两个子 agent 做了深度调查：
- **nsys 时间线**（`results/lfm_fusion/nsys/FINDINGS.md`）：5 个候选按"收益/风险"排序 + 2 个明确否决
- **FX / Inductor 图挖掘**（`results/lfm_fusion/fx/FINDINGS.md`）：独立验证 + 机制修正

### 5.2 注入方式（方法学，值得单独说）

所有改动通过 **`LFM_FUSION_PATCH` 环境变量 opt-in**，不设变量时走的是**逐字未改动的 sglang 原路径**。
→ **A/B 的 baseline 是真 baseline，同一棵树、同一份 server 参数。**

**踩过的坑**：模型类被 model registry **懒加载**，`sitecustomize` 执行时 `lfm2_moe` 还没导入，
用定时器打 patch 是**竞态**。改用 `sys.meta_path` finder，在该模块 exec 完成的**瞬间**打补丁
（`lf_inject/sitecustomize.py`）。

**另外**：server log 会被检查 patch 生效标记 —— 否则一个静默失效的 patch 会被当成"与 baseline 相同"记录下来。

---

### 5.3 七项改动逐个详解

改动分两类：

| 类 | 项 | 说明 |
|---|---|---|
| **写新 kernel** | G3 `conv`(2 个)、G6 `gate`(1 个)、G7 `moesum`(1 个) | **手写 Triton，共 301 行** |
| **只改接线** | G1 `norm`、G4 `qkrope`、G2 `scale`、G5 `idx` | 融合能力**早就有**，调用点没用上 |

---

#### G1 `norm` —— residual 加法从未被融合（接线）

**问题代码**（`sglang/srt/models/lfm2_moe.py:433-456`）：

```python
def forward(self, layer_id, positions, hidden_states, residual, forward_batch, **kwargs):
    residual = hidden_states                    # ← 传进来的 residual 参数被直接覆盖
    normed = self.operator_norm(hidden_states)  # ← 没传 residual → 走非融合分支

    hidden_states = self.conv(normed, forward_batch)

    hidden_states = hidden_states + residual    # ← 单独一个 elementwise kernel
    hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
                    #                ↑ 又一个单独的 kernel
    return hidden_states, residual
```

三个关键观察：
1. 函数签名**收了 `residual` 参数**，第一行就把它覆盖掉 —— 传进来的值从没被用过
2. `RMSNorm.forward_cuda(x, residual)` **本来就会走 `fused_add_rmsnorm`**（`layers/layernorm.py:139-147`），一趟做完加法和归一化
3. `Lfm2MoeModel.forward` **本来就在层间传递 residual**

→ **接线全都在，只是这一层没接上。**

**修复**：改成 llama / qwen2 / 所有正常模型都在用的 **deferred-residual** 写法（`models/llama.py:304-316`）：

```python
if residual is None:                        # 只有第一层
    residual = hidden_states
    normed = self.operator_norm(hidden_states)
else:                                        # 其余层：走融合分支
    normed, residual = self.operator_norm(hidden_states, residual)

hidden_states = self.conv(normed, forward_batch) if not self.is_attention_layer \
                else self.self_attn(positions, normed, forward_batch)

hidden_states, residual = self.ffn_norm(hidden_states, residual)   # 融合
hidden_states = self.feed_forward(hidden_states)
return hidden_states, residual
```

**数学等价性**（写 `x` 为进入本层的激活）：

```
原版：  a = op(rms(x));  h1 = a + x;  out = h1 + ffn(rms(h1))
新版：  rms(x, r) → r := x,      n := rms(x)          ← 加法在 norm kernel 内部完成
        a = op(n)
        rms(a, r) → r := a + x = h1,  n2 := rms(h1)
        返回 (ffn(n2), h1)
```

下一层拿到 `ffn(n2) + h1`，与原版的 `out` 是同一个值。
**残差不在本层结清，而是当作"欠账"传给下一层，由下一层的 norm kernel 顺手结清。**

**效果**：每层省 2 个 kernel × 24 层 = **48 个**。

---

#### G2 `scale` —— 每次 forward 有 22 个 kernel 在"乘以 1"（接线）

**问题代码**（`lfm2_moe.py:156-169`）：

```python
def forward(self, hidden_states):
    router_logits, _ = self.gate(hidden_states)
    topk_output = self.topk(hidden_states, router_logits)
    final_hidden_states = self.experts(hidden_states, topk_output)
    return final_hidden_states * self.routed_scaling_factor   # ← 这里
```

而 LFM2.5 的 `config.json` 里 **`"routed_scaling_factor": 1.0`**。

→ 每次 forward **22 个 GPU kernel** 在把整个 `[T, 2048]` 激活张量逐元素乘以 1：
读一遍、乘 1、写一遍，**什么都没发生**。

**修复**（3 行）：
```python
if self.routed_scaling_factor == 1.0:
    return final_hidden_states
return final_hidden_states * self.routed_scaling_factor
```

**这是 bit-exact 的** —— 有限的 bf16 数乘 1.0 就是它自己。

> 注：代码里那个 factor 不放进 `FusedMoE` 而手动乘是有正当理由的
> （放进去会引入与 HuggingFace 的数值差异），所以 factor ≠ 1 时那个乘法该留着。
> **只跳过 = 1 的情况。**

---

#### G3 `conv` —— ★ 手写 Triton kernel ①②，ShortConv 的胶水不合并

**问题代码**（`lfm2_moe.py:321-377`）：

```python
proj, _ = self.in_proj(hidden_states)        # GEMM -> [T, 3H]
B_gate, C_gate, x = proj.chunk(3, dim=-1)    # 3 个 strided 视图
Bx = B_gate * x                              # elementwise -> [T, H]
Bx_t = Bx.transpose(0, 1).contiguous()       # 物化 -> [H, T]
conv_out = causal_conv1d_fn(Bx_t, ...).transpose(0, 1)   # 视图, [T, H]
output, _ = self.out_proj(C_gate * conv_out) # elementwise，读的是转置视图
```

`causal_conv1d_fn` 是**不透明的外部 CUDA 算子**，要求 `[dim, seqlen]` 布局且
`x.stride(-1) == 1`（`causal_conv1d.py:59-60`）。
所以**布局转换躲不掉，只能被吸收**进相邻的 elementwise 工作里。

**关键诊断 —— 问题不是流量，是访问不合并**：

```
18 个 conv 层 × 500 MB 流量 = 8.79 GB  用了 10.3 ms  →  0.83 TB/s
H200 HBM 峰值 ~4.8 TB/s                              →  仅 17% 峰值
```

`Bx.transpose(0,1).contiguous()` 和 `C_gate * conv_out` 里的转置读**都是跨步访问**，
每次取回的 cache line 大部分被丢弃。

**修复**（`scripts/lfm_fusion/lf_triton_shortconv.py`，189 行）：
conv **两侧各一个 tiled Triton kernel**，把 chunk + gating mul + transpose 折叠进一趟，
转置用 `tl.trans` **在寄存器/共享内存里完成**，不发跨步全局访问。

```python
@triton.jit
def _fused_gate_transpose_kernel(proj_ptr, out_ptr, T, H, ...):
    # 沿 H 合并读入 [BLOCK_T, BLOCK_H] tile
    b = tl.load(proj_ptr + base, mask=mask)                    # B_gate
    x = tl.load(proj_ptr + base + 2*H*stride_h, mask=mask)     # x
    bx = (b.to(tl.float32) * x.to(tl.float32)).to(b.dtype)
    # 转置写出：[BLOCK_H, BLOCK_T]，沿 T 合并
    tl.store(out_ptr + out_off, tl.trans(bx), mask=...)
```

**隔离结果**（correctness 门禁先于计时）：

| T | input side | output side | 带宽 | 每 forward 省 |
|---:|---:|---:|---|---:|
| 1024 | 0.94× | 0.71× | — | −0.22 ms |
| 2048 | 1.29× | 0.93× | 0.9 → 0.7 TB/s | +0.27 ms |
| 4096 | 2.24× | 1.76× | 0.9 → 1.3 TB/s | +1.47 ms |
| **16000** | **5.93×** | **4.33×** | **0.98 → 3.46 TB/s** | **+7.86 ms** |

**带宽从 17% 提到 ~72% 峰值。每个测试形状都 bit-exact**（max\|diff\| = 0.0）。

**形状门控**：融合 kernel 有 ~30 µs 的地板（Triton 的 Python launch 路径），
T < 2048 时打不过原生 elementwise。低于 `CONV_FUSION_MIN_TOKENS` 走原路径。
tile 尺寸来自**实测扫描**（`lf_tune_shortconv.py`，每形状 32 组配置，先验正确性再计时），不是猜的。

decode 路径**根本不转置**（`causal_conv1d_update` 直接吃 `[T,H]`），
所以这个组件**结构上就是 prefill-only**。

---

#### G4 `qkrope` —— 融合原语早已存在，这个模型没调用（接线）

**问题代码**（`lfm2_moe.py:236-263`）：

```python
q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
q = q.reshape(T, num_q_heads, head_dim)
k = k.reshape(T, num_kv_heads, head_dim)
q = self.q_layernorm(q.reshape(-1, head_dim)).reshape(...)   # 独立 RMSNorm
k = self.k_layernorm(k.reshape(-1, head_dim)).reshape(...)   # 独立 RMSNorm
q, k = self.rotary_emb(positions, q, k)                       # 独立 RoPE
```

而 `sgl_kernel.fused_qk_norm_rope` **早就存在**
（`sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu`），
把两个 head-wise RMSNorm 和 RoPE 合并成**一个 in-place CUDA kernel**，
**Qwen3-MoE 已经在调用它**（`models/qwen3_moe.py:559-585`）。**LFM2.5 没有。**

**测得的现有链路**：decode 18 次调用 32.5 µs（**1.65%** kernel 时间）；
prefill 30 次调用 5581 µs（**3.61%**）。

**修复**：在 packed QKV 上直接调融合 kernel。
LFM2.5 `head_dim = 2048/32 = 64`（该 kernel 支持），`rope_type` 为 `default` 无 `rope_scaling`
→ yarn 参数退化为恒等 `(1.0, 0, 0, 1.0)`：

```python
if qkv.dtype == torch.bfloat16 and self.head_dim == 64:
    pos = positions.view(-1).to(dtype=torch.int32, device=qkv.device).contiguous()
    fused_qk_norm_rope(
        qkv, self.num_local_q_heads, self.num_local_kv_heads, self.num_local_kv_heads,
        self.head_dim, self.q_layernorm.variance_epsilon,
        self.q_layernorm.weight, self.k_layernorm.weight,
        self._lfm_rope_theta, self.rotary_emb.is_neox_style, pos, 1.0, 0, 0, 1.0)
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
```

不满足条件时**回退原实现**（保留 fallback）。

> **这是 G1 的同一模式，也是第三个实例：融合原语已存在，调用点没用。**
> （第四个实例在另一个模型上：OLMo-2，见 `docs/kernel_fusion_catalogue.md`）

---

#### G5 `idx` —— 18 个 kernel 只为搬 12 字节（接线）

`req_pool_indices.to(torch.int32)` 在**每一个 conv 层**里重算一遍（18 次/forward）。
这个 kernel 只搬 **12 字节**，是纯 launch 开销，但占**低批 decode kernel 时间 ~1.3%**。

**修复**：按 forward 缓存，用**源张量的 identity 作 key**，保证不会返回陈旧缓存：

```python
cached = getattr(forward_batch, "_lfm_int32_idx", None)
if cached is not None and cached[0] is req_pool_indices:
    return cached[1]
out = req_pool_indices.to(torch.int32)
forward_batch._lfm_int32_idx = (req_pool_indices, out)
```

---

#### G6 `gate` —— ★ 手写 Triton kernel ③，strided rows 让向量化失效

decode 路径的 `B_gate * x` 读的是 `proj` 的**跨步行**。
这**是合并访问**，但仍只跑到 54% 峰值 —— 因为跨步的**行**让 PyTorch 的 `TensorIterator`
无法向量化，退化成标量 `elementwise_kernel` 而不是 `vectorized_elementwise_kernel<8>`
（**由 trace 里的 kernel 名直接确认**）。

改用一个直接读 `proj` 的 Triton kernel（`_gate_mul_kernel`）绕开。

---

#### G7 `moesum` —— ★ 手写 Triton kernel ④，消除 MoE 归约的 HBM 往返

**问题**：MoE 的 top-k 归约把 `[T, H]` 写回 HBM，**紧接着下一层**的 `fused_add_rmsnorm`
又把它读回来。**两者都是行方向的操作** —— 多跑了一整趟 HBM 往返。

**修复**（`scripts/lfm_fusion/lf_triton_moesum.py`，112 行）：
让 `FusedMoE` 返回 **4 个加权专家输出**（不归约），一个 kernel 做完
**归约 + 残差加 + RMSNorm**：

```
每行加载 top-k 分量 → 求和 → 加残差 → 算 RMS → 乘权重
                   → 同时写出「归一化输出」和「更新后的残差」
```

**隔离结果**：

| T | stock | fused | 加速 | 带宽 | bit-exact |
|---:|---:|---:|---:|---:|---|
| 1 | 90.5 µs | 36.8 µs | **2.46×** | 0.8 GB/s | ✅ |
| 8 | 95.6 | 35.7 | **2.68×** | 6.1 | ✅ |
| 32 | 94.8 | 36.0 | **2.64×** | 23.9 | ✅ |
| 128 | 24.9 | 33.8 | 0.74× | 101 | ✅ |
| 1024 | 24.8 | 34.3 | 0.72× | 798 | ✅ |
| 4096 | 42.2 | 36.9 | 1.14× | 2961 | ✅ |
| 16000 | 145.7 | 111.8 | 1.30× | 3821 | 4.9e-4 |

residual 输出全程 bit-exact；归一化输出到 T=4096 精确，T=16000 差 4.9e-4。

> **这与 G3 的形状依赖正好相反。** G3 在 T<2048 无用，G7 在**小 T 才是赢面** ——
> 因为省的是 **launch + 一次 HBM 往返**，T=1 时那几乎就是全部成本。
> **两个 kernel 形状依赖相反，合起来覆盖了整个范围。**

双侧门控：`T <= 32 或 T >= 4096`。

---

### 5.4 L3 的端到端结果

`lf_e2e.py` 复用 canonical serving harness，只变 `LFM_FUSION_PATCH`；
模型、serving 参数、backend、CUDA graph 设置完全一致。
6 次重复/臂，Welch t + **精确 Student-t 尾**。

#### 单项与组合

| regime | `qkrope` | `gate+idx` | `norm+scale+conv` | `moesum` | 六项 | **七项全开** |
|---|---:|---:|---:|---:|---:|---:|
| A 低批 decode | +0.93% | −0.00% (n.s.) | +3.89% | +4.55% | +4.60% | **+6.57%** |
| B 并发 decode | **+5.42%** | +0.65% (n.s.) | +3.65% | +3.08% | +6.01% | **+6.21%** |
| C 长 prefill | +1.99% | +0.40% (n.s.) | +3.47% | — | +5.81% | **+5.30%** |

七项全开的 p = **4.6e-14 / 2.4e-08 / 1.2e-05**。

#### 组件按机制互补 —— 四种不同形状的收益

- **`norm+scale`** 消除的是**每 forward 固定数量**的 kernel 和全激活读写，与该 forward 做多少计算无关
  → decode 每 forward 才 ~2 ms，占比大（+4.2%）；长 prefill ~157 ms，被稀释（+1.6%）
- **`conv`** 消除的是**随 token 数增长**的流量，且要 T≥2048 才划算
  → decode 够不到（精确中性，p=0.22/0.95），长 prefill 跑在 T=4000–16000（+2.33%）
- **`qkrope`** 消除的是 6 个注意力层里的工作 → **并发 decode 最受益**（+5.42%）
- **`moesum`** 消除的是 launch + HBM 往返，**小 T 最赚** → **低批 decode 最受益**（+4.55%）

> **只测一个 regime，四种收益一个都看不全。**

#### 诚实负面

`gate+idx` **三个 regime 全不显著**。机制在 kernel 级真实可测（1~2%），
但**没能兑现到端到端**。保留在报告里。

---

## 6. 正确性验证

### 6.1 一个结构性发现：token-identity 对这个模型不可用

`scale` / `conv` / `moesum` 的 residual 部分是 bit-exact 的。
但 `norm` 和 `qkrope` **代数等价而非 bit-exact**（`fused_add_rmsnorm` 累加顺序与精度不同）。

**三层证据**：
1. **原语单测**：`fused_add_rmsnorm` vs 手动 `add` + `rmsnorm` —— residual 差 **0.0**，
   归一化输出差 0.03（量级 4.34），约 **2 个 bf16 ulp**，且**融合版更准**（加法保持在更高精度）
2. **重构单测**：6 层代数替身栈跑两遍 —— 相对偏差 1.1%，符合 bf16 累加漂移，无结构性错误
3. **整模型**：12 个 prompt 的 next-token 分布 —— top-1 11/12 一致，但 KL 最高到 0.99

第 3 层看着吓人，直到机制被指出：
**LFM2.5 走 top-4/32 专家路由，专家选择是离散 argmax。**
bf16 级扰动偶尔会翻转选中哪个专家，输出就不连续地变了。

> **所以 token-identity 对这个模型是结构性不可用的门禁** —— 任何数值上非恒等的改动都会触发它。
> **必须改用任务指标。**

### 6.2 用 bit-exact 的对照臂免费标定噪声底

GSM8K 全量 1319 题，贪心解码：

| 臂 | 各次结果 | 均值 |
|---|---|---:|
| baseline | 0.348 / 0.349 / 0.344 | 0.3470 |
| **`scale`（可证 bit-exact）** | 0.338 / 0.339 / 0.340 | **0.3390** |
| `norm` | 0.362 / 0.368 / 0.361 | 0.3637 |
| `norm+scale` | 0.359 / 0.359 / 0.359 | 0.3590 |
| **`conv`（bit-exact）** | 0.342 / 0.350 | **0.3460** |
| `qkrope` | 0.352 / 0.346 | 0.3490 |
| **`moesum`（bit-exact）** | 0.343 / 0.347 | **0.3450** |
| 全部七项 | 0.371 / 0.364 / 0.370 | 0.3683 |

**注意 `scale` 臂**：它**数学上必然等于 baseline**，却读数低 **0.8 点**。
这不是 bug，是它**免费帮我们标定出了 harness 的系统噪声**
（`--parallel 32` 让 batch 组成在不同 server 实例间不同，而 batch 相关的 reduction 会改变贪心输出）。

三个噪声度量：between-arm 系统噪声 ≥ 0.8 点；within-arm 跨度 0.0–0.8 点；
n=1319, p≈0.35 的二项抽样误差 ±2.6 点。**全部 8 个臂跨度 2.5 点，在三个度量下都在噪声内。**

> **口径：未检测到质量回归。**
> 不是"质量提升" —— 这个实验分辨不了这么小的差异，**而那个 bit-exact 的臂就是证据**。

---

## 7. NCU 硬件层证据（headroom 从哪来）

`docs/2026-07-10/v9_ncu_hardware_ceiling_evidence.md`，
**在 v8 tuning 出来的最优 config 下**测的（所以是"tuning 之后仍剩下的空间"）：

### prefill 段（agent trace, in=2700, b=1）

| 模型 | kernel | SM% | DRAM% | Occ% |
|---|---|---|---|---|
| LFM2.5 | nvjet_gemm | 86.2 | 17.4 | **14.7** |
| LFM2.5 | act_and_mul | 42.0 | 78.3 | 73.9 |

### decode 段

| 模型 | regime | kernel | SM% | DRAM% | Occ% |
|---|---|---|---|---|---|
| LFM2.5 | b32 | flash_attn | 45.8 | 45.1 | 24.9 |
| LFM2.5 | b32 | nvjet_gemm | 10.1 | 68.1 | 14.2 |

**结论**：
1. **Occupancy 普遍只有 12–25%** —— Hopper 的 warp 调度槽有 75–88% 是空的。
   occupancy 由 kernel 的 launch/寄存器/tile 配置决定，**serving config knob 改不了它**
2. **decode 最热的 flash_attn：SM 48% / DRAM 46%** —— 既没被算力卡、也没被带宽卡，
   卡在延迟/依赖 stall 上。**两个硬件维度都还剩一半**

> 这是"gap 不在 serving config 层"的直接硬件证据。

---

## 8. L2 + L3 串联 —— ★ 实验 3 已完成（2026-08-03）

> **本节原标题是「当前最大的结构问题：三层从未串联」。实验 3 已经把 L2+L3 那一层补上了，
> 结论比预期好得多。原分析保留在 §8.4 供对照。**

### 8.1 实验设计

同一棵树、同一份 serving 参数，**只有 `SGLANG_MOE_CONFIG_DIR` 和 `LFM_FUSION_PATCH` 两个变量**：

```
2 (config: nocfg / cfg) × 2 (kernel: baseline / all7) × 2 (顺序: fwd / rev) × 8 reps
```

**顺序做了 counterbalance**（`rk_e2e.py` 是顺序执行 arm，存在位置效应——
L2 的 decode 那栏就被这个坑过，见 §4.4）。合并后每格 n=16。

config 生效已在 server log 中确认：

```
Using MoE kernel config from .../lfm25_pr_candidate/configs/triton_3_5_1/
                             E=32,N=1792,device_name=NVIDIA_H200.json
```

### 8.2 ★ 结果摘要

**C 长 prefill**（req/s，n=16/格）：

| | baseline | + kernel rewrite (all7) | 增量 |
|---|---:|---:|---:|
| **无 tuned config** | 12.119 ± 0.116 | 12.869 ± 0.182 | **+6.18%**（p=4.5e-13） |
| **有 tuned config** | 14.939 ± 0.123 | **16.392 ± 0.200** | **+9.73%**（p=9.5e-19） |

**A 低批 decode**：

| | baseline | all7 | 增量 |
|---|---:|---:|---:|
| 无 tuned config | 1.6863 | 1.7992 | +6.70%（p=2.1e-41） |
| 有 tuned config | 1.6872 | 1.7944 | +6.35%（p=1.8e-34） |

> **完整分析、六项 vs 七项拆解、`moesum` 边际贡献、以及为什么增量反而变大，
> 全部在 §9.5 和 `docs/2026-08-03/exp3_kernel_on_tuned_baseline.md`。本节只给摘要。**

### 8.3 三个结论

**① 完整的分层图现在画得出来了（长 prefill）：**

```
cookbook 基线                    12.12 req/s      ——
+ L2 tuned MoE config            14.94 req/s      +23.3%
+ L3 kernel rewrite              16.39 req/s      +9.73%  ← 叠在 tuned 之上
                                                  ─────
总计 vs 基线                                      +35.3%
```

**★ 这直接回答了 Dey 的问题**：best kernel autotuning 之后，kernel rewrite **仍然贡献 +9.73%，
t=24.0，8/8 重复不重叠**。

**② 增量不但没缩水，反而变大了（+6.18% → +9.73%）。**
这**超过**了 §9 正交假设的预测（+6.62%）。机制见 §9.5。

**③ A regime 是完美的阴性对照。**
L2 的 guarded 策略对 `M ≤ 32` 逐字段保持默认，所以 decode 路径**本来就不该受影响**——
实测 baseline 1.6863 vs 1.6872（差 0.05%）。
而 kernel 增量几乎不变（+6.70% → +6.35%）。**两个层次在 decode 上互不干扰，符合设计预期。**

### 8.4 【已解决】原先的问题分析（保留供对照）

在实验 3 之前，三个研究**都**把 serving 配置冻结在 cookbook：

| 研究 | 文件 | serving 配置 |
|---|---|---|
| L2 kernel config tuning | `rk_e2e.py:33-39` | `cap=32, chunk=-1, lpm, mem=0.85` |
| L3 kernel rewrite | `lf_e2e.py:42-48` | `cap=32, chunk=-1, lpm, mem=0.85` |

而 **L1 找到的长 prefill 赢家是 `cap=24, chunk=2048, fcfs, mem=0.75`** —— 完全不同的配置。

且实验 1 已核实 `17f7a1da1` 树里**没有** `E=32,N=1792` 的任何 config，
`lf_e2e.py` 也没有 `SGLANG_MOE_CONFIG_DIR`。所以当时的结构是三条平行分支：

```
                      ┌── L1 serving tuning ──→ +56.9%  (长 prefill, TRADE-OFF)
  cookbook 基线 ──────┼── L2 kernel config ───→ +22.1% ~ +23.3%
  (12.28 req/s)       └── L3 kernel rewrite ──→ +5.30%
```

当时的风险是：原样交出去会讲成
❌ *"kernel autotuning 给了 +23%，我们手写 Triton kernel 只给了 +5%"*——正好反驳论点。

**L2+L3 已经串联（§8.2）。L1 仍未串联进去**（见 §10 实验 5，可选）。

> **【2026-08-03 更新】X 已实测：+9.73%（p=9.5e-19）。见 §9.5。**
> 串联后的完整链条（同一 session、同一棵树、counterbalanced n=16/臂）：
> `12.119 ──L2: +23.26%──→ 14.939 ──L3: +9.73%──→ 16.392 req/s`

---

## 9. 正交性分析 —— X 应该是多少

### 9.1 代码层面：7 项全部避开 MoE GEMM

L2 只改 `fused_moe_kernel` 的 tile 参数。L3 的 7 项动的是：

| 项 | 动的是 | 碰 `fused_moe_kernel` 吗 |
|---|---|:--:|
| `conv` (Triton×2) | ShortConv 的转置胶水 | ❌ |
| `qkrope` | 注意力层的 norm+RoPE | ❌ |
| `norm` | RMSNorm 的残差加 | ❌ |
| `scale` | FusedMoE **输出之后**的乘法 | ❌ |
| `gate` / `idx` | conv 路径 | ❌ |
| `moesum` | `moe_sum_reduce` + 下层 RMSNorm | ⚠️ **相邻**，但不是 GEMM |

**7 项全部避开 GEMM 本身。**
（`moesum` 是唯一结构性接触点 —— 它改了 `FusedMoE` 的返回值，让它吐 4 个未归约的专家输出。）

### 9.2 时间占比（nsys 实测，长 prefill T=16000，单个 MoE 层）

```
fused_moe_kernel #1      3176.8 us
fused_moe_kernel #2      1602.3 us
                        ─────────
MoE GEMM                 4779.1 us  ←  73.6%   ← L2 只动这块
其他全部 15 个 kernel     1711.1 us  ←  26.4%   ← L3 只动这块
```

### 9.3 Amdahl 推导

若两者作用在时间轴的**不相交部分**：

```
基线                       t=1.0000   thr=1.0000
+ L2 tuned MoE config      t=0.8108   thr=1.2334   (+23.34%)
+ L3 kernel rewrite        t=0.7604   thr=1.3150   (+31.50%)

→ L3 叠在 L2 之上的增量 = 1.0662 = +6.62%
→ 对比：L3 叠在未 tune 基线上 = +5.30%
```

**★ 正交的话，L3 的增量应该从 +5.30% 涨到 +6.62%，而不是缩水。**

原因很直白：**分母变小了。** L3 省的绝对时间不变，
但 MoE GEMM 被 L2 砍掉之后总时间从 1.0 降到 0.81 —— **同样一块肉，占比自然更大**。

> ⚠️ **本文档早先版本曾预测"会缩到 +2~4%"，那是把「同类优化次可加」的经验规律
> 错误地套到了「不同类优化」上，方向反了。已撤回。**

### 9.4 但有三个理由可能吃掉一部分

1. **`moesum` 与 L2 有结构性接触** —— 它改了 `FusedMoE` 的返回，
   而 L2 改的是 GEMM 的 block 划分，两者共用 `intermediate_cache` 布局。**可能有真实交互。**
2. **e2e 里有不缩水的部分** —— throughput 包含调度、tokenize、HTTP，这些不会因 kernel 变快而变快。
   **这是把理想的 1.3150× 拉低的主要力量。**
3. **我们自己的 waterfall 数据撞过** —— serving 1.78× × kernel 1.22× → 理论 2.17×，
   **实测 1.70×**（超额部分只兑现 60%）。那次隔得更远，兑现率反而更低，
   所以不是可靠下界，但足以说明**理论值不能直接报**。

**诚实预期区间：+2% ~ +6.6%，中位数猜 +4~5%。必须实测。**

> 🔴 **本节 9.1–9.4 是实测之前的预测，已被 §9.5 的实测取代。**
> 实测 X = **+9.73%**，**高于**本节的正交上界 +6.62%。
> 上面这个"+2%~+6.6%"的区间**是错的，方向偏保守**，保留在此仅供对照。
> 正确的机制拆解见 §9.5：Amdahl 部分 +2.06 点、`moesum` 与 tuned MoE GEMM 的真实交互 +1.49 点。

---

## 9.5 ★ 实测结果（2026-08-03，GPU 4）

**X = +9.73%（p = 9.5e-19）。比本节的正交上界 +6.62% 还高。**

完整记录：`docs/2026-08-03/exp3_kernel_on_tuned_baseline.md`
设计：`{L2 关,开} × {臂顺序 正,逆}` 的 2×2，合并顺序后每臂 n=16，来自 2 个独立
server lifetime。顺序对照不是可选项——今天 regime C 的 baseline 正序 12.020 / 逆序
12.219，位置效应 1.7%，比要测的效应的一半还大。

### 四根柱子（regime C 长 prefill，counterbalanced n=16/臂）

| Bar | 内容 | req/s | 相对 Bar 2 |
|---|---|---:|---:|
| 2 | cookbook 默认 = autotuning ceiling | 12.119 | 1.000× |
| 3 | + L2 tuned MoE config | 14.939 | 1.233× |
| 4 | + L3 kernel rewrite（七项） | **16.392** | **1.352×** |

| 比较 | 变化 | p |
|---|---:|---|
| L3 增量，**无** L2（复现 7/27 的 +5.30%） | +6.18% | 4.5e-13 |
| **L3 增量，有 L2** | **+9.73%** | **9.5e-19** |
| L2 单独（复现 PR 草稿的 +23.34%） | +23.26% | 1.1e-33 |
| 整栈 Bar2 → Bar4 | +35.25% | 1.3e-29 |

**§9.3 的正交推导方向对了，量不够。** 兑现率 1.14 —— 超可加。

### 为什么比正交上界还高：§9.4.1 猜中了

补测**六项臂**（去掉 `moesum`）后，+3.54 个百分点干净地拆成两半：

| 来源 | 无 L2 | 有 L2 | 贡献 |
|---|---:|---:|---:|
| 六项（`all`，全部避开 MoE） | +6.41% | +8.47% | **+2.06 点 = Amdahl** |
| `moesum` 边际（`all7 − all`） | **−0.08%**（p=0.88，中性） | **+1.69%**（p=2.8e-04） | **+1.49 点 = 真实交互** |
| 合计（`all7`） | +6.18% | +9.73% | +3.54 点 |

- **Amdahl 部分被定量验证**：六项省下的**绝对**时间几乎是常数
  （4.978 → 5.253 ms/req，比值 1.06），把这个常数代进调优后的基线预测 7.99%，实测 8.47%。
- **交互部分是 §9.4.1 说的那个**：`moesum` 是唯一改 `FusedMoE` 返回值的一项，
  它在未调优的 MoE 上**一文不值**，在调优后的 MoE 上值 **+1.69%**。
  七项的绝对节省比值是 1.23（不是 1.06），多出来的正是它。

### 附带修正两处

1. **§9.4.2 猜的"e2e 里不缩水的部分会把理论值拉低"没有发生**——实测高于理论，
   说明在这个 regime 上调度/HTTP 开销不是主导。
2. **7/27 报告读出的"`moesum` 在长 prefill 帮倒忙"（六项 +5.81% > 七项 +5.30%）要更正**：
   在它自己的基线上 `moesum` 是**中性**（−0.08%, p=0.88）而非负面；在干净基线上它是
   **第二大贡献项**。

### 还有一件事：L2 在 decode 上精确中性

regime A 低批 decode 同样做了 2×2：

| 比较 | 变化 | p |
|---|---:|---|
| L2 单独 | **+0.05%** | **0.34（不显著）** |
| L3 增量，无 L2 | +6.70% | 2.1e-41 |
| L3 增量，有 L2 | +6.35% | 1.8e-34 |

独立复现 PR 草稿的「−0.13%, p=0.079」。这是 guarded 策略的直接产物：`M ≤ 32` 的桶
逐字段等于默认启发式，而 CUDA graph 捕获的 decode batch 全落在那一段。

**推论：L3 在 regime A/B 上的 +6.57% / +6.21% 不需要重测。**
「脏基线」问题只影响 prefill，三个 regime 里只有 C 要重做。

### 9.6 这个 case 为什么强

> **kernel rewrite 和 config autotuning 作用在时间轴的不相交区域。**
> **autotuning 把 73.6% 那块打到极限之后，剩下 26.4% 只有 rewrite 能动。**
> **而且 autotuning 越成功，rewrite 的相对价值越高**（实测：+6.18% → +9.73%）。

这不只是"还有 X%"，是**机制性解释** —— 两者根本在优化不同的东西，
所以 autotuning 无论多强都到不了 rewrite 那块地。

还顺带解释了为什么 LFM2.5 的 **serving-config** autotuning 在 3/6 regime 零收益：
**该模型在那些 regime 下的瓶颈根本不在 serving 参数能碰到的地方**（§7 的 NCU 数据佐证）。

---

## 10. 缺口与待补实验

| # | 实验 | 目的 | 预计 | 状态 |
|---|---|---|---|---|
| 1 | 查 `17f7a1da1` 树里有无 `E=32,N=1792` config | 确认 L3 是否叠在 L2 上 | — | ✅ **已完成**：没有 |
| 2 | 裁剪空间**网格穷举**取代 TPE | 把 ceiling 从软变硬 | 2–4h | ⬜ |
| 3 | **L2+L3 串联**：装上 tuned config 重跑 `baseline,all7` | ★ 拿到 Dey 要的那个 X | 3–4h | ✅ **已完成**：X = **+9.73%**（§9.5） |
| 4 | NCU headroom 串进主线叙事 | Mason 证据链第 4 步 | 1h | ⬜ |
| 5 | L1+L2+L3 三层全串联 | 完整 waterfall | 4–6h | ⬜ 可选 |

### 关于实验 2 的说明

**问题**：`docs/2026-06-30/lfm2.5_conditional_autotuning.md` 的 ceiling 只有 25 次 TPE trial，
而且**我们自己的报告承认 TPE 坏掉了** —— 前 7 个 trial 把 `triton MoE` 和差 batching 绑一起，
之后 18 个 trial 再没试过 `triton + 好 batching`。

**审稿人一定会问：「这不是 ceiling，是搜索失败。」**

**缓解**：`docs/2026-07-22/lfm25_serving_autotuning_plateau.md` 的 100-trial 无热启动研究
**已经在很大程度上补上了这个洞**（100 trial、最后 20 个改进 0%、事后交错验证）。
如果时间紧，实验 2 可以降级为可选。

### 实验 3 的执行注意

> ✅ **已按这个方式执行完毕（2026-08-03）**，实际做法比原计划多两点，都很关键：
> 1. **不改树、也不放文件进树**，直接 `export SGLANG_MOE_CONFIG_DIR=configs/regime_kernel/profiles/lfm25_pr_candidate`。
>    `lf_e2e.py` 用 `os.environ.copy()` 启 server，所以**两臂自动都看得到**，config 成为基线属性而非臂间差异。
> 2. **必须做顺序对照**。`lf_e2e.py` 顺序跑臂、一臂一个 server lifetime，位置效应实测
>    1.7%，比要测的效应的一半还大。最终设计是 `{L2 关,开} × {正序,逆序}` 的 2×2，
>    合并后每臂 n=16、来自 2 个独立 lifetime。
>
> 另外补测了**六项臂**（去掉 `moesum`）以解释超可加从哪来，见 §9.5。

`lf_e2e.py` 已经是环境变量切换臂，**不要**为了加 config 而建第二棵 worktree
（Gemma-3 那次因此撞上 stride 问题，attention backend 直接拒绝）。
**把 config 文件放进同一棵树即可，两臂都会看到它。**

```bash
# 实际执行的命令
GPU=4 REPS=8 PORT=52141 REGIME=C_long_prefill    bash scripts/lfm_fusion/exp3_layered.sh
GPU=4 REPS=8 PORT=52142 REGIME=A_low_batch_decode bash scripts/lfm_fusion/exp3_layered.sh
SUITE=six_ ARMS_FWD=baseline,all ARMS_REV=all,baseline \
    GPU=4 REPS=8 PORT=52143 REGIME=C_long_prefill bash scripts/lfm_fusion/exp3_layered.sh
```

### 建议的主 regime：**C 长 prefill**

- L1（+56.9%）、L2（+23.3%）、L3（**+9.73%，叠在 L2 之上**）**三层在这个 regime 都有结果**，唯一一个
- `conv` 手写 kernel 在这里最有效（+2.33%）
- **A/B 上 L2 是中性的**（regime A 实测 +0.05%, p=0.34），图会退化成两段
- prefill 更容易展示 kernel-bound 特性（NCU 显示 prefill 在 compute roof 上）

> ⚠️ 原文这里写的理由是「次可加性损失最小（兑现率 0.90）」。**这个理由已作废**——
> regime C 实测是**超可加**（1.14）。选 C 的真正理由是上面第一条和第三条。

---

## 11. 方法学产出（比那 6% 更可迁移）

### 11.1 ★ 同类优化强烈次可加

| regime | 各项之和 | 一起测 | 兑现率 |
|---|---:|---:|---:|
| C 长 prefill | 5.86% | **5.30%** | 0.90 |
| A 低批 decode | 9.37% | **6.57%** | 0.70 |
| B 并发 decode | 12.80% | **6.21%** | **0.49** |

并发 decode 上：`qkrope` 单独 +5.42%，再加单独值 +3.65% 的 `norm+scale+conv` 只多买到 **0.12 点**；
再加单独值 +3.08% 的 `moesum` 又只多买到 **0.19 点**。
三者都在消除**同一份"固定每-forward 开销"的余量**，消完之后别的东西成为瓶颈。

**兑现率的排序精确跟踪 regime 的饱和程度**：
长 prefill 每 forward 工作最多、最能把开销藏起来，损失最小（0.90）；
并发 decode 最饱和，损失最大（0.49，不到一半）。

> **规则：消除同一"种类"成本的优化不会相加。**
> **报告各项分别测量之和会高估整个 stack，且系统越饱和高估越严重。**
> **任何会真实部署的组合都必须按组合测量。**

> **⚠️ 边界条件（2026-08-03 补）：这条规则只适用于「同类」优化。**
> 跨层组合（L2 调 GEMM tile 参数 + L3 改 GEMM 周边）实测是**超可加**的：
> 兑现率 **1.14**，L3 增量从 +6.18% 涨到 +9.73%（§9.5）。原因有两个，
> 必须分开说，否则「超可加」听起来比实际神奇：
>
> | 来源 | 贡献 | 性质 |
> |---|---:|---|
> | 六个 MoE 之外的改动 | +2.06 点 | **平凡**：绝对节省近乎常数（4.98→5.25 ms/req），分母被 L2 压小 19% |
> | `moesum` × tuned MoE config | +1.49 点 | **真实交互**：−0.08%（p=0.88）→ +1.69%（p=2.8e-04） |
>
> **完整表述：削同一份成本的优化次可加；削不同层成本的优化可以超可加，
> 但其中一部分只是 Amdahl 的分母效应，不是新的收益。**

**实践含义：最便宜的组件反而最有价值。**
`qkrope` 是纯调用点改动，单独就拿下并发 decode 的大部分空间。

### 11.2 ★ 一条可机械检查的 signature

最锋利的观察：**两个最大的赢家都是"sglang 已有融合原语、这个模型的调用点没用"**
（`fused_add_rmsnorm`、`fused_qk_norm_rope`），加上一个乘以 1.0 和一个冗余 `.to(int32)`。
两个真正需要写 kernel 的，都是相邻行方向工作的机械融合，
而且 **Inductor 自己就能推导出其中一个**。**全程没有发明任何新东西。**

> **枚举代码库里已有的融合原语，检查哪些模型的调用点没用它们**
> —— 纯静态、不需要 profiling，这一条就找到了最大的两个赢家。

这条 signature 已固化进 SLO-agent 的 `fusion_scan.py`（4 种 gap 形态），
并在其他模型上复现（Gemma-3、OLMo-2、GraniteMoe）。

### 11.3 对项目既有结论的修正

之前的立场"成熟 bf16 MoE 上 kernel 层不转化为端到端收益"**仍然成立**，但补上边界条件：

> **覆盖空缺是"这个模型文件受到过多少优化关注"的函数，不是 sglang 的属性。**

上游优化过的模型族（Qwen3-30B：1 个未融合 norm、0 个独立 add）在融合层没剩空间。
新加入的架构（LFM2.5：61 + 48 + 36 + 一条未融合的 QK-norm+RoPE 链 + 一次 MoE 归约往返）
带着 **6.6%** 的纯开销 —— **不在它的新算子里，在周围的调用点胶水上**。

（原文曾写作"架构成熟度"，已被 `docs/2026-07-28/cross_architecture_audit.md` 推翻 ——
最新的 Qwen3-Next 几乎干净，成熟的 Gemma-3 最差；真正的分界是**模型家族**。）

---

## 12. 诚实范围

- 绝对值 ~5–6.6%（L3），**单模型单卡 TP1**
- `norm` / `qkrope` **非 bit-identical**，质量结论依赖噪声底 0.8 点的任务指标
- **大部分收益来自补漏用的融合原语，不是新 kernel**
- 一个组件（`gate+idx`）是**实测负面**
- §11.1 表明这个 stack **不会**交付各部分之和
- **L2 的 baseline 很弱**：对手是两档启发式，不是认真调过的配置。
  正确表述是"**这个 model/GPU 组合从来没人调过**"，不是"我们把 kernel 优化快了 1.6×"
- **L2 的 Triton 版本**：在 3.5.1 上 tune 和验证；上游 main 现在 pin torch 2.11.0。
  Triton 3.6 用户通过跨版本 fallback 拿到它，**理想情况应在 3.6 上重扫**
- **L1 的两个大赢（长 prefill / shared-prefix）都是 TRADE-OFF**，不是白拿
- **★ L1 / L2 / L3 三层从未串联测量**（§8）

### 过程中发现并修正的自身错误

1. **统计方法**：原用正态近似算 p，在 n=6 下 anti-conservative。
   改用精确 Student-t 后**无结论翻转**，但"全部 p<0.005"是错的（`qkrope` 在 C 实为 0.018），已改逐格 p 值
2. **数据丢失 bug**：`lf_bench_shortconv.py --tokens` 做局部扫描会**静默覆盖**完整曲线，
   已加 `--out` 并恢复数据
3. **正确性门禁选错**：最初用 token-identity，被专家路由的离散性证伪，改用 GSM8K
4. **一次实验失败被正确捕获**：regime A 的一臂 `rc=-9`（两个 server 争资源），
   harness 记为 `launch_failed` 而非静默丢弃，已单独重测
5. **L2 的 M 键错了一个 `top_k` 因子** —— 只有活体 trace 能暴露
6. **L2 的 decode 回归是位置效应** —— 顺序对照后消失
7. **本文档 §9.3 的早期预测方向错了** —— 已撤回并重推

---

## 13. 产物地图

### 文档

| 层 | 文档 |
|---|---|
| L1 全网格 | `docs/2026-07-24/qwen_serving_ceiling_results.md`（含 LFM）<br>`docs/2026-07-24/qwen_serving_ceiling_methodology.md` |
| L1 TPE 收敛 | `docs/2026-07-22/lfm25_serving_autotuning_plateau.md` |
| L1 早期（有硬伤，仅供参考） | `docs/2026-06-30/lfm2.5_conditional_autotuning.md` |
| L2 | `docs/2026-07-27/regime_kernel_results.md`<br>`docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md` |
| L3 | `docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md`<br>`docs/2026-07-27/lfm_fusion_results.md` |
| NCU | `docs/2026-07-10/v9_ncu_hardware_ceiling_evidence.md`<br>`docs/2026-07-22/ncu_roofline_fused_moe_analysis.md` |
| 跨模型 signature | `docs/kernel_fusion_catalogue.md` |
| mentor 要求对照 | `docs/2026-08-03/deliverables_vs_mentor_requirements.md` |
| 分层实验交接 | `docs/2026-08-03/HANDOFF_lfm25_layered_experiment.md` |

### 脚本

| 用途 | 路径 |
|---|---|
| workload / regime 定义 | `scripts/serving_ceiling_lib.py` |
| L2 e2e harness | `scripts/regime_kernel/rk_e2e.py` |
| L3 e2e harness | `scripts/lfm_fusion/lf_e2e.py` |
| L3 算子审计 | `scripts/lfm_fusion/lf_audit.py` |
| **手写 Triton kernel** | `scripts/lfm_fusion/lf_triton_shortconv.py`（189 行，3 kernel）<br>`scripts/lfm_fusion/lf_triton_moesum.py`（112 行，1 kernel） |
| 融合补丁（注入层） | `scripts/lfm_fusion/lfm_fusion_patch.py`（572 行） |
| tile 尺寸扫描 | `scripts/lfm_fusion/lf_tune_shortconv.py` |
| 正确性 | `scripts/lfm_fusion/lf_correctness.py` |

### 原始数据

| 内容 | 路径 |
|---|---|
| L1 全网格 | `results/2026-07-24_serving_ceiling/` |
| L1 验证 pass | `results/2026-07-24_serving_ceiling_validation/` |
| L1 TPE 收敛 | `results/2026-07-22_lfm25_plateau_100/` |
| L2 | `results/regime_kernel/` |
| L3 e2e | `results/lfm_fusion/e2e/lfm25/` |
| L3 nsys | `results/lfm_fusion/nsys/FINDINGS.md` |
| L3 FX | `results/lfm_fusion/fx/FINDINGS.md` |
| NCU | `results/2026-07-10_v9_ncu_realworkload/` 等 |

### 复现

```bash
# L3 算子审计（LFM2.5 + Qwen 对照）
python scripts/lfm_fusion/lf_audit.py --model lfm25 --gpu N
python scripts/lfm_fusion/lf_audit.py --model qwen  --gpu N

# L3 端到端（七项全开）
python scripts/lfm_fusion/lf_e2e.py --regime C_long_prefill --gpu N \
       --arms baseline,all7 --reps 6

# L2 端到端
python scripts/regime_kernel/rk_e2e.py --model lfm25 --regime C_long_prefill --gpu N

# L1 复现
bash results/2026-07-22_lfm25_plateau_100/reproduce.sh
```
