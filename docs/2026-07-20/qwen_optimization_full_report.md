# Qwen3-30B-A3B 优化全纪录：做了什么 · 结果 · 有无提升 · 分析

**模型**：Qwen3-30B-A3B-Instruct-2507（MoE，E=128，top-8，hidden=2048，moe_intermediate=768，无 shared expert）
**硬件**：H200（GPU0/1），bf16，sglang（editable @17f7a1da1，triton 3.5.1）
**日期**：2026-07-19 ~ 2026-07-20 · **所有数字均为本项目实测**（非 PR 自称），对标 sglang 真实 GPU 代码 + cudagraph
**报告目的**：给团队/Dey/Ofer 一个关于 Qwen 这条线的完整、诚实口径。

---

## 0. 一句话总结

在**成熟 bf16/H200 上的 Qwen3-30B**，我们验证的**可复现同模型端到端提升 = MoE kernel-config 自动调优（★e2e 实测 prefill +34~43%，v42 §1.7）+ 开启投机解码（decode c32 +30.6%）**；**自己重写/融合 kernel 在全 regime 端到端 ≈0**（隔离层 1.23× 不迁移，b1 仅 +1.17%）。真正"tuning 之外"的空间在**算法层（spec decoding）**和**架构层（线性注意力，需换模型）**，不在重写 bf16 MoE kernel。

---

## 1. 我们在 Qwen 上做过的所有事情（含结果与判定）

| # | 做了什么 | 层次 | 隔离/micro | **端到端实测** | 有无提升 | 判定 |
|---|---|---|---|---|---|---|
| 1 | **config 自动调优**（重 tune MoE triton config vs 默认启发式） | 配置 | — | decode +13%、**prefill +35~54%** | ✅ 有 | **主力杠杆**（autotuning） |
| 2 | **自写 small-M(decode) MoE kernel**（跳过 align/sort、融合 act/sum、fp32 累加、tensor-core dot；详见 §1.5） | kernel | b1 **1.23×** 且更准 | b1 **+1.17%**(真信号,\|t\|=6.5) / b2 −4.3% / b4 −11.7%（n=15 t 检验）；agent c1 −0.7% / c32 −7% | ❌ ≈0，b≥2 真回归 | 通用改动不成立 |
| 3 | **shared-expert gate 融合**（linear+sigmoid+mul 三算子）| kernel | 隔离 2-3× | Qwen3 无 shared expert，**不适用**；换 Qwen1.5-MoE 测得全 batch ~1.0× | ❌ 0 | 对 Qwen3 不适用 |
| 4 | **投机解码（spec decoding）** | 算法 | — | decode **c1 +6.6% / c32 +30.6%**（exact，不改分布） | ✅ 有 | **最大可实现杠杆** |
| 5 | **decode step 组成审计**（哪块占时间） | 诊断 | — | MoE 41% + dense 32% + attn 16% = 89% memory-bound | 诊断 | 解释了为何 kernel 杠杆小 |
| 6 | **MoE HBM 带宽 vs batch** | 诊断 | b≥32 达 74–84% HBM | — | 诊断 | decode 已近内存屋顶 |
| 7 | **roofline 天花板** | 诊断 | decode 理论上界 ~1.85× | — | 诊断 | config 够不到的 memory 侧空间 |
| 8 | **线性注意力架构对比**（Qwen3 vs LFM2.5，长上下文 scaling） | 架构 | — | Qwen decode scaling +57% vs LFM +24%（bs=32, 512→8192）；Qwen bs=32×16k **OOM** | ✅（架构级，非同模型） | 选型洞察 |

---

## 1.5 附：自写 custom MoE kernel 具体改了什么（实现细节）

> 代码：`scripts/custom_moe_patch.py`。它 monkeypatch `fused_experts_impl`（Qwen3-MoE 的真实 decode 路径），仅在 **M≤4 + bf16 + gated-silu + 非量化 + shape 匹配** 时接管，否则回退 sglang 原实现（保留 fallback）。

### sglang 原路径（baseline，为大 M 吞吐优化）
1. `moe_align_block_size`：把 token 按 expert **排序/分组**成对齐 block，让每个 expert 的 token 拼成连续 tile；
2. `fused_moe_kernel` 做 w1（gate+up）grouped GEMM → 单独 `silu_and_mul` 激活 → w2（down）grouped GEMM；
3. 再做 topk 加权求和。
→ 排序 + 分组 GEMM 的开销在大 M 下被摊薄，是**吞吐最优**设计；但在 decode（M=1~4）下，排序/分组几乎是纯开销。

### 我的 custom kernel 改了 4 处
1. **完全跳过 `moe_align_block_size`**：不排序不分组。改为**按 (token, expert) pair 并行**，共 `P = M × topk` 个 program（decode b=1 只有 8 个 pair）。省掉 decode 下无收益的排序/gather 开销。
2. **kernel 1 `_w1_act`：融合 w1 GEMM + SwiGLU**。每个 pair 用 tiled `tl.dot`（沿 H 分块）同时算 gate=x·Wg 和 up=x·Wu，**在 kernel 内直接做 `silu(gate)*up`**，写出激活 [P×I]。→ baseline 是"GEMM 后再单独 silu_and_mul"，这里合成一个 kernel、少一次 [P×I] 的写回+读入。
3. **kernel 2 `_w2_sum`：融合 w2 GEMM + 路由加权 + 缩放 + 规约**。每个 pair 算 down=act·W2，乘上 `topk_weight × routed_scaling_factor`，用 **`atomic_add` 直接累加**进输出（把 topk 个专家贡献就地求和）。→ baseline 是"GEMM 后再单独做加权求和"，这里合成一个 kernel。
4. **fp32 累加**：输出张量为 fp32、`tl.dot` 与规约全程 fp32 累加，末尾再转回 bf16 → 数值比 sglang 的 bf16 路径**更准**（隔离测 max rel err ~3.95%，且更接近 fp32 参考）。
   - 补充：即使 M=1，也用 tensor-core `tl.dot`（把 M pad 到 BM=16 tile + mask `m<1`），而非退化成 gemv。

### 为什么 b=1 真赢、b≥2 真输（已用 n=15 t 检验坐实，见 §4.2）
- **b=1（+1.17%，真信号）**：只有 8 个 pair，分组无收益；跳过 align/sort + 两处融合把 decode 下的固定开销削掉。注意这**不是省 launch**（cudagraph 已隐藏），而是 kernel **GPU 计算本身更省**。
- **b≥2（−4% ~ −12%，真回归）**：sglang 的 expert 分组能让**同一 expert 的多个 token 复用一次权重加载**（Wg/Wu/W2 每 expert-tile 只读一次，摊到多 token）；而我的 per-pair 方案对每个 (token,expert) 都**重新加载权重** → 显存流量随 M 上升，加上 `atomic_add` 竞争，很快被反超。

### 一句话
custom kernel = **「去掉 align/sort + 把 GEMM/激活/加权求和融成 2 个 kernel + fp32 累加」**，专为 M=1 decode 定制；它在 b=1 拿到**真实但极小**的 +1.17%，但因为放弃了 expert 权重复用，b≥2 就是净负 → **作为通用改动不成立**。

---

## 1.6 附：我们对 kernel 做的 tuning（≠ sglang 参数，≠ kernel 重写）

> 这里澄清一个容易混的区分。我们在 kernel 层做过**两件不同的事**：
> - **A. MoE triton kernel 的 config 调优（本节）** —— 调 kernel 自己的 meta 参数，属于 **autotuning**；
> - **B. 自写 custom MoE kernel（§1.5）** —— 重写 kernel 逻辑，属于**重写**。
>
> A 调的是 **Triton kernel 的启动配置**，和 sglang server 参数（`--chunked-prefill-size` 等）**无关**。

### 1.6.1 tune 了哪个 kernel
**只有一个：`fused_moe_kernel`（Triton）** —— MoE 的 grouped GEMM（w1=gate+up、w2=down 的专家矩阵乘），Qwen3-MoE 的 decode 和 prefill 都走它。
调的 meta 参数：`BLOCK_SIZE_M / BLOCK_SIZE_N / BLOCK_SIZE_K / GROUP_SIZE_M / num_warps / num_stages`，按 `(E, N, dtype, H200, M-bucket)` 生成 tuned JSON。
- 工具：sglang 官方 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`（CUDA-graph 计时、flush L2、100 iters）。
- 脚本 / 数据：`scripts/run_v23_config_evidence.py`、`results/2026-07-19_v23_config_evidence/fused_moe_config_speedup.json`。

> ★**Triton 版本（关键，复现必读）**：我们的 tuning 基于 **Triton 3.5.1**（当前 `sglang-dev` 环境；autotune 日志显示 `target=cuda:90`=H200/SM90）。
> **但存在版本错配**：sglang 的 config 目录 `configs/triton_3_5_1/` **没有**我们 shape 的 config → 运行时虽是 3.5.1，却**回退加载为 triton 3.2.0 tune 的旧 config**（`configs/triton_3_2_0/E=128,N=768,H200.json`，覆盖全 batch），并打印 "Fallback to triton version 3.2.0 ... Performance might be sub-optimal!"。
> - config 加载机制：`get_moe_configs` 按 `triton.__version__` 找 `configs/triton_{ver}/`，找不到则遍历其他版本目录回退；支持 `SGLANG_MOE_CONFIG_DIR` 环境变量覆盖目录（见 `fused_moe_triton_config.py:76-115`）。
> - **迁移影响**：新机器若也是 triton 3.5.1，需把 tuned config 放进 `configs/triton_3_5_1/` 才会被优先加载；若 triton 版本不同，最优 block/stages 会变，**必须在新机器重新 tune**。

### 1.6.2 对标的 baseline（三个 config）
- **default**：`get_default_config` 启发式（没有任何 tuned JSON 时 sglang 用的）。
- **fallback**：旧 `triton_3_2_0/E=128,N=768,H200.json`（**sglang 今天实际加载的**——因为 triton 3.5.1 没有我们 shape 的 config，回退并打印 "Performance might be sub-optimal!"）。
- **ours**：我们对自己 shape 重新 tune 生成的（仅 batch=32；`results/autotune_qwen3_moe/`）。
- ⚠️ **注意**：§1.6.3 表里的 "tuned" 列其实是 **fallback（triton_3_2_0）** 的数字（`ours` 只在 b=32 有值，且 vs fallback 仅 +0.6%）。所以这张表实为 **"fallback tuned config vs default 启发式"**。真正 e2e 该验证的正是这个大 gap（见 §1.7）。

### 1.6.3 结果 —— tuned vs default 启发式（**kernel 时间**，U 形）

**Qwen3-30B-A3B（E=128, N=768）：**

| batch(=M) | regime | default (µs) | tuned (µs) | **kernel 提速** |
|---:|---|---:|---:|---:|
| 1 | **decode 单请求** | 34.95 | 31.01 | **1.13× (+13%)** |
| 8 | decode 小并发 | 136.76 | 130.51 | 1.05× (+5%) |
| 16 | decode | 206.09 | 199.82 | 1.03× |
| 32 | **decode 并发** | 270.69 | 260.60 | 1.04× (+4%) |
| 64 | decode | 300.50 | 298.84 | 1.01× |
| 128 | decode | 311.72 | 301.34 | 1.03× |
| 256 | **prefill 起点** | 442.30 | 326.83 | **1.35× (+35%)** |
| 512 | prefill | 495.65 | 339.38 | **1.46× (+46%)** |
| 1024 | prefill | 570.20 | 401.87 | **1.42× (+42%)** |
| 2048 | 长 prefill | 801.66 | 547.81 | **1.46× (+46%)** |
| 4096 | **长 prefill** | 1373.08 | 891.80 | **1.54× (+54%)** |

**DeepSeek-V2-Lite 交叉验证（E=64, N=1408，有 shared expert，top-6）：**

| batch(=M) | regime | default (µs) | tuned (µs) | **kernel 提速** |
|---:|---|---:|---:|---:|
| 1 | decode 单请求 | 41.99 | 37.38 | **1.12× (+12%)** |
| 256 | prefill 起点 | 438.93 | 298.22 | **1.47× (+47%)** |
| 4096 | 长 prefill | 1753.75 | 1049.54 | **1.67× (+67%)** |

### 1.6.4 per-regime 小结
- **Decode（小 M）**：单请求 b=1 **+13%（Qwen）/+12%（DeepSeek）**；并发 b=8~128 只有 **+1~5%**。
- **Prefill（大 M）**：**+35~54%（Qwen）/+47~67%（DeepSeek）**。
- **为什么 U 形**：大 M 下 kernel 是 compute-bound，好的 block config 能真正吃满算力 → prefill 收益大；decode 小 M 是 memory-bound，config 空间小 → 收益小。

### 1.6.5 三条必须说清的 caveat（诚实）
1. **这些是 MoE kernel 的隔离时间（µs），不是完整端到端 prefill/decode。** 端到端里 MoE 只占一部分（decode ~41%），所以 **e2e 提升会明显小于这些 kernel 数字**。这张表是"kernel 层"证据，不是"e2e 层"证据。
2. **大 gap 主要是 "有 tuned config" vs "default 启发式"。** 我们自己重 tune 的 vs sglang 今天加载的**旧 fallback config**，在 b=32 只快 **0.6%** → 旧回退 config 已够好，"按新 triton 版本重调"几乎无收益；真正价值是"**别掉进 default 启发式**"这个坑。
3. **这本质上是 autotuning。** 它作用在 **kernel meta 参数**层（不是 server 参数、不是重写 kernel），但仍属"配置/自动调优"范畴 → 与本报告"同模型端到端提升都来自配置/特性开关层"的结论一致。

### 1.6.6 A vs B 一句话对比
| | A. kernel-config 调优（本节） | B. custom kernel 重写（§1.5） |
|---|---|---|
| 对象 | Triton `fused_moe_kernel` 的 meta 参数 | 重写 MoE 计算逻辑 |
| 手段 | autotuning（block/warps/stages） | 去 align/sort + 融合 + fp32 |
| kernel 层 | decode +13% / prefill +54% | b1 1.23× |
| **端到端** | **prefill +34~43%（v42 实测，见 §1.7）、decode ≈0** | b1 **+1.17%**（真信号）、b≥2 真回归 |
| 结论 | ✅ prefill 有真实 e2e 收益；属 autotuning | ❌ 通用改动不成立 |

---

## 1.7 ★kernel-config tuning 的端到端验证（v42，全 regime，n=3）

> §1.6 只给了**隔离 kernel 时间（µs）**。本节补上**端到端**：那个 kernel 层 +35~54% 的 gap，到底转不转化成端到端 prefill/decode？

![kernel-config tuning e2e](../../results/2026-07-20_v42_kernel_e2e/kernel_config_e2e.png)

### 1.7.1 方法（干净 A/B）
- 用 sglang config 加载机制做开关（`fused_moe_triton_config.py`）：
  - **default（启发式）**：`SGLANG_MOE_CONFIG_DIR` 指向空目录 → `get_moe_configs` 返回 None → 用 `get_default_config`。
  - **tuned**：sglang 现状 → 加载 fallback `triton_3_2_0` tuned config（覆盖全 batch）。
- `bench_one_batch`（H200/GPU0，cudagraph ON），一次运行同时给 **prefill 吞吐 + decode 中位延迟**；MoE 的 M：prefill≈batch×input_len（大）、decode=batch（小）。
- 扫 batch∈{1,32,64} × input_len∈{256,2048,4096}，**每配置 3 次重复取中位数**。
- 脚本 `scripts/run_v42_kernel_e2e.py`；数据 `results/2026-07-20_v42_kernel_e2e/summary.json`。

### 1.7.2 结果 —— tuned config vs default 启发式（**端到端**）

| regime | prefill M | **prefill 吞吐 default→tuned** | **decode 延迟 default→tuned** |
|---|---:|---|---|
| b=1, in=256 | 256 | 7640→7614 (**−0.3%**) | 4.47→4.30ms (+3.9%) |
| b=1, in=2048 | 2048 | 28983→41651 (**+43.7%**) | 4.61→4.40ms (+4.6%) |
| b=1, in=4096 | 4096 | 40774→54729 (**+34.2%**) | 4.72→4.49ms (+5.1%) |
| b=32, in=256 | 8192 | 50594→71515 (**+41.3%**) | 9.15→8.72ms (+4.9%) |
| b=32, in=2048 | 65536 | 50042→70613 (**+41.1%**) | 9.37→8.88ms (+5.6%) |
| b=32, in=4096 | 131072 | 46842→64449 (**+37.6%**) | 10.32→10.46ms (−1.3%) |
| b=64, in=256 | 16384 | 53291→76322 (**+43.2%**) | 10.21→10.37ms (−1.5%) |
| b=64, in=2048 | 131072 | 50143→70693 (**+41.0%**) | 11.83→12.40ms (−4.6%) |
| b=64, in=4096 | 262144 | 46809→64421 (**+37.6%**) | 15.80→15.53ms (+1.7%) |

（prefill 吞吐越高越好；decode 延迟 gain 正=tuned 更快）

### 1.7.3 结论
1. **Prefill 端到端真实兑现 +34~43%**（只要 prefill M≥2048）。这是完整 prefill 阶段吞吐，**不是隔离 kernel µs** → §1.6 的 kernel gap **在 prefill 上确实转化为端到端**。
2. **为什么 prefill e2e ≈ kernel 提升（都 ~+40%）**：prefill 是 compute-bound，MoE grouped GEMM 占算力主导，config 好坏几乎 1:1 传到端到端。
3. **短 prefill（M=256）无提升（−0.3%）**：M 太小，default 与 tuned 选的 block 差不多。
4. **Decode 端到端 ≈0（−4.6% ~ +5.6%，噪声带内）**：decode 是 memory-bound、MoE 只占 41%，config 调优对 decode e2e 无实质影响（与 §1.6 预期一致）。
5. **对我们 agent 负载高度相关**：真实 agent in:out≈13:1，**prefill 主导** → 这条 kernel-config tuning 是**同模型上端到端最实在的杠杆之一**（+40% prefill），且完全属 autotuning，agent 可自动完成。

### 1.7.4 caveat
- 这里 "tuned" = sglang 现状加载的 fallback（triton_3_2_0）；"default" = 关掉所有 config 的纯启发式。**我们自己重 tune vs fallback 只 +0.6%**（§1.6.5），所以图里的 +40% 是"**有 tuned config vs 没有**"的价值，不是"我们比 sglang 更好"。真实意义：**agent 给未覆盖 shape 补 tuned config，能拿 +40% prefill**——这正是 sglang 打印 "sub-optimal!" 想让你做的事。

### 1.7.5 总 e2e（prefill+decode 合计时间，out=32）
`bench_one_batch` 的 `total_latency`（一次完整 prefill+decode 的墙钟）：

| regime | 总 e2e default→tuned | 总提升 |
|---|---|---|
| b=1, in=256 | 0.173→0.166s | +4.3% |
| b=1, in=2048 | 0.215→0.187s | **+15.1%** |
| b=1, in=4096 | 0.248→0.215s | **+15.4%** |
| b=32, in=256 | 0.510→0.450s | +13.4% |
| b=32, in=2048 | 1.604→1.209s | **+32.6%** |
| b=32, in=4096 | 3.125→2.369s | **+31.9%** |
| b=64, in=2048 | 2.988→2.242s | **+33.2%** |
| b=64, in=4096 | 6.094→4.565s | **+33.5%** |

→ 总 e2e 提升介于纯 prefill(+40%)与纯 decode(0%)之间，**取决于 in:out 比**（这里 out=32 短→prefill 主导→总提升大；output 越长 decode 占比越高，总提升越被稀释）。数据 `results/2026-07-20_v42_kernel_e2e/total_e2e.json`。

---

## 1.8 ★真实 server + agent 数据集的全 regime 端到端验证（v43）

> §1.7 是 `bench_one_batch`（单批、无排队）。本节用**真实 sglang server + `bench_serving`**，跑我们**人造 regime 全套** + **sglang agent 数据集（mooncake toolagent）**，A/B 同前（default 启发式 vs tuned config）。这是最贴近真实部署的证据。

![server e2e regimes + agent](../../results/2026-07-20_v43_server_e2e/server_e2e_regimes.png)

### 1.8.1 方法
- 每个 config 起一个 server（default 用空 `SGLANG_MOE_CONFIG_DIR`，tuned=现状 fallback），跑全部 8 个 regime 的 `bench_serving`，记录 TTFT / TPOT / E2E 延迟 / 输出吞吐。
- 脚本 `scripts/run_v43_server_e2e.py`；数据 `results/2026-07-20_v43_server_e2e/`。

### 1.8.2 结果（tuned vs default，正=tuned 更好）

| regime（in/out/并发） | TTFT | TPOT | E2E 延迟 | 输出吞吐 |
|---|---|---|---|---|
| tiny_latency (8/4/1) | −1.3% | +2.4% | −1.0% | −6.7% |
| short_in_short (128/32/16) | +7.0% | −3.7% | +7.2% | +3.2% |
| sched_overhead (128/16/64) | −0.3% | −1.1% | −9.3% | −7.2% |
| **prefill_medium (4096/16/4)** | **+34.1%** | **+15.9%** | **+23.0%** | **+19.8%** |
| **prefill_long (16384/16/2)** | **+24.2%** | +6.4% | **+24.6%** | **+19.6%** |
| decode_medium (128/512/16) | +2.4% | +4.2% | +4.1% | +4.1% |
| decode_heavy (128/1024/32) | +1.1% | +0.5% | +0.7% | +1.2% |
| **agent_toolagent (mooncake)** | **+27.1%** | **+13.8%** | **+17.5%** | +2.7% |

### 1.8.3 结论
1. **★agent 数据集（真实 toolagent workload）真实受益：TTFT +27%、TPOT +14%、E2E +17.5%**。agent 有长 prompt（prefill 重），正中 tuned config 强项 → 这是最贴近真实部署的正面证据。
2. **Prefill-heavy regime 大赢：E2E +23~25%、TTFT +24~34%**，与 v42 bench_one_batch 的 prefill +34~43% 方向一致（server 端因排队/调度略稀释）。
3. **Decode-heavy / 短序列 ≈0 或噪声**（decode_heavy 仅 +0.5~1.2%；tiny/sched 出现 −6~−9% 的小负值，属高并发短序列的调度噪声）。
4. **总结论坐实**：kernel-config tuning 的端到端收益**集中在 prefill-heavy 和真实 agent 负载**（E2E +17~25%），decode-heavy 无实质收益 —— 与"prefill compute-bound / decode memory-bound"的机理完全一致。

### 1.8.4 caveat
- 同 §1.7.4：这里 "tuned" 是 sglang 现状 fallback config，"default" 是关掉 config 的启发式；+提升 = "有 tuned config vs 没有"。
- tiny/sched 的小负值样本量小（num_prompts 少）、高并发噪声大，不宜过度解读。

---

## 2. 三张核心图（Dey 要的"tuning 以外还有多少空间"）

> 全部为 Qwen3-30B-A3B / decode / H200 / bf16 实测。文件在 `results/2026-07-20_v34_figures/`。

### 图1 — decode step 组成（`fig1_decode_composition.png`）

![decode step 组成](../../results/2026-07-20_v34_figures/fig1_decode_composition.png)

**这张图是什么**：把一步 decode 的 GPU kernel 时间按算子类型拆开的饼图/柱图。
**数据**：**MoE 41% + dense_gemm(qkv/o/lm_head) 32% + attention 16% = 89%**，其余 norm/act/sample/misc ≈ 11%。
**解析**：decode 前三大块**全是 memory-bound 的权重/KV 流式读取**（b1 下光 lm_head 单 token 就要读 vocab×hidden≈600MB 权重）。这是"为什么抠单个 kernel 的算力，端到端杠杆很小"的根因——整步本质是在**读权重**，不是在算。任何只提升算力/省 launch 的 kernel 改动，最多动 89% 里很小一角。

### 图2 — MoE 达到的 HBM 带宽 vs batch（`fig2_moe_bandwidth_vs_batch.png`）

![MoE HBM 带宽 vs batch](../../results/2026-07-20_v34_figures/fig2_moe_bandwidth_vs_batch.png)

**这张图是什么**：sglang fused_moe kernel 在不同 batch 下实际打满的 HBM 带宽百分比曲线。
**数据**：b≥32 达 **74–84% HBM**（近内存屋顶，无损 kernel 空间 <1.3×）；b=4096 掉到 **29%**（此时转 compute-bound，即 prefill 区）。
**解析**：**decode = memory-bound**（kernel 已近内存屋顶，config 和 kernel 都难再压）；**prefill = compute-bound**（另一套故事，config-tuning 已在这里拿到 +50%）。所以"还有多少空间"这个问题**必须按 regime 分开答**——decode 和 prefill 的瓶颈根本不同。

### 图3 — ★headroom BEYOND tuning（`fig3_headroom_beyond_tuning.png`，核心图）

![headroom beyond tuning](../../results/2026-07-20_v34_figures/fig3_headroom_beyond_tuning.png)

**这张图是什么**：以 **best-tuned config 为 baseline（=1.0×）**，展示"在把 config 调到最优之后，别的手段还能再拿多少"的分组柱状图（decode，exact 方法）。
**数据**：

| 手段 | 单请求 c=1 | 并发 c=32 |
|---|---|---|
| best-tuned config（baseline） | 1.00× | 1.00× |
| + kernel 重写（实测 e2e） | **+1.5%** | —（未测/预期更低）|
| + spec decoding（实测 e2e，exact） | **+6.6%** | **+30.6%** |
| roofline 天花板（理论上界，exact） | 1.85× | 1.85× |

**解析（一句话）**：config 调到平台期后，decode 理论上还有 ~**1.85×** 空间（全在 memory 侧，config 够不到）；**spec decoding 已实测兑现 +6.6%(c1)/+30.6%(c32)，是目前最大可实现杠杆**（它一次验证多 token，同时摊薄 MoE+dense+lm_head+attn 全部 89% 的 memory 读取）；而**纯 kernel 重写只兑现 +1.5%**（因为 MoE 仅占 41%，且 sglang kernel 已近内存屋顶）。

---

## 3. 补充图 — 线性注意力如何"扩大" tuning 以外的空间（`results/2026-07-20_v39_ctxscan/ctx_scaling.png`）

![线性注意力 context scaling](../../results/2026-07-20_v39_ctxscan/ctx_scaling.png)

**这张图是什么**：两张并排子图。左：Qwen3-30B（全注意力）vs LFM2.5-8B（混合线性注意力）**decode 每步延迟 随上下文长度**曲线（batch=32）；右：同数据**归一化到 ctx=512** 后的 scaling 因子。
**数据**：

| context | Qwen3(ms) | LFM(ms) | Qwen 归一 | LFM 归一 |
|---:|---:|---:|---:|---:|
| 512 | 8.42 | 5.44 | 1.00× | 1.00× |
| 2048 | 8.68 | 5.83 | 1.03× | 1.07× |
| 8192 | 13.25 | 6.74 | **1.57×** | **1.24×** |
| 16384 | **OOM** | (单发 prefill 亦 OOM) | — | — |

**解析**：全注意力每生成一个 token 要**读全部历史 KV cache**，上下文越长 decode 越慢、显存越涨 → Qwen 延迟 +57%、且 bs=32×16k **直接 OOM**。线性注意力把历史压进**固定大小 O(1) 状态**，decode 几乎不随上下文涨 → LFM 只 +24%、显存足迹小仍能跑。**这是 tuning 和 kernel 重写都够不到的架构级杠杆**——但注意它是"**换一类模型**"，不是把 Qwen 本身变快（Qwen3 与 LFM 是不同模型、质量不同）。

---

## 4. 关键分析与教训

### 4.1 为什么 kernel 重写在 Qwen 上端到端 ≈0
1. **decode 是带宽墙**：89% 时间在流式读权重/KV（图1），kernel 省的是算力/launch，不是读带宽。
2. **sglang kernel 已近内存屋顶**：b≥32 打满 74–84% HBM（图2），无损空间 <1.3×。
3. **cudagraph 已吃掉 launch 开销**：融合省的那点 launch，cudagraph 早已隐藏 → 端到端无感。
4. **MoE 只占 41%**：即便 MoE kernel 拿到隔离 1.23×，摊到整步也只剩个位数,且我的按-pair 处理在 b≥2 被 sglang 专家分组反超。

### 4.2 方法学教训（已固化）
- **单点端到端会误导**：自写 MoE kernel 隔离 1.23×、"M≤4 都赢"，一扫 regime 就被证伪（b4 −11%）。**必须扫 batch/context/并发 + 真实 server 负载**。
- **信号 vs 噪声必须用统计检验**（Chendi 要求，已做）：把 b1 的 "+1.4%" 用 **n=15 交错重复 + Welch t 检验**验证 → **+1.17%，\|t\|=6.51，是真信号（非波动）**，但 b2 −4.3%(\|t\|=3.2)、b4 −11.7%(\|t\|=9.9) 是**真回归**。见 `noise_verification_custom_moe_b1.md`。→ 以后每个改动都应过"多次重复 + t 检验"闸门。
- **必须对标 sglang 真实 GPU 代码 + cudagraph**：早期用朴素 PyTorch baseline 得出过误导性"SwiGLU 加速"，已撤回。
- **所有数字自测**，不采信 PR 自称。

### 4.3 提升来源总分类（回答"是不是全靠 autotuning"）
| 来源 | 端到端 | 是 autotuning? | 同模型变快? |
|---|---|---|---|
| config 调优 | prefill +35~54% | ✅ | ✅ |
| spec decoding | decode c32 +30.6% | ❌（算法层） | ✅（开特性） |
| 线性注意力架构 | scaling +24 vs +57% | ❌ | ❌（换模型） |
| 重写 bf16 MoE kernel | ≈0 | — | ✅ 但无效 |

**结论**：同模型上可复现的端到端提升 = **config 调优 + 开启 spec decoding**；kernel 层证伪 ≈0；架构层是选型洞察。若把"开特性"也算进广义配置，则**目前所有同模型端到端提升都来自"配置/特性开关层"，没有一个来自我们自写 kernel**。

---

## 5. 对项目定位的启示
- 若 agent 目标 = **"自动把某模型在某机器上调到最优"** → 价值在**穷举 config + 特性开关空间**（config tuning + spec + chunk/并发），这本身是有价值的产品，且我们已有正面证据。
- 若 agent 目标 = **"发现 kernel 级新提升"** → 在成熟 bf16 模型上 payoff 很低（本报告证据）；kernel 空间集中在**新架构 / AMD / 量化 / sglang 未覆盖的边角**。

## 6. 相关文档与产物
- 本报告：`docs/2026-07-20/qwen_optimization_full_report.md`
- 图：`results/2026-07-20_v34_figures/fig1_3*.png`、`results/2026-07-20_v39_ctxscan/ctx_scaling.png`
- 全 regime 扫描 + 最终矩阵：`docs/2026-07-20/regime_sweep_kernel_changes.md`
- **噪声验证（n=15 + t 检验，b1 +1.17% 真信号）：`docs/2026-07-20/noise_verification_custom_moe_b1.md`**
- 新架构线性注意力：`docs/2026-07-20/new_architecture_linear_attention_e2e.md`
- 图说明（Dey）：`docs/2026-07-20/headroom_beyond_tuning_figures.md`
- kernel 攻坚全过程：`docs/2026-07-20/kernel_optimization_attempt_log.md`
- config-tuning 验证：`docs/2026-07-19/pr_validation_report.md`
- 自写 kernel / patch：`scripts/custom_moe_patch.py`、`scripts/serve_with_patch.py`
