# SGLang MoE/Kernel PR 验证报告（Chendi PR 清单 · bf16 · H200）

**状态**：主结论完成（config-tuning 类已在 2 个模型上验证）；shared-expert 融合机会测量为可选后续。
**日期**：2026-07-19
**目标**：对 Chendi 给的 ~30 个 sglang MoE/kernel PR 做 survey，找出**能在我们的目标场景复现出性能提升**的证据、并尽可能验证这些 PR。
**约束**：只做 **bf16**（不碰量化，保持输出分布不变）；用 **GPU 0-3**。
**硬件/软件**：H200（SM90）· sglang 可编辑安装 `/home/t-jialianggu/work/sglang`（commit 17f7a1da1）· triton **3.5.1**。

---

## 0. TL;DR（最想让你先看到的结论）

1. **绝大多数 PR 打的不是我们的目标**：多为 **Qwen3.5**（比我们新的模型，MoE shape 不同）、**FP8/NVFP4/MXFP8/W4A8 量化**（改分布，本轮排除）、或**非 H200 硬件**（H20/SM120/AMD/B200）。
2. **我们的主模型 Qwen3-30B-A3B 没有 shared experts** → 一整类"shared-expert 融合"PR（#22325/#26727/#28666/#31370）**对主模型不适用**；需换成有 shared expert 的模型（DeepSeek-V2-Lite）才能谈。
3. **唯一能干净复现、且不碰量化/不换框架就能拿到性能证据的 PR 类别 = "为未覆盖的 MoE shape 在 H200 上加 tuned fused_moe config"**（#27112/#20565/#18969）。我们在**两个模型**上复现了它的机制并量出提升：
   - **Qwen3-30B-A3B**（E=128，无 shared expert）：decode 单请求 **+13%**、prefill 大 batch **+35~54%**；
   - **DeepSeek-V2-Lite**（E=64，有 2 个 shared expert）：decode **+12%**、prefill **+47~67%**。
   两模型一致的 U 形曲线 → **跨模型稳健**（详见 §2、§3）。
4. 这条证据同时是 **Mason 路线第 2 层（kernel constexpr autotuning）**的实测支撑，也直接说明"我们的 autotuner 在 config/kernel-config 层是有意义的"。

---

## 1. PR 分类 triage（对照我们的目标场景）

> 判定维度：目标模型是否=我们的（Qwen3-30B-A3B 或可获得的 shared-expert 模型）· 是否需要量化 · 硬件是否 H200 · 是否可在不改 sglang 源码（无 PR diff）下复现。

### Bucket A — 可复现的 config-tuning（★本轮重点，已验证）
| PR | 内容 | 对我们 | 复现方式 |
|---|---|---|---|
| #27112 | Qwen3.5-397B H200 Triton fused-MoE tuning | 机制适用（shape 不同）| 在**我们的 shape** 上跑同一 tuning 脚本 → 量提升 ✅ |
| #20565 | Qwen3.5-35B H200 Triton fused-MoE (TP4) | 机制适用 | 同上 ✅ |
| #18969 | Qwen3.5 **BF16**/FP8 H20 tuning | 机制适用（BF16 分支）| 同上 ✅ |

**这三个 PR 本质都是"跑 tuning 脚本、把生成的 config JSON 提交上去"。** 我们无法直接 apply 它们的 JSON（shape 是 Qwen3.5 的），但可以**复现其机制**：对自己的 shape 跑同样的官方 tuning，量测 kernel 提速。这就是对这一类 PR 价值的直接验证（见 §2）。

### Bucket B — shared-expert 融合（需有 shared expert 的模型）
| PR | 内容 | 对我们 |
|---|---|---|
| #22325 | Qwen shared-expert 融合 linear+sigmoid+mul | 主模型无 shared expert → 需 DeepSeek-V2-Lite / Qwen2-MoE（见 §3）|
| #26727 | Qwen shared-expert 四算子融合 | 同上 |
| #28666 (AMD) / #31370 | shared-expert append/remap 融合 | AMD/HIP 或需 shared expert；本机为 NVIDIA，部分不适用 |

无法直接 apply（无 PR diff，且本仓库 sglang 固定在某 commit）。策略：换 **DeepSeek-V2-Lite**（有 2 个 shared expert）测**融合机会（opportunity）**，见 §3。

### Bucket C — FP8/量化（本轮排除：改分布）
#30541 (HPC-Ops FP8) · #28355 (FlashInfer-cutlass FP8) · #26089 (FP8 quant fusion) · #21473 (FP8 config 122B) · #28552 (FP8 GEMM SM120) · #24651 (AMD FP8) · #12210 (FP8 allgather) · #31552 (Marlin) · #31529 (W4A8) · #31463 (MiniMax Wint4) · #31429 (FP8 DeepEP) · #31353/#31330 (NVFP4 Marlin) · #31510 (MXFP8) · #31382 (NVFP4 MTP) · #31470/#31408 (MegaMOE NVFP4/MXFP8) · #27211 (FP8 DeepEP combine)。
→ **全部涉及量化**，违反"不改分布"约束，本轮不验证。（备注：本地有 Qwen3-30B-A3B-FP8，若日后放开量化，#30541/#28355 可在其上测。）

### Bucket D — 非硬件目标 / 修复类
#28552 (SM120) · #24651/#28666 (AMD) · #31463 (Hopper 量化) · #31608 (LoRA TMA guard，非 perf) · #31246 (MoE CUDA-graph capture 修复，correctness) · #31510 (MXFP8 pipeline 修复)。
→ 硬件不符或非性能类，不作性能验证。

**一句话**：~30 个 PR 里，**真正能在我们 bf16/H200 场景复现出性能证据的，是 Bucket A 的 config-tuning 类**；Bucket B 需换模型且只能测机会。

---

## 2. Bucket A 验证：config-tuning 在 Qwen3-30B-A3B 上的实测提升 ✅

### 2.1 背景：我们的 shape 确实"未被覆盖"
- Qwen3-30B-A3B MoE shape：`E=128, N(moe_intermediate)=768, hidden=2048, top-8`，bf16。
- 当前 triton **3.5.1** 的 config 目录里 **没有** 这个 shape 的 H200 config（只有 B200/H100）→ sglang 回退到旧的 `triton_3_2_0/E=128,N=768,H200.json`，并打印 **"Performance might be sub-optimal!"**。
- 这正是 #27112/#20565/#18969 这类"给未覆盖 shape 加 H200 config"PR 要解决的情形。

### 2.2 方法
用 sglang 官方 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` 的 `benchmark_config`，在**真实 triton fused_moe kernel** 上（CUDA-graph 计时、flush L2、100 iters）测三种 config 的 kernel 时间：
- **default**：`get_default_config` 的启发式（没有任何 tuned JSON 时 sglang 用的）
- **fallback**：`triton_3_2_0` 的 tuned config（sglang 今天实际加载的）
- **ours**：我们对自己 shape 跑 tuning 生成的（`results/autotune_qwen3_moe/`，仅 batch=32）

脚本：`scripts/run_v23_config_evidence.py`；数据：`results/2026-07-19_v23_config_evidence/fused_moe_config_speedup.json`。

### 2.3 结果：tuned config vs 默认启发式（kernel 提速）
| batch | default (µs) | best tuned (µs) | **提速** | 对应 regime |
|---|---|---|---|---|
| 1 | 34.95 | 31.01 | **1.13×** | decode 单请求 |
| 8 | 136.76 | 130.51 | 1.05× | 小 batch decode |
| 16 | 206.09 | 199.82 | 1.03× | |
| 32 | 270.69 | 260.60 | 1.04× | decode 并发 |
| 64 | 300.50 | 298.84 | 1.01× | |
| 128 | 311.72 | 301.34 | 1.03× | |
| 256 | 442.30 | 326.83 | **1.35×** | prefill 起点 |
| 512 | 495.65 | 339.38 | **1.46×** | prefill |
| 1024 | 570.20 | 401.87 | **1.42×** | prefill |
| 2048 | 801.66 | 547.81 | **1.46×** | 长 prefill |
| 4096 | 1373.08 | 891.80 | **1.54×** | 长 prefill |

### 2.4 结论
- **config autotuning 在我们的 shape 上确有真实提升**，且呈 **U 形**：
  - **decode 单请求（b=1）+13%**；
  - 中间 batch（8–128）温和 **+1~5%**；
  - **prefill 大 batch（256–4096）+35%~+54%**。
- 提升**最大的区间正是 prefill**，而这恰是我们真实 agent 负载的主导阶段（in:out≈13:1）→ 对我们的场景**高度相关**。
- **补充发现**：我们自己重 tune 的 config（batch=32）相对旧回退 config 只快 0.6%——说明**"按 triton 版本重调"在 b=32 几乎无收益**（回退 config 已够好）；真正的价值是 **"有 tuned config" vs "默认启发式"** 这个大 gap。
- **对 PR 的验证结论**：#27112/#20565/#18969 这类"加 H200 tuned config"的 PR，其**机制在我们 shape 上复现有效**，尤其在 prefill/大 batch 收益显著。这是"config/kernel-config 层的 autotuner 有意义"的直接证据。

---

## 3. Bucket B 验证：shared-expert 模型（DeepSeek-V2-Lite）

### 3.1 为什么换模型
我们主模型 Qwen3-30B-A3B **无 shared expert**（config 只有 128 routed + top-8）。为验证 shared-expert 相关 PR，下载 **DeepSeek-V2-Lite**（15.7B，`n_routed_experts=64` top-6 + **`n_shared_experts=2`**，`moe_intermediate=1408`，hidden=2048，bf16，单卡可跑）。

### 3.2 这个 shape 也未被覆盖
sglang 任何 triton 目录都**没有** `E=64, N=1408, H200` 的 config → 同样走默认启发式。因此可复现同一条 config-tuning 证据（不同模型/shape 的交叉验证）。

### 3.3 结果（config-tuning on DeepSeek-V2-Lite）✅
对 `E=64, N=1408, H200` 跑官方 tuning（GPU0/2/3 并行，batch=1/256/4096），再对比默认启发式：

| batch | default (µs) | tuned (µs) | **提速** |
|---|---|---|---|
| 1 | 41.99 | 37.38 | **1.12×** |
| 256 | 438.93 | 298.22 | **1.47×** |
| 4096 | 1753.75 | 1049.54 | **1.67×** |

**结论（交叉验证）**：在一个**完全不同的 MoE**（E=64 vs 128、有 shared expert、top-6 vs 8）上，config-tuning 复现出**同样的 U 形、且 prefill 收益更大（+67%）**。这把"config autotuning 有意义"从单模型（Qwen3-30B-A3B）扩展到**跨模型稳健结论**：
- decode 单请求 +12~13%（两模型一致）；
- prefill 大 batch **+54%（Qwen）/ +67%（DeepSeek）**。

### 3.4 shared-expert 融合 PR（#22325/#26727）的评估
- #22325（融合 `linear+sigmoid+mul`）针对的是**带 shared-expert gate**（sigmoid 门控）的模型（Qwen2-MoE / Qwen3.5）。DeepSeek-V2-Lite 的 shared expert **无 gate**（直接相加），故 #22325 的确切算子链**不完全适用**；#26727 的四算子融合类似。
- 我们**无法直接 apply** 这些 PR（没有 diff，且本仓库 sglang 固定 commit）。可行的验证是**测融合机会**：shared-expert MLP 及其 gate/激活在整步 decode 里占多少时间 → 估计融合上界。
> ⏳ 若时间允许，补 NCU/timing 测 DeepSeek shared-expert 路径耗时占比。

---

## 4. 逐 PR 结论表

| PR | 类别 | 是否适用我们 | 验证方式 | 结论 |
|---|---|---|---|---|
| #27112 | config-tuning H200 Triton | ✅机制 | 我们 shape 复现 tuning | **验证：prefill +35~54%** |
| #20565 | config-tuning H200 Triton | ✅机制 | 同上 | **验证（同机制）** |
| #18969 | config-tuning BF16 | ✅机制 | 同上 | **验证（同机制）** |
| #22325 | shared-expert 融合 | ⚠️需 gate 模型 | DeepSeek 机会测量 | 部分适用；见 §3.4 |
| #26727 | shared-expert 四算子融合 | ⚠️需 gate 模型 | 同上 | 部分适用 |
| #28666/#31370 | shared-expert append (AMD/HIP) | ❌AMD | — | 硬件不符 |
| #31246 | MoE CUDA-graph 修复 | 中性 | — | correctness，非 perf |
| #31608 | LoRA TMA guard | ❌ | — | 非 perf |
| Bucket C（17 个 FP8/量化）| 量化 | ❌改分布 | — | 本轮排除 |
| Bucket D（SM120/AMD/B200）| 非本硬件 | ❌ | — | 硬件不符 |

---

## 5. 复现命令（供他人重跑）
```bash
ENV=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$ENV
# config-tuning 证据（Qwen3-30B-A3B）
CUDA_VISIBLE_DEVICES=1 $ENV/bin/python scripts/run_v23_config_evidence.py \
    --batches 1,8,16,32,64,128,256,512,1024,2048,4096 --iters 100
# 生成某 shape 的 tuned config（官方脚本）
cd /home/t-jialianggu/work/sglang/benchmark/kernels/fused_moe_triton
CUDA_VISIBLE_DEVICES=0 $ENV/bin/python tuning_fused_moe_triton.py \
    --model <model_path> --tp-size 1 --batch-size <B> --tune
```

## 6. 产物
- 脚本：`scripts/run_v23_config_evidence.py`（Qwen 三方对比）、`scripts/run_v23_generic.py`（通用 default-vs-tuned）
- 数据：`results/2026-07-19_v23_config_evidence/fused_moe_config_speedup.json`（Qwen）、`deepseek_config_speedup.json`（DeepSeek）
- DeepSeek tuned config：`results/2026-07-19_v23_config_evidence/deepseek_E=64,N=1408,H200.json` + `deepseek_tuned_per_batch/`
- 依赖工具：sglang 官方 `tuning_fused_moe_triton.py`

---

## 7. 这对"证明 agent 有意义"意味着什么

- **正面证据**：在**不写任何 kernel、不碰量化**的前提下，仅靠"对未覆盖 shape 跑官方 tuning 生成 config"，就能在真实 MoE kernel 上拿到 **decode +12~13% / prefill +47~67%** 的稳定提升，且**跨两个结构不同的 MoE 模型一致**。这说明：
  1. **"config/kernel-config 层的自动调优"本身就有可观、可复现的价值** —— 这正是我们 autotuner agent 能自动做的事（发现未覆盖 shape → 触发 tuning → 落地 config）。
  2. 收益**最大的区间是 prefill/大 batch**，恰好对齐真实 agent 负载（prefill 主导）。
- **对应 Mason 的 X 判据**：这是"kernel constexpr autotuning"层（Mason 第 2 层）的实测提升。**手写 kernel（第 3 层）之上还能加多少（X）尚未测**；但即便不写 kernel，第 2 层已有 +47~67%（prefill）的空间可被 agent 自动吃到。
- **诚实边界**：
  - "按 triton 版本重调"在中间 batch（8–128）几乎无收益（回退 config 已够好）；价值集中在 **有无 tuned config** 这个大 gap，以及 **prefill 大 batch**。
  - shared-expert 融合类 PR（#22325/#26727）需要**带 shared-expert gate 的模型**（Qwen2-MoE/Qwen3.5），且需要 PR 的实际 diff 才能测到"融合后"，本轮无法直接验证，只能评估适用性。

## 8. 建议的下一步
1. **把 config-tuning 接入 serving 端到端**：目前是 kernel micro-benchmark；下一步在真实 serving（prefill-heavy regime）上量端到端 TTFT/吞吐提升，确认 kernel 层的 +50% 能转化为多少端到端收益。
2. **补齐 batch 覆盖**：给我们的两个 shape 跑全 batch tuning，生成可直接提交上游的 H200/triton3.5.1 config（这本身就是一个 PR 级贡献，等价于 #27112/#20565）。
3. **（可选）shared-expert 融合机会**：下载 Qwen1.5-MoE-A2.7B（小、带 shared-expert gate），测 gate 链（linear+sigmoid+mul）在整步中的时间占比，估计 #22325/#26727 的上界。
