# 三级优化方法论：在新模型上复现 LFM2.5 结果的操作手册

**日期**：2026-08-04
**目的**：把 LFM2.5 上跑通的三级优化流程固化成可执行方法，让另一个 agent 在
OLMo-2 等模型上复现，目标是 **2-3 个模型的可比结果 → 可发表**。

**读者**：接手在新模型上执行的 agent。**本文默认你没有 LFM2.5 的上下文。**

---

## 0. 我们要复现的是什么

> **⚠️ 2026-08-04 更正（用户澄清）**：本节原先写的是「不是优化一个模型，是复现一个论断」，
> 并把成功标准定为「L3 增量那一行横着看是否也平」。**那个框架太窄，会误导接手的 agent。**
>
> **真正的目标是复现「整套发现流程」本身**：在一个没人看过的模型上，
> 从审计出发**找到新的优化机会**、实现、验证。
> 也就是这个项目最初的命题 —— **这套 agent 流程能不能在新模型上再产出一次价值。**

### 两种框架的差别（会直接改变你怎么排优先级）

| | ❌ 窄框架（原文） | ✅ 实际目标 |
|---|---|---|
| 成功标准 | L3 增量跨基线是否恒定 | **有没有找到并落地新的优化点** |
| 重心 | 阶段 5（2³ 矩阵，~20h/模型） | **阶段 0/2/3：审计 → 读码 → 实现** |
| 选模型 | 选能撑起矩阵的 | **选没人探过、headroom 未知的** |
| 矩阵的角色 | 主交付物 | **最后的验证；没找到东西就不该建** |
| 失败长什么样 | 那一行不平 | **审计找不到真 gap，或找到的修不动** |

**最关键的推论**：窄框架下 `olmoe`（L3 headroom 仅 4.70%）是好选择，因为它能补矩阵的格子；
**实际目标下它是差选择**，因为几乎注定找不到新机会。

> **给 L3 是空的模型建 8 个格子，是在给一个不存在的效应做精密测量。**
> **先审计，找到东西再决定要不要建矩阵。**

### 那个论断仍然是有用的参照（但不是目标）

LFM2.5 上（4 个 regime 独立支撑）：

> 把 serving config 和 kernel config 都调到实测上限之后，
> kernel rewrite 仍然贡献 +6.4% ~ +8.4%，全部统计显著。

| L3 增量叠在 | cookbook | L2 | L1 | **L1+L2** |
|---|---:|---:|---:|---:|
| A 低批 decode | +6.70% | +6.35% | +7.46% | **+7.35%** |
| B 并发 decode | +6.72% | +6.86% | +6.62% | **+7.14%** |
| C 长 prefill | +6.18% | +9.73% | +6.26% | **+6.38%** |
| D medium | +8.29% | +8.13% | +8.40% | **+8.38%** |

**横着看几乎是常数** —— 这是 LFM2.5 上的结论。
**如果新模型上也做了矩阵且这一行也平，那是加分；不平也是有价值的结果。
但这不是你去新模型上要回答的第一个问题。**

---

## 1. ⚠️ 开工前必读：三层不是都适用于所有模型

| 层 | 是什么 | 适用条件 | OLMo-2（dense）能做吗 |
|---|---|---|---|
| **L1** serving config tuning | 4 个 server 旋钮的全网格 | **总是适用** | ✅ |
| **L2** kernel config tuning | 热 kernel 的 tile 参数 | **需要最热的 kernel 的 tile 是「可改的」** —— 见 §0.3 的三种形态 | ⚠️ **不能照抄** |
| **L3** kernel rewrite | 改模型代码 | 需要先审计出 gap | ✅ |

### ★ L2 在 OLMo-2 上的问题

LFM2.5 的 L2 调的是 **`fused_moe_kernel` 的 tile 参数**（`E=32, N=1792`）。
sglang 用 JSON config 文件驱动这个 kernel，所以「换 config」是一个干净的开关。

**OLMo-2 是 dense 模型，`architectures: Olmo2ForCausalLM`，没有 MoE。**
→ **那个 config 机制根本不存在。**

**必须先做的一步**：profile 一次，找出 dense 模型上**最热且 config 可调**的 kernel。
候选：

- `nvjet_*` / cuBLAS GEMM —— 厂商闭源，**确实不可做**
- Triton attention backend 的 tile 参数（若用 `--attention-backend triton`）
- flashinfer 的 autotune 开关
- **★ 任何硬编码 `tl.constexpr` tile 且无 `@triton.autotune` 的热 Triton kernel**
  —— 这一条 2026-08-04 才补上，在 Falcon-H1 上值 **端到端 +27.63%**。
  linear-attention / mamba / SSM 这类较新的算子尤其容易中招，因为它们的
  Triton 实现常常是从参考实现直接搬来的，默认值从没为任何卡调过。

> ⚠️ **本节标题「L2 在 OLMo-2 上的问题」在 2026-08-04 被部分推翻。**
> 结论对 OLMo-2 本身仍然成立（它最热的是 cuBLAS GEMM），
> 但**推理过程「没有 MoE → 没有 config 机制 → L2 不适用」是错的**。
> Falcon-H1 同样没有 MoE，却有三个占 59% prefill 时间、tile 硬编码为 16 的
> Triton kernel。**判据在 §0.3。**

**如果找不到可改 tile 的热 kernel，就诚实记录「L2 在这个模型上不适用」，
矩阵变成 2² 而不是 2³。** 这本身是个发现：**L2 这一层的适用性依赖架构，
而 L1/L3 不依赖。**

> **不要为了凑齐三层而编一个 L2。** 一个诚实的 4 格矩阵比一个勉强的 8 格有价值。

---

## 2. 完整流程（六个阶段）

```
阶段 0  可行性勘察          ~1h    决定 L2 做不做，选定 regime
阶段 1  L1 serving 上限     ~6h    192 全网格 + 验证 pass
阶段 2  L3 gap 审计         ~1h    ★ 最高杠杆，纯静态+一次 profile
阶段 3  L3 实现与隔离验证    ~4h    每个组件独立测
阶段 4  L2（若适用）        ~4h
阶段 5  ★ 2³ 全因子矩阵     ~7h    每 regime 两次调用
```

**总计约 20-25 GPU 小时**，多卡并行可压到 6-8 小时挂钟。

---

## 阶段 0：可行性勘察（先做，别跳）

### 0.1 模型能不能跑起来

```bash
python -m sglang.launch_server --model-path <PATH> --port <P> --tensor-parallel-size 1
```

**常见坑**（都遇到过）：
- `context-length` 超过模型的 `max_position_embeddings` → sglang 直接拒绝。
  OLMo-2-1B 必须 `ctx=4096`（已记录在 `serving_ceiling_lib.MODELS`）
- 需要 `--trust-remote-code`
- 小模型（1B）跑得极快，**测量窗口会很短** → 见 §5.3 的噪声警告

### 0.2 注册模型

两个 harness 各注册一次（OLMo-2 **两处都已注册**，可直接用）：

| 文件 | 位置 | 作用 |
|---|---|---|
| `scripts/serving_ceiling_lib.py` | `MODELS` | L1 用 |
| `scripts/lfm_fusion/lf_lib.py` | `MODELS` | L3 审计/A-B 用 |

### 0.3 ★ 决定 L2 做不做

```bash
python scripts/lfm_fusion/lf_audit.py --model <M> --regime C_long_prefill --gpu <G>
```

> **⚠️ 2026-08-04 修正**：本节原先只问「有没有**用户可换的 config**（如 fused-MoE 的
> JSON）」，并据此断言 dense 模型上 L2 不适用。**这个判据太窄，在 Falcon-H1 上漏掉了
> 本轮最大的一个发现（端到端 +27.63%）。**

看 kernel 时间构成，然后问**正确的那个问题**：

> **最热的那个 kernel 的 tile 参数是谁定的？有没有人为这张卡定过？**

三种形态，**前两种都能做 L2**：

| 形态 | 长什么样 | 例子 | L2 |
|---|---|---|:--:|
| ① 外部 config 文件 | JSON 按 `E,N,device_name` 查表 | LFM2.5 的 fused-MoE，H200 缺文件 | ✅ |
| ② **硬编码 constexpr 默认值** | `BLOCK_SIZE_M: tl.constexpr = 16`，无 `@triton.autotune`，调用点不传 | **Falcon-H1 的 mamba SSD kernel** | ✅ **本轮新增** |
| ③ 厂商闭源 kernel | cuBLAS / `nvjet_*` / cutlass | 大多数 dense GEMM | ❌ |

**形态 ② 的检查方法**（对最热的 Triton kernel）：

```bash
# 1. tile 参数有没有默认值？
grep -n "BLOCK_SIZE.*tl.constexpr.*=" <kernel_file>.py
# 2. 有没有 autotune？
grep -n "@triton.autotune" <kernel_file>.py
# 3. ★ 调用点传不传？（这一条最关键，很多人只查前两条）
grep -A30 "<kernel_name>\[" <kernel_file>.py | grep -c "BLOCK_SIZE"
```

**三条都是「默认值 / 无 / 0」→ 这个 kernel 从来没人为任何卡调过 → L2 可做。**

Falcon-H1 实测：`_chunk_state_fwd` / `_chunk_scan_fwd` / `_state_passing_fwd`
三条全中，合计 **59% 的 prefill kernel 时间**跑在 16×16×16 上。
换成 64×64×64 后**端到端长 prefill +27.63%，TTFT −32.4%，输出逐 token 相同**。
细节见 `docs/2026-08-04/pipeline_replication_olmo2_falconh1.md` §3。

> **不要为了凑齐三层而编一个 L2**（原则不变），
> **但也不要因为「没有 JSON config」就放过形态 ②。**

**注入方式**：`scripts/lfm_fusion/ssd_inject/sitecustomize.py` 演示了怎么在不改
上游源码的前提下覆盖 tile —— 包住 kernel 对象，在 launch 时补 kwargs。
比手写 microbench 可靠得多（那些 kernel 有 30+ 个 stride 参数，重构极易出错），
而且 grid lambda 读 `META["BLOCK_SIZE_M"]` 会自动适配。

⚠️ **扫 tile 时必须丢弃每个配置的第一次运行** —— 首次要付 Triton 编译
（实测 1.1–1.8 s vs 稳态 0.68 s）。若 stock 配置已被别的实验编译过，
不丢首次就是**拿冷配置比热配置**，本轮第一版数据就因此作废。

### 0.4 选 regime

沿用现有的 6 个（`serving_ceiling_lib.WORKLOADS`）：
`R_short_decode` / `R_medium_balanced` / `R_long_prefill` / `R_concurrent_decode` /
`shared_prefix` / `tool_agent`。

**不要改 workload 定义** —— 跨模型可比性依赖于此。

---

## 阶段 1：L1 serving 上限

### 做法

4 个旋钮的**全网格穷举**（不是采样）：

| 旋钮 | 取值 |
|---|---|
| `max_running_requests` | 8, 16, 24, 32, 48, 64, 96, 128 |
| `chunked_prefill_size` | −1, 2048, 8192 |
| `schedule_policy` | lpm, fcfs |
| `mem_fraction_static` | 0.75, 0.80, 0.85, 0.90 |

**8 × 3 × 2 × 4 = 192。** 再对前 35 个做 **5 重复验证 pass**。

```bash
python scripts/run_serving_ceiling_campaign.py --init --models <M>
python scripts/run_serving_ceiling_campaign.py --gpu <G> --worker w<G>
```

### ★ 为什么必须穷举而不是 TPE

我们**试过 TPE 并且失败了**：25 次 trial 的 Optuna 研究，前 7 个 trial 把
`triton MoE` 和差 batching 绑在一起，之后 18 个 trial 再没试过好的组合，
**报出的 ceiling 比 cookbook 还低 6%**。

审稿人会问「这不是 ceiling，是搜索失败」。**穷举没有这个问题**：
*the whole space is enumerated, so no sampling bias exists at all*。

### ★ 必须同时记录延迟

**只记吞吐会得出错误结论。** LFM2.5 上：

- 长 prefill 吞吐 +56.9%，**TTFT p95 同时从 208ms 降到 94ms**
- shared prefix 吞吐 +93.6%，**TTFT p95 从 7.4s 降到 389ms**

**这些不是「拿延迟换吞吐」，是三赢。** 如果只看吞吐会误判为 trade-off。

反过来在真实 trace 上，吞吐可能完全不动而延迟大幅改善（见 §5.4）。

### 输出

`results/<date>_serving_ceiling_validation/analysis/<model>/ceiling_per_regime.json`
—— 每个 regime 的最优旋钮 + cookbook 基线 + 三个指标。**阶段 5 要用。**

---

## 阶段 2：★ L3 gap 审计（最高杠杆的一步）

**这一步的产出决定后面所有工作。做对了 4 小时能拿到 +6%，做错了会朝错的方向优化一整天。**

### 2.1 换一个计数口径 —— 这是核心

**❌ 不要数时间占比。** 时间占比会告诉你「MoE 占 70%」，然后你去优化 MoE，
**但那里通常已经没有空间了**（它是所有人都在优化的东西）。

**✅ 数「一个融合实现根本不会启动的 kernel」的个数。**

```bash
python scripts/lfm_fusion/lf_audit.py --model <M> --regime <R> --gpu <G>
python scripts/lfm_fusion/lf_audit.py --model qwen  --regime <R> --gpu <G>   # ← 对照组
```

方法：`bench_one_batch --profile` + **关闭 CUDA graph**（让每个算子单独现形）
→ 按 kernel 名分桶。

### 2.2 ★★ 必须有对照组

LFM2.5 的结果：

| 模型 | 未融合 RMSNorm | 独立 residual add | gating mul |
|---|---:|---:|---:|
| **LFM2.5** | **61** | **48** | **36** |
| Qwen3-30B（对照） | **1** | **0** | **0** |

> **没有对照组，61 和 48 只是两个数字，说明不了任何事。**
> **有了对照组，它们变成「这个模型文件的实现漏了」。**

**对照组选什么**：一个**被上游充分优化过的**同类模型。Qwen3-30B 是好选择
（框架的主要用户群）。

### 2.3 两条判定信号

**信号 A：计数能被层数整除**

```
48 = 2 个 residual add × 24 层
36 = 2 个 gating mul  × 18 个 conv 层
```

**能整除 = 每一层都在犯同一个错 = 实现漏了，不是偶然。**

**信号 B：与对照组的计数差**

**差值才是信号，绝对值不是。**

### 2.4 ★ 纯静态的补充扫描（不用 GPU，秒级）

```bash
cd /home/t-jialianggu/work/SLO-agent
PYTHONPATH=$PWD/src python -m sglang_agent_kernel_lab.cli scan --framework-src <sglang路径>
```

它检查 4 种形态：`never_wired` / `rank_guarded` / `path_guarded` / `residual_not_deferred`。

**这一条 signature 就找到了 LFM2.5 最大的两个赢家**：

> **枚举代码库里已有的融合原语，检查哪些模型的调用点没用它们。**

⚠️ **精度不高**：`never_wired` 形态 32-40 个候选里只有 3 个是真的。
**它给候选，不给判定。** 必须人工读代码确认。

参考 `.github/skills/fusion-gap-hunting/SKILL.md`（已有的成熟 skill）。

### 2.5 从「有 gap」到「gap 在哪一行」

数出计数之后要读源码。**判定标准：这是「可优化」还是「bug」？**

LFM2.5 的 `norm` 是后者，**三个事实必须同时看见**：

1. 函数签名**收了 `residual` 参数，第一行就覆盖** —— 传进来的值从没被用过
2. `RMSNorm.forward_cuda(x, residual)` **本来就会走 `fused_add_rmsnorm`**
3. 模型主循环**本来就在层间传递 residual**

→ **接线全都在，只是这一层没接上。**

> **参数被声明、被传递、然后被丢弃 —— 这是 bug，不是优化机会。**
> 这类的收益最大且最容易验证。

---

## 阶段 3：L3 实现与验证

### 3.1 ★★ 修复原则：不发明，去抄

**先找「正常的模型是怎么写的」。** LFM2.5 的 `norm` 修复照抄
`models/llama.py:304-316` 的 deferred-residual 写法。

**手写 kernel 是最后手段**，而且 LFM2.5 上两个真正需要写 kernel 的，
都是**相邻行方向工作的机械融合**，Inductor 自己就能推导出其中一个。

> **全程没有发明任何新东西。**

### 3.2 ★★ 注入方式：让 baseline 成为真 baseline

**这一步决定后面所有数字可不可信。**

用**环境变量 opt-in**（`LFM_FUSION_PATCH` 那套），不设变量时走
**逐字未改动的原路径** —— 同一棵树、同一 commit、同一份 server 参数。

**❌ 不要建第二棵 worktree** —— Gemma-3 那次因此撞上 stride 问题。

**踩过的坑**：模型类被 registry **懒加载**，`sitecustomize` 执行时目标模块还没导入，
**用定时器打 patch 是竞态**。用 `sys.meta_path` finder 在模块 exec 完成的瞬间打。

**防线**：**server log 必须检查 patch 生效标记** —— 否则一个静默失效的 patch
会被记成「与 baseline 相同」，得出「这个优化没用」的错误结论。

### 3.3 每个组件独立 e2e 验证，不攒批

LFM2.5 上写完 ShortConv kernel **24 分钟内**就做了 e2e 验证。

**理由**：隔离加速和端到端收益是两回事。早验证才能早发现
「microbenchmark 赢了但 e2e 没有」。

### 3.4 ★ 手写 kernel 必须有形状门控

两个手写 kernel 的形状依赖**正好相反**：

| kernel | 赢在哪 | 门控 |
|---|---|---|
| `conv` | **大 T**（要摊掉 ~30µs 的 Triton launch 地板） | `T ≥ 2048` |
| `moesum` | **小 T**（省的就是 launch + 一次 HBM 往返） | `T ≤ 32 或 T ≥ 4096` |

**门控阈值必须实测扫描得出，不能猜。** `lf_tune_shortconv.py` 每形状扫 32 组配置，
**先验正确性再计时**。

⚠️ **门控是有代价的**：在 regime E 上，L1 的赢家把 chunk 设成 2048，
**恰好卡在 `conv` 的门控边界**，导致 L3 从 +7.24% 塌到 **+1.84%**。
**新模型上必须检查 L1 赢家的 chunk 设置会不会打死门控。**

---

## 阶段 4：L2（若适用）

**前提：阶段 0.3 确认存在 config 驱动的热 kernel。**

### 关键做法

- 每个 token-count 桶扫 **468–894 个候选**，桶集合与上游对齐
- **每个候选先过正确性门禁再计时**（~9000 配置，0 失败）
- ★ **guarded 策略**：只在 oracle 证明有 headroom 的桶特化，
  其余**逐字段等于默认**

### 三个必须避免的错误（都栽过）

1. **在 server 从不执行的 kernel 变体上 tune**（如 expert bias 的有无）
2. **CUDA graph 捕获会把 config 烘焙在捕获时** → decode 事后改 config 无效
3. ★ **`M` 是 token 数，不是 `tokens × top_k`** —— profile 的键错了一个 `top_k`
   因子，真实 headroom 被藏在错位的桶后面。**只有活体 trace 能暴露这一点。**

---

## 阶段 5：★ 2³ 全因子矩阵（论文的主表）

### 5.1 工作单元

`scripts/lfm_fusion/exp3_layered.sh` 在**固定 serving 配置**下跑
`{L2 关/开} × {L3 关/开} × {正序/逆序}`，填**一行的 4 格**。

**每个 regime 跑两次填满 8 格**：

```bash
# cookbook serving
GPU=<G> REPS=8 PORT=<P> REGIME=<X>       bash scripts/lfm_fusion/exp3_layered.sh
# L1 ceiling serving（旋钮来自阶段 1）
WARMUP=<W> REPS=<R> SUITE=l1_ GPU=<G> PORT=<P2> REGIME=<X>_tuned \
    bash scripts/lfm_fusion/exp3_layered.sh
```

### 5.2 ★★ 顺序对照不是可选项

harness **顺序执行 arm**（一个 server lifetime 一个臂），存在**位置效应**。

**我们被这个坑过并差点报出反向结论**：

| 顺序 | 结果 |
|---|---:|
| default 先跑 | **−0.37%**（p=4.9e-04，"显著回归"） |
| candidate 先跑 | **+0.12%** |

**符号翻转，两次都是「先跑的更快」。** 合并后：−0.13%，p=0.079，**不显著**。

→ **必须 `{正序, 逆序}` 都跑，合并后 n=16。**

### 5.3 ⚠️ 测量窗口是噪声的真因

**实测各 workload 的窗口**：

| workload | 窗口 | 评价 |
|---|---:|---|
| `tool_agent` | **38 s** | ✅ 极稳（drift 0.7%） |
| `R_short_decode` | 4.7 s | ✅ |
| `shared_prefix` | 8.7 s | ✅ |
| `R_concurrent_decode` | 1.5 s | ✅ 尚可 |
| `R_long_prefill` | **0.31 s** | ⚠️ drift 36.5% |
| `R_long_prefill` @ L1 ceiling | **0.20 s** | ⚠️⚠️ |

在 0.2 秒的窗口里，**「9% 的差异」绝对值只有 16 毫秒** —— 一次 GC、
一次调度 tick 就是这个量级。

**在小模型上这个问题会更严重**（OLMo-2-1B 比 LFM2.5-8B 快得多）。

**处理**（按有效性）：
1. ★ **加长窗口**：提高 `--num-prompts`。⚠️ **但这改变 workload 定义**，
   要用就**整行重测**，不能与已有格子混比
2. 加 server lifetime，把 lifetime 当随机效应
3. **至少**：两个顺序**分别列出**，不要只报几何平均

### 5.4 ★ 真实 trace 上吞吐是错的口径

LFM2.5 的 `tool_agent`（唯一真实 trace）：

| 指标 | 全部三层 |
|---|---:|
| request throughput | **+0.48%**（噪声级） |
| **TTFT p95** | **537ms → 218ms，−59%** |

**真实 agentic trace 的吞吐由请求间依赖（think time）决定，不是服务器能力。**
**吞吐本来就是天花板。**

> **在 agentic workload 上必须报延迟。只报吞吐会得出「优化无效」的错误结论。**

⚠️ 同时注意 TTFT/TPOT 权衡：LFM2.5 上 L2 让 TTFT p95 −29.2%
但 **TPOT p95 +29.7%**。

---

## 3. 正确性：闸门可能对你的模型不适用

### token-identity 什么时候失效

LFM2.5 走 **top-4/32 路由，专家选择是离散 argmax**。任何**代数等价但非 bit-exact**
的改动（约 2 bf16 ulp）都可能**翻转选中的专家**，输出不连续地变。

实测：12 个 prompt top-1 有 11/12 一致，但 **KL 最高 0.99**。

> **不要降低闸门标准，要判定这个闸门是否结构性不可用，然后换一个。**

**dense 模型（如 OLMo-2）没有这个问题** —— token-identity 应该可用，**先试它**。

### 退到任务指标时，用 bit-exact 对照臂标定噪声

LFM2.5 的 `scale` 臂**数学上必然等于 baseline**，却读数低 **0.8 点**。

→ **between-arm 系统噪声 ≥ 0.8 点，由构造得到，不靠假设。**

**设计一个可证 bit-exact 的臂**（如「跳过乘以 1.0」），它就是免费的噪声标尺。

**口径**：**「未检测到质量回归」，不是「质量没变」**。

⚠️ 用户已指示 **GSM8K 可以只跑 20 题**（太慢）。但要知道：
n=20 时二项误差从 ±2.6 点涨到 **±21 点** —— 那已经不是质量闸门，
**只是「没崩」检查**。报数时必须说明。

---

## 4. 统计

- **Welch t + 精确 Student-t 尾**（`scipy.stats.t.sf`）。
  正态近似在 df≈5-10 下 **anti-conservative**，正好影响判决可能翻转的边缘臂
- 每格 **n=16**（2 顺序 × 8 重复）起
- **报逐格 p 值，不要笼统声明**。我们曾写「全部 p<0.005」，
  实际有一项是 p=0.018

---

## 5. ★ 必须记录的负面结果

**一个只报告成功的流程不可信。** LFM2.5 上刻意保留的：

| 负面结果 | 为什么保留 |
|---|---|
| `gate+idx` 三 regime 全不显著 | 机制在 kernel 级真实（1-2%），**没兑现到 e2e** |
| 否决 `topkGatingSigmoid`（占 decode 7.52%，**看起来最大**） | graph 节点间隙只有 0.064-0.128µs，**没有证据支持的收益** |
| 否决 MoE down-GEMM 原子归约 | 能省 11.5GB 流量，但要改 Triton GEMM 主循环，风险过高 |
| 两次自我纠错（统计方法、数据丢失 bug） | **主动降低了自己结论的强度** |
| **C 上 L2 在 L1 配置下是 −5.19% 回归** | kernel config **依赖 tune 时的 serving 配置** |
| **E 上 L3 因门控失效塌到 +1.84%** | 形状门控可能被 serving 配置打死 |

---

## 6. OLMo-2 的具体起点

### 已有资产

- ✅ 两个 harness **都已注册** `olmo2`（`ctx=4096`）
- ✅ 已有一个确认的 gap：**`_apply_qk_norm` 在非 capture 模式下走
  `forward_native`（慢路）**，见 `docs/kernel_fusion_catalogue.md`
- ✅ 已有 e2e 数据（`results/lfm_fusion/processed/fusion_ab_olmo2.csv`）：

  | regime | 增量 | p |
  |---|---:|---|
  | A 低批 decode | +0.45% | 2.7e-04 |
  | B 并发 decode | +0.70% | 0.143 **n.s.** |
  | **C 长 prefill** | **+14.51%** | 3.3e-05 |

- ✅ 上游 issue #33415 + draft PR #33416 已开

### ⚠️ 必须先解决的三件事

1. **L2 不适用**（dense，无 MoE）→ 阶段 0.3 找替代，或诚实记录 2² 矩阵
2. **1B 模型太快** → 窗口会很短，先量 `dur_s`，可能要加 `--num-prompts`
3. **只有一个 gap 已确认** → 阶段 2 的完整审计**还没做过**（现有结果来自单点发现）

### 建议顺序

```
阶段 0（含 L2 可行性判定）  →  阶段 2 完整审计（对照组用 Qwen3-30B）
  →  阶段 1 L1 全网格（可与阶段 2 并行，不同 GPU）
  →  阶段 3 实现新发现的 gap
  →  阶段 5 矩阵
```

**阶段 2 优先于阶段 1** —— 如果审计发现 OLMo-2 只有那一个 gap，
L3 那一层会很薄，**要早知道**。

---

## 6b. ★ 候选模型清单（本机已有，两个 harness 都已注册）

**不要从零挑模型。** 已经做过一次 **11 个模型的跨架构审计**
（`results/lfm_fusion/processed/cross_architecture_audit_summary.csv`，
方法见 `docs/2026-07-28/cross_architecture_audit.md`），L3 的 headroom 排序已知。

### 审计结果（低批 decode，占 kernel 时间比例）

| 模型 | 架构 | 层 | `removable_pct` | `all_gaps_pct` | L2 可做？ | 优先级 |
|---|---|---:|---:|---:|:--:|:--:|
| **gemma3** (1B) | dense + 滑窗注意力 | 26 | **37.06%** | **46.32%** | ❌ dense | ⚠️ 见下 |
| **olmo2** (1B) | dense (AllenAI) | 16 | **14.71%** | **27.74%** | ❌ dense | **★★★** |
| **exaone4** (1.2B) | dense (LG) | 30 | 3.54% | 15.66% | ❌ dense | ★★ |
| **phi4mini** (3.8B) | phi3 dense (MS) | 32 | 6.43% | 13.87% | ❌ dense | ★★ |
| *lfm25 (8B)* | *MoE + gated short conv* | *24* | *4.06%* | *11.31%* | *✅* | *已完成* |
| **olmoe** (1B-7B) | **MoE 64E (AllenAI)** | 16 | 0.43% | 4.70% | **✅** | **★★★** |
| qwen3next | MoE(512E) + GDN | 48 | 0.24% | 0.64% | ✅ | ✗ 太干净 |
| qwen06 (0.6B) | dense llama 式 | 28 | 0.46% | 0.57% | ❌ | ✗ 太干净 |
| granite (2B) | dense (IBM) | 40 | 0.23% | 0.30% | ❌ | ✗ 太干净 |
| qwen (30B) | MoE + 全注意力 | 48 | 0.18% | 0.23% | ✅ | **对照组** |
| qwen32 (32B) | dense llama 式（大） | 64 | 0.04% | 0.05% | ❌ | ✗ 太干净 |

- `removable_pct` = 一个融合实现**根本不会启动**的 kernel 占 kernel 时间的比例
- `all_gaps_pct` = 全部可疑桶（含不确定的）
- **`qwen` 是标准对照组** —— 0.18%，是「已被充分优化」的定义

### ★ 推荐组合：olmo2 + olmoe（同家族，一 dense 一 MoE）

**这是本次最有价值的一对，理由是它构成一个受控对比：**

| | olmo2 | olmoe |
|---|---|---|
| 家族 | AllenAI | **同一个** |
| 架构 | dense | **MoE 64E / top-8** |
| 层数 | 16 | **16（相同）** |
| hidden | 2048 | **2048（相同）** |
| ctx | 4096 | 4096 |
| L3 headroom | **27.74%** | 4.70% |
| L2 可做 | ❌ | **✅** |

**同家族、同层数、同 hidden size，只差 dense/MoE。**
→ 可以直接回答「L2 那一层的适用性和收益是不是纯粹由 MoE 的有无决定」。

**olmoe 的 L2 前景（已核实）**：
```
MoE shape = E=64, N=1024
上游 configs 里搜 "E=64,N=1024*":
  → 只有 triton_3_1_0/E=64,N=1024,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,...json
  → 没有 H200，没有 bf16
```
**和 LFM2.5 完全相同的缺口形态**（上游给别的卡调了，唯独没给 H200）。
LFM2.5 上这个缺口值 **+23.3%**。

⚠️ 但注意 olmoe 的 **L3 headroom 只有 4.70%**（LFM2.5 是 11.31%），
**L3 那一层可能很薄**。这恰好是有用的对比：**一个 L2 强 / L3 弱的模型。**

### ⚠️ gemma3 的特殊情况：headroom 最大，但已经被我们修过

gemma3 的 `removable_pct` 是全场最高的 **37.06%**，
但**那个 gap 我们已经在 2026-07-28 修了并测过**：
`Gemma3RMSNorm.forward_cuda` 跑 eager PyTorch，一行 fall-through 修复，
端到端 **2.07× / 1.75× / 1.57×**（`docs/2026-07-28/PR_DRAFT_gemma3_rmsnorm_v2.md`）。

**所以 gemma3 不是「未开发」，是「已经摘过一次果子」。** 用它要么：
- 在**已打补丁**的基线上重新审计，找剩下的（`docs/2026-07-28/` 有记录，剩下的价值很低）
- 或者把它当作**已完成的第三个案例**引用，不再投入 GPU 时间

> **建议：gemma3 作为「已有案例」写进论文，不作为新 campaign 的目标。**

### 其他模型的注意点

**exaone4（1.2B, 30 层, LG）** —— `all_gaps_pct` 15.66%，是 dense 里第二好的。
`ctx=65536`，注意 sglang 的 context-length 设置。

**phi4mini（3.8B, 32 层, Microsoft）** —— 13.87%，且**是这批里唯一超过 3B 的候选**
（其余都是 1-2B）。**如果担心「结论只在小模型成立」，这是最便宜的反驳。**
`ctx=131072`，必须显式限制。

**falconh1（1.5B, TII）** —— hybrid mamba 架构，`lf_lib` 已注册但**不在审计表里**，
headroom 未知。**属于未探索区域**，风险和潜在回报都高。

**granite / qwen06 / qwen32 / qwen3next —— 不要选。** 全部低于 0.65%，
L3 那一层会是空的，做出来的矩阵没有信息量。

### 三模型组合建议

| 组合 | 覆盖的轴 | 成本 |
|---|---|---|
| **olmo2 + olmoe** ★推荐 | 同家族 dense vs MoE，直接检验 L2 的适用性 | 2 × 20h |
| olmo2 + phi4mini | 小模型 vs 中模型，反驳「只在小模型成立」 | 2 × 22h |
| olmo2 + olmoe + phi4mini | 上面两条都覆盖 | 3 × 20h |

**加上已完成的 LFM2.5 和 gemma3，论文可以有 4-5 个模型。**

### ⚠️ 全部候选的共同风险：模型太小 → 测量窗口太短

LFM2.5 是 **8B**，本机其余候选是 **1-3.8B**。

`R_long_prefill` 在 LFM2.5 上的窗口已经只有 0.31 秒（drift 36.5%）。
**1B 模型会快 3-8 倍，窗口可能降到 50 毫秒级** —— 那时任何噪声都会淹没 6% 的效应。

**阶段 0 必须先量 `dur_s`**，若 `R_long_prefill` 的窗口 < 0.5 秒：
- 提高 `--num-prompts`（⚠️ 改变 workload 定义，**整行重测，不能与已有格子混比**）
- 或**在该模型上放弃 `R_long_prefill`，用其余 5 个 regime**（诚实记录原因）

---

## 7. 论文角度：什么样的结果才算复现

### 强复现（最理想）

新模型上 **L3 增量横跨四种基线基本恒定**，且量级相当（+3% 以上）。
→ 支持「kernel rewrite 与前两层正交」是**普遍规律**。

### 弱复现（仍有价值）

L3 增量存在且显著，但**随基线衰减**。
→ 说明正交性**依赖架构**，需要给出条件。**这也是可发表的。**

### 反例（最有价值，如果发生）

新模型上 **serving/config tuning 之后 L3 增量消失**。
→ 直接限定 LFM2.5 结论的适用范围。**必须照实报。**

### 跨模型可比的前提

- **同一套 workload 定义**（不要改 `WORKLOADS`）
- **同一个 cookbook 基线定义**（`cap32/chunk−1/lpm/mem0.85`）
- **同一个统计口径**（Welch + 精确 t，counterbalanced，n≥16）
- **同一张卡**（1×H200）

---

## 8. 这个流程为什么有效（可迁移的部分）

```
① 找一个挡路的既有结论，检查它的适用边界      ← 最容易被跳过，价值最高
② 换计数口径：数「不该存在的 kernel」，不是时间占比
③ ★ 选对照组（没有对照组，数字说明不了任何事）
④ 定位到代码行，判断是「可优化」还是「bug」
⑤ 抄正常实现，不发明
⑥ ★ 环境变量注入，baseline 是逐字未改的原路径
⑦ 每个组件独立 e2e 验证，不攒批
⑧ ★ 子 agent 从不同视角查同一对象，允许它们纠正主线
⑨ ★ 记录否决了什么，以及为什么
⑩ ★ 测组合，不要报各项之和
⑪ 闸门失效时换闸门，不是降标准
⑫ ★ 主动纠正自己的统计方法和 bug
```

**打星的六条是真正产生价值的部分。**

### 关于 ①（在新模型上尤其重要）

LFM2.5 那次的起点是质疑「sglang 热路径已全部融合，没有空缺」这个结论 ——
**它是在一个模型上得出的**。

**在 OLMo-2 上，等价的挡路结论可能是「小模型没什么可优化的」或
「dense 模型比 MoE 简单，上游肯定优化过了」。先明确写下你在质疑什么。**

### 关于 ⑩（次可加性）

| regime | 各项之和 | 一起测 | 兑现率 |
|---|---:|---:|---:|
| C 长 prefill | 5.86% | 5.30% | 0.90 |
| A 低批 decode | 9.37% | 6.57% | 0.70 |
| B 并发 decode | 12.80% | 6.21% | **0.49** |

**兑现率精确跟踪 regime 的饱和程度。**

> **规则：消除同一「种类」成本的优化不会相加。任何会部署的组合必须按组合测量。**

---

## 9. 关键文件索引

| 用途 | 路径 |
|---|---|
| **本文的证据来源** | `docs/2026-08-04/agent_workflow_evidence_chain.md` |
| **LFM2.5 主表（要复现的形式）** | `docs/2026-08-03/LFM25_ablation_matrix_EN.md` |
| LFM2.5 全记录（中文） | `docs/2026-08-03/LFM25_FINAL_CASE_full_record.md` |
| 已有的 L3 skill | `.github/skills/fusion-gap-hunting/SKILL.md` |
| 其余 17 个 skill | `.github/skills/` |
| **审计脚本**（阶段 2） | `scripts/lfm_fusion/lf_audit.py` |
| **L1 campaign**（阶段 1） | `scripts/run_serving_ceiling_campaign.py` |
| **矩阵驱动**（阶段 5） | `scripts/lfm_fusion/exp3_layered.sh` |
| 矩阵分析器 | `scripts/lfm_fusion/exp3_analyze.py` |
| L3 A/B harness | `scripts/lfm_fusion/lf_e2e.py` |
| L2 harness | `scripts/regime_kernel/rk_e2e.py` |
| workload 定义（**别改**） | `scripts/serving_ceiling_lib.py` |
| 静态扫描器 | `/home/t-jialianggu/work/SLO-agent`（`fusion_scan.py`） |
| 跨模型 gap 目录 | `docs/kernel_fusion_catalogue.md` |
| OLMo-2 已有结果 | `results/lfm_fusion/processed/fusion_ab_olmo2.csv` |

---

## 10. 诚实边界

- **这不是全自动流程。** researcher-in-the-loop：agent 提出、实现、测量，
  人决定继续还是叫停。
- **步骤 ①（质疑既有结论）无法机械化** —— 依赖「知道项目里有哪些结论、
  在什么条件下得出」。
- **步骤 ④（判断是 bug 还是优化点）需要同时读懂三处不相邻的代码。**
  静态扫描器只能给候选。
- **静态扫描精度不高**：`never_wired` 形态 32-40 候选里只有 3 个真。
- **L2 的适用性依赖架构**，L1/L3 不依赖。
- **跑 GPU 前必须确认卡是空的** —— 这台机器多人共用，
  跑在别人的卡上会双向污染。
