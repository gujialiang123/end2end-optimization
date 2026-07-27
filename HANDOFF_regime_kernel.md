# Regime-aware Kernel Specialization — 交接文档

**日期:** 2026-07-27 · **仓库:** `/home/t-jialianggu/work/EndtoEnd-auto-optimization`
**远端:** `github.com/gujialiang123/end2end-optimization`,分支 `main`
**最新 commit:** 见 `git log`(2026-07-27 追加 K1 跨模型 + LFM2.5 fusion 两条线) · **未提交文件:0**(除长期存在的 `result.jsonl`)

这份文档给接手的新会话用。读完这一份就能继续工作,不需要回看之前的对话。

---

## 0. 一句话现状

P0 全部完成并推送。**核心结论已确立但经历过两次重要修正**,当前结论是诚实且经过对照实验支持的。剩余工作是可选的加强项(见 §8)。

---

## 1. 这轮要回答的问题

> 不同 serving regime 会产生不同的 kernel workload,因此同一模型同一 GPU 上,不同 regime 的最优 kernel 实现/配置可能不同。Agent 能否根据 profiling 自动选择、调优、验证、部署 regime-specific kernel variant?

四个 RQ:
- **RQ1** regime 是否产生不同 kernel workload
- **RQ2** 一个 regime 调的配置迁到别的 regime 会不会退化
- **RQ3** 少量 regime profile 是否优于单一 global profile,能否接近 per-shape oracle
- **RQ4** 能否把 profiling→候选→correctness→benchmark→accept/reject 做成闭环

---

## 2. 实验环境(所有结果都在这个 frame 下)

| 项 | 值 |
|---|---|
| GPU | 1× NVIDIA H200(TP1) |
| dtype | BF16 |
| sglang | 0.5.12.post1 @ `17f7a1da1`(源码在 `/home/t-jialianggu/work/sglang`) |
| torch / triton / CUDA / driver | 2.9.1+cu128 / 3.5.1 / 12.8 / 580.105.08 |
| 模型 A | LFM2.5-8B-A1B `/data/hf/LFM2.5-8B-A1B`(E=32, N=1792, top_k=4, **use_expert_bias=true**, 无 shared expert) |
| 模型 B | Qwen3-30B-A3B `/data/hf/models/Qwen3-30B-A3B-Instruct-2507`(E=128, N=768, top_k=8) |
| 热点 kernel | fused-MoE Triton |
| python | `/home/t-jialianggu/.conda/envs/sglang-dev/bin/python` |

跑任何脚本前必须设:
```bash
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$ENVDIR PATH=$ENVDIR/bin:$PATH HF_HOME=$PWD/.hf_cache \
       TRITON_CACHE_DIR=/tmp/regime_kernel_triton_cache
```

三个 regime(复用 serving campaign 的冻结定义):
- **A 低批 decode** = `R_short_decode`(input 100, output 256, 8 请求, 并发 1)
- **B 并发 decode** = `R_concurrent_decode`(input 200, output 256, 32 请求, 并发 32)
- **C 长 prefill** = `R_long_prefill`(input 4000, output 32, 4 请求, 并发 4)

---

## 3. 切入点(为什么这个实验成立且便宜)

SGLang 的 MoE kernel config 是一张 **`M → 配置` 的查找表**,运行时用最近邻取:
```python
config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
```
而**我们两个模型在这台 H200 上都没有调优过的配置**(启动日志原话):
- LFM2.5 `E=32,N=1792`:**配置文件不存在**,走两分支启发式默认
- Qwen `E=128,N=768`:回退到 **triton 3.2.0** 的配置(我们跑 3.5.1)

**不用写新 CUDA kernel**。切换方式(默认路径完全不变):
- microbench:`override_config()` context manager
- 端到端:`SGLANG_MOE_CONFIG_DIR` 环境变量指向自建 profile 目录

---

## 4. ⚠️ 三个必须知道的坑(都是踩过并修好的)

### 4.1 `M` 是 token 数,不是 `tokens × top_k`
`fused_experts_impl` 里:`M = min(num_tokens, CHUNK_SIZE)`。
我最初假设 `M = tokens × top_k`,导致 profile 的 key **大了 4 倍**,配置被错位到相邻桶,**反而掩盖了真实空间**。
只有 trace 活服务器才发现。**新会话如果要改 profile 生成逻辑,务必确认 key 是 token 数。**

### 4.2 必须跑服务器真实执行的 kernel 变体
LFM2.5 有 `use_expert_bias=true`,服务器跑 **with-bias** 变体。我最初调的是 no-bias 变体:
- no-bias 在 M=4 测出 1.067× → **部署后端到端损失 25%**
- with-bias 真实空间只有 1.007×

**microbench 必须传 `--bias`**(`rk_microbench.py` 已支持)。

### 4.3 CUDA graph 重放 decode
config 在 capture 时烘焙进图,**稳态 decode 一次配置查找都没有**。
capture 用 bs `[1,2,4,8,12,16,24,32]`。实测 regime A 的 MoE 查找其实是 **prompt prefill**(M=101–125)。
tracer 记录到 0 行 ≠ tracer 坏了。

**附带**:tracer 本身有个坑 —— `fused_moe.py` 用 `from ... import try_get_optimal_moe_config` 把原函数绑进自己命名空间,**只 patch config 模块无效,必须同时 patch 调用方模块**(已修)。

---

## 5. 已确立的结论(全部经 correctness 门禁,~9000 配置零正确性失败)

### 5.1 Kernel 配置调优 = **shape-dependent,不是 regime-dependent** ✅(已修正的说法)

| 真实 M | oracle 加速(with-bias) |
|---:|---:|
| ≤32 | 1.00–1.09×(**无空间**) |
| 64 / 128 / 256 | 1.484× / 1.440× / 1.449× |
| 1024 / 2048 / 8192 | 1.491× / 1.632× / 1.639× |

**crossover 在 M ≈ 64。**

**routing 交叉验证(决定性负面结果)**:固定 M,把 uniform 调的配置和 skewed 调的配置交叉应用 —— **4 个 M 里 3 个,"针对 skewed 调的配置"在 skewed 上反而输给 uniform 调的配置**。routing-specific 调优是在拟合噪声。
→ 因果链是 `regime → M 分布 → 最优配置`,而**运行时本来就按 M 分发**。

### 5.2 迁移会灾难性退化 ✅(RQ2)
Qwen 的 low-M profile 用到 prefill:**0.123×(慢 8 倍)**。

### 5.3 单一 global profile 有害 ✅(RQ3)
大 M 上 LFM 0.618× / Qwen 0.385×,**比不调还差**。3 个 profile 恢复 73–100% oracle。

### 5.4 端到端(LFM2.5,serving 参数冻结,只换 kernel profile,6 次重复)

| regime | global-best | regime-aware(朴素) | **guarded(M 修正)** |
|---|---:|---:|---:|
| A 低批 decode | 0.923× | 0.745× | **0.998×** 中性 |
| B 并发 decode | 1.004× | 1.060× | **1.014×** |
| C 长 prefill | **0.796×** | 1.170× | **1.223×**(6/6 不重叠) |

**guarded 策略**:只在 oracle >1.15× 的 M 桶特化(8/14),其余写运行时默认值 → 同时拿到 prefill +22.3% 和其他 regime 零回归。

### 5.5 **K1:换 kernel 实现才是真正 regime-dependent 的** ✅(最强结果)

| backend | A 低批 decode | B 并发 decode | C 长 prefill |
|---|---:|---:|---:|
| auto / triton | 1.000× / 0.999× | 1.000× / 1.006× | 1.000× / 1.004× |
| **triton_kernel** | **0.650×** | 0.966× | 0.996× |
| **flashinfer_cutlass** | 0.965× | **1.017×** | **0.664×** |

**排序随 regime 翻转**(cutlass 并发 decode 最好、长 prefill 最差,35 个百分点摆动)。

**为什么这比 config 更强**:config 运行时**已经**按 M 自动分发;**backend 是启动时定死、全程不变** → "按 regime 选 backend"是**真正缺失的能力**。

**但不对称**:选对最多 +1.7%,选错损失 34–35% → **避坑杠杆,不是提速杠杆**。

### 5.6 RQ4 Agent 闭环 ✅
不同诊断走**不同动作序列**:
- M=4 判 `low_occupancy_launch_bound`(算术强度 0.083)→ 小 tile+深 pipeline(接受 1.081×)→ 宽 K(**正确拒绝** 0.997×)
- M=8192 判 `compute_bound`(强度 170.7)→ 大 tile(1.266×)→ 深 pipeline(累计 1.336×)

且只用 48 候选就超过 240 候选穷举。

### 5.7 waterfall(serving tuning vs kernel,两者**不是简单叠加**)

| regime | cookbook | +kernel | tuned serving | tuned+kernel |
|---|---:|---:|---:|---:|
| A | 1.000× | 0.998× | 0.995× | 0.998× |
| B | 1.000× | 1.014× | 1.001× | 1.013× |
| C | 1.000× | 1.223× | **1.779×** | 1.696× |

C 上 serving 1.78× + kernel 1.22× → 叠加只有 1.70×。**假设**(未验证):tuned serving 的 `chunk=2048` 改变了 M 分布,提前吃掉了 kernel 的空间。CI 较大(±1.0),需要更多重复。

### 5.8 部署注意:Triton JIT 重编译
引入新 config 会造成间歇性停顿。冷 cache 下 8 次里 2 次掉到 14.6/16.5(vs 22.2);**warm cache 下 8/8 干净**。
→ 部署必须预热 Triton cache;不预热的 benchmark 均值会被严重低估。

---

## 6. ⚠️ Baseline 的局限(必须在任何汇报里说明)

**LFM2.5 的 baseline 是 SGLang 的两分支启发式默认值**,不是认真调过的配置:
```python
config = {BLOCK_SIZE_M:64, BLOCK_SIZE_N:64, BLOCK_SIZE_K:32, GROUP_SIZE_M:8}
if M <= E:
    config = {BLOCK_SIZE_M:16, BLOCK_SIZE_N:32, BLOCK_SIZE_K:64, ...}
```
整个 M 范围只有两档,没有 `num_warps`/`num_stages`。

**诚实表述**:不是"我们把 kernel 优化快了 1.6×",而是**"这个模型/GPU 组合从来没人调过,补上调优值 1.6×"**。上游日志自己写着 "Performance might be sub-optimal!"。

**Qwen 是有用的对照**:它**有**真实调优配置(只是版本差一个 minor),空间只有 0.96–1.23×。**有人调过的空间小,没调过的空间大** —— 这个对比本身是结果。

---

## 7. 代码与数据地图

### 脚本 `scripts/regime_kernel/`
| 文件 | 作用 |
|---|---|
| `rk_lib.py` | 共享:模型/regime 定义、剪枝搜索空间、routing 生成、环境快照 |
| `rk_microbench.py` | **核心**:单 (model, tokens, routing) 下跑所有候选,correctness 门禁 + CUDA event 计时。`--bias` 必须加 |
| `rk_campaign.py` | sqlite 工作队列,多 GPU 并行,断点续跑。stage: sweep/routing/transfer/bias |
| `rk_profiles.py` | 构建 default/global-best/regime-aware/oracle 策略,生成 `SGLANG_MOE_CONFIG_DIR` 目录 |
| `rk_guarded_profile.py` | 只在 oracle 超阈值处特化,其余写运行时默认 |
| `rk_e2e.py` | 端到端:复用 canonical serving harness,只变 `SGLANG_MOE_CONFIG_DIR`。支持 `--suffix`/`--tag`/serving knob 覆盖 |
| `rk_backends.py` | **K1**:按 regime 比较不同 MoE runner backend |
| `rk_routing_cross.py` | routing 交叉验证(证明 routing 调优是拟合噪声) |
| `rk_agent.py` | 闭环 agent:诊断→选动作→候选→correctness→bench→accept/reject |
| `rk_process.py` / `rk_plots.py` | 原始→tidy CSV;从 CSV 出 10 张图 |
| `rk_trace/sitecustomize.py` | opt-in tracer,`RK_KERNEL_TRACE=/path.jsonl` + `PYTHONPATH` 启用 |

### 数据 `results/regime_kernel/`
- `raw/{sweep,bias,routing,transfer,routing_cross}/` — 原始 JSON(含每个候选的计时+correctness)
- `processed/*.csv` — 15 张 tidy 表,**绘图只读这里**
- `plots/*.{png,svg}` — 10 张图
- `e2e/`、`backends/`、`agent/`、`traces/` — 各阶段结果
- `configs/regime_kernel/profiles/*/` — 可直接 `SGLANG_MOE_CONFIG_DIR` 部署的 profile

### 文档 `docs/`
- `regime_kernel_status.md` — 仓库审计、可复用资产、缺失项、成本估算
- `regime_kernel_experiment_plan.md` — 方法论、搜索空间、测量协议、P0/P1
- `regime_kernel_results.md` — **主报告**,含 §0b 的诚实范围说明和修正记录

---

## 8. 剩余可做的工作(按价值排序)

> **2026-07-27 更新**:§8.1 已完成(见下方 §8.0),并且新开了一条 fusion 线
> (`docs/lfm_fusion_results.md`),拿到本项目第一个同模型正向 kernel e2e 结果。

### 8.0 ✅ 已完成:Qwen 的 backend 对比(原 §8.1)
3 regime × 4 backend × 5 rep,0 失败。**结论比预期更强:regime→backend 规则不可迁移。**

| backend | A decode LFM/Qwen | B 并发 LFM/Qwen | C 长 prefill LFM/Qwen |
|---|---:|---:|---:|
| **triton_kernel** | **0.650 / 0.641** | 0.966 / 1.008 | 0.996 / **0.647** |
| **flashinfer_cutlass** | 0.965 / 0.934 | **1.017 / 1.047** | **0.664** / **1.027** |

两个 regime 两模型一致(triton_kernel 低批 decode 都灾难;cutlass 并发 decode 都最好),
但**长 prefill 完全反转** —— 断崖换到了另一个 backend。把一个模型的规则用到另一个
模型,最差 **−34%**。→ 静态查找表有害,必须按部署实测。详见 results 文档 §11c。

### 8.0b ✅ 新增:LFM2.5 fusion(`docs/lfm_fusion_results.md`,**已做两轮**)
v33 的"sglang 热路径已全部融合"是**在 Qwen 一个模型上**得出的。LFM2.5 每 forward
有 **61 个未融合 RMSNorm + 48 个独立 residual add + 36 个 gating mul**(Qwen 对照:
1 / 0 / 0),外加一条未融合的 QK-norm+RoPE 链。

**最终 E2E(七个组件全开,6 重复,精确 Welch t,p=4.6e-14 / 2.4e-08 / 1.2e-05)**:
低批 decode **+6.57%** · 并发 decode **+6.21%** · 长 prefill **+5.30%**。

两个最大的赢家都是**"sglang 已有融合原语、这个模型的调用点没用"**
(`fused_add_rmsnorm`、`fused_qk_norm_rope`)。两个手写 Triton kernel:
ShortConv 的 gate+transpose(bit-exact,**Inductor 自己也会推导出结构等价的
kernel**,只在 T≥2048 有用)和 MoE 归约+下一层 norm 的融合(**相反形状:小 T 才
是赢面**,T=1/8/32 上 2.46/2.68/2.64×)。单项最大是 `qkrope`(并发 decode
**+5.42%**),纯调用点改动。

**★ 最重要的方法学产出:同类优化强烈次可加。** 各项之和 vs 一起测:
A 0.98 / B **0.57** / C 0.87。并发 decode 上 qkrope 单独 +5.42%,再加单独值
+3.65% 的组件只多买到 0.12 点 —— 两者在消除同一份"固定每-forward 开销"余量。
**与 regime-kernel 的 waterfall 非叠加(serving 1.78×+kernel 1.22×→1.70×)同源。**

### 8.2 把 backend 纳入 agent 候选动作
`rk_agent.py` 现在只会调 config。计划 §十 明确要求 agent 能 "switch backend"。
考虑到 §5.5 显示 backend 是避坑杠杆,agent 应该学会**排除**坏 backend 而不是追求最优。
**§8.0 的跨模型结果让这条的优先级上升**:既然规则不可迁移,就只能靠 agent 按部署实测。

### 8.2b 把 fusion 审计做成 agent 的机械检查(**最推荐**)
现在有**两条**可机械检查的 signature:
1. **随层数线性增长的 kernel 计数**(48 = 2 × 24 层),Qwen 对照组定义了"干净"
   (1 / 0 / 0)。脚本已就绪:`scripts/lfm_fusion/lf_audit.py`。
2. **枚举代码库里已有的融合原语,检查哪些模型的调用点没用它们** —— 这一条就找到了
   四个赢家里的三个,而且完全不需要 profiling,是纯静态检查。

### 8.2c 把审计跑到其他新架构上(**最便宜的下一步,~15 分钟/模型**)
"架构成熟度决定 fusion 空缺"目前是**单模型观察**。在第二、第三个新架构上复现
才能变成规律 —— 而这正是把它做成 agent 检查的前提。

### 8.3 验证 waterfall 的非叠加机制
用 tracer 在 tuned serving(`chunk=2048`)下测 M 分布,和 cookbook 对比,验证"serving tuning 改变了 M 分布"这个假设。约 20 分钟。

### 8.4 `_down` 伴随配置
TMA 路径在这台机器上**是激活的**(`support_tensor_descriptor()=True`),所以缺 `_down` 配置有真实代价。
**约束**:`down_config["BLOCK_SIZE_M"]` 必须**等于**主 config 的,否则 runtime assert 失败。

### 8.5 未做且判断为不值得
- **K4 fusion**:LFM2.5 无 shared expert;仓库现有 fusion patch(pr29007/pr31438)针对别的模型/算子。计划原话"没有现成实现时不要从零实现 fusion"。
- **写新 CUDA kernel**:计划明确排除在 P0 之外。

---

## 9. 操作纪律(重要)

1. **GPU 归属**:这台机器多人共用。跑之前必查
   ```bash
   nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
   ```
   并用 `ps -o user= -p <pid>` 确认归属。**t-delwinkim** 和 **t-ntakbir** 经常占卡,不要碰他们的进程。不确定就问用户。
2. **kill 只能用具体 PID**(`kill -TERM <pid>`),环境禁止 `pkill`/`killall`。
3. **git**:`result.jsonl` 长期有本地改动。push 前 `git stash push result.jsonl -q` → `git pull --rebase` → push → `git stash pop`。commit message 含多行/特殊字符时**写到文件再 `-F`**,否则引号嵌套会炸。
4. **commit trailer**:`Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
5. **测量纪律**(前面吃过亏):
   - 短 workload 必须预热到稳态(`WARMUP_RUNS` 按 workload 定制)
   - correctness 先于 performance,不过关的候选不进结果表
   - 单次网格搜索的 argmax 会系统性高估,最终结论必须多次重复
   - micro speedup 和 E2E speedup **严格分开报**

---

## 10. 给新会话的开场建议

```
读 docs/regime_kernel_results.md(主报告)、docs/lfm_fusion_results.md
(fusion 线主报告)和 HANDOFF_regime_kernel.md(本文件)。
P0 与 §8.1 已完成。建议接 §8.2b(把 fusion 审计做成 agent 的机械检查)
或 §8.4/§9.1(ShortConv 的 layout copy 与 gating 融合)。
```

### 新增的坑(fusion 线,2026-07-27)
- **token-identity 对 LFM2.5 是结构性不可用的正确性门禁**。它 top-4/32 专家路由是
  **离散 argmax**,任何 bf16 级扰动都可能翻转选中的专家 → 输出不连续变化。
  用任务指标(GSM8K)代替。
- **用一个 bit-exact 的 arm 免费标定 harness 噪声底**。`scale`(乘以 1.0)必然与
  baseline 数学相同,但 GSM8K 读数低 0.8 点 → 这就是 `--parallel 32` 的 batch 组成
  差异造成的系统噪声。**任何精度相关的结论都应该配一个 bit-exact 对照。**
- **patch 必须校验真的生效**。`lf_e2e.py` 会在 server log 里找 patch marker,
  否则一个静默失效的 patch 会被当成"与 baseline 相同"记录下来。
- 模型类是被 model registry **懒加载**的,sitecustomize 里用定时器打 patch 是竞态;
  用 `sys.meta_path` finder 在模块 exec 完成的瞬间打(`lf_inject/sitecustomize.py`)。

关键 CSV 一览:
- `processed/sweep_headroom_bias.csv` — 真实变体下每个 M 的可调空间(**看这个判断哪里值得调**)
- `processed/backend_comparison.csv` — K1 结果
- `processed/routing_cross_matrix.csv` — routing 交叉验证(负面结果)
- `processed/e2e_summary.csv` — 全部 E2E,含三轮迭代对照
- `processed/measured_M_distribution.csv` — 实测 regime → M 映射
