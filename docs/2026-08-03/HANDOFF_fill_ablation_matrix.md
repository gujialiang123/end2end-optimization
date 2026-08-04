# 交接：补全 LFM2.5 消融矩阵的空格

**写于**：2026-08-03 23:45 · **更新**：2026-08-04 00:45
**接手方**：另一个 agent，**GPU 3 和 4**
**目标文档**：`docs/2026-08-03/LFM25_ablation_matrix_EN.md`（英文，给 mentor 看的主表）

---

## ★★ 00:45 更新：进度比本文档原先记载的多，而且出了两个反直觉的结果

上一轮已经跑完了 **regime C 的 L1 全部 4 格** 和 **regime B 的全部 4 格**，
F 和 A_tuned 正在跑。**下面三条必须先读，它们改变了后续该怎么跑。**

### ① ★ L2 在 L1 的 serving 配置下是**回归**，不是增益

regime C，L1 ceiling 配置（`cap8/chunk2048/fcfs/mem0.9`），n=60/臂：

| | req/s | vs cookbook(12.119) |
|---|---:|---:|
| L1 | 21.530 | +77.7% |
| **L1 + L2** | **20.413** | **+68.4%** ← 比只有 L1 还低 |
| L1 + L3 | 22.879 | +88.8% |
| L1 + L2 + L3 | 21.717 | +79.2% |

**L2 叠在 L1 之上 = −5.19%。** 而同一个 L2 在 cookbook 配置下是 **+23.26%**。

已核实 config 确实加载了（`cfg` 格的 server log 有
`Using MoE kernel config from .../E=32,N=1792,device_name=NVIDIA_H200.json`，
`nocfg` 格没有）。**不是实验失误。**

**机制（推测，待验证）**：L1 的赢家开了 `chunked_prefill_size=2048`，
这**改变了每次 forward 的 token 数 M 的分布**。而 tuned MoE config 的桶是在
cookbook 配置（`chunk=-1`，4000 token 整块进）下扫出来的。
**换了 serving 配置 = 换了 M 分布 = 选中的桶不对了。**

> **这条结论本身就是交付材料**，而且和我们已有的
> 「regime→backend 规则跨模型不可迁移」是同一类：
> **kernel config tuning 依赖于它被 tune 时的 serving 配置，不能跨 serving 配置迁移。**
> 需要验证：跑一次 NCU 或看 M 分布直方图，确认桶确实错位了。

**直接后果**：regime C 上**最优组合是 `L1 + L3`（22.879，+88.8%），不是全三层。**
「三层叠加」这个叙事在这个 regime 上是错的，要改成「L1 和 L3 叠加，L2 要看 serving 配置」。

### ② regime B 全 4 格完成，替换掉旧的 n=6 数字

| | req/s | 增量 |
|---|---:|---:|
| S0 cookbook | 21.661 | — |
| + L2 | 22.026 | **+1.68%** |
| + L3 | 23.118 | **+6.72%** (t=25.8) |
| + L2 + L3 | 23.537 | **+8.66%** vs S0 |

n=16/臂，counterbalanced。**旧的 +6.21%（n=6）作废，用 +6.72%。**
L2 在 B 上有小幅正收益（+1.68%），和 L2+L3 的复合 +8.66% 一致（1.0168×1.0686=1.0866）。

### ③ ⚠️ regime F（真实 trace）上 kernel 改动**几乎不提升吞吐**

只有 fwd 一半，但信号很清楚（n=8/臂）：

| 指标 | baseline | +L3 | 变化 |
|---|---:|---:|---:|
| request throughput | 5.265 | 5.286 | **+0.38%** (t=4.6) |
| **TTFT p95** | 537.0 ms | 495.9 ms | **−7.66%** (t=−16.8) ✅ |
| TTFT mean | 323.2 ms | 298.6 ms | **−7.63%** (t=−14.7) ✅ |
| TPOT p95 | 23.44 ms | 27.52 ms | +17.4% (t=0.9 **不显著**) |
| TPOT mean | 7.19 ms | 7.04 ms | −2.05% (t=−0.6 n.s.) |

**吞吐几乎不动，但 TTFT 显著降低 7.6%。**

这符合预期：真实 agentic trace 的吞吐由**请求之间的依赖关系**决定（有 think time），
不是服务器能力，**吞吐本来就是天花板**。**在这个 regime 上必须报延迟，不能只报吞吐。**

> ⚠️ **主表现在只记 request throughput 一个指标，对 F 是错的口径。**
> 填 F 那一行时必须同时填 TTFT p95。

---

## 0. 三十秒读懂

有一张 6 regime × 8 列（L1/L2/L3 的 2³ 全因子）的表，**48 格里填了 19 格**。
你的任务是填剩下的。**不是 29 个实验——一次 harness 调用填 4 格。**

一切都已经跑通过，脚本、分析器、profile 都在。你主要做三件事：
1. 给 `lf_e2e.py` 加 8 个 regime 条目（纯代码，~20 行）
2. 反复调用 `exp3_layered.sh`
3. 跑 `exp3_analyze.py` 并把数字填进表

**⚠️ 开工前必读 §4（一个已知会毁掉数据的坑）和 §5（顺序依赖）。**

---

## 1. 环境

```bash
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python      # 不是 gemma-sglang
GPU=3,4                                                       # 用户指定
```

- sglang 0.5.12.post1 @ `17f7a1da1`，torch 2.9.1+cu128，Triton 3.5.1
- 模型 `/data/hf/LFM2.5-8B-A1B`，bf16，TP=1
- `exp3_layered.sh` 自己会 `export CUDA_HOME` 和 `HF_HOME`，不用手动设

**GPU 争用**：2026-08-03 23:40 时 8 张卡全被 t-vinkapoor 和 t-ntakbir 占着。
开跑前必须确认 GPU 3/4 真的空了：

```bash
for g in 3 4; do
  for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); do
    ps -o user=,cmd= -p $p | cut -c1-90
  done
done
```

**别在有别人进程的卡上跑**——会双向污染，两边数据都作废。

---

## 2. 现在的表长什么样

| Regime | workload | S0 | L1 | L2 | L3 | L1+L2 | L1+L3 | L2+L3 | L1+L2+L3 |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **A** low-batch decode | `R_short_decode` | ✅ | 🔄 | ✅ | ✅ | ⬜ | 🔄 | ✅ | ⬜ |
| **B** concurrent decode | `R_concurrent_decode` | ✅ | † | ✅ | ✅ | ⬜ | ⬜ | ✅ | ⬜ |
| **C** long prefill | `R_long_prefill` | ✅ | ✅ | ✅ | ✅ | **✅** | **✅** | ✅ | **✅** |
| **D** medium balanced | `R_medium_balanced` | † | † | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **E** shared prefix | `shared_prefix` | † | † | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **F** tool agent（**唯一真实 trace**） | `tool_agent` | ✅ | † | ⬜ | 🔄 | ⬜ | ⬜ | ⬜ | ⬜ |

- ✅ = 已测且内部一致
- 🔄 = 正在跑（00:45 时 GPU 3 在跑 A_tuned，GPU 4 在跑 F）
- † = 数字有，但来自**另一个 campaign**（07-24 serving ceiling），有自己的 cookbook 基线，**只有 ratio 可比**
- ⬜ = 空

**★ regime C 整行已满**（8/8），是唯一完整的一行。**结论见开头 ①：最优不是全三层。**

**已确认的核心结果**（regime C 长 prefill，serving = cookbook）：

```
S0  cookbook       12.119 ± 0.116 req/s   1.000×
    + L2           14.939 ± 0.123         1.233×   (+23.26%)
    + L2 + L3      16.392 ± 0.200         1.353×   (+35.25%)
                                                    ↑ L3 在 L2 之上 = +9.73%, p=9.5e-19
```

**L3 在三种基线之上的增量**（regime C）：

| 叠在 | 增量 | n | 状态 |
|---|---:|---:|---|
| cookbook | +6.18% | 16 | ✅ |
| L1 ceiling | +6.34%（见 §6 的警告） | 60 | ⚠️ 两个顺序严重不一致 |
| L2 | **+9.73%** | 16 | ✅ 最强的那个 |
| **L1 + L2** | ⬜ | — | **❌ 就缺这个** |

---

## 3. 工作单元：一次调用 = 4 格

`scripts/lfm_fusion/exp3_layered.sh` 在**固定 serving 配置**下跑
`{L2 关/开} × {L3 关/开} × {正序/逆序}`，即 4 次 `lf_e2e.py` 调用、8 个 server lifetime。

```bash
GPU=3 REPS=8 PORT=52141 REGIME=<regime> bash scripts/lfm_fusion/exp3_layered.sh
```

**每个 regime 跑两次就填满整行 8 格**：

| 调用 | REGIME 参数 | 填哪 4 格 |
|---|---|---|
| 第一次（cookbook serving） | `X_xxx` | S0, L2, L3, L2+L3 |
| 第二次（L1 ceiling serving） | `X_xxx_tuned` | L1, L1+L2, L1+L3, L1+L2+L3 |

而且**整行内部一致**（同一棵树、同一 campaign、同一协议）→ **† 标记消失，绝对值可比**。

---

## 4. ⚠️ 三个会毁数据的坑（都踩过）

### 4.1 泄漏的 server（已修但要知道）

`lf_e2e.py` 用 `setsid` 起 server。**Ctrl-C / kill 父进程杀不掉它**，它会继续占端口，
下一次运行的健康检查会打到这台旧 server 上，整个 A/B 变成测那台遗留进程。

已加 `assert_port_free(port)` 会直接报错退出。但**每次中断后都要手动确认**：

```bash
ps -u $(whoami) -o pid,cmd | grep launch_server | grep -v grep
# 有残留就 kill <pid>
```

**每个并行任务用不同的 PORT**（52141, 52142, ...）。

### 4.2 ★ 不要在脚本运行期间编辑 `exp3_layered.sh`

bash **按字节偏移增量读取脚本**。运行中改动会让它中途 `unexpected EOF` 挂掉。
今天 regime A 的后三格就是这么丢的，白跑一格。

**要改就先复制一份改，或者等它跑完。**

### 4.3 ★ WARMUP 必须按 serving 配置重新设

`serving_ceiling_lib.WARMUP_RUNS` 是**按 cookbook 旋钮标定的**。换了 serving 配置就是
另一个稳态。

实测：在 `cap8/chunk2048/fcfs/mem0.9` 上，用它默认的 4 次 warm-up，**前两个计分重复
仍在从 20 爬到 23 req/s**——读出来就是"kernel 收益消失了"。

exp5 因此把 `REPS` 提到 30、`WARMUP` 提到 12。

**规则：跑任何 `*_tuned` regime 时必须显式给 `WARMUP`。**

```bash
WARMUP=12 REPS=30 GPU=3 PORT=52141 REGIME=C_long_prefill_tuned SUITE=l1_ \
    bash scripts/lfm_fusion/exp3_layered.sh
```

cookbook 的三个 regime（A/B/C）用默认即可。D/E/F 的 cookbook 版也用默认
（`WARMUP_RUNS` 里已有它们的条目：medium=2, shared_prefix=1, tool_agent=0）。

---

## 5. ★ 顺序依赖：先决定 `_down` config

**这件事必须在跑任何新格子之前定。**

server log 里有：

```
Using MoE kernel config with down_moe=False. Performance might be sub-optimal!
Config file not found at .../E=32,N=1792,device_name=NVIDIA_H200_down.json
```

一个 MoE 层跑**两个** grouped GEMM——上投影 `w13` 和下投影 `w2`——sglang 用**两个独立
配置文件**分别调（`fused_moe_triton_config.py:33` 按 `down_moe` 标志拼文件名）。
我们只做了上投影的。缺 `_down` 时 sglang 拿上投影的配置凑合给下投影用。

**所以约一半的 MoE GEMM 工作跑在不是为它调的配置上 = L2 没到自己的 ceiling。**

补它有硬约束（`fused_moe_triton_config.py:265`）：

```python
assert config["BLOCK_SIZE_M"] == down_config["BLOCK_SIZE_M"]
```

下投影的 `BLOCK_SIZE_M` **必须等于**上投影的，只能在这个约束下扫。

**为什么必须先定**：加上 `_down` **改变了「L2 开」这条臂本身是什么**。
先跑的所有格子都要重测。

现成的候选：`configs/regime_kernel/profiles/lfm25_bias_guarded_tma/` 下已有一个 `_down.json`，
值得先测一下能不能直接用。

### 建议顺序（00:45 修订）

```
1. 让 GPU 3/4 上正在跑的 A_tuned 和 F 跑完                    ← 别打断
2. F 的 cfg 两格（L2 / L2+L3 在真实 trace 上）★★ 最高价值
3. F 和 A_tuned 的 L1 版本                                     ← 补全两行
4. D 和 E 各两次调用
5. 验证 ①（L2 在 L1 配置下回归）的机制 —— 看 M 分布或 NCU
6. 决定 _down config（原先排第 1，现在降级，理由见下）
```

> **`_down` 为什么降级**：发现 ① 说明 **L2 本身就依赖 serving 配置**，
> 在 L1 配置下甚至是负的。在搞清楚这件事之前扫 `_down` 是在优化一个
> **还不知道该在什么配置下用**的东西。**先弄明白 ①，再决定要不要投入 1-2h 扫 `_down`。**

## 6. 【已完成】regime C 的 L1 四格（保留供参考）

> **00:45 更新：这两格已经跑完了**，结果见开头 ①。本节保留是因为里面的噪声分析仍然适用。

exp5 只跑完 4 格里的 2 格就被 GPU 争用打断，**缺的正好是 `cfg` 那两格**。

已有（`results/lfm_fusion/e2e/lfm25_exp3_l1_C_nocfg_{fwd,rev}/`）：
- L1 only = 21.530 req/s
- L1 + L3 = 22.879 req/s

缺：
- **L1 + L2**
- **L1 + L2 + L3** ← **这是"在前面两个阶段基础上还能提升多少"的唯一答案**

```bash
# 只补 cfg 两格：把 exp3_layered.sh 复制一份，注释掉两行 nocfg
cp scripts/lfm_fusion/exp3_layered.sh /tmp/exp3_cfg_only.sh
sed -i 's/^run nocfg_/#run nocfg_/' /tmp/exp3_cfg_only.sh
WARMUP=12 REPS=30 GPU=3 PORT=52141 REGIME=C_long_prefill_tuned SUITE=l1_ \
    bash /tmp/exp3_cfg_only.sh
```

### ⚠️ 关于噪声：根因已查明，是**测量窗口太短**

原先这里写的是「加 server lifetime」。**那个诊断不完整，已修正。**

`R_long_prefill` 是 `in=4000, out=32, conc=4, **n=4**` —— **总共 4 个请求**。
实测每次测量窗口：

| regime | 窗口 | 评价 |
|---|---:|---|
| F tool agent | **37.98 s** | ✅ 极稳 |
| A low-batch decode | 4.75 s | ✅ |
| A low-batch decode **_tuned** | 4.77 s | ✅ |
| B concurrent decode | 1.47 s | ✅ 尚可 |
| C long prefill | **0.307 s** | ⚠️ |
| C long prefill **_tuned** | **0.196 s** | ⚠️⚠️ 最差 |

**所谓「两个顺序差一倍」的 9%，绝对值是 16 毫秒**（fwd baseline 0.1785s vs
rev 0.1947s）。在 180 毫秒的窗口里，一次 Python GC、一次调度 tick、一次 allocator
抖动就是这个量级。

看尾部更清楚：两组的**最快值几乎相同**（0.1715 vs 0.1765，差 5ms），
**差异全在最慢值**（0.1969 vs 0.2319，差 35ms）—— 典型的偶发抖动，不是系统性差异。

`serving_ceiling_lib.py` 里本来就标注过：
```python
"R_long_prefill": 4,   # ~0.3 s/run, drift 36.5 % -> needs the most
"tool_agent": 0,       # ~42 s/run, drift 0.7 % -> already steady state
```

**→ 问题只出在 `R_long_prefill`，其余 5 个 workload 的窗口都够长。**
而 L1 的赢家把它进一步加速，**窗口从 0.31s 砍到 0.20s，噪声反而更大**。

**处理方式**（按有效性排序）：

1. **★ 加长窗口** —— 把 `R_long_prefill` 的 `--num-prompts` 从 4 提到 40
   （`serving_ceiling_lib.py:96-101`）。窗口从 0.2s 变 2s，噪声降一个数量级。
   **⚠️ 但这会改变 workload 定义**：启动瞬态被摊薄，绝对吞吐会升高，
   **无法与已测的格子直接比较**。要用就得**整行重测**，别混用。
2. **加 server lifetime** —— 每个顺序跑 3-4 个 lifetime 再池化，把 lifetime
   当随机效应。**不改 workload 定义，与已有数据可比。** 代价是时间 ×3。
3. **至少**：报数时把两个顺序**分别列出**，不要只报几何平均。

**建议**：C 那一行已经有 n=60 的数据了，效应（+6.3%）远大于顺序间的不一致，
**不必重跑**；但报数时要写「两个顺序分别为 +3.99% / +8.75%」。
**D/E/F 不受影响**（窗口分别约 3s / 20s / 38s）。

## 7. 需要改的代码

`scripts/lfm_fusion/lf_e2e.py` 的 `REGIME_SERVING`（42-59 行）现在只有：

```python
"A_low_batch_decode":   dict(workload="R_short_decode",      cap=32, chunk=-1, policy="lpm", mem=0.85),
"B_concurrent_decode":  dict(workload="R_concurrent_decode", cap=32, chunk=-1, policy="lpm", mem=0.85),
"C_long_prefill":       dict(workload="R_long_prefill",      cap=32, chunk=-1, policy="lpm", mem=0.85),
"C_long_prefill_tuned": dict(workload="R_long_prefill",      cap=8,  chunk=2048, policy="fcfs", mem=0.9),
```

**要加 8 个条目**（cookbook 旋钮统一是 `cap=32, chunk=-1, lpm, mem=0.85`；
tuned 旋钮来自 07-24 验证 pass，源数据
`results/2026-07-24_serving_ceiling_validation/analysis/lfm25/ceiling_per_regime.json`）：

```python
# --- cookbook serving（新增 3 个 workload）---
"D_medium_balanced":  dict(workload="R_medium_balanced", cap=32, chunk=-1, policy="lpm", mem=0.85),
"E_shared_prefix":    dict(workload="shared_prefix",     cap=32, chunk=-1, policy="lpm", mem=0.85),
"F_tool_agent":       dict(workload="tool_agent",        cap=32, chunk=-1, policy="lpm", mem=0.85),

# --- L1 ceiling serving（验证 pass 的赢家）---
"A_low_batch_decode_tuned":  dict(workload="R_short_decode",      cap=8,   chunk=-1,   policy="fcfs", mem=0.85),
"B_concurrent_decode_tuned": dict(workload="R_concurrent_decode", cap=64,  chunk=8192, policy="fcfs", mem=0.75),
"D_medium_balanced_tuned":   dict(workload="R_medium_balanced",   cap=8,   chunk=2048, policy="fcfs", mem=0.9),
"E_shared_prefix_tuned":     dict(workload="shared_prefix",       cap=96,  chunk=2048, policy="lpm",  mem=0.9),
"F_tool_agent_tuned":        dict(workload="tool_agent",          cap=128, chunk=8192, policy="lpm",  mem=0.75),
```

**不需要改 harness 逻辑**——`run_workload` 已经按 workload 名字分发，
`shared_prefix` / `tool_agent` 在 `serving_ceiling_lib.WORKLOADS` 里已经接好。

### ⚠️ `REGIME_SHORT` 的推导会撞车

`exp3_layered.sh` 第 39 行：

```bash
REGIME_SHORT=$( [ "$REGIME" = "C_long_prefill" ] && echo "" || echo "${REGIME%%_*}_" )
```

`${REGIME%%_*}` 取第一个下划线前的部分，所以
`D_medium_balanced` 和 `D_medium_balanced_tuned` **都得到 `D_`**
→ 结果目录会互相覆盖。

**必须靠 `SUITE` 区分**（exp5 就是这么做的，用 `SUITE=l1_`）：

```bash
# cookbook 版
GPU=3 PORT=52141 REGIME=F_tool_agent          bash scripts/lfm_fusion/exp3_layered.sh
# L1 ceiling 版 —— 一定要带 SUITE
WARMUP=? REPS=? SUITE=l1_ GPU=4 PORT=52142 REGIME=F_tool_agent_tuned \
    bash scripts/lfm_fusion/exp3_layered.sh
```

结果目录名：`lfm25_exp3_{SUITE}{REGIME_SHORT}{nocfg,cfg}_{fwd,rev}`

---

## 8. 分析

```bash
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python
$PY scripts/lfm_fusion/exp3_analyze.py --regime <REGIME>
$PY scripts/lfm_fusion/exp3_analyze.py --regime <REGIME> --suite l1_
```

输出落到 `results/lfm_fusion/e2e/exp3_layered_*_summary.json`，
结构是 `cells[{nocfg,cfg}_{fwd,rev}][{baseline,all7}] = [30 个数]`。

**每次分析完先检查 fwd/rev 一致性**（见 §6）。不一致就别急着填表。

---

## 9. 时间估算

| Regime | cookbook 次 | L1 ceiling 次 | 备注 |
|---|---|---|---|
| A | ✅ 已完成 | ~35 min | |
| B | ~35 min | ~35 min | |
| C | ✅ 已完成 | 🔄 缺 cfg 两格，~40 min | **最高优先级** |
| D | ~30 min | ~35 min | |
| E | ~50 min | ~55 min | 20 s/run |
| F | ~70 min | ~75 min | 42 s/run，**真实 trace** |

**≈ 7 GPU-小时串行；GPU 3 和 4 并行 → ≈ 3.5 小时。**
时间主要花在 server 启动（8 lifetime × ~3.5 min），E/F 由 benchmark 时间主导。

⚠️ 如果按 §6 加 lifetime 数，时间会翻 2-3 倍。**先跑 regime C 那两格看噪声，
再决定 D/E/F 要不要加。**

---

## 10. 每个新 regime 预期能教什么（预测，不是结果）

- **D medium balanced**：唯一一个 token 数**卡在两个 Triton kernel 门控之间**的
  （`conv` 要 T≥2048，`moesum` 要 T≤32 或 T≥4096）。预期**生效组件最少**，
  是「四种不同形状的收益」这个论断的**阴性对照**。
- **E shared prefix**：radix cache 大量复用 → 实际 prefill 的 token 数远低于名义
  2048。`conv` 还能不能过 T≥2048 那道门是未知的，答案决定**在 synthetic workload
  上调的 shape 门控能不能扛住 prefix caching**。
- **F tool agent**：**唯一的真实 trace**。现在每个 kernel 层的数字都是 synthetic 测的，
  而 Mason 明确要求 shape 来自真实端到端运行。这一行把研究从"在我们自己设计的
  benchmark 上测的"变成"**在我们没设计的流量上测的**"。

---

## 11. 用户交代的其他事

- **GSM8K 太慢**，用户说 **20 题左右就够**。相关脚本 `sglang.test.few_shot_gsm8k`
  有 `--num-questions` 参数。**注意**：题数降到 20 后二项抽样误差从 ±2.6 点涨到
  **±21 点**，那已经不是质量闸门只是"没崩"检查——报数时必须说明。
- 用户在**同时看结果**，有阶段性数字就报。
- 文档：中文放 `docs/YYYY-MM-DD/`，**给 mentor 的英文版是
  `docs/2026-08-03/LFM25_ablation_matrix_EN.md`**，代码和 commit message 用英文。
- 每步实验存原始数据并 push。
- **跑 GPU 前先确认卡是空的**。

---

## 12. 关键文件

| 用途 | 路径 |
|---|---|
| **主表（要填的）** | `docs/2026-08-03/LFM25_ablation_matrix_EN.md` |
| 详细过程（中文） | `docs/2026-08-03/LFM25_FINAL_CASE_full_record.md` |
| L2×L3 实验全记录 | `docs/2026-08-03/exp3_kernel_on_tuned_baseline.md` |
| 驱动脚本 | `scripts/lfm_fusion/exp3_layered.sh` |
| A/B harness | `scripts/lfm_fusion/lf_e2e.py`（`REGIME_SERVING` 在 42 行） |
| 分析器 | `scripts/lfm_fusion/exp3_analyze.py` |
| workload 定义 | `scripts/serving_ceiling_lib.py`（`WORKLOADS` / `WARMUP_RUNS`） |
| L1 逐 regime ceiling | `results/2026-07-24_serving_ceiling_validation/analysis/lfm25/ceiling_per_regime.json` |
| MoE tuned config | `configs/regime_kernel/profiles/lfm25_pr_candidate/` |
| 候选 `_down.json` | `configs/regime_kernel/profiles/lfm25_bias_guarded_tma/` |
| 手写 Triton kernel | `scripts/lfm_fusion/lf_triton_shortconv.py`、`lf_triton_moesum.py` |
| 融合补丁（注入层） | `scripts/lfm_fusion/lfm_fusion_patch.py` |
