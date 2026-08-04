# 补全 LFM2.5 消融矩阵：6 regime × 8 列，48 格全测

**日期**：2026-08-04（03:00–03:45 完成最后一批）· **GPU**：H200 #3 和 #4 并行
**模型**：LFM2.5-8B-A1B · TP1 · bf16 · sglang 0.5.12.post1 @ `17f7a1da1` · Triton 3.5.1
**任务来源**：`docs/2026-08-03/HANDOFF_fill_ablation_matrix.md`（含 00:45 更新版）
**主表（英文，给 mentor）**：`docs/2026-08-03/LFM25_ablation_matrix_EN.md`

---

## 0. 一句话

**48 格全部填完**，全部在本次 campaign 内、同一棵树、同一协议、每臂 n≥16 且做了臂顺序对照，
所以「只有 ratio 可比」的 † 标记全部消失，**行内绝对值可以直接比较**。

最重要的结论：**在 serving 调优动不到 3% 的四个 regime 上，kernel rewrite 稳定值 6.2–8.4%，
且完全不受下面两层影响。** 交付的核心论点从站在 1 个 regime 上变成站在 4 个上。

---

## 1. 矩阵是什么

`L1`（serving config）× `L2`（MoE kernel config）× `L3`（kernel rewrite，7 项）的 2³ 全因子，
横跨 6 个 workload。一次 `exp3_layered.sh` 调用 = `{L2 关/开} × {正序/逆序}` 4 次
`lf_e2e.py`、8 个 server lifetime，填 4 格；每个 regime 跑两次（cookbook serving + L1 ceiling
serving）填满整行 8 格。**共 12 次调用。**

| 层 | 改什么 | 怎么切换 |
|---|---|---|
| L1 | 4 个 serving 旋钮 | 换 `lf_e2e.py` 的 `REGIME_SERVING` 条目 |
| L2 | fused-MoE Triton tile 参数 | `SGLANG_MOE_CONFIG_DIR` 环境变量（**两臂共享**，所以是基线属性不是臂间差异） |
| L3 | 7 处代码（含 2 个手写 Triton kernel） | `LFM_FUSION_PATCH` 环境变量 |

---

## 2. 结果：L3 的增量对比下层占掉的空间

| Regime | 输入形状 | **L1 单独** | L3 在 cookbook 上 | **L3 在 L1 上** | L3 在 L1+L2 上 |
|---|---|---:|---:|---:|---:|
| **A** 低批 decode | in≈100, conc=1 | **−0.57%** | +6.70% | **+7.46%** | +7.35% |
| **B** 并发 decode | in≈200, conc=32 | +1.11% | +6.72% | **+6.62%** | +7.14% |
| **D** 中等均衡 | in≈800, conc=8 | +2.52% | +8.29% | **+8.40%** | +8.38% |
| **C** 长 prefill | in≈4000, conc=4 | +77.7% | +6.18% | **+6.26%** | +6.38% |
| **E** shared prefix | sys 2048×8 组 | +93.8% | +7.24% | **+1.84%** | +2.51% |
| **F** 真实 trace（TTFT p50） | mooncake toolagent | −44% | −7.91% | **−3.64%** | −5.56% |

### 2.1 主结论：A/B/C/D 四行上 L3 与下层正交

前四行里三列数字只差**几分之一个点**。也就是说 kernel rewrite 削掉的那部分开销，
serving 调优和 MoE config 调优**都碰不到**。这是交付要证明的那件事。

### 2.2 E 和 F 是例外，且方向可解释

E 上 L3 从 +7.24% 掉到 +1.84%，F 上 TTFT 收益从 −7.9% 掉到 −3.6%。
**两者的 L1 赢家都是靠开 `chunked_prefill_size` 取胜的**，而分块预填充削的正是
kernel 工作在削的同一份 per-forward 开销 —— 消完了别人就没得消。

这是本项目此前只在 **kernel 层内部**观察到的次可加性（兑现率 0.90/0.70/0.49），
**第一次出现在跨层组合上**。

### 2.3 C 证明了关键不是「L1 有多大」而是「削的是不是同一份成本」

C 上 L1 也有 **+77.7%**，但 L3 完好保留 +6.26%。
差别在于 C 的 L1 赢家是靠**把 prefill 批得更好**来提吞吐的，
不是靠消除 elementwise 流量，所以不和 kernel 工作抢同一块肉。

> **规则**：判断两层优化会不会互相吃掉，看的是**机制是否重叠**，不是各自幅度大小。

### 2.4 L2 只跟一件事相关：prefill 量

```
A (in≈100)   +0.05%  n.s.
B (in≈200)   +1.68%
D (in≈800)   +1.85%
E (sys 2048) +12.49%
C (in≈4000)  +23.26%
```

单调，无例外。原因是那份 config 用的 guarded 策略把 `M ≤ 32` 的桶
逐字段写成默认启发式，而 CUDA graph 捕获的 decode batch 全落在那一段。

---

## 3. 三个单独值得说的发现

### 3.1 ★ regime A 的 L1「ceiling」没能复现

07-24 那次 192 配置穷举在 A 上选出 `cap8/chunk-1/fcfs/mem0.85`，记录为 **+0.38%**。
在本次 harness 里重测（n=24，双向对照）：**−0.57%**，两个顺序一致
（1.6788 / 1.6745 vs cookbook 1.6863）。

**即 192 个配置里最好的那个，换一次独立测量就赢不了 cookbook。**

这比正数更有利于交付：「autotuning 到顶了」的最强形式是
**「两层独立的调优、其中一层在自己空间里穷举，合计动不到 1%」**。

> 注意这**不是**说 07-24 的数据错了 —— 那次的 +0.38% 本身就在噪声量级内。
> 两次测量都对，只是这个 regime 上根本没有可复现的 serving 增益。

### 3.2 ★ 真实 trace 上吞吐是错的口径 —— 而且五个 regime 的吞吐本来就是延迟

**（2026-08-04 补充，用户提问后做的分析，结论比原先强得多）**

先证实机制。`bench_serving` 的 `--request-rate` 默认 `inf`：

- **5 个 synthetic workload**：所有请求 t=0 全部发出，只受 `--max-concurrency` 闸门限制。
  **闭环、打满** —— 一个请求完成立刻补下一个，吞吐 = 服务器能力。
- **tool_agent（mooncake）**：`get_mooncake_request_over_time()` 按 trace 原始时间戳
  `await asyncio.sleep()` 到点才发。**开环、到达率驱动** ——
  吞吐 = trace 的到达率，**与服务器快慢无关**。

用 Little's Law（在飞请求数 N = 吞吐 × 延迟）量化，不需要新测量：

| Regime | 并发上限 | 吞吐 | E2E mean | **在飞 N** | **利用率** |
|---|---:|---:|---:|---:|---:|
| A 低批 decode | 1 | 1.686 | 591.5 ms | 1.00 | **100%** |
| C 长 prefill | 4 | 12.119 | 321.4 ms | 3.90 | **97%** |
| D 中等均衡 | 8 | 6.900 | 1153.9 ms | 7.96 | **100%** |
| B 并发 decode | 32 | 21.661 | 1463.5 ms | 31.70 | **99%** |
| E shared prefix | 64 | 14.220 | 3923.5 ms | 55.79 | **87%** |
| **F tool agent** | 64 | 5.265 | 901.7 ms | **4.75** | **7%** |

**F 只用掉 64 个并发槽里的 4.75 个 —— 服务器 93% 的时间在空等 trace 发下一个请求。**
把它变快退不掉更多请求，因为**根本没有请求在排队**。

反过来说：**前五个 regime 的「吞吐提升」本来就是延迟提升。**
N 被钉死时 `吞吐 = N / 延迟`，所以延迟降 6% **就是** 吞吐升 6%：

| Regime | E2E mean | TTFT p50 | TPOT p50 | 吞吐（**由延迟预测**） | 吞吐（**实测**） |
|---|---:|---:|---:|---:|---:|
| A 低批 decode | −6.29% | −5.73% | −6.32% | +6.71% | **+6.70%** |
| C 长 prefill | −6.27% | −6.19% | −6.45% | +6.69% | **+6.18%** |
| D 中等均衡 | −7.67% | −5.80% | −7.84% | +8.30% | **+8.29%** |
| B 并发 decode | −6.42% | −7.61% | −6.26% | +6.87% | **+6.72%** |
| E shared prefix | −6.62% | −7.43% | −7.19% | +7.09% | **+7.24%** |
| **F tool agent** | **−6.04%** | **−7.91%** | −4.47% | +6.43% | **+0.40%** ← N 可动 |

**五个 regime 上预测值与实测值差不到半个点。** F 差了 6 个点，
因为它是唯一 N 可以自由变动的 workload —— 那 6% 变成了**空闲时间**，没变成吞吐。

> **★ 结论：kernel rewrite 在全部六个 workload 上都把端到端延迟砍掉 6.0–7.7%，
> 真实 trace 也不例外。根本不存在「弱 regime」。**
> 只存在**一个吞吐指标看不见结果的 regime**，而它恰好是唯一一个到达模式不是我们自己设计的。
> 而且在生产环境里，**挂 SLO 的是延迟，不是吞吐**。

脚本：`scripts/lfm_fusion/exp3_littles_law.py`

### 3.2.1 F 上四层的 TTFT 阶梯

```
S0 cookbook   321.2 ms
+ L2          225.3 ms   (−30%)
+ L1          179.2 ms   (−44%)
+ L1 + L2     138.4 ms   (−57%)
```

按吞吐排三层无法区分（总跨度 0.6%）；按 TTFT 排，顺序是 L1 > L2 > L3，且每层都是大效应。

### 3.3 ★ regime D 推翻了自己的事前预测

交接文档 §10 预测 D 会是**阴性对照**：它的 token 数（800）卡在两个 Triton kernel 的门控
之间（`conv` 要 T≥2048，`moesum` 要 T≤32 或 T≥4096），预期「生效组件最少」。

**实测 D 是全矩阵最高的 kernel 增益：+8.29%**（A +6.70%、B +6.72%、C +6.18%）。

预测把「生效组件最少」等同于「效应最小」了。剩下 5 项调用点改动显然不依赖那两个门控。
**具体是哪几项贡献的，D 上没做过 per-component 消融，这是未解问题。**

---

## 4. 两个不需要 GPU 就解掉的阻塞项

### 4.1 `_down` MoE config：现成候选**不能用，会让 server 崩**

交接文档 §5 把它列为「必须先决定，否则后面全要重测」。

一个 MoE 层跑两个 grouped GEMM（上投影 `w13`、下投影 `w2`），sglang 用两个独立配置文件调，
并断言两者 `BLOCK_SIZE_M` 相等（`fused_moe_triton_config.py:264`）。

逐桶比对 `lfm25_bias_guarded_tma/` 下那个现成 `_down.json`，**零冲突** —— 但这是假象：
**两个文件是各自独立做最近桶查找的**，而 down 文件缺 `24/48/96/1536/3072` 五个桶。
按真实查找逻辑模拟：

```
M = 97..112:   up 落 96 桶 (BLOCK_SIZE_M=16)
               down 落 128 桶 (BLOCK_SIZE_M=32)   → assert 触发，启动即崩
```

**而 regime A 的输入正好是 100 token。**

用它必须在 `BLOCK_SIZE_M` 相等约束下、按 up 的 19 个桶重扫下投影（≈2 GPU-小时），
且**已填的每个 L2 格都要重测**。本次交付不含它 ——
**所以本矩阵里的 L2 是低于它自身 ceiling 的**，约一半 MoE GEMM 工作跑在不是为它调的配置上。

判定逻辑已固化为 `scripts/check_moe_down_config.py`。

### 4.2 「L2 依赖 serving 配置」拿到了不依赖计时的证据

C 上 L2 叠在 L1 之上读出 −5.19%，但那个数**不可用**（见 §5.2）。
机制却可以确定性地证实：sglang 为每个 prefill batch 打印 `#new-token`，
而桶选择是最近邻，所以映射是确定的。`scripts/analyze_moe_bucket_usage.py` 直接读 server 日志：

| serving 配置 | prefill 次数 | 选中的桶 |
|---|---:|---|
| cookbook（`chunk=-1`） | 36 | **4096**(24)、**8192**(12) |
| L1 ceiling（`chunk=2048`） | 171 | 512(38)、**1024(117)**、1536(6)、2048(10) |

**两个分布完全不重叠。** config 是在 M≥4000 处扫出来的，而 L1 配置下那两个桶
**一次都没被选中**；68% 的 forward 落在 1024 桶，其 `BLOCK_SIZE_M=128`
意味着只切出 8 个 block 去喂 132 个 SM。

> **结论：tuned kernel config 绑定于它被 tune 时的 serving 配置，不能跨 serving 配置迁移。**
> 这和我们已有的「regime→backend 规则跨模型不可迁移」是同一类命题。

---

## 5. 方法学：这轮踩到和修掉的坑

### 5.1 泄漏的 server 会被健康检查当成自己人

`lf_e2e.py` 用 `setsid` 起 server，**杀父进程杀不掉它**。
`wait_health` 只探端口、不验证应答者是不是刚 spawn 的那个，于是下一轮 A/B 静默地
测那台遗留 server（症状：报「patch never applied」、吞吐莫名变化、server log 0 字节）。

已加 `assert_port_free(port)` 直接拒绝启动 + server log 为空判为外来响应者。

### 5.2 短窗口 workload 的方差不能用「加 lifetime」解释

交接文档 00:45 更新版修正了我上一轮的诊断，这个修正是对的，记在这里：

| regime | 单次测量窗口 |
|---|---:|
| F tool agent | 37.98 s ✅ |
| A 低批 decode | 4.75 s ✅ |
| B 并发 decode | 1.47 s ✅ |
| C 长 prefill | 0.307 s ⚠️ |
| **C 长 prefill _tuned** | **0.196 s** ⚠️⚠️ |

C 上「两个顺序差 9%」的绝对值只有 **16 毫秒** —— 一次 GC、一次调度 tick 就是这个量级。
**根因是窗口太短（`R_long_prefill` 只有 4 个请求），不是 lifetime 之间真有系统性差异。**

**直接后果**：C 的 `L1 → L1+L2` 那一格（−5.19%）是**跨 lifetime 比较**，
两个 nocfg lifetime 相差 1.86 req/s、两个 cfg 相差 0.17，
而池化均值之差只有 1.12 —— **小于单个 lifetime 自身的跨度**。
p 值假设重复独立，但同一 lifetime 内的重复是相关的。**这一格标注为不可引用。**

其余 5 个 workload 窗口都够长，不受影响。

### 5.3 运行期间不能编辑 bash 脚本

bash 按字节偏移增量读脚本，运行中改动会让它中途 `unexpected EOF`。上一轮因此丢了一格。

### 5.4 会话中断会杀掉 attached 的后台进程

本轮有两次实验被会话中断杀掉（A_tuned 丢三格、F 的 server 空转 17 分钟）。
**改用 `nohup` + `detach:true` 后不再发生。**

### 5.5 两个防呆已写进 wrapper（不靠人记得）

- `_tuned` regime 自动 `SUITE=l1_`：否则 `D_medium_balanced` 和 `D_medium_balanced_tuned`
  的结果目录都推导成 `D_`，**互相覆盖**；
- `_tuned` regime 不给 `WARMUP` 直接拒绝启动：warmup 表是按 cookbook 旋钮标定的，
  换 serving 配置就是换稳态，用默认值曾产出「+0.0%，kernel 收益消失了」的假数据。

---

## 6. 两个独立复现（跨 11 天、不同脚本）

| 量 | 07-24 campaign | 本次 | 差 |
|---|---:|---:|---|
| L1 on B | +1.11% | +1.11% | 0 |
| L1 on E | +93.61% | +93.76% | 0.15 点 |
| L1 on A | +0.38% | **−0.57%** | **未复现，见 §3.1** |

---

## 7. 复现

```bash
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
PY=~/.conda/envs/sglang-dev/bin/python
cd $REPO

# cookbook serving 行（A/B/C 已有，D/E/F 如下）
GPU=3 PORT=52154 REGIME=D_medium_balanced bash scripts/lfm_fusion/exp3_layered.sh
GPU=4 PORT=52164 REGIME=E_shared_prefix   bash scripts/lfm_fusion/exp3_layered.sh
GPU=4 PORT=52162 REGIME=F_tool_agent      bash scripts/lfm_fusion/exp3_layered.sh

# L1 ceiling serving 行（必须显式给 WARMUP，见 §5.5）
WARMUP=6  REPS=12 GPU=3 PORT=52152 REGIME=A_low_batch_decode_tuned  bash scripts/lfm_fusion/exp3_layered.sh
WARMUP=6  REPS=12 GPU=3 PORT=52156 REGIME=B_concurrent_decode_tuned bash scripts/lfm_fusion/exp3_layered.sh
WARMUP=12 REPS=30 GPU=3 PORT=52151 REGIME=C_long_prefill_tuned      bash scripts/lfm_fusion/exp3_layered.sh
WARMUP=6  REPS=16 GPU=3 PORT=52155 REGIME=D_medium_balanced_tuned   bash scripts/lfm_fusion/exp3_layered.sh
WARMUP=4  REPS=10 GPU=4 PORT=52165 REGIME=E_shared_prefix_tuned     bash scripts/lfm_fusion/exp3_layered.sh
WARMUP=1  REPS=8  GPU=4 PORT=52166 REGIME=F_tool_agent_tuned        bash scripts/lfm_fusion/exp3_layered.sh

# 分析
$PY scripts/lfm_fusion/exp3_analyze.py --regime <REGIME> [--suite l1_]
cd scripts/lfm_fusion && $PY exp3_latency.py --regime F_tool_agent --level {nocfg,cfg}   # F 必看
$PY scripts/lfm_fusion/exp3_littles_law.py                      # §3.2 闭环 vs 开环
$PY scripts/analyze_moe_bucket_usage.py <server.log> ...        # §4.2
$PY scripts/check_moe_down_config.py <up.json> <down.json>      # §4.1
```

| 产物 | 路径 |
|---|---|
| 主表（英文） | `docs/2026-08-03/LFM25_ablation_matrix_EN.md` |
| 逐次原始结果 | `results/lfm_fusion/e2e/lfm25_exp3_*/`（72 个目录） |
| 统计汇总 | `results/lfm_fusion/e2e/exp3_layered_*_summary.json`（12 份） |
| 运行日志 | `logs/2026-08-04/m_*.log` |
| bucket 直方图 | `logs/2026-08-04/moe_bucket_usage.txt` |
| F 延迟表 | `logs/2026-08-04/F_latency{,_cfg}.txt` |
| 闭环/开环分析 | `logs/2026-08-04/littles_law.txt` |

> `logs/` 被 `.gitignore` 排除，只在本机；所有关键数字都在 `results/` 的 JSON 里。

---

## 8. 仍然开放

1. **per-component 消融只在 A/B 跑过。** D 的 +8.29% 是全矩阵最高，
   但不知道 7 项里哪几项贡献的。
2. **`_down` config**（§4.1）—— L2 全程低于自身 ceiling。
3. **C 的 `L1 → L1+L2` 那一格**（§5.2）—— 唯一不可引用的数。
   要修得把 `R_long_prefill` 的 `--num-prompts` 从 4 提到 40，
   但那会改变 workload 定义，**必须整行重测**，不能和现有格子混用。

---

## 9. 一句方法学总结

> 同一个 kernel 改动，在 6 个 workload 上给出 +1.84% 到 +8.29% 的吞吐、
> 以及在真实 trace 上「吞吐 +0.4% 但 TTFT −7.9%」。
> **单个 regime、单个指标的数字不构成结论 —— 结论在矩阵里，不在格子里。**

补一句更准确的（§3.2）：

> **那六个吞吐数字里有五个其实是延迟数字。** 闭环打满时 `吞吐 = N / 延迟`，
> 两者是同一个测量的两种写法；只有当 N 能自由变动（真实到达模式）时它们才分家，
> 而那时候**吞吐是失效的那一个**。
> 换句话说：这个 kernel 改动在六个 workload 上的真实效果是**一致的 6.0–7.7% 延迟下降**，
> 之前看到的「regime 之间差异很大」有一部分是指标的假象。
