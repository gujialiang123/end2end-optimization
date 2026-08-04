# Agent 工作流证据链：LFM2.5 的 kernel 机会是怎么被找到、验证、修好的

**日期**：2026-08-04
**目的**：回答「agent 具体是如何发现机会 → 迭代验证 → 做出修复的」
**方法**：**不靠回忆**。全部从 git 历史、脚本源码、子 agent 产出的 FINDINGS 文件重建，
每一条都可核验。

> 相关但角度不同的文档：`docs/2026-08-03/evidence_chain_how_gaps_were_found.md`
> 讲的是 SLO-agent（确定性 CLI）如何**重新发现**这些 gap。**本文讲原始发现过程。**

---

## 0. 一句话

**不是「profile 一下发现热点然后优化」。是「拿一个已有结论去证伪，发现它有边界条件，
然后沿着边界找」。最贵的一步是选对对照组。**

全过程 **07-27 19:28 → 23:20，约 4 小时，8 个 commit**，每个 commit 都是一轮
「假设 → 实现 → 测量 → 保留或撤回」。

---

## 1. 起点：一个挡路的既有结论

项目里有个 v33 审计结论，此后成为**降低 kernel 融合优先级的依据**：

> 「对 Qwen3-30B，sglang 热路径已全部 CUDA 融合，**没有可补的空缺**。」

**第一件事不是找机会，是质疑这个结论的适用范围。**

理由写在 `scripts/lfm_fusion/lf_audit.py:8-13` 的 docstring 里：

```
Why redo it for LFM2.5: v33 concluded "for Qwen3-30B every hot path is already
CUDA-fused, there is no gap to fill".  LFM2.5 is a different architecture
(18/24 layers are gated short convolutions) and its sglang implementation does
*not* use the fused residual+RMSNorm path, so the v33 conclusion cannot be
assumed to carry over.
```

**关键判断**：那个结论是**在一个模型上**得出的，而 LFM2.5 是本机上最新的架构，
**从没被算子级审计过**。

---

## 2. 发现步骤 1：换一个计数口径

### 沿用的部分

方法照抄 v33：`bench_one_batch --profile` + **关闭 CUDA graph**（让每个算子单独现形）
→ 按 kernel 名分桶。

### ★ 改掉的部分 —— 这一步是关键

v33 统计的是**时间占比**。时间占比会告诉你「MoE 占 70%」，然后你去优化 MoE，
**但那里已经没有空间了**。

**改成数「融合实现根本不会执行的 kernel」的个数，并且拿 Qwen 做对照。**

结果（每次 forward 的 kernel 启动次数）：

| 模型 | 未融合 RMSNorm | 独立 residual add | gating mul |
|---|---:|---:|---:|
| **LFM2.5** | **61** | **48** | **36** |
| Qwen3-30B（对照） | **1** | **0** | **0** |

### 为什么这个口径有决定性 —— 两点缺一不可

**① 计数是结构性的，不是约数**：

```
48 = 2 个 residual add × 24 层
36 = 2 个 gating mul  × 18 个 conv 层
```

**能被层数整除，意味着每一层都在犯同一个错**——这是实现漏了，不是某处的偶然。

**② 对照组排除了「这是 sglang 的通病」**：

Qwen 一整个 forward 只有 1 个未融合 norm、0 个独立 add。

> **没有对照组，61 和 48 只是两个数字，说明不了任何事。**
> **有了对照组，它们变成「这个模型文件的实现漏了」。**

---

## 3. 发现步骤 2：从「有 gap」到「gap 在哪一行」

数出 48 个独立 add 之后，还要找到**代码在哪**。这一步靠读源码，不靠工具。

在 `lfm2_moe.py:433-456`：

```python
def forward(self, layer_id, positions, hidden_states, residual, forward_batch, **kwargs):
    residual = hidden_states                    # ← 收了 residual 参数，第一行就覆盖
    normed = self.operator_norm(hidden_states)  # ← 没传 residual → 走非融合分支
    ...
    hidden_states = hidden_states + residual    # ← 独立 kernel
```

**三个事实必须同时看见才能定性**：

1. 函数签名**收了 `residual` 参数**，第一行就覆盖 —— 传进来的值从没被用过
2. `RMSNorm.forward_cuda(x, residual)` **本来就会走 `fused_add_rmsnorm`**
3. `Lfm2MoeModel.forward` **本来就在层间传递 residual**

→ **接线全都在，只是这一层没接上。**

> 这不是「发现一个可以优化的地方」，是**发现一个 bug**：
> 参数被声明、被传递、然后被丢弃。

---

## 4. 修复方法：不发明，去抄

**没有设计新方案，而是去找「正常的模型是怎么写的」**——`models/llama.py:304-316`
的 deferred-residual 写法，改成一样的。

数学等价性是**手工推导并写进代码注释**的：

```
原版：  a = op(rms(x));   h1 = a + x;    out = h1 + ffn(rms(h1))
新版：  r' = x;           n = rms(x);    a = op(n)
        r'' = a + x = h1; n2 = rms(h1);  返回 (ffn(n2), h1)
```

下一层拿到 `ffn(n2) + h1`，与原版的 `out` 是同一个值。

---

## 5. ★ 注入方式：让 baseline 成为真 baseline

**这一步决定了后面所有数字可不可信。**

所有改动通过 **`LFM_FUSION_PATCH` 环境变量 opt-in**。不设变量时走的是
**逐字未改动的 sglang 原路径**——同一棵树、同一个 commit、同一份 server 参数。

### 踩过的坑（记录在 `lfm_fusion_patch.py` 注释里）

模型类被 model registry **懒加载**。`sitecustomize` 执行时 `lfm2_moe` 还没导入，
**用定时器打 patch 是竞态**。改用 `sys.meta_path` finder，在该模块 exec 完成的
**瞬间**打补丁。

### 还加了一道防线

**server log 会被检查 patch 生效标记**——否则一个静默失效的 patch 会被当成
「与 baseline 相同」记录下来，从而得出「这个优化没用」的错误结论。

---

## 6. 迭代：8 个 commit，每个是一轮假设-测量-裁决

`git log --reverse -- scripts/lfm_fusion/` 就是迭代记录：

| 时间 | commit | 这一轮做了什么 | 结果 |
|---|---|---|---|
| 19:28 | `ba18c2e` | 审计 → 找到 norm/scale，实现，e2e | decode **+3.8/+4.0%** |
| 20:53 | `09cc7ed` | **手写 ShortConv 两个 Triton kernel** | 隔离 **5.9×/4.3×**，bit-exact |
| 21:17 | `0574701` | 那两个 kernel 的 e2e 验证 | 长 prefill **+2.3%**，decode **精确中性** |
| 21:40 | `f222b93` | **两个子 agent 并行深挖** + 三个新调用点改动 | 5 候选排序，**否决 2 个** |
| 22:21 | `a94df72` | 全栈组合验证 | **发现强次可加性** |
| 22:28 | `94f2af1` | ⚠️ **自我纠错：统计方法** | p 值被高估，修正 |
| 22:32 | `c3c8c88` | ⚠️ **自我纠错：数据丢失 bug** | 恢复被覆盖的曲线 |
| 23:20 | `81d9d28` | **第 4 个手写 kernel（moesum）** | 最终 **+6.57/+6.21/+5.30%** |

**注意 20:53:35 → 21:17:33 只隔 24 分钟**：写完 kernel **立刻**做 e2e 验证，
不等攒够再一起测。这是刻意的——隔离加速和端到端收益是两回事，
早验证才能早发现「microbenchmark 赢了但 e2e 没有」。

---

## 7. ★ 最有价值的一轮：两个子 agent 并行深挖（21:40）

主 agent 起了**两个子 agent**，从**不同视角**查同一个对象：

| 子 agent | 视角 | 产出 |
|---|---|---|
| **nsys 时间线** | 硬件实测 | `results/lfm_fusion/nsys/FINDINGS.md` |
| **FX / Inductor 图** | 编译器 | `results/lfm_fusion/fx/FINDINGS.md` |

**关键设计**：两个都跑在**已经打过 norm/scale/conv 补丁**的路径上，
所以它们找到的是**剩下什么**，不是**已经修好的东西**。

### 它们各自纠正了主线的错误

**nsys 纠正了一个方向性错误**：

> decode 开着 CUDA graph 时每层只有 **1.5 µs** 的设备空闲，关掉 graph 是 **689 µs**。
> **launch 开销在 graph 打开后根本不是 decode 的问题。**

→ 没有这条，会继续朝「减少 launch」优化 decode，**方向就错了**。

同时确认 **prefill 即使 decode 开着 graph 也是 eager 的**（两条 prefill trace
吻合到 0.5% 以内），所以 **prefill 的 launch 开销是真的**（~89 µs/层）。

**FX 纠正了机制解释** —— 这条最锋利：

主线原本认为 conv 那里的问题是「访问不合并」，**一个原因**。
FX 查出来是**两个独立效应**：

1. 转置和转置读**确实不合并**（14%/21% 峰值）
2. **但 `B_gate * x` 是合并的，仍然只有 54%** —— 因为**跨步的行**让
   `TensorIterator` 无法向量化，退化成标量 `elementwise_kernel` 而不是
   `vectorized_elementwise_kernel<8>`

**这是由 trace 里的 kernel 名直接确认的，不是推测。**
→ 直接催生了第三个 Triton kernel（`gate`）。

**FX 还独立验证了设计**：

> Inductor **自己就从未修改的模块推导出了 ShortConv 融合**，
> 生成的 kernel 与手写的**结构等价**。

→ **一个编译器独立选择了同样的方案**，说明手写版不是拍脑袋。

同时它说明为什么**必须**手写：`causal_conv1d_fwd/_update` 在 Inductor 眼里是
`ExternKernelSchedulerNode`，即**硬屏障**，融合只能发生在 conv 的两侧
——**这正是那两个 kernel 做的事**。

### ★ 否决的两个候选 —— 这比找到的更能说明方法可信

**候选 1：top-k + MoE alignment**

`topkGatingSigmoid` 加对齐占 graph decode kernel 时间的 **7.52%**，
**乍看是最大的目标**。但时间线否决了这个解读：graph 节点间隙只有 0.064–0.128 µs
——融合掉它们**没有证据支持的 7.5% 收益**。**作为纯融合被否决。**

**候选 2：在 MoE down GEMM 里直接原子归约**

能省掉同样的 22 个 sum kernel，并避免 11.5 GB 的 prefill 中间流量。
但它要改 Triton GEMM 主循环，风险评估后**近期否决**。

> **一个只会说「找到了 N 个机会」的流程是不可信的。**
> **能说清「为什么这两个看起来最大的反而不做」才是可信的。**

---

## 8. ⚠️ 两次自我纠错（22:28 和 22:32）

这两个 commit 没有增加任何性能，但它们是**流程可信度的核心证据**。

### 纠错 1：统计方法本身是错的（`94f2af1`）

> 分析用**正态近似**算 Welch p 值。在这些 run size 产生的 df≈5-10 下这是
> **anti-conservative** 的，而这恰恰影响那些**判决可能翻转的边缘臂**。

改用 `scipy stats.t.sf` 后：

- **没有任何结论翻转**（所有 improvement 仍是 improvement，`gate+idx` 仍不显著）
- **但文档里「全部 p<0.005」是错的** —— `qkrope` 在长 prefill 实为 **p=0.018**
- 把笼统声明**换成逐格 p 值**，原话：*so nothing has to be taken on trust*

> **这是主动降低自己结论的强度。没人要求做这件事。**

### 纠错 2：一个静默的数据丢失 bug（`c3c8c88`）

> 用 `--tokens` 重跑 `lf_bench_shortconv.py` 做局部抽查，
> **覆盖了 `shortconv_bench.json`**，静默销毁了图表所依赖的 8 点交叉曲线。
> **是工作树显示了一处意外修改才发现的。**

修复：加 `--out` 让局部扫描写到别处，并恢复完整曲线。

> **这个 bug 不影响任何已发布的数字，但如果不修，下次就会。**

---

## 9. ★ 最有价值的产出来自「测组合」这个决定（22:21）

**如果只报告各项单独的数字，会得到一个高估的 stack。**

| regime | 各项之和 | 一起测 | 兑现率 |
|---|---:|---:|---:|
| C 长 prefill | 5.86% | **5.30%** | 0.90 |
| A 低批 decode | 9.37% | **6.57%** | 0.70 |
| B 并发 decode | 12.80% | **6.21%** | **0.49** |

并发 decode 上：`qkrope` 单独 +5.42%，再加单独值 +3.65% 的组只多买到 **0.12 点**。

**兑现率精确跟踪 regime 的饱和程度**——长 prefill 每 forward 工作最多、
最能把开销藏起来，损失最小（0.90）；并发 decode 最饱和，损失最大（0.49，不到一半）。

> **规则：消除同一「种类」成本的优化不会相加。**
> **报告各项分别测量之和会高估整个 stack，且系统越饱和高估越严重。**

这条规则**只能通过「多花时间测组合」发现**。跳过这一步不会有任何报错。

---

## 10. 正确性：闸门自己被证伪了

第一版正确性闸门是 **token-identity**（贪心输出逐 token 相同）。它**失效了**：

LFM2.5 走 **top-4/32 路由，专家选择是离散 argmax**。`norm` 和 `qkrope` 是
**代数等价但非 bit-exact**（约 2 个 bf16 ulp，而且**融合版更准**）。
bf16 级扰动偶尔翻转选中的专家，输出就不连续地变了。

实测：12 个 prompt 里 top-1 有 11/12 一致，但 **KL 最高到 0.99**。

> **没有降低闸门标准，而是判定这个闸门对这个模型结构性不可用，然后换了一个。**

改用 GSM8K 全量 1319 题。**并且用一个 bit-exact 的对照臂免费标定噪声底**：

`scale` 臂**数学上必然等于 baseline**，却读数低 **0.8 点**
→ between-arm 系统噪声 ≥ 0.8 点，**由构造得到，不靠假设**。

所以口径是 **「未检测到质量回归」，不是「质量没变」**。

---

## 11. 这个流程的可复用形状

```
① 找一个挡路的既有结论，检查它的适用边界        ← 最容易被跳过，价值最高
② 换一个计数口径（数「不该存在的 kernel」）
③ ★ 选一个对照组（没有对照组，数字说明不了任何事）
④ 定位到具体代码行，判断是「可优化」还是「bug」
⑤ 抄正常实现，不发明
⑥ ★ 用环境变量注入，让 baseline 是逐字未改的原路径
⑦ 每个组件独立 e2e 验证，不攒批
⑧ ★ 起子 agent 从不同视角查同一对象，允许它们纠正主线
⑨ ★ 记录否决了什么，以及为什么
⑩ ★ 测组合，不要报各项之和
⑪ 闸门失效时换闸门，不是降标准
⑫ ★ 主动纠正自己的统计方法和 bug
```

**打星的六条是这次真正产生价值的部分。其余是常规工程。**

---

## 12. 三条可机械化的 signature

从这次过程中提炼、**已在其他模型上复现**的规律：

### signature 1：已有融合原语，调用点没用（**纯静态，不需要 GPU**）

> **枚举代码库里已有的融合原语，检查哪些模型的调用点没用它们。**

这一条就找到了**最大的两个赢家**（`fused_add_rmsnorm`、`fused_qk_norm_rope`）。
已在 Gemma-3、OLMo-2、GraniteMoe 上复现，固化进 SLO-agent 的 `fusion_scan.py`。

### signature 2：结构性整除的 kernel 计数

`48 = 2 × 24 层`、`36 = 2 × 18 层`。**能被层数整除说明每层都在犯同一个错。**

### signature 3：与对照模型的计数差

同一个 profiling 口径，跑一个已被充分优化的模型作对照。
**差值才是信号，绝对值不是。**

---

## 13. 诚实边界

- **这不是全自动的。** 全程 researcher-in-the-loop：agent 提出、实现、测量，
  人决定继续还是叫停。多次被叫停（例如 GPU 被别人占用时）。
- **步骤①（质疑既有结论）目前无法机械化。** 它依赖「知道项目里有哪些结论、
  它们是在什么条件下得出的」。
- **步骤④（判断是 bug 还是可优化点）需要读懂三处不相邻的代码**并同时持有；
  静态扫描器目前只能给候选，不能给判定。
- **signature 1 已经机械化**（`fusion_scan.py`），但精度不高：
  `never_wired` 形态 32–40 个候选里只有 3 个是真的。
- 子 agent 的价值是**独立视角**，不是并行加速。两个子 agent 都**纠正了主线的错误**
  ——如果它们只是重复主线的判断，就没有意义。
- **本文是事后重建，不是实时日志。** 依据是 git 历史、脚本注释和 FINDINGS 文件，
  这些都是当时写下的；但「当时怎么想的」这部分带有事后整理的成分。

---

## 14. 原始记录在哪

| 内容 | 路径 |
|---|---|
| 完整迭代序列 | `git log --reverse --format="%h\|%ad\|%s" -- scripts/lfm_fusion/` |
| 审计脚本（发现机制） | `scripts/lfm_fusion/lf_audit.py` |
| 审计原始输出 | `results/lfm_fusion/audit/` |
| **子 agent：nsys 时间线** | `results/lfm_fusion/nsys/FINDINGS.md` |
| **子 agent：FX/Inductor** | `results/lfm_fusion/fx/FINDINGS.md` |
| 注入层（含竞态的记录） | `scripts/lfm_fusion/lfm_fusion_patch.py` |
| 手写 Triton kernel | `scripts/lfm_fusion/lf_triton_shortconv.py`、`lf_triton_moesum.py` |
| e2e A/B harness | `scripts/lfm_fusion/lf_e2e.py` |
| 逐项配对结果 | `results/lfm_fusion/processed/fusion_ab*.csv` |
| 全报告 | `docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md` |
| 自我纠错 1（统计） | `git show 94f2af1` |
| 自我纠错 2（数据丢失） | `git show c3c8c88` |
| 源码级移植 | `gujialiang123/sglang` PR #1 |
