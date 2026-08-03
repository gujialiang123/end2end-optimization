# 交接：LFM2.5 分层实验（autotuning ceiling vs kernel rewrite）

**写于**：2026-08-03 · **原因**：上下文过长，换窗口继续
**上一份交接**：`docs/2026-08-03/HANDOFF_kernel_fusion_agent_loop.md`（那一份的任务已全部完成）

---

## 0. 当前状态一句话

Mentor 要的交付物是「**证明 best autotuning 是上限，kernel rewrite 还能再进一步**」。
**LFM2.5 上两半证据都已经有了，而且用的是同一个基线**——但从没画成一张图，
且 ceiling 那一半的可信度有硬伤。**实验 1 已完成**（结论见 §3），剩 2 个实验。

> **【2026-08-03 更新】实验 3 也已完成**，见
> `docs/2026-08-03/exp3_kernel_on_tuned_baseline.md`。
> 结果与本文档 §4「实验 3」的预期**相反**：kernel 增量没有缩到 +2~4%，而是
> 从 +6.18% 涨到 **+9.73%**（p=9.5e-19，counterbalanced n=16/臂）。
> 另外确认 tuned MoE config 在 decode 上精确中性（+0.05%, p=0.34），
> **所以 regime A/B 的 +6.57% / +6.21% 不需要重测**。
> **现在只剩实验 2（硬化 ceiling）和实验 4（NCU，可选）。**

---

## 1. Mentor 的交付要求（Copilot 整理，用户确认过）

### Debadeepta 的原话
> "Our aim is not to get torch compile working. It is to show that for different
> regimes we can genetically rewrite kernels to improve **beyond what the best
> auto tuning config provides**."

### 成功标准
> 有没有一个可信、可复现的实验说明：**autotuning 到这里就停了，但 kernel
> rewrite 又向前推进了 X%。**

### Mason 要的证据链
```
真实 serving workload → 识别 regime 和真实 shapes → best config autotuning 基线
→ profile 剩余瓶颈 → 选一个 kernel → rewrite/fuse → 正确性 + microbench
→ 端到端 serving benchmark
```

### 明确要停掉的
通用 FX importer、让所有模型支持 torch.compile、完整 autonomous agent framework、
无止境改 slides。

### 用户的决定
**focus 在 LFM2.5 上**，用 **GPU 4**。

---

## 2. LFM2.5 上我们已经有什么（都已核实，非记忆）

### 2.1 Config autotuning：**找不到任何提升**

`docs/2026-06-30/lfm2.5_conditional_autotuning.md`

| 臂 | R_concurrent_decode |
|---|---|
| **cookbook 默认**（3 次独立 server lifetime） | **23.74 ± 0.12 req/s**（stddev 0.5%） |
| Optuna v2 "best"（25 trial，条件化空间 288 组合） | 22.32 → **比基线低 6%** |
| 手工修正（MoE backend 换回 triton） | 23.53 → **持平** |

报告原文结论：
> 对 (LFM2.5-8B-A1B × 1× H200 × bf16)，**sglang 团队的开箱默认就是最优**

**基线配置**（报告 139 行）：
```
mem-fraction-static 0.85 · schedule-policy lpm ·
max-running-requests 32 · chunked-prefill-size -1 ·
schedule-conservativeness 1.0 · max-prefill-tokens 16384 ·
disable-radix-cache false · disable-cuda-graph false
```

### 2.2 Kernel 工作：**+5.30% ~ +6.57%**，全部显著

`docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md`

| regime | 七项全开 | p |
|---|---|---|
| A 低批 decode | **+6.57%** | 4.6e-14 |
| B 并发 decode | **+6.21%** | 2.4e-08 |
| C 长 prefill | **+5.30%** | 1.2e-05 |

七项明细（**两个是真手写 Triton kernel**）：

| 项 | 类型 | 收益 |
|---|---|---|
| `conv` | **手写 Triton** | 长 prefill +2.33% |
| `moesum` | **手写 Triton** | 低批 decode +4.55% |
| `qkrope` | 调用点 | 并发 decode +5.42% |
| `norm+scale` | 调用点 | decode +3.89% |
| `gate+idx` | 调用点 | **三 regime 全不显著**（诚实负面，要保留） |

### 2.3 ★ 两者基线一致（这是关键，已核实）

- 6/30 autotuning 基线：`mem 0.85 / lpm / cap 32 / chunk -1`
- 7/27 kernel A/B（`scripts/lfm_fusion/lf_e2e.py:42-48`）：
  ```python
  cap=32, chunk=-1, policy="lpm", mem=0.85
  ```
  **完全一致。**

而且 A/B 设计很干净：`LFM_FUSION_PATCH` 环境变量切换，**同一棵树、同一份 server 参数**，
不设变量走逐字未改的 sglang。正是后来在 Gemma-3 上才总结出的消融方法。

### 2.4 方法学产出（超出要求，是加分项）

- **同类优化强烈次可加**：兑现率 **0.90 / 0.70 / 0.49**，随 regime 饱和度单调下降
  → 「各项分别测量之和会高估整个 stack，系统越饱和高估越严重」
- **regime→backend 规则跨模型不可迁移**：用错最差 **−34%**

---

## 3. ★ 实验 1 结果（已完成，2026-08-03）

**问题**：7/27 那批 kernel A/B 跑的时候，树里有没有装 PR #32687 的 tuned MoE config？

**方法**：报告记录环境为 sglang `17f7a1da1`（2026-07-09）。直接查那个 commit 的文件树。

```bash
cd /home/t-jialianggu/work/sglang
git ls-tree -r 17f7a1da1 --name-only | grep "E=32,N=1792"
# → 0 个结果
```

**结论：没有装。** 那棵树里 `E=32,N=1792` 一个设备的 config 都没有
（LFM2.5 的 MoE shape 是 `E=32, N=1792`，见 `docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md`）。

**这意味着**：

```
Bar 1  sglang 裸默认
Bar 2  cookbook 默认              ← 25 次 Optuna 无法超越 = ceiling
Bar 3  + tuned MoE kernel config  ← ★ 缺这一格，从没和 Bar 4 一起测过
Bar 4  + kernel rewrite/fusion    ← 现有的 +5.30~6.57% 是叠在 Bar 2 上的，不是 Bar 3
```

**所以 Bar 4 必须在装了 tuned config 的基线上重测。**
> ⚠️ **2026-08-03 修正**：本行原文写「按次可加性规律（0.90/0.70/0.49），增量很可能从
> +6% 缩到 +2~4%」。**那个预测方向是错的，已撤回** —— 它把「同类优化次可加」的经验规律
> 错误地套到了「不同类优化」上。
>
> 正确的 Amdahl 推导：L2 只改 `fused_moe_kernel` 的 tile 参数（占长 prefill kernel 时间
> **73.6%**），L3 的 7 项**全部避开 GEMM 本身**（占 26.4%）。若两者作用在时间轴的不相交
> 部分，L3 省下的绝对时间不变而总时间从 1.0 降到 0.81，**分母变小 → 相对增量变大**：
>
> ```
> 基线                   t=1.0000  thr=1.0000
> + L2 tuned MoE config  t=0.8108  thr=1.2334  (+23.34%)
> + L3 kernel rewrite    t=0.7604  thr=1.3150  (+31.50%)
> → L3 叠在 L2 之上 = +6.62%（对比：叠在未 tune 基线上 = +5.30%）
> ```
>
> 三个可能吃掉一部分的因素：`moesum` 与 `FusedMoE` 有结构性接触；e2e 含不缩水的
> 调度/tokenize/HTTP；我们自己的 waterfall 撞过（1.78×1.22 → 实测 1.70× 而非 2.17×）。
>
> **诚实预期区间：+2% ~ +6.6%，中位数猜 +4~5%。完整推导见
> `docs/2026-08-03/LFM25_FINAL_CASE_full_record.md` §9。**

> 这不是坏消息。「在最优 autotuning 之上仍有 X% 且统计显著」，X 小但基线干净，
> 远胜 X 大但基线脏——我们已经因为基线脏栽过两次（#32383、#32670）。

---

## 4. 还要补的实验

### 实验 2（★ 最重要）：把 ceiling 从软变硬

**问题**：现在的 ceiling 只有 25 次 TPE trial，而且**我们自己的报告说 TPE 坏掉了**——
前 7 个 trial 把 `triton MoE` 和差 batching 绑一起，之后 18 个 trial 再没试过
`triton + 好 batching`。

**审稿人一定会问**：「这不是 ceiling，是搜索失败。」

**做法**：在**裁剪后的可行空间**做**网格穷举**，不用 TPE。
LFM2.5 只有 `fa3` attention backend 可用（triton 不支持 hybrid 架构，
flashinfer 的 JIT 被 conda env libcuda 链接问题挡住），去掉这一维后空间应该
远小于 288。**能穷举的话，ceiling 就是硬的。**

- 工具：`harness/autotune_v2_lfm.py`、`autotune_v3_lfm.py`（已存在）
- 预计 2–4h，多卡可并行

### 实验 3：Bar 4 在 Bar 3 之上重测

> **✅ 2026-08-03 已完成。结果见 `docs/2026-08-03/exp3_kernel_on_tuned_baseline.md`。**
> 下面这段的预期（「增量很可能从 +6% 缩到 +2~4%」，见 §3 末）**已被实测推翻**：
> 实际是 **+6.18% → +9.73%**，超可加。原因拆成 Amdahl(+2.06 点) 与
> `moesum` × MoE config 的真实交互(+1.49 点)。以下保留原文作为记录。

装上 PR #32687 的 config（`patches/` 里有，或见
`docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md`），重跑 `lf_e2e.py` 的
`baseline` vs `all7` 两臂。

```bash
python scripts/lfm_fusion/lf_e2e.py --regime C_long_prefill --gpu 4 \
    --arms baseline,all7 --reps 6
```

**注意**：`lf_e2e.py` 已经是环境变量切换，**不要**为了加 config 而建第二棵 worktree
（Gemma-3 那次因此撞上 stride 问题，attention backend 直接拒绝）。
把 config 文件放进同一棵树即可，两臂都会看到它。

### 实验 4（可选，1h）：NCU headroom 串进叙事

`results/2026-07-08_v5_ncu/`、`v6_ncu`、`2026-07-10_v9_ncu_realworkload/` 有数据，
但没接进主线。需要一句话：「NCU 显示这个 kernel 有 X% headroom，所以我们选它」
——这是 Mason 证据链的第 4 步。

---

## 5. 建议的主 regime

**C 长 prefill**：
- `conv` 手写 kernel 在这里最有效（+2.33%）
- 次可加性损失最小（兑现率 0.90，B 并发 decode 只有 0.49）
- prefill 更容易展示 kernel-bound 特性

B 并发 decode 收益最大（+6.21%）但饱和度最高，叠加后缩水最多。

---

## 6. 关键文件

| 用途 | 路径 |
|---|---|
| autotuning ceiling（LFM） | `docs/2026-06-30/lfm2.5_conditional_autotuning.md` |
| autotuning ceiling（Qwen，对照） | `docs/2026-06-25/autotuning_ceiling_report.md` |
| kernel 工作全报告 | `docs/2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md` |
| MoE config PR 草稿 | `docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md` |
| e2e A/B harness | `scripts/lfm_fusion/lf_e2e.py` |
| 手写 kernel | `scripts/lfm_fusion/lf_triton_moesum.py`、`lf_triton_shortconv.py` |
| 融合补丁 | `scripts/lfm_fusion/lfm_fusion_patch.py` |
| 算子审计 | `scripts/lfm_fusion/lf_audit.py` |
| 原始数据 | `results/lfm_fusion/` |
| NCU 数据 | `results/2026-07-08_v5_ncu/`、`v6_ncu`、`2026-07-10_v9_ncu_realworkload/` |

---

## 7. ⚠️ 一个必须处理的矛盾

`docs/2026-06-25/autotuning_ceiling_report.md`（Qwen3-30B）里写着：

> "**This challenges the original 'agent for kernel rewriting' motivation**:
> the gap between default and ceiling is *huge*... Agent value should be
> redirected toward *automating the search itself*, **not rewriting kernels**."

**如果这份和 kernel 成果一起交上去，等于在论证 Debadeepta 的目标不成立。**

这是苹果比橘子：
- Qwen 的 `8.86×` 是 **default → tuned**（众所周知的巨大空档）
- LFM 的 `+6%` 是 **在已调优基线之上**的增量

**两个数不可比，但图上一放读者一定会比。**

**处理方式**：LFM2.5 的故事恰好把这条反驳变成支撑——
在 LFM2.5 上 **config autotuning 完全找不到提升**（25 trial 全部低于或持平默认），
所以「autotuning 是上限」在这个模型上是**更强的形式**：不是收益递减，是零收益。
交付时应以 LFM2.5 为主线，Qwen 那份作为「不同模型 ceiling 高度差异极大」的对照。

---

## 8. 上一阶段已完成的（不用重做）

### SLO-agent（Chendi 的产品仓库）
- **PR #9**：`scan` 阶段 + `kernel_fusion_gap` mode，回测 5/5 重现历史案例
- **PR #30**：`knowledge/case_studies/` 6 个确认 + 5 个否决案例（base = #9 分支）
- 分支 `docs/case-studies-from-prior-campaigns`，工作区干净，116 测试
- **注意**：唯一失败的 `test_repo_wiki` 哈希是仓库既有问题，与我们无关

### 上游 sglang
- **issue #33415** + **draft PR #33416**：OLMo-2 prefill 1.24×（等 review）
- **PR #32670**（Gemma-3 RMSNorm）、**#32687**（LFM MoE config）：更早提的

### Review
- 已给 **sglang#33293**（GraniteMoe fusion）发了 review，主要意见是
  `torch.compiler.is_compiling()` 应该门控快路径以避免 BS1 退化 4.45%

---

## 9. 环境与坑

```bash
ENV=~/.conda/envs/gemma-sglang; CU13=$ENV/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU13 PATH=$CU13/bin:$ENV/bin:$PATH LD_LIBRARY_PATH=$CU13/lib
```

**注意**：LFM 那批实验用的是 **`sglang-dev` 环境 + sglang 0.5.12.post1 @ `17f7a1da1`**
（torch 2.9.1+cu128 / Triton 3.5.1），不是 gemma-sglang。重跑时要对齐，
否则 Triton 版本不同会改变 MoE config 查找路径。

| 坑 | 说明 |
|---|---|
| `import sglang` AssertionError | `CUDA_HOME` 要指向 `site-packages/nvidia/cu13` |
| 用 `sgl.Engine` 的脚本 | 必须有 `if __name__ == "__main__":`，否则 spawn 子进程递归 |
| 消融臂 | **同一棵树用环境变量切换**，不要建第二棵 worktree |
| 噪声基线 | greedy 换 seed 恒为 0，要用配对 McNemar |
| 数值验证 | 对 fp64，不能 bf16 比 bf16；看**平均**不看最大 |
| 基线 | 必须含所有在飞的上游修复，否则报别人的功劳（已栽两次） |
| **泄漏 server**（8/3 新增） | server 用 `setsid` 起，**Ctrl-C / 杀脚本杀不掉它**。`wait_health` 只探端口不认进程，于是下一次跑会静默地测那台旧 server。已在 `lf_e2e.py` 加 `assert_port_free`；仍要养成 `ps -eo pid,cmd \| grep launch_server` 的习惯 |
| **运行中改脚本**（8/3 新增） | bash 按字节偏移增量读脚本，运行期间编辑会让它中途 `unexpected EOF` 挂掉 |
| **`--skip-correctness`**（8/3 新增） | 7/27 那批 LFM e2e **全部**跳过了正确性闸门，`correctness.json` 里 outputs 全空。改用 `--correctness-nogate`：记录但不否决 |

---

## 10. 用户偏好

- **严格区分「发现」和「验证」**——不能把只做了验证的工具说成是它发现的
- **撤回要显式标注**，不能悄悄改掉
- 文档写**中文**放 `docs/YYYY-MM-DD/`；代码和 commit message 写**英文**
- 每步实验都要存文档/log/原始数据并 push
- commit message 要讲清楚**为什么**，不只是做了什么
- 跑 GPU 前先问，除非已给 auto 权限
