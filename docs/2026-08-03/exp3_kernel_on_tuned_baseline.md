# 实验 3：把 kernel 增量重测在装了 tuned MoE config 的干净基线上

**日期**：2026-08-03 · **GPU**：H200 #4 · **模型**：LFM2.5-8B-A1B (TP1, bf16)
**上游树**：`sglang @ 17f7a1da1`（editable 安装于 `~/.conda/envs/sglang-dev`）
**任务来源**：`docs/2026-08-03/HANDOFF_lfm25_layered_experiment.md` §4「实验 3」

---

## 0. 一句话结论

**预期它会缩水，结果它变大了。**

在长 prefill 上，kernel rewrite 的端到端增量从脏基线上的 **+6.18%** 变成干净基线上的
**+9.73%**（p = 9.5e-19）。低批 decode 上则**完全不受影响**（+6.70% → +6.35%），
因为 tuned MoE config 在 decode 上本就是中性的。

交接文档按本项目自己的次可加规律（兑现率 0.90 / 0.70 / 0.49）预测增量会掉到 +2~4%。
**这个预测是错的，而且错的方向本身是个新结果**：两层优化在长 prefill 上是**超可加**的
（兑现率 1.14）。

补测六项臂后，这 +3.54 个百分点能干净地拆成两半：

- **+2.06 点是 Amdahl**：六个 MoE 之外的改动省下的**绝对**时间几乎不变（4.98 → 5.25 ms/req），
  但请求总时间被 config 压掉 19%，同一份节省就占了更大比例；
- **+1.49 点是真实的 kernel 交互**：`moesum` 是七项里唯一伸进 `FusedMoE` 的，它在
  未调优的 MoE 上是 **−0.08%（p=0.88，中性）**，在调优后的 MoE 上是 **+1.69%（p=2.8e-04）**。

---

## 1. 为什么要做这个实验（discovery）

7/27 的 kernel 报告在 regime C 上给出 +5.30%。它跑在 `sglang 17f7a1da1` 上，
而那棵树**没有 LFM2.5 的 MoE tuned config**——`E=32,N=1792` 对任何设备都不存在。

今天在活树上二次确认（不只是查历史 commit）：

```bash
$ pip show sglang | grep Editable
Editable project location: /home/t-jialianggu/work/sglang/python
$ git -C /home/t-jialianggu/work/sglang log --oneline -1
17f7a1da1a fix(bench_serving): handle mooncake dict records ...
$ find /home/t-jialianggu/work/sglang/python -name "E=32,N=1792*"
（无输出）
```

而 `docs/2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md`（已开为上游 PR #32687）
测得那份 config 在**同一个 regime C** 上值 **+23.34%**。

所以 7/27 的 +5.30% 是叠在一个**缺了 +23% 的基线**上的。这正是本项目已经栽过两次的
「脏基线」问题（#32383、#32670），必须在交付前修掉。

---

## 2. 四根柱子怎么定义

```
Bar 1  sglang 裸默认
Bar 2  cookbook 默认 serving config    ← autotuning ceiling
Bar 3  + tuned MoE kernel config       ← PR #32687
Bar 4  + kernel rewrite / fusion       ← 七项 all7
```

**Bar 1 == Bar 2**：`docs/2026-06-30/lfm2.5_conditional_autotuning.md` 里，288 组合的
条件化空间上跑 25 次 Optuna，**没有一个配置超过 cookbook 默认**（best 比默认低 6%，
手工修正后持平）。所以在 LFM2.5 上 "autotuning 的上限" 就是 cookbook 默认本身。

**冻结的 serving 参数**（三个 regime 一致，与 6/30 和 7/27 完全相同）：

```
mem-fraction-static 0.85 · schedule-policy lpm · max-running-requests 32 ·
chunked-prefill-size -1 · schedule-conservativeness 1.0 ·
max-prefill-tokens 16384 · attention fa3 · moe-runner auto · cuda graph on
```

Bar 3 的加法只通过 `SGLANG_MOE_CONFIG_DIR` 指向
`configs/regime_kernel/profiles/lfm25_pr_candidate`，**不建第二棵 worktree**
（Gemma-3 那次因此撞上 stride 问题）。server 日志确认它被读到：

```
Using MoE kernel config from .../lfm25_pr_candidate/configs/triton_3_5_1/
E=32,N=1792,device_name=NVIDIA_H200.json.
```

---

## 3. 实验设计：2×2，而不是一次 A/B

`{tuned MoE config 关, 开} × {臂顺序 正序, 逆序}`，每格 8 次计分重复。

**config 轴**不只是为了拿 Bar 3。它让 Bar2→Bar4（脏基线增量）和 Bar3→Bar4（干净基线增量）
**在同一个 session、同一棵树上测出来**，而不是拿今天的数去比一个月前的数。

**顺序轴不是可选项。** `lf_e2e.py` 是顺序跑臂、一臂一个 server lifetime，而 PR #32687
的端到端工作已经抓到过这个 harness 产生**纯粹由顺序造成的符号翻转**（"先跑的那个更快"，
−0.37% 和 +0.12%）。今天的数据里位置效应依然可见（regime C 的 baseline：正序 12.020、
逆序 12.219，差 1.7%），**比要测的 kernel 效应的一半还大**。7/27 的原实验是单一顺序、
每臂 6 次，这一点当时没有处理。

合并两个顺序后每臂 n = 16，来自 **2 个独立的 server lifetime**。

```bash
GPU=4 REPS=8 PORT=52141 REGIME=C_long_prefill \
    bash scripts/lfm_fusion/exp3_layered.sh
/home/t-jialianggu/.conda/envs/sglang-dev/bin/python \
    scripts/lfm_fusion/exp3_analyze.py --regime C_long_prefill
```

统计用 Welch t + **精确 Student-t 尾**（n 小，正态近似 anti-conservative）。

---

## 4. 过程中撞到的两个坑（都已修进 harness）

### 4.1 泄漏的 server 被健康检查当成自己人 —— 第一批数据因此作废

第一次跑到一半我把脚本停了。但 server 是用 `setsid` 起的，**它活了下来**，继续占着
GPU 4 和端口 52140。重跑时 `wait_health` 只探
`http://127.0.0.1:<port>/health`，**从不检查应答的是不是它刚 spawn 的那个进程**，
于是瞬间"健康"，整个 A/B 测的是那台遗留 server。

症状极具迷惑性：

| 现象 | 真实原因 |
|---|---|
| all7 报 "patch never applied (silent no-op)" | 旧 server 当然没打补丁 |
| baseline 从 12.28 跳到 13.0–13.3 req/s | 换了一台已经热起来的 server |
| `server_all7.log` 是 **0 字节** | 新进程根本没起来 |

**修复**（`scripts/lfm_fusion/lf_e2e.py`）：

- `assert_port_free(port)`：启动前端口若已有人应答，直接 `SystemExit` 并提示怎么找到
  泄漏进程；
- `check_patch_applied`：server 日志为空一律判为"健康检查打到了别人的 server"。

**教训**：`stop_bash` / Ctrl-C 只杀父进程；`setsid` 起的 server 必须显式
`ps -eo pid,cmd | grep launch_server` 找出来 kill。

### 4.2 correctness gate 在 7/27 那批实验里**从未被执行过**

今天第一次真跑这个闸门，`all7` 以 2/5 被拒。回头查发现：

```
lfm25/{A,B,C}/correctness.json          outputs: []
lfm25_all/{A,B,C}/correctness.json      outputs: []
lfm25_all7/{A,B,C}/correctness.json     outputs: []
lfm25_conv/{A,B,C}/correctness.json     outputs: []
lfm25_moesum/{A,B}/correctness.json     outputs: []
```

**7/27 那一整批 LFM2.5 的 e2e A/B 全部带 `--skip-correctness`**，记录下来的签名全是空的。

那 2/5 是真问题吗？**不是**：

| prompt | 一致 | 说明 |
|---|---|---|
| "List the first ten prime numbers..." | ✅ 逐 token 相同 | 正常文本 |
| "The capital of France is" | ✅ 逐 token 相同 | 正常文本 |
| 其余 3 个 | ❌ | **baseline 自己就退化**成 `So. So. So.` / `Kernel Kernel` / `):):):`|

三个不一致的 prompt 上，baseline 本身陷入重复退化循环，两臂的分歧只是"在第几个 `So`
后面插句点"——即 top-2 logits 近乎持平时被任意数值扰动翻转。token-identity 在这里
**没有判别力**。而且逆序单元根本没法用这个闸门：先跑的那一臂定义签名，于是 baseline
会被 all7 的签名判为失败。

**修复**：加 `--correctness-nogate`，**记录签名但不否决**。
理由很简单——**记录比跳过强**。这些 kernel 真正的正确性证据是 7/27 报告里的
GSM8K（1319 题，8 个臂跨度 2.5 点，在三个噪声度量下都在噪声内）。

原始探针留档：`results/lfm_fusion/e2e/exp3_correctness_probe/`。

---

## 5. 结果

### 5.1 Regime C 长 prefill（in≈4000, out=32, conc=4）

每格均值（单个 server lifetime, n=8, req/s）：

| | 正序 | 逆序 |
|---|---|---|
| 无 config | baseline 12.020 ± 0.061 → all7 12.846 ± 0.169 | all7 12.892 ± 0.191 → baseline 12.219 ± 0.059 |
| 有 config | baseline 14.943 ± 0.109 → all7 16.372 ± 0.256 | all7 16.412 ± 0.114 → baseline 14.934 ± 0.135 |

合并两顺序（n=16/臂）：

| 比较 | req/s | 变化 | t | p |
|---|---|---:|---:|---|
| kernel 增量，**无** tuned MoE config | 12.119 → 12.869 | **+6.18%** | 13.44 | 4.5e-13 |
| kernel 增量，**有** tuned MoE config | 14.939 → 16.392 | **+9.73%** | 24.03 | 9.5e-19 |
| tuned MoE config 单独（baseline 臂） | 12.119 → 14.939 | **+23.26%** | 64.64 | 1.1e-33 |
| tuned MoE config 单独（all7 臂） | 12.869 → 16.392 | +27.38% | 50.52 | 2.1e-30 |
| **整栈** Bar2 → Bar4 | 12.119 → 16.392 | **+35.25%** | 71.66 | 1.3e-29 |

**两项独立复现**：

- `+6.18%` 复现了 7/27 的 `+5.30%`（同树、同 harness、隔一个月、这次带顺序对照）；
- `+23.26%` 复现了 PR 草稿的 `+23.34%`（隔六天，独立 session）。

**加性**：`1.2326 × 1.0618 = 1.3088` 预测，实测 `1.3525`，**兑现率 1.14 —— 超可加**。

### 5.2 Regime A 低批 decode（in≈100, out=256, conc=1）

| 比较 | req/s | 变化 | p |
|---|---|---:|---|
| kernel 增量，**无** tuned MoE config | 1.686 → 1.799 | **+6.70%** | 2.1e-41 |
| kernel 增量，**有** tuned MoE config | 1.687 → 1.794 | **+6.35%** | 1.8e-34 |
| tuned MoE config 单独（baseline 臂） | 1.686 → 1.687 | **+0.05%** | **0.34（不显著）** |
| 整栈 Bar2 → Bar4 | 1.686 → 1.794 | +6.41% | 1.2e-35 |

**config 在 decode 上精确中性**（+0.05%, p=0.34），独立复现 PR 草稿的
「−0.13%, p=0.079」。这不是巧合而是设计的直接产物：那份 config 用的是 **guarded 策略**，
`M ≤ 32` 的桶**逐字段等于 `get_default_config` 的默认启发式**，而 CUDA graph 捕获的
decode batch size 全部落在这一段。

**推论：7/27 在 regime A 和 B 上的 +6.57% / +6.21% 不需要重测。**
「脏基线」问题只影响 prefill。交接文档担心三个 regime 都要重来，实际上只有 C 要。

### 5.3 交付用的四柱数据（regime C）

| Bar | 内容 | req/s | 相对 Bar 2 |
|---|---|---:|---:|
| 1 | sglang 裸默认 serving config | — | （见 6/30 报告） |
| **2** | **cookbook 默认 = autotuning ceiling** | **12.119** | **1.000×** |
| **3** | **+ tuned MoE config（PR #32687）** | **14.939** | **1.233×** |
| **4** | **+ kernel rewrite（七项）** | **16.392** | **1.352×** |

**Bar 3 → Bar 4 = +9.73%，p = 9.5e-19。这就是 Debadeepta 要的那个数。**

---

## 6. 为什么会超可加

七项里只有 **`moesum`** 伸进了 `FusedMoE` 内部：它让该层返回**四个未合并的专家输出**
`[T,4,H]`（`no_combine`），再把 reduction 与 residual add、后续 RMSNorm 融成一个
Triton kernel。其余六项（`norm`/`scale`/`conv`/`gate`/`idx`/`qkrope`）都在 MoE 之外。

于是有一个可证伪的机制假设：

> tuned config 让 fused-MoE GEMM 变快之后，**被 `moesum` 消掉的那部分 combine 开销
> 在剩余时间里占比上升**，所以 `moesum` 的相对收益变大。

有两条既有证据支持它：

1. 7/27 的表里 **regime C 的 `moesum` 单项是空的（"—"），从没单独测过**；
2. 在脏基线上，六项臂 `all` 是 **+5.81%**，七项臂 `all7` 反而是 **+5.30%**
   ——加上 `moesum` 掉了 0.51 点。也就是说**在未调优的 MoE 上 moesum 是负担**。

如果假设成立，那么在装了 config 之后，`all7 − all` 应该由负转正。

### 6.1 验证：补测六项臂 `all`，用 `all7 − all` 反推 moesum

同样的 2×2（`{config 关,开} × {正逆序}`），把七项臂换成六项臂
（`norm,scale,conv,gate,idx,qkrope`，即**去掉 moesum**）。

```bash
SUITE=six_ ARMS_FWD=baseline,all ARMS_REV=all,baseline \
    REGIME=C_long_prefill GPU=4 REPS=8 PORT=52143 \
    bash scripts/lfm_fusion/exp3_layered.sh
$PY scripts/lfm_fusion/exp3_moesum_marginal.py
```

| 臂 | 无 config | 有 config |
|---|---:|---:|
| baseline（七项组） | 12.119 | 14.939 |
| baseline（六项组） | 12.103 | 14.861 |
| **`all` = 六项** | **12.879** | **16.119** |
| **`all7` = 六项 + moesum** | **12.869** | **16.392** |

**moesum 的边际贡献（`all7` vs `all`）：**

| 基线 | 变化 | t | p |
|---|---:|---:|---|
| 无 tuned MoE config | **−0.08%** | −0.16 | **0.88（精确中性）** |
| 有 tuned MoE config | **+1.69%** | 4.15 | **2.8e-04** |

**假设成立。** moesum 在未调优的 MoE 上一文不值，在调优后的 MoE 上值 +1.69%。

### 6.2 但 moesum 只解释一半——另一半是 Amdahl

| | 无 config | 有 config | 变化 |
|---|---:|---:|---:|
| 六项（`all`） | +6.41% | **+8.47%** | +2.06 点 |
| moesum 边际 | −0.23 点 | +1.26 点 | +1.49 点 |
| **合计（`all7`）** | **+6.18%** | **+9.73%** | **+3.54 点** |

六项**本身**也涨了 2.06 点，而它们碰都没碰 MoE。原因在绝对时间上一目了然：

```
六项节省   4.978 ms/req（未调优） → 5.253 ms/req（调优后）   比值 1.06
七项节省   4.805 ms/req（未调优） → 5.934 ms/req（调优后）   比值 1.23
```

**六项省下的绝对时间几乎是常数**（1.06×），但每个请求的总时间被 config 压掉了
19%（82.6 ms → 67.3 ms），于是同一份绝对节省占了更大的比例。这是教科书式的 Amdahl：
把常数节省 4.978 ms 代进调优后的基线，预测 `4.978/(67.29−4.978) = 7.99%`，
实测 8.47%——**Amdahl 解释了 2.06 点里的 1.58 点**。

七项的比值 1.23 则明显偏离常数，多出来的正是 moesum 与 MoE kernel 的真实交互。

**结论：超可加有两个来源，一个是平凡的（Amdahl），一个是真实的（moesum × MoE config）。
把它们分开报，比笼统说"超可加"诚实得多。**

---

## 7. 对交付叙事的影响

1. **主线 regime 应该是 C 长 prefill**，理由和交接文档不同：不是因为次可加损失最小，
   而是因为**只有 C 上这三层是可分离且都为正的**，能画出一张真正有三段的柱状图。
   A/B 上 Bar 2 == Bar 3，图会退化成两段。
2. **交付的数字是 +9.73%，不是 +5.30%，也不是 +6%。** 而且它是在
   *config autotuning 找不到任何提升* + *MoE kernel config 已调到最优* 之上的增量。
3. 交接文档 §7 指出的矛盾（6/25 Qwen 报告论证「不要改 kernel」）**没有被削弱**：
   Qwen 的 8.86× 是 default→tuned，LFM 的 +9.73% 是 on-top-of-tuned，两个数不可比，
   图上必须分开画。
4. **次可加不是普适规律。** 本项目此前只见过 0.90/0.70/0.49，据此写进了方法学结论。
   今天见到 1.14。正确的表述是：**同类优化（都在削同一份固定开销）次可加；异类优化
   （一个削 GEMM、一个削 GEMM 周边）可以超可加**，而且超可加还要再分成
   **平凡的 Amdahl 部分**和**真实的 kernel 交互部分**（§6.2）。这条要写回方法学，
   不能只报好消息。
5. **7/27 报告里 regime C 的一句结论要撤回。** 那里说七项（+5.30%）不如六项（+5.81%）、
   即 `moesum` 在长 prefill 上帮倒忙。今天测得在脏基线上它确实是 −0.08%（p=0.88，
   其实是**中性**而非负面），但**在干净基线上是 +1.69%（p=2.8e-04）**。
   原结论在它自己的基线上不算错，但作为对 `moesum` 的评价必须显式更正。

---

## 8. 复现

```bash
# 数据（每个 regime 约 30 分钟，8 个 server lifetime）
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization
GPU=4 REPS=8 PORT=52141 REGIME=C_long_prefill    bash scripts/lfm_fusion/exp3_layered.sh
GPU=4 REPS=8 PORT=52142 REGIME=A_low_batch_decode bash scripts/lfm_fusion/exp3_layered.sh
SUITE=six_ ARMS_FWD=baseline,all ARMS_REV=all,baseline \
    GPU=4 REPS=8 PORT=52143 REGIME=C_long_prefill bash scripts/lfm_fusion/exp3_layered.sh

# 分析
PY=~/.conda/envs/sglang-dev/bin/python
$PY scripts/lfm_fusion/exp3_analyze.py --regime C_long_prefill
$PY scripts/lfm_fusion/exp3_analyze.py --regime A_low_batch_decode
$PY scripts/lfm_fusion/exp3_analyze.py --regime C_long_prefill --suite six_
cd scripts/lfm_fusion && $PY exp3_moesum_marginal.py
```

> **不要在 bash 脚本运行期间编辑它。** bash 按字节偏移增量读取脚本，改动会让它在
> 中途 `unexpected EOF` 挂掉——今天 regime A 的后三格就是这么丢的，白跑一格。

| 产物 | 路径 |
|---|---|
| 原始逐次结果 | `results/lfm_fusion/e2e/lfm25_exp3_{,A_,six_}{nocfg,cfg}_{fwd,rev}/` |
| 统计汇总 | `results/lfm_fusion/e2e/exp3_layered_*_summary.json` |
| moesum 边际 | `results/lfm_fusion/e2e/exp3_moesum_marginal_C_long_prefill.json` |
| 运行日志 | `logs/2026-08-03/exp3_*.log` |
| 分析输出 | `logs/2026-08-03/exp3_analysis*.txt`、`exp3_moesum_marginal.txt` |
| correctness 探针 | `results/lfm_fusion/e2e/exp3_correctness_probe/` |

> `logs/` 被 `.gitignore` 排除，日志只在本机。关键数字都在 `results/` 里的 JSON。

---

## 9. 遗留与注意

1. **`sglang` 工作树有一处未提交改动**：`model_runner.py` 里 2026-06-11 打的
   flashinfer_cutlass autotune allowlist 补丁。两臂共享，**不影响 A/B**，但
   `lf_e2e.py` 文档字符串里"baseline 是逐字未改的 sglang"这句话严格说不成立，
   本实验的 baseline 是"17f7a1da1 + 那一处补丁"。7/27 的实验也在同一状态下跑。
2. **发现一个 PR #32687 可以补强的地方**：server 日志里还有
   `Config file not found ... E=32,N=1792,device_name=NVIDIA_H200_down.json` ——
   down projection 的 config 也缺。`configs/regime_kernel/profiles/lfm25_bias_guarded_tma/`
   下已经有一个 `_down.json`，值得测一下再决定要不要并进 PR。
3. **regime B 并发 decode 没有重测**。按 §5.2 的机制（config 只动 prefill），
   预期与 A 相同、即 +6.21% 成立，但这是推断不是测量。要严谨的话应该补一次。
4. 本次实验**没有**碰 autotuning ceiling 的硬化（交接文档的实验 2）。那仍然是
   审稿人最容易攻击的一环：25 次 TPE，而我们自己的报告说这次搜索失败了。

---

## 10. 一句方法学总结

> 把一个优化的收益报在"当时手头那棵树"上是不够的。同一个改动，在缺了一层的基线上
> 是 +6.18%，在补齐那层之后是 +9.73%——**方向还不是我们预期的那个**。
> 基线不是背景，它是被测量的一部分。
