# 交接文档：kernel fusion gap 的自由探索 → agent loop 化

**写于**：2026-08-03 · **用途**：上下文过长，换窗口继续
**当前状态**：SLO-agent 有 **5 个 commit 未推送**，**PR 未开**（用户要求开 PR，不要 push main）

---

## 0. 一句话现状

把我们两周里人工找 kernel fusion gap 的方法，做成了 SLO-agent 里的 `scan` 阶段 +
`kernel_fusion_gap` opportunity mode。**回测 5/5 重现了历史案例**，包括最难的 OLMo-2。
剩下最后一步：**推分支 + 开 PR**。

---

## 1. 两个仓库

| 仓库 | 路径 | 角色 | 状态 |
|---|---|---|---|
| EndtoEnd-auto-optimization | `/home/t-jialianggu/work/EndtoEnd-auto-optimization` | 我们的实验仓库（自由探索模式） | 已推送到 main，干净 |
| **SLO-agent** | `/home/t-jialianggu/work/SLO-agent` | Chendi 的产品仓库（agent loop） | **分支 `feat/kernel-fusion-gap-mode`，5 commit 未推送** |

**用户明确要求**：SLO-agent **不要直接 push main**，改动合理就**开 PR**。

SLO-agent 远端：`https://github.com/Lurkrazy/SLO-agent.git`（SSH key 没配，用 HTTPS clone 的）

---

## 2. 未完成的事（按优先级）

### 2.1 ★ 推分支 + 开 PR（唯一必做项）

```bash
cd /home/t-jialianggu/work/SLO-agent
git push -u origin feat/kernel-fusion-gap-mode
# 然后开 PR，标题建议：
#   feat: a scan stage for fused kernels a call site never reaches
```

PR 正文要点（详见 §4 的成果）：
- 新增 `scan` 阶段 + `kernel_fusion_gap` mode
- 回测 5/5 重现历史案例
- 3 处 `pr_wiki.py` 移植性修复（**这些不属于新功能，但不修 loop 在别人机器上跑不起来**，PR 里要单独说明）
- 107 个测试通过（原 88 + 新增 19），**唯一失败是仓库既有的** `test_repo_wiki` 哈希不匹配，与本改动无关

### 2.2 用户还想要的（未做完）

用户原话：「做完这一步 你可以继续探索新模型上的机会，同时用旧仓库的自由探索和新仓库的
agentloop，我要看自由模式和loop模式会不会有显著区别」

- 已写了 `docs/2026-08-02/free_exploration_vs_agent_loop.md`（EndtoEnd 仓库，已推送）
- **但新模型探索做得不够**：只扫了框架源码，没在新模型上做端到端验证

---

## 3. 历史脉络（为什么会做到这一步）

### 3.1 起点：6 个 kernel fusion 案例

全部记录在 `EndtoEnd-auto-optimization/docs/kernel_fusion_catalogue.md`（已推送）。

| # | 模型 | 原语 | 收益 | 怎么发现的 |
|---|---|---|---|---|
| 1 | LFM2.5 | `fused_add_rmsnorm` | +2.35% decode | 静态扫描 1a |
| 2 | LFM2.5 | `fused_qk_norm_rope` | +5.42% 并发 decode | 静态扫描 1b |
| 3 | Gemma-3 | `gemma_rmsnorm` | 2.13×（诚实增量 +36.6%） | profiling + 读代码 |
| 4 | Gemma-3 | 同上的 rank 守卫 | 含在 #3 | profiling + 读代码 |
| 5 | Gemma-3 | `fused_qk_norm_rope` | **+0.5%~+1.1%** | 静态扫描（自动） |
| 6 | **OLMo-2** | 它自己的融合 norm | **prefill 1.24× / +24%** | 扫描 + profiling 交叉 |

**统一模式**：融合 kernel 已经写好编译好了，某个模型的调用点没调用它。

### 3.2 三个必须记住的教训（踩过坑）

1. **基线要含所有在飞的上游修复**。案例 5 对 main 测出 1.39×，加消融臂后发现
   **97% 是 PR #32670 的功劳**，自己只有 +1%。这个坑踩过**两次**（另一次是 #32383）。
2. **数值验证要对 fp64，不能拿 bf16 比 bf16**。两次误判（"4% 精度损失"、"87% rope 不一致"）
   都是这个原因。且要看**平均**相对误差，最大值会被近零元素主导。
3. **greedy 解码换 seed 不是噪声基线**（恒为 0），要用配对 McNemar。

### 3.3 对 SLO-agent 的判断（分析过程）

读完 3000 行 `src/` 后的结论：

- 它是**证据账本 + 闸门校验器**，不是发现器。loop 从 `observe` 开始，等你把跑好的数递进去
- `src/` 里**零 FX、零 torch.compile、零 profiler**（纯 stdlib）
- FX 只在 `.copilot/skills/` 里，那是**从 NVIDIA TensorRT-LLM 移植的 27 个技能文档**，
  可执行文件被刻意没抄，路径指向 TRT-LLM 而非 sglang
- **它的 `result_gates` 很硬（repeats≥3、regression_veto、指纹校验），正是我们缺的**

**缺的三样**（就是我补的）：源码扫描型观察、语义等价闸门、按执行路径分流的检测。

---

## 4. 已完成的改动（SLO-agent，5 个 commit）

```
1ae5811 docs: record the fourth shape and what narrowing it cost
524525a fix(scan): nn.LayerNorm has nothing to dispatch, so it is not a gap
4e11dc9 feat(scan): find the residual gap by its convention, since it has no name
3f828bd docs: back-test record, and the false positive that sharpens the OLMo-2 case
c7315cf feat: a scan stage, so a gap in the source can enter the ledger
```

### 4.1 新文件

| 文件 | 作用 |
|---|---|
| `src/sglang_agent_kernel_lab/fusion_scan.py` | 核心扫描器，纯 stdlib、AST 分析 |
| `campaigns/kernel_fusion_gap.json` | 新 campaign 定义 |
| `scripts/back_test_known_cases.py` | 回测 6 个历史案例 |
| `tests/test_fusion_scan.py` | 19 个新测试 |
| `docs/kernel_fusion_gap_backtest.md` | 回测记录 |
| `docs/back_test_result.json` | 回测原始数据 |

### 4.2 改的文件

- `campaigns.py`：加 `kernel_fusion_gap` 到枚举 + `WORKLOAD_INDEPENDENT_MODES` + `is_workload_independent()`
- `evaluator.py`：加 `_kernel_fusion_gap_opportunity_gates()`；dataset/workload/concurrency/target_m
  对该 mode 变成**不适用**（不是可选）；observation 改用 `scan_sha256` 定位证据
- `cli.py`：加 `scan` 子命令
- `pr_wiki.py`：**3 处移植性修复**（`is_file()` / `exists()` / `resolve()` 在无权限路径上抛
  `PermissionError`，表现为 "unexpected error" 看不出哪个检查失败；现在 fail closed 且带诊断）

### 4.3 扫描器识别的 4 种形态

| 形态 | 找什么 | 对应案例 |
|---|---|---|
| `never_wired` | 模型文件从不提同僚在用的原语 | 1, 2, 5 |
| `rank_guarded` | `forward_cuda` 只对某些 rank 走 kernel | 4 |
| `path_guarded` | kernel 调用和 `forward_native` 在**执行路径**（非输入）条件的两侧 | **6** |
| `residual_not_deferred` | 自己加 residual 而不用 deferred 惯例 | 1 |

**`path_guarded` 是最关键的一条**，因为：
- grep 看得见 `self.q_norm(`，会误判为"已接入"
- decode 的 profile 是干净的（graph 捕获时录的是融合路径）
- `--disable-cuda-graph`（看单个 kernel 的标准手段）会**强制走 eager 分支**
- 只有 prefill 一直在丢

**技术关键点**：OLMo-2 的融合侧是 `self.q_norm(x)` 这种**模块调用**，不含任何 kernel 名。
所以扫描器要把「同一模块在别处以 `forward_native` 形式出现」也算作融合调用——
**这条规则是从 miss 变成 hit 的关键**。

### 4.4 回测结果：5/5

```
 #  model     shape                     result        detail
 1  LFM2.5    never_wired               REDISCOVERED  lfm2_moe.py in the candidate list
 2  LFM2.5    never_wired               REDISCOVERED  lfm2_moe.py in the candidate list
 3  Gemma-3   never_wired_at_the_time   N/A           upstream #32383 closed this shape
 4  Gemma-3   rank_guarded              REDISCOVERED  Gemma3RMSNorm on `x.dim() == 2`
 5  Gemma-3   never_wired               REDISCOVERED  gemma3_causal.py in the candidate list
 6  OLMo-2    path_guarded              REDISCOVERED  Olmo2Attention on `get_is_capture_mode`
```

案例 3 报 N/A 是**正确的**：上游 #32383 已经修掉那个形态。

### 4.5 两个扫描的精度差异（必须诚实说明）

| 扫描 | 输出 | 精度 |
|---|---|---|
| `never_wired` | 32~40 候选 | **只有 3 个已知为真**，高召回低精度 |
| `rank_guarded` + `path_guarded` | 5 个 | 全部结构上为真 |

原因：**"提没提这个名字"不是判据**——多个模型通过 helper 调用。
`qwen3.py` 就是典型：它不提 `fused_qk_norm_rope`，但调了共享的 `apply_qk_norm`，
那个 helper 会**先试融合 kernel**。18 个模型用这个共享 helper。

**而 OLMo-2 自己定义了一个 `_apply_qk_norm`（同名少个下划线），私有副本直接跳过融合尝试。**
profiling 分得很清楚：qwen3 是 1 次 eager-norm 调用（0.32%），OLMo-2 是 97 次（6.62%）。

### 4.6 loop 实跑结果（已验证）

**OLMo-2 案例全部闸门通过**：
```
eligible: True
  PASS  primary_model / serving_regime / semantic_equivalence
  PASS  eager_calls_per_layer 6.06 >= 1.0
  PASS  gap_pct_of_kernel_time 6.62 >= 3
  PASS  gap_stage prefill
```

**两个必须被拒的也正确拒了**：
```
Qwen3-0.6B (0.32%)：           eligible False, failed [eager_calls_per_layer, gap_pct_of_kernel_time]
OLMo-2 用 per-head kernel：    eligible False, failed [semantic_equivalence]
                               —— 所有性能闸门仍然全绿
```

**第二条是最有价值的**：同样 6.62% 的缺口、所有性能指标达标，但因为数学不等价而被一票否决。
（OLMo-2 是跨 head 归一化，per-head kernel 会**静默算错**不是变慢）

---

## 5. 一个新发现（本轮补充验证的）

OLMo-2 的 `forward_native` 回退与融合路径 **bit-identical**：

```
forward_native vs fused dispatch:  max abs diff 0.0,  bit-identical True
native vs fp64 : 0.1410%
fused  vs fp64 : 0.1410%
```

所以 8/8 greedy 生成一致是**精确相等**，不是碰巧。那个 `forward_native` 调用
**不是在换精度，是白白丢性能**。

---

## 6. 环境与坑（复现必读）

```bash
ENV=~/.conda/envs/gemma-sglang; CU13=$ENV/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU13 PATH=$CU13/bin:$ENV/bin:$PATH LD_LIBRARY_PATH=$CU13/lib
```

| 坑 | 说明 |
|---|---|
| `import sglang` 报 AssertionError | `CUDA_HOME` 必须指向 `site-packages/nvidia/cu13` |
| SLO-agent 找不到模块 | 要 `PYTHONPATH=$PWD/src`（没装 editable） |
| PR wiki 引用 chendi 私有路径 | 我修了 3 处让它 fail closed；**测试时临时移开了 `entries/` 和 `annotations.json`，已恢复** |
| wiki 查询词格式 | 必须是 `sglang-32670` 或 `#32670`，**裸数字不匹配** |
| GPU 白名单 | `init` 只允许 GPU `{2,3}`，除非 `--allow-shared-gpu` |
| 用 `sgl.Engine` 的脚本 | 必须有 `if __name__ == "__main__":`，否则 spawn 子进程无限递归，父进程只报 "scheduler died" |
| OLMo-2 + fa3 + CUDA graph | 在**未打补丁的 main 上就崩** `cudaErrorIllegalAddress`，与我们改动无关，用 `--attention-backend triton` |

### 关键路径

| 用途 | 路径 |
|---|---|
| sglang 干净基线 | `/tmp/sglang_fqr_base`（origin/main @ `89f4a80c1f`） |
| Gemma-3 补丁树 | `/tmp/sglang_fqr` |
| OLMo-2 补丁树 | `/tmp/sglang_olmo` |
| loop 运行产物 | `/home/t-jialianggu/slo_runs/` |
| GSM8K 数据 | `/home/t-jialianggu/slo_runs/gsm8k.jsonl`（1319 题） |
| 新下的模型 | `/home/t-jialianggu/models/{OLMo-2-0425-1B-Instruct,EXAONE-4.0-1.2B,OLMoE-1B-7B-0924-Instruct}` |
| 现成模型 | `/data/hf/models/{gemma-3-1b-it,Qwen3-0.6B,Qwen3-30B-A3B-Instruct-2507,gemma-4-26B-A4B-it}` |

**注意 `/data/hf/models` 不可写**，要下模型放 `/home/t-jialianggu/models`（home 只剩 ~190G）。

---

## 7. 验证命令

```bash
cd /home/t-jialianggu/work/SLO-agent
PYTHONPATH=$PWD/src python -m unittest discover -s tests   # 107 tests, 1 既有失败
python -m compileall -q src && git diff --check
python scripts/back_test_known_cases.py --checkout /tmp/sglang_fqr_base
```

完整 loop 跑法见 `docs/kernel_fusion_gap_backtest.md` §7。

---

## 8. 未完成的技术债

1. **`residual_not_deferred` 精度不够**：最初报 48 个候选（含 vision encoder 等误报），
   收紧后是 8 个。最后一个 commit (`1ae5811`) 记录了收紧的代价。**还没在真实模型上验证过**。
2. **`never_wired` 精度低**：32 候选 3 真。这是设计如此（高召回），但 PR 里要说清楚。
3. **新模型探索不足**：只扫了源码，没做端到端。用户想看"自由模式 vs loop 模式"的对比，
   目前的对比文档偏分析、缺新案例。
4. **OLMo-2 的 PR 还没提给上游 sglang**：prefill 1.24×、改动 3 行、GSM8K p=1.000，
   补丁在 `EndtoEnd-auto-optimization/patches/olmo2_fused_qk_norm/`。

---

## 9. 用户偏好（重要）

- **严格区分「发现」和「验证」**：不能把工具只做了验证的说成是它发现的。用户多次揪出这类夸大。
- **撤回要显式标注**，不能悄悄改掉。
- 文档写**中文**，放 `docs/YYYY-MM-DD/`；代码和 commit message 写**英文**。
- 每步实验都要存文档/log/原始数据并 push。
- commit message 要详细讲清楚**为什么**，不只是做了什么。
