# 证据链：这几个 kernel fusion 机会到底是怎么被找到的

**日期**：2026-08-03
**目的**：回答「在 SLO-agent 里完整复现了找到这些机会」这句话背后的证据

---

## 0. 先纠正一个前提（重要）

你问的是「agent 工作记录、对话记录、每一轮对话、模型是什么、用了什么工具」。

这个问法假设 SLO-agent 是一个 **LLM agent**，会跑多轮对话。**它不是。**

我 grep 过整个 `src/`（3000 行）：

```bash
grep -rn "openai|anthropic|llm|prompt|completion|api_key" src/ --include=*.py
# 唯一命中：evaluator.py:437  "llm_score_used": False
```

那个字段是**硬编码的审计声明**，意思是「本次判定没有用 LLM 打分」。

> **SLO-agent 是确定性的 Python CLI**：一个账本 + 闸门校验器。
> 没有模型、没有对话、没有 token、没有随机性。同样输入永远同样输出。

所以**不存在**「agent 的对话记录」这种东西。但存在**比对话更硬的证据**：
内容寻址的 append-only JSONL 记录，每条都能用 sha256 复核。

---

## 1. 有两条独立的证据链，必须分开看

| | 链 A：**原始发现** | 链 B：**loop 重新发现** |
|---|---|---|
| 谁做的 | **我**（Claude Opus 5，在 GitHub Copilot CLI 里） | **确定性代码**（`fusion_scan.py`） |
| 有没有 LLM | 有（就是我） | **没有** |
| 有没有对话 | 有（就是这个 session） | **没有** |
| 用了什么 | grep / 读源码 / GPU profiling / 跑 A/B | AST 分析，秒级，不用 GPU |
| 时间 | 2026-07-31，约 3.5 小时 | 2026-08-02，1.7 秒 |
| 产出 | 发现 + 量化 + 验证 | **只有发现**（候选清单） |

**我之前说的「5/5 重现」指的是链 B。** 它重新找到了这些缺口的**位置**，
但**没有**重新产生那些性能数字——那些数字来自链 A。

下面两条都完整展开。

---

# 链 A：原始发现（OLMo-2 案例，最有代表性）

## A.1 我是什么

- **模型**：Claude Opus 5（`claude-opus-5`）
- **运行环境**：GitHub Copilot CLI
- **可用工具**：`bash` / `grep` / `glob` / `view` / `edit` / `create` / `sql` 等
- **对话记录**：就是这个 session 本身（我们从 7-31 一直聊到现在）

## A.2 关键的那几步（真实顺序）

### 第 1 步：你给的指令

> 「另找3-5个模型 从头到尾扫一遍 找到可能的优化点」

我据此下载了 3 个非 Qwen 家族的小模型（OLMo-2、OLMoE、EXAONE-4）。

### 第 2 步：静态可用性扫描

工具：`scripts/fx_fusion/scan_models_pipeline.py`（我当时写的）

```
=== olmo2  (Olmo2ForCausalLM) ===
  head_dim=128 heads=16/16 layers=16 dtype=bfloat16
  eligible for fused_qk_norm_rope: no
    - q_norm normalises across heads (RMSNorm(self.config.hidden_size)),
      kernel is per-head -- NOT equivalent
```

**结论：OLMo-2 用不了那个 kernel。** 到这一步它应该被排除。

### 第 3 步：Profiling（GPU 1，`--disable-cuda-graph`）

```
=== olmo2 ===
  profiled decode: 516 kernels, 97 eager-norm-signature calls = 6.62% of kernel time
=== olmoe ===
  profiled decode: 356 kernels, 1 eager-norm-signature calls  = 0.13%
=== qwen3 ===
  profiled decode: 384 kernels, 1 eager-norm-signature calls  = 0.32%
```

**结论：OLMo-2 有 6.62% 的缺口，是 Qwen 系的 20 倍。**

### 第 4 步：★ 两个结论撞在一起

这一步是**整个发现的关键**，而且**只有人能做**：

> 扫描说「**用不了**那个 kernel」，profiling 说「**有** 6.62% 的缺口」。
> 两个都对。所以问题变成：**那它到底为什么是 eager 的？**

我当时的原话（session 记录）：

> "OLMo-2 那 6.62% 修不了——它的 norm 语义根本不匹配这个 kernel。
> **「有缺口」和「能用现成 kernel 补」是两件事。**"

然后我去读了源码。

### 第 5 步：读代码，找到真凶

```bash
grep -n "q_norm\|k_norm\|rotary_emb\|class Olmo2Attention" \
     python/sglang/srt/models/olmo2.py
```

定位到 `olmo2.py:165-192`：

```python
if self.alt_stream is not None and get_is_capture_mode():
    q_by_last = self.q_norm(q_by_last)        # ← 走融合 kernel
else:
    q = self.q_norm.forward_native(q)          # ← 显式绕过
```

**三个观察同时成立**，缺一不可：

1. `q` 来自 `qkv.split(...)`，本来就是 2-D `[tokens, q_size]`
2. `q_norm = RMSNorm(hidden_size)`，宽度**正好对上**
3. 所以 `self.q_norm(q)` **直接就能命中融合 kernel**，`forward_native` 是白丢

而 `get_is_capture_mode()` **只在 CUDA graph 捕获期为真** → decode 重放的是融合分支（看着正常），
**prefill 从不被捕获，一直在丢**。

### 第 6 步：验证 + 量化

| 验证 | 命令 / 结果 |
|---|---|
| 数值等价 | `torch.equal` → **bit-identical，max diff 0.0** |
| 生成一致 | `verify_generation_identical.py` → **8/8** |
| 任务指标 | `gsm8k_paired_test.py` → 65.50%→65.25%，**McNemar p=1.000** |
| prefill 直测 | 87.80ms → 70.79ms = **1.24×** |
| 四 regime A/B | `e2e_ab_gemma3.py --reps 7` → prefill-heavy **+17.6% (p<0.001)**，decode 全 n.s. |

**decode 无变化恰恰证明机制判断对了**——decode 本来就在跑融合路径。

## A.3 链 A 的时间线（git commit 时间戳，可复核）

```
07-31 18:07  result: sglang already ships the kernel Gemma-3 needs
07-31 18:27  verify: the fused kernel matches Gemma-3's real rope path
07-31 19:04  result(gemma3): accuracy is a wash, 21 wins to 19 losses
07-31 20:20  result(gemma3): wire up the fused kernel
07-31 20:42  result(gemma3): worth +0.5 to +1.1%, not the 1.39x it first read
07-31 21:26  result(olmo2): bypasses its own fused norm on every path but capture  ← OLMo-2
07-31 21:41  skill(fusion-gap-hunting): v3, with the four traps this run walked into
```

**约 3.5 小时**，含 GPU 上的 profiling 和多轮 A/B。

---

# 链 B：SLO-agent 重新发现（确定性，可完全复核）

## B.1 我实际敲的命令（4 条，按顺序）

```bash
cd /home/t-jialianggu/work/SLO-agent
export PYTHONPATH=$PWD/src

# ① init —— 锁定 GPU UUID、sglang commit、运行时指纹
python -m sglang_agent_kernel_lab.cli init \
  --campaign $PWD/campaigns/kernel_fusion_gap.json \
  --artifact-root /home/t-jialianggu/slo_runs/fusion_gap_001 \
  --sglang-checkout /tmp/sglang_fqr_base \
  --gpu 2 \
  --dataset-name gsm8k --dataset-path /home/t-jialianggu/slo_runs/gsm8k.jsonl \
  --tokenizer /data/hf/models/gemma-3-1b-it \
  --pr-wiki-index $PWD/knowledge/pr_wiki/index.json \
  --pr-wiki-manifest $PWD/knowledge/pr_wiki/manifest.json

# ② scan —— 读源码（新加的阶段），不用 GPU，1.7 秒
python -m sglang_agent_kernel_lab.cli scan \
  --state $ART/state.json \
  --primitive fused_qk_norm_rope \
  --requires 'q_norm|q_layernorm' 'rotary_emb|rope'

# ③ observe —— 递入测量值，过闸门
python -m sglang_agent_kernel_lab.cli observe \
  --state $ART/state.json --input /home/t-jialianggu/slo_runs/obs_olmo2.json

# ④ next —— 状态机决定下一步
python -m sglang_agent_kernel_lab.cli next --state $ART/state.json
```

## B.2 产生的记录（append-only，内容寻址）

`/home/t-jialianggu/slo_runs/fusion_gap_001/records.jsonl`：

```
1. record_type=provenance   id=sha256:aac895f07efc054e8...  2026-08-02T18:40:51.056661Z
2. record_type=scan         id=sha256:0d6a7f41759bc4bcd...  2026-08-02T18:40:52.745383Z
3. record_type=observation  id=sha256:b8930a4576fc53d3b...  2026-08-02T18:40:52.879130Z
4. record_type=opportunity  id=sha256:dbc35b1bbfe848389...  2026-08-02T18:40:52.879518Z
```

**注意时间戳：`scan` 到 `opportunity` 只花了 0.134 秒。** 因为全是确定性代码。

## B.3 ② scan 的输入 → 输出（这是「发现」发生的地方）

**输入**：只有 sglang 代码树路径 + 一个原语名。**没有模型、没有 GPU、没有 prompt。**

**输出**（`scan` 记录的 payload，原文）：

```json
{
  "checkout": "/tmp/sglang_fqr_base",
  "kernel_names_seen": 634,
  "counts": {
    "never_wired_candidates": 35,
    "rank_guarded": 4,
    "path_guarded": 1
  }
}
```

其中那个 **`path_guarded: 1`** 就是 OLMo-2，完整记录：

```json
{
  "kind": "path_guarded",
  "file": "python/sglang/srt/models/olmo2.py",
  "line": 165,
  "class": "Olmo2Attention",
  "backend": "_apply_qk_norm",
  "kernels": [],
  "dispatching_modules": ["self.k_norm", "self.q_norm"],
  "input_guards": [],
  "path_guards": ["self.alt_stream is not None and get_is_capture_mode()"],
  "why_grep_misses_it": true,
  "note": "The fused call is present in this file, so a scan that only asks
           whether the name appears reads this as already wired. The condition
           selects an execution path, not an input, so the fall-through runs on
           whichever phase the condition excludes -- prefill, when it is
           cuda-graph capture."
}
```

**这一条就是链 A 花 3.5 小时找到的东西，代码在 1.7 秒内定位到了同一行（`olmo2.py:165`）。**

注意 `"kernels": []` 但 `"dispatching_modules": ["self.q_norm"]`——
这正是**规则的关键**：融合侧是模块调用 `self.q_norm(x)`，**不含任何 kernel 名字**。
只按 kernel 名字匹配会漏掉它（我第一版就漏了）。

## B.4 ③ observe 的输入（我递进去的）

`/home/t-jialianggu/slo_runs/obs_olmo2.json`：

```json
{
 "campaign_id": "kernel_fusion_gap",
 "model": "OLMo-2-0425-1B-Instruct",
 "scan_sha256": "0d6a7f41759bc4bcde290581dd612305afdeb4b1942237d2d6905bb8a48d36f1",
 "regime": {"serving": "prefill"},
 "measurements": {
   "semantic_equivalence": true,
   "eager_calls_per_layer": 6.06,
   "gap_pct_of_kernel_time": 6.62,
   "gap_stage": "prefill"
 },
 "artifacts": [{"kind": "operator_audit",
                "path": ".../results/fx_fusion/model_pipeline_profiled.json",
                "sha256": "..."}],
 "wiki_query_terms": ["sglang-32670", "sglang-32383", "rmsnorm", "olmo"]
}
```

> ⚠️ **这里必须说清楚**：`6.62` 和 `6.06` 这两个数字**来自链 A 的 profiling**，
> 不是 loop 测的。SLO-agent **按设计就不跑 benchmark**
> （README 原文："does not launch SGLang benchmarks automatically"）。

## B.5 ③ observe 的输出（闸门逐条判定）

`opportunity` 记录里的原始内容：

```
  PASS  primary_model            model in primary_models                   observed=OLMo-2-0425-1B-Instruct
  PASS  serving_regime           regime.serving == campaign.regime.serving observed=prefill
  PASS  semantic_equivalence     semantic_equivalence is true              observed=True
  PASS  eager_calls_per_layer    eager_calls_per_layer >= 1.0              observed=6.06
  PASS  gap_pct_of_kernel_time   gap_pct_of_kernel_time >= 3               observed=6.62
  PASS  gap_stage                gap_stage in [prefill, decode, both]      observed=prefill

  eligible: True
  wiki 选中: ["sglang-32670"]
  证据锚定: workload_sha256 = 0d6a7f41759bc4bcde290581dd612305...  ← 就是 scan 的 sha
```

最后一行值得注意：因为 fusion gap 跟负载无关，**observation 用 scan 的内容哈希代替
workload 哈希**来锚定证据——依然是钉死在一份确定的东西上，只是不再假装它依赖某个 prompt。

## B.6 两个阴性对照（同样是真跑的）

**对照 1 — Qwen3-0.6B**（缺口只有 0.32%）：

```
 eligible: False   failed: [primary_model, eager_calls_per_layer, gap_pct_of_kernel_time]
   FAIL  eager_calls_per_layer    observed=0.036
   FAIL  gap_pct_of_kernel_time   observed=0.32
```

**对照 2 — OLMo-2 硬要用 per-head kernel**（★ 最有价值的一条）：

```
 eligible: False   kill_reasons: [semantic_equivalence]
   PASS  primary_model            observed=OLMo-2-0425-1B-Instruct
   PASS  serving_regime           observed=prefill
   FAIL  semantic_equivalence     observed=False
   PASS  eager_calls_per_layer    observed=6.06     ← 性能闸门
   PASS  gap_pct_of_kernel_time   observed=6.62     ← 全绿
   PASS  gap_stage                observed=prefill
```

**同样 6.62% 的缺口、所有性能指标达标，因为数学不等价被一票否决。**
这就是防「静默算错」的机制——OLMo-2 是跨 head 归一化，硬套 per-head kernel
不会报错、不会跑挂，**只会算错**。

## B.7 全部 6 个历史案例的回测

```bash
python scripts/back_test_known_cases.py --checkout /tmp/sglang_fqr_base
```

```
 #  model     shape                     result        detail
 1  LFM2.5    never_wired               REDISCOVERED  lfm2_moe.py in the candidate list
 2  LFM2.5    never_wired               REDISCOVERED  lfm2_moe.py in the candidate list
 3  Gemma-3   never_wired_at_the_time   N/A           upstream #32383 closed this shape
 4  Gemma-3   rank_guarded              REDISCOVERED  Gemma3RMSNorm on `x.dim() == 2`
 5  Gemma-3   never_wired               REDISCOVERED  gemma3_causal.py in the candidate list
 6  OLMo-2    path_guarded              REDISCOVERED  Olmo2Attention on `get_is_capture_mode`

5/5 rediscovered (1 not applicable to current main)
```

原始数据：`SLO-agent/docs/back_test_result.json`

---

## 2. 诚实的边界（这些必须一起说）

### ❌ loop **没有**做的事

1. **没有测任何性能数字。** 6.62%、1.24× 全部来自链 A。SLO-agent 按设计不跑 benchmark。
2. **没有跑到 `evaluate`。** 停在 `phase: act`。`propose`/`evaluate` 我后来单独试过，
   但那也只是把链 A 的 A/B 结果递进去让它审。
3. **没有 LLM 参与任何判断。** 全是阈值比较。

### ⚠️ 「5/5 重现」不等于「loop 会先于人发现」

**扫描器的规则是我从已经理解的案例里反推写出来的。**

比如 `path_guarded` 那条规则（「模块调用 + 同模块的 forward_native 在互斥分支上」），
是**先有** OLMo-2 这个案例，我才知道要这么写。

所以诚实的说法是：**这个方法现在可复用了，不是说它有预测能力。**

### ⚠️ 精度差异很大，两个扫描不是一回事

| 扫描 | 输出 | 精度 |
|---|---|---|
| `never_wired` | 35 个候选 | **只有 3 个已知为真**，高召回低精度 |
| `rank_guarded` + `path_guarded` | 5 个 | 全部结构上为真 |

`qwen3.py` 就是典型误报：它不提 `fused_qk_norm_rope`，但调了共享 helper
`apply_qk_norm`，那个 helper 会先试融合 kernel。**「提没提这个名字」不是判据。**

---

## 3. 一句话回答你的问题

> **「在 SLO-agent 里完整复现了找到这几个机会」——准确说法是：**
>
> loop 重新**发现**了这些缺口的位置（5/5，1.7 秒，确定性可复核），
> 但**没有复现**这些优化（没测性能、没做验证）。
>
> 而且它靠的**不是 LLM 推理**，是我把人工发现的模式**固化成了确定性规则**。
> 规则来自已知案例，所以这证明的是「方法可复用」，不是「它能预测」。

---

## 4. 所有证据文件

| 证据 | 路径 |
|---|---|
| loop 运行记录（4 条） | `/home/t-jialianggu/slo_runs/fusion_gap_001/records.jsonl` |
| loop 状态 | `/home/t-jialianggu/slo_runs/fusion_gap_001/state.json` |
| observe 输入 | `/home/t-jialianggu/slo_runs/obs_olmo2.json` |
| 阴性对照输入 | `/home/t-jialianggu/slo_runs/obs_qwen3.json`、`obs_olmo_wrong.json` |
| 回测原始数据 | `SLO-agent/docs/back_test_result.json` |
| 扫描器实现 | `SLO-agent/src/sglang_agent_kernel_lab/fusion_scan.py` |
| 链 A 的 profiling | `EndtoEnd/results/fx_fusion/model_pipeline_profiled.json` |
| 链 A 的 A/B | `EndtoEnd/results/fx_fusion/e2e_ab_olmo2.json` |
| 链 A 的精度闸门 | `EndtoEnd/results/fx_fusion/gsm8k_paired_olmo2.json` |
| 链 A 的完整记录 | `EndtoEnd/docs/2026-07-31/gemma3_fused_qknorm_rope_full_record.md` |
| 6 案例全集 | `EndtoEnd/docs/kernel_fusion_catalogue.md` |
| 上游 issue | [sglang#33415](https://github.com/sgl-project/sglang/issues/33415) |
| 上游草稿 PR | [sglang#33416](https://github.com/sgl-project/sglang/pull/33416) |

**全部可复核**：`records.jsonl` 里每条记录的 `record_id` 就是 payload 的 sha256，
改一个字符哈希就对不上。
