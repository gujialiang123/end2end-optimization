# 自由探索 vs Agent loop：同一个问题，两种模式的实测对比

**日期**：2026-08-02 · **硬件**：H200（GPU 0–3） · **基线**：sglang `89f4a80c1f`

用户的问题是：**自由模式和 loop 模式会不会有显著区别？**
这份文档是实测答案，不是推测——两种模式在同一个代码库上跑，产出逐条对照。

---

## 0. 结论先说

| | 自由探索（旧仓库） | Agent loop（SLO-agent） |
|---|---|---|
| **能识别的缺口形态** | 3 种 | **4 种**（多一个 residual convention） |
| **已知案例重现** | — | **5/5**（第 6 个已被上游修掉，正确报 N/A） |
| **今晚新发现** | 0 | **2 个未见过的候选**（Exaone4、JetNemotron） |
| **假阳性拦截** | 靠人记得去查 | **自动**：4 SCALED + 1 nn.LayerNorm + 1 dispatch chain |
| **耗时** | 两周 | **秒级**（无 GPU） |
| **能否判断值不值得** | 能（人去 profile） | **不能**——必须外接 profiling |

**最重要的一条**：两种模式**不是替代关系**。
loop 把「找」从两周压到秒级，但它**产出的是候选不是结论**——
今晚新找到的 Exaone4，结构缺口真实存在，profiling 一测只占 **0.45%**，被闸门正确否决。

---

## 1. 实验设计

同一份 sglang checkout，两种模式各跑一遍，比较产出。

- **自由模式**：直接 grep / 读代码 / 挑模型 profile，就是过去两周的做法
- **loop 模式**：`scan` → `observe` → 闸门判定，规则事先写死，不许中途改

---

## 2. 自由模式今晚做了什么

按老办法找了三个方向：

```bash
# 方向1：layers/ 里显式调 forward_native
grep -rn "\.forward_native(" python/sglang/srt/layers/**/*.py
#   -> layernorm.py:452，查证是 hidden_size 不受支持的能力回退，不是缺口

# 方向2：被 env/flag 关掉的融合
grep -rn "envs\.\|get_server_args()\." python/sglang/srt/layers/layernorm.py
#   -> SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM，NPU 专用，与 CUDA 无关

# 方向3：MoE 里的同类形态
grep -rln "forward_native" python/sglang/srt/layers/moe/
#   -> fused_moe_native.py，是刻意保留的 native 参考实现
```

**产出：0 个新缺口。** 三条线索全是合理设计。

但自由模式确认了一件 loop 做不到的事——`get_is_capture_mode()` 这个 idiom 在
qwen3_next / qwen2_moe / grok / nemotron_h / kimi_linear 里都出现，
**只有 OLMo-2 的 else 分支调了不同的、更慢的实现**。其余几个两条分支算的是同一件事，
只是流并行方式不同（`qwen3_next.py:394` 两边都调 `in_proj_qkvz` + `in_proj_ba`）。

> 这是自由模式的价值：**它能判断一个形态是不是设计意图**。

---

## 3. loop 模式产出了什么

### 3.1 已知案例重现：5/5

```
 #  model     shape                     result        detail
 1  LFM2.5    never_wired               REDISCOVERED  lfm2_moe.py 在候选列表里
 2  LFM2.5    never_wired               REDISCOVERED  lfm2_moe.py 在候选列表里
 3  Gemma-3   never_wired_at_the_time   N/A           上游 #32383 已修掉这个形态
 4  Gemma-3   rank_guarded              REDISCOVERED  Gemma3RMSNorm on `x.dim() == 2`
 5  Gemma-3   never_wired               REDISCOVERED  gemma3_causal.py 在候选列表里
 6  OLMo-2    path_guarded              REDISCOVERED  Olmo2Attention on `get_is_capture_mode`
```

包括最难的 OLMo-2（grep 看不见、decode profile 干净）。

### 3.2 新增了一种自由模式没系统扫过的形态

**`fused_add_rmsnorm` 从来不被任何模型文件写出名字。**
它是通过「调 norm 时传两个参数」触发的：`self.input_layernorm(hidden_states, residual)`。

按名字扫它返回 **137 个候选**——完全没用。

loop 改成扫**调用约定**：一个 forward 收了 `residual` 参数、把它覆盖掉、
自己手写 `+ residual`、且 norm 只传一个参数。逐步收紧：

```
第一版（只看 + residual）        : 48 个  ← 没用，含 vision encoder
要求「收了 residual 参数」      :  9 个  ← 只留真正在协议内的
排除 nn.LayerNorm                :  8 个
拆出 SCALED                      :  4 plain + 4 scaled
```

**4 个 plain-add：`Lfm2DecoderLayer`、`Lfm2MoeDecoderLayer`（已知）、
`Exaone4DecoderLayer`、`JetNemotronDecoderLayer`（全新）。**

### 3.3 自动拦截的假阳性（loop 最值钱的部分）

| 候选 | 表象 | 真相 | 谁拦的 |
|---|---|---|---|
| **PhiMoE** | 形态跟 LFM2.5 一模一样 | 用 `nn.LayerNorm`（PyTorch 原生），**没有 dispatch 可言** | norm 构造函数检查 |
| **MiniCPM ×2** | 手写 residual add | 实际是 `residual + hidden * (scale/depth)`，**带缩放**，融合 kernel 表达不了 | SCALED 标记 → 等价闸门 |
| **GraniteMoeHybrid ×2** | 同上 | 同上 | 同上 |
| **Ernie4_5_VLRotary** | kernel 在形状守卫后面 | 另一条分支调了**别的** kernel，不是缺口 | early-return 传播 |
| **qwen3.py** | 从不提 `fused_qk_norm_rope` | 走共享 helper `apply_qk_norm`，helper 内部先试融合 | profiling（0.32%）|

**这五类全是「不报错、不跑挂、只算错或白忙」的失败。**
自由模式靠人记得去查；loop 把它们写成了规则。

---

## 4. 关键实测：新发现值不值得？

**Exaone4** 是 loop 新找到的、结构上真实的缺口。GPU 3 上 profile：

```
exaone4 (Exaone4ForCausalLM), 30 层, head_dim=64, bf16
  profiled decode: 530 kernels, 1 eager-norm 调用 = 0.45% of kernel time
```

**0.45%——不值得做。** loop 自己也这么判：

```
eligible: False
  PASS  semantic_equivalence     True
  FAIL  eager_calls_per_layer    0.033  (需 >= 1.0)
  FAIL  gap_pct_of_kernel_time   0.45   (需 >= 3)
```

对照 OLMo-2：

```
eligible: True
  PASS  eager_calls_per_layer    6.06
  PASS  gap_pct_of_kernel_time   6.62
```

> **这是整个实验最重要的一条**：
> loop 能把「找」加速三个数量级，但**判断值不值得仍然要 profiling**。
> 扫描器给候选，闸门给判决，两者缺一不可。

---

## 5. 显著区别在哪

### 5.1 loop 强在：规模、一致性、不遗忘

- **规模**：秒级扫完 212 个模型文件、227 个 `forward*` 方法，不需要 GPU
- **一致性**：同样的规则每次都跑，不会因为今天累了就跳过某个检查
- **不遗忘**：`semantic_equivalence` 这条是踩了 OLMo-2 和 EXAONE-4 两次才写下来的，
  写进闸门后**永远不会再漏**

最能说明问题的一条：`residual convention` 这个形态，
**自由模式两周里从没想过要系统扫它**——LFM2.5 那次是偶然读到的。
loop 一旦把它写成规则，立刻多出 2 个候选。

### 5.2 自由模式强在：判断力、发现新形态

- **判断意图**：`get_is_capture_mode()` 在 5 个模型里出现，只有 OLMo-2 是缺口——
  这个区分**需要读懂两条分支各自在算什么**
- **发现形态本身**：loop 的规则全是从已知案例反推的。
  rank_guarded 来自 Gemma-3，path_guarded 来自 OLMo-2，residual convention 来自 LFM2.5。
  **没有那两周就没有这些规则。**

### 5.3 一句话

> **自由模式发现「什么形态算缺口」，loop 负责「把这个形态扫遍全库且永不遗忘」。**

今晚的数据支持这个分工：
loop 重现 5/5 + 新增 2 个候选，但已验证的那个（Exaone4）只值 0.45%；
自由模式今晚 0 个新缺口，却正确否掉了 3 条假线索。

---

## 6. 一个必须承认的局限

**back-test 的 5/5 不能说明 loop「本来能先找到」。**

扫描器的规则是从已经理解的案例里反推出来的——
`path_guarded` 这条规则的存在，就是因为我们先花时间搞懂了 OLMo-2。
诚实的说法是：**这套方法现在可复用了，不是它有预测能力。**

真正的检验是：**在一个我们从没看过的框架/模型上，它能不能找出人没找到的东西。**
今晚的 Exaone4 和 JetNemotron 是这个方向的第一步——
Exaone4 已被证明不值得（0.45%），JetNemotron 还没测。

---

## 7. 证据文件

| 内容 | 位置 |
|---|---|
| back-test 脚本与结果 | `SLO-agent/scripts/back_test_known_cases.py`、`SLO-agent/docs/back_test_result.json` |
| loop 实现（4 种扫描） | `SLO-agent/src/sglang_agent_kernel_lab/fusion_scan.py` |
| campaign 定义 | `SLO-agent/campaigns/kernel_fusion_gap.json` |
| loop 测试 | `SLO-agent/tests/test_fusion_scan.py`（19 个） |
| loop 侧完整记录 | `SLO-agent/docs/kernel_fusion_gap_backtest.md` |
| Exaone4 profiling | `results/fx_fusion/exaone4_profile.json` |
| 全部 fusion 案例 | `docs/kernel_fusion_catalogue.md` |
| 方法论 | `.github/skills/fusion-gap-hunting/SKILL.md` (v3) |
