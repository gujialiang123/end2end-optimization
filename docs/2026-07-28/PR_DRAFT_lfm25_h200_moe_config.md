# PR 草稿 — LFM2.5 的 H200 MoE tuned config（补 #22791 漏掉的那块）

**目标仓库**：`sgl-project/sglang` · **状态**：候选，尚未开 PR
**要提交的文件**：`python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=32,N=1792,device_name=NVIDIA_H200.json`
**本地副本**：`configs/regime_kernel/pr_candidate/`
**原始数据**：`results/regime_kernel/processed/pr_candidate_e2e.csv`、`pr_candidate_buckets.csv`、`raw/pr_fill/`

---

## 一句话

上游 **#22791**（`[MoE] Add LFM2 MoE tuning support + tuned configs for H100/B200/MI325X`，2026-04-22）已经为 LFM2 的 MoE shape 做了这件事，覆盖 H100、B200、MI325X ——**唯独没有 H200**。补上它，长 prefill **+23.3%**，decode 中性。

---

## 为什么这个空缺是真的

`get_moe_configs` 的查找 key 是 `E={专家数},N={中间维度},device_name={GPU}`，**`device_name` 是文件名的一部分**。版本回退只换 `triton_*` 目录，**文件名不变** —— 所以 H200 永远拿不到 H100 的 config。实测：

```
triton: 3.5.1 | device_name: NVIDIA_H200
LFM2.5     E=32,N=1792,device_name=NVIDIA_H200.json
           -> *** NOT FOUND in any triton_* dir ***

E=32,N=1792 present for these devices:
  triton_3_5_1/E=32,N=1792,device_name=NVIDIA_B200.json
  triton_3_5_1/E=32,N=1792,device_name=NVIDIA_H100_80GB_HBM3.json
  triton_3_6_0/E=32,N=1792,device_name=AMD_Instinct_MI325X.json
```

找不到就走 `get_default_config` 的两档启发式（`fused_moe_triton_config.py:246-260`）：

```python
config = {BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8}
if M <= E or (is_marlin and M <= 32):
    config = {BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 32, BLOCK_SIZE_K: 64, GROUP_SIZE_M: 1}
```

**整个 M 范围两档**，`num_warps`/`num_stages` 交给 Triton 默认。

---

## 这个 config 长什么样：decode 逐字段等于默认，只有 prefill 被特化

沿用 2026-07-26 研究里的 **guarded** 策略：**只在 oracle 证明有 ≥1.15× 空间的桶特化，其余原样写入默认启发式的值**。结果是一条非常干净的分界：

| M | 是否等于默认启发式 | 本 PR 新扫的 kernel 加速 |
|---:|---|---:|
| 1, 2, 4, 8, 16, **24**, 32 | ✅ **全部等于默认** | 24: 1.078×（< 1.15，故保持默认）|
| **48** | 特化 | **1.372×** |
| 64 | 特化 | （2026-07-26）|
| **96** | 特化 | **1.391×** |
| 128, 256, 512, 1024 | 特化 | （2026-07-26）|
| **1536** | 特化 | **1.562×** |
| 2048 | 特化 | （2026-07-26）|
| **3072** | 特化 | **1.626×** |
| 4096, 8192 | 特化 | （2026-07-26）|

**M ≤ 32 全部与默认逐字段相同** ——CUDA graph 捕获的 decode batch size 落在这一段，所以 decode 路径行为不变，收益全部来自 prefill。这不是事后解释，是 guarded 策略的直接产物。

桶集合与 #22791 的 H100/B200 文件**完全对齐**（19 个桶）。原研究只有 14 个，缺 `24, 48, 96, 1536, 3072` —— 这五个是**为这个 PR 新扫的**（每个 468–894 个候选，`warmup=25, iters=100, repeats=5`）。

---

## 端到端

LFM2.5-8B-A1B · 1×H200 · BF16 · sglang 0.5.12.post1 · Triton 3.5.1
只有 `SGLANG_MOE_CONFIG_DIR` 在变，serving flag 全部冻结。

| regime | default（无 config 文件） | 本 PR 的 config | 变化 | p |
|---|---:|---:|---:|---|
| **C 长 prefill** (in=4000, out=32, conc=4) | 12.277 req/s | **15.142** | **+23.34%** | 1.3e-10 |
| A 低批 decode (in=100, out=256, conc=1) | 1.6847 | 1.6825 | −0.13% | 0.079 **中性** |

长 prefill **8/8 次分布完全不重叠**（default 最高 12.34 < candidate 最低 14.75）。

### decode 那一栏做了顺序对照，这点值得单独说

第一次测 decode 得到 **−0.37%，p=4.9e-04** —— 统计显著的小回归。但 `rk_e2e.py` 是**顺序执行** arm（先 8 次 default，再 8 次 candidate），所以我把顺序反过来重跑：

| 顺序 | ratio |
|---|---:|
| default 先 | 0.9963（**−0.37%**）|
| candidate 先 | 1.0012（**+0.12%**）|

**符号翻转，而且两次都是"先跑的那个更快"** —— 这是位置效应，不是 config 效应。把两个顺序合并（counterbalanced，每臂 n=16）后：**−0.13%，p=0.079，不显著**。

顺带排除了另一个假设：新加的 96 桶确实改变了 M=100 的归属（原来落到 128 桶），但实测 96 桶的配置在 M=100 上**更快**（1.399× vs 1.379×），所以它不可能是回归来源。

---

## 诚实边界

1. **baseline 很弱。** 对手是两档启发式，不是认真调过的配置。正确表述是"**这个 model/GPU 组合从来没人调过**"，不是"我们把 kernel 优化快了 1.6×"。上游自己在日志里也写 "Performance might be sub-optimal!"。

2. **Triton 版本。** 我们在 **Triton 3.5.1 / torch 2.9.1** 上 tune 和验证；上游 main 现在 pin `torch==2.11.0`。文件放进 `triton_3_5_1/`（我们实测的版本）。Triton 3.6 的用户会通过既有的跨版本 fallback 拿到它 —— 仍然远好于两档启发式，但会打印 "Performance might be sub-optimal!"。**理想情况应在 3.6 上重扫一遍**；这一点必须在 PR 正文里主动说明，不能装作没有。

3. **只测了一张卡、一个模型。** H200 单卡 TP1。

---

## 提交前 checklist

- [x] 确认 H200 文件在所有 `triton_*` 目录下都不存在
- [x] 确认 `device_name` 精确匹配，不会 fallback 到 H100
- [x] 桶集合与 #22791 的 H100/B200 对齐（19 个）
- [x] 小 M 桶与 `get_default_config` 逐字段比对
- [x] 补扫 5 个缺失桶
- [x] 端到端 A/B（长 prefill +23.34%，8/8 不重叠）
- [x] decode 顺序对照，排除位置效应
- [ ] 在 Triton 3.6 上重扫（可选，但会让 PR 更强）
- [ ] 用上游官方 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` 复现一遍，便于 reviewer 自证
