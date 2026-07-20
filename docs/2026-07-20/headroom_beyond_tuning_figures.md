# "Tuning 以外还有多少空间" — Qwen3-30B-A3B 图表说明（回答 Dey）

**日期**：2026-07-20 · 全部数字为本项目实测（decode，Qwen3-30B-A3B，H200，bf16）。

## Fig 1 — decode step 组成（`fig1_decode_composition.png`，v33 实测）
一步 decode 的 GPU kernel 时间：**MoE 41% + dense_gemm(qkv/o/lm_head) 32% + attention 16% = 89%**，其余 norm/misc/act/sample ≈ 11%。
→ **decode 前三大块全是 memory-bound 的权重/KV 读取**（b1 下 lm_head 单 token 就读 vocab×hidden≈600MB）。这解释了为何"抠单个 kernel 算力"端到端杠杆小——整步本质是流式读权重。

## Fig 2 — MoE 达到的 HBM 带宽 vs batch（`fig2_moe_bandwidth_vs_batch.png`，v27 实测）
sglang fused_moe：b≥32 达 **74–84% HBM**（近内存屋顶，无损 kernel 空间 <1.3×）；b4096 掉到 **29%**（转 compute-bound，prefill 区，config-tuning 已 +50%）。
→ **decode = memory-bound（config/kernel 都难再压）；prefill = compute-bound（另一套故事）**。所以 headroom 图必须按 regime 分开。

## Fig 3 — ★headroom BEYOND tuning（`fig3_headroom_beyond_tuning.png`，核心图，回答 Dey）
以 **best-tuned config 为 baseline（=1.0）**，展示 config 调参以外的手段能再拿多少（decode，exact 方法）：
| | 单请求 c=1 | 并发 c=32 |
|---|---|---|
| best-tuned config | 1.00× | 1.00× |
| + kernel 重写（实测 e2e） | **+1.5%** | —（未测）|
| + spec decoding（实测 e2e，exact） | **+6.6%** | **+30.6%** |
| roofline 天花板（理论上界，exact） | 1.85× | 1.85× |

**这张图一句话**：config tuning 到平台期后，decode 理论上还有 ~1.85× 的空间（全在 memory 侧，config 够不到）；**spec decoding 这类 exact 方法已实测兑现 +6.6%(c1)/+30.6%(c32)**，是目前最大的可实现杠杆；**而纯 kernel 重写实测只兑现 +1.5%**（因为 MoE 只占 41%，且 sglang kernel 已近内存屋顶）。

## 结论（给 Dey / 团队）
1. **"tuning 以外确实有空间"**：decode 理论天花板 ~1.85×，config 碰不到。
2. **但"怎么拿"很关键**：kernel 重写杠杆小（+1.5%）；**spec decoding 是最大可实现杠杆（c32 +30.6%）**，因为它一次验证多 token，同时摊薄 MoE+dense+lm_head+attention 全部 89% 的 memory-bound 读取。
3. **prefill 是另一条线**（compute-bound，config 已 +50%），需单独画。
