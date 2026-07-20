# 对话上下文快照 — 2026-07-20 晚（kernel tuning e2e gap + 迁移到新机器）

> 本文件为「当前对话」的完整上下文存档，供在另一台机器 clone 仓库后**接着做实验**。
> 本机磁盘已满，停止本机实验。以下记录：已完成的事、当前未决的关键实验、以及下一步该怎么跑。

---

## 0. 一句话现状
Qwen3-30B-A3B 优化研究已得出诚实结论：**成熟 bf16/H200 MoE 上，kernel 融合/重写端到端 ≈0；真实"tuning 以外"的 e2e 杠杆是架构（线性注意力）+ spec decoding。** 
**但留下一个必须补的 gap（用户在对话末尾指出）：§1.6 的 kernel-config tuning 只有隔离 kernel 时间（µs），从未做端到端 A/B 验证。** 这是迁移到新机器后要做的第一件事。

---

## 1. 本次对话（2026-07-20 晚）具体做了什么

### 1.1 完成了三项 autopilot 任务（更早，已 push）
- **① 全 regime 重测 kernel 改动**：custom MoE kernel b1 +1.4%、b2 −2%、b4 −11%；agent server c1 −0.7%、c32 −7% → 全 regime e2e ≈0。gate 融合全 batch ~1.0×。
- **② 新架构 e2e（唯一正结果）**：线性注意力 LFM2.5 vs 全注意力 Qwen3，decode 随上下文 scaling **+24% vs +57%**（bs=32, 512→8192），Qwen bs=32×16k **OOM**。图 `results/2026-07-20_v39_ctxscan/ctx_scaling.png`。
- **③ 最终矩阵 + 诚实结论**。

### 1.2 噪声验证（Chendi 要求）— v41
- 问题：custom MoE b1 "+1.4%" 是真信号还是波动？
- 方法：**n=15 交错重复独立启动 + Welch t 检验**（cudagraph ON，与原始条件一致）。
- 结果：**b1 +1.17%，|t|=6.51 → 真信号（非波动）**，但极小（~0.05ms）；b2 −4.3%（|t|=3.2）、b4 −11.7%（|t|=9.9）**真回归**。
- 产物：`docs/2026-07-20/noise_verification_custom_moe_b1.md`、`scripts/run_v41_noise_verify.py`、`results/2026-07-20_v41_noise/`。

### 1.3 报告补充 — `docs/2026-07-20/qwen_optimization_full_report.md`
- **§1.5**：custom MoE kernel 具体改了什么（去 `moe_align_block_size` + 融合 w1+SwiGLU / w2+加权求和 + fp32 累加 + M=1 也用 tensor-core dot）。机制：b1 赢因削开销、b≥2 输因放弃 expert 权重复用。
- **§1.6**：kernel-config tuning（Triton `fused_moe_kernel` meta 参数 BLOCK_M/N/K、GROUP_M、num_warps、num_stages）。per-regime kernel 时间：**decode +13%（b1）、prefill +35~54%（Qwen）/+47~67%（DeepSeek）**，U 形。
- 4 张图已内嵌（fig1-3 + ctx_scaling）。

### 1.4 b=32 强制 custom 实测（回答"为啥 b32 更慢"）
- 实测：baseline b32 ~7.66ms vs 强制 custom ~22.63ms → **custom 慢 2.95×**，rel err 恶化到 7.2%。**注意：这个数据点还没写进任何文档**（下一步可补进 §1.5 或噪声文档作为机制证据）。

---

## 2. ★未决的关键实验（新机器上第一件事）

### 2.1 kernel-config tuning 的端到端验证（用户明确要求，尚未做）
**背景**：报告 §1.6 全是隔离 kernel 时间（µs），**没有 e2e**。用户问"kernel level tuning 之后你有做 end2end 的性能验证吗？" → 答案是**没有**。

**必须搞清的关键点（已查明）**：
- 我们 shape `E=128,N=768,H200` 在 triton **3.5.1 无 config** → sglang **回退加载** `triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json`（fallback），并打印 "Performance might be sub-optimal!"。
- **§1.6 那个 +35~54% 大数是 "ours vs default 启发式"**；但 sglang 实际根本不用 default，它用 fallback。
- **我们重 tune 的 vs fallback 只 +0.6% kernel（b=32）** → 所以真正该测的 e2e 是 **"ours tuned config" vs "sglang 实际在跑的 fallback"**，预期 e2e 提升很小。

**我们的 tuned config 产物**：`results/autotune_qwen3_moe/E=128,N=768,device_name=NVIDIA_H200.json`（仅 batch=32 那次；注意 §1.6 的 default 对比数据来自 `results/2026-07-19_v23_config_evidence/`）。

**建议的 e2e A/B 方案（在新机器上跑）**：
1. **baseline（sglang 现状）**：直接起服务/bench_one_batch，它会自动加载 fallback config。
2. **ours**：把 `results/autotune_qwen3_moe/E=128,N=768,device_name=NVIDIA_H200.json` 拷进 `sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/` 让 sglang 优先加载我们的（需确认加载优先级：先查当前 triton 版本目录，再回退旧版本）。
3. **对比**：prefill throughput（in=256/512/1024/2048/4096）+ decode TPOT（b=1/8/32），各 **≥3 次重复 + 中位数/t 检验**（Chendi 标准）。
4. **诚实预期**：因为 ours-vs-fallback kernel 只 +0.6%，e2e 大概率 <1%；但 prefill 大 batch 可能因 default→tuned 的差在某些未覆盖点更明显——**需实测确认，别再只报 kernel µs**。
5. 更彻底：对每个 batch bucket 都重新 tune（`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`），生成 per-batch config，再测 e2e。

**config 加载机制排查（新机器上先做）**：
```
grep -nE "get_moe_configs|get_config_file_name|config_file_path|triton.__version__" \
  sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
```
确认 sglang 如何按 triton 版本查找 config、是否有 env 覆盖、放进哪个目录能被优先加载。

### 2.2 其他候选（用户之前提到、可选）
- **FP8 backend/dispatch 扫描**（用户已放开量化）：本地有 `Qwen3-30B-A3B-Instruct-2507-FP8`；sglang 有 `--moe-runner-backend {triton,cutlass,flashinfer_trtllm,flashinfer_cutlass,...}`。公平比较（同 FP8 模型同 workload 只换 backend）+ GSM8K accuracy 闸门，看能否打败 cookbook 默认 backend。对应已合并 PR #27401/#22664 的 dispatch 手法。
- **重复计算/CPU-profile 审计**（#28744 类，硬件无关）：长 prompt 起 Qwen3 服务抓 CPU profile 找重复 tokenization/preprocessing。

---

## 3. 环境与复现要点（迁移必读）

- **conda 环境**：`/home/t-jialianggu/.conda/envs/sglang-dev/bin/python`（triton 3.5.1）。新机器需重建等价环境。
- **必需环境变量**：
  ```
  export CUDA_HOME=$ENVDIR HF_HOME=$PWD/.hf_cache PATH=$ENVDIR/bin:$PATH
  export HF_HUB_CACHE=$PWD/.hf_cache/hub HF_DATASETS_CACHE=$PWD/.hf_cache/datasets  # bench_serving 需要可写缓存
  ```
- **sglang**：editable install @ commit `17f7a1da1`。新机器需 clone 同版本 sglang 并 `pip install -e`。
- **模型路径**（新机器需重新获取）：
  - `/data/hf/models/Qwen3-30B-A3B-Instruct-2507`（主模型，无 shared expert）
  - `/data/hf/models/Qwen3-30B-A3B-Instruct-2507-FP8`（FP8）
  - `/data/hf/LFM2.5-8B-A1B`（混合线性注意力）
  - `/home/t-jialianggu/models/DeepSeek-V2-Lite`、`Qwen1.5-MoE-A2.7B-Chat`
- **custom MoE patch 用法**：`CUSTOM_MOE=1 CUSTOM_MOE_MAX_M=4`；server 注入用 `scripts/_siteinject/sitecustomize.py`（`PYTHONPATH=$PWD/scripts/_siteinject CUSTOM_MOE_INJECT=1`）。
- **bench_one_batch 要点**：tp=1 in-process（monkeypatch 可读 stats）；不支持 spec decoding；单次噪声大需 ≥3 次；第一个 median 是 warmup 重复值要丢。
- **测量纪律（Chendi 固化）**：信号 vs 噪声必须多次重复 + t 检验；隔离 kernel 时间 ≠ e2e，必须做端到端 A/B。
- **git 注意**：`.hf_cache/`、`logs/`、`*.ncu-rep/*.nsys-rep/*.sqlite/*.trace.json.gz` 已 gitignore（大文件，会触发 GitHub 100MB 限制，需在新机器重新生成）。

---

## 4. 关键文档索引（都在仓库里）
- **主报告**：`docs/2026-07-20/qwen_optimization_full_report.md`（§1.5 kernel 重写、§1.6 kernel-config 调优、4 图）
- 噪声验证：`docs/2026-07-20/noise_verification_custom_moe_b1.md`
- 全 regime 扫描 + 矩阵：`docs/2026-07-20/regime_sweep_kernel_changes.md`
- 新架构线性注意力：`docs/2026-07-20/new_architecture_linear_attention_e2e.md`
- config-tuning kernel 证据：`docs/2026-07-19/pr_validation_report.md`
- 项目状态：`plan.md`
- 关键脚本：`scripts/custom_moe_patch.py`、`run_v41_noise_verify.py`、`run_v23_config_evidence.py`、`serve_with_patch.py`、`_siteinject/sitecustomize.py`

---

## 5. 下一步 TODO（新机器，按优先级）
1. **[最高] kernel-config tuning 的 e2e A/B**（§2.1）：ours tuned config vs sglang fallback，测 prefill throughput + decode TPOT，≥3 repeat + t 检验。**补上报告缺的 e2e 层证据。**
2. 把 b=32 强制 custom 慢 2.95× 的实测数据点补进文档（机制证据）。
3.（可选）FP8 backend 扫描 + accuracy 闸门（§2.2）。
4.（可选）重复计算 CPU-profile 审计（§2.2）。
