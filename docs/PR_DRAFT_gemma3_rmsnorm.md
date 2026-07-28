# PR 草稿 — `fix(gemma3): dispatch Gemma3RMSNorm to the fused CUDA kernel`

**目标仓库**：`sgl-project/sglang` · **分支**：`fix/gemma3-rmsnorm-cuda-fused`
**本地分支位置**：`/tmp/sglang_pr_main`（基于 `origin/main` @ `a82ead53b`）
**补丁文件**：`results/lfm_fusion/pr_gemma3/0001-fix-gemma3-rmsnorm-cuda.patch`
**状态**：**草稿，未提交。** 需要用户确认后才发。

> ⚠️ 提 PR 是公开行为且会署名，我不会未经许可提交。这份文档是给你审的。

---

## 一句话

`Gemma3RMSNorm.forward_cuda()` 是 `return self.forward_native(x)`，导致 Gemma-3 的每一次归一化在 CUDA 上跑 **eager PyTorch**（~6 个 kernel），而**同文件里 100 行之上的 `GemmaRMSNorm` 早就在用融合 kernel**。改过来后端到端 **2.13× / 2.00× / 1.52×**。

---

## PR 正文（可直接粘贴）

### Motivation

`Gemma3RMSNorm.forward_cuda` returns `self.forward_native(x)`, so on CUDA every
Gemma-3 normalisation runs as eager PyTorch — `pow`, `mean`, `add`, `rsqrt`,
`mul`, plus the fp32 up/down casts. That is roughly six kernels and seven HBM
round trips per norm, where one fused kernel would do.

`gemma_rmsnorm` already implements exactly this class's semantics,
`out = (x / RMS(x)) * (1 + weight)`, is pre-built in `sgl_kernel`, and is what
`GemmaRMSNorm` in this same file already uses for gemma/gemma2.

This does not look like a deliberate accuracy trade-off:

* the two classes have **identical reference implementations** — both upcast to
  fp32, both apply `(1.0 + weight)`, both cast back at the end;
* Gemma-3's **CPU** path dispatches to `gemma3_rmsnorm_cpu` and its **NPU** path
  to `npu_gemma_rms_norm`. CUDA is the only backend that falls through.

Gemma-3-1B runs 6 norms per layer (`input`, `post_attention`,
`pre_feedforward`, `post_feedforward`, `q_norm`, `k_norm`) across 26 layers =
**157 norm calls per forward**.

### Implementation notes

Two details that are easy to get wrong:

1. **`self.weight` is fp32.** It is created by `nn.Parameter(torch.zeros(dim))`
   while activations are half precision. Handing the fused kernel a mismatched
   weight **does not raise — it silently produces NaNs.** The cast is cached per
   module and keyed on dtype so a re-cast module is not served a stale buffer.
2. **`q_norm`/`k_norm` receive `[tokens, heads, head_dim]`.** RMSNorm reduces
   over the last dimension only, so flattening to 2-D and restoring the shape is
   exact. A rank-2 guard would leave **2 of the 6 norms per layer** on the slow
   path — worth checking, since that alone was the difference between 1.56× and
   2.07× in my measurements.

Anything the fused kernel cannot serve (non-CUDA/XPU, fp32 activations,
mismatched trailing dim) keeps `forward_native`.

### Tests

`Gemma3RMSNorm` previously had **no test**, which is part of why this went
unnoticed. Adds `TestGemma3RMSNorm` covering 2-D and 3-D inputs, bf16/fp16, and
both fp32 and half weights.

The test is mutation-checked — it is load-bearing, not decorative:

| mutation | result |
|---|---|
| remove the weight dtype cast | **64 subtests fail** |
| drop the 3-D shape restore | **24 subtests fail** |
| unmodified patch | 2 passed, 104 subtests passed |

### Benchmarks

1× H200, `gemma-3-1b-it`, BF16, `sglang.bench_serving`, **8 repetitions per
arm**, Welch t with the exact Student-t tail. Both arms verified to resolve to
the same backend / attention / CUDA-graph settings.

| regime | baseline | patched | **speedup** | p |
|---|---:|---:|---:|---:|
| low-batch decode | 0.839 req/s | 1.784 req/s | **2.128×** | 3.5e-22 |
| concurrent decode | 21.671 req/s | 43.247 req/s | **1.996×** | 4.2e-18 |
| long prefill | 17.156 req/s | 26.088 req/s | **1.521×** | 4.5e-15 |

Profiling confirms the mechanism rather than just the outcome: eager-norm kernel
launches go **157 → 0** per forward, and decode CUDA kernel time falls
**3.81 ms → 1.89 ms**.

### Accuracy

GSM8K, 1319 questions, greedy, 3 runs: **0.2260 → 0.2213**. The binomial error
at n=1319 and p≈0.22 is ±2.2 points, so this is inside the noise.

Numeric verification across **120 combinations** of shape, dtype, magnitude and
weight dtype: worst relative deviation **9.3e-3**, no NaN/Inf. That is
half-precision rounding, and it is the same trade-off `GemmaRMSNorm` already
ships for gemma/gemma2.

### Scope / caveats

* Measured on **gemma-3-1b-it only**. Decode on a 1B model is heavily
  launch-bound, so the ~2× should be read as an upper bound — larger Gemma-3
  checkpoints will see less, since they do more compute per forward to hide the
  overhead behind. I could not measure them (gated on Hugging Face).
* The change is **not bit-identical** to the eager path, as above.
* Functional testing was done on **v0.5.12.post1** (`17f7a1da1`). The change was
  then ported to `main`; the port compiles and the diff is byte-identical, but
  `main`'s test suite could not run in my environment (it needs a newer
  `transformers` than I have — `PreTrainedConfig` import fails). **A maintainer
  should re-run CI on main.**

---

## 我做过的验证（比 PR 正文更细）

| 验证 | 方法 | 结果 |
|---|---|---|
| 数值正确性 | 120 组合（10 shape × 2 dtype × 3 量级 × 2 weight dtype） | 0 失败，最差相对偏差 9.3e-3 |
| 不破坏现有测试 | 打补丁前后各跑一遍 `test_layernorm.py` | **完全相同**（201 failed / 3 passed / 423 subtests passed，失败是环境预存、在 `TestLayerNorm` 另一个类） |
| 新测试有效性 | 两次变异测试 | 分别抓到 64 / 24 个失败 |
| 端到端 | 真实源码补丁经 PYTHONPATH 加载（**非 monkeypatch**），8 次重复 | 2.13× / 2.00× / 1.52×，p ≤ 4.5e-15 |
| 机制闭环 | 打补丁后重跑算子审计 | eager norm 157 → **0**，kernel 时间 −50.4% |
| 质量 | GSM8K 全量 1319 题 × 3 | 噪声内 |
| 可移植性 | 补丁应用到 `origin/main` | 手动适配成功，改动逐字一致，编译通过 |

**原始证据**：`results/lfm_fusion/pr_gemma3/`（补丁、120 组数值验证 JSON、三份 E2E/质量 log）。

---

## 发之前还需要做的

1. **你确认要发**（会署你的名）
2. 把分支 push 到 `gujialiang123/sglang` 的 fork，再开 PR
3. 建议在 PR 里主动 @ 维护者说明"main 的 CI 我没能在本地跑通，请帮忙跑一遍"
4. 可选：先开 issue 描述现象，让维护者确认是不是他们已知的取舍

**命令**（确认后我可以执行）：
```bash
cd /tmp/sglang_pr_main
git push fork fix/gemma3-rmsnorm-cuda-fused
# 然后到 GitHub 上从 gujialiang123/sglang 向 sgl-project/sglang 开 PR
```
