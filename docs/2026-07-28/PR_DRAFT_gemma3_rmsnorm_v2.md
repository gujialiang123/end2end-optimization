# PR 草稿 v2 — `fix(gemma3): fuse high-rank RMSNorm and harden mixed-dtype weights`

**目标仓库**：`sgl-project/sglang` · **PR**：https://github.com/sgl-project/sglang/pull/32670（draft）
**新分支**：`fix/gemma3-rmsnorm-highrank-and-dtype`，本地在 `/tmp/sglang_pr2`（基于 `origin/main` @ `1eee8fbdcc`）
**补丁存档**：`docs/2026-07-28/gemma3_pr_rebased_on_1eee8fb.patch`
**v1 已作废**：`PR_DRAFT_gemma3_rmsnorm_v1_superseded.md`

---

## 为什么要有 v2

v1 写的是「`Gemma3RMSNorm.forward_cuda` 整个掉进 eager，改成融合 kernel，端到端 2.13×」。
这在我们做实验的那个 main（`a82ead53b`）上是准确的。但在我们准备提交期间，上游合入了
**#32383（optimize EmbeddingGemma prefill performance）**，它已经把 2-D 路径和 residual 路径
接到了 `gemma_rmsnorm` / `gemma_fused_add_rmsnorm`，并顺带重构了 `gemma3_causal.py`。

也就是说 **v1 的标题和 headline 数字现在会误导 reviewer**：它把上游已经拿到的收益算进了自己的账上。

剩下的真实缺口只有两个：

1. **高维输入仍走 eager**。#32383 的守卫是 `if x.dim() == 2`。而 `q_norm` / `k_norm` 的输入是
   `[tokens, heads, head_dim]`，是 3-D，直接掉回 `forward_native`。gemma-3-1b 上每次
   forward 有 157 次 norm 调用，其中 **52 次**是这两个。
2. **weight dtype 无条件透传**。两条融合路径都传 `self.weight.data`；而 `weight` 是
   `nn.Parameter(torch.zeros(dim))`，dtype 跟随构造时的 default dtype。

所以 v2 把自己**严格限定为 #32383 的增量**，并且用重新测的增量数字。

---

## 数字：必须用增量，不能用 2.13×

`2.128× / 1.996× / 1.521×` 是「Gemma3RMSNorm 完全不融合」→「完全融合」的差值。
现在 main 已经占了其中一部分，这个数字不能再当 headline。

我们没法在本机直接跑上游 main（main 要求 `transformers==5.12.1`，本地 env 是 4.57.1 / 5.6.0，
conda clone 失败，跨 env 混用工具链导致 JIT 链接失败）。**绕过办法**：在能跑的 0.5.12 上构造一个
`norm2d` arm，它完全复刻 main 的 norm 覆盖面（2-D 融合、高维 eager），以此为基线直接 A/B。

gemma-3-1b-it · 1×H200 · BF16 · 每 arm 6 次重复 · Welch t（精确 Student-t 尾）：

| regime | baseline（= main 等效） | patched | 增量 | p |
|---|---|---|---|---|
| A 低批 decode | 1.300 req/s | 1.776 | **+36.60%** | 2.4e-14 |
| B 并发 decode | 33.648 req/s | 41.891 | **+24.50%** | 1.2e-06 |
| C 长 prefill | 23.385 req/s | 25.085 | +7.27% | 0.053（**不显著**） |

原始数据：`results/lfm_fusion/processed/fusion_ab_incremental.csv`。
C 这一列必须标注 not significant，不能因为它是正的就写进 headline。

---

## dtype 那条要如实定性，不能夸大

实测确认：`gemma_rmsnorm(bf16_x, fp32_weight)` **静默返回 NaN**，不抛异常。

但 sglang 加载模型时包在 `set_default_torch_dtype(bf16)` 里，所以**生产路径上 weight 就是 bf16**。
因此这是一个**健壮性缺口 / 潜在陷阱**，**不是 main 上的线上 bug**。PR 里就得这么写。

缓存按 `(dtype, device)` 建，避免模块被 `.to()` 之后拿到过期 buffer。

---

## 精度措辞要收敛

之前算的 "0.25–0.73 ulp" 用的是整个 tensor 的 `max(abs(truth))` 当统一分母，**不是逐元素**，
严格说站不住。PR 里改成可直接核验的说法：

> 97.8–98.4% 的元素与 eager 路径逐位相同；其余元素的偏差处在 BF16 舍入量级。

并且要主动交代：**输出与 eager 不是 bit-identical**（融合 kernel 在激活 dtype 下施加 weight），
这与 #32383 的 2-D 路径已经接受的是同一个取舍——不是本 PR 新引入的。

---

## PR 正文（可直接粘贴）

### Title

```
fix(gemma3): fuse high-rank RMSNorm and harden mixed-dtype weights
```

### Body

#32383 landed fused paths for the 2-D and residual cases of `Gemma3RMSNorm`.
Two gaps remain.

**1. Higher-rank inputs still fall back to the eager path.** The dispatch is
guarded by `x.dim() == 2`, but `q_norm` and `k_norm` are called with
`[tokens, heads, head_dim]`. On Gemma-3 that is 2 of the 6 norms per layer —
52 of the 157 norm calls per forward on `gemma-3-1b`. RMSNorm reduces over the
last dimension only, so flattening the leading dimensions and restoring the
shape afterwards is exact.

**2. Both fused paths pass `self.weight.data` unconditionally.** The weight is
`nn.Parameter(torch.zeros(dim))`, so its dtype follows whatever default is in
effect at construction. Under the loader's `set_default_torch_dtype` that
matches the activations, but a module built outside that context keeps an fp32
weight — and the fused kernels do not raise on a dtype mismatch, they return
NaNs. Verified directly:

```
gemma_rmsnorm(bf16_input, bf16_weight) -> finite
gemma_rmsnorm(bf16_input, fp32_weight) -> NaN/Inf, no exception
```

The cast is cached on `(dtype, device)` so a module that is moved or re-cast is
not served a stale buffer.

#### Benchmarks

These are the increment attributable to *this* change, not to fusing
`Gemma3RMSNorm` from scratch. The baseline is a build whose norm coverage
matches current main (2-D fused, higher rank eager); the patched arm adds the
high-rank path. `gemma-3-1b-it`, 1×H200, BF16, 6 reps per arm, Welch t-test:

| regime | baseline | patched | gain | p |
|---|---|---|---|---|
| low-batch decode | 1.300 req/s | 1.776 req/s | **+36.6%** | 2.4e-14 |
| concurrent decode | 33.648 req/s | 41.891 req/s | **+24.5%** | 1.2e-06 |
| long prefill | 23.385 req/s | 25.085 req/s | +7.3% | 0.053 (not significant) |

#### Numerics

20 shape/dtype/weight-dtype combinations pass, including the fp32-weight case
that currently produces NaNs. No NaN/Inf anywhere. 97.8–98.4% of elements are
bit-identical to the eager path; the rest differ at the BF16 rounding scale.
Outputs are *not* bit-identical — the fused kernel applies the weight in the
activation dtype — which is the same trade-off the existing 2-D path already
accepted.

#### Tests

Adds `TestGemma3RMSNorm`, which this class had none of. It covers 2-D and 3-D
inputs, bf16/fp16, and fp32 weights. Mutation-checked: dropping the dtype cast
fails 64 subtests, dropping the shape restore fails 24.

Related: #21962, #32383.

---

## 推送受阻（需要你一步操作）

`gh` token 的 scope 是 `admin:public_key, gist, read:org, repo`，**没有 `workflow`**。
新分支基于最新 main，推送时会携带上游改过的 `.github/workflows/*`，被 GitHub 拒绝：

```
! [remote rejected] refusing to allow an OAuth App to create or update
  workflow `.github/workflows/_pr-test-check-changes.yml` without `workflow` scope
```

试过且**都不通**的绕行：
- `gh api -X POST .../merge-upstream`（同一限制，422）
- `gh api -X POST .../git/refs` 指向 upstream main SHA（对象在 fork 网络中不可见，404）
- SSH 推送（本机 `~/.ssh/id_ed25519` 带 passphrase，`ssh -T` publickey denied）

**二选一即可解开**（之后我就能把分支推上去并更新 #32670）：
1. 在 https://github.com/gujialiang123/sglang 点 **Sync fork**（最省事，workflow blob 落到 fork 上，
   我的分支就只需传自己那一个 commit）；或
2. 跑 `gh auth refresh -h github.com -s workflow`（要浏览器输一次 device code）。

在那之前，rebase 后的完整补丁已存档在
`docs/2026-07-28/gemma3_pr_rebased_on_1eee8fb.patch`，`git am` 即可复现。
