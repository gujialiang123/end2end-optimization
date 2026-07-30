# docs/

Two kinds of directory:

* **`YYYY-MM-DD/`** — dated experiment records. A report lives under the date the
  work was *done*, not the date it was last edited. This is the primary
  convention; anything reporting a measurement belongs here.
* **named directories** — living documents that are maintained rather than
  dated: `architecture/`, `development/`, `problem-package/`, `idea-pool/`,
  `research/`, `skills/`, `reports/`, `paper_audit/`.

Reusable *methodology* does not live here at all — it goes to
`.github/skills/<name>/SKILL.md`.

---

## Index of the kernel-optimisation line (2026-07-26 → 07-28)

Read top-down; each entry says what it is and whether it is still current.

| doc | what's in it |
|---|---|
| [`architecture_primer.md`](architecture_primer.md) | **推理侧模型架构入门，给 infra 视角写的。** 一层 Transformer 的四件事、Q/K/V 和多头、QK-Norm 为什么存在（含 softmax 饱和的实测数据）、GQA 如何决定 KV cache、prefill/decode 为何瓶颈相反。所有 shape 取自 gemma-3-1b 真实 config，每节标注了我们做过的哪个优化落在这一层。 |

### 2026-07-30 — how SGLang onboards models and dispatches backends

| doc | what's in it |
|---|---|
| [`2026-07-30/sglang_model_onboarding_and_backend_dispatch.md`](2026-07-30/sglang_model_onboarding_and_backend_dispatch.md) | **调研：SGLang 内部到底怎么接模型、怎么选 backend，以及和 torch.compile/FX 体系的接口在哪。** 三个可复核的结论：212 个模型是手写的（通用路径靠类名字符串匹配，不是 FX）；backend 决策是手写 if/elif，看硬件不看图；从 vLLM 移植了 FX pass 基础设施但**融合 pass 没跟过来**——挂载点是空的。附四个实验方案，以及已完成的预实验（post-grad 图上 RMSNorm pattern 干净可匹配，风险已排除）。 |

### 2026-07-29 — the Triton 3.6 re-tune (retracted)

| doc | what's in it |
|---|---|
| [`2026-07-29/RETRACTION_triton36_baseline_contamination.md`](2026-07-29/RETRACTION_triton36_baseline_contamination.md) | **Read this before the one below.** The 3.6 "default" baseline was my own tuned config, loaded through the cross-version fallback because the experiment imported the PR branch worktree. Zero BK=32 kernels in 8000+ compiled on 3.6 proves the default path never ran, and the numbers decompose exactly into tuning's own gain times a 1.04x version bump. The 3.5.1 measurements and PR #32687 are unaffected. |

### 2026-07-29 — the Triton 3.6 re-tune

| doc | what's in it |
|---|---|
| [`2026-07-29/triton_36_retune_findings.md`](2026-07-29/triton_36_retune_findings.md) | **A negative result worth more than the positive one it replaces.** Re-tuning the LFM2.5 MoE shape on Triton 3.6 found 0/19 buckets worth specialising: the compiler upgrade (+29.8% end-to-end) already took everything hand-tuning bought on 3.5.1 (+23.3%), and 3.6's *default* beats 3.5.1's *tuned*. Includes what it means for PR #32687 and the toolchain fixes needed to measure it. |

### 2026-07-28 — cross-architecture audit, and the upstream PR

| doc | what it is |
|---|---|
| [`2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md`](2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md) | 第二个上游候选 → [sgl-project/sglang#32687](https://github.com/sgl-project/sglang/pull/32687) (draft)。LFM2.5 的 H200 MoE tuned config。#22791 已为同一 shape 提交了 H100/B200/MI325X，唯独漏了 H200。长 prefill **+23.3%**（8/8 不重叠），decode 经顺序对照后中性。 |
| [`2026-07-28/four_kernel_cases_comparison.md`](2026-07-28/four_kernel_cases_comparison.md) | **对照报告：四种干预方式的收益对比。** 调 config（LFM，+22.3%）vs 重写 kernel（Qwen b=1，+1.17% 且 b≥2 回归）vs 补融合（LFM 七组件，+6.6%）vs 接线（Gemma-3，+36.6%）。收益与"写了多少 kernel 代码"成反比，并给出机制解释。 |
| [`2026-07-28/cross_architecture_audit.md`](2026-07-28/cross_architecture_audit.md) | 11 models / 8 families / 0.6 B–80 B. **Refutes** the "architecture maturity" hypothesis. §7.1 further corrects "family" as the predictor (IBM granite at 0.30% is the counter-example); §7.2 documents a systematic over-estimate in the audit itself when a gap sits behind a CUDA-graph capture guard. |
| [`2026-07-28/three_fusion_cases.md`](2026-07-28/three_fusion_cases.md) | Every case in technical detail — the three that paid off, the four that did **not**, and a precision analysis answering whether any of it is lossy. **Start here for the engineering.** |
| [`2026-07-28/PR_DRAFT_gemma3_rmsnorm_v2.md`](2026-07-28/PR_DRAFT_gemma3_rmsnorm_v2.md) | The upstream PR and everything backing it → [sgl-project/sglang#32670](https://github.com/sgl-project/sglang/pull/32670) (draft). Rescoped after upstream #32383 landed the 2-D half: the claim is now the high-rank + dtype increment, measured against a main-equivalent baseline. Supersedes [`..._v1_superseded.md`](2026-07-28/PR_DRAFT_gemma3_rmsnorm_v1_superseded.md). |

### 2026-07-27 — LFM2.5 fusion work

| doc | what it is |
|---|---|
| [`2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md`](2026-07-27/LFM_KERNEL_OPTIMIZATION_FULL_REPORT.md) | Self-contained two-day summary for the team. Carries a correction notice: its "architecture maturity" framing was superseded on 07-28. |
| [`2026-07-27/lfm_fusion_results.md`](2026-07-27/lfm_fusion_results.md) | The full methodology record — audit, the seven components, the sub-additivity result, the noise-floor calibration. |
| [`2026-07-27/regime_kernel_results.md`](2026-07-27/regime_kernel_results.md) | Regime-aware kernel specialisation, incl. §11c (a regime→backend rule does **not** transfer across models). |

### 2026-07-26 — regime-kernel study setup

| doc | what it is |
|---|---|
| [`2026-07-26/regime_kernel_status.md`](2026-07-26/regime_kernel_status.md) | Repo audit, reusable assets, cost estimate. |
| [`2026-07-26/regime_kernel_experiment_plan.md`](2026-07-26/regime_kernel_experiment_plan.md) | Methodology, search space, measurement protocol. |

### Methodology extracted from all of the above

**[`.github/skills/fusion-gap-hunting/SKILL.md`](../.github/skills/fusion-gap-hunting/SKILL.md)** —
how to find kernels a model runs unfused that the framework already ships a
fused implementation for. Includes a runnable scanner and eight verification
disciplines. **Read this before writing any kernel.**

---

## Where the raw data is

`results/lfm_fusion/` — `audit/` (per-model operator audits), `e2e/` (A/B runs),
`processed/` (tidy CSVs), `correctness/` (quality gates), `pr_gemma3/` (the PR
evidence bundle), `nsys/` and `fx/` (the two investigation reports), `plots/`.

Project-level history and the running log are in `plan.md`; the handoff for a
fresh session is `HANDOFF_regime_kernel.md`.
