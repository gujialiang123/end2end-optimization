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

### 2026-07-28 — cross-architecture audit, and the upstream PR

| doc | what it is |
|---|---|
| [`2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md`](2026-07-28/PR_DRAFT_lfm25_h200_moe_config.md) | 第二个上游候选：LFM2.5 的 H200 MoE tuned config。#22791 已为同一 shape 提交了 H100/B200/MI325X，唯独漏了 H200。长 prefill **+23.3%**（8/8 不重叠），decode 经顺序对照后中性。 |
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
