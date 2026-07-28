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
| [`2026-07-28/cross_architecture_audit.md`](2026-07-28/cross_architecture_audit.md) | 6 models / 3 families / 0.6 B–80 B. **Refutes** the "architecture maturity" hypothesis — the predictor is model family, not novelty, type or size. |
| [`2026-07-28/three_fusion_cases.md`](2026-07-28/three_fusion_cases.md) | The three same-pattern hits in technical detail: before/after code, why each was missed, measured gain. **Start here for the engineering.** |
| [`2026-07-28/PR_DRAFT_gemma3_rmsnorm.md`](2026-07-28/PR_DRAFT_gemma3_rmsnorm.md) | The upstream PR and everything backing it → [sgl-project/sglang#32670](https://github.com/sgl-project/sglang/pull/32670) (draft). |

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
