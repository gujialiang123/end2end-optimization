# LFM2.5 kernel-fusion gaps — audit, patch and end-to-end result

**Date:** 2026-07-27 · **Model:** LFM2.5-8B-A1B · **Control:** Qwen3-30B-A3B
**Frame:** 1× H200, TP1, BF16, sglang 0.5.12.post1 @ `17f7a1da1`, torch
2.9.1+cu128, Triton 3.5.1, driver 580.105.08.

Scripts `scripts/lfm_fusion/`, data `results/lfm_fusion/`.

---

## 0. One-paragraph summary

The v33 decode audit concluded that for Qwen3-30B **every hot path in SGLang is
already CUDA-fused, so there is no gap to fill**, and that finding has been the
basis for deprioritising kernel-fusion work ever since. That conclusion is
correct — *for Qwen*. LFM2.5-8B-A1B, a newer hybrid architecture where 18 of 24
layers are gated short convolutions rather than attention, had never been
audited at the operator level. It turns out to carry **two real fusion gaps that
Qwen does not have**, worth **+3.8 % to +4.0 % end-to-end on the two decode
regimes** and +0.9 % on long prefill, with GSM8K accuracy unchanged. This is the
first same-model kernel-level change in this project to produce a *positive,
statistically significant, whole-regime* end-to-end result.

| regime | `scale` only | `norm` only | **both** |
|---|---:|---:|---:|
| A low-batch decode | +1.40 % (p<1e-4) | +2.35 % (p<1e-4) | **+3.82 %** (p<1e-4) |
| B concurrent decode | +1.02 % (p=0.005) | +2.89 % (p<1e-4) | **+3.96 %** (p<1e-4) |
| C long prefill | +0.73 % (n.s.) | +1.42 % (p=2e-4) | **+0.91 %** (p=0.031) |

5 repetitions per arm after per-workload warm-up; Welch t against the baseline
arm; ratios on request throughput.

---

## 1. Why look here at all

Everything this project has measured says kernel rewriting is a dead end:
a hand-written small-M MoE kernel was 1.23× in isolation and +1.2 % end-to-end
(with real regressions at b≥2), shared-expert gate fusion was 2–3× in isolation
and 1.00× end-to-end, and re-tuning already-covered configs was 0. The v33
audit explained why: decode is 89 % memory-bound weight and KV reading, so
arithmetic wins do not convert.

But v33's *other* conclusion — "no un-fused gaps remain" — was established on
**one model**. Its own table of CPU-only fused kernels (`fused_linear_sigmoid_mul`,
`fused_gdn_gating`, `fused_rmsnorm_gated`) notes each serves a *different*
architecture. That is a hint that the coverage gap is a property of
**architecture maturity**, not of SGLang as a whole. LFM2.5 is the newest
architecture available on this machine and had never been profiled here.

---

## 2. The audit

`scripts/lfm_fusion/lf_audit.py` reruns the v33 method — `sglang.bench_one_batch
--profile` with CUDA graphs disabled so each operator appears as its own kernel
— then buckets CUDA kernel time by name and, separately, counts kernels matching
*fusion-gap signatures*: work a fused implementation would not perform at all.

Kernel counts are per forward pass and are identical across regimes, which is
itself the tell — these are structural, not workload-dependent.

| model | regime / stage | total kernel time | unfused RMSNorm | residual add | gating mul | layout copy |
|---|---|---:|---:|---:|---:|---:|
| **LFM2.5** | A decode | 1.99 ms | **61** (5.45 %) | **48** (2.60 %) | **36** (1.80 %) | 22 (1.46 %) |
| **LFM2.5** | B decode | 4.60 ms | **61** (2.60 %) | **48** (1.38 %) | **36** (1.92 %) | 34 (1.41 %) |
| **LFM2.5** | C prefill | 157.14 ms | **61** (4.27 %) | **50** (1.40 %) | **36** (3.65 %) | 53 (2.93 %) |
| Qwen3-30B | A decode | 3.71 ms | **1** (0.05 %) | **0** | **0** | 4 (0.18 %) |
| Qwen3-30B | C prefill | 247.11 ms | **1** (0.02 %) | **0** | **0** | 52 (3.15 %) |

**The control is decisive.** Qwen issues *one* un-fused RMSNorm and *zero*
standalone residual adds in an entire forward pass; LFM2.5 issues 61 and 48.
The counts are not approximate — 48 = 2 residual additions × 24 layers, and
36 = 2 gating multiplies × 18 convolution layers.

### 2.1 The architectural trade the audit exposes

Comparing the two models on the same long-prefill workload:

| bucket | LFM2.5 | Qwen3-30B |
|---|---:|---:|
| MoE | 70.8 % | 54.2 % |
| **attention** | **2.8 %** | **21.6 %** |
| short conv | 0.7 % | — |
| dense GEMM | 12.5 % | 16.1 % |
| **norm + elementwise** | **12.8 %** | **5.6 %** |

LFM2.5's architecture does exactly what it promises: attention plus the
convolution replacing it costs **3.5 % against Qwen's 21.6 %**. But it hands a
large part of that structural win back as **12.8 % of un-fused glue versus
Qwen's 5.6 %**. The headroom is not in the new operator — `causal_conv1d` is
already fast and takes 0.7 % — it is in the *plumbing around it that upstream
has not yet fused*.

---

## 3. The two gaps, and the fix

`scripts/lfm_fusion/lfm_fusion_patch.py`. Both are opt-in through
`LFM_FUSION_PATCH`; with the variable unset the stock path is untouched, so the
baseline arm is genuinely unmodified SGLang. Injection uses a `sys.meta_path`
finder (`lf_inject/sitecustomize.py`) that patches the class the moment the
module finishes importing — the model registry imports it lazily, so a timer
would have been a race.

### G1 `norm` — the residual add is never fused

`Lfm2MoeDecoderLayer.forward` calls RMSNorm without a residual and then adds the
residual with a separate elementwise kernel:

```python
residual = hidden_states
normed = self.operator_norm(hidden_states)
hidden_states = <attn or conv>(normed)
hidden_states = hidden_states + residual                       # separate kernel
hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
```

`RMSNorm.forward_cuda(x, residual)` already dispatches to `fused_add_rmsnorm`,
which does both in one pass — and `Lfm2MoeModel.forward` *already threads a
`residual` through the layer loop*. The layer simply ignores the value it is
handed and overwrites it. Rewriting to the deferred-residual convention that
every other SGLang model uses removes 2 kernels per layer.

Equivalence, writing `x` for the activation entering the layer:

```
stock:  a = op(rms(x));  h1 = a + x;  out = h1 + ffn(rms(h1))
patched: rms(x, r) -> r := x, n := rms(x);  a = op(n)
         rms(a, r) -> r := a + x = h1, n2 := rms(h1);  return (ffn(n2), h1)
```

and the next layer consumes `ffn(n2) + h1`, which is the same value.

### G2 `scale` — a full-tensor multiply by 1.0

`Lfm2MoeSparseMoeBlock.forward` ends with

```python
return final_hidden_states * self.routed_scaling_factor
```

For LFM2.5 `routed_scaling_factor == 1.0`, so this is an elementwise multiply by
one over the whole activation — **22 no-op kernel launches per forward**. The
comment above it explains why the factor is applied manually rather than inside
`FusedMoE` (numerical differences against HuggingFace), which is a good reason
to keep the multiply *when the factor is not 1*. Skipping it when the factor is
exactly 1.0 is bit-exact.

### G3 `conv` — audited, not fixed

`Lfm2MoeShortConv.forward` materialises `Bx.transpose(0, 1).contiguous()` on the
prefill path and transposes the result back. This is the bulk of the
`layout_copy` line (53 kernels, 2.9 % on long prefill). Fixing it means changing
the memory layout expected by `causal_conv1d_fn`, which is a real kernel change
rather than a call-site change, so it is recorded and left.

---

## 4. Correctness

**G2 is bit-exact** — multiplying a finite bf16 value by 1.0 is the identity.

**G1 is algebraically exact but not bit-exact**, because `fused_add_rmsnorm`
accumulates differently from a bf16 `add` followed by a normalisation. Three
levels of evidence:

1. **Primitive.** `fused_add_rmsnorm` versus manual `add` + `rmsnorm` on random
   bf16 tensors: the residual matches **exactly** (max abs diff 0.0) and the
   normalised output differs by 0.03 on a magnitude of 4.34 — about 2 bf16 ulp,
   with the fused version being the *more* accurate of the two since it keeps
   the addition in higher precision.
2. **Restructuring.** A 6-layer stand-in stack (real `RMSNorm`, random linear
   maps) run both ways: relative deviation 1.1 % after 6 layers, consistent with
   bf16 accumulation drift and with no structural error.
3. **Whole model.** Next-token distributions over 12 prompts: top-1 token agrees
   on 11/12, but the head-restricted logprob deviation reaches 2.9 and the
   symmetrised KL reaches 0.99 on one prompt.

Level 3 looks alarming until the mechanism is named: **LFM2.5 routes through
top-4 of 32 experts, and expert selection is a discrete argmax over router
logits.** A bf16-level perturbation occasionally flips which expert fires, which
changes the output discontinuously. So token-identity is structurally
unavailable for *any* numerically non-identical change to this model, and it is
the wrong gate.

The right gate is task quality. See §5.

---

## 5. Quality gate — and a noise floor that came for free

Few-shot GSM8K, greedy, full 1319-question set, 3 evaluations per server launch
(`sglang.test.few_shot_gsm8k`; raw in
`results/lfm_fusion/correctness/accuracy_*.json`).

| arm | run 1 | run 2 | run 3 | mean | within-arm spread |
|---|---:|---:|---:|---:|---:|
| baseline | 0.348 | 0.349 | 0.344 | 0.3470 | 0.005 |
| **`scale`** (bit-exact) | 0.338 | 0.339 | 0.340 | **0.3390** | 0.002 |
| `norm` | 0.362 | 0.368 | 0.361 | 0.3637 | 0.007 |
| `norm+scale` | 0.359 | 0.359 | 0.359 | 0.3590 | 0.000 |

The `scale` arm is **provably bit-exact**, so its accuracy *must* equal
baseline's. It does not: it reads **0.8 points lower**. That is not a defect, it
is a measurement of the harness — `--parallel 32` means batch composition
differs between server instances, and batch-dependent reductions change greedy
outputs. So we get the noise floor by construction instead of assuming one:

* **between-arm systematic noise ≥ 0.8 points**, measured on a change that
  cannot possibly alter the model;
* within-arm spread 0.0–0.7 points;
* binomial sampling error at n=1319, p≈0.35 is ±2.6 points at 95 %.

All four arms span 0.339–0.364, i.e. **2.5 points — inside the noise band on
every one of those three measures.** The patched arms happen to sit above
baseline rather than below, which is at least consistent with `fused_add_rmsnorm`
keeping the addition in higher precision, but the separation is far too small to
claim that.

**Verdict: no quality regression detected.** Not "quality improved" — the
experiment cannot resolve a difference this small, and the bit-exact arm proves
it.

---

## 6. End-to-end

`scripts/lfm_fusion/lf_e2e.py`, reusing the canonical serving harness so numbers
are comparable with the serving-ceiling and regime-kernel campaigns. Arms differ
*only* by the `LFM_FUSION_PATCH` variable; model, serving knobs, backend and
CUDA-graph settings are identical, and the server log is checked for the patch
marker so a silent no-op cannot be scored as a baseline-equal result.

| regime | baseline req/s | `scale` | `norm` | **`norm+scale`** |
|---|---:|---:|---:|---:|
| A low-batch decode | 1.688 ± 0.002 | 1.712 (+1.40 %) | 1.728 (+2.35 %) | **1.753 (+3.82 %)** |
| B concurrent decode | 21.767 ± 0.085 | 21.988 (+1.02 %) | 22.395 (+2.89 %) | **22.628 (+3.96 %)** |
| C long prefill | 12.282 ± 0.075 | 12.371 (+0.73 %, n.s.) | 12.457 (+1.42 %) | **12.393 (+0.91 %)** |

Every arm except `scale` on long prefill is significant at p<0.05, and the two
components are close to additive on the decode regimes (1.40+2.35 vs 3.82
measured; 1.02+2.89 vs 3.96).

### Why the gain is regime-dependent

The removed work is a **fixed number of kernels and a fixed number of
full-activation passes per forward**, independent of how much compute that
forward performs. Decode does very little work per forward, so fixed overhead is
a large fraction of it; long prefill does ~80× more kernel work per forward
(157 ms vs 2 ms), so the same absolute saving is proportionally small. This is
the mirror image of the configuration result, which only paid off at large M.

**This is a genuine regime-aware kernel result and it points the opposite way
from the config study**: config tuning helped prefill and did nothing for
decode; fusion helps decode and does little for prefill. They are complementary,
and neither is visible if you only measure one regime.

---

## 7. What this changes about the project's standing conclusions

The previous position was "kernel-level work does not convert to end-to-end
gains on mature bf16 MoE serving". That stands. What this adds is a boundary
condition:

> The coverage gap is a function of **architecture maturity**, not of SGLang.
> A model family that upstream has optimised (Qwen3-30B: 1 un-fused norm, 0
> stray adds) has nothing left at the fusion layer. A recently added
> architecture (LFM2.5: 61 un-fused norms, 48 stray adds, 22 multiplies by one)
> carries several percent of pure overhead — not in its novel operator, which is
> already fast, but in the call-site plumbing around it.

This is precisely the niche the plan assigns to the agent — *"automatically find
and fill SGLang's coverage gaps"* — and it is the first time that framing has
produced a positive, significant, whole-regime end-to-end number on the same
model. It is also cheap to detect: the entire audit is one profiled
`bench_one_batch` run per model, and the signature is a *kernel count that
scales with layer count*, which is mechanically checkable rather than a
judgement call.

**Honest scope.** The absolute number is small (≈4 %). It is one model on one
GPU. The `norm` component is not bit-identical and its quality claim rests on a
task metric inside a noisy harness, not on token identity. And the win comes
from applying an existing fused primitive at a call site that failed to use it —
**no new kernel was written**, which is the same shape as every other positive
result this project has found.

---

## 8. Reproduce

```bash
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$ENVDIR PATH=$ENVDIR/bin:$PATH HF_HOME=$PWD/.hf_cache

# operator-level audit (LFM2.5 + Qwen control)
python scripts/lfm_fusion/lf_audit.py --model lfm25 --regime A_low_batch_decode --gpu 5
./scripts/lfm_fusion/run_audit_all.sh 5

# end-to-end A/B, all four arms
python scripts/lfm_fusion/lf_e2e.py --regime A_low_batch_decode --gpu 4 \
    --arms baseline,scale,norm,norm+scale --reps 5
python scripts/lfm_fusion/lf_analyze.py          # -> processed/fusion_ab.csv

# correctness
python scripts/lfm_fusion/lf_correctness.py collect  --arm baseline   --gpu 5
python scripts/lfm_fusion/lf_correctness.py collect  --arm norm+scale --gpu 5
python scripts/lfm_fusion/lf_correctness.py compare  --arms baseline,norm+scale
python scripts/lfm_fusion/lf_correctness.py accuracy --arm baseline --gpu 5 \
    --num-questions 1319 --reps 3
```

## 9. Next

1. **G3, the ShortConv layout copy** — 2.9 % of long-prefill kernel time, needs a
   layout change in the `causal_conv1d_fn` call rather than a call-site edit.
2. **Fuse the ShortConv gating multiplies** into the conv kernel (36 kernels,
   1.8–3.7 %); `causal_conv1d_update` already takes an `activation` argument, so
   a gating argument is a natural extension.
3. **Turn the audit into an agent check.** The gap signature is a kernel count
   that scales with layer count, and the control model shows what "clean" looks
   like. This is a mechanical rule an agent can apply to any newly added
   architecture without a human reading the model file.
4. **Upstream G2.** The multiply by `routed_scaling_factor == 1.0` is a
   two-line, bit-exact fix that costs 22 kernel launches per forward.
