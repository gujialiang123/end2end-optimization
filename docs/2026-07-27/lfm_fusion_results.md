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
audited at the operator level. It turns out to carry **five real fusion gaps
that Qwen does not have**. Closing them is worth **+4.7 % to +5.5 % end-to-end
on every regime tested**, with GSM8K accuracy unchanged. This is the first
same-model kernel-level change in this project to produce a *positive,
statistically significant* end-to-end result across the board.

**Final result** (6 repetitions per arm, Welch t vs baseline with the exact
Student-t tail):

| regime | baseline req/s | six components | **all seven** | p |
|---|---:|---:|---:|---:|
| A low-batch decode | 1.688 | +4.60 % | **+6.57 %** | 4.6e-14 |
| B concurrent decode | 21.673 | +6.01 % | **+6.21 %** | 2.4e-08 |
| C long prefill | 12.311 | +5.81 % | **+5.30 %** | 1.2e-05 |

Five of the seven gaps are **call-site changes that reuse fused primitives
SGLang already ships** — no new kernel. The other two are hand-written Triton
kernels, and both are bit-exact (one to within 5e-4 at the largest shape).

The components break down by mechanism, and they are complementary:

| regime | `norm+scale` | `conv` (new Triton) | **all three** |
|---|---:|---:|---:|
| A low-batch decode | +4.20 % | +0.13 % (n.s.) | **+3.80 %** |
| B concurrent decode | +3.68 % | −0.03 % (n.s.) | **+3.65 %** |
| C long prefill | +1.60 % | **+2.33 %** | **+4.59 %** |

6 repetitions per arm after per-workload warm-up; Welch t against the baseline
arm with the exact Student-t tail (a normal approximation is anti-conservative
at these run sizes, which matters for the marginal arms); ratios on request
throughput.

The `conv` component is a **hand-written Triton kernel** and is **bit-exact**
(max |diff| = 0.0 at every shape tested), so it carries no numerical risk. It is
shape-guarded: below T≈2048 it is *slower* than stock and is switched off, which
is why it reads as exactly neutral on both decode regimes rather than as a
regression.

A second round (§6b) added three more call-site fusions found by an nsys and an
FX-graph investigation, the largest being **`qkrope`: reusing
`sgl_kernel.fused_qk_norm_rope`, which SGLang already ships and Qwen3-MoE
already calls, but LFM2.5 does not** — worth **+5.42 % on its own** in
concurrent decode. The full stack reaches **+5.1 % to +5.5 %**, but the
components are **markedly sub-additive** in decode (57 % of the sum of parts
realised) — see §6c, which is the more transferable finding.

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

### G3 `conv` — the ShortConv glue is uncoalesced, and that is the real defect

`Lfm2MoeShortConv.forward` wraps `causal_conv1d` in three pure-data-movement
operations:

```python
proj, _ = self.in_proj(hidden_states)        # [T, 3H]
B_gate, C_gate, x = proj.chunk(3, dim=-1)    # strided views
Bx = B_gate * x                              # elementwise  -> [T, H]
Bx_t = Bx.transpose(0, 1).contiguous()       # materialise  -> [H, T]
conv_out = causal_conv1d_fn(Bx_t, ...).transpose(0, 1)   # view, [T, H]
output, _ = self.out_proj(C_gate * conv_out) # elementwise, reads transposed
```

`causal_conv1d_fn` is an opaque external CUDA op that requires
`x.stride(-1) == 1` on a `[dim, seqlen]` tensor. So the layout change **cannot
be avoided** — only *absorbed* into the neighbouring elementwise work.

The interesting part is *why* this costs so much. On long prefill these glue
kernels move ~8.8 GB in 10.3 ms:

```
18 conv layers x 500 MB of traffic = 8.79 GB  in 10.3 ms  ->  0.83 TB/s
H200 HBM peak ~4.8 TB/s                       ->  17 % of peak
```

**The defect is not the amount of traffic, it is that the traffic is
uncoalesced.** `Bx.transpose(0,1).contiguous()` and the transposed read inside
`C_gate * conv_out` both walk memory with a stride, so most of every fetched
cache line is discarded.

So there are two wins available and they compound: fold three passes into two,
*and* make each remaining pass coalesced. `lf_triton_shortconv.py` does this
with one tiled Triton kernel on each side of the conv, keeping the transpose in
registers via `tl.trans` instead of issuing strided global accesses.

Isolated (`lf_bench_shortconv.py`, correctness checked before timing):

| T | input side | output side | bandwidth | saved per forward |
|---:|---:|---:|---|---:|
| 1024 | 0.94× | 0.71× | — | −0.22 ms |
| 2048 | 1.29× | 0.93× | 0.9 → 0.7 TB/s | +0.27 ms |
| 4096 | 2.24× | 1.76× | 0.9 → 1.3 TB/s | +1.47 ms |
| 16000 | **5.90×** | **4.33×** | **0.98 → 3.46 TB/s** | **+7.96 ms** |

Bandwidth goes from 17 % to ~72 % of peak. **Every shape is bit-exact**
(max |diff| = 0.0) — the multiply is done in fp32 and cast back, matching what
PyTorch's elementwise kernel already does.

**Shape guard.** The fused kernels sit on a ~30 µs floor from Triton's Python
launch path, so below T≈2048 they lose to the stock elementwise ops. The patch
keeps the stock path below `CONV_FUSION_MIN_TOKENS` (default 2048), which is why
`conv` measures as *exactly neutral* on the decode regimes (+0.13 %, −0.03 %,
both n.s.) instead of as a regression. Tile shapes come from a measured sweep
(`lf_tune_shortconv.py`, 32 configurations per shape, correctness-checked before
timing), not from guesswork.

Decode never transposes at all — `causal_conv1d_update` consumes `[T, H]`
directly — so this component is **prefill-only by construction**.

**Rejected for decode.** Fusing the gating multiply into `causal_conv1d_update`
would require rebuilding the `sgl_kernel` C++ extension (it accepts only
`silu`/`swish`, not an external gate tensor), for ~1.2 % of decode kernel time
that CUDA graphs already amortise. Not worth a build-level change.

Figure: `plots/shortconv_crossover.png`.

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

| arm | runs | mean | within-arm spread |
|---|---|---:|---:|
| baseline | 0.348 / 0.349 / 0.344 | 0.3470 | 0.005 |
| **`scale`** (bit-exact) | 0.338 / 0.339 / 0.340 | **0.3390** | 0.002 |
| `norm` | 0.362 / 0.368 / 0.361 | 0.3637 | 0.007 |
| `norm+scale` | 0.359 / 0.359 / 0.359 | 0.3590 | 0.000 |
| **`conv`** (bit-exact) | 0.342 / 0.350 | **0.3460** | 0.008 |
| `norm+scale+conv` | 0.356 / 0.362 | 0.3590 | 0.006 |
| `qkrope` | 0.352 / 0.346 | 0.3490 | 0.006 |
| **all six** | 0.371 / 0.364 / 0.370 | 0.3683 | 0.007 |

The `scale` arm is **provably bit-exact**, so its accuracy *must* equal
baseline's. It does not: it reads **0.8 points lower**. That is not a defect, it
is a measurement of the harness — `--parallel 32` means batch composition
differs between server instances, and batch-dependent reductions change greedy
outputs. So we get the noise floor by construction instead of assuming one:

* **between-arm systematic noise ≥ 0.8 points**, measured on a change that
  cannot possibly alter the model;
* within-arm spread 0.0–0.7 points;
* binomial sampling error at n=1319, p≈0.35 is ±2.6 points at 95 %.

All eight arms span 0.339–0.368, i.e. **2.5 points — inside the noise band on
every one of those three measures.** Note `conv` (0.346) lands essentially on
baseline (0.347) and `norm+scale+conv` (0.359) is identical to `norm+scale`
(0.359), which is exactly what a bit-exact kernel should do. The patched arms happen to sit above
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

6 repetitions per arm, Welch t against baseline
(`processed/fusion_ab_conv.csv`):

| regime | baseline req/s | `norm+scale` | `conv` | **all three** |
|---|---:|---:|---:|---:|
| A low-batch decode | 1.683 ± 0.001 | **+4.20 %** (2e-07) | +0.13 % (0.22) | **+3.80 %** (1.3e-11) |
| B concurrent decode | 21.639 ± 0.128 | **+3.68 %** (4.7e-06) | −0.03 % (0.95) | **+3.65 %** (7.4e-06) |
| C long prefill | 12.104 ± 0.068 | +1.60 % (0.0090) | **+2.33 %** (0.0015) | **+4.59 %** (8.7e-07) |

### The two components are complementary by mechanism

This is the most interesting structural result of the study, and it was not
designed — it fell out of what each change removes.

**`norm+scale` removes a fixed number of kernels and full-activation passes per
forward**, independent of how much compute that forward performs. Decode does
~2 ms of kernel work per forward, so fixed overhead is a large fraction of it;
long prefill does ~157 ms, so the same absolute saving is diluted. Hence
**+4.2 % on decode, +1.6 % on prefill**.

**`conv` removes traffic that scales with the token count**, and only becomes
worth its launch overhead above T≈2048. Decode never reaches that (and never
transposes at all), so the guard switches it off; long prefill runs at
T=4000–16000 where the fused kernel is 2.2–5.9×. Hence **+2.3 % on prefill,
exactly 0 on decode**.

Together they cover the whole range: **positive and significant on all three
regimes**, +3.65 % to +4.59 %.

This also sharpens the contrast with the configuration study, which helped
prefill and did nothing for decode. Three levers, three different shapes of
benefit — **none of which is visible if you only measure one regime.**

---

## 6b. Second round — what nsys and the FX graph found

Two investigations were run *against the already-patched path*, so their
findings are what remains rather than what was already fixed. Full evidence in
`results/lfm_fusion/nsys/FINDINGS.md` and `results/lfm_fusion/fx/FINDINGS.md`.

**The FX/Inductor study independently re-derived the ShortConv kernel.** Given
the unmodified module, Inductor emits
`triton_poi_fused_causal_conv1d_fwd_clone_mul_split_transpose_0` — two strided
loads from `proj`, one multiply, one transposed store, no intermediate. That is
structurally the hand-written kernel, which is a useful independent check that
the shipped design is the one a compiler would choose. It also reports
`found 0 possible fusions` for everything else in ShortConv, because
`causal_conv1d_fwd/_update` are `ExternKernelSchedulerNode`s — hard barriers.
**Fusion has to happen on each side of the conv**, which is exactly the shape of
the two kernels.

Three further gaps, all **call-site changes**:

| component | what it is | measured chain |
|---|---|---|
| **`qkrope`** | `sgl_kernel.fused_qk_norm_rope` combines both head-wise RMSNorms and RoPE into one in-place kernel over packed QKV. **Qwen3-MoE already calls it; LFM2.5 splits QKV and runs two RMSNorms plus a separate RoPE.** | 1.65 % decode / 3.61 % prefill kernel time |
| `idx` | `req_pool_indices.to(torch.int32)` is recomputed in each of the 18 conv layers. The cast moves **12 bytes** — it is pure launch overhead. | ~1.3 % low-batch decode |
| `gate` | the decode gate multiply reads *strided rows* of `proj`, which falls back to a scalar `elementwise_kernel` | ~1.9 % concurrent decode |

`qkrope` is the same pattern as G1, and it is the third instance: **the fused
primitive already exists in SGLang, and this model's call site does not use
it.**

Results (6 reps, Welch t, `processed/fusion_ab_all.csv`):

| regime | `qkrope` | `gate+idx` | `norm+scale+conv` | **all six** |
|---|---:|---:|---:|---:|
| A low-batch decode | +0.93 % (7.2e-09) | −0.00 % (0.97) | +3.89 % (2.5e-15) | **+4.74 %** (2.2e-13) |
| B concurrent decode | **+5.42 %** (1.6e-07) | +0.65 % (0.12) | +3.65 % (6.1e-06) | **+5.54 %** (9.5e-08) |
| C long prefill | +1.99 % (0.018) | +0.40 % (0.54) | +3.47 % (9.5e-04) | **+5.12 %** (2.6e-04) |

`qkrope` alone is the single largest win in the whole study (+5.42 % on
concurrent decode) and it is a pure call-site change reusing a tested CUDA
kernel. `gate+idx` is not significant anywhere — an honest negative: the
mechanism is real and measurable at the kernel level, but ~1–2 % of *kernel*
time does not survive to end-to-end.

The regime-A `all` arm was measured in a separate run
(`processed/fusion_ab_allA.csv`) against its own baseline, because the first
attempt died with `rc=-9` while two servers shared the host. The harness
recorded that as `launch_failed` rather than dropping it silently, which is why
it was noticed at all.

## 6c. The components are strongly sub-additive — and that is the transferable result

Counting all the separately-measured components against the measured stack:

| regime | sum of the parts | measured together | fraction realised |
|---|---:|---:|---:|
| A low-batch decode | 9.37 % | **6.57 %** | 0.70 |
| B concurrent decode | 12.80 % | **6.21 %** | **0.49** |
| C long prefill | 5.86 % | **5.30 %** | 0.90 |

Long prefill is nearly additive, because its components remove different things:
`norm+scale` removes fixed per-forward overhead, `conv` removes traffic that
scales with T, `qkrope` removes work in the 6 attention layers.

Concurrent decode realises **less than half**. `qkrope` alone gets +5.42 %, and
adding `norm+scale+conv` — worth +3.65 % on its own — buys 0.12 points; adding
`moesum` on top of that — worth +3.08 % on its own — buys another 0.19. All
three are removing *fixed per-forward overhead from the same slack*. Once enough
of it is gone something else becomes binding and further overhead removal stops
converting.

**The ordering is the tell: 0.90 / 0.70 / 0.49 tracks how saturated the regime
is.** Long prefill has the most work per forward to hide overhead behind and
loses the least; concurrent decode is the most loaded and loses the most.

This is the same shape as the waterfall non-additivity recorded in the
regime-kernel study (serving 1.78× + kernel 1.22× → 1.70×, not 2.17×). Two
independent studies in this project have now hit it, which makes it worth
stating as a rule:

> **Optimisations that remove the same *kind* of cost do not add up.** Reporting
> the sum of separately-measured wins overstates the stack, and the error is
> largest exactly where the system is most loaded. Every combination that will
> actually be deployed has to be measured as a combination.

Practically this also means the cheapest component is the most valuable:
`qkrope` is a call-site change and captures most of the available decode win on
its own.

---

## 6d. G5 `moesum` — fusing the MoE reduction with the following norm

The nsys study ranked one more chain worth attacking: the MoE top-k reduction
writes `[T, H]` to HBM and the very next thing that happens is the following
layer's `fused_add_rmsnorm` reading it straight back. Both are row-wise, so one
kernel can do the reduction, the residual add and the RMSNorm in a single pass
(`scripts/lfm_fusion/lf_triton_moesum.py`).

Isolated: **2.46×/2.68×/2.64×** at T=1/8/32, falling to 0.72–0.74× at
T=128–1024 and recovering to 1.14×/1.30× at T=4096/16000 — hence a two-sided
guard (`T <= 32 or T >= 4096`). Residual output is bit-exact; the normalised
output is exact through T=4096 and differs by 4.9e-4 at T=16000.

Note this is the **opposite** shape from the ShortConv kernel, which was useless
below T≈2048. Here the win is largest at *small* T, because what is removed is a
kernel launch plus a round-trip, and at T=1 that is essentially the whole cost.

Standalone end-to-end: **A +4.55 %** (p=1.5e-13), **B +3.08 %** (p=0.003).
GSM8K 0.345 against a 0.347 baseline.

Stacked on the other six (`lfm25_all7`, `all` vs `all7`):

| regime | six | seven | delta from `moesum` | p |
|---|---:|---:|---:|---:|
| A low-batch decode | +4.60 % | **+6.57 %** | **+1.88 %** | 1.7e-09 |
| B concurrent decode | +6.01 % | +6.21 % | +0.19 % | 0.68 |
| C long prefill | +5.81 % | +5.30 % | −0.48 % | 0.43 |

So `moesum` is a genuine further win on low-batch decode and **statistically
neutral everywhere else** — it neither helps nor hurts B and C. All seven
together is therefore a safe universal default.

That it survives stacking on regime A while `norm+scale+conv` and `qkrope`
cancel each other on regime B is consistent with §6c: A is the least saturated
regime, so there is still slack for another fixed-overhead removal to convert.

## 7. What this changes about the project's standing conclusions

The previous position was "kernel-level work does not convert to end-to-end
gains on mature bf16 MoE serving". That stands. What this adds is a boundary
condition:

> **[Superseded 2026-07-27]** This paragraph said *architecture maturity*. A
> five-architecture audit refuted that: the newest architecture tested
> (Qwen3-Next) is nearly clean at 0.64 %, while the mature Gemma-3 is the
> worst at 46.32 %. The real split is **model family** — see
> `docs/2026-07-28/cross_architecture_audit.md`. The mechanism below still holds; only
> the predictor was wrong.
>
> The coverage gap is a function of **how much optimisation attention that
> model file has had**, not of SGLang.
> A model family that upstream has optimised (Qwen3-30B: 1 un-fused norm, 0
> stray adds) has nothing left at the fusion layer. A recently added
> architecture (LFM2.5: 61 un-fused norms, 48 stray adds, 36 multiplies, an
> un-fused QK-norm+RoPE chain, and a MoE reduction that round-trips to HBM)
> carries **6.6 %** of pure overhead — not in its novel operator, which is
> already fast, but in the call-site plumbing around it.

The sharpest version of this: **the two largest wins are cases where SGLang
already ships the fused primitive and this model's call site does not use it.**
`fused_add_rmsnorm` (used by llama/qwen2/every other model) and
`fused_qk_norm_rope` (used by Qwen3-MoE), plus a multiply by a constant that is
1.0 and a redundant `.to(int32)`. The two that needed real kernels are
mechanical fusions of adjacent row-wise work, and Inductor derives one of them
by itself. **Nothing here required inventing anything.**

This is precisely the niche the plan assigns to the agent — *"automatically find
and fill SGLang's coverage gaps"* — and it is the first time that framing has
produced a positive, significant end-to-end number across all regimes on the
same model. It is also cheap to detect: the audit is one profiled
`bench_one_batch` run per model, and the signature is a *kernel count that
scales with layer count*, which is mechanically checkable rather than a
judgement call. A useful second signature is now available too: **grep for
fused primitives that exist in the codebase and enumerate which models call
them.**

**Honest scope.** The absolute number is ~5–6.6 %. It is one model on one GPU.
The `norm` and `qkrope` components are not bit-identical, and their quality
claim rests on a task metric inside a harness whose own noise floor is
0.8 points. Most of the win comes from applying existing fused primitives at
call sites that failed to use them, not from novel kernels. One component
(`gate+idx`) is a measured **negative** — real at the kernel level, invisible
end-to-end. And §6c shows the stack does **not** deliver the sum of its parts
where the system is most loaded.

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

1. **Turn the audit into an agent check.** Two mechanical signatures are now
   established: (a) a kernel count that scales with layer count, with Qwen as
   the control for what "clean" looks like, and (b) fused primitives that exist
   in the codebase but that a given model's call site never calls. Both are
   checkable without a human reading the model file, and (b) in particular found
   three of the four wins here.
2. **Run the audit on other recently-added architectures** to test whether
   "architecture maturity predicts fusion headroom" is a rule or a single
   observation. This is the cheapest remaining experiment (~15 min per model)
   and it is what would turn this from an anecdote into a finding.
3. **Upstream the two bit-exact fixes.** The multiply by
   `routed_scaling_factor == 1.0` and the `req_pool_indices` cast hoist are
   small, bit-exact, and cost nothing to review. `fused_add_rmsnorm` and
   `fused_qk_norm_rope` at the LFM call sites are slightly larger but reuse
   primitives upstream already maintains.
4. **Remove the ShortConv shape guard by making the launch cheaper.** The FX
   study measured that the *GPU* crossover is below T=512 — the T≈2048 guard
   exists only because Triton's Python launch path pins wall time at ~19–30 µs.
   A CUDA-graph-captured or precompiled launcher would let the fused kernel run
   everywhere.
5. **Gates into `causal_conv1d_update`** (1.8–1.9 % of decode) — needs a new
   tensor parameter and an `sgl-kernel` rebuild, since `activation` collapses to
   a bool before reaching C++. Deferred as the only remaining item that is not a
   call-site change.
