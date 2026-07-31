---
name: fusion-gap-hunting
description: Find kernels a model runs unfused that the framework already ships a fused implementation for, by combining a static AST scan of the dispatch code with an FX trace sweep or an operator-level profile audit.
version: 2
stage: [1, 2]
inputs:
  - framework_src: path to the serving framework checkout (e.g. /home/.../sglang)
  - model: model key runnable by the bench harness
  - gpu: a single free GPU id
outputs:
  - fusion_gap_candidates.json    # static scan, cheap, high recall / low precision
  - gap.json                      # FX rank/dtype sweep, no profiler, any backend
  - audit.json                    # measured per-kernel gap counts and time share
  - verdict: confirmed | refuted, with the numbers that decided it
triggers:
  - "a model family is newly supported, or is not the framework's primary user base"
  - "before deciding to hand-write a kernel for a model — run this first, it is cheaper"
  - "an operator audit shows a large 'elementwise'/'other'/'norm' bucket"
depends_on: [pytorch-profiling, e2e-bench-runner, noise-aware-scoring]
---

# fusion-gap-hunting

## WHEN

Run this **before** any attempt to write a new kernel for a model. It is the
cheapest high-yield check available, and in the 2026-07 study it found a **2.13×**
end-to-end win in a model file that had been in the framework for months.

Concretely, trigger on:

- a model whose family is **not** the framework's primary user base (see WHY —
  this, not architecture novelty, is the predictor);
- an operator-level profile where `elementwise`, `other` or `norm` buckets are
  more than a few percent of kernel time;
- any newly added architecture, *before* concluding "the hot path is already
  optimised" — that conclusion is per-model, never framework-wide.

## WHY

The framework accumulates fused CUDA kernels over time. Each new model file has
to *opt in* to them. Nothing enforces that it does, and nothing fails when it
doesn't — the model is still **correct**, just slower. So the gap is invisible
to tests and only shows up in a profile.

**Measured evidence (2026-07-27/28, sglang, 1× H200).** Three independent hits
of the same pattern, none of which required writing a kernel:

| model | primitive that already existed | who used it | who didn't | e2e |
|---|---|---|---|---|
| LFM2.5 | `fused_add_rmsnorm` | llama, qwen2, nearly all | `Lfm2MoeDecoderLayer` | +2.35 % |
| LFM2.5 | `fused_qk_norm_rope` | Qwen3-MoE | `Lfm2MoeAttention` | +5.42 % |
| **Gemma-3** | `gemma_rmsnorm` | **`GemmaRMSNorm`, same file, ~100 lines above** | `Gemma3RMSNorm.forward_cuda` | **+112.8 %** |

**The single most important calibration:** in the same study, hand-writing two
Triton kernels (with tile sweeps, correctness gates and shape guards) consumed
most of the effort and returned **~6 %**. Finding one un-called primitive and
changing ~10 lines returned **2.13×**.

> **Leverage is in finding the right place, not in writing something clever.**
> Always run this skill before proposing to write a kernel.

**What predicts a gap — refuted and corrected.** The initial hypothesis was
"newer architecture → more gaps". A six-model audit **refuted** it:

| model | family | size | gap % of decode kernel time |
|---|---|---:|---:|
| Gemma-3-1B | Google | 1 B | **46.32 %** |
| LFM2.5-8B | Liquid | 8 B | **11.31 %** |
| Qwen3-Coder-Next | Qwen | ~80 B | 0.64 % |
| Qwen3-0.6B | Qwen | 0.6 B | 0.57 % |
| Qwen3-30B | Qwen | 30 B | 0.23 % |
| Qwen3-32B | Qwen | 32 B | 0.05 % |

The *newest* architecture tested is 72× cleaner than a *mature* one. Four Qwen
models spanning dense / MoE / linear-attention and 0.6 B–80 B are all under 1 %;
both non-Qwen models are over 11 %. Architecture type, architecture age and
model size are each ruled out by this table (note Qwen3-0.6B is **smaller** than
Gemma-3-1B and 81× cleaner).

> **Predictor: how much optimisation attention that specific model file has
> received — which tracks the family's prominence in the framework's user base.**
> Actionable form: **audit models from families that are not the framework's
> primary users.**

## HOW

Two techniques. Use **both** — they have opposite error profiles and the
combination is what works.

### Step 1 — static scan (seconds, no GPU, high recall / low precision)

Three scans. None needs the model to run; `impl/scan_fusion_gaps.py` runs all
three.

**1a. Backends that fall through to the reference implementation.**

```bash
grep -rn "def forward_cuda" -A 2 "$SRC"/python/sglang/srt/layers/*.py \
  | grep -B1 "return self.forward_native" | grep "def forward_cuda"
```

A hit means: this op has a dispatch mechanism, other backends have real
implementations, and CUDA is falling back to eager PyTorch. **Then check whether
the sibling backends (`forward_cpu`, `forward_hip`, `forward_npu`) call a fused
kernel.** If they do and CUDA does not, that is the Gemma-3 shape exactly.

**1b. Models that never name a primitive their peers use.**

```bash
# example: who has q_norm + rotary but never calls the fused version?
for f in "$SRC"/python/sglang/srt/models/*.py; do
  grep -q "q_norm\|q_layernorm" "$f" && grep -q "rotary_emb" "$f" \
    && ! grep -q "fused_qk_norm_rope" "$f" && echo "$f"
done
```

**1c. Backends that reach the kernel only for *some* inputs (AST, not grep).**

Scan 1a only catches a body that hands *everything* to `forward_native`. The
quieter and more common shape is a body that calls a real kernel on one branch
and falls back on another, gated by a property of the input:

```python
def forward_cuda(self, x, residual=None):
    ...
    if x.dim() == 2:
        return gemma_rmsnorm(x, self.weight.data, self.eps)   # fused
    return self.forward_native(x)                             # eager
```

This is Gemma-3 on current `main`, and it is invisible to both earlier scans:
the source *does* name the primitive, and an aggregate profile shows the op as
fused *and* eager depending on the caller. Grep cannot see it either, because
what matters is which branch each call sits in — so this scan parses the AST,
records every call with the `if` conditions guarding it, and reports a guard
only when the region where it is **false** reaches no kernel of its own.

```bash
python .github/skills/fusion-gap-hunting/impl/scan_fusion_gaps.py \
    --src "$SRC" --out fusion_gap_candidates.json
```

Measured on sglang `main` (2026-07-31): 227 `forward*` methods under `layers/`,
20 where a kernel call and a `forward_native` call coexist, **4** where an
input-property guard leaves the fall-back path unfused. One is
`Gemma3RMSNorm.forward_cuda` — **the scan finds the case that took profiling
plus manual code reading to find originally.** The other three are `forward_cpu`
/ `forward_npu` / `forward_xpu`, which this hardware cannot confirm.

The scan is backend-agnostic by construction, so **the same scan applies to a
new backend** (Maia, XPU) with no changes — the pattern it looks for is "this
dispatch has a kernel that some shapes cannot reach", not anything CUDA.

### Step 2 — confirm the candidate

Two arbiters. Prefer the FX sweep when the target is not CUDA, or when the
question is only "is it fused"; use the operator audit when you need time share.

**2a. FX rank/dtype sweep (seconds, no profiler, hardware-agnostic).**

```bash
python scripts/fx_fusion/fx_dispatch_gap_detector.py --out gap.json
```

Traces the module once per input shape and reports, per shape, whether the graph
contains a **registered kernel** (an op whose namespace is outside
`aten`/`prims`) or expanded pointwise math. The verdict is the *asymmetry*:

```
rank=2 [64, 1152]        FUSED  kernel: ['sgl_kernel.gemma_rmsnorm.default']
rank=3 [1, 64, 1152]     EAGER  markers: ['mean','pow','rsqrt']
rank=4 [1, 64, 4, 1152]  EAGER  markers: ['mean','pow','rsqrt']
→ DISPATCH GAP FOUND
```

**All shapes expanded means no gap** — the framework simply has no kernel here,
and there is nothing to miss. Validate this direction with a negative control on
a tree where `forward_cuda` delegates unconditionally; if that also reports a
gap, the detector is reporting "saw eager math" rather than "saw a kernel some
shapes cannot reach", which is the failure this check exists to catch.

Judge fused on the **presence of a registered kernel**, never on the absence of
`pow`/`mean`/`rsqrt`. Absence is also produced by a graph break, an unrelated
implementation, or an op outside whatever marker list you wrote.

**2b. Operator audit (one profiled run, ~10 min, gives time share).**

```bash
python scripts/lfm_fusion/lf_audit.py --model <key> --regime A_low_batch_decode --gpu <id>
```

Runs `bench_one_batch --profile` with **CUDA graphs disabled** (so every
operator appears as its own kernel), buckets kernel time by name, and counts
*fusion-gap signatures*. Report **per layer**, not absolute — the counts are
structural, so a deeper model naturally issues more.

The decisive signature for eager norms is the reduction kernel:
`reduce_kernel<...MeanOps...>` alongside `rsqrt_kernel` and `pow_tensor_scalar`
at equal call counts is an RMSNorm decomposed into primitives.

### Step 3 — confirm the mechanism, then fix

Fix by calling the primitive the sibling/peer already calls. Then **re-run the
audit with the fix applied** (Step 5).

## OUTPUT CONTRACT

`audit.json`:

```jsonc
{
  "model": "gemma3", "regime": "A_low_batch_decode",
  "arch": "dense + sliding-window attention",
  "layers": 26, "tp": 1,
  "stages": {
    "decode": {
      "total_kernel_us": 3814.1,
      "buckets":      [{"bucket": "norm", "pct": 15.98, "calls": 157}],
      "fusion_gaps":  [{"gap": "eager_norm_decomp",
                        "calls": 157, "calls_per_layer": 6.04,
                        "pct_of_kernel_time": 15.98,
                        "removable_by_fusion": true}],
      "top_kernels":  [{"kernel": "...", "total_us": 741.8, "calls": 317,
                        "bucket": "layout_copy", "gap": "layout_copy"}]
    }
  }
}
```

`calls_per_layer` is the comparable number across models. `pct_of_kernel_time`
is **kernel time**, never end-to-end — keep them separate in every report.

## FAILURE MODES

**Static scan has false positives — always confirm by measurement.**

- *"Doesn't name the primitive" ≠ "runs eager."* Qwen3-0.6B does not mention
  `fused_qk_norm_rope`, but the audit showed it already calls
  `fused_qknorm_warp` via a helper. The grep was wrong; the profile was right.
- *"Primitive exists in the repo" ≠ "exists on this platform."* `QuickGELU`
  looks identical to the Gemma-3 case — `forward_cuda` returns
  `forward_native` while `forward_hip` calls `gelu_quick`. But `gelu_quick` is
  imported only under `elif _is_hip` and is **not in the CUDA build** of
  `sgl_kernel`. Check the import guard and the actual module contents.
- *"A shape guard sits in front of a kernel" ≠ "other shapes run eager."*
  `Ernie4_5_VLRotaryEmbedding.forward_cuda` guards
  `triton_ernie45_rope_fused_inplace` on `positions.ndim == 2`, then calls a
  **different** kernel for rank 1, and only reaches `forward_native` when that
  second kernel is absent from the build. Scan 1c flagged it until the scan
  learned to propagate early returns and check whether the region where the
  guard is *false* reaches a kernel of its own. Any guard-based scan needs that
  check or the whole dispatch-chain idiom reads as a gap.
- *Seeding kernel names from one import path under-reports.* Harvesting only
  `from sgl_kernel import` found 63 names; adding `sglang.kernels.*` found 629,
  and the difference is exactly the in-tree Triton kernels the newer model files
  call.
- `NewGELU` carries an explicit `# TODO: Implement the CUDA kernel` — a known
  absence, not an un-called primitive.

Precision of scan 1a in the 2026-07 run: **1 real hit out of 3 candidates.**
That is fine — candidates are cheap, the audit is the arbiter.

**Traps when implementing the fix** (each cost real debugging time):

1. **dtype mismatch fails silently.** `nn.Parameter(torch.zeros(dim))` is fp32
   while activations are bf16. Fused kernels typically require matching dtypes
   and **return NaNs rather than raising**. Cache the cast per module, keyed on
   dtype so a re-cast module is not served a stale buffer.
2. **Rank > 2 inputs.** `q_norm`/`k_norm` receive `[tokens, heads, head_dim]`.
   A rank-2 guard silently leaves those on the slow path — 2 of 6 norms per
   layer, the difference between 1.56× and 2.13×. Flattening is exact when the
   op reduces over the last dimension only.
3. **Dispatch bound at construction.** `MultiPlatformOp.__init__` assigns
   `self._forward_method = self.dispatch_forward()`. Replacing the class method
   afterwards does nothing for already-constructed modules — patch the
   constructor too. (Only affects monkeypatch-style A/B; a source patch is fine.)
4. **Lazy model imports.** Model modules load long after `sitecustomize`. Use a
   `sys.meta_path` finder that patches when the target module finishes
   executing; a timer is a race.

## VERIFICATION DISCIPLINE

Non-negotiable, in this order. Skipping any of these has produced a wrong
conclusion in this project before.

1. **Correctness before performance.** Never time a variant that failed its
   gate.
2. **Choose a gate the model can actually satisfy.** Token identity is
   *structurally unavailable* for top-k routed MoE models: expert selection is a
   discrete argmax, so a bf16-level perturbation flips which expert fires and
   the output changes discontinuously. Use a task metric instead.
3. **Calibrate the noise floor with a bit-exact arm.** Include a change that
   *provably* cannot alter the result (e.g. skipping a multiply by 1.0). In this
   study that arm moved GSM8K by **0.8 points** — that is the harness noise,
   measured rather than assumed. Any claim smaller than it is not a claim.
4. **Verify the patch actually applied.** Print a marker and assert on it, or a
   silently no-op patch scores as "identical to baseline" and you will believe
   it.
5. **Re-run the audit with the fix applied.** This is the mechanical closure —
   the gap signature should go to zero. It is also what catches an *incomplete*
   fix: 52 residual eager calls (exactly 2.00/layer) exposed the rank-2 bug.
6. **Multiple repetitions + exact Welch t.** A normal approximation is
   anti-conservative at n≈6 and will overstate significance on the marginal
   arms. Use `scipy.stats.t.sf`.
7. **Measure combinations as combinations.** Optimisations that remove the same
   *kind* of cost do not add up: measured realisation of the sum of parts was
   **0.90 / 0.70 / 0.49** across three regimes, and that ordering tracks how
   saturated the regime is. Reporting a sum overstates the stack, worst exactly
   where the system is most loaded.
8. **Treat the audit's `pct_of_kernel_time` as an upper bound, never as an
   expected gain — especially for the *second* gap in the same model.** After
   the Gemma-3 norm fix landed, the audit priced the next gap (52 standalone
   residual adds, 2.00/layer) at **3.00 % of decode kernel time**. Implementing
   it delivered **−0.09 % and +0.44 %, neither significant**. The dominant
   fixed overhead was already gone, so the remaining same-class overhead had
   nothing left to convert against.

   Practical rule: **after a large fix lands, re-audit to find the next gap, but
   re-measure end-to-end before believing it — and before enlarging a PR for
   it.** In this case the data retroactively justified keeping the change out of
   the PR.

## PR READINESS

If the fix is going upstream, add:

- apply it as a **real source patch**, not a monkeypatch, and A/B *that*
  (`lf_e2e.py` supports an `@src:<tree>` arm via PYTHONPATH). The source patch
  measured *better* than the monkeypatch — 1.996× vs 1.754× on one regime;
- run the framework's own tests **patched and unpatched** and compare. Identical
  results, including identical pre-existing failures, is the evidence of no
  regression;
- if the op had no test, add one — and **mutation-test it** by reintroducing
  each trap to confirm it fails. An added test that cannot fail proves nothing;
- check whether upstream `main` already fixed it, and whether the fall-through
  might be a deliberate accuracy choice. For Gemma-3 both were checked: still
  broken on `main`, and the two classes' reference implementations are
  byte-identical, so it is not an accuracy trade-off;
- **re-check `main` again right before you post, and re-scope to whatever gap is
  actually left.** Upstream moves while you measure. Between our audit
  (`a82ead53b`) and our submission, #32383 landed the 2-D and residual halves of
  exactly this fix. Our `2.13× / 2.00× / 1.52×` headline silently became a claim
  on someone else's work — the honest number was the *increment* over a
  main-equivalent baseline (`+36.6% / +24.5% / +7.3% n.s.`), roughly a third of
  what we had been about to claim.

  Two habits fall out of this, and both are cheap:

  - **Diff the specific function against `origin/main` on the day you post**, not
    the day you found it. A `git log -p <file>` on the touched file is enough.
  - **When you cannot run current `main` directly, build a baseline arm that
    reproduces its coverage** rather than reusing your old baseline. We could not
    run `main` locally (it needs `transformers==5.12.1`; our envs had 4.57.1 and
    5.6.0, conda clone failed, cross-env toolchains broke the JIT link), so we
    added a `norm2d` arm that fuses exactly the 2-D case and leaves higher rank
    eager. That arm *is* main's behaviour, so the A/B against it is the increment
    — and it costs one component flag, not a working `main` install.

  Corollary for reporting: if a regime comes back `p = 0.053`, it is **not** a
  result. Print it with the verdict attached and keep it out of the headline.

- **Re-check that the opportunity still exists, not just that your fix still
  applies.** These are different failures and we hit both within four days.
  Upstream #32383 landed the half of the Gemma-3 fix we were about to claim --
  our *fix* was partly redundant. Then a Triton 3.5.1 -> 3.6 bump erased a
  1.37-1.74x MoE tuning headroom entirely: 0 of 19 buckets still cleared the
  threshold, and 3.6's untuned path beat 3.5.1's *tuned* path by 5.2%. Nothing
  was measured wrong either time; the ground moved.

  (Caveat added later: the Triton half of that example was retracted -- the 3.6
  "baseline" turned out to be our own tuned config, loaded via the cross-version
  config fallback because the experiment imported the PR branch worktree. The
  #32383 half stands. See docs/2026-07-29/RETRACTION_triton36_baseline_contamination.md.
  The rule below is still worth following; just note that one of its two
  motivating cases was a measurement bug, not an expiring opportunity.)

  So every recorded opportunity needs the toolchain version stamped on it, and
  anything older than a few days gets re-measured before it is claimed. In
  practice this is cheap -- one microbenchmark of the default path on the new
  version answers it.

  And it reorders the candidate list: **check whether upgrading a dependency
  already gets the win before proposing hand-tuning.** Here the upgrade was
  worth +29.8% end-to-end, free and zero-maintenance, against +23.3% for a
  hand-tuned config that then expired.

## ROADMAP

- ~~Automate scan 1a/1b into `impl/scan_fusion_gaps.py`~~ — done, plus scan 1c
  (guarded fall-through) and the per-platform import-guard check that rejects
  `QuickGELU` automatically.
- **Second gap class, not yet automated: fusion the compiler could do but a
  graph barrier prevents.** Gemma-3 slices q and k out of one qkv tensor and
  norms each separately, which Inductor emits as two kernels; two *independent*
  norms fuse laterally into one, so the slice is the barrier. Merging them is
  1.94× at 8–32 tokens, 1.15× at 2048, and **0.56× at 4096**, numerically
  identical throughout. Neither scan here can see it — it is a property of the
  traced graph, not of the source. The rule to implement: *≥2 chains with the
  same signature, whose inputs trace back to slices of one producer, that
  Inductor placed in different kernels.* Note this class is the one where a
  compiler beats a hand-written kernel: the merge needs a per-head weight, which
  `sgl_kernel.gemma_rmsnorm`'s 1-D weight cannot express but `torch.compile`
  writes in three lines.
- Drive scan 1c's shape sweep from **observed** shapes: hook the real model's
  forward and record what each module is actually handed, instead of sweeping
  ranks 1–4 and hoping the interesting one is in there.
- Extend the audit's `GAP_SIGNATURES` beyond norms (activation decompositions,
  attention epilogues).
- Sample more non-primary-family models. The family conclusion currently rests
  on **only two** non-Qwen families, which is its weakest link.
