# PR #31558 reproduction — avoid FLA l2-norm recompilation by token count

> 2026-07-21 · host `aifx-clou000001`, 8×H200 · **sglang v0.5.15.post1** (env `sglang-v515`,
> triton 3.6.0 / torch 2.11 / transformers 5.12.1) · model `Qwen/Qwen3.6-35B-A3B-FP8`
> (hybrid **linear-attention + VLM**, 32 linear + 8 full-attn layers, 256 experts).
>
> First of the "reproduce a recent upstream PR as a fair A = stable, B = stable + PR
> patch" experiments. This is the fast, high-signal smoke-test candidate from the
> plan (Qwen VLM + #31558).

## 0. TL;DR

**Reproduced — mechanism exactly, e2e positively and significantly.**
1. **Mechanism (microbench, exact):** baseline compiles a **new Triton kernel per
   distinct token count** — 10 unique token counts → **10 compiled variants**;
   the patch (`do_not_specialize=["T"]`) compiles **1** and reuses it for all.
2. **End-to-end (server, true-cold Triton cache):** on a VLM served cold and hit
   with **8 different image resolutions**, the patch cuts the first-pass total TTFT
   over the 8 resolutions by **13.7%** (4.005 s → 3.454 s), **Welch t = 20.9,
   p ≪ 0.001**. Per newly-seen resolution, baseline pays a **consistent ~70 ms
   l2norm compile stall** that the patch avoids (+25–33% TTFT on the small/fast
   resolutions where 70 ms is a large fraction).
3. **No regression:** steady-state (warm) TTFT is identical (~170 ms both arms);
   fixed-single-resolution control is identical (both compile once).
4. **Important confound found & controlled:** Triton caches compiled kernels to
   **disk** (`~/.triton/cache`, 2.3 GB / 3370 entries here), so the effect is
   invisible across ordinary server restarts — it only appears with a **fresh
   `TRITON_CACHE_DIR`**. The plan explicitly called for this.

## 1. The patch

Upstream PR #31558 (`optimize: avoid fla l2-norm recompilation by token count`,
merge `42a058c`, 2026-07-18) changes 3 lines in the FLA l2norm kernel. Its file path
on `main` (`python/sglang/kernels/ops/attention/fla/l2norm.py`) **does not exist** in
v0.5.15.post1 (PR was written after a later refactor), but the identical un-optimized
code lives at `python/sglang/srt/layers/attention/fla/l2norm.py`. I **manually ported
the exact diff** (cleaner than a cherry-pick that would drag in unrelated refactors):

```diff
-@triton.jit
+@triton.jit(do_not_specialize=["T"])
 def l2norm_fwd_kernel(
     x, y, eps,
-    NB: tl.constexpr,
-    T: tl.constexpr,
+    T,
     D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
 ):
 ...
 if D <= 512:
-    NB = triton.cdiv(T, 2048)
     def grid(meta): ...
-    l2norm_fwd_kernel[grid](x, y, eps, NB=NB, T=T, ...)
+    l2norm_fwd_kernel[grid](x, y, eps, T=T, ...)
```

`T` = flattened token count. As a `tl.constexpr` it is a compile-time specialization
key, so every distinct token count JIT-compiles a fresh kernel. In a VLM, different
image resolutions → different token counts → a fresh cubin each time.
`do_not_specialize=["T"]` makes `T` a runtime scalar → one kernel serves all.

Baseline/patched l2norm files saved verbatim under
`patches/l2norm_v0.5.15.post1_{baseline,pr31558}.py`.

## 2. Mechanism microbench (v46)

`scripts/run_v46_l2norm_recompile_microbench.py` calls `l2norm_fwd` on 10 distinct
token counts and counts Triton's compiled-kernel cache (`JITFunction.device_caches`,
the triton-3.6 API) before/after.

| arm | unique token counts | **kernels compiled during sweep** | cached after |
|---|---|---|---|
| baseline | 10 | **10** | 11 |
| patched | 10 | **0** | 1 |

Exactly reproduces the upstream "6 cubins → 1 cubin" claim (we used 10 shapes → 10 vs 1).
Per-call latency including compile also dropped ~7× (0.43 ms → 0.06 ms) on first touch.
Raw: `results/2026-07-21_v46_l2norm_recompile/microbench.jsonl`.

## 3. End-to-end server A/B (v47)

`scripts/run_v47_pr31558_server_ab.py`: for each arm, place the right l2norm.py,
launch a **fresh** sglang server, wait until ready, then fire streaming VLM requests
(OpenAI `/v1/chat/completions`, capturing true TTFT from the first content token).

- **dynamic** mode: cycle 8 resolutions `{360p,720p,1080p,512x512,640x800,768x1024,
  900x1200,1024x1024}`, 1 random image each, `max_tokens=8`.
- **fixed** control: single resolution (720p), 3 rounds.
- **true-cold** variant: fresh `TRITON_CACHE_DIR` per server start (the correct way
  to measure compile cost) — 3 cold-start repeats per arm.

### 3.1 True-cold first-pass total TTFT (headline)

Sum of TTFT over the first pass across all 8 distinct resolutions, from a cold server
+ cold Triton cache:

| arm | n | mean | std | raw (s) |
|---|---|---|---|---|
| baseline | 3 | **4.005 s** | 0.035 | 3.97, 4.04, 4.00 |
| patched | 3 | **3.454 s** | 0.029 | 3.44, 3.44, 3.49 |

**−13.7%, Welch t = 20.9, df ≈ 3.9 → p ≪ 0.001.** Both arms extremely low-variance.

### 3.2 Per-resolution first-seen TTFT (the compile stall)

| resolution | baseline (ms) | patched (ms) | Δ (stall avoided) |
|---|---|---|---|
| 360p (1st request) | 1690.2 | 1625.3 | +64.9 (+3.8%) |
| 720p | 277.4 | 204.3 | **+73.2 (+26.4%)** |
| 1080p | 787.5 | 736.8 | +50.7 (+6.4%) |
| 512×512 | 227.7 | 152.9 | **+74.8 (+32.9%)** |
| 640×800 | 240.1 | 171.7 | **+68.5 (+28.5%)** |
| 768×1024 | 251.3 | 176.7 | **+74.6 (+29.7%)** |
| 900×1200 | 275.2 | 206.0 | **+69.2 (+25.2%)** |
| 1024×1024 | 255.6 | 180.8 | **+74.8 (+29.3%)** |

The compile stall is a **strikingly consistent ~70 ms** per newly-seen resolution —
i.e. the cost of one l2norm JIT compile — which the patch pays once and reuses. It is
a large fraction (+25–33%) of TTFT on the small/fast resolutions and a small fraction
on 360p (dominated by generic first-request init) and 1080p (dominated by prefill of
many image tokens).

### 3.3 Steady-state & fixed control (no regression)

- Warm (rounds 1–2, dynamic): baseline 170.3 ms vs patched 170.2 ms per request — identical.
- Fixed single resolution (720p), first request: baseline 905.6 ms vs patched 902.7 ms
  — identical (both compile the single shape once; nothing to save). This is the
  correct negative control: the benefit is specific to **multiple distinct shapes**.

Raw: `results/2026-07-21_v47_pr31558_server/server_ab.jsonl` (original dyn+fixed),
`server_ab_coldcache.jsonl` (3× true-cold per arm), `analysis_coldcache.txt`.

## 4. Honest calibration vs the upstream headline

Upstream reported mean TTFT −40% (rate=1) / −26.5% (burst) on 2×H200 Qwen3-VL. We got
−13.7% on the *aggregate first-pass* and +25–33% on *individual newly-seen small
resolutions*. The difference is workload/metric composition, not a contradiction:
- Our aggregate is diluted by the first request (360p, +3.8%: generic init dominates)
  and 1080p (+6.4%: prefill-of-many-tokens dominates). Upstream's resolution mix and
  their mean-TTFT-over-a-stream metric weight the compile-bound small-shape requests
  more heavily.
- The **per-new-shape compile cost (~70 ms) is the physically real, reproducible
  effect**, and it is exactly the quantity the patch removes.

## 5. Verdict

PR #31558 **reproduces cleanly and positively** on 8×H200 / v0.5.15.post1:
- mechanism is exact (10 compiles → 0),
- it converts to a **significant, low-variance e2e TTFT win in the intended
  dynamic-shape VLM + cold-cache regime** (−13.7% aggregate, t≈21; ~70 ms per new
  resolution),
- with **no regression** in steady state or fixed-shape serving.
This is a good headline candidate for "an agent-authored kernel/JIT fix that lands on
real end-to-end latency," with the honest caveat that the win is specific to
dynamic-shape workloads with a cold Triton cache (the plan's exp-1 scenario); the
fixed-shape control correctly shows ~0.

## 6. Environment / reproduce

- env `sglang-v515` (conda clone of `sglang-dev` + upgraded deps): sglang
  v0.5.15.post1 editable at `/home/t-jialianggu/work/sglang-v515`, transformers
  5.12.1, kernels 0.14.1, flashinfer 0.6.12, triton 3.6.0, torch 2.11, CUDA 13.0,
  `CUDA_HOME=$CONDA_PREFIX`.
- `PYTHONPATH=/home/t-jialianggu/work/sglang-v515/python` needed for `bench_serving`
  (benchmark pkg excluded from the editable wheel).
- **Always clear `TRITON_CACHE_DIR` to measure compile-related effects** — the on-disk
  cache masks them across restarts.
- Scripts: `scripts/run_v46_l2norm_recompile_microbench.py`,
  `scripts/run_v47_pr31558_server_ab.py`. Patches in `patches/`.
