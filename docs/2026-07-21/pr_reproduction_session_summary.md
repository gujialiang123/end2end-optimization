# Upstream-PR reproduction line — session summary (2026-07-21)

> Host `aifx-clou000001`, 8×H200. Env `sglang-v515` (sglang **v0.5.15.post1**,
> the current stable release). Goal (from the user's proposal): reproduce recent
> upstream SGLang PRs as fair **A = stable release, B = stable + single PR patch**
> A/Bs, to show an agent-authored kernel/runtime fix lands on real end-to-end perf.
> Two PRs reproduced cleanly and positively; a third assessed as much higher effort.

## Results at a glance

| PR | area | model | headline result | verdict |
|---|---|---|---|---|
| **#31558** | JIT / compile-side (FLA l2norm recompile by token count) | Qwen3.6-35B-A3B-FP8 (linear-attn VLM) | mechanism exact (10 compiles → 0); **cold-start VLM TTFT −13.7%**, Welch t≈21; ~70 ms compile stall avoided per new image resolution | ✅ reproduced, positive |
| **#29007** | communication / memory (MoE TP allreduce → NCCL symmetric memory) | DeepSeek-V4-Flash-FP8 (294 GB, TP8) | **TPOT −9.2% / E2E −10.6% / throughput +10.6%** at c1; +5–7% through c16 | ✅ reproduced, positive (≥ upstream) |
| **#31438** | CPU critical path (parallelize VLM multimodal preprocessing) | Qwen3.6-35B-A3B-FP8 | **+8.5–14.5% image-burst throughput** at default 2 workers; **bit-identical** greedy output | ✅ reproduced, positive |
| #30514 | GPU CUDA kernel (Q8KV8 FP8 sparse-MLA prefill) | DeepSeek-V3.2 (689 GB) | not attempted — needs an **sgl-kernel CUDA rebuild** (modifies `csrc/.../kernel.cuh`), not a pure-Python toggle; + accuracy gate | ⏸ deferred (high effort) |

These two cover two of the three evidence categories the plan wanted: **CPU/JIT
runtime** (#31558) and **communication/memory layout** (#29007). The third
(GPU kernel/backend, #30514) is the CUDA-rebuild candidate.

## Why these two were clean to reproduce

Both PRs are **pure-Python** changes whose pre-patch code still exists verbatim (or
near-verbatim) in v0.5.15.post1, so the A/B is a clean **file-swap in one env** — no
sgl-kernel rebuild, no "stable vs a week of main" confound:
- #31558: 3-line change to `l2norm.py` (ported by hand; PR's `main` path didn't exist).
- #29007: 5-file change; cherry-picked with 2 small conflicts resolved (kept v0.5.15's
  extra `_DpGatheredBufferWrapper` fields and its `post_reorder_triton_kernel` call,
  applied only the symmetric-memory wraps).

Baseline/patched file snapshots + upstream + ported diffs are under `patches/`.

## Key methodology points (honest calibration)

- **#31558 confound found:** Triton caches compiled kernels to disk
  (`~/.triton/cache`), masking the recompile cost across server restarts. The effect
  only appears with a fresh `TRITON_CACHE_DIR` — with that control, the −13.7% is
  rock-solid (both arms std ≈ 0.03 s, t≈21). Without it you'd wrongly conclude ~0.
  Also, n=3 first showed a scary decode-b8 −8.8% that was pure noise (n=8 → +0.9%,
  p=0.93) — signal-vs-noise needs repeats + a t-test.
- **#29007 is the cleanest possible A/B:** identical flags on both arms
  (`--enable-symm-mem`), same model, same workload; the only difference is whether the
  MoE output tensor lands in the symmetric pool. The win (−9.2% TPOT) is purely the
  faster all-reduce path, and it holds across concurrency.
- **Env engineering** was the real cost for #29007: the `nccl_allocator` JIT
  extension (`--enable-symm-mem`) needed CUDA+NCCL headers/libs wired onto
  `CPATH`/`LIBRARY_PATH` and a `libnccl.so` symlink before DeepSeek-V4 would serve at
  all (3 successive link errors fixed). Documented for reuse.

## Artifacts (all pushed to main)

- Docs: `docs/2026-07-21/pr31558_fla_l2norm_recompile_repro.md`,
  `docs/2026-07-21/pr29007_dsv4_symm_mem_allreduce_repro.md`, this summary.
- Scripts: `run_v46_l2norm_recompile_microbench.py`, `run_v47_pr31558_server_ab.py`,
  `analyze_v45_server_ab.py`, `run_v48_dsv4_pr29007_ab.py`.
- Raw: `results/2026-07-21_v46_l2norm_recompile/`, `..._v47_pr31558_server/`,
  `..._v48_dsv4_pr29007/`.
- Patches: `patches/l2norm_v0.5.15.post1_{baseline,pr31558}.py`,
  `patches/pr29007/{baseline,patched}/`, upstream + ported diffs.

## Suggested next steps (for review)

1. **#30514 (DeepSeek-V3.2, strongest agent evidence)** — feasible but heavier:
   requires rebuilding sgl-kernel from source (CUDA `.cuh` change) on both arms, the
   DSA sparse-prefill serving path (8K+ context to activate), and a GSM8K accuracy
   gate (the PR is FP8-vs-BF16 query, not bit-exact). Best done as a dedicated run.
2. **#31438 (VLM multimodal preprocessing parallelization)** — likely another clean
   Python-side reproduction on the same Qwen3.6 VLM already downloaded; good
   complement to #31558 for the "CPU critical path" story.
3. Add a **bit-exact greedy A/B** for #29007 (both servers up on separate ports) to
   nail the semantics-unchanged claim, if wanted for the writeup.

## Environment (both PRs)

env `sglang-v515`: sglang v0.5.15.post1 editable at `/home/t-jialianggu/work/sglang-v515`
(git worktree), transformers 5.12.1, kernels 0.14.1, flashinfer 0.6.12, triton 3.6.0,
torch 2.11, CUDA 13.0. `sglang-dev` (v0.5.12) left intact for the earlier v44/v45 work.
Models under `/home/t-jialianggu/work/models/` (gitignored): Qwen3.6-35B-A3B-FP8 (35 GB),
DeepSeek-V4-Flash-FP8 (294 GB).
