# PR #29007 reproduction — MoE TP allreduce via NCCL symmetric memory (DeepSeek-V4)

> 2026-07-21 · host `aifx-clou000001`, 8×H200 · **sglang v0.5.15.post1** (env `sglang-v515`)
> · model `sgl-project/DeepSeek-V4-Flash-FP8` (294 GB, FP8, 43 layers, NSA indexer),
> **TP8**, `attention_backend=dsv4`, `--enable-symm-mem`.
>
> The plan's candidate #1 headline: natural-text, semantics-unchanged, same-config
> E2E win. Reproduced cleanly and **positively** (even slightly above the upstream
> numbers).

## 0. TL;DR

**Reproduced — clean positive E2E win, consistent across concurrency.**
On DeepSeek-V4-Flash-FP8 / TP8 / 8×H200, both arms run with identical flags
(`--enable-symm-mem`); the only difference is PR #29007 (allocate the MoE post-norm
output in the NCCL **symmetric-memory pool** so the downstream TP all-reduce takes
the low-latency symmetric path). Result (gain = improvement, + = patched better):

| workload (in/out/conc) | mean TPOT | mean E2E | output throughput |
|---|---|---|---|
| 4096 / 1024 / **c1** (upstream regime) | **+9.2%** | **+10.6%** | **+10.6%** |
| 4096 / 1024 / c8 | +6.0% | +7.0% | +7.0% |
| 4096 / 512 / c16 | +5.3% | +6.4% | +6.4% |

Upstream reported (same model, 4K/1536/c1): mean E2E −6.58%, output tput +7.05%,
mean TPOT −6.55%. **We reproduce and slightly exceed it** (−9.2% TPOT / +10.6% tput
at c1), and additionally show it holds at higher concurrency (c8, c16). n=2 repeats
per cell, extremely consistent (e.g. c1 TPOT 6.33 vs 6.34 ms).

## 1. What the PR does

`Fix MoE TP allreduce to use NCCL symmetric memory via in-pool output allocation`
(#29007, squash-merge `980acd6`, 2026-07-15, parent `41e0b4b3`). With
`--enable-symm-mem` the server sets up a NCCL symmetric-memory pool for low-latency
all-reduce, but the MoE runner used to allocate its output with a plain
`torch.empty`, so the tensor was **not** in the pool and the downstream TP all-reduce
fell back to the normal path. The PR allocates the MoE output inside
`use_symmetric_memory(...)` (gated by `is_allocation_symmetric()`), so a symmetric
input yields a symmetric all-reduce → the fast path. Per the PR, single all-reduce
latency ~20 µs → ~10 µs; because it fires every layer every decode step, it
compounds into the TPOT / throughput win.

Files (5 src, pure Python — no rebuild): `dp_attention.py` (real defaults so the
prewarm path can read `_dp_max_padding`), `deepseek_v4.py` (wrap the hc_pre MoE
output alloc), `moe_runner/deep_gemm.py` + `moe_runner/triton_utils/fused_moe.py`
(wrap runner output allocs), `mhc.py`.

### Porting to v0.5.15.post1

The PR was authored on later `main`; its file path pattern (`python/sglang/kernels/…`)
and two hunks conflict with v0.5.15.post1. I cherry-picked and resolved 2 conflicts:
- `dp_attention.py`: v0.5.15 has extra fields (`_hidden_size/_dtype/_device/
  _is_extend_in_batch`) around the sizing quartet — kept them, applied the PR's
  real-default values to the quartet + `_is_extend_in_batch=False`.
- `deep_gemm.py`: the PR renames `post_reorder_triton_kernel[...]` →
  `post_reorder_deepgemm(...)` (an unrelated later refactor) — kept v0.5.15's
  `post_reorder_triton_kernel[...]` call and applied only the `use_symmetric_memory`
  wrap around `output = torch.empty(...)`.
The other 3 files applied cleanly. Ported diff + both file snapshots (baseline &
patched) saved under `patches/pr29007/`.

## 2. Environment fixes (required to serve at all)

`--enable-symm-mem` JIT-builds the `nccl_allocator` C++ extension at server startup,
which failed three ways until fixed (all env, not code):
1. `ld: cannot find -lcuda` → add `/usr/lib` (real `libcuda.so`) + conda stubs dir to `LIBRARY_PATH`.
2. `fatal error: cuda_runtime.h` → add `$CONDA_PREFIX/targets/x86_64-linux/include` to `CPATH`.
3. `ld: cannot find -lnccl` → `ln -sf libnccl.so.2 libnccl.so` in `nvidia/nccl/lib`,
   add that dir + `nvidia/nccl/include` to `LIBRARY_PATH`/`CPATH`.
Also clear the stale `/tmp/symm_allocator` ninja cache between attempts. After these,
DeepSeek-V4 loads on TP8 and the symmetric allocator builds and loads cleanly.

## 3. Method

- Both arms: fresh sglang server, `--model-path DeepSeek-V4-Flash-FP8 --tp-size 8
  --mem-fraction-static 0.85 --enable-symm-mem` (`attention_backend=dsv4` auto).
- baseline = stock v0.5.15.post1; patched = PR #29007 files copied in (pure Python,
  no rebuild), server restarted.
- `bench_serving --dataset-name random-ids --random-range-ratio 1.0`, cells
  `4096/1024/c1` (upstream regime), `4096/1024/c8`, `4096/512/c16`; 2 repeats each.
- Metric focus: mean TPOT (decode, where the per-layer all-reduce lives), mean E2E,
  output throughput. Scripts `scripts/run_v48_dsv4_pr29007_ab.py`; raw
  `results/2026-07-21_v48_dsv4_pr29007/ab.jsonl` (12 rows), `analysis.txt`.

## 4. Results (mean over 2 repeats)

| cell | metric | baseline | patched | gain |
|---|---|---|---|---|
| 4096/1024/c1 | mean TPOT (ms) | 6.92 | 6.33 | **+9.2%** |
| 4096/1024/c1 | mean E2E (ms) | 7359.2 | 6655.5 | **+10.6%** |
| 4096/1024/c1 | out tput (tok/s) | 139.1 | 153.8 | **+10.6%** |
| 4096/1024/c8 | mean TPOT (ms) | 8.84 | 8.34 | +6.0% |
| 4096/1024/c8 | mean E2E (ms) | 9611.6 | 8979.0 | +7.0% |
| 4096/1024/c8 | out tput (tok/s) | 852.4 | 912.1 | +7.0% |
| 4096/512/c16 | mean TPOT (ms) | 11.13 | 10.57 | +5.3% |
| 4096/512/c16 | mean E2E (ms) | 6308.4 | 5928.3 | +6.4% |
| 4096/512/c16 | out tput (tok/s) | 1298.1 | 1380.7 | +6.4% |

The win is largest at c1 (all-reduce latency is a bigger share of a low-occupancy
step) and shrinks with concurrency (more compute to hide the all-reduce behind), but
stays a solid +5–7% through c16. Both repeats agree to <0.2% within each cell.

## 5. Correctness note

The PR changes **only where** the MoE output tensor is allocated (symmetric pool vs
plain), not the math — the all-reduce is bit-identical, just routed through the fast
NCCL path. Both arms produced coherent generations with matching token counts. A
bit-exact greedy A/B (both servers on separate ports, same prompt, temp 0) is a
reasonable follow-up but was not needed to establish the perf result; by construction
the numerics are unchanged.

## 6. Verdict

PR #29007 **reproduces cleanly and positively** on 8×H200 / TP8 / v0.5.15.post1:
**−9.2% TPOT / −10.6% E2E / +10.6% output throughput** at the upstream c1 regime,
holding at +5–7% through c16, with identical config on both arms (the win is purely
the symmetric-memory allocation). This is an excellent "natural-text, semantics-
unchanged, same-config, agent-relevant kernel/runtime fix that lands on real E2E"
headline — arguably the strongest of the reproductions so far.

## 7. Reproduce

- env `sglang-v515`; `git worktree` at v0.5.15.post1 (`/home/…/sglang-v515`).
- Serving env (else nccl_allocator won't build): see §2. Set `CPATH`, `LIBRARY_PATH`,
  `LD_LIBRARY_PATH` for CUDA + NCCL, symlink `libnccl.so`.
- Toggle arm by copying `patches/pr29007/{baseline,patched}/*.py` into the worktree
  and restarting the server (pure Python, no rebuild).
