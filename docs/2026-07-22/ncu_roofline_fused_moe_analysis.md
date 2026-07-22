# NCU roofline analysis — fused_moe kernel, decode (memory-bound) vs prefill (compute-bound)

> 2026-07-22 · host `aifx-clou000001`, H200 · sglang (Triton fused MoE) · model shape
> **Qwen3-30B-A3B** (E=128, shard_intermediate N=1536, hidden H=2048, topk=8, bf16).
> Follow-up to `docs/2026-07-22/prefill_vs_decode_bottleneck_report.md` (Chendi's ask):
> use **Nsight Compute (NCU)** to prove, at the hardware-counter level, *which* kernel
> dominates each stage and *why* — roofline of the key kernel.

## 0. TL;DR

Profiled the sglang Triton **`fused_moe_kernel`** (the MoE grouped-GEMM — the single
largest kernel in both stages: 33–68% of decode, 41% of prefill per the bottleneck
report) at decode-scale and prefill-scale token counts M, with NCU's full roofline
section. The kernel **crosses the roofline** exactly as the report claims:

| M (tokens) | regime | duration µs | **Compute (SM) %** | **DRAM %** | mem TB/s | **bound** |
|---:|---|---:|---:|---:|---:|---|
| 32 | decode | 167.5 | 19.7 | **87.9** | 4.23 | **MEMORY** |
| 128 | decode | 187.9 | 15.1 | **89.8** | 4.32 | **MEMORY** |
| 2048 | prefill | 330.2 | **64.5** | 57.9 | 2.78 | **COMPUTE** |
| 4096 | prefill | 521.4 | **67.4** | 43.3 | 2.08 | **COMPUTE** |

(Numbers are the dominant w1 grouped-GEMM launch; the w2 launch behaves the same, see
`roofline_summary.csv`. H200 HBM3e peak ≈ 4.8 TB/s.)

- **Decode (M=32, 128): DRAM at 88–90% of peak, compute at only 15–20%** → the kernel
  sits on the **memory-bandwidth roof**. Each decode step re-streams the full active-
  expert weights for a handful of tokens; it is bandwidth-bound, so faster GEMM math
  cannot help. This is the hardware-counter proof of the report's "decode is
  memory-bound weight streaming."
- **Prefill (M=2048, 4096): compute rises to 64–67%, DRAM drops to 43–58%** → the
  kernel moves up to the **compute roof**. Many tokens amortize the one-time weight
  read (higher arithmetic intensity), so better GEMM tiling / config tuning translates
  to real speedups — matching the +34–43% prefill e2e we measured.

**One kernel, two regimes, opposite roofline positions** — the bottleneck moves with M.

## 1. Method

- **Environment (blockers fixed):**
  - GPU perf counters are admin-only here (`RmProfilingAdminOnly=1` → `ERR_NVGPUCTRPERM`).
    We have passwordless `sudo` (group `wheel`), so NCU is run under `sudo -E`.
  - Under `sudo` the login env is stripped, so the sglang fused-activation **JIT build**
    fails (`ld: cannot find -lcudart`). Fixed by passing `CPATH` / `LIBRARY_PATH` /
    `LD_LIBRARY_PATH` (conda `targets/x86_64-linux/{include,lib}` + `lib`) into the sudo env.
- **Isolation:** `scripts/run_v50_ncu_moe_microbench.py` calls `fused_moe` once at a
  fixed M with the real Qwen3-30B-A3B shape, so NCU captures one clean grouped-GEMM
  launch (no CUDA-graph, no surrounding model). Warmup launches are skipped
  (`--launch-skip 2`), the first timed w1+w2 GEMM captured (`--launch-count 2`).
- **NCU:** `ncu --set full --kernel-name regex:fused_moe_kernel --export ...` (the
  `full` set includes all SpeedOfLight, roofline, memory- and compute-workload
  sections). Driver: `scripts/run_v50_ncu_roofline.sh 32 128 2048 4096`.
- **Reads:** SoL `sm__throughput...pct_of_peak` (compute %), `gpu__dram_throughput...
  pct_of_peak` (DRAM %), `Memory Throughput [Tbyte/s]`, `gpu__time_duration`.
- All numbers our own on H200. Config: sglang fallback `triton_3_2_0` MoE config
  (production default for this uncovered shape).

## 2. Artifacts

- Per-M NCU reports (open in **ncu-ui** for the charts):
  `results/2026-07-22_v50_ncu_roofline/fused_moe_M{32,128,2048,4096}.ncu-rep`
- Summary CSV: `results/2026-07-22_v50_ncu_roofline/roofline_summary.csv`
- Per-run NCU logs: `results/2026-07-22_v50_ncu_roofline/ncu_M*.log`
- Scripts: `scripts/run_v50_ncu_moe_microbench.py`, `scripts/run_v50_ncu_roofline.sh`

## 3. How to view + WHERE to screenshot (ncu-ui)

Open a report: `ncu-ui results/2026-07-22_v50_ncu_roofline/fused_moe_M32.ncu-rep`
(or File▸Open). Pick the kernel launch in the top selector. Then:

### Screenshot A — the Roofline chart (the headline figure)
1. In the report, open the **"GPU Speed Of Light Throughput"** section.
2. Expand the **Roofline** chart (in `--set full` it appears as
   **"GPU SOL — Roofline"**, and the precision-specific ones
   **"Hierarchical Roofline (Half / Tensor)"** — bf16 MoE uses **Half/Tensor**).
3. **Screenshot the Roofline chart.** The achieved point (the dot) tells the story:
   - **decode report (M=32/128):** the dot sits **on the sloped memory-bandwidth
     ceiling, far left (low arithmetic intensity)** → memory-bound.
   - **prefill report (M=2048/4096):** the dot sits **under the flat compute ceiling,
     to the right (higher AI)** → compute-bound.
   Put the M=32 and M=4096 roofline charts **side by side** — that single image is the
   clearest "the same kernel moves across the roofline" evidence.

### Screenshot B — proof it's memory-bound (for the decode reports)
1. Still on the kernel, open **"GPU Speed Of Light Throughput"** — the top bar chart
   shows **Compute (SM) vs Memory** throughput. **Screenshot it:** Memory ≈ 88–90%,
   Compute ≈ 15–20% (bar hugging the right for Memory, short for Compute) = memory-bound.
2. Open **"Memory Workload Analysis"** → its **Memory Chart** (L1/L2/DRAM traffic
   diagram). **Screenshot the chart:** DRAM is the saturated stage (~88% of peak, ~4.2
   TB/s). This is *where* the bottleneck is — the HBM read of expert weights.
3. (Optional) **"Compute Workload Analysis"** shows the SM pipes idle (low utilization),
   confirming compute is not the limiter.

### Screenshot C — proof prefill is compute-bound (for the prefill reports)
1. Same **"GPU Speed Of Light Throughput"** bar chart on `fused_moe_M4096.ncu-rep`:
   Compute ≈ 67%, Memory ≈ 43% — bars **flipped** vs decode. **Screenshot it.**
2. **"Compute Workload Analysis"** now shows the tensor/FMA pipes as the busy ones.

> Quick answer to "where should we take a screenshot?": **(1)** the **Roofline chart**
> in *GPU Speed Of Light Throughput* (put decode-M32 and prefill-M4096 side by side),
> and **(2)** the **Memory Workload Analysis → Memory Chart** on the decode (M=32)
> report to show DRAM saturated at ~88%. Those two images carry the whole argument.

## 4. Consequence (ties back to the optimization results)

- **Decode = on the memory roof** → MoE kernel/config math changes give ≈0 e2e (as we
  measured); the real levers are **spec decoding** and **serving concurrency** (reduce
  or hide weight re-reads), not GEMM tiling.
- **Prefill = on the compute roof** → MoE config/tiling tuning converts to real e2e
  gains (**+34–43% prefill**), because the kernel is compute-limited there.

## 5. Supplementary — real `bench_one_batch` decode capture

A real decode-heavy step (batch=32, in=8, out=32, `--disable-cuda-graph`) profiled
under NCU (SpeedOfLight) confirms the **`fused_moe_kernel` dominates the MoE path
(93% of profiled MoE-path kernel time)**; the small `moe_sum_reduce` (4.7%, DRAM
88% → memory-bound) and `moe_align_block_size` (2%) are the only other MoE kernels.
Ranking + caveats: `results/2026-07-22_v50_ncu_roofline/real_decode_kernel_ranking.txt`,
report `real_decode.ncu-rep`.

Caveat (honest): the SoL %/roofline numbers in that *real* capture are **not
reliable** — the `--launch-skip` window landed on launches of ambiguous effective M,
so its compute/DRAM % differ from the controlled microbench. **Use the isolated v50
microbench (`fused_moe_M*.ncu-rep`) as the authoritative roofline crossover.** The
`fa3` attention kernel was not captured (name didn't match the filter).

## 6. Suggested follow-ups

- Same NCU roofline for the **attention kernel** (#2 kernel: 38.8% of long-context
  prefill, 16–21% of decode) — it should show the same flip but driven by O(seq²)
  compute at long context. Needs the attention backend + KV-cache set up standalone,
  or a real `bench_one_batch --disable-cuda-graph` capture filtered to the fmha/flash
  kernels.
- A real end-to-end `bench_one_batch` NCU capture (decode-heavy + long-prefill) to rank
  all kernels by NCU-measured duration and confirm the composition numbers in the
  bottleneck report at hardware-counter precision.
