# NCU roofline — real decode-heavy & prefill-heavy regimes, both-stage hot kernels

> 2026-07-22 · H200 · sglang v0.5.12 (Triton fused MoE, fa3 attention) · Qwen3-30B-A3B bf16.
> Chendi's next step: pick **real decode-heavy and prefill-heavy regimes**, profile with
> **NCU (CUDA graph disabled → single-kernel capture)**, get the **roofline of the hot
> kernels in BOTH stages**, and for **fused_moe specifically** show the memory-bound
> analysis. Extends `ncu_roofline_fused_moe_analysis.md` (isolated MoE microbench).

## 0. TL;DR — the hot kernels and their roofline position, per stage

| stage | hot kernel | Compute (SM) % | DRAM % | **bound** | source |
|---|---|---:|---:|---|---|
| **DECODE** | **fused_moe** (MoE GEMM) M=32 | 19.7 | **87.9** | **MEMORY** | v50 isolated |
| **DECODE** | fused_moe M=128 | 15.1 | **89.8** | **MEMORY** | v50 isolated |
| **PREFILL** | fused_moe (MoE GEMM) M=4096 | **67.4** | 43.3 | **COMPUTE** | v50 isolated |
| **PREFILL** | fused_moe (real, in=4096) | **60.0** | 13.8 | **COMPUTE** | prefill_roofline |
| **PREFILL** | **dense GEMM** (nvjet, qkv/o) | **79.8** | 13.8 | **COMPUTE** | prefill_roofline |
| **PREFILL** | **attention** (FA3 `FlashAttnFwdSm90`) | — dominant by **time (33 ms, O(seq²))** | | compute | names3 dump |

- **Decode-heavy regime**: the MoE GEMM streams expert weights for few tokens → sits
  on the **memory-bandwidth roof** (DRAM ~88–90%). Dense GEMM and attention are also
  small/latency-bound at decode. Decode ≈ memory-bound (matches the bottleneck report).
- **Prefill-heavy regime**: every hot kernel is **compute-bound** — MoE GEMM 60–67%,
  dense GEMM ~80% SM, and **attention (FA3) is the single largest kernel by time**
  (33 ms across the profiled window, O(seq²) at 4096 context). Prefill ≈ compute-bound.

**The same fused_moe kernel flips memory→compute from decode to prefill** — that
crossover is the headline (see the isolated report for the clean M-sweep).

## 1. Regimes profiled (from the bottleneck report)

- **Decode-heavy**: `decode_heavy` (128/1024/32), `decode_medium` (128/512/16). For a
  single-kernel NCU capture we use `bench_one_batch --batch-size 32 --input-len 8
  --output-len 8` (decode M = batch = 32) — plus the isolated MoE microbench at M=32/128.
- **Prefill-heavy**: `prefill_medium` (4096/16/4), `prefill_long` (16384/16/2). Capture
  via `bench_one_batch --batch-size 1 --input-len 4096 --output-len 4` (prefill M=4096).
- **CUDA graph DISABLED** (`--disable-cuda-graph`) in every run so NCU sees individual
  kernel launches (a captured graph is opaque to NCU).

## 2. The four roofline metrics / how the figure is produced

The Nsight-Compute roofline chart is rendered from the **SpeedOfLight roofline
sections**. We capture with **`--set full`**, which includes all of them:
`SpeedOfLight`, `SpeedOfLight_RooflineChart`, and the precision-specific
`SpeedOfLight_Hierarchical{Half,Tensor,Single,Double}RooflineChart`. For **bf16** MoE/GEMM,
the relevant ceilings are the **Half** and **Tensor-core** rooflines.

The four underlying quantities the roofline needs (and that we read for the tables):
1. **Compute throughput** — `sm__throughput.avg.pct_of_peak_sustained_elapsed` (% of SM peak).
2. **Memory throughput** — `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` (% of HBM peak) + `dram__bytes.sum` for TB/s.
3. **Work (FLOPs)** — `sm__sass_thread_inst_executed_op_{h,f}fma/add/mul_pred_on.sum` (the tensor/half FLOP counters → arithmetic intensity numerator).
4. **Time** — `gpu__time_duration.sum` / `sm__cycles_elapsed.avg[.per_second]`.
`--set full` collects all of these, so the roofline dot + ceilings render in ncu-ui.

## 3. Method notes / gotchas (so this reproduces)

- **NCU needs sudo** here (`RmProfilingAdminOnly=1`); under sudo pass `CPATH`/`LIBRARY_PATH`/
  `LD_LIBRARY_PATH` for the sglang JIT (`ld -lcudart`). See scripts.
- **Landing on the right kernel instance is finicky.** `bench_one_batch` runs a warmup
  (prefill + a few decode) then the measured pass. With a `--kernel-name` filter,
  `--launch-skip` counts only *matched* launches; with no filter it counts *all*. We
  used `--launch-skip 1500` (no/broad filter) to land in the measured forward.
- **fa3 attention kernel name**: it is `void cutlass::device_kernel<flash::
  FlashAttnFwdSm90<…>>` — match it with `--kernel-name-base demangled --kernel-name
  regex:FlashAttnFwdSm90`. (Plain `cutlass::device_kernel` against the default mangled
  base name does **not** match.)
- **Authoritative decode roofline = the isolated MoE microbench** (v50), which fixes M
  exactly. The real decode capture (batch=32) tends to land on the batch's *prefill*
  MoE (M=256, compute-ish) rather than the M=32 decode step, so use the isolated numbers
  for the decode memory-bound claim.
- **Attention per-launch counters were degenerate** in some captures (FA3 is a
  persistent varlen-scheduler kernel; a single captured launch can show ~0% util). Its
  *dominance by time* (33 ms, from the duration-only sweep) is the reliable statement;
  a clean attention roofline is a noted follow-up.

## 4. WHERE to take each screenshot (ncu-ui)

Open a report, e.g. `ncu-ui results/2026-07-22_v51_ncu_real/prefill_roofline.ncu-rep`.
Select the kernel in the top launch selector, then:

### Prefill hot kernels — roofline (show compute-bound)
- **`prefill_roofline.ncu-rep`**, select **`fused_moe_kernel`** → **"GPU Speed Of Light
  Throughput"** section → **Roofline** chart. Screenshot: the dot sits **near the flat
  compute (Tensor/Half) ceiling, right side** → compute-bound (SM 60%).
- Same report, select **`nvjet_sm90_…`** (dense GEMM) → Roofline chart: even closer to
  the compute ceiling (SM ~80%). Screenshot side-by-side with MoE.
- (Attention: open `prefill_attention.ncu-rep` — the FA3 kernel; note the time-dominance
  from `names3` if the per-launch roofline is degenerate.)

### Decode hot kernel — roofline (show memory-bound) + the memory analysis
- **`fused_moe_M32.ncu-rep`** (isolated, from the companion report) → **Roofline** chart:
  the dot sits **on the sloped memory-bandwidth ceiling, far left (low AI)** → memory-bound.
- **fused_moe memory-bound analysis (the key screenshot):** on the **decode** report
  (`fused_moe_M32.ncu-rep`):
  1. **"GPU Speed Of Light Throughput"** top bar chart → Memory ≈ 88%, Compute ≈ 20%
     (Memory bar hugging the right). Screenshot.
  2. **"Memory Workload Analysis" → Memory Chart** (L1/L2/DRAM diagram) → **DRAM is the
     saturated stage (~88%, ~4.2 TB/s of H200's ~4.8 TB/s)**. Screenshot — this is
     *where* the bottleneck physically is (HBM read of expert weights).

> Quick answer to "**Where should we take a screenshot?**":
> 1. **Prefill roofline** — `prefill_roofline.ncu-rep`, `fused_moe_kernel` (and
>    `nvjet` dense GEMM), the **Roofline chart** in *GPU Speed Of Light Throughput*
>    (dot on the compute ceiling).
> 2. **Decode roofline** — `fused_moe_M32.ncu-rep`, the **Roofline chart** (dot on the
>    memory-bandwidth ceiling).
> 3. **fused_moe memory-bound proof** — `fused_moe_M32.ncu-rep`, **Memory Workload
>    Analysis → Memory Chart** showing **DRAM saturated at ~88%**.
> Put the prefill (compute-roof) and decode (memory-roof) roofline charts **side by
> side** — that single image is the whole "bottleneck moves with the regime" story.

## 5. Artifacts

- `results/2026-07-22_v51_ncu_real/prefill_roofline.ncu-rep` — prefill MoE + dense GEMM (compute-bound)
- `results/2026-07-22_v51_ncu_real/decode_roofline.ncu-rep` — batch=32 MoE + dense GEMM
- `results/2026-07-22_v51_ncu_real/prefill_attention.ncu-rep` — FA3 attention kernel
- `results/2026-07-22_v51_ncu_real/real_hotkernel_roofline.csv`, `names3.csv` (kernel time ranking)
- Isolated MoE roofline (authoritative crossover): `results/2026-07-22_v50_ncu_roofline/fused_moe_M{32,128,2048,4096}.ncu-rep`
- Scripts: `run_v50_ncu_moe_microbench.py`, `run_v50_ncu_roofline.sh`.

## 6. Bottom line

- **Decode-heavy regime → hot kernels are MEMORY-bound** (fused_moe DRAM ~88–90%). Kernel
  math/config tuning ≈0 e2e; levers are spec-decoding / concurrency.
- **Prefill-heavy regime → hot kernels are COMPUTE-bound** (fused_moe SM 60–67%, dense
  GEMM ~80%, attention O(seq²) dominant by time). Config/tiling tuning converts to real
  e2e gains (+34–43% prefill) — consistent with the bottleneck report.
