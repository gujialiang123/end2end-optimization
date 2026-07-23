# High-concurrency TTFT rerun — comparison & analysis

> Streaming rerun of the four v4 slide points to obtain client-observed TTFT (the original
> non-streaming client could not). Workload/seed identical to v4; 5 measured reps (rep0 dropped).
> H200, BF16, TP1, CUDA graph ON. Source: `results/2026-07-23_high_concurrency_ttft_rerun/`.
> Date: 2026-07-23.

## Metric conventions
- **throughput speedup** = candidate / baseline − 1 (or shown as ×).
- **latency improvement** = 1 − candidate / baseline; **positive = lower latency (better)**.
- **Client-observed TTFT includes admission and scheduler queueing** (that is the user-visible
  effect of `max_running_requests`).

## Configurations
- **A. cookbook baseline**: cap=32, chunk=-1, lpm, mem=0.85
- **B. full high-concurrency**: cap=128, chunk=2048, fcfs, mem=0.90
- **C. cap-only ablation**: cap=128, chunk=-1, lpm, mem=0.85 (isolates admission capacity)

## Full high-concurrency vs cookbook

| model | regime | throughput | TTFT p50 | TTFT p95 | TPOT p50 |
|---|---|---|---|---|---|
| Qwen | C64/O512 | 7.41 → 11.68 req/s (**1.58×**) | 2202 → 117 ms (**−95%**) | 4336 → 160 ms (**−96%**) | 8.3 → 10.4 ms |
| Qwen | C128/O256 | 14.78 → 36.01 req/s (**2.44×**) | 3265 → 210 ms (**−94%**) | 6446 → 245 ms (**−96%**) | 8.2 → 12.7 ms |
| LFM2.5 | C64/O512 | 12.35 → 17.24 req/s (**1.40×**) | 1386 → 204 ms (**−85%**) | 2654 → 233 ms (**−91%**) | 4.8 → 6.7 ms |
| LFM2.5 | C128/O256 | 23.85 → 52.07 req/s (**2.18×**) | 2028 → 217 ms (**−89%**) | 4112 → 332 ms (**−92%**) | 4.8 → 8.4 ms |

→ Throughput **improves AND** TTFT p50/p95 **improve** (−85% to −96%). This is **removal of an
admission bottleneck, NOT a Pareto tradeoff.** (The cookbook `max-running-requests=32` throttles
admission once offered concurrency exceeds 32, so requests queue and TTFT explodes.)

TPOT rises modestly (e.g. 8.2 → 12.7 ms) because more requests decode concurrently; this is a
small decode-side cost, dwarfed by the TTFT improvement. Not framed as a tradeoff because the
primary user-visible latency (TTFT) improves dramatically.

## Cap-only ablation vs cookbook (is admission capacity the cause?)

| model | regime | cap-only throughput | full throughput | full-vs-cap throughput residual |
|---|---|---|---|---|
| Qwen | C64/O512 | **1.57×** | 1.58× | +0.7% |
| Qwen | C128/O256 | **2.47×** | 2.44× | −1.4% |
| LFM2.5 | C64/O512 | **1.40×** | 1.40× | −0.7% |
| LFM2.5 | C128/O256 | **2.23×** | 2.18× | −2.0% |

→ **Cap-only reproduces essentially all of the gain** (full-vs-cap residual is −2.0% to +0.7%,
within noise). **Admission capacity (`max_running_requests` 32→128) is the dominant cause.** The
other knobs (chunked-prefill 2048, fcfs, mem 0.90) contribute negligibly — even slightly negative
at C128. Cap-only also delivers the same TTFT collapse (e.g. Qwen C128 baseline 3265ms → cap-only
133ms → full 210ms).

**Honest labeling consequence**: the candidate should be called a **"high-concurrency
configuration"**, but the analysis shows the effect is driven by admission capacity; chunking/fcfs/
memory are not material here. (Do NOT attribute the gain to chunked prefill in this regime.)

## Reproduction fidelity vs original v4 CSV
Streaming-rerun throughput matches the original non-streaming v4 numbers closely (e.g. Qwen C64
7.41 vs 7.41; Qwen C128 36.0 vs cap128 36.36; LFM C64 17.24 vs 17.33; LFM C128 52.1 vs 50.46),
confirming the workload was faithfully reproduced.

## Data files
- `summary.csv` (aggregate, 12 rows: 2 models × 2 regimes × 3 configs)
- `per_run_metrics.csv` (60 rows: 5 reps each)
- `per_request_metrics.csv` (5760 raw per-request TTFT/E2E records)
- `analysis.json` (3-way comparison)
- `environment.json`, `workload_audit.md`, `reproduce.sh`
- figure: `high_concurrency_ttft.png` / `.svg`
