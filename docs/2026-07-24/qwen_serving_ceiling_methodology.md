# Qwen + LFM2.5 serving-ceiling campaign — methodology

**Campaign:** `results/2026-07-24_serving_ceiling/`
**Purpose:** measure how much of the end-to-end performance gap can be closed by
**serving-level configuration alone**, so that whatever remains is correctly
attributed to kernels, backends, policies or communication.

---

## 1. What is fixed and what is tuned

The experiment isolates serving configuration. Everything that could change the
execution path is frozen and **verified from the server log of every single
configuration**, not assumed.

| category | setting | how it is enforced |
|---|---|---|
| model | Qwen3-30B-A3B-Instruct-2507 · LFM2.5-8B-A1B | explicit `--model-path` |
| dtype / parallelism | BF16 · TP1 · 1× H200 | `--tensor-parallel-size 1` |
| attention backend | `fa3` | parsed from `attention_backend=` in the log |
| MoE runner backend | `auto` | parsed from `moe_runner_backend=` in the log |
| CUDA Graph | enabled, capture completed | `disable_cuda_graph=False` **and** "Capture cuda graph end" in the log |
| context length | 8192 | `--context-length 8192` |
| scheduling conservativeness | 1.0 | `--schedule-conservativeness 1.0` |
| speculative decoding / kernel config | untouched | never passed |

**Tuned (the only degrees of freedom):**

| knob | values | server argument |
|---|---|---|
| `max_running_requests` | 8, 16, 24, 32, 48, 64, 96, 128 | `--max-running-requests` |
| `chunked_prefill_size` | −1, 2048, 8192 | `--chunked-prefill-size` |
| `schedule_policy` | lpm, fcfs | `--schedule-policy` |
| `mem_fraction_static` | 0.75, 0.80, 0.85, 0.90 | `--mem-fraction-static` |

**8 × 3 × 2 × 4 = 192 unique configurations per model.**

### Why full grid enumeration rather than TPE

The goal is not to find one throughput winner. We need all-point Pareto fronts,
the negative and flat results, and the cross-regime transfer structure. TPE
concentrates samples around one objective and would undersample exactly the
regions that make the honest argument. The sampler is therefore a deterministic
full grid; the whole space is enumerated, so no sampling bias exists at all.

`chunked_prefill_size = 8192` is effectively identical to `−1` at
`--context-length 8192`, since the longest canonical input is ≈4 000 tokens.
This is deliberate and documented: the pair acts as a built-in duplicate
measurement and yields a free estimate of run-to-run noise.

### No warm start

No `enqueue_trial`, no seeded cookbook, no reused study, no copied trials. The
cookbook configuration is **config_id 74** inside the same grid, measured by the
same harness under the same protocol, so the baseline is never a differently
measured external number. This directly fixes the bias identified in the
2026-07-02 v3 study (see the Phase-0 audit).

---

## 2. Workload suite

All six workloads are driven by **one** streaming client,
`sglang.bench_serving --output-details`, so TTFT, TPOT/ITL and E2E are defined
identically everywhere and every metric comes from raw per-request records.

| workload | client arguments | shape |
|---|---|---|
| `R_short_decode` | `random-ids`, in 100, out 256, n 8, conc 1 | latency-bound single stream |
| `R_medium_balanced` | `random-ids`, in 800, out 256, n 16, conc 8 | mixed |
| `R_long_prefill` | `random-ids`, in 4000, out 32, n 4, conc 4 | prefill-dominated |
| `R_concurrent_decode` | `random-ids`, in 200, out 256, n 32, conc 32 | decode-dominated |
| `shared_prefix` | `generated-shared-prefix`, 8 groups × 16 prompts, system 2048, question 128, out 256 | prefix-cache sensitive |
| `tool_agent` | `mooncake --mooncake-workload toolagent`, n 200 | real FAST'25 agent trace |

The four synthetic regimes reproduce the recovered v4/2026-06-25 specification
(`results/2026-06-25_autotuning/true-default-bf16/regimes_resolved.yaml`);
the two agentic workloads reproduce `scripts/run_v7_agentic_bench.py`.
Definitions are frozen in `results/2026-07-24_serving_ceiling/workloads.yaml`.

Two deliberate, documented deviations from the historical specs:

1. **Streaming client.** The original v4 client posted to `/generate` without
   `"stream": true` and therefore could not produce TTFT at all. The workload
   shape (input size, output length, request count, concurrency) is preserved;
   only the client is upgraded. This is the same deviation the v48 study made.
2. **`random-ids` instead of `random`.** The `random` dataset requires
   downloading ShareGPT into a read-only `HF_HUB_CACHE`, which fails on this
   host. `random-ids` produces deterministic token ids of exactly the requested
   length — closer to the specified fixed-length shape than ShareGPT text, and
   identical across every configuration, so comparisons are unaffected.
   `prompt_words` is mapped 1 word → 1 token.

---

## 3. Execution protocol

For each of the 384 (model, configuration) tasks:

1. pick a bindable port from a per-worker window (bind test, not connect test);
2. wait until the GPU has ≥110 GB free;
3. launch the server in its **own process group**;
4. wait for `/health`, then parse the log and record the resolved configuration
   and the CUDA-graph capture;
5. run **all six workloads against that one server**, in a randomized order
   (seeded per config) to remove ordering bias;
6. save the raw `--output-details` JSON and per-request parquet;
7. terminate the whole process group, then verify port release and GPU-memory
   release before continuing.

Running all six workloads from a single launch is what makes the 6×6 transfer
matrix a set of real measurements instead of an extrapolation: every
configuration has already been measured on every regime.

### Parallelism and resumption

A sqlite work-queue hands tasks to N single-GPU workers (7 used here, GPU 0–6).
Task state is `pending → running → done|failed`, so:

* interrupting and relaunching workers never repeats a completed configuration;
* a worker crash leaves its task recoverable;
* every configuration has a stable hash (`cap32_chunk-1_pollpm_mem0.85`).

### Failure policy

Failures are classified and **never fake-scored**. Infrastructure failures (port
conflict, stale server, SIGKILL during load) are retried once and do not count
as completed configurations; genuine configuration failures (OOM, invalid
argument, CUDA-graph failure) would be retained in `failures.csv` with zero
substituted for nothing.

During this campaign 5 launch failures occurred, all traced to a port-rebind
race (`[Errno 98] address already in use`) caused by probing ports with
`connect()` while a shutting-down server still held the listening socket. The
harness now tests ports by actually binding them and rotates ports between
tasks. All affected tasks were requeued and completed; 18 per-run rows belonging
to interrupted configurations were dropped so that no partial configuration
entered the matrix. **Final state: 384/384 complete, 0 unresolved failures.**

---

## 4. Metrics

Recorded per (model, configuration, workload, repetition):

* **throughput** — request, input-token, output-token, total-token;
* **latency** — TTFT, TPOT/ITL and E2E at mean / p50 / p95 / p99, computed from
  the raw `ttfts` and `itls` arrays, never estimated from throughput;
* **operational** — benchmark wall time, server startup time, completion rate,
  failed-request count, output-token counts;
* **raw** — every request's TTFT, ITL list, input and output length (parquet).

TTFT is the client-observed interval from request submission to the first
streamed token, so it **includes scheduler and admission queueing** — which is
the user-visible consequence of `max_running_requests`.

TPOT is the mean inter-token latency of a request excluding the first token.

---

## 5. Analysis

### Sign convention

```
throughput improvement = candidate / baseline − 1
latency    improvement = 1 − candidate / baseline
```

A positive latency improvement always means **lower latency**.

### Classification

* **WIN** — a primary metric improves and no guardrail metric significantly
  regresses;
* **REGRESSION** — the primary metric significantly degrades;
* **TRADE-OFF** — one primary metric significantly improves **while another
  significantly worsens**;
* **FLAT / INCONCLUSIVE** — changes are inside the noise band.

For the coverage pass (one measurement per cell) a ±3 % noise band is used and
the result is reported as coverage-grade evidence. For the validation pass the
95 % confidence interval over 5 repetitions replaces the fixed band. **A
trade-off is only claimed when both directions are significant** — normal
measurement noise is never called a trade-off.

### Pareto

Three non-dominance views are computed per regime:

1. TTFT p95 (min) × output-token throughput (max) — the slide view;
2. E2E p95 (min) × request throughput (max);
3. full five-metric non-dominance (request thr, output thr, TTFT p95, TPOT p95,
   E2E p95).

In every plot lower TTFT is on the **left** and higher throughput at the
**top**, so the preferred direction is upper-left; this is labelled on the axes.

### Transfer matrices

For each source configuration (cookbook plus each regime's validated winner) and
each target regime, the ratio against the **target regime's own cookbook** is
reported, so all cells are dimensionless and >1.00× always means better. Five
separate matrices are produced (request thr, output thr, TTFT p95, TPOT p95,
E2E p95) rather than one ambiguous compressed heatmap.

---

## 6. Validation pass

The coverage pass is one measurement per (configuration, workload). Final
"best" claims must not rest on a single-run ranking, so a second pass re-runs a
selected subset with **5 repetitions**:

* the cookbook baseline;
* per regime: the highest request-throughput, highest output-throughput, lowest
  TTFT p95, lowest TPOT p95, lowest E2E p95 and balanced/Pareto configurations;
* a sample of Pareto points;
* one or two clear regression configurations, so the negative results are
  validated as carefully as the positive ones.

After de-duplication this is 27 configurations for Qwen and 35 for LFM2.5
(62 tasks). Output: `results/2026-07-24_serving_ceiling_validation/`.

---

## 7. Reproduction

```bash
bash results/2026-07-24_serving_ceiling/reproduce.sh      # GPUS="0 1 2 3 4 5 6"
```

Scripts:

| file | role |
|---|---|
| `scripts/serving_ceiling_lib.py` | grid, workload definitions, server lifecycle, log parsing |
| `scripts/run_serving_ceiling_campaign.py` | sqlite work-queue, parallel workers, resume, failure policy |
| `scripts/analyze_serving_ceiling.py` | deltas, Pareto, classification, transfer matrices, summary matrix |
| `scripts/render_serving_ceiling_figures.py` | all slide figures (PNG + SVG) |
| `scripts/run_serving_ceiling_validation.py` | selection + 5-repetition validation pass |
| `scripts/update_performance_gap_slides.py` | six-slide draft deck |
