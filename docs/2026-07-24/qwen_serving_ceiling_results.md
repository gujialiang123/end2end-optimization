# Qwen + LFM2.5 serving-ceiling campaign — results

**Date:** 2026-07-25 · **Campaign:** `results/2026-07-24_serving_ceiling/`
**Coverage pass:** 384/384 tasks (2 models × 192 configurations × 6 workloads),
**0 unresolved failures**, 2 304 per-run measurements, 148 992 per-request records.
**Validation pass:** `results/2026-07-24_serving_ceiling_validation/` — 62
configurations × 5 repetitions, 1 860 per-run measurements.

Methodology: `docs/2026-07-24/qwen_serving_ceiling_methodology.md`
Audit of prior data: `docs/2026-07-24/serving_tuning_data_audit.md`
Slide provenance: `docs/2026-07-24/qwen_serving_ceiling_slide_claims.md`

Sign convention: `throughput delta = candidate/baseline − 1`,
`latency delta = 1 − candidate/baseline` (**positive latency delta = lower latency**).

---

## 0. Data quality first: the steady-state correction

The first version of this campaign produced a wrong answer for the short
workloads, and finding that is the most transferable result here.

Comparing the *same* configuration measured once (coverage) and five times
(validation) showed disagreements of up to **5.2×** on `R_long_prefill`. The
cause was not sampling noise but **first-touch effects** — Triton JIT and
radix-cache population — in workloads whose measurement window is under a second:

| workload | run length | drift rep0 → rep4, no warm-up |
|---|---:|---:|
| `R_long_prefill` | 0.33 s | **+36.5 %** |
| `shared_prefix` | 20 s | +9.2 % |
| `R_medium_balanced` | 2.8 s | +2.1 % |
| `R_concurrent_decode` | 5.3 s | +1.5 % |
| `R_short_decode` | 7.0 s | +0.9 % |
| `tool_agent` | 42 s | +0.7 % |

Discarding repetition 0 was **not** sufficient. The fix is explicit unscored
warm-up passes budgeted by measured drift (4 passes for `R_long_prefill`, 2 for
medium/concurrent, 1 for short/shared-prefix, 0 for the already-steady
tool-agent trace). After the fix:

* drift falls to **≤1.2 %** on every workload;
* the mean relative 95 % CI half-width over 5 repetitions is **0.1–2.6 %**;
* and, decisively, the warmed **1-repetition coverage now agrees with the warmed
  5-repetition validation across all 372 shared cells**:

| workload | coverage / validation (mean) | std | min | max |
|---|---:|---:|---:|---:|
| `R_short_decode` | 1.000 | 0.003 | 0.988 | 1.005 |
| `R_medium_balanced` | 0.999 | 0.008 | 0.966 | 1.022 |
| `R_long_prefill` | 0.991 | 0.044 | 0.889 | 1.161 |
| `R_concurrent_decode` | 1.000 | 0.012 | 0.942 | 1.043 |
| `shared_prefix` | 0.990 | 0.023 | 0.913 | 1.028 |
| `tool_agent` | 0.992 | 0.014 | 0.947 | 1.003 |

Both un-warmed passes are retained in full as evidence, not deleted:
`results/2026-07-24_serving_ceiling_nowarmup/` and
`results/2026-07-24_serving_ceiling_validation_nowarmup/`.

> **Lesson to carry forward:** a benchmark must be shown to be in steady state
> *before* any configuration comparison is made. Sub-second measurement windows
> are especially dangerous, and dropping the first repetition does not fix them.

---

## 1. Headline

> Serving-level tuning removes workload-specific configuration cliffs and lets
> you choose a point on a latency/throughput frontier. It does **not** produce
> one configuration that is best across workloads, and in the regimes where the
> cookbook already matches the traffic the entire 192-point grid moves request
> throughput by **under 3 %**.

---

## 2. Per-regime results — coverage pass, 192 configurations per model

### LFM2.5-8B-A1B

| regime | winning knobs | req-thr | TTFT p95 | TPOT p95 | class | worst cfg | % cfgs beating cookbook | % dominated by cookbook |
|---|---|---:|---:|---:|:--:|---:|---:|---:|
| short decode | cap96 · chunk8192 · fcfs · mem0.90 | +0.4 % | +6.0 % | −0.0 % | WIN | −1.9 % | 18.8 % | 32.8 % |
| medium balanced | cap16 · chunk2048 · lpm · mem0.75 | +2.6 % | **−8.4 %** | −0.1 % | REGRESSION | −7.0 % | 16.1 % | 63.0 % |
| **long prefill** | cap24 · chunk2048 · fcfs · mem0.75 | **+77.5 %** | +64.6 % | +3.3 % | WIN | −19.2 % | 71.9 % | 25.0 % |
| concurrent decode | cap48 · chunk−1 · fcfs · mem0.75 | +1.6 % | +3.9 % | +1.5 % | WIN | **−64.9 %** | 8.9 % | 79.7 % |
| **shared-prefix** | cap96 · chunk2048 · lpm · mem0.75 | **+94.1 %** | +94.8 % | −3.8 % | TRADE-OFF | −53.9 % | 69.8 % | 17.7 % |
| tool-agent | cap48 · chunk2048 · lpm · mem0.80 | +0.4 % | −1.1 % | **−221 %** | REGRESSION | −1.1 % | 46.4 % | 35.4 % |

### Qwen3-30B-A3B

| regime | winning knobs | req-thr | TTFT p95 | TPOT p95 | class | worst cfg | % cfgs beating cookbook | % dominated by cookbook |
|---|---|---:|---:|---:|:--:|---:|---:|---:|
| short decode | cap48 · chunk−1 · fcfs · mem0.85 | +0.8 % | +2.3 % | +0.4 % | FLAT | −0.9 % | 58.3 % | 34.9 % |
| medium balanced | cap96 · chunk8192 · fcfs · mem0.80 | +1.2 % | +23.1 % | −0.0 % | WIN | −0.7 % | 82.3 % | 4.7 % |
| **long prefill** | cap64 · chunk−1 · lpm · mem0.75 | **+19.7 %** | +35.2 % | +16.6 % | WIN | −11.5 % | 76.0 % | 23.4 % |
| concurrent decode | cap128 · chunk−1 · fcfs · mem0.85 | +0.8 % | +14.0 % | +0.0 % | WIN | **−61.9 %** | 41.7 % | 52.1 % |
| **shared-prefix** | cap128 · chunk8192 · lpm · mem0.75 | **+25.5 %** | +84.7 % | **−46.2 %** | TRADE-OFF | −63.9 % | 55.7 % | 43.8 % |
| tool-agent | cap64 · chunk8192 · lpm · mem0.90 | +0.2 % | −5.3 % | **−38.0 %** | REGRESSION | −16.4 % | 19.3 % | 52.1 % |

### Validation pass (5 repetitions, CI-backed) — the same structure holds

| regime | LFM2.5 req-thr | class | Qwen req-thr | class |
|---|---:|:--:|---:|:--:|
| short decode | +0.4 % | WIN | +0.9 % | WIN |
| medium balanced | +1.8 % | REGRESSION | +0.9 % | WIN |
| long prefill | +56.9 % | TRADE-OFF | +19.6 % | WIN |
| concurrent decode | +1.1 % | FLAT | +0.5 % | WIN |
| shared-prefix | **+93.6 %** | TRADE-OFF | **+20.9 %** | TRADE-OFF |
| tool-agent | +0.3 % | WIN | +0.0 % | REGRESSION |

The two passes agree on every qualitative conclusion: two regimes with a real
cliff (shared-prefix, long-prefill), three saturated regimes (~1 %), and
tool-agent as the honest negative.

---

## 3. What the numbers actually say

### 3.1 Three of six regimes are saturated

`short decode`, `concurrent decode` and `tool agent` top out at **+0.2 % to
+1.6 %** on both models. In `concurrent decode` **80 % (LFM) / 52 % (Qwen)** of
all 192 configurations are dominated by the cookbook. Serving configuration has
essentially nothing left to give there.

### 3.2 Two regimes have a genuine cliff — and it is a *capacity* cliff

`shared_prefix` (+94.1 % / +25.5 %) and `long_prefill` (+77.5 % / +19.7 %) are
where the cookbook is mismatched. Both winners raise `max_running_requests`
above the cookbook's 32 and enable chunking. **This is a multi-knob
configuration and must never be attributed to chunked prefill alone.**

### 3.3 The shared-prefix win is a trade-off, not a free lunch

The Qwen shared-prefix throughput winner buys +25.5 % request throughput and
+84.7 % TTFT p95 at the cost of **−46.2 % TPOT p95**: aggregate tokens arrive
sooner, but each individual stream decodes slower. Reported as TRADE-OFF, never
as a clean win.

### 3.4 The honest negative: optimising throughput can destroy TPOT

On `tool_agent` — the only *real* trace in the suite — the throughput winner
gains **+0.4 % (LFM) / +0.2 % (Qwen)**, which is inside the noise band, while
degrading TPOT p95 by **221 % / 38 %**. Chasing a single objective on an agentic
workload produces a configuration that is strictly worse for the user.

### 3.5 The downside is an order of magnitude larger than the upside

| regime | best gain | worst loss |
|---|---:|---:|
| LFM concurrent decode | +1.6 % | **−64.9 %** |
| Qwen concurrent decode | +0.8 % | **−61.9 %** |
| Qwen shared-prefix | +25.5 % | **−63.9 %** |
| LFM shared-prefix | +94.1 % | **−53.9 %** |

Serving knobs destroy performance far more easily than they improve it. The
dominant cliff driver is `max_running_requests = 8`, which starves batching —
the same mechanism the v48 plateau study isolated independently on LFM2.5.

---

## 4. Cross-regime transfer — the strongest evidence against a universal config

Because one server launch evaluates all six workloads, every cell of the
transfer matrix is a real measurement. Ratios are against **each target
regime's own cookbook**, so >1.00× always means better.
Source: `analysis/*/transfer_matrix_*.csv`.

Three structural findings, stable across both models and both passes:

1. **A regime winner can be catastrophic elsewhere.** The low-capacity
   long-prefill winner collapses to **0.36×** under concurrent decode.
2. **Off-diagonal transfer is almost never > 1.00×**, with one systematic
   exception: the `shared_prefix` column, where nearly any higher-capacity
   configuration beats a badly mismatched cookbook.
3. **Diagonal gains are small except at the two cliffs**, which is the
   quantitative statement of "no universal configuration".

---

## 5. Cross-model comparison

Identical protocol, so ratios are comparable (raw values are never merged):

| regime | LFM2.5 | Qwen |
|---|---:|---:|
| short decode | +0.4 % | +0.8 % |
| medium balanced | +2.6 % | +1.2 % |
| long prefill | **+77.5 %** | **+19.7 %** |
| concurrent decode | +1.6 % | +0.8 % |
| shared-prefix | **+94.1 %** | **+25.5 %** |
| tool-agent | +0.4 % | +0.2 % |

The reachable gain is **regime-dominated but model-scaled**: both models agree
on *which* regimes are saturated and which have cliffs, yet disagree by 3–4× on
*how large* those cliffs are. A configuration recommendation is therefore not
safe to transfer across models even when the regime matches. This independently
reproduces the v7 finding that the same chunking candidate helps LFM2.5
shared-prefix (+28.6 % req/s) while being neutral on Qwen (−2.9 %).

---

## 6. What this means for the performance-gap story

1. The cookbook is a **fair** baseline, not a strawman: in the saturated regimes
   the full 192-configuration grid cannot beat it by more than ~1.6 %, and most
   configurations are strictly worse.
2. Serving tuning matters where the workload has a **structural mismatch**
   (shared-prefix, long-prefill here; high concurrency in the 2026-07-23 stress
   study). It is mostly a way to **avoid cliffs** rather than to gain speed.
3. Because the ceiling is ~1 % in three of six regimes and every cliff fix is
   regime-specific, the residual end-to-end gap **cannot** be closed by serving
   configuration. That is the hand-off to profiling: kernels, backends,
   scheduling internals and communication.

**Do not write:** "serving tuning has no value" or "the serving optimization
space is exhausted".
**Do write:** "serving tuning removes workload-specific configuration cliffs and
selects points on a latency/throughput frontier, but it does not provide a
universal configuration or consistently move the end-to-end frontier outward."

---

## 7. Provenance and limitations

* Environment (`environment.json`): 1× H200, TP1, BF16, sglang 0.5.12.post1 @
  `17f7a1da1`, torch 2.9.1+cu128, triton 3.5.1, CUDA 12.8, driver 580.105.08.
  Resolved path verified **per configuration** from the server log:
  `attention_backend=fa3`, `moe_runner_backend=auto`, CUDA graph captured.
* `chunked_prefill_size = 8192` is effectively equal to `−1` at
  `--context-length 8192`; those two levels act as a built-in duplicate and give
  a free noise estimate. Only `chunk = 2048` chunks the long inputs.
* Coverage cells are single measurements taken **after** warm-up; the validation
  pass supplies 5-repetition means and 95 % CIs for the selected configurations.
* Failures: 10 launch failures were recorded across both passes, all traced to a
  port-rebind race (`[Errno 98] address already in use`), classified as
  infrastructure, retried and completed. No configuration was ever fake-scored.
  Final state: coverage 384/384, validation 62/62, **0 unresolved failures**.
