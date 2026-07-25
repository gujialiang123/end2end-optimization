# Qwen + LFM2.5 serving-ceiling campaign — results

**Date:** 2026-07-24 · **Campaign:** `results/2026-07-24_serving_ceiling/`
**Coverage pass:** 384/384 tasks complete (2 models × 192 configurations),
**0 unresolved failures**, 2 304 per-run measurements, one streaming client.

Methodology: `docs/2026-07-24/qwen_serving_ceiling_methodology.md`
Audit of prior data: `docs/2026-07-24/serving_tuning_data_audit.md`

Sign convention throughout:
`throughput delta = candidate/baseline − 1`, `latency delta = 1 − candidate/baseline`
(so a **positive latency delta always means lower latency**).

---

## 1. Headline

> Serving-level tuning removes workload-specific configuration cliffs and lets
> you pick a point on a latency/throughput frontier. It does **not** produce one
> configuration that is best across workloads, and in four of the twelve
> model × regime cells the reachable gain is under 2 %.

Two facts carry that claim, and both are measured on the full grid:

1. **Where the cookbook is already matched to the traffic, the ceiling is low.**
   In `R_medium_balanced` and `R_long_prefill` the best of 192 configurations
   beats the cookbook by **0.5 – 1.1 %** on request throughput, and **98 %** of
   configurations are dominated by the cookbook in `R_long_prefill`.
2. **Where it is mismatched, the cliff is large.** In `shared_prefix` the best
   configuration is **+78.6 %** (LFM2.5) and **+27.7 %** (Qwen) request
   throughput, with TTFT p95 **−84 %** and **−80 %**.

---

## 2. Per-regime results (coverage pass, best-throughput configuration)

### LFM2.5-8B-A1B

| regime | winning knobs | req-thr | TTFT p95 | TPOT p95 | class | worst cfg | median cfg | % cfgs that beat cookbook | % dominated by cookbook |
|---|---|---:|---:|---:|:--:|---:|---:|---:|---:|
| short decode | cap128 · chunk8192 · fcfs · mem0.90 | **+1.5 %** | +25.8 % | −0.2 % | WIN | −9.2 % | −0.5 % | 14.6 % | 79.7 % |
| medium balanced | cap48 · chunk−1 · fcfs · mem0.90 | +1.1 % | −1.1 % | +0.2 % | FLAT | −12.2 % | −3.6 % | 6.8 % | 90.1 % |
| long prefill | cap8 · chunk−1 · lpm · mem0.90 | +0.5 % | +0.7 % | +0.5 % | FLAT | −60.3 % | −9.3 % | 1.0 % | 97.9 % |
| concurrent decode | cap64 · chunk8192 · fcfs · mem0.75 | **+2.9 %** | +19.1 % | +0.8 % | WIN | −67.8 % | −3.6 % | 33.9 % | 63.0 % |
| **shared-prefix** | cap96 · chunk2048 · lpm · mem0.90 | **+78.6 %** | **+84.0 %** | −21.5 % | TRADE-OFF | −54.8 % | **+21.1 %** | 65.1 % | 21.9 % |
| tool-agent | cap128 · chunk8192 · lpm · mem0.85 | +0.5 % | +1.7 % | **−82.6 %** | REGRESSION | −1.1 % | −0.0 % | 40.6 % | 34.4 % |

### Qwen3-30B-A3B

| regime | winning knobs | req-thr | TTFT p95 | TPOT p95 | class | worst cfg | median cfg | % cfgs that beat cookbook | % dominated by cookbook |
|---|---|---:|---:|---:|:--:|---:|---:|---:|---:|
| short decode | cap96 · chunk−1 · fcfs · mem0.75 | **+5.3 %** | **+85.5 %** | +0.4 % | WIN | −1.0 % | +4.7 % | **99.0 %** | 1.0 % |
| medium balanced | cap8 · chunk−1 · fcfs · mem0.75 | +0.7 % | +1.5 % | +0.4 % | FLAT | −24.6 % | −1.5 % | 14.6 % | 69.8 % |
| long prefill | cap96 · chunk−1 · lpm · mem0.80 | +0.5 % | +1.3 % | +2.4 % | FLAT | −72.4 % | −12.1 % | 1.0 % | 97.9 % |
| concurrent decode | cap64 · chunk−1 · fcfs · mem0.85 | +1.0 % | +17.4 % | +0.1 % | WIN | −70.6 % | −2.4 % | 23.4 % | 74.0 % |
| **shared-prefix** | cap96 · chunk2048 · fcfs · mem0.90 | **+27.7 %** | **+79.6 %** | −31.7 % | TRADE-OFF | −64.2 % | +1.8 % | 58.3 % | 41.1 % |
| tool-agent | cap24 · chunk8192 · lpm · mem0.90 | +1.6 % | **+68.6 %** | −16.3 % | TRADE-OFF | −22.6 % | +1.2 % | 87.0 % | 13.0 % |

### How to read the classifications

* **WIN** — the primary metric improves and no guardrail metric regresses.
* **TRADE-OFF** — at least one primary metric improves *and* another worsens.
  In `shared_prefix` the throughput winner buys +78.6 % request throughput at
  the cost of **−21.5 % TPOT p95** (LFM) / **−31.7 %** (Qwen): tokens arrive
  sooner in aggregate but each stream is slower.
* **REGRESSION** — the LFM `tool_agent` throughput winner gains a statistically
  meaningless +0.5 % while degrading TPOT p95 by **82.6 %**. This is the honest
  negative result: chasing request throughput on an agentic trace can wreck the
  per-token latency that an agent loop actually feels.
* **FLAT** — `medium balanced` and `long prefill` on both models. The cookbook
  is already near-optimal; the whole 192-point grid moves it by ≤1.1 %.

---

## 3. The search space is not uniformly good — it contains cliffs

The `worst cfg` column is as important as the winner. Serving knobs can destroy
performance far more easily than they improve it:

| regime | best gain | worst loss |
|---|---:|---:|
| LFM long prefill | +0.5 % | **−60.3 %** |
| LFM concurrent decode | +2.9 % | **−67.8 %** |
| Qwen long prefill | +0.5 % | **−72.4 %** |
| Qwen concurrent decode | +1.0 % | **−70.6 %** |
| Qwen shared-prefix | +27.7 % | **−64.2 %** |

The downside is one to two orders of magnitude larger than the upside in the
regimes where the cookbook is already well matched. The dominant cliff driver is
`max_running_requests = 8`, which starves batching — the same mechanism the v48
plateau study isolated on LFM2.5.

---

## 4. Cross-regime transfer: the strongest evidence against a universal config

Every configuration was measured on all six workloads from a single server
launch, so the transfer matrix uses real measurements rather than extrapolation.
Ratios are computed against **each target regime's own cookbook**, so >1.00×
always means better.

### LFM2.5 — request-throughput transfer (`analysis/lfm25/transfer_matrix_request_throughput.csv`)

| source config ↓ / target → | short dec | med bal | long pref | conc dec | shared-pfx | tool-agent |
|---|---:|---:|---:|---:|---:|---:|
| cookbook | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| short-decode winner | 1.01 | 0.93 | 0.92 | 1.00 | 1.51 | 1.00 |
| medium-balanced winner | 1.00 | 1.01 | **0.77** | 0.99 | 1.08 | 1.00 |
| long-prefill winner | 1.00 | 1.00 | 1.00 | **0.36** | **0.46** | 0.99 |
| concurrent-decode winner | 1.00 | 0.96 | **0.71** | 1.03 | 1.55 | 0.99 |
| shared-prefix winner | 1.00 | 0.96 | 0.92 | 0.90 | **1.79** | 1.00 |
| tool-agent winner | 1.00 | 0.95 | 0.93 | 1.01 | 1.49 | 1.00 |

Three conclusions, all visible in one table:

1. **A winner can be catastrophic elsewhere.** The long-prefill winner
   (`cap=8`) is optimal for its own regime and delivers **0.36×** — a 64 % loss —
   under concurrent decode, and 0.46× on shared-prefix.
2. **Off-diagonal gains are almost always ≤ 1.00×.** Apart from the
   shared-prefix column (where *any* higher-capacity config helps because the
   cookbook is badly mismatched there), no source configuration improves a
   regime it was not tuned for.
3. **Diagonal gains are small except for shared-prefix.** 1.01× / 1.01× / 1.00×
   / 1.03× / 1.79× / 1.00×. Only one of six regimes has real headroom.

The same structure holds for Qwen and for the TTFT p95 / TPOT p95 / E2E p95
transfer matrices (`analysis/*/transfer_matrix_*.csv`).

---

## 5. Cross-model comparison

Both models were run under an identical protocol, so the *ratios* are
comparable (raw values are not merged):

| regime | LFM2.5 | Qwen |
|---|---:|---:|
| short decode | +1.5 % | +5.3 % |
| medium balanced | +1.1 % | +0.7 % |
| long prefill | +0.5 % | +0.5 % |
| concurrent decode | +2.9 % | +1.0 % |
| shared-prefix | **+78.6 %** | **+27.7 %** |
| tool-agent | +0.5 % | +1.6 % |

The reachable gain is **regime-dominated, not model-dominated**: both models
agree that long-prefill and medium-balanced are saturated and that shared-prefix
has a cliff — but they disagree by a factor of ~3 on how big that cliff is. A
configuration recommendation transferred from one model to another is therefore
not safe even when the regime matches. This independently reproduces the v7
observation that the same chunking candidate helps LFM2.5 shared-prefix
(+28.6 % req/s) and is neutral on Qwen (−2.9 %).

---

## 6. What this means for the performance-gap story

1. The cookbook is a *fair* baseline, not a strawman: in 4 of 12 cells the full
   192-configuration grid cannot beat it by 2 %.
2. Serving tuning is worth doing where the workload has a structural mismatch
   (shared-prefix here; high concurrency in the 2026-07-23 stress study), and it
   is mostly a way to **avoid cliffs** rather than to gain speed.
3. Because the six-regime ceiling is ~1 % in the saturated regimes, the
   remaining end-to-end gap cannot be closed by serving configuration. That is
   the hand-off to profiling: kernels, backends, scheduling policy internals and
   communication.

**Do not write:** "serving tuning has no value", or "the serving optimization
space is exhausted". **Do write:** "serving tuning removes workload-specific
configuration cliffs and selects points on a latency/throughput frontier, but it
does not provide a universal configuration or consistently move the end-to-end
frontier outward."

---

## 7. Provenance and limitations

* Environment: `results/2026-07-24_serving_ceiling/environment.json` — H200,
  TP1, BF16, sglang 0.5.12.post1 @ `17f7a1da1`, torch 2.9.1+cu128, triton 3.5.1,
  CUDA 12.8, driver 580.105.08. Resolved path verified per config from server
  logs: `attention_backend=fa3`, `moe_runner_backend=auto`, CUDA graph captured.
* **Numbers in §2–§5 are single-measurement coverage values.** A 5-repetition
  validation pass over the selected configurations (cookbook, per-regime
  throughput/output/TTFT/TPOT/E2E winners, a Pareto sample and clear
  regressions; 62 unique configs) provides the confidence intervals used for
  final claims — see `results/2026-07-24_serving_ceiling_validation/`.
* `chunked_prefill_size = 8192` is effectively equal to `−1` at
  `--context-length 8192`, so those two levels act as a built-in duplicate and
  give a free noise estimate. Only `chunk = 2048` chunks the long inputs.
* The synthetic regimes are short by construction (recovered v4 spec); their
  statistical strength comes from the repeated validation pass.
* Failures: 5 launch failures were recorded during the campaign, all traced to a
  port-rebind race (`[Errno 98] address already in use`), classified as
  infrastructure, requeued, and re-run to success. No configuration was ever
  fake-scored, and 18 per-run rows from interrupted configurations were dropped
  so that no partial configuration entered the matrix. Final state: 384/384
  complete.
