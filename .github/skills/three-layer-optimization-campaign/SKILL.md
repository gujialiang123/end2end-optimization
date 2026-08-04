---
name: three-layer-optimization-campaign
description: Run the full serving-config / kernel-config / kernel-rewrite factorial on a new model, producing a 6-regime x 8-cell ablation matrix that tests whether kernel rewriting still pays on top of both tuning ceilings.
version: 1
stage: [1, 2, 3]
inputs:
  - model: model key, registered in both serving_ceiling_lib.MODELS and lf_lib.MODELS
  - control_model: an already well-optimized model for the audit control (default: qwen)
  - gpus: list of free GPU ids
outputs:
  - ceiling_per_regime.json       # L1: per-regime serving ceiling + latency
  - audit.json                    # L2/L3: per-kernel gap counts vs control
  - exp3_layered_*_summary.json   # the 2^3 factorial cells
  - matrix.md                     # the deliverable table
triggers:
  - "a second or third model is needed to establish that a single-model result generalizes"
  - "someone asks whether kernel rewriting still pays after autotuning"
  - "a new model family is added and its optimization headroom is unknown"
depends_on: [fusion-gap-hunting, e2e-bench-runner, noise-aware-scoring, regime-bench-harness]
---

# three-layer-optimization-campaign

## WHEN

Run this when the question is **not** "make this model faster" but **"does the
layered claim hold on another model"**.

The claim under test, established on LFM2.5-8B-A1B:

> After both serving config and kernel config are tuned to their measured
> ceiling, kernel rewriting still contributes **+6.4 % to +8.4 %** end-to-end,
> significant in four independent regimes.

The evidence form is a **6 regime x 8 column matrix** (the 2^3 factorial of
L1/L2/L3). The row that carries the argument is the kernel-rewrite increment
measured on four different baselines:

| L3 increment on | cookbook | L2 | L1 | L1+L2 |
|---|---:|---:|---:|---:|
| low-batch decode | +6.70 % | +6.35 % | +7.46 % | +7.35 % |
| concurrent decode | +6.72 % | +6.86 % | +6.62 % | +7.14 % |
| long prefill | +6.18 % | +9.73 % | +6.26 % | +6.38 % |
| medium balanced | +8.29 % | +8.13 % | +8.40 % | +8.38 % |

**Reading across is flat. That is the claim, not any single number.**

Success on a new model is **not** "also got +6 %". It is "is that row also flat".
If the increment decays as the baseline improves, that is an equally publishable
result: it bounds the claim. Report it either way.

## WHY

Three failure modes this skill exists to prevent, all of which were hit during
the LFM2.5 campaign.

**A gain measured on a dirty baseline is not the gain you can claim.** The
original kernel work reported +5.30 % on long prefill against a tree that
shipped no tuned MoE config at all. Re-measured on a baseline that had it, the
same change was worth **+9.73 %**. The direction was not the one anyone
predicted. **The baseline is not background; it is part of what you measure.**

**Sequential A/B is not a controlled experiment.** The harness runs one arm per
server lifetime. Measuring only one order produced **-0.37 % at p=4.9e-04** — a
statistically significant regression that reversed to **+0.12 %** when the arms
were swapped. Whichever ran first was faster. Counterbalancing is not optional.

**Throughput is the wrong lens on real traces.** On the mooncake tool-agent
trace, the full three-layer stack moves request throughput by **+0.48 %** and
TTFT p95 from **537 ms to 218 ms (-59 %)**. A throughput-only report concludes
"the optimization does nothing".

## CANDIDATES

**Do not pick a model from scratch.** An 11-model cross-architecture audit
already ranked the L3 headroom
(`results/lfm_fusion/processed/cross_architecture_audit_summary.csv`).
`removable_pct` is the share of kernel time spent in kernels a fused
implementation would never launch.

| model | arch | layers | removable | all gaps | L2 possible | pick |
|---|---|---:|---:|---:|:--:|:--:|
| gemma3 (1B) | dense + sliding-window attn | 26 | **37.06 %** | 46.32 % | no | already harvested |
| **olmo2 (1B)** | dense (AllenAI) | 16 | **14.71 %** | **27.74 %** | no | **yes** |
| exaone4 (1.2B) | dense (LG) | 30 | 3.54 % | 15.66 % | no | maybe |
| phi4mini (3.8B) | phi3 dense | 32 | 6.43 % | 13.87 % | no | maybe |
| *lfm25 (8B)* | *MoE + gated short conv* | *24* | *4.06 %* | *11.31 %* | *yes* | *done* |
| **olmoe (1B-7B)** | **MoE 64E (AllenAI)** | 16 | 0.43 % | 4.70 % | **yes** | **yes** |
| qwen3next / qwen06 / granite / qwen32 | — | — | < 0.5 % | < 0.7 % | — | no |
| qwen (30B) | MoE + full attn | 48 | 0.18 % | 0.23 % | yes | **control** |

**The recommended pair is olmo2 + olmoe**, because it is a controlled contrast:
same family, same 16 layers, same 2048 hidden size, same 4096 context — the only
difference is dense versus 64-expert MoE. That isolates whether L2's
applicability and payoff are determined purely by the presence of MoE.

olmoe's L2 prospect has been checked: its MoE shape is `E=64, N=1024`, and the
only upstream config matching it is
`triton_3_1_0/E=64,N=1024,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8`.
**No H200, no bf16** — the same gap shape that was worth +23.3 % on LFM2.5.
Its L3 headroom is only 4.70 % though, so expect a strong-L2 / weak-L3 model,
which is a useful contrast rather than a problem.

**gemma3 has the largest headroom in the table but is not a fresh target**: that
gap was already found and fixed in 2026-07 (`Gemma3RMSNorm.forward_cuda` running
eager PyTorch; a one-line fall-through, 2.07x / 1.75x / 1.57x end-to-end). Cite
it as a completed case rather than spending GPU time on it.

**phi4mini is the only candidate above 3B.** If the concern is that the result
only holds on small models, it is the cheapest rebuttal.

⚠️ **Every candidate here is 1-3.8B, against LFM2.5's 8B.** `R_long_prefill`
already measures in a 0.31 s window on LFM2.5; a 1B model may drive it to tens
of milliseconds, below the noise. Measure `dur_s` in stage 0 and, if that
workload's window is under ~0.5 s, either raise `--num-prompts` (which changes
the workload definition, so re-measure the whole row) or drop that regime for
that model and say why.

## HOW

Six stages, ~20-25 GPU-hours serially, 6-8 h wall clock across several GPUs.

Full procedure, with the rationale for every step:
`docs/2026-08-04/METHODOLOGY_three_layer_optimization.md`.

### Stage 0 — feasibility (~1 h, do not skip)

```bash
python -m sglang.launch_server --model-path <PATH> --port <P> --tensor-parallel-size 1
python scripts/lfm_fusion/lf_audit.py --model <M> --regime C_long_prefill --gpu <G>
```

Decide **whether L2 applies at all**. L2 retunes the tile parameters of a
config-driven hot kernel. On LFM2.5 that is `fused_moe_kernel`, driven by a JSON
config file, so swapping configs is a clean switch.

**A dense model has no such kernel.** OLMo-2 is `Olmo2ForCausalLM` with no MoE:
the mechanism does not exist. Look for an alternative (Triton attention backend
tiles, flashinfer autotune) and if there is none, **record that L2 does not
apply and produce a 2^2 matrix**. An honest four-cell row beats a contrived
eight-cell one. That L2's applicability is architecture-dependent while L1 and
L3 are not is itself a finding.

Also measure `dur_s` here. Small models finish the synthetic workloads in
fractions of a second; see FAILURE MODES.

### Stage 1 — L1 serving ceiling (~6 h)

Full grid, 8 x 3 x 2 x 4 = **192 configurations**, then a 5-repetition
validation pass over the top 35.

```bash
python scripts/run_serving_ceiling_campaign.py --init --models <M>
python scripts/run_serving_ceiling_campaign.py --gpu <G> --worker w<G>
```

**Enumerate, do not sample.** A 25-trial TPE study on this space bound its first
seven trials to a bad batching setting and never revisited the good region,
reporting a ceiling **6 % below the cookbook**. A reviewer will read that as a
failed search rather than a ceiling. Exhaustive enumeration has no such hole.

**Record TTFT p95 and TPOT p95 alongside throughput.** On LFM2.5 the two large
serving wins were not throughput-for-latency trades: long prefill gained 56.9 %
while TTFT p95 fell from 208 ms to 94 ms, and shared prefix gained 93.6 % while
TTFT p95 fell from 7.4 s to 389 ms. Throughput alone would have mislabeled both.

### Stage 2 — L3 gap audit (~1 h, highest leverage)

**Count kernels a fused implementation would never launch. Do not rank by time
share.** Time share points at the MoE, which is where everyone already looks and
where no headroom remains.

```bash
python scripts/lfm_fusion/lf_audit.py --model <M>             --regime <R> --gpu <G>
python scripts/lfm_fusion/lf_audit.py --model <control_model> --regime <R> --gpu <G>
```

**The control is what makes the count mean anything:**

| model | unfused RMSNorm | standalone residual add | gating mul |
|---|---:|---:|---:|
| LFM2.5 | 61 | 48 | 36 |
| Qwen3-30B (control) | 1 | 0 | 0 |

Two signals decide it:

- **the counts divide by the layer count** — `48 = 2 x 24`, `36 = 2 x 18`. Exact
  divisibility means every layer makes the same mistake, so it is an
  implementation omission rather than an accident.
- **the difference from the control** — the difference is the signal, never the
  absolute number.

Add the static scan, which needs no GPU and runs in seconds:

```bash
cd /home/t-jialianggu/work/SLO-agent
PYTHONPATH=$PWD/src python -m sglang_agent_kernel_lab.cli scan --framework-src <sglang>
```

It encodes the signature that found the two largest LFM2.5 wins: **enumerate the
fused primitives the codebase already ships, then check which models' call sites
fail to use them.** Precision is low — 3 of 32-40 `never_wired` candidates were
real — so it produces candidates, not verdicts. See `fusion-gap-hunting`.

Then read the source. **The question is whether this is an optimization
opportunity or a bug.** The largest LFM2.5 win was the latter, and it took three
non-adjacent facts held at once: the layer signature accepts a `residual`
argument and overwrites it on the first line; `RMSNorm.forward_cuda(x, residual)`
already dispatches to `fused_add_rmsnorm`; and the model loop already threads a
residual between layers. **A parameter declared, passed, and discarded is a bug.**

### Stage 3 — implement and validate L3 (~4 h)

**Copy, do not invent.** Find how a correctly-wired model does it first. The
LFM2.5 fix was `models/llama.py:304-316` applied verbatim.

**Toggle by environment variable so the baseline is the byte-identical stock
path** — same tree, same commit, same server args. Do not build a second
worktree; that is how the Gemma-3 run hit a stride mismatch the attention
backend rejected outright.

Two things that will bite:
- model classes are **lazily imported** by the registry, so a timer-based patch
  is a race. Use a `sys.meta_path` finder that patches the instant the module
  finishes executing.
- **check the server log for a patch marker.** A silently inactive patch
  otherwise records as "identical to baseline" and reads as "this optimization
  does nothing".

Validate each component end-to-end **as it lands**, not in a batch. Isolated
speedup and end-to-end gain are different quantities.

**Hand-written kernels need shape guards, and the thresholds must be swept, not
guessed.** The two LFM2.5 kernels have opposite dependence: the short-conv pair
needs `T >= 2048` to amortize Triton's ~30 us launch floor, while the MoE
reduction saves that launch plus a round trip and so wins at `T <= 32` or
`T >= 4096`. **Guards can be defeated by the serving config**: on shared prefix
the L1 winner sets `chunked_prefill_size=2048`, landing on the conv guard
boundary and collapsing the kernel gain from +7.24 % to +1.84 %.

### Stage 4 — L2, if it applies (~4 h)

Sweep 468-894 candidates per token-count bucket, **correctness-gate every
candidate before timing it** (~9000 configurations, 0 failures on LFM2.5), and
emit a **guarded** profile that specializes only buckets where an oracle proved
headroom, leaving the rest field-for-field identical to the default.

Three errors that cost a full iteration each:
1. tuning a kernel variant the server never executes;
2. CUDA-graph capture bakes the config in at capture time, so decode cannot be
   retuned afterwards;
3. **`M` is the token count, not `tokens x top_k`** — the profile keys were off
   by a factor of `top_k`, hiding real headroom behind misaligned buckets. Only a
   live trace exposed it.

### Stage 5 — the 2^3 factorial (~7 h)

One invocation is a 2x2x2 at a fixed serving config and fills **four cells of a
row**. Two invocations per regime fill the row and make it internally
consistent.

```bash
GPU=<G> REPS=8 PORT=<P> REGIME=<X> bash scripts/lfm_fusion/exp3_layered.sh
WARMUP=<W> REPS=<R> SUITE=l1_ GPU=<G> PORT=<P2> REGIME=<X>_tuned \
    bash scripts/lfm_fusion/exp3_layered.sh
```

The `_tuned` regime carries the L1 winner knobs from stage 1.

## OUTPUT CONTRACT

```
results/<date>_serving_ceiling_validation/analysis/<model>/ceiling_per_regime.json
  {<workload>: {n_configs_validated: int,
                cookbook: {req_per_s, ci95, ttft_p95_ms, tpot_p95_ms},
                ceiling:  {hash, req_per_s, ci95, ttft_p95_ms, tpot_p95_ms},
                gain_over_cookbook: float}}

results/lfm_fusion/e2e/exp3_layered_<regime>_summary.json
  {metric: "request_throughput", regime: str,
   cells: {"<nocfg|cfg>_<fwd|rev>": {"<arm>": [float, ...]}}}
```

Every reported cell must state **n, the Welch t or p, and that both arm orders
are pooled**. A cell measured in one order only is not a cell.

## FAILURE MODES

**Measurement window shorter than the noise.** Measured windows:
`tool_agent` 38 s, `R_short_decode` 4.7 s, `shared_prefix` 8.7 s,
`R_concurrent_decode` 1.5 s, `R_long_prefill` **0.31 s**, and
`R_long_prefill` at the L1 ceiling **0.20 s**. At 0.2 s a "9 % disagreement
between arm orders" is **sixteen milliseconds** — one GC, one scheduler tick.
Raising `--num-prompts` fixes it but **changes the workload definition**, so the
whole row must be re-measured and cannot be compared against existing cells.
**This gets worse on small models**: measure `dur_s` in stage 0.

**Leaked servers.** `lf_e2e.py` spawns with `setsid`, so Ctrl-C does not kill
the server. The next run's health check will happily benchmark the stale
process. `assert_port_free` now catches this; after any interruption still run
`ps -u $(whoami) -o pid,cmd | grep launch_server`.

**Editing the driver while it runs.** bash reads scripts by incremental byte
offset; a mid-run edit kills it with `unexpected EOF`. Three cells were lost
this way.

**Warm-up calibrated for a different serving config.** `WARMUP_RUNS` was tuned
on the cookbook knobs. At `cap8/chunk2048/fcfs/mem0.9` the first two scored
repetitions were still climbing from 20 to 23 req/s under its default, which
reads as "the kernel gain vanished". Always pass `WARMUP` explicitly for
`*_tuned` regimes.

**Token-identity as a correctness gate.** With top-k routing, expert selection
is a discrete argmax, so any algebraically-equivalent-but-not-bit-exact change
can flip an expert and change the output discontinuously — 11/12 prompts
top-1-identical but KL up to 0.99. **Do not lower the gate; decide whether it is
structurally unusable and replace it.** Dense models do not have this problem;
try token-identity first there. When falling back to a task metric, include a
**provably bit-exact arm** as a free noise ruler: the LFM2.5 `scale` arm must
equal baseline by construction and read 0.8 points lower, which establishes the
between-arm noise floor without assuming one.

**Shared GPUs.** This host is multi-tenant. Confirm the cards are idle before
launching; running alongside someone else corrupts both sets of measurements.

## ROADMAP

- Stage 0's L2 feasibility check is manual. It should become a scan for
  config-file-driven kernels in the framework tree.
- The static scanner's `never_wired` precision is ~10 %. The `path_guarded`
  shape is much better and is the one that found OLMo-2.
- Nothing currently automates step one of the method — questioning the boundary
  of an existing conclusion. That remains the highest-value and least
  mechanizable step.
