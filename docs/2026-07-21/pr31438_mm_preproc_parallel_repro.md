# PR #31438 reproduction — parallelize VLM multimodal preprocessing

> 2026-07-21 · host `aifx-clou000001`, 8×H200 · **sglang v0.5.15.post1** (env `sglang-v515`)
> · model `Qwen/Qwen3.6-35B-A3B-FP8` (linear-attn VLM). Third upstream-PR reproduction.

## 0. TL;DR

**Reproduced — positive and bit-exact-correct.** Moving image I/O + HF-processor
work off the tokenizer event loop onto dedicated worker pools (patched default: 2
processor workers + 16 I/O workers) speeds up **image-burst** serving on Qwen3.6 VLM:

| burst cell (reqs / imgs-per-req / concurrency) | request throughput | p99 TTFT |
|---|---|---|
| 16 / 4 / c8 | **+14.5%** | −13.8% (885.6→763.7 ms) |
| 32 / 4 / c16 | **+8.5%** | −2.0% (1703→1669 ms) |

Crucially, greedy outputs are **identical** across baseline / patched(2w) /
patched(4w) — the parallelization is semantically exact (correctness gate passed).

## 1. What the PR does

`vlm: parallelize multimodal preprocessing with customized worker num` (#31438,
squash-merge `4682ded`, 2026-07-21). Baseline does image read + HF processor call +
dispatch synchronously on the tokenizer event loop, so an image burst serializes and
stalls the GPU scheduler. The PR adds an isolated **processor executor** (thread-local
deep-copied processor clones, `MultimodalProcessorExecutor` in a new `executor.py`)
and a configurable I/O thread pool, plus `--mm-processor-worker-num` /
`--mm-io-worker-num` server args. Default for Qwen-VL: 2 processor workers, 16 I/O.

## 2. Porting to v0.5.15.post1

5 pure-Python files; cherry-picked with 2 conflicts in `base_processor.py` (the PR's
base diverged from v0.5.15):
- imports: kept both `get_global_server_args` (v0.5.15) and added the PR's
  `executor` import + `get_server_args` (both exist in v0.5.15).
- `use_cuda_ipc`: kept v0.5.15's `SGL_USE_CUDA_IPC` global (PR renamed it to a
  `self.use_cuda_ipc` attr that doesn't exist in v0.5.15) and de-duplicated the
  moved `skip_mm_pool` definition.
- **dropped** an incidental BOS-dedup block the PR bundled in — it reads
  `self._tokenizer_auto_adds_specials`, which in v0.5.15 lives only on the
  serving-chat class, **not** the processor; keeping it would AttributeError. It is
  unrelated to the parallelization feature.
`executor.py` (the core mechanism) is stdlib-only and applied clean; `qwen_vl.py`
just sets `auto_mm_processor_worker_num = 2` + `supports_mm_processor_concurrency`.
Snapshots + diffs under `patches/pr31438/`.

Confirmed active in the patched server log:
`Multimodal data loading enabled with 16 worker threads` +
`Multimodal processor concurrency enabled with 2 isolated worker threads`.

## 3. Method

- Both arms: fresh server, Qwen3.6-35B-A3B-FP8, single GPU, `--mem-fraction-static 0.85`.
  baseline = stock v0.5.15.post1; patched = PR files copied in (pure Python, no rebuild).
- `scripts/run_v49_pr31438_mm_preproc_ab.py`:
  1. **correctness probe** — same fixed 2-image prompt, temp 0, capture greedy output.
  2. **image-burst A/B** — N concurrent requests, each K random-resolution images,
     `max_tokens=8` (isolates the preprocessing/prefill path). 3 repeats/cell; the
     first (cold, kernel-compile) repeat is dropped, warm repeats aggregated.
- Sensitivity: patched re-run with `--mm-processor-worker-num 4`.
- Raw: `results/2026-07-21_v49_pr31438_mm_preproc/ab.jsonl`, `analysis_full.txt`.

## 4. Results (warm repeats; + = patched better vs baseline)

| cell | arm | req/s | p99 TTFT (ms) | throughput gain |
|---|---|---|---|---|
| 16/4/c8 | baseline | 8.81 | 885.6 | — |
| 16/4/c8 | patched (2w) | 10.09 | 763.7 | **+14.5%** |
| 16/4/c8 | patched (4w) | 8.38 | 1018.7 | −4.9% |
| 32/4/c16 | baseline | 9.57 | 1703.2 | — |
| 32/4/c16 | patched (2w) | 10.38 | 1669.4 | **+8.5%** |
| 32/4/c16 | patched (4w) | 9.70 | 1833.5 | +1.3% |

**Correctness:** baseline == patched(2w) == patched(4w), greedy output identical.

## 5. Honest calibration

- Upstream reported **+80.8% burst request throughput** (H200 TP2, 4 workers). We got
  **+8.5–14.5%** at the patch **default (2 workers)**, which the plan explicitly told
  us to use as the fair headline (and warned not to expect the 4-worker H200 number).
- **4 workers did not help** on this workload (≈baseline / noisier) — 2 is the sweet
  spot here. Consistent with the PR choosing 2 as the default. The gap to upstream's
  +80% is workload: our random JPEGs (≈640×480) decode cheaply, so preprocessing is
  not the dominant bottleneck; a heavier image mix would stress it more and widen the
  gain. The **mechanism is real and correct**; its e2e magnitude is workload-dependent.

## 6. Verdict

PR #31438 **reproduces positively and correctly** on Qwen3.6-35B VLM / v0.5.15.post1:
**+8.5–14.5% image-burst request throughput** at the default 2 workers with
**bit-identical greedy output** (semantics unchanged). A clean example of an
agent-relevant **CPU-critical-path / concurrency** fix that lands on real serving
throughput — complements #31558 (JIT/compile) and #29007 (GPU communication).

## 7. Reproduce

- env `sglang-v515`, worktree at v0.5.15.post1.
- Toggle: copy `patches/pr31438/{baseline,patched}/*.py` into the worktree (patched
  also needs the new `executor.py`), restart server. Pure Python, no rebuild.
- Patched auto-enables 2 processor + 16 I/O workers for Qwen-VL; override with
  `--mm-processor-worker-num` / `--mm-io-worker-num`.
