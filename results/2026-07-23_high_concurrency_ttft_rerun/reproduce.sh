#!/usr/bin/env bash
# Reproduce the high-concurrency TTFT rerun (2 models x 3 configs x 2 regimes, 6 reps).
# Single H200, BF16, TP1, CUDA graph on. Streaming client captures client-observed TTFT.
set -euo pipefail
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python
cd "$REPO"
# smoke test first (optional):
#   CUDA_VISIBLE_DEVICES=<gpu> $PY -m sglang.launch_server --model-path /data/hf/LFM2.5-8B-A1B ...
#   $PY scripts/v51_stream_bench.py --url http://127.0.0.1:32207 --num-prompts 64 --max-new 512 --concurrency 64 --out /tmp/smoke.json
# full matrix:
GPU=7 PORT=32207 $PY scripts/run_v51_high_conc_ttft.py
echo "See results/2026-07-23_high_concurrency_ttft_rerun/{summary.csv,comparison.md,high_concurrency_ttft.png}"
