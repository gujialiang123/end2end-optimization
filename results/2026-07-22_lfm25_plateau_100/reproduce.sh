#!/usr/bin/env bash
# Reproduce the v48 LFM2.5 serving-knob plateau study (clean, no warm start).
# Single H200 GPU. All steps use the same environment recorded in environment.json.
set -euo pipefail

REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python
GPU=${GPU:-3}
cd "$REPO"

# 0) (optional) smoke test — 3 sampler-generated trials, verifies invariants
# $PY scripts/run_v48_lfm25_plateau.py --gpu $GPU --port 31700 --n-success 3 --max-attempts 12

# 1) full study: 100 unique COMPLETE trials, fresh Optuna study (no warm start)
$PY scripts/run_v48_lfm25_plateau.py --gpu $GPU --port 31700 --n-success 100 --max-attempts 400

# 2) cookbook baseline reference (5 repeats, measured separately, NOT enqueued)
$PY scripts/run_v48_baseline.py --gpu $GPU --port 31701 --repeats 5

# 3) post-search validation: cookbook + top-5 configs, interleaved x5
$PY scripts/run_v48_validate.py --gpu $GPU --port 31702 --repeats 5 --top 5

# 4) plots (convergence raw + normalized, TTFT/throughput Pareto) + plateau stats
$PY scripts/run_v48_plots.py

echo "Done. See results/2026-07-22_lfm25_plateau_100/"
