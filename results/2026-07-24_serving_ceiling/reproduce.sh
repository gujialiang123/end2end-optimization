#!/usr/bin/env bash
# Reproduce the 2026-07-24 serving-ceiling campaign end to end.
#
# Hardware assumed: NVIDIA H200 (one GPU per worker). Any number of free GPUs
# can be used; the sqlite work-queue load-balances tasks across workers and is
# resume-safe, so interrupting and re-launching workers never repeats a
# completed configuration.
set -euo pipefail

REPO="/home/t-jialianggu/work/EndtoEnd-auto-optimization"
ENVDIR="/home/t-jialianggu/.conda/envs/sglang-dev"
PY="$ENVDIR/bin/python"
OUT="$REPO/results/2026-07-24_serving_ceiling"
VAL="$REPO/results/2026-07-24_serving_ceiling_validation"
GPUS="${GPUS:-0 1 2 4 5 6}"

cd "$REPO"

# ---------------------------------------------------------------- 0. sanity
$PY scripts/serving_ceiling_lib.py    # prints 192 configs + the six workloads

# ------------------------------------------------- 1. coverage pass (1 rep)
$PY scripts/run_serving_ceiling_campaign.py --init --models qwen,lfm25 --outroot "$OUT"
for G in $GPUS; do
  setsid nohup $PY scripts/run_serving_ceiling_campaign.py \
      --gpu "$G" --port $((33100+G)) --worker "g$G" --outroot "$OUT" --reps 1 \
      > "$REPO/logs/serving_ceiling/worker_g$G.log" 2>&1 &
done
wait   # ~16.5 GPU-hours total; ~2.7 h wall-clock on six H200s

# ------------------------------------------------------------- 2. analysis
$PY scripts/analyze_serving_ceiling.py --outroot "$OUT"

# -------------------------------------------- 3. validation pass (5 reps)
$PY scripts/run_serving_ceiling_validation.py --init --outroot "$OUT" --valroot "$VAL" --reps 5
for G in $GPUS; do
  setsid nohup $PY scripts/run_serving_ceiling_validation.py \
      --outroot "$OUT" --valroot "$VAL" --gpu "$G" --port $((33200+G)) \
      --worker "v$G" --reps 5 \
      > "$REPO/logs/serving_ceiling/val_g$G.log" 2>&1 &
done
wait

# re-run the analysis over the repeated measurements so that final claims carry
# 95 % confidence intervals instead of a single-run ranking
$PY scripts/analyze_serving_ceiling.py --outroot "$VAL"

# --------------------------------------------------- 4. figures and slides
$PY scripts/render_serving_ceiling_figures.py --outroot "$OUT"
$PY scripts/update_performance_gap_slides.py --outroot "$OUT" \
    --out "$OUT/performance_gap_slides_1to6_draft.pptx"

echo "done: $OUT"
