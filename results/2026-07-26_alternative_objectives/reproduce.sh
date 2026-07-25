#!/usr/bin/env bash
# Reproduce the alternative-objective study. No new search: the warmed 192-point
# grid is re-selected under 8 objectives, then only the missing configs are
# validated with 5 repetitions.
set -euo pipefail
REPO="/home/t-jialianggu/work/EndtoEnd-auto-optimization"
PY="/home/t-jialianggu/.conda/envs/sglang-dev/bin/python"
R="$REPO/results/2026-07-26_alternative_objectives"
GPUS="${GPUS:-0 1 2 3 4 5 6}"
cd "$REPO"

# Phases 1-3: selection policies + validation-coverage audit + plan
$PY scripts/analyze_alternative_serving_objectives.py

# Phase 4: run ONLY the missing configurations (cookbook anchors queued first)
$PY scripts/run_alternative_objective_validation.py --init
for G in $GPUS; do
  setsid nohup $PY scripts/run_alternative_objective_validation.py \
      --gpu "$G" --port $((41000+G*100)) --worker "a$G" --reps 5 \
      > "$REPO/logs/alt_objectives/worker_a$G.log" 2>&1 &
done
wait

# Phase 5: validated selection with bootstrap 95% CIs
$PY scripts/finalize_alternative_objectives.py

# Phase 6: figures
$PY scripts/render_alternative_objective_figures.py
echo "done: $R"
