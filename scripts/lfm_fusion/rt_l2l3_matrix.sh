#!/usr/bin/env bash
# Run the L2/L3/L2+L3 ablation (the 2x2 {MoE config off/on} x {L3 off/on},
# both arm orders) on the new real/agentic RT_ workloads, skipping the
# expensive L1 grid. Each GPU processes a list of workloads sequentially;
# GPUs run in parallel. All cells use the cookbook serving knobs already
# baked into REGIME_SERVING, so no WARMUP override is needed.
set -euo pipefail
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
cd "$REPO"

GPU=$1; PORT=$2; shift 2
REGIMES=("$@")
LOGDIR=$REPO/results/2026-08-10_rt_l2l3
mkdir -p "$LOGDIR"

for REG in "${REGIMES[@]}"; do
  echo "########## GPU$GPU  $REG  (port $PORT) $(date -Is) ##########"
  GPU=$GPU PORT=$PORT REGIME=$REG REPS=8 \
    bash scripts/lfm_fusion/exp3_layered.sh \
    > "$LOGDIR/${REG}.log" 2>&1
  echo "########## GPU$GPU  $REG  DONE $(date -Is) ##########"
done
