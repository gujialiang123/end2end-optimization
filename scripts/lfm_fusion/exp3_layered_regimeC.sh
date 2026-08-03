#!/usr/bin/env bash
# Experiment 3: does the kernel rewrite still pay once the baseline already has
# the tuned MoE config (upstream PR #32687)?
#
# The 2026-07-27 kernel A/B reported +5.30 % on regime C, but it ran on sglang
# 17f7a1da1, a tree that ships no E=32,N=1792 config for any device. That
# baseline is therefore missing a layer worth +23.34 % on this very regime, so
# the +5.30 % is an increment over a dirty baseline.
#
# Design is 2x2: {MoE config off, on} x {arm order forward, reversed}.
#   * the config axis gives Bar2->Bar3 and lets us re-measure Bar2->Bar4 in the
#     same session as Bar3->Bar4, instead of comparing against a month-old run;
#   * the order axis is not optional -- lf_e2e.py runs arms sequentially, one
#     server lifetime each, and the #32687 e2e work already caught this harness
#     producing a sign flip purely from arm order ("whichever ran first was
#     faster"). A single order cannot resolve a few-percent effect here.
#
# Both arms of a pair see the identical SGLANG_MOE_CONFIG_DIR, so the config is
# a property of the baseline, not a difference between arms.
set -euo pipefail

REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
PROFILE=$REPO/configs/regime_kernel/profiles/lfm25_pr_candidate
GPU=${GPU:-4}
REPS=${REPS:-8}
PORT=${PORT:-52140}

export CUDA_HOME=$ENVDIR
export HF_HOME=$REPO/.hf_cache
cd "$REPO"

run () {           # $1 tag  $2 arms  $3 config-dir ("" = leave unset)
  local tag=$1 arms=$2 cfgdir=$3
  echo "=============== $tag | arms=$arms | cfg=${cfgdir:-<none>} ==============="
  if [ -n "$cfgdir" ]; then export SGLANG_MOE_CONFIG_DIR="$cfgdir";
  else unset SGLANG_MOE_CONFIG_DIR || true; fi
  "$ENVDIR/bin/python" scripts/lfm_fusion/lf_e2e.py \
      --regime C_long_prefill --gpu "$GPU" --port "$PORT" \
      --arms "$arms" --reps "$REPS" --tag "_exp3_$tag"
}

run nocfg_fwd baseline,all7 ""
run nocfg_rev all7,baseline ""
run cfg_fwd   baseline,all7 "$PROFILE"
run cfg_rev   all7,baseline "$PROFILE"

echo "all four cells done"
