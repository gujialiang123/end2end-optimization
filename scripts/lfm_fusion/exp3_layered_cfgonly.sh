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
#
# --correctness-nogate: the greedy signature is recorded but does not veto an
# arm. Three of the five gate prompts drive LFM2.5 into a repetition loop where
# the baseline itself emits "So. So. So." forever; the arms then differ only in
# where a period lands, which is a near-tied logit flipping, not a wrong kernel.
# The two prompts that produce real text are token-identical. Gating on this
# would also make the reversed order unrunnable, since the first arm defines the
# signature. Correctness for these kernels rests on the GSM8K run in the
# 2026-07-27 report, not on this probe.
set -euo pipefail

REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
PROFILE=$REPO/configs/regime_kernel/profiles/lfm25_pr_candidate
GPU=${GPU:-4}
REGIME=${REGIME:-C_long_prefill}
# Result dirs are named from the first underscore-delimited token, so
# D_medium_balanced and D_medium_balanced_tuned would both land in "D_" and
# overwrite each other. A _tuned regime therefore gets SUITE=l1_ automatically
# unless the caller set SUITE explicitly. C_long_prefill keeps an empty short
# for backwards compatibility with the dirs already committed, and
# C_long_prefill_tuned keeps l1_C_, which is what exp5 produced.
REGIME_SHORT=$( [ "$REGIME" = "C_long_prefill" ] && echo "" || echo "${REGIME%%_*}_" )
case "$REGIME" in
  *_tuned) SUITE=${SUITE:-l1_} ;;
esac
SUITE=${SUITE:-}
REPS=${REPS:-8}
# Warm-up is deliberately not defaulted. serving_ceiling_lib.WARMUP_RUNS was
# calibrated against the cookbook knobs, and a different serving config is a
# different steady state: on cap8/chunk2048/fcfs/mem0.9 the first two scored
# repetitions were still climbing from 20 to 23 req/s under its four warm-ups,
# which read as "the kernel gain vanished". Set WARMUP when the regime is not
# one of the three cookbook ones.
WARMUP=${WARMUP:-}
PORT=${PORT:-52140}

# A tuned regime is by definition not the config the warm-up table was
# calibrated on, and running one under the default table is what produced the
# "+0.0 %, the kernel gain vanished" cell on 2026-08-03. Refuse rather than
# silently produce it again.
case "$REGIME" in
  *_tuned)
    if [ -z "$WARMUP" ]; then
      echo "refusing to run $REGIME without WARMUP." >&2
      echo "serving_ceiling_lib.WARMUP_RUNS is calibrated on the cookbook knobs;" >&2
      echo "a tuned serving config is a different steady state. Use WARMUP=12 REPS=30." >&2
      exit 2
    fi ;;
esac

export CUDA_HOME=$ENVDIR
export HF_HOME=$REPO/.hf_cache
cd "$REPO"

run () {           # $1 tag  $2 arms  $3 config-dir ("" = leave unset)
  local tag=$1 arms=$2 cfgdir=$3
  echo "=============== $tag | arms=$arms | cfg=${cfgdir:-<none>} ==============="
  if [ -n "$cfgdir" ]; then export SGLANG_MOE_CONFIG_DIR="$cfgdir";
  else unset SGLANG_MOE_CONFIG_DIR || true; fi
  "$ENVDIR/bin/python" scripts/lfm_fusion/lf_e2e.py \
      --regime "$REGIME" --gpu "$GPU" --port "$PORT" \
      --arms "$arms" --reps "$REPS" ${WARMUP:+--warmup "$WARMUP"} \
      --tag "_exp3_${SUITE}${REGIME_SHORT}$tag" --correctness-nogate
}

#run nocfg_fwd "${ARMS_FWD:-baseline,all7}" ""
#run nocfg_rev "${ARMS_REV:-all7,baseline}" ""
run cfg_fwd   "${ARMS_FWD:-baseline,all7}" "$PROFILE"
run cfg_rev   "${ARMS_REV:-all7,baseline}" "$PROFILE"

echo "all four cells done"
