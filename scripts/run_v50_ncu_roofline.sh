#!/bin/bash
# v50 NCU roofline capture of the sglang Triton fused_moe grouped-GEMM kernel at
# decode (small M) vs prefill (large M) token counts. Requires sudo (GPU perf counters
# are admin-only on this host: RmProfilingAdminOnly=1). Exports .ncu-rep (open in
# ncu-ui for the roofline chart) + a CSV summary of SOL / roofline metrics.
set -u
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
PY=$ENVDIR/bin/python
NCU=$ENVDIR/bin/ncu
REPO=/home/t-jialianggu/work/end2end-optimization
SCRIPT=$REPO/scripts/run_v50_ncu_moe_microbench.py
OUT=$REPO/results/2026-07-22_v50_ncu_roofline
mkdir -p $OUT

export CUDA_HOME=$ENVDIR CUDA_VISIBLE_DEVICES=0 PATH=$ENVDIR/bin:$PATH
export HF_HOME=/home/t-jialianggu/work/hf_cache
# JIT builds (sglang fused activation) need CUDA headers+libs on the toolchain path;
# under sudo the login env is stripped, so pass them explicitly.
INC=$ENVDIR/targets/x86_64-linux/include
LIB=$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:$ENVDIR/lib64
export CPATH=$INC${CPATH:+:$CPATH}
export LIBRARY_PATH=$LIB${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# fused_moe launches the grouped-GEMM kernel twice per call (w1, w2). warmup run()=2
# launches; skip those, capture the next 2 (w1+w2 of the first timed iter).
for M in "$@"; do
  echo "===== NCU roofline: fused_moe M=$M ====="
  sudo -E env "PATH=$PATH" "CUDA_HOME=$CUDA_HOME" "CUDA_VISIBLE_DEVICES=0" "HF_HOME=$HF_HOME" \
    "CPATH=$CPATH" "LIBRARY_PATH=$LIBRARY_PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    $NCU --set full \
      --kernel-name "regex:fused_moe_kernel" \
      --launch-skip 2 --launch-count 2 \
      --export "$OUT/fused_moe_M${M}" --force-overwrite \
      $PY $SCRIPT --M $M --iters 3 \
      > "$OUT/ncu_M${M}.log" 2>&1
  echo "exit=$? -> $OUT/fused_moe_M${M}.ncu-rep"
done
