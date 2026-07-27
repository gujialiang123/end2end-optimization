#!/bin/bash
# HANDOFF §8.1 -- Qwen backend comparison (K1 cross-model validation).
# 3 regimes x 4 backends x 5 reps on one GPU. ~2 GPU-h.
set -x
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$ENVDIR PATH=$ENVDIR/bin:$PATH HF_HOME=$PWD/.hf_cache \
       TRITON_CACHE_DIR=/tmp/regime_kernel_triton_cache
GPU=${1:-4}
PORT=${2:-51000}
for R in C_long_prefill B_concurrent_decode A_low_batch_decode; do
  echo "=========== REGIME $R ==========="
  python scripts/regime_kernel/rk_backends.py --model qwen --regime $R \
    --gpu $GPU --port $PORT --reps 5 \
    --backends auto,triton,triton_kernel,flashinfer_cutlass
done
echo "ALL DONE"
