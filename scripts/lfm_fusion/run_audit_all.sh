#!/bin/bash
# Operator-level fusion-gap audit: LFM2.5 across 3 regimes + Qwen3-30B control.
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python
GPU=${1:-5}
for R in B_concurrent_decode C_long_prefill; do
  echo "########## lfm25 $R ##########"
  $PY scripts/lfm_fusion/lf_audit.py --model lfm25 --regime $R --gpu $GPU
done
for R in A_low_batch_decode C_long_prefill; do
  echo "########## qwen $R (control) ##########"
  $PY scripts/lfm_fusion/lf_audit.py --model qwen --regime $R --gpu $GPU
done
echo "AUDIT ALL DONE"
