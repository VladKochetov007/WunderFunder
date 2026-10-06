#!/usr/bin/env bash
# k-fold distilled d256 for folds 2,3: teacher train caches -> d256 fold parent -> head FT (1 ep, like ft256d_f*h)
set -euo pipefail
cd ~/WunderFunder-creative
f=$1 g=$2
log=artifacts/logs/kfd256_f$f.log
echo "START $(date -Is) fold $f gpu $g" >> $log
for t in h256 h256s4; do
  [ -f cache/pred_${t}_f${f}_train.npy ] || CUDA_VISIBLE_DEVICES=$g .venv/bin/python scripts/predict_run.py ${t}_f$f train --epoch 3 >> $log 2>&1
done
WNN_GPU_FRACTION=0.45 scripts/fold_job.sh d256 $f $g --hidden 256 --distill h256_f$f,h256s4_f$f >> $log 2>&1
bash research/iterations/2026-10-02-slow-fast/ft_sweep.sh ft256d $f d256 $g "--freeze inp,gru --epochs 1 --lr 5e-4" >> $log 2>&1 || true
echo "DONE $(date -Is) fold $f" >> $log
