#!/usr/bin/env bash
# Discriminating test: is the head-FT gain from the gcorr LOSS or from VALID data? FT head with gcorr on train+valid folds.
#   ft_train.sh NAME FOLD GPU EXTRA_FLAGS
set -euo pipefail
cd ~/WunderFunder-creative
name=$1 fold=$2 gpu=$3 extra=$4
out="${name}_f${fold}"; log="artifacts/logs/${out}.log"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=0.45 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
echo "START $(date -Is) $out" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name ${out}a --init d256_f${fold}@ep3 --train-source train+valid --valid-fold $fold \
  --eval-source vfold --full-valid --loss gcorr --propensity prop_f$fold --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
  --batch 16 --aux-struct 0.3 --aux-next 0.3 --hidden 256 --layers 2 --seed 3 --fast --freeze inp,gru $extra >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py ${out}a valid --output-name ${out} >> $log 2>&1
echo "DONE $(date -Is) $out" >> $log
