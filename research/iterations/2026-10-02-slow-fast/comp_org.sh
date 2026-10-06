#!/usr/bin/env bash
# Composite arm without FT: parent on train half 0 + valid folds != 0 with custom flags only (no propensity/since defaults), predict valid + train.
#   comp_org.sh ARM SEED GPU "flags"
set -euo pipefail; cd ~/WunderFunder-creative
arm=$1 seed=$2 gpu=$3 pf=$4; P=cmp_${arm}_s$seed; log=artifacts/logs/$P.log
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=0.23 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
echo "START $(date -Is) $P pf=[$pf]" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name ${P}_p --train-source train+valid --valid-fold 0 --eval-source vfold --full-valid \
  --hidden 128 --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --seed $seed --fast --train-half 0 $pf >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py ${P}_p valid --epoch 3 --output-name $P >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py ${P}_p train --epoch 3 --output-name $P >> $log 2>&1
echo "DONE $(date -Is) $P" >> $log
