#!/usr/bin/env bash
# Composite-validation arm: parent on train half 0 + valid folds != 0, then valid FT on folds != 0, then predict valid + train.
#   comp_arm.sh ARM SEED GPU "parent flags" "ft flags"
set -euo pipefail
cd ~/WunderFunder-creative
arm=$1 seed=$2 gpu=$3 pf=$4 ftf=$5
P=cmp_${arm}_s$seed; log=artifacts/logs/$P.log
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=0.65 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
echo "START $(date -Is) $P pf=[$pf] ft=[$ftf]" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name ${P}_p --train-source train+valid --valid-fold 0 --eval-source vfold --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity prop_f0 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 --hidden 128 \
  --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 --seed $seed --fast --train-half 0 $pf >> $log 2>&1
scripts/limited .venv/bin/python scripts/train_rnn.py --name ${P}_ft --train-source valid --n-valid-folds 4 --valid-fold 0 --eval-source vfold \
  --full-valid --loss gcorr --mask-weight --batch 16 --aux-struct 0.3 --aux-next 0.3 --hidden 128 --layers 2 --chunk 2000 \
  --init ${P}_p@ep3 --seed $seed --fast $ftf >> $log 2>&1
last=$(ls artifacts/runs/${P}_ft/model_ep*.pt | sort -V | tail -1 | sed 's/.*model_ep\([0-9]*\).pt/\1/')
scripts/limited .venv/bin/python scripts/predict_run.py ${P}_ft valid --epoch $last --output-name $P >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py ${P}_ft train --epoch $last --output-name $P >> $log 2>&1
echo "DONE $(date -Is) $P" >> $log
