#!/usr/bin/env bash
# comp_ftonly.sh ARM SEED GPU "ft flags": valid FT (folds!=0) on existing composite fullft parent, predict valid+train
set -euo pipefail; cd ~/WunderFunder-creative
arm=$1 s=$2 g=$3 ftf=$4; P=cmp_${arm}_s$s; log=artifacts/logs/$P.log
export CUDA_VISIBLE_DEVICES=$g WNN_GPU_FRACTION=0.23 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
echo "START $(date -Is) $P ft=[$ftf]" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name ${P}_ft --train-source valid --n-valid-folds 4 --valid-fold 0 --eval-source vfold \
  --full-valid --loss gcorr --mask-weight --batch 16 --aux-struct 0.3 --aux-next 0.3 --hidden 128 --layers 2 --chunk 2000 \
  --init cmp_fullft_s${s}_p@ep3 --seed $s --fast $ftf >> $log 2>&1
last=$(ls artifacts/runs/${P}_ft/model_ep*.pt | sort -V | tail -1 | sed "s/.*model_ep\([0-9]*\).pt/\1/")
scripts/limited .venv/bin/python scripts/predict_run.py ${P}_ft valid --epoch $last --output-name $P >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py ${P}_ft train --epoch $last --output-name $P >> $log 2>&1
echo "DONE $(date -Is) $P" >> $log
