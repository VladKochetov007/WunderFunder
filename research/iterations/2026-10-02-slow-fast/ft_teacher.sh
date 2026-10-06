#!/usr/bin/env bash
# ft_teacher.sh TEACHER FOLD GPU : full FT of a fold teacher on valid folds != FOLD (true mask), predict train+valid as pred_<T>ft_*
set -euo pipefail; cd ~/WunderFunder-creative
t=$1 k=$2 g=$3; N=${t%_f*}ft_f$k; log=artifacts/logs/$N.log
export CUDA_VISIBLE_DEVICES=$g WNN_GPU_FRACTION=0.26 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
echo "START $(date -Is) $N" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name $N --train-source valid --n-valid-folds 4 --valid-fold $k --eval-source vfold \
  --full-valid --loss gcorr --mask-weight --batch 16 --aux-struct 0.3 --aux-next 0.3 --hidden 256 --layers 2 --chunk 2000 \
  --init ${t}@ep3 --seed 3 --fast --epochs 2 --lr 1e-4 >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py $N valid --epoch 1 >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py $N train --epoch 1 >> $log 2>&1
echo "DONE $(date -Is) $N" >> $log
