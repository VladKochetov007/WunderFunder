#!/usr/bin/env bash
# Cross-fitted PRIVILEGED teacher for distillation: sees raw rows t+L (non-causal, --lookahead), trained on
# one half of train (--train-half) with the fold recipe minus validation data, then predicts train and valid.
#   scripts/teacher_job.sh NAME FOLD HALF GPU LOOKAHEAD [extra flags]
# Merge the two halves with scripts/merge_teachers.py so every train sequence gets an out-of-sample target.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 fold=$2 half=$3 gpu=$4 look=$5; shift 5
run="${name}_h${half}_f${fold}"
log="artifacts/logs/${run}.log"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=${WNN_GPU_FRACTION:-0.5} WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4
export WNN_CPU_PCT=${WNN_CPU_PCT:-10} WNN_MEM_MAX=${WNN_MEM_MAX:-40G}
scripts/limited .venv/bin/python scripts/train_rnn.py --name "$run" --train-source train --train-half "$half" \
  --valid-fold "$fold" --eval-source vfold --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 \
  --propensity "prop_f${fold}" --prop-power 2 --prop-eps 0.03 --hidden 128 --epochs 4 --ema-teacher 1.0 \
  --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 --seed 3 --fast --lookahead "$look" "$@" >> "$log" 2>&1
for s in train valid; do
  scripts/limited .venv/bin/python scripts/predict_run.py "$run" $s --epoch 3 >> "$log" 2>&1
done
echo "DONE $(date -Is) $run" >> "$log"
