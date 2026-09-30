#!/usr/bin/env bash
# Shift-robustness protocol: train on TRAIN ONLY (no validation sequences), predict all validation sequences.
# Pure validation chains are streams the model never saw, the closest local proxy for the unseen test streams.
#   scripts/trainonly_job.sh NAME GPU [extra flags]     (SEED env, default 3)
# Writes cache/pred_NAME_valid.npy and links it as pred_NAME_f{0..3}_valid.npy for fold_eval.py --folds 0,1,2,3.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 gpu=$2; shift 2
log="artifacts/logs/${name}.log"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=${WNN_GPU_FRACTION:-0.25} WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4
export WNN_CPU_PCT=${WNN_CPU_PCT:-10} WNN_MEM_MAX=${WNN_MEM_MAX:-40G}
scripts/limited .venv/bin/python scripts/train_rnn.py --name "$name" --train-source train --eval-source holdout \
  --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity prop_f0 --prop-power 2 --prop-eps 0.03 \
  --hidden 128 --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --seed "${SEED:-3}" --fast "$@" >> "$log" 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py "$name" valid --epoch 3 >> "$log" 2>&1
for k in 0 1 2 3; do ln -sf "pred_${name}_valid.npy" "cache/pred_${name}_f${k}_valid.npy"; done
echo "DONE $(date -Is) $name" >> "$log"
