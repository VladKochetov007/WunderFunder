#!/usr/bin/env bash
# Train one fold model with the reference recipe (EMA-4 + since-weight, the members of the public 0.6622
# pair) plus extra flags, then write its predictions for all validation sequences.
#   scripts/fold_job.sh NAME FOLD GPU [extra train_rnn.py flags]
# Produces artifacts/runs/NAME_fFOLD and cache/pred_NAME_fFOLD_valid.npy; log in artifacts/logs/NAME_fFOLD.log.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 fold=$2 gpu=$3; shift 3
run="${name}_f${fold}"
log="artifacts/logs/${run}.log"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=${WNN_GPU_FRACTION:-0.9} WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4
export WNN_CPU_PCT=${WNN_CPU_PCT:-10} WNN_MEM_MAX=${WNN_MEM_MAX:-40G}
recipe=(--train-source train+valid --valid-fold "$fold" --eval-source vfold --full-valid --loss wmse
        --aux-struct 0.3 --aux-next 0.3 --propensity "prop_f${fold}" --prop-power 2 --prop-eps 0.03
        --valid-true-mask 0.03 --hidden 128 --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema
        --since-weight 32 --since-floor 0.25 --seed "${SEED:-3}" --fast)
echo "START $(date -Is) gpu=$gpu $run ${*}" >> "$log"
scripts/limited .venv/bin/python scripts/train_rnn.py --name "$run" "${recipe[@]}" "$@" >> "$log" 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py "$run" valid >> "$log" 2>&1
echo "DONE $(date -Is) $run" >> "$log"
