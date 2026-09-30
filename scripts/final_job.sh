#!/usr/bin/env bash
# Final member for a submission: parent on train + ALL validation sequences (EMA-4 + since-weight recipe of
# fold_job.sh), then the head-only gcorr fine-tune on all validation rows with true masks (the stage that
# the chain-fold study found best for hidden 256: research/quant-probes-2026-09-29.md).
#   scripts/final_job.sh NAME GPU SEED [extra flags, e.g. --hidden 256]
# PARENT_EXTRA: flags for the parent only (e.g. "--distill fin256_s3,fin256_s4 --distill-alpha 0.5").
# Produces artifacts/runs/NAME (parent) and artifacts/runs/NAMEh (packaged member). The holdout score in
# the logs is in-sample (the holdout is trained on) and is not evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 gpu=$2 seed=$3; shift 3
log="artifacts/logs/${name}.log"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=${WNN_GPU_FRACTION:-0.9} WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4
export WNN_CPU_PCT=${WNN_CPU_PCT:-10} WNN_MEM_MAX=${WNN_MEM_MAX:-40G}
# PARENT_SOURCE=train: parents never see validation (only the head fine-tune does)
if [ "${PARENT_SOURCE_MODE:-trainvalid}" = train ]; then PARENT_SOURCE="--train-source train"; else PARENT_SOURCE="--train-source train+valid --include-heldout-valid"; fi
echo "START $(date -Is) gpu=$gpu $name ${*}" >> "$log"
scripts/limited .venv/bin/python scripts/train_rnn.py --name "$name" \
  $PARENT_SOURCE --eval-source holdout --full-valid --loss wmse \
  --aux-struct 0.3 --aux-next 0.3 --propensity prop_f0 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
  --hidden 128 --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --seed "$seed" --fast "$@" ${PARENT_EXTRA:-} >> "$log" 2>&1
scripts/limited .venv/bin/python scripts/train_rnn.py --name "${name}h" --init "${name}@ep3" \
  --train-source valid --include-heldout-valid --eval-source holdout --full-valid --loss gcorr --mask-weight \
  --batch 16 --epochs 1 --aux-struct 0.3 --aux-next 0.3 --hidden 128 --seed "$seed" --fast \
  --freeze inp,gru --lr 5e-4 "$@" >> "$log" 2>&1
echo "DONE $(date -Is) $name" >> "$log"
