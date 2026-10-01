#!/usr/bin/env bash
# Finish a team-recipe parent whose fit completed: export valid (+train for GRU256 teachers) predictions at epoch 3
# and run the head-only gcorr FT on all validation.   tc_finish.sh NAME HIDDEN [IS_TEACHER(1/0)]
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=$1 hidden=$2 teacher=${3:-1}
.venv/bin/python scripts/predict_run.py "$name" valid --epoch 3
[[ $hidden == 256 && $teacher == 1 ]] && .venv/bin/python scripts/predict_run.py "$name" train --epoch 3
.venv/bin/python scripts/train_rnn.py --name "${name}_h" --init "${name}@ep3" --train-source valid --include-heldout-valid \
  --eval-source holdout --full-valid --mask-weight --seed 3 --epochs 1 --batch 16 --chunk 2000 --fast \
  --inherit-init-preprocessing --loss gcorr --freeze inp,gru --lr 5e-4 --checkpoint-policy fixed --checkpoint-epoch 0 \
  --no-eval-during-fit
.venv/bin/python scripts/predict_run.py "${name}_h" valid --epoch 0
echo "TC_FINISH_DONE $name"
