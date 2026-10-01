#!/usr/bin/env bash
# Team-candidate recipe in our code: final since parent on train + ALL validation (holdout included), fixed
# epoch 3, then head-only gcorr FT on all validation; exports parent (train+valid) and head-FT predictions.
#   tc_parent.sh NAME SEED HIDDEN CHUNK [DISTILL_TEACHERS] [ALPHA]
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=$1 seed=$2 hidden=$3 chunk=$4 teachers=${5:-} alpha=${6:-0.5}
extra=()
[[ -n $teachers ]] && extra=(--distill "$teachers" --distill-alpha "$alpha")
.venv/bin/python scripts/train_rnn.py --name "$name" --seed "$seed" --train-source train+valid --include-heldout-valid \
  --eval-source holdout --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity prop_v1 --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --hidden "$hidden" --layers 2 --batch 32 --chunk "$chunk" --epochs 4 \
  --lr 0.001 --wd 0.01 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast --since-weight 32 --since-floor 0.25 \
  --checkpoint-policy fixed --checkpoint-epoch 3 --no-eval-during-fit "${extra[@]}"
.venv/bin/python scripts/predict_run.py "$name" valid --epoch 3
[[ $hidden == 256 && -z $teachers ]] && .venv/bin/python scripts/predict_run.py "$name" train --epoch 3
.venv/bin/python scripts/train_rnn.py --name "${name}_h" --init "${name}@ep3" --train-source valid --include-heldout-valid \
  --eval-source holdout --full-valid --mask-weight --seed 3 --epochs 1 --batch 16 --chunk 2000 --fast \
  --inherit-init-preprocessing --loss gcorr --freeze inp,gru --lr 5e-4 --checkpoint-policy fixed --checkpoint-epoch 0 \
  --no-eval-during-fit
.venv/bin/python scripts/predict_run.py "${name}_h" valid --epoch 0
echo "TC_DONE $name"
