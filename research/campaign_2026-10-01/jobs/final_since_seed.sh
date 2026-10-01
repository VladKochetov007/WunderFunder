#!/usr/bin/env bash
# Extra final since GRU128 trunk (incumbent ab67 parent recipe, new parent seed, fixed epoch 3) plus the
# head-only gcorr FT stage (FT seed 3, as for both incumbent members); exports parent and head predictions.
#   final_since_seed.sh SEED
set -euo pipefail
cd "$(dirname "$0")/../../.."
seed=$1
n=c20261001_final_since_s${seed}
.venv/bin/python scripts/train_rnn.py --name "$n" --seed "$seed" --train-source train+valid --eval-source holdout \
  --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity prop_v1 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --hidden 128 --layers 2 --batch 32 --chunk 2000 --epochs 4 --lr 0.001 --wd 0.01 \
  --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast --precision fp16 --since-weight 32 --since-floor 0.25 \
  --checkpoint-policy fixed --checkpoint-epoch 3
.venv/bin/python scripts/train_rnn.py --name "${n}_gcorrh" --init "${n}@ep3" --train-source valid --eval-source holdout \
  --full-valid --mask-weight --aux-struct 0.3 --aux-next 0.3 --hidden 128 --seed 3 --epochs 1 --batch 16 --fast \
  --loss gcorr --freeze inp,gru --lr 5e-4 --checkpoint-policy fixed --checkpoint-epoch 0
.venv/bin/python scripts/predict_run.py "$n" valid --epoch 3
.venv/bin/python scripts/predict_run.py "${n}_gcorrh" valid --epoch 0
echo "FINAL_SINCE_DONE $n"
