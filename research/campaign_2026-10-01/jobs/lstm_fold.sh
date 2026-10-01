#!/usr/bin/env bash
# LSTM128 since member on clean fold K (identical to the clean since-parent recipe except --cell lstm, seed 5),
# then GCLP FT with FT seed 3 exporting head-only and two-stage results. Diversity member for GRU seed bags.
#   lstm_fold.sh FOLD
set -euo pipefail
cd "$(dirname "$0")/../../.."
fold=$1
tag=c20261001_lstm_s5_f${fold}
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
.venv/bin/python scripts/train_rnn.py --name "$tag" --train-source train+valid --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --hidden 128 --layers 2 --cell lstm --batch 32 --chunk 2000 --lr 0.001 \
  --wd 0.01 --seed 5 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 --fast \
  --split-manifest "$manifest" --checkpoint-policy fixed --checkpoint-epoch 3 --no-eval-during-fit --epochs 4
.venv/bin/python scripts/predict_run.py "$tag" valid --epoch 3
ft=c20261001_ft_lstm_s5_f${fold}
WNN_IN_CAMPAIGN_JOB=1 bash scripts/ft_gclp.sh "${tag}@ep3" "$ft" "$fold" 3 --split-manifest "$manifest" \
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
.venv/bin/python scripts/predict_run.py "${ft}h" valid --epoch 0
echo "LSTM_DONE $tag $ft"
