#!/usr/bin/env bash
# R11 screen: clean since parent recipe + event-preserving row thinning (p 0.5, q ~ U(0.1, 0.4)), then GCLP FT
# (FT seed 3) exporting parent, head-stage and two-stage predictions.   thin_fold.sh FOLD SEED
set -euo pipefail
cd "$(dirname "$0")/../../.."
fold=$1 seed=$2
tag=c20261001_thin_since_s${seed}_f${fold}
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
.venv/bin/python scripts/train_rnn.py --name "$tag" --train-source train+valid --full-valid --loss wmse \
  --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --hidden 128 --layers 2 --batch 32 --chunk 2000 --epochs 4 --lr 0.001 --wd 0.01 \
  --seed "$seed" --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 --fast \
  --thin-prob 0.5 --thin-range 0.1,0.4 --split-manifest "$manifest" --checkpoint-policy fixed --checkpoint-epoch 3 \
  --no-eval-during-fit
.venv/bin/python scripts/predict_run.py "$tag" valid --epoch 3
ft=c20261001_ft_thin_since_s${seed}_f${fold}
WNN_IN_CAMPAIGN_JOB=1 bash scripts/ft_gclp.sh "${tag}@ep3" "$ft" "$fold" 3 --split-manifest "$manifest" \
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
.venv/bin/python scripts/predict_run.py "${ft}h" valid --epoch 0
echo "THIN_DONE $tag"
