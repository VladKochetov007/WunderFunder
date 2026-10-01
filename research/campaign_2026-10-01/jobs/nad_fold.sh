#!/usr/bin/env bash
# H-059 screen: distilled GRU256 since student with noise-adaptive teacher weight a(u) = A + (FAR - A)(1 - e^{-u/TAU}),
# u = rows until the training label changes; otherwise identical to d256_fold.sh (fold teachers s3+s4, seed 3).
#   nad_fold.sh FOLD A FAR TAU
set -euo pipefail
cd "$(dirname "$0")/../../.."
fold=$1 a=$2 far=$3 tau=$4
tag=c20261002_nad${a/./}_${far/./}_t${tau}_s3_f${fold}
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
.venv/bin/python scripts/train_rnn.py --name "$tag" --train-source train+valid --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --hidden 256 --layers 2 --batch 32 --chunk 2000 --lr 0.001 \
  --wd 0.01 --seed 3 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --distill "c20260929_since_s3_f${fold},c20260929_since_s4_f${fold}" --distill-alpha "$a" \
  --distill-alpha-far "$far" --distill-alpha-tau "$tau" --fast \
  --split-manifest "$manifest" --checkpoint-policy fixed --checkpoint-epoch 3 --no-eval-during-fit --epochs 4
.venv/bin/python scripts/predict_run.py "$tag" valid --epoch 3
ft=${tag/c20261002_/c20261002_ft_}
WNN_IN_CAMPAIGN_JOB=1 FT_STAGES=head bash scripts/ft_gclp.sh "${tag}@ep3" "$ft" "$fold" 3 --split-manifest "$manifest" \
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
echo "NAD_DONE $tag"
