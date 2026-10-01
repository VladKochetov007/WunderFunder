#!/usr/bin/env bash
# Distilled GRU256 "since" student on clean fold K (teachers: that fold's since parents s3+s4 @ep3),
# then GCLP FT (seed 3) exporting both the frozen-trunk head stage and the two-stage result.
#   d256_fold.sh FOLD SEED [MAX_BATCHES] [ALPHA] [HIDDEN]   (defaults 0, 0.5, 256 = the original arm names)
set -euo pipefail
cd "$(dirname "$0")/../../.."
fold=$1 seed=$2 maxb=${3:-0} alpha=${4:-0.5} hidden=${5:-256}
tag=c20261001_d${hidden}_s${seed}_f${fold}
[[ $alpha != 0.5 ]] && tag=c20261001_d${hidden}a${alpha/./}_s${seed}_f${fold}
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
extra=()
if (( maxb > 0 )); then
  tag=${tag}_smoke
  extra=(--max-batches "$maxb" --epochs 1 --checkpoint-epoch 0)
else
  extra=(--epochs 4 --checkpoint-epoch 3)
fi
.venv/bin/python scripts/train_rnn.py --name "$tag" --train-source train+valid --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --hidden "$hidden" --layers 2 --batch 32 --chunk 2000 --lr 0.001 \
  --wd 0.01 --seed "$seed" --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --distill "c20260929_since_s3_f${fold},c20260929_since_s4_f${fold}" --distill-alpha "$alpha" --fast \
  --split-manifest "$manifest" --checkpoint-policy fixed --no-eval-during-fit "${extra[@]}"
(( maxb > 0 )) && exit 0
.venv/bin/python scripts/predict_run.py "$tag" valid --epoch 3
ft=c20261001_ft_${tag#c20261001_}
WNN_IN_CAMPAIGN_JOB=1 bash scripts/ft_gclp.sh "${tag}@ep3" "$ft" "$fold" 3 --split-manifest "$manifest" \
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
.venv/bin/python scripts/predict_run.py "${ft}h" valid --epoch 0
echo "D256_DONE $tag $ft"
