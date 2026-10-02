#!/usr/bin/env bash
# Teacher-ensemble upgrade (H-060): distilled GRU256 since student on clean fold K whose teachers are the fold's two
# since parents plus a fold-bound distilled GRU256 student, then GCLP FT head stage + export (same as d256_fold.sh).
#   te_fold.sh FOLD ALPHA STUDENT_TEACHER   e.g. te_fold.sh 0 0.7 c20261001_d256a07_s3_f0
set -euo pipefail
cd "$(dirname "$0")/../../.."
fold=$1 alpha=$2 st=$3 seed=3
tag=c20261002_te${alpha/./}_s${seed}_f${fold}
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
[[ -f cache/pred_${st}_train.npy ]] || .venv/bin/python scripts/predict_run.py "$st" train --epoch 3
[[ -f cache/pred_${st}_valid.npy ]] || .venv/bin/python scripts/predict_run.py "$st" valid --epoch 3
.venv/bin/python scripts/train_rnn.py --name "$tag" --train-source train+valid --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --hidden 256 --layers 2 --batch 32 --chunk 2000 --lr 0.001 \
  --wd 0.01 --seed "$seed" --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --distill "c20260929_since_s3_f${fold},c20260929_since_s4_f${fold},${st}" --distill-alpha "$alpha" --fast \
  --split-manifest "$manifest" --checkpoint-policy fixed --no-eval-during-fit --epochs 4 --checkpoint-epoch 3
.venv/bin/python scripts/predict_run.py "$tag" valid --epoch 3
ft=c20261002_ft_${tag#c20261002_}
WNN_IN_CAMPAIGN_JOB=1 bash scripts/ft_gclp.sh "${tag}@ep3" "$ft" "$fold" 3 --split-manifest "$manifest" \
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
.venv/bin/python scripts/predict_run.py "${ft}h" valid --epoch 0
echo "TE_DONE $tag $ft"
