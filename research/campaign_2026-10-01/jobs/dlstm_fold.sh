#!/usr/bin/env bash
# H-064 architecture-diversity student: distilled LSTM since student on clean fold K (teachers: the fold's since
# parents s3+s4, as for the GRU256 student), then GCLP FT exporting the head stage. Bag partner for the GRU256 student.
#   dlstm_fold.sh FOLD HIDDEN ALPHA
set -euo pipefail
cd "$(dirname "$0")/../../.."
export WNN_VRAM_GB=${WNN_VRAM_GB:-4.6} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fold=$1 hidden=$2 alpha=$3 seed=3
tag=c20261002_dlstm${hidden}a${alpha/./}_s${seed}_f${fold}
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
.venv/bin/python scripts/train_rnn.py --name "$tag" --train-source train+valid --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --hidden "$hidden" --layers 2 --cell lstm --batch 32 --chunk 2000 --lr 0.001 \
  --wd 0.01 --seed "$seed" --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --distill "c20260929_since_s3_f${fold},c20260929_since_s4_f${fold}" --distill-alpha "$alpha" --fast \
  --split-manifest "$manifest" --checkpoint-policy fixed --no-eval-during-fit --epochs 4 --checkpoint-epoch 3
.venv/bin/python scripts/predict_run.py "$tag" valid --epoch 3
ft=c20261002_ft_${tag#c20261002_}
WNN_IN_CAMPAIGN_JOB=1 FT_STAGES=head bash scripts/ft_gclp.sh "${tag}@ep3" "$ft" "$fold" 3 --split-manifest "$manifest" \
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
[[ -f cache/pred_${ft}h_valid.npy ]] || .venv/bin/python scripts/predict_run.py "${ft}h" valid --epoch 0
echo "DLSTM_DONE $tag $ft"
