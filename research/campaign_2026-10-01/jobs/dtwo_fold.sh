#!/usr/bin/env bash
# H-067 two-timescale student: distilled twoclock since student on clean fold K (row GRU192 x2 + event GRU128 stepping
# on a6 endpoint rows; teachers: the fold's since parents s3+s4, recipe of dtcn_fold.sh), then GCLP FT exporting the
# head stage with the whole trunk frozen (inp,gru covers inp.*, gru.* and gru_event.*; make_model rejects less).
#   dtwo_fold.sh FOLD [MAX_BATCHES]   (MAX_BATCHES: smoke run under its own tag)
set -euo pipefail
cd "$(dirname "$0")/../../.."
# hard-set: the campaign launcher exports WNN_LOADER_WORKERS=4 WNN_LOADER_RAM_GB=6
export WNN_VRAM_GB=4.6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WNN_LOADER_WORKERS=3 WNN_LOADER_RAM_GB=3
fold=$1 max_batches=${2:-0} hidden=192 event_hidden=128 alpha=0.5 seed=3
tag=c20261002_dtwo${hidden}e${event_hidden}a${alpha/./}_s${seed}_f${fold}
smoke=()
if (( max_batches > 0 )); then
  tag=c20261002_dtwosmoke${hidden}e${event_hidden}_s${seed}_f${fold}
  smoke=(--max-batches "$max_batches")
fi
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
.venv/bin/python scripts/train_rnn.py --name "$tag" --train-source train+valid --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --arch twoclock --hidden "$hidden" --layers 2 --event-hidden "$event_hidden" \
  --batch 32 --chunk 2000 --lr 0.001 --wd 0.01 --seed "$seed" --ema-teacher 1.0 --ema-decay 0.999 --eval-ema \
  --since-weight 32 --since-floor 0.25 \
  --distill "c20260929_since_s3_f${fold},c20260929_since_s4_f${fold}" --distill-alpha "$alpha" --fast \
  --split-manifest "$manifest" --checkpoint-policy fixed --no-eval-during-fit --epochs 4 --checkpoint-epoch 3 "${smoke[@]}"
.venv/bin/python scripts/predict_run.py "$tag" valid --epoch 3
ft=c20261002_ft_${tag#c20261002_}
WNN_IN_CAMPAIGN_JOB=1 FT_STAGES=head bash scripts/ft_gclp.sh "${tag}@ep3" "$ft" "$fold" 3 --split-manifest "$manifest" \
  --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit --freeze inp,gru "${smoke[@]}"
[[ -f cache/pred_${ft}h_valid.npy ]] || .venv/bin/python scripts/predict_run.py "${ft}h" valid --epoch 0
echo "DTWO_DONE $tag $ft"
