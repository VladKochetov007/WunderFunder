#!/usr/bin/env bash
# Cheap optimizer/recipe screen: the alpha-0.5 GRU256 since student recipe on clean fold K plus extra train_rnn flags,
# stopped after epoch 2 of the 4-epoch schedule (OPT_REPORT F2: arm contrasts at ep2 track ep3). Parent predictions only;
# the matched control is the existing c20261001_d256_s3_f{K} at epoch 2 (exported once).
#   dflag_fold.sh FOLD TAG [TRAIN_RNN_FLAGS...]
set -euo pipefail
cd "$(dirname "$0")/../../.."
export WNN_VRAM_GB=4.6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WNN_LOADER_WORKERS=3 WNN_LOADER_RAM_GB=3
fold=$1 tag=$2; shift 2
name=c20261002_${tag}_s3_f${fold}
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
.venv/bin/python scripts/train_rnn.py --name "$name" --train-source train+valid --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --hidden 256 --layers 2 --batch 32 --chunk 2000 --lr 0.001 \
  --wd 0.01 --seed 3 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --distill "c20260929_since_s3_f${fold},c20260929_since_s4_f${fold}" --distill-alpha 0.5 --fast \
  --split-manifest "$manifest" --checkpoint-policy fixed --no-eval-during-fit --epochs 4 --checkpoint-epoch 2 \
  --stop-after-epoch 2 "$@"
.venv/bin/python scripts/predict_run.py "$name" valid --epoch 2
ctrl=c20261001_d256_s3_f${fold}
[[ -f cache/pred_${ctrl}_ep2_valid.npy ]] || .venv/bin/python scripts/predict_run.py "$ctrl" valid --epoch 2 --output-name "${ctrl}_ep2"
echo "DFLAG_DONE $name"
