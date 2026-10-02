#!/usr/bin/env bash
# Full-schedule distilled GRU since student on clean fold K with extra train_rnn flags (C1 learning curve / width,
# C3 since x alpha factorial; RESEARCH_RESET.md). 4 epochs, fixed checkpoint epoch 3, parent predictions only.
#   dcurve_fold.sh FOLD TAG SEED [TRAIN_RNN_FLAGS...]   (later flags override the recipe defaults below)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export WNN_VRAM_GB=4.6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WNN_LOADER_WORKERS=3 WNN_LOADER_RAM_GB=3
fold=$1 tag=$2 seed=$3; shift 3
name=c20261003_${tag}_s${seed}_f${fold}
manifest=research/campaign_2026-09-29/artifacts/splits/fold${fold}.json
.venv/bin/python scripts/train_rnn.py --name "$name" --train-source train+valid --full-valid \
  --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity "c20260929_prop_f${fold}" --prop-power 2 \
  --prop-eps 0.03 --valid-true-mask 0.03 --hidden 256 --layers 2 --batch 32 --chunk 2000 --lr 0.001 \
  --wd 0.01 --seed "$seed" --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --distill "c20260929_since_s3_f${fold},c20260929_since_s4_f${fold}" --distill-alpha 0.5 --fast \
  --split-manifest "$manifest" --checkpoint-policy fixed --no-eval-during-fit --epochs 4 --checkpoint-epoch 3 "$@"
.venv/bin/python scripts/predict_run.py "$name" valid --epoch 3
echo "DCURVE_DONE $name"
