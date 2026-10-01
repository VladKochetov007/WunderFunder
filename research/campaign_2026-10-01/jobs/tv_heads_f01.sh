#!/usr/bin/env bash
# Head-bag member: head-only gcorr FT of each clean since parent on train + fold-safe valid with the parent's
# propensity/true-mask row weights (frozen trunk, 1 epoch, batch 16, lr 5e-4, FT seed 3).
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in 0 1; do for s in 3 4; do
  n=c20261001_tvh_since_s${s}_f${f}
  .venv/bin/python scripts/train_rnn.py --name "$n" --init "c20260929_since_s${s}_f${f}@ep3" --inherit-init-preprocessing \
    --train-source train+valid --full-valid --loss gcorr --freeze inp,gru --lr 5e-4 --epochs 1 --batch 16 --chunk 2000 \
    --seed 3 --fast --propensity "c20260929_prop_f${f}" --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
    --split-manifest "research/campaign_2026-09-29/artifacts/splits/fold${f}.json" --checkpoint-policy fixed \
    --checkpoint-epoch 0 --no-eval-during-fit
  .venv/bin/python scripts/predict_run.py "$n" valid --epoch 0
done; done
