#!/usr/bin/env bash
# N2 screen: GCLP FT of the clean since parents with the pooled-memory correlation loss (decay .995 per chunk)
# in both stages; everything else identical to the reference FT jobs (seed 3, fixed epoch 0).
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in "$@"; do for s in 3 4; do
  WNN_IN_CAMPAIGN_JOB=1 bash scripts/ft_gclp.sh "c20260929_since_s${s}_f${f}@ep3" "c20261001_ftpool_since_s${s}_f${f}" \
    "$f" 3 --split-manifest "research/campaign_2026-09-29/artifacts/splits/fold${f}.json" \
    --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit --loss gcorr_pool --pool-decay 0.995
  .venv/bin/python scripts/predict_run.py "c20261001_ftpool_since_s${s}_f${f}h" valid --epoch 0
  # matched head-stage reference (already fitted by the clean panel; export only)
  [[ -f "cache/pred_c20260929_ft_since_s${s}_f${f}h_valid.npy" ]] || \
    .venv/bin/python scripts/predict_run.py "c20260929_ft_since_s${s}_f${f}h" valid --epoch 0
done; done
