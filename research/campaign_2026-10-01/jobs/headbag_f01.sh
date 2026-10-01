#!/usr/bin/env bash
# Head bag on frozen clean since parents (folds 0/1): extra head-only GCLP stages with FT seeds 5 and 6
# (seed 3 head stage already exists from the reference FT; pooled-memory head from the N2 job).
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in 0 1; do for s in 3 4; do for k in 5 6; do
  out=c20261001_hb_since_s${s}_f${f}_k${k}
  FT_STAGES=head WNN_IN_CAMPAIGN_JOB=1 bash scripts/ft_gclp.sh "c20260929_since_s${s}_f${f}@ep3" "$out" "$f" "$k" \
    --split-manifest "research/campaign_2026-09-29/artifacts/splits/fold${f}.json" \
    --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
done; done; done
