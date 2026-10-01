#!/usr/bin/env bash
# Distilled head bag member: head-only FT of each clean since parent toward 0.5*y + 0.5*mean(fold teachers:
# d256 parent, since s3 parent, since s4 parent), wMSE on scored valid rows, FT seed 3.
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in 0 1; do
  teachers="c20261001_d256_s3_f${f},c20260929_since_s3_f${f},c20260929_since_s4_f${f}"
  for s in 3 4; do
    FT_STAGES=head WNN_IN_CAMPAIGN_JOB=1 bash scripts/ft_gclp.sh "c20260929_since_s${s}_f${f}@ep3" \
      "c20261001_dh_since_s${s}_f${f}" "$f" 3 --split-manifest "research/campaign_2026-09-29/artifacts/splits/fold${f}.json" \
      --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit --loss wmse --distill "$teachers" --distill-alpha 0.5
  done
done
