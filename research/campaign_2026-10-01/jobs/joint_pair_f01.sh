#!/usr/bin/env bash
# N1 on the clean since pair, folds 0 and 1 (joint vs solo extra head FT from the reference FT checkpoints).
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in 0 1; do
  research/campaign_2026-10-01/jobs/joint_heads.sh "$f" jpair c20260929_ft_since_s3_f{fold}@ep0 c20260929_ft_since_s4_f{fold}@ep0
done
