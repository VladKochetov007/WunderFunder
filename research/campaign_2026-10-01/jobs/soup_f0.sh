#!/usr/bin/env bash
# H-062 weight soup screen on clean fold 0: soups of the alpha-0.7 and alpha-0.5 GRU256 students' fixed-epoch
# checkpoints, parent predictions, plus the GCLP head-FT stage on the alpha-0.7 ep2+ep3 soup.
set -euo pipefail
cd "$(dirname "$0")/../../.."
manifest=research/campaign_2026-09-29/artifacts/splits/fold0.json
mk() { [[ -d artifacts/runs/$2 ]] || .venv/bin/python scripts/weight_soup.py "$1" "$2" "${@:3}"; }
mk c20261001_d256a07_s3_f0 c20261002_soup23_d256a07_s3_f0 2 3
mk c20261001_d256a07_s3_f0 c20261002_soup123_d256a07_s3_f0 1 2 3
mk c20261001_d256_s3_f0 c20261002_soup23_d256_s3_f0 2 3
for r in c20261002_soup23_d256a07_s3_f0 c20261002_soup123_d256a07_s3_f0 c20261002_soup23_d256_s3_f0; do
  .venv/bin/python scripts/predict_run.py "$r" valid --epoch 3
done
WNN_IN_CAMPAIGN_JOB=1 bash scripts/ft_gclp.sh "c20261002_soup23_d256a07_s3_f0@ep3" c20261002_ft_soup23_d256a07_s3_f0 0 3 \
  --split-manifest "$manifest" --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit
.venv/bin/python scripts/predict_run.py c20261002_ft_soup23_d256a07_s3_f0h valid --epoch 0
echo "SOUP_DONE"
