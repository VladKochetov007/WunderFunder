#!/usr/bin/env bash
# Export clean since parents (@ep3) for folds 2/3 on valid (confirmation of the parent+FT output mix).
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in 2 3; do for s in 3 4; do
  .venv/bin/python scripts/predict_run.py "c20260929_since_s${s}_f${f}" valid --epoch 3
done; done
