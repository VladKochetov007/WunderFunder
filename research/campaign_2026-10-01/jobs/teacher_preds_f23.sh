#!/usr/bin/env bash
# Export fold-2/3 clean since parents (@ep3) on train (valid already exported) as distillation teachers.
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in 2 3; do for s in 3 4; do
  [[ -f "cache/pred_c20260929_since_s${s}_f${f}_train.npy" ]] || .venv/bin/python scripts/predict_run.py "c20260929_since_s${s}_f${f}" train --epoch 3
done; done
