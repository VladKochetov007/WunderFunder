#!/usr/bin/env bash
# Export fold-0/1 clean since parents (@ep3) on train and valid as distillation teachers.
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in 0 1; do for s in 3 4; do for sp in valid train; do
  .venv/bin/python scripts/predict_run.py "c20260929_since_s${s}_f${f}" "$sp" --epoch 3
done; done; done
