#!/usr/bin/env bash
# Export reference head-stage FT checkpoints (already fitted in the clean panel) for folds 2/3.
set -euo pipefail
cd "$(dirname "$0")/../../.."
for f in 2 3; do for s in 3 4; do
  .venv/bin/python scripts/predict_run.py "c20260929_ft_since_s${s}_f${f}h" valid --epoch 0
done; done
