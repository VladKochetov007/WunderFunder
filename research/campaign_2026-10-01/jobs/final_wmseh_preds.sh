#!/usr/bin/env bash
# Export the incumbent parents' wMSE head-only FT stages (fitted 09-27, never exported) for a holdout head-bag check.
set -euo pipefail
cd "$(dirname "$0")/../../.."
for s in 3 4; do .venv/bin/python scripts/predict_run.py "ab67_final_since_s${s}_wmseh" valid; done
