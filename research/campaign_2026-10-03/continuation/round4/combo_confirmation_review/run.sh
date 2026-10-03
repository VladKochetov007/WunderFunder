#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
research/campaign_2026-10-03/continuation/run_cpu.sh round2-critic \
  env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/campaign_2026-10-03/continuation/round4/combo_confirmation_review/audit.py \
  > research/campaign_2026-10-03/continuation/round4/combo_confirmation_review/checks.log 2>&1
