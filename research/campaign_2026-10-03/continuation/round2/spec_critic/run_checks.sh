#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
critic=research/campaign_2026-10-03/continuation/round2/spec_critic
exec research/campaign_2026-10-03/continuation/run_cpu.sh round2-critic \
  env PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v .venv/bin/python \
  "$critic/checks.py" --output "$critic/results.json"
