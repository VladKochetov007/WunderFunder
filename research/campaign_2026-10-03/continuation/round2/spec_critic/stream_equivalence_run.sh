#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
critic=research/campaign_2026-10-03/continuation/round2/spec_critic
exec research/campaign_2026-10-03/continuation/run_cpu.sh round2-critic \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts /usr/bin/time -v \
  .venv/bin/python "$critic/stream_equivalence_check.py"
