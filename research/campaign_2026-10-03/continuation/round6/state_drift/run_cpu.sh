#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
research/campaign_2026-10-03/continuation/run_cpu.sh round2-critic env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/campaign_2026-10-03/continuation/round6/state_drift/controls.py "$@"
