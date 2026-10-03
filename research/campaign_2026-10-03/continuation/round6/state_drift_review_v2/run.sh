#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
branch=research/campaign_2026-10-03/continuation/round6/state_drift_review_v2
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B "$branch/final_audit.py" > "$branch/final_audit.log" 2>&1
