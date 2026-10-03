#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
branch=research/campaign_2026-10-03/continuation/round6/state_drift_critic
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer env PYTHONDONTWRITEBYTECODE=1 python -B "$branch/adversaries.py" > "$branch/run.log" 2>&1
