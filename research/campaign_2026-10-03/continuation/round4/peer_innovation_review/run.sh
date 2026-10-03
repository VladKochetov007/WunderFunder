#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -u research/campaign_2026-10-03/continuation/round4/peer_innovation_review/review.py
