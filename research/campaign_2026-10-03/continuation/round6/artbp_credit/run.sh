#!/usr/bin/env bash
set -euo pipefail
scope=research/campaign_2026-10-03/continuation/round6/artbp_credit
exec research/campaign_2026-10-03/continuation/run_cpu.sh round4-capture env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$scope" bash "$scope/run_inside_cpu.sh"
