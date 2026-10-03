#!/usr/bin/env bash
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
scope=research/campaign_2026-10-03/continuation/round4/live_capture_v2
exec research/campaign_2026-10-03/continuation/run_cpu.sh round4-capture env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="src:$scope" .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp="$scope/test_tmp" "$scope/test_live_capture.py"
