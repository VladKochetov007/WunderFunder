#!/usr/bin/env bash
set -euo pipefail
scope=research/campaign_2026-10-03/continuation/round4/live_capture_v3
.venv/bin/ruff check "$scope/live_capture.py" "$scope/runner.py" "$scope/baselines.py" "$scope/fit_panel.py" "$scope/test_live_capture.py" "$scope/prepare_runtime.py"
.venv/bin/python "$scope/live_capture.py"
.venv/bin/python "$scope/prepare_runtime.py"
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp="$scope/test_tmp_final" "$scope/test_live_capture.py"
.venv/bin/python "$scope/runner.py" --output "$scope/cpu_smoke_corrected"
