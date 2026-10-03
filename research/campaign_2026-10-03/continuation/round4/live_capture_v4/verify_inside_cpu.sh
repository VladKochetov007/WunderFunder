#!/usr/bin/env bash
set -euo pipefail
scope=research/campaign_2026-10-03/continuation/round4/live_capture_v4
.venv/bin/ruff format "$scope/live_capture.py" "$scope/test_rng_roundtrip.py" "$scope/prepare_runtime.py"
.venv/bin/ruff check "$scope/live_capture.py" "$scope/test_rng_roundtrip.py" "$scope/prepare_runtime.py"
.venv/bin/python "$scope/live_capture.py"
.venv/bin/python "$scope/prepare_runtime.py"
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp="$scope/test_tmp" "$scope/test_rng_roundtrip.py"
.venv/bin/python "$scope/runner.py" --output "$scope/cpu_smoke"
