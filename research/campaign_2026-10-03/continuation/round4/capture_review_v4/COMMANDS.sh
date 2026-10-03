#!/usr/bin/env bash
# From repository root; retest.py requires a fresh own cpu_reproduction directory.
set -euo pipefail
own=research/campaign_2026-10-03/continuation/round4/capture_review_v4
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/retest.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/verify_inputs.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/freeze.py"
