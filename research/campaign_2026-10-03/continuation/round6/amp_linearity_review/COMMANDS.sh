#!/usr/bin/env bash
# Pure CPU tiny fixtures/code/provenance only; never run author main/tests/freeze.
set -euo pipefail
own=research/campaign_2026-10-03/continuation/round6/amp_linearity_review
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/review_tests.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/verify.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/freeze.py"
