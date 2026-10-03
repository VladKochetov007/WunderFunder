#!/usr/bin/env bash
set -euo pipefail
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B research/campaign_2026-10-03/continuation/round6/amp_linearity_result_review/review.py
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B research/campaign_2026-10-03/continuation/round6/amp_linearity_result_review/freeze.py
# No tensor inputs, analysis rerun, GPU execution or receipt creation.
