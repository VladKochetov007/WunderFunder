#!/usr/bin/env bash
# CPU tiny fixtures and JSON/code verification only; no author/controller main.
set -euo pipefail
own=research/campaign_2026-10-03/continuation/round6/amp_linearity_review_v2
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/retest.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/verify.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/review_constructor_guards.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/freeze.py"
