#!/usr/bin/env bash
set -euo pipefail
own=research/campaign_2026-10-03/continuation/round4/capture_review_v2
# retest requires fresh cpu_reproduction output; other commands are read-only inputs.
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/retest.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/followup.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/rng_prestate.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/moment_cast.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/verify_inputs.py"
