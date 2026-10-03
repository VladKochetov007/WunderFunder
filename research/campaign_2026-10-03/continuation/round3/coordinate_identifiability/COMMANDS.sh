#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. No raw data read or model fit.
own=research/campaign_2026-10-03/continuation/round3/coordinate_identifiability
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B "$own/identifiability_toy.py"
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B "$own/finalize.py"
