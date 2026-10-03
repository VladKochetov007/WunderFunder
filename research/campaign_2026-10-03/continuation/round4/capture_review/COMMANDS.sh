#!/usr/bin/env bash
set -euo pipefail
# From repo root. adversaries.py requires a fresh cpu_capture directory.
own=research/campaign_2026-10-03/continuation/round4/capture_review
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B -u "$own/adversaries.py"
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B "$own/verify_protected.py"
