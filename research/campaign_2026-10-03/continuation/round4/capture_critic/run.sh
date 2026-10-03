#!/usr/bin/env bash
set -euo pipefail
# Repository root; synthetic CPU source checks, no training data.
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B -u research/campaign_2026-10-03/continuation/round4/capture_critic/fixtures.py
