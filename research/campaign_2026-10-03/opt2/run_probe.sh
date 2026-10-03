#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec research/campaign_2026-10-03/continuation/run_cpu.sh opt2-builder \
  env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/campaign_2026-10-03/opt2/probe.py
