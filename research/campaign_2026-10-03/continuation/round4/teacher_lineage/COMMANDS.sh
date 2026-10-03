#!/usr/bin/env bash
set -euo pipefail
own=research/campaign_2026-10-03/continuation/round4/teacher_lineage
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B -u "$own/audit.py"
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B "$own/freeze.py"
