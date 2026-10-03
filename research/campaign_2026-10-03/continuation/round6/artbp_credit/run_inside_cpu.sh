#!/usr/bin/env bash
set -euo pipefail
scope=research/campaign_2026-10-03/continuation/round6/artbp_credit
.venv/bin/ruff format "$scope/credit_oracle.py" "$scope/run_probe.py"
.venv/bin/ruff check "$scope/credit_oracle.py" "$scope/run_probe.py"
.venv/bin/python "$scope/run_probe.py"
