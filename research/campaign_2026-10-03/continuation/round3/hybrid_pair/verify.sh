#!/usr/bin/env bash
set -euo pipefail
hybrid_dir=research/campaign_2026-10-03/continuation/round3/hybrid_pair
export PYTHONDONTWRITEBYTECODE=1 RUFF_NO_CACHE=true
.venv/bin/ruff check --fix "$hybrid_dir/laws.py" "$hybrid_dir/synthetic.py" "$hybrid_dir/probe.py" "$hybrid_dir/summary.py"
.venv/bin/ruff format "$hybrid_dir/laws.py" "$hybrid_dir/synthetic.py" "$hybrid_dir/probe.py" "$hybrid_dir/summary.py"
.venv/bin/ruff check "$hybrid_dir/laws.py" "$hybrid_dir/synthetic.py" "$hybrid_dir/probe.py" "$hybrid_dir/summary.py"
.venv/bin/python "$hybrid_dir/synthetic.py"
