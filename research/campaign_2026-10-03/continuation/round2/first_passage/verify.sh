#!/usr/bin/env bash
set -euo pipefail
firstpassage_dir=research/campaign_2026-10-03/continuation/round2/first_passage
export PYTHONDONTWRITEBYTECODE=1 RUFF_NO_CACHE=true
.venv/bin/ruff check --fix "$firstpassage_dir/path_laws.py" "$firstpassage_dir/decoder.py" "$firstpassage_dir/synthetic.py" "$firstpassage_dir/evaluate.py" "$firstpassage_dir/finalize.py" "$firstpassage_dir/loser_bound.py"
.venv/bin/ruff format "$firstpassage_dir/path_laws.py" "$firstpassage_dir/decoder.py" "$firstpassage_dir/synthetic.py" "$firstpassage_dir/evaluate.py" "$firstpassage_dir/finalize.py" "$firstpassage_dir/loser_bound.py"
.venv/bin/ruff check "$firstpassage_dir/path_laws.py" "$firstpassage_dir/decoder.py" "$firstpassage_dir/synthetic.py" "$firstpassage_dir/evaluate.py" "$firstpassage_dir/finalize.py" "$firstpassage_dir/loser_bound.py"
.venv/bin/python "$firstpassage_dir/synthetic.py"
