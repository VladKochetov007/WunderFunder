#!/usr/bin/env bash
set -euo pipefail
puzzle_dir=research/campaign_2026-10-03/eda/puzzle_a
export RUFF_NO_CACHE=true
export MPLCONFIGDIR="$puzzle_dir/.mplconfig"
export XDG_CACHE_HOME="$puzzle_dir/.cache"
.venv/bin/ruff format "$puzzle_dir/continuation_probe.py" "$puzzle_dir/continuation_events.py" "$puzzle_dir/continuation_summary.py" "$puzzle_dir/eda_puzzle_a.py"
.venv/bin/ruff check "$puzzle_dir/continuation_probe.py" "$puzzle_dir/continuation_events.py" "$puzzle_dir/continuation_summary.py" "$puzzle_dir/eda_puzzle_a.py"
.venv/bin/marimo export html "$puzzle_dir/eda_puzzle_a.py" -o "$puzzle_dir/eda_puzzle_a.html"
