#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
role=round2-objective
root=research/campaign_2026-10-03/continuation/round2/objective
run_cpu=research/campaign_2026-10-03/continuation/run_cpu.sh
# PREREG is immutable. Reproduction uses it, never chooses a new subset.
for probe in synthetic_probe stream_probe analyze_probe compact_results sham_probe finalize_results; do
  "$run_cpu" "$role" env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "$root/$probe.py"
done
"$run_cpu" "$role" env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  -o "cache_dir=$root/.pytest_cache" --basetemp="$root/repro_test_tmp" "$root/test_objective.py"
