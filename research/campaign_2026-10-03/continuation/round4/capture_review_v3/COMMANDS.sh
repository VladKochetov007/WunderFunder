#!/usr/bin/env bash
# Run from repository root. Reproduction requires a fresh reviewer output folder.
set -euo pipefail
own=research/campaign_2026-10-03/continuation/round4/capture_review_v3
# recover_e385.py is intentionally write-once: refuses to replace existing archive.
# run_cpu.sh reviewer .venv/bin/python -B "$own/recover_e385.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/retest.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/torch_engine_boundary.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/verify_inputs.py"
