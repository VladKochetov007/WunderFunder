#!/usr/bin/env bash
# From repository root, using a fresh own review output directory for reproduction.
set -euo pipefail
own=research/campaign_2026-10-03/continuation/round6/capture_measurement_review
# The original hash pass succeeded before the recorded metadata schema error.
# For this recorded review, the actual second command resumed without rehashing captures:
# run_cpu.sh reviewer .venv/bin/python -B "$own/audit.py" --resume-verified-byte-pass
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/audit.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/inspect_capture_metadata.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/check_current_guard.py"
research/campaign_2026-10-03/continuation/run_cpu.sh reviewer .venv/bin/python -B "$own/finalize.py"
