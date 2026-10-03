#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Each command stays in the mandated CPU scope.
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft /usr/bin/time -v -o research/campaign_2026-10-03/continuation/round3/decoder_audit/raw_replay_resource.log .venv/bin/python -B -u research/campaign_2026-10-03/continuation/round3/decoder_audit/raw_bound_replay.py
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft /usr/bin/time -v -o research/campaign_2026-10-03/continuation/round3/decoder_audit/synthetic_resource.log .venv/bin/python -B -u research/campaign_2026-10-03/continuation/round3/decoder_audit/common_reference_falsifier.py
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B -u research/campaign_2026-10-03/continuation/round3/decoder_audit/summarize_audit.py
research/campaign_2026-10-03/continuation/run_cpu.sh round3-hft .venv/bin/python -B -u research/campaign_2026-10-03/continuation/round3/decoder_audit/finalize_audit.py
