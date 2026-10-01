#!/usr/bin/env bash
set -euo pipefail
own=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-09-30/integration
.venv/bin/python "$own/diagnostics_proof.py" > "$own/diagnostics_proof_v1.log" 2>&1
.venv/bin/python "$own/cpu_proof.py" --label v2 --with-diagnostics > "$own/cpu_proof_v2.log" 2>&1
