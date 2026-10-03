#!/usr/bin/env bash
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
out=research/campaign_2026-10-03/continuation/round3/washout_usage
for index in {1..23}; do
  printf -v label '%02d' "$index"
  if [[ -f $out/seq_${label}.json ]]; then continue; fi
  research/campaign_2026-10-03/continuation/run_cpu.sh round3-matched-usage \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts /usr/bin/time -v \
    .venv/bin/python -B "$out/probe.py" --sequence "$index" > "$out/seq_${label}.log" 2>&1
  printf 'completed sequence %s\n' "$index"
done
