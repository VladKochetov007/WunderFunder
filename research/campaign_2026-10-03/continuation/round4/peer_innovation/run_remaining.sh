#!/usr/bin/env bash
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
out=research/campaign_2026-10-03/continuation/round4/peer_innovation
for index in {1..23}; do
  printf -v label '%02d' "$index"
  [[ ! -f $out/seq_${label}.json ]] || continue
  research/campaign_2026-10-03/continuation/run_cpu.sh round3-matched-usage \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts /usr/bin/time -v \
    .venv/bin/python -B "$out/extract.py" --sequence "$index" > "$out/seq_${label}.log" 2>&1
  printf 'completed%s\n' "$index"
done
