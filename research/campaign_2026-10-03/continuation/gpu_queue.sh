#!/usr/bin/env bash
# Controller-owned queue. Wait time is outside per-job deadlines; fail closed.
set -euo pipefail
cd "$(dirname "$0")/../../.."
state=research/campaign_2026-10-03/continuation
queue=${1:-research/campaign_2026-10-01/jobs/queue6.txt}
while true; do
  if [[ -f $state/RESOURCE_FAILURE || -f research/campaign_2026-09-29/STOP_NEW_JOBS ]]; then
    echo 'launch barrier present'; exit 1
  fi
  if [[ -f research/campaign_2026-10-01/QUEUE_PAUSE ]]; then sleep 5; continue; fi
  line=$(awk 'NF && $1 !~ /^#/ {print; exit}' "$queue")
  [[ -n $line ]] || { echo 'queue complete'; exit 0; }
  read -r name timeout threads script args <<< "$line"
  [[ $name =~ ^[a-zA-Z0-9_-]+$ && $timeout =~ ^[0-9]+$ && $threads == 8 ]] || exit 64
  while true; do
    if ! units=$(systemctl --user list-units --type=service --state=active --no-legend); then
      printf '%s systemd service inspection failed\n' "$(date -Is)" >> "$state/RESOURCE_FAILURE"; exit 1
    fi
    [[ $units == *wnn-top1-c2026* ]] || break
    sleep 5
  done
  # Recheck pause at the launch boundary, including after an external job exits.
  [[ ! -f research/campaign_2026-10-01/QUEUE_PAUSE ]] || continue
  [[ ! -f $state/RESOURCE_FAILURE && ! -f research/campaign_2026-09-29/STOP_NEW_JOBS ]] || exit 1
  printf '%s launch %s\n' "$(date -Is)" "$name"
  read -r -a arguments <<< "$args"
  if ! env WNN_CAMPAIGN_VRAM_BYTES=5500000000 .venv/bin/python scripts/campaign_job.py launch \
      --name "$name" --kind gpu --phase codex-continuation --timeout "$timeout" --threads 8 -- \
      "$state/run_gpu.sh" "$script" "${arguments[@]}"; then
    printf '%s launch failed: %s\n' "$(date -Is)" "$name" >> "$state/RESOURCE_FAILURE"
    exit 1
  fi
  sed -i "0,/^$name /s//# started $(date +%H:%M) $name /" "$queue"
  sleep 2
  while true; do
    if ! properties=$(systemctl --user show "wnn-top1-$name.service" -p LoadState -p ActiveState -p Result); then
      printf '%s systemd job inspection failed: %s\n' "$(date -Is)" "$name" >> "$state/RESOURCE_FAILURE"; exit 1
    fi
    [[ $properties == *ActiveState=active* || $properties == *ActiveState=activating* ]] || break
    sleep 5
  done
  if ! .venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["status"],d.get("exit_code")); sys.exit(0 if d["status"]=="finished" and d.get("exit_code")==0 else 1)' "research/campaign_2026-09-29/jobs/$name.json"; then
    printf '%s GPU job failed: %s\n' "$(date -Is)" "$name" >> "$state/RESOURCE_FAILURE"
    exit 1
  fi
done
