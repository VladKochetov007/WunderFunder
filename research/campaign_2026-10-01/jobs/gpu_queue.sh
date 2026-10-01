#!/usr/bin/env bash
# Sequential GPU queue: launch each campaign job only when no campaign GPU unit is active, then wait for it to
# finish. Avoids queued waiters whose systemd RuntimeMaxSec would expire while blocked on gpu.lock.
#   gpu_queue.sh QUEUE_FILE   (lines: NAME TIMEOUT THREADS SCRIPT [ARGS...]; '#' comments; re-read each step)
set -uo pipefail
cd "$(dirname "$0")/../../.."
queue=$1
active() { systemctl --user list-units --type=service --state=active --no-legend 2>/dev/null | grep -q "wnn-top1-c2026"; }
while true; do
  [[ -f research/campaign_2026-09-29/STOP_NEW_JOBS ]] && { echo "barrier present; stopping queue"; exit 1; }
  [[ -f research/campaign_2026-10-01/QUEUE_PAUSE ]] && { sleep 20; continue; }
  line=$(grep -v '^\s*#' "$queue" | grep -v '^\s*$' | head -1)
  [[ -z $line ]] && { echo "queue empty"; exit 0; }
  while active; do sleep 10; done
  read -r name timeout threads script args <<<"$line"
  sed -i "0,/^$name /s//# started $(date +%H:%M) $name /" "$queue"
  echo "$(date +%H:%M:%S) launch $name"
  WNN_SLICE=wnn-campaign.slice .venv/bin/python scripts/campaign_job.py launch --name "$name" --kind gpu --phase queue \
    --timeout "$timeout" --threads "$threads" -- env WNN_IN_CAMPAIGN_JOB=1 WNN_THREADS="$threads" \
    WNN_CPU_QUOTA="${threads}00%" WNN_MEM_MAX=30G scripts/limited "$script" $args || echo "launch failed $name"
  sleep 15
  while active; do sleep 10; done
  echo "$(date +%H:%M:%S) done $name $(python3 -c "import json;d=json.load(open('research/campaign_2026-09-29/jobs/$name.json'));print(d['status'],d.get('exit_code'),round(d.get('elapsed_seconds') or 0))" 2>/dev/null)"
done
