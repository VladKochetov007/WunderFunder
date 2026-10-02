#!/usr/bin/env bash
# One-screen campaign status: queue tail, active units, current training progress, GPU, memory, agent status files.
cd "$(dirname "$0")/../../.."
date +%H:%M
grep -hE '^[0-9:]+ (launch|done)' research/campaign_2026-10-01/jobs/gpu_queue6.log 2>/dev/null | tail -3
for u in $(systemctl --user list-units --state=active --no-legend --plain 'wnn-top1-*.service' | awk '{print $1}'); do
  n=${u#wnn-top1-}; n=${n%.service}; echo "RUN $n: $(tail -1 research/campaign_2026-09-29/logs/$n.log 2>/dev/null | cut -c1-70)"
done
echo "GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) | RAM avail $(free -g | awk '/Mem/{print $7}')G | pending: $(grep -cvE '^\s*(#|$)' research/campaign_2026-10-01/jobs/queue6.txt)"
for f in research/campaign_2026-10-02/engine/STATUS_L2.txt research/campaign_2026-10-02/tcn/NOTES.txt; do
  [[ -f $f ]] && echo "$(basename $(dirname $f))/$(basename $f): $(head -1 $f | cut -c1-90)" || true
done
