#!/usr/bin/env bash
# Keep wnn-campaign.slice measured memory under the 18 GB guard by reclaiming FILE cache only
# (swappiness=0; no limit changes, no anon reclaim). Logs each reclaim.
G=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice
LOG=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/slice_filecache_guard_v1.log
while true; do
  cur=$(cat $G/memory.current); file=$(awk '$1=="file"{print $2}' $G/memory.stat)
  if (( cur > 15500000000 )); then
    want=$(( cur - 14000000000 )); room=$(( file - 1500000000 )); (( want > room )) && want=$room
    if (( want > 0 )); then
      echo "$want swappiness=0" > $G/memory.reclaim 2>/dev/null
      echo "$(date -u +%FT%TZ) cur=$cur file=$file req=$want after=$(cat $G/memory.current)" >> $LOG
    fi
  fi
  sleep 10
done
