#!/usr/bin/env bash
# Scoped adaptation of the existing research file-cache guard; never reclaims swap.
set -u
group=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice
worker=wnn-overnight-nll-fit-20261020.service
while [[ $(systemctl --user show "$worker" -p MainPID --value) != 0 ]]; do
  current=$(cat "$group/memory.current")
  if (( current > 4400000000 )); then
    available=$(awk '$1=="inactive_file" {print $2}' "$group/memory.stat")
    requested=$((current - 4000000000))
    (( requested > available )) && requested=$available
    if (( requested > 0 )); then
      printf '%s swappiness=0\n' "$requested" > "$group/memory.reclaim" 2>/dev/null || true
      printf '%s before=%s requested=%s after=%s\n' \
        "$(date -u +%FT%TZ)" "$current" "$requested" "$(cat "$group/memory.current")"
    fi
  fi
  sleep 5
done
