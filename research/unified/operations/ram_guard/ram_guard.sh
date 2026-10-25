#!/usr/bin/env bash
# Periodic RAM guard (user-requested 2026-10-25). Every 120 s: log memory; terminate ONLY allowlisted,
# auto-restarting editor language servers above a size threshold. Never touches user apps, VS Code itself,
# other projects' jobs, or wnn-* research units.
LOG=/home/vlad/development/wunder/wnn_connectome_starterpack/research/unified/operations/ram_guard/ram_guard.log
declare -A LIMIT_MB=( [cpptools]=2048 [rust-analyzer]=3072 [ty]=2048 [pyright]=3072 [pylance]=3072 [clangd]=3072 [cpptools-srv2]=1536 )
while true; do
  read -r total used avail <<<"$(free -m | awk '/^Mem:/{print $2, $3, $7}')"
  echo "$(date -u +%FT%TZ) used_mb=$used avail_mb=$avail" >> "$LOG"
  for name in "${!LIMIT_MB[@]}"; do
    for pid in $(pgrep -x "$name" 2>/dev/null); do
      rss=$(( $(ps -o rss= -p "$pid" 2>/dev/null || echo 0) / 1024 ))
      if [ "$rss" -gt "${LIMIT_MB[$name]}" ]; then
        kill -TERM "$pid" 2>/dev/null && echo "$(date -u +%FT%TZ) TERM $name pid=$pid rss_mb=$rss limit_mb=${LIMIT_MB[$name]}" >> "$LOG"
      fi
    done
  done
  if [ "$avail" -lt 6000 ]; then
    echo "$(date -u +%FT%TZ) LOW_MEMORY avail_mb=$avail top: $(ps -eo comm,rss --sort=-rss | sed -n 2,6p | tr '\n' ' ')" >> "$LOG"
  fi
  sleep 120
done
