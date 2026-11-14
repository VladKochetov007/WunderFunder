#!/usr/bin/env bash
# Freeze our wnn-l14 fit scopes when the CPU package is >= 95 C for 3 consecutive 20 s samples; thaw at <= 85 C.
O=$(dirname "$(readlink -f "$0")"); hot=0; frozen=0
units() { systemctl --user list-units 'wnn-l14-*.scope' --no-legend --plain | awk '{print $1}'; }
while [ ! -e "$O/GUARD_STOP" ]; do
  t=$(sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}')
  if [ "$t" -ge 95 ]; then hot=$((hot+1)); else hot=0; fi
  if [ $frozen -eq 0 ] && [ $hot -ge 3 ]; then
    for u in $(units); do systemctl --user freeze "$u"; done; frozen=1; echo "$(date -u +%FT%TZ) freeze t=$t" >> "$O/thermal_guard.log"
  elif [ $frozen -eq 1 ] && [ "$t" -le 85 ]; then
    for u in $(units); do systemctl --user thaw "$u"; done; frozen=0; echo "$(date -u +%FT%TZ) thaw t=$t" >> "$O/thermal_guard.log"
  fi
  sleep 20
done
