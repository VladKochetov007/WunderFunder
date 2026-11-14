#!/usr/bin/env bash
# Freeze only this wave's live fit units after three hot readings.
set -u
hot=0
while systemctl --user list-units 'wnn-q14-fit-*' --state=running --no-legend | rg -q .; do
  temp=$(sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}')
  if [ "${temp:-0}" -ge 92 ]; then hot=$((hot+1)); else hot=0; fi
  if [ "$hot" -ge 3 ]; then
    units=$(systemctl --user list-units 'wnn-q14-fit-*' --state=running --no-legend --plain | awk '{print $1}')
    for unit in $units; do systemctl --user freeze "$unit"; done
    echo "freeze CPU=$temp"
    while [ "${temp:-100}" -gt 84 ]; do
      sleep 15
      temp=$(sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}')
    done
    for unit in $units; do systemctl --user thaw "$unit"; done
    echo "thaw CPU=$temp"
    hot=0
  fi
  sleep 15
done
