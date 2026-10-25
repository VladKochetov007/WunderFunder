#!/usr/bin/env bash
# Freeze our c3mix fit units when CPU package >=97C for 3 consecutive 20s samples; thaw at <=88C.
log=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-25/optimizer_panel_r1/thermal_guard_muon6.log
hot=0; frozen=0
while systemctl --user is-active -q wnn-muon6; do
  t=$(sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}')
  if [ "$t" -ge 97 ]; then hot=$((hot+1)); else hot=0; fi
  if [ $frozen -eq 0 ] && [ $hot -ge 3 ]; then
    for u in wnn-muon6; do systemctl --user freeze $u 2>/dev/null; done; frozen=1; echo "$(date -u +%FT%TZ) freeze t=$t" >> $log
  elif [ $frozen -eq 1 ] && [ "$t" -le 88 ]; then
    for u in wnn-muon6; do systemctl --user thaw $u 2>/dev/null; done; frozen=0; echo "$(date -u +%FT%TZ) thaw t=$t" >> $log
  fi
  sleep 20
done
