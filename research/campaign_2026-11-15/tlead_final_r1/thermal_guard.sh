#!/usr/bin/env bash
# Final-day guard: freeze the whole wnn-campaign.slice (only our fits/exports) on sustained >=97C, thaw <=88C.
set -u
read_temp() { sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4); exit}'; }
hot=0; frozen=0
while :; do
  t=$(read_temp)
  if (( frozen )); then
    if [[ -n "$t" ]] && (( t <= 88 )); then systemctl --user thaw wnn-campaign.slice; frozen=0; hot=0; echo "$(date -u +%T) thaw $t"; fi
  else
    if [[ -z "$t" ]] || (( t >= 97 )); then hot=$((hot+1)); else hot=0; fi
    if (( hot >= 3 )); then systemctl --user freeze wnn-campaign.slice && frozen=1; echo "$(date -u +%T) freeze ${t:-na}"; fi
  fi
  sleep 15
done
