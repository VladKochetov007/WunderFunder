#!/usr/bin/env bash
# Keep GPU training ahead of CPU research agents inside wnn-campaign.slice: every 30 s give active wnn-top1-* units
# and their wnn-top1-*-scope.scope children CPUWeight=1000 / IOWeight=500 (agents' scopes keep the default 100).
while true; do
  for u in $(systemctl --user list-units --state=active --no-legend --plain 'wnn-top1-*' | awk '{print $1}'); do
    [[ $(systemctl --user show "$u" -p CPUWeight --value) == 1000 ]] || \
      systemctl --user set-property --runtime "$u" CPUWeight=1000 IOWeight=500 || true
  done
  sleep 30
done
