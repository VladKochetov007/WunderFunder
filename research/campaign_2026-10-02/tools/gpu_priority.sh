#!/usr/bin/env bash
# Keep GPU training units ahead of CPU research agents inside wnn-campaign.slice: every 30 s give active
# wnn-top1-* units CPUWeight=1000 (agents' scopes keep the default 100) and IOWeight=500.
while true; do
  for u in $(systemctl --user list-units --type=service --state=active --no-legend 'wnn-top1-*' | awk '{print $1}'); do
    [[ $(systemctl --user show "$u" -p CPUWeight --value) == 1000 ]] || \
      systemctl --user set-property --runtime "$u" CPUWeight=1000 IOWeight=500 && true
  done
  sleep 30
done
