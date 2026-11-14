#!/usr/bin/env bash
# Sequential local queue: one GPU fit at a time (a single fit already uses ~94% GPU).
# Reads queue.txt line by line at run time, so lines can be appended while it runs.
# Stops when STOP exists, or pauses while the CPU package is >= 92 C.
O=$(dirname "$(readlink -f "$0")"); cd "$O"
while systemctl --user is-active -q wnn-l14-a.scope; do sleep 20; done
i=0
while :; do
  i=$((i+1)); line=$(sed -n "${i}p" queue.txt); [ -z "$line" ] && break
  [ -e STOP ] && { echo "$(date -u +%FT%TZ) STOP" >> queue.log; break; }
  name=${line%% *}
  [ -e "$name.fit.log" ] && continue
  while [ "$(sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}')" -ge 92 ]; do sleep 30; done
  echo "$(date -u +%FT%TZ) start $line" >> queue.log
  MEMMAX=18000000000 WNN_SCOPE_UNIT=wnn-l14-q$i ./run_cell.sh $line >> queue.log 2>&1 || echo "$(date -u +%FT%TZ) FAILED $name" >> queue.log
  rg -o '"weighted_pearson": [0-9.]+' "$name.fit.log" | tail -1 | sed "s/^/$name /" >> queue.log
done
echo QUEUE_DONE
