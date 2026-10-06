#!/bin/bash
cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast; Q=$D/org_queue.txt
while true; do
  line=$(head -n 1 $Q); [ -z "$line" ] && { echo "empty $(date -Is)"; exit 0; }
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>3900{print \$1; exit}")
  if [ -n "$g" ]; then
    sed -i 1d $Q; kind=${line%% *}; rest=${line#* }; head=${rest%%|*}; ft=${rest#*|}; set -- $head; arm=$1 seed=$2; shift 2
    if [ $kind = O ]; then setsid nohup bash $D/comp_org.sh $arm $seed $g "$*" >/dev/null 2>&1 </dev/null &
    else setsid nohup bash $D/comp_arm.sh $arm $seed $g "$*" "$ft" >/dev/null 2>&1 </dev/null & fi
    echo "$(date -Is) $kind $arm s$seed gpu$g"; sleep 120
  else sleep 60; fi
done
