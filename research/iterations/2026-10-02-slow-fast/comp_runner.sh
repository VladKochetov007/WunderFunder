#!/usr/bin/env bash
cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast; Q=$D/comp_queue.txt
while true; do
  line=$(head -n 1 $Q); [ -z "$line" ] && { echo "empty $(date -Is)"; exit 0; }
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " '$2>3700{print $1; exit}')
  if [ -n "$g" ]; then
    sed -i 1d $Q; head=${line%%|*}; ft=${line#*|}; set -- $head; arm=$1 seed=$2; shift 2
    setsid nohup bash $D/comp_arm.sh $arm $seed $g "$*" "$ft" > /dev/null 2>&1 < /dev/null &
    echo "$(date -Is) launched $arm s$seed gpu$g"; sleep 120
  else sleep 60; fi
done
