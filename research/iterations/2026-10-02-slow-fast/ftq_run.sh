#!/usr/bin/env bash
cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast; Q=$D/ftq.txt
while true; do
  line=$(head -n 1 $Q); [ -z "$line" ] && { echo "queue empty $(date -Is)"; exit 0; }
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>3900{print \$1; exit}")
  if [ -n "$g" ]; then
    sed -i 1d $Q; set -- $line; kind=$1 arm=$2 s=$3; shift 3
    if [ $kind = W ]; then setsid nohup $D/comp_wise.sh $1 $s $g >/dev/null 2>&1 </dev/null &
    else setsid nohup $D/comp_ftonly.sh $arm $s $g "$*" >/dev/null 2>&1 </dev/null & fi
    echo "$(date -Is) $kind $arm s$s gpu$g"; sleep 120
  else sleep 45; fi
done
