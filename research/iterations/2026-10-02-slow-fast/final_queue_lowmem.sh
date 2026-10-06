#!/usr/bin/env bash
# Memory-aware queue for final-data members: lines "NAME SEED KIND extra args..." -> final_mu.sh with EXTRA_ARGS
cd ~/WunderFunder-creative
Q=research/iterations/2026-10-02-slow-fast/final_queue.txt; S=research/iterations/2026-10-02-slow-fast/final_mu_lowmem.sh
while true; do
  line=$(head -n 1 $Q); [ -z "$line" ] && { echo "final queue empty $(date -Is)"; exit 0; }
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " '$2>3900{print $1; exit}')
  if [ -n "$g" ]; then
    sed -i 1d $Q; set -- $line; name=$1 seed=$2 kind=$3; shift 3
    GPUFRAC=0.25 EXTRA_ARGS="$*" setsid nohup $S $name $seed $g $kind > /dev/null 2>&1 < /dev/null &
    echo "$(date -Is) launched $name seed$seed $kind gpu$g extra=[$*]"; sleep 90
  else sleep 60; fi
done
