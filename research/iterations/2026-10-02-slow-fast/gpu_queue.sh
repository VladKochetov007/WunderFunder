#!/usr/bin/env bash
# Memory-aware job queue: each line of QUEUE is "NAME FOLD [SEED=..] args..."; launches fold_job.sh when a GPU has >7.5 GB free.
cd ~/WunderFunder-creative
Q=research/iterations/2026-10-02-slow-fast/queue.txt
while true; do
  line=$(head -n 1 $Q); [ -z "$line" ] && { echo "queue empty $(date -Is)"; exit 0; }
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " '$2>7500{print $1; exit}')
  if [ -n "$g" ]; then
    sed -i 1d $Q
    set -- $line; name=$1 fold=$2; shift 2; seed=3
    if [[ "$1" == SEED=* ]]; then seed=${1#SEED=}; shift; fi
    for x in artifacts/logs/${name}_f${fold}.log; do [ -f $x ] && mv $x $x.old; done
    SEED=$seed WNN_GPU_FRACTION=0.3 setsid nohup scripts/fold_job.sh $name $fold $g "$@" > /dev/null 2>&1 < /dev/null &
    echo "$(date -Is) launched $name f$fold seed$seed on gpu $g"; sleep 90
  else sleep 60; fi
done
