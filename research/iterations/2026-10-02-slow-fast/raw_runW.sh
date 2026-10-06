#!/bin/bash
cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
for s in 3 4; do
  while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>3900{print \$1; exit}"); [ -n "$g" ] && break; sleep 60; done
  setsid nohup $D/final_rawW.sh $s $g >/dev/null 2>&1 </dev/null & echo "$(date -Is) frawW s$s gpu$g"; sleep 150
done
