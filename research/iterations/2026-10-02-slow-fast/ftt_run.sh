#!/bin/bash
cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
for job in "h256_f0 0" "h256s4_f0 0" "h256_f1 1" "h256s4_f1 1"; do
  while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>4600{print \$1; exit}"); [ -n "$g" ] && break; sleep 60; done
  set -- $job; setsid nohup $D/ft_teacher.sh $1 $2 $g >/dev/null 2>&1 </dev/null & echo "$(date -Is) $job gpu$g"; sleep 150
done
