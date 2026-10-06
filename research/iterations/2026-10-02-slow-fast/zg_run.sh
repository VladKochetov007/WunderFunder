#!/bin/bash
cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
for job in "za 3 a" "zvol 3 vol" "zpxev 3 px,ev" "za 4 a" "zvol 4 vol" "zpxev 4 px,ev"; do
  while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>3900{print \$1; exit}"); [ -n "$g" ] && break; sleep 60; done
  set -- $job; setsid nohup $D/final_zg.sh $1 $2 $g $3 >/dev/null 2>&1 </dev/null & echo "$(date -Is) $job gpu$g"; sleep 150
done
