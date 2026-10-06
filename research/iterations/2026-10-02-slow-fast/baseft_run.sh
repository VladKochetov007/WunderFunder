#!/bin/bash
cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
pick(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>3900{print \$1; exit}"; }
while [ -z "$(pick)" ]; do sleep 60; done; g=$(pick)
(export WNN_PREDICT_FP32=1 CUDA_VISIBLE_DEVICES=$g WNN_GPU_FRACTION=0.23; scripts/limited .venv/bin/python scripts/predict_run.py orgbase valid --output-name orgbase32 && scripts/limited .venv/bin/python scripts/predict_run.py orgbase train --output-name orgbase32) > artifacts/logs/orgbase32.log 2>&1 &
sleep 120
for job in "baseft4 3 --epochs 2 --lr 1e-4" "baseft35 3 --epochs 2 --lr 3e-5" "baseft4 4 --epochs 2 --lr 1e-4"; do
  while [ -z "$(pick)" ]; do sleep 60; done; g=$(pick); set -- $job; a=$1 s=$2; shift 2
  setsid nohup bash $D/comp_baseft.sh $a $s $g "$*" >/dev/null 2>&1 </dev/null & echo "$(date -Is) $a s$s gpu$g"; sleep 150
done
