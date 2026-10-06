#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2 prior=$3
for mode in uniform soft; do
 while systemctl --user is-active --quiet "wunder-wave33-${mode}-${prior}"; do sleep 10;done
done
D="$PWD/research/iterations/2026-10-05-wave37"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.35 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=24G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave6/scripts:$HOME/WunderFunder/src"
for split in train valid; do
 if [ ! -f "cache/pred_w21final_distilled_s${seed}_${split}.npy" ]; then
  scripts/limited .venv/bin/python research/iterations/2026-10-05-wave6/scripts/predict_run.py "w21final_distilled_s${seed}" "$split" --epoch 1 >> "$D/parent-${seed}.log" 2>&1
 fi
done
