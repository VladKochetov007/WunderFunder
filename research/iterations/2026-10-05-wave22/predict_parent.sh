#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave6"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.6 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=24G WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
test ! -e "cache/pred_w19distilled_s${seed}_train.npy"
scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "w19distilled_s${seed}" train --epoch 1 > "research/iterations/2026-10-05-wave22/predict-$seed.log" 2>&1
