#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
lane=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave6"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.60 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
case "$lane" in
 strong) names=(fbc1k_s5); epoch=3;;
 full) names=(ffull_s3 ffull_s4); epoch=1;;
 raw) names=(frawW_s3ft frawW_s4ft); epoch=1;;
 *) exit 2;;
esac
for name in "${names[@]}"; do
 for split in valid train; do
  test ! -e "cache/pred_w18final_${name}_${split}.npy"
  scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$name" "$split" --epoch "$epoch" --output-name "w18final_${name}" >> "research/iterations/2026-10-05-wave18/teacher-${lane}.log" 2>&1
 done
done
echo "DONE $(date -Is) $lane" >> "research/iterations/2026-10-05-wave18/teacher-${lane}.log"
