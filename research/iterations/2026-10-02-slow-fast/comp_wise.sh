#!/usr/bin/env bash
# comp_wise.sh ALPHA SEED GPU : WiSE of composite fullft arm, predict valid+train
set -euo pipefail; cd ~/WunderFunder-creative
a=$1 s=$2 g=$3; tag=$(echo $a | tr -d .); P=cmp_wise${tag}_s$s; log=artifacts/logs/$P.log
export CUDA_VISIBLE_DEVICES=$g WNN_GPU_FRACTION=0.25 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
.venv/bin/python research/iterations/2026-10-02-slow-fast/wise.py cmp_fullft_s${s}_p@ep3 cmp_fullft_s${s}_ft@ep1 ${P}_ft $a >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py ${P}_ft valid --epoch 1 --output-name $P >> $log 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py ${P}_ft train --epoch 1 --output-name $P >> $log 2>&1
echo "DONE $(date -Is) $P" >> $log
