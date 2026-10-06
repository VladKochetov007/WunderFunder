#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2 kind=$3
D="$PWD/research/iterations/2026-10-05-wave43"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.45 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
pseed=$((seed-10));extra=()
if [ "$kind" = full ]; then
 parent=d256bc1k_f0; if [ "$pseed" = 4 ]; then parent=d256bc1ks4_f0;fi
 teacher=w6fullteacher;family=d256bc1k_f0,w6family_full,w6family_raw,orgbase
else
 test "$kind" = comp
 parent=w15d256_s${pseed};teacher=w18comp_teacher;family=w15d256_s3,w6family_full,w6family_raw,orgbase;extra=(--train-half 0)
fi
P=w43_${kind}_s${seed};log=artifacts/logs/$P.log;test ! -e "artifacts/runs/$P"
scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train+valid --valid-fold 0 --full-valid --eval-source vfold "${extra[@]}" \
 --loss wmse --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 --aux-struct .3 --aux-next .3 --hidden 256 --layers 2 --chunk 2000 --batch 16 \
 --distill "$teacher" --distill-alpha .7 --family-teachers "$family" --epochs 2 --lr .0001 --init "${parent}@ep3" --seed "$seed" --fast > "$log" 2>&1
scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" valid --epoch 1 >> "$log" 2>&1
if [ "$kind" = comp ]; then scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" train --epoch 1 --groups-file research/iterations/2026-10-05-wave2/complement_hold.npy >> "$log" 2>&1;fi
echo "DONE $(date -Is) $P" >> "$log"
