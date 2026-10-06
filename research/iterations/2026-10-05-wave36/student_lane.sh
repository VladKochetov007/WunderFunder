#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2 kind=$3
D="$PWD/research/iterations/2026-10-05-wave36"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.35 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
test -f "$D/model-check.json"
extra=()
if [ "$kind" = full ]; then
 parent=w19distilled;teacher=w22teacher;family=d256bc1k_f0
elif [ "$kind" = comp ]; then
 parent=w24comp_distilled;teacher=w33teacher;family=w15d256_s3;extra=(--train-half 0)
else exit 2;fi
P=w36_${kind}_s${seed};log=artifacts/logs/$P.log
test ! -e "artifacts/runs/$P"
scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train+valid "${extra[@]}" --valid-fold 0 --eval-source vfold --full-valid \
 --loss wmse --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 --aux-struct .3 --aux-next .3 --hidden 256 --layers 2 --chunk 2000 --batch 16 \
 --distill "$teacher" --distill-alpha .7 --family-teachers "${family},w6family_full,w19family_twostream,orgbase" --family-class-mode off \
 --epochs 2 --lr .0001 --init "${parent}_s${seed}@ep1" --seed "$seed" --fast > "$log" 2>&1
scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" valid --epoch 1 >> "$log" 2>&1
if [ "$kind" = comp ]; then scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" train --epoch 1 --groups-file research/iterations/2026-10-05-wave2/complement_hold.npy >> "$log" 2>&1;fi
echo "DONE $(date -Is) $P" >> "$log"
