#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
while systemctl --user is-active --quiet "wunder-wave22-wide-${seed}"; do sleep 10; done
D="$PWD/research/iterations/2026-10-05-wave6"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.60 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
P=w24comp_distilled_s${seed};log=artifacts/logs/$P.log
hidden=256;init="w15d256_s${seed}@ep3"
test ! -e "artifacts/runs/$P"
scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train+valid --train-half 0 --valid-fold 0 --eval-source vfold --full-valid \
 --loss wmse --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 --aux-struct .3 --aux-next .3 --hidden "$hidden" --layers 2 --chunk 2000 --batch 16 \
 --distill w24comp_teacher --distill-alpha .7 --family-teachers w15d256_s3,w6family_full,w19family_twostream,orgbase \
 --epochs 2 --lr .0001 --init "$init" --seed "$seed" --fast > "$log" 2>&1
scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" valid --epoch 1 >> "$log" 2>&1
scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" train --epoch 1 >> "$log" 2>&1
echo "DONE $(date -Is) $P" >> "$log"
