#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2 mode=$3
D="$PWD/research/iterations/2026-10-05-wave6"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.60 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
P=w22_${mode}_s${seed};log=artifacts/logs/$P.log
hidden=256;init="w19distilled_s${seed}@ep1"
if [ "$mode" = wide ]; then hidden=384;init="w22wide_init_s${seed}@ep0";fi
test ! -e "artifacts/runs/$P"
scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train+valid --valid-fold 0 --eval-source vfold --full-valid \
 --loss wmse --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 --aux-struct .3 --aux-next .3 --hidden "$hidden" --layers 2 --chunk 2000 --batch 16 \
 --distill w22teacher --distill-alpha .7 --family-teachers d256bc1k_f0,w6family_full,w19family_twostream,orgbase \
 --epochs 2 --lr .0001 --init "$init" --seed "$seed" --fast > "$log" 2>&1
scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" valid --epoch 1 >> "$log" 2>&1
echo "DONE $(date -Is) $P" >> "$log"
