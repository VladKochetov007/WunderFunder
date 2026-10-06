#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2 kind=$3
D="$PWD/research/iterations/2026-10-05-wave41"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.45 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=24G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
test -f "$D/parent-parity.json"
common=(--train-source valid --valid-fold 0 --eval-source vfold --n-valid-folds 4 --full-valid --loss gcorr --mask-weight --batch 16 --epochs 1 --aux-struct .3 --aux-next .3 --hidden 256 --layers 2 --chunk 2000 --seed "$seed" --fast)
parent="w41plain_${kind}_s${seed}@ep1"
for mode in head ft; do
 P=w41_${kind}_${mode}_s${seed};log=artifacts/logs/$P.log;test ! -e "artifacts/runs/$P"
 if [ "$mode" = head ]; then extra=(--freeze inp,gru --lr .0005);else extra=(--lr .0002);fi
 scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --init "$parent" "${extra[@]}" "${common[@]}" > "$log" 2>&1
 scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" valid --epoch 0 >> "$log" 2>&1
 if [ "$kind" = comp ]; then scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" train --epoch 0 --groups-file research/iterations/2026-10-05-wave2/complement_hold.npy >> "$log" 2>&1;fi
 parent="${P}@ep0"
 echo "DONE $(date -Is) $P" >> "$log"
done
