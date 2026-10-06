#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave8"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.65 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
for mode in event8 all; do
    P=w8adaptive_${mode}_s${seed}
    test ! -e "artifacts/runs/${P}_p"
    log="artifacts/logs/$P.log"
    scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "${P}_p" --train-source train+valid --train-half 0 --valid-fold 0 --eval-source vfold --full-valid \
        --loss wmse --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 --aux-struct .3 --aux-next .3 --hidden 64 --event-hidden 256 --arch dual --layers 2 --chunk 2000 --batch 16 \
        --epochs 4 --ema-teacher 1 --ema-decay .999 --eval-ema --since-weight 32 --since-floor .25 --optimizer muon --muon-lr-mult 6 \
        --seed "$seed" --fast --adaptive-schedule "$mode" > "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "${P}_ft" --train-source valid --valid-fold 0 --eval-source vfold --full-valid \
        --loss gcorr --mask-weight --aux-struct .3 --aux-next .3 --hidden 64 --event-hidden 256 --arch dual --layers 2 --chunk 2000 --batch 16 \
        --epochs 2 --lr .0001 --init "${P}_p@ep3" --seed "$seed" --fast --adaptive-schedule "$mode" >> "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "${P}_ft" valid --epoch 1 --output-name "$P" >> "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "${P}_ft" train --epoch 1 --output-name "$P" --groups-file "$PWD/research/iterations/2026-10-05-wave2/complement_hold.npy" >> "$log" 2>&1
    echo "DONE $(date -Is) $P" >> "$log"
done
