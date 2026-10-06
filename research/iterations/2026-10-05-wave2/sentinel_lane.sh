#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave2"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.60 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
for mode in zero age; do
    P=w2sent_${mode}_s${seed}
    log="artifacts/logs/$P.log"
    test ! -e "artifacts/runs/${P}_p" || { echo "Run exists $P"; exit 1; }
    scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "${P}_p" --train-source train+valid --valid-fold 0 --eval-source vfold --full-valid \
        --loss wmse --aux-struct .3 --aux-next .3 --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 --hidden 128 \
        --epochs 4 --ema-teacher 1 --ema-decay .999 --eval-ema --since-weight 32 --since-floor .25 --seed "$seed" --fast --train-half 0 \
        --optimizer muon --muon-lr-mult 6 --raw-only --sentinel-mode "$mode" > "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "${P}_ft" --train-source valid --valid-fold 0 --eval-source vfold --full-valid \
        --loss gcorr --mask-weight --batch 16 --aux-struct .3 --aux-next .3 --hidden 128 --layers 2 --chunk 2000 \
        --init "${P}_p@ep3" --seed "$seed" --fast --epochs 2 --lr 1e-4 --raw-only --sentinel-mode "$mode" >> "$log" 2>&1
    for split in valid train; do
        scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "${P}_ft" "$split" --epoch 1 --output-name "$P" >> "$log" 2>&1
    done
    echo "DONE $(date -Is) $P" >> "$log"
done
