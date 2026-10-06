#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave4"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.60 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
for mode in state pool; do
    P=w4warm_${mode}_s${seed}
    test ! -e "artifacts/runs/$P"
    log="artifacts/logs/$P.log"
    scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source valid --valid-fold 0 --eval-source vfold --full-valid \
        --loss wmse --mask-weight --aux-struct .3 --aux-next .3 --hidden 128 --layers 2 --chunk 2000 --batch 16 \
        --epochs 3 --lr .001 --init "cmp_fullft_s${seed}_ft@ep1" --seed "$seed" --fast --warm-init "$mode" > "$log" 2>&1
    for split in valid train; do
        scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" "$split" --epoch 2 >> "$log" 2>&1
    done
    echo "DONE $(date -Is) $P" >> "$log"
done
