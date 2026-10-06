#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave5"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.65 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
if [ "$seed" = 3 ]; then raw=cmp_rawwmse_s3_ft; else raw=cmp_rawwmse65r2_s4_ft; fi
for mode in independent coupled; do
    P=w5joint_${mode}_s${seed}
    test ! -e "artifacts/runs/$P"
    log="artifacts/logs/$P.log"
    scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source valid --valid-fold 0 --eval-source vfold --full-valid \
        --loss wmse --mask-weight --aux-struct .3 --aux-next .3 --hidden 128 --layers 2 --chunk 2000 --batch 16 \
        --epochs 2 --lr .0001 --init "cmp_fullft_s${seed}_ft@ep1" --init-other "${raw}@ep1" --seed "$seed" --fast --joint-mode "$mode" > "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" valid --epoch 1 >> "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" train --epoch 1 --groups-file "$PWD/research/iterations/2026-10-05-wave2/complement_hold.npy" >> "$log" 2>&1
    echo "DONE $(date -Is) $P" >> "$log"
done
