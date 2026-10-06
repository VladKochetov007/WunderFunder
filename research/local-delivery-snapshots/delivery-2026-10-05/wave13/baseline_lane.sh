#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave13"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.25 WNN_LOADER_RAM_GB=3 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=24G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src" WNN_PREDICT_FP32=1
if [ "$seed" = 3 ]; then
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" w13base_direct valid --epoch 0 --output-name w13frozen > "$D/frozen-predict.log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" w13base_direct train --epoch 0 --output-name w13frozen --groups-file "$PWD/research/iterations/2026-10-05-wave2/complement_hold.npy" >> "$D/frozen-predict.log" 2>&1
fi
for mode in fullcorr fullwmse headwmse; do
    P=w13base_${mode}_s${seed}
    test ! -e "artifacts/runs/$P"
    log="artifacts/logs/$P.log"
    loss=wmse; lr=.00003; extra=()
    if [ "$mode" = fullcorr ]; then loss=gcorr; fi
    if [ "$mode" = headwmse ]; then lr=.001; extra=(--freeze inp,gru,head.0); fi
    scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source valid --valid-fold 0 --eval-source vfold --full-valid \
        --loss "$loss" --mask-weight --hidden 128 --layers 2 --chunk 2000 --batch 16 --raw-only --linear-baseline --precision fp32 \
        --epochs 2 --lr "$lr" --init w13base_direct@ep0 --seed "$seed" --fast "${extra[@]}" > "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" valid --epoch 1 >> "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" train --epoch 1 --groups-file "$PWD/research/iterations/2026-10-05-wave2/complement_hold.npy" >> "$log" 2>&1
    echo "DONE $(date -Is) $P" >> "$log"
done
