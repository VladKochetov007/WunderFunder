#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave6"
export CUDA_VISIBLE_DEVICES=5 WNN_GPU_FRACTION=.65 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
for seed in 3 4; do
  for mode in direct distilled; do
    P=w12compress512_${mode}_s${seed}
    test ! -e "artifacts/runs/$P"
    log="artifacts/logs/$P.log"
    alpha=0; if [ "$mode" = distilled ]; then alpha=.9; fi
    scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source valid --valid-fold 0 --eval-source vfold --full-valid \
        --loss wmse --mask-weight --aux-struct .3 --aux-next .3 --hidden 512 --layers 2 --chunk 2000 --batch 16 \
        --distill w6fullteacher --distill-alpha "$alpha" --family-teachers d256bc1k_f0,w6family_full,w6family_raw,orgbase \
        --epochs 2 --lr .0001 --init "w12merge_s${seed}@ep0" --seed "$seed" --fast > "$log" 2>&1
    scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" valid --epoch 1 >> "$log" 2>&1
    echo "DONE $(date -Is) $P" >> "$log"
  done
done
