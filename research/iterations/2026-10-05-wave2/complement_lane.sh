#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave2"
export CUDA_VISIBLE_DEVICES=$1 WNN_GPU_FRACTION=.60 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
scripts/limited .venv/bin/python "$D/prepare_complement.py" > "$D/complement-prepare.log" 2>&1
for seed in 3 4; do
    for kind in direct residual; do
        P=w2comp_${kind}_s${seed}
        test ! -e "artifacts/runs/$P" || { echo "Run exists $P"; exit 1; }
        extra=()
        if [ "$kind" = residual ]; then extra=(--distill w2_reference --complement-scale .2); fi
        scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train --train-half 1 \
            --train-index-file "$D/complement_fit.npy" --valid-fold 0 --eval-source vfold --full-valid \
            --loss wmse --propensity prop_f0 --prop-power 2 --prop-eps .03 --aux-struct .3 --aux-next .3 \
            --hidden 128 --layers 2 --chunk 2000 --batch 16 --epochs 2 --lr 1e-4 --init "cmp_fullft_s${seed}_ft@ep1" \
            --seed "$seed" --fast "${extra[@]}" > "artifacts/logs/$P.log" 2>&1
        for split in valid train; do
            scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" "$split" --epoch 1 >> "artifacts/logs/$P.log" 2>&1
        done
        echo "DONE $(date -Is) $P" >> "artifacts/logs/$P.log"
    done
done
