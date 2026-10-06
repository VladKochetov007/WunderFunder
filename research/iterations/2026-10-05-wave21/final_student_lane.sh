#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave6"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.65 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
P=w21final_distilled_s${seed};log=artifacts/logs/$P.log
test ! -e "artifacts/runs/$P"
scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train+valid --include-heldout-valid --full-valid --eval-source holdout \
 --loss wmse --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 --aux-struct .3 --aux-next .3 --hidden 256 --layers 2 --chunk 2000 --batch 16 \
 --distill w21final_teacher --distill-alpha .7 --family-teachers w18final_fbc1k_s5,w18final_family_full,w21final_family_twostream,orgbase \
 --epochs 2 --lr .0001 --init "fbc1k_s${seed}@ep3" --seed "$seed" --fast > "$log" 2>&1
echo "DONE $(date -Is) $P; FINAL TRAINING METRICS ARE IN-SAMPLE" >> "$log"
