#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave2"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.65 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
P=w15h256_s${seed}
log=artifacts/logs/$P.log
test ! -e "artifacts/runs/$P"
scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train+valid --train-half 0 --valid-fold 0 --eval-source vfold --full-valid \
 --loss wmse --aux-struct .3 --aux-next .3 --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 \
 --hidden 256 --layers 2 --chunk 2000 --batch 16 --epochs 4 --ema-teacher 1. --ema-decay .999 --eval-ema --since-weight 32 --since-floor .25 \
 --seed "$seed" --fast --optimizer adamw --lr .001 > "$log" 2>&1
for split in valid train; do
 scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" "$split" --epoch 3 >> "$log" 2>&1
done
echo "PARENT DONE $(date -Is)" >> "$log"
# Require complete, atomically renamed prediction caches from both independent parents.
while ! test -f cache/pred_w15h256_s3_train.npy || ! test -f cache/pred_w15h256_s4_train.npy; do sleep 10; done
P=w15d256_s${seed}
log=artifacts/logs/$P.log
test ! -e "artifacts/runs/$P"
scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train+valid --train-half 0 --valid-fold 0 --eval-source vfold --full-valid \
 --loss wmse --aux-struct .3 --aux-next .3 --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 \
 --hidden 256 --layers 2 --chunk 1000 --batch 16 --epochs 4 --ema-teacher 1. --ema-decay .999 --eval-ema --since-weight 32 --since-floor .25 \
 --distill w15h256_s3,w15h256_s4 --distill-alpha .5 --distill-boot-h 8 \
 --seed "$seed" --fast --optimizer muon --muon-lr-mult 3 > "$log" 2>&1
for split in valid train; do
 scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" "$split" --epoch 3 >> "$log" 2>&1
done
echo "DONE $(date -Is) $P" >> "$log"
