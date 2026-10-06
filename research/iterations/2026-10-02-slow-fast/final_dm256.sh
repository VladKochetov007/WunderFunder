#!/usr/bin/env bash
# Final-data demean version of the distilled 2x256 member (find256 recipe) + H2 head FT.  final_dm256.sh NAME SEED GPU
set -euo pipefail
cd ~/WunderFunder-creative
name=$1 seed=$2 gpu=$3
log=artifacts/logs/${name}.log
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=0.45 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
echo "START $(date -Is) $name seed=$seed" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name $name --train-source train+valid --include-heldout-valid --full-valid \
  --eval-source holdout --loss wmse --distill fin256_s3,fin256_s4 --distill-alpha 0.5 --aux-struct 0.3 --aux-next 0.3 \
  --propensity prop_f0 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 --hidden 256 --layers 2 --chunk 2000 --epochs 4 \
  --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 --seed $seed --fast --a-mode demean >> $log 2>&1
scripts/limited .venv/bin/python scripts/train_rnn.py --name ${name}h2 --train-source valid --include-heldout-valid --full-valid \
  --eval-source holdout --loss gcorr --mask-weight --batch 16 --epochs 2 --lr 5e-4 --aux-struct 0.3 --aux-next 0.3 \
  --hidden 256 --layers 2 --chunk 2000 --init ${name}@ep3 --freeze inp,gru --seed $seed --fast --a-mode demean >> $log 2>&1
echo "DONE $(date -Is) $name" >> $log
