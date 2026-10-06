#!/usr/bin/env bash
# Final-data members with Muon x6 parents (raw view) + standard AdamW H2 head FT on valid with true mask.
#   final_mu.sh NAME SEED GPU KIND   KIND in {h128, c500, d256}
set -euo pipefail
cd ~/WunderFunder-creative
name=$1 seed=$2 gpu=$3 kind=$4
log=artifacts/logs/${name}.log
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=0.45 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
case $kind in
  h128) arch=(--hidden 128 --layers 2 --chunk 2000); extra=() ;;
  c500) arch=(--hidden 128 --layers 2 --chunk 500); extra=() ;;
  d256) arch=(--hidden 256 --layers 2 --chunk 2000); extra=(--distill fin256_s3,fin256_s4 --distill-alpha 0.5) ;;
esac
echo "START $(date -Is) $name seed=$seed kind=$kind" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name $name --train-source train+valid --include-heldout-valid --full-valid \
  --eval-source holdout --loss wmse "${extra[@]}" --aux-struct 0.3 --aux-next 0.3 --propensity prop_f0 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 "${arch[@]}" --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --seed $seed --fast --optimizer muon --muon-lr-mult 6 ${EXTRA_ARGS:-} >> $log 2>&1
scripts/limited .venv/bin/python scripts/train_rnn.py --name ${name}h2 --train-source valid --include-heldout-valid --full-valid \
  --eval-source holdout --loss gcorr --mask-weight --batch 16 --epochs 2 --lr 5e-4 --aux-struct 0.3 --aux-next 0.3 \
  "${arch[@]}" --init ${name}@ep3 --freeze inp,gru --seed $seed --fast >> $log 2>&1
echo "DONE $(date -Is) $name" >> $log
