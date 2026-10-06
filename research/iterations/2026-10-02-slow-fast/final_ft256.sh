#!/usr/bin/env bash
# Valid fine-tune variants on existing full-data parents.  final_ft.sh NAME PARENT SEED GPU "FT flags"
set -euo pipefail
cd ~/WunderFunder-creative
name=$1 parent=$2 seed=$3 gpu=$4 ft=$5
log=artifacts/logs/${name}.log
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=0.25 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=8 WNN_MEM_MAX=16G
echo "START $(date -Is) $name parent=$parent ft=[$ft]" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name $name --train-source valid --include-heldout-valid --full-valid \
  --eval-source holdout --loss gcorr --mask-weight --batch 16 --aux-struct 0.3 --aux-next 0.3 --hidden 256 --layers 2 --chunk 2000 \
  --init ${parent}@ep3 --seed $seed --fast $ft >> $log 2>&1
echo "DONE $(date -Is) $name" >> $log
