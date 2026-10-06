#!/usr/bin/env bash
# Full-data raw-only corr-loss (mix) member: Muon x6 parent on train+valid, then full-network FT on valid (true mask).
#   final_raw.sh SEED GPU
set -euo pipefail; cd ~/WunderFunder-creative
s=$1 g=$2; N=w2fts_s$s; log=artifacts/logs/$N.log
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$PWD/research/iterations/2026-10-05-wave2/scripts:$HOME/WunderFunder/src"
export CUDA_VISIBLE_DEVICES=$g WNN_GPU_FRACTION=0.60 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=10 WNN_MEM_MAX=40G
echo "START $(date -Is) $N" >> $log
scripts/limited .venv/bin/python research/iterations/2026-10-05-wave2/scripts/train_rnn.py --name $N --train-source train+valid --include-heldout-valid --full-valid \
  --eval-source holdout --loss wmse --arch twostream --aux-struct 0.3 --aux-next 0.3 --propensity prop_f0 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --hidden 128 --layers 2 --chunk 2000 --epochs 4 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25 \
  --seed $s --fast --optimizer muon --muon-lr-mult 6 >> $log 2>&1
scripts/limited .venv/bin/python research/iterations/2026-10-05-wave2/scripts/train_rnn.py --name ${N}ft --train-source valid --include-heldout-valid --full-valid \
  --eval-source holdout --loss gcorr --mask-weight --batch 16 --aux-struct 0.3 --aux-next 0.3 --hidden 128 --layers 2 --chunk 2000 \
  --arch twostream --init ${N}@ep3 --seed $s --fast --epochs 2 --lr 1e-4 >> $log 2>&1
echo "DONE $(date -Is) $N" >> $log
