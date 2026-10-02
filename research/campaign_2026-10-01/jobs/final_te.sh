#!/usr/bin/env bash
# Final teacher-ensemble student (H-060; run only if c20261002_te07_f1 confirms): GRU256 since student on train + dev
# valid (holdout reserved), teachers = incumbent since parents s3+s4 + the final alpha-0.7 student, then GCLP FT.
#   final_te.sh SEED ALPHA
set -euo pipefail
export WNN_VRAM_GB=${WNN_VRAM_GB:-4.6} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 14 GB slice: three teacher arrays + respawned loader workers OOMed at the epoch-0 boundary (16:16) with 4 workers
export WNN_LOADER_WORKERS=3 WNN_LOADER_RAM_GB=3  # hard-set: the campaign launcher already sets these
cd "$(dirname "$0")/../../.."
seed=$1 alpha=$2 st=c20261002_final_d256a07_s3
[[ -f cache/pred_${st}_train.npy ]] || .venv/bin/python scripts/predict_run.py "$st" train --epoch 3
n=c20261002_final_te${alpha/./}_s${seed}
.venv/bin/python scripts/train_rnn.py --name "$n" --seed "$seed" --train-source train+valid --eval-source holdout \
  --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity prop_v1 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --hidden 256 --layers 2 --batch 32 --chunk 2000 --epochs 4 --lr 0.001 --wd 0.01 \
  --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast --since-weight 32 --since-floor 0.25 \
  --distill "ab67_final_since_s3,ab67_final_since_s4,${st}" --distill-alpha "$alpha" --checkpoint-policy fixed \
  --checkpoint-epoch 3 --no-eval-during-fit
FTW=(--train-source valid --eval-source holdout --full-valid --mask-weight --seed 3 --epochs 1 --batch 16 --chunk 2000
     --fast --inherit-init-preprocessing --loss gcorr --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit)
.venv/bin/python scripts/train_rnn.py --name "${n}_gcorrh" --init "${n}@ep3" "${FTW[@]}" --freeze inp,gru --lr 5e-4
for r in "$n" "${n}_gcorrh"; do
  ep=0; [[ $r == "$n" ]] && ep=3
  .venv/bin/python scripts/predict_run.py "$r" valid --epoch "$ep"
done
echo "FINAL_TE_DONE $n"
