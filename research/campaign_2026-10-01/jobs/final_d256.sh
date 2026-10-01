#!/usr/bin/env bash
# Final distilled GRU256 since student on train + dev valid (holdout reserved, as for the incumbent parents),
# teachers = the incumbent's since parents, then GCLP FT exporting head-only and two-stage members.
#   final_d256.sh SEED
set -euo pipefail
cd "$(dirname "$0")/../../.."
seed=$1
for t in ab67_final_since_s3 ab67_final_since_s4; do
  [[ -f cache/pred_${t}_train.npy ]] || .venv/bin/python scripts/predict_run.py "$t" train
  [[ -f cache/pred_${t}_valid.npy ]] || .venv/bin/python scripts/predict_run.py "$t" valid
done
n=c20261001_final_d256_s${seed}
.venv/bin/python scripts/train_rnn.py --name "$n" --seed "$seed" --train-source train+valid --eval-source holdout \
  --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity prop_v1 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --hidden 256 --layers 2 --batch 32 --chunk 2000 --epochs 4 --lr 0.001 --wd 0.01 \
  --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast --since-weight 32 --since-floor 0.25 \
  --distill ab67_final_since_s3,ab67_final_since_s4 --distill-alpha 0.5 --checkpoint-policy fixed --checkpoint-epoch 3
FTW=(--train-source valid --eval-source holdout --full-valid --mask-weight --seed 3 --epochs 1 --batch 16 --chunk 2000
     --fast --inherit-init-preprocessing --loss gcorr --checkpoint-policy fixed --checkpoint-epoch 0)
.venv/bin/python scripts/train_rnn.py --name "${n}_gcorrh" --init "${n}@ep3" "${FTW[@]}" --freeze inp,gru --lr 5e-4
.venv/bin/python scripts/train_rnn.py --name "${n}_gcorr" --init "${n}_gcorrh@ep0" "${FTW[@]}" --freeze "" --lr 2e-4
for r in "$n" "${n}_gcorrh" "${n}_gcorr"; do
  ep=0; [[ $r == "$n" ]] && ep=3
  .venv/bin/python scripts/predict_run.py "$r" valid --epoch "$ep"
done
echo "FINAL_D256_DONE $n"
