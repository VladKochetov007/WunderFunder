#!/usr/bin/env bash
# Final distilled GRU256 since student with a chosen distillation weight (same data/teachers/FT as final_d256.sh,
# whose alpha-0.5 student scored 0.6641 public alone): train + dev valid, holdout reserved.
#   final_d256a.sh SEED ALPHA
set -euo pipefail
cd "$(dirname "$0")/../../.."
seed=$1 alpha=$2
n=c20261002_final_d256a${alpha/./}_s${seed}
.venv/bin/python scripts/train_rnn.py --name "$n" --seed "$seed" --train-source train+valid --eval-source holdout \
  --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity prop_v1 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --hidden 256 --layers 2 --batch 32 --chunk 2000 --epochs 4 --lr 0.001 --wd 0.01 \
  --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast --since-weight 32 --since-floor 0.25 \
  --distill ab67_final_since_s3,ab67_final_since_s4 --distill-alpha "$alpha" --checkpoint-policy fixed \
  --checkpoint-epoch 3 --no-eval-during-fit
FTW=(--train-source valid --eval-source holdout --full-valid --mask-weight --seed 3 --epochs 1 --batch 16 --chunk 2000
     --fast --inherit-init-preprocessing --loss gcorr --checkpoint-policy fixed --checkpoint-epoch 0 --no-eval-during-fit)
.venv/bin/python scripts/train_rnn.py --name "${n}_gcorrh" --init "${n}@ep3" "${FTW[@]}" --freeze inp,gru --lr 5e-4
.venv/bin/python scripts/train_rnn.py --name "${n}_gcorr" --init "${n}_gcorrh@ep0" "${FTW[@]}" --freeze "" --lr 2e-4
for r in "$n" "${n}_gcorrh" "${n}_gcorr"; do
  ep=0; [[ $r == "$n" ]] && ep=3
  .venv/bin/python scripts/predict_run.py "$r" valid --epoch "$ep"
done
echo "FINAL_D256A_DONE $n"
