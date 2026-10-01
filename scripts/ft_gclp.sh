#!/usr/bin/env bash
# Two-stage gcorr LP-FT on validation rows with true masks (recipe of ab63_gcorr / ab67_*_gcorr).
#   scripts/ft_gclp.sh PARENT[@epN] OUT FOLD|final SEED [campaign/train flags]
# FT_STAGES=head stops after the frozen-trunk stage and exports OUTh instead of the full-FT OUT.
# Parent preprocessing and model dimensions are inherited; --freeze applies to the head stage only.
# FOLD=k trains on validation chains outside fold k and predicts all of valid;
# FOLD=final trains on dev sequences (never the reserved holdout); FOLD=finalall also on the holdout.
set -euo pipefail
cd "$(dirname "$0")/.."
parent=$1 out=$2 fold=$3 seed=$4; shift 4
PY=.venv/bin/python
head_freeze=inp,gru
extra=()
while (( $# )); do
  case "$1" in
    --freeze|--head-freeze)
      head_freeze=$2; shift 2 ;;
    --freeze=*|--head-freeze=*)
      head_freeze=${1#*=}; shift ;;
    *) extra+=("$1"); shift ;;
  esac
done
if [[ $fold == final ]]; then
  split=(--eval-source holdout)
elif [[ $fold == finalall ]]; then
  split=(--eval-source holdout --include-heldout-valid)  # all valid; holdout score is in-sample
else
  split=(--valid-fold "$fold" --eval-source vfold)
fi
common=(--train-source valid --n-valid-folds 4 --full-valid --loss gcorr --mask-weight --batch 16 --epochs 1
        --chunk 2000 --inherit-init-preprocessing --seed "$seed" --fast "${split[@]}" "${extra[@]}")
runner=(scripts/limited)
if [[ ${WNN_IN_CAMPAIGN_JOB:-0} == 1 ]]; then
  runner=()
fi
"${runner[@]}" "$PY" scripts/train_rnn.py --name "${out}h" --init "$parent" \
  "${common[@]}" --freeze "$head_freeze" --lr 5e-4 > "artifacts/logs/${out}h.log" 2>&1
if [[ ${FT_STAGES:-both} == head ]]; then
  # head-only LP: keep the frozen-trunk stage as the member (wide parents lose their gain under full FT)
  "${runner[@]}" "$PY" scripts/predict_run.py "${out}h" valid --epoch 0 >> "artifacts/logs/${out}h.log" 2>&1
  echo "FT_DONE ${out}h model_ep0.pt"
  exit 0
fi
"${runner[@]}" "$PY" scripts/train_rnn.py --name "$out" --init "${out}h@ep0" \
  "${common[@]}" --freeze "" --lr 2e-4 > "artifacts/logs/${out}.log" 2>&1
"${runner[@]}" "$PY" scripts/predict_run.py "$out" valid --epoch 0 >> "artifacts/logs/${out}.log" 2>&1
echo "FT_DONE $out model_ep0.pt"
