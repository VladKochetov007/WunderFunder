#!/usr/bin/env bash
# Two-stage gcorr LP-FT on validation rows with true masks (recipe of ab63_gcorr / ab67_*_gcorr).
#   scripts/ft_gclp.sh PARENT[@epN] OUT FOLD|final SEED [extra arch flags matching the parent]
# FOLD=k trains on validation chains outside fold k and predicts all of valid;
# FOLD=final trains on dev sequences (never the reserved holdout); FOLD=finalall also on the holdout.
set -euo pipefail
cd "$(dirname "$0")/.."
parent=$1 out=$2 fold=$3 seed=$4; shift 4
PY=.venv/bin/python
export WNN_GPU_FRACTION=${WNN_GPU_FRACTION:-0.25} WNN_LOADER_RAM_GB=${WNN_LOADER_RAM_GB:-4} WNN_LOADER_WORKERS=2
if [[ $fold == final ]]; then
  split=(--eval-source holdout)
elif [[ $fold == finalall ]]; then
  split=(--eval-source holdout --include-heldout-valid)  # all valid; holdout score is in-sample
else
  split=(--valid-fold "$fold" --eval-source vfold)
fi
common=(--train-source valid --n-valid-folds 4 --full-valid --loss gcorr --mask-weight --batch 16 --epochs 1
        --aux-struct 0.3 --aux-next 0.3 --hidden 128 --layers 2 --seed "$seed" --fast "${split[@]}" "$@")
WNN_MEM_MAX=8G WNN_CPU_PCT=15 scripts/limited "$PY" scripts/train_rnn.py --name "${out}h" --init "$parent" \
  --freeze inp,gru --lr 5e-4 "${common[@]}" > "artifacts/logs/${out}h.log" 2>&1
WNN_MEM_MAX=8G WNN_CPU_PCT=15 scripts/limited "$PY" scripts/train_rnn.py --name "$out" --init "${out}h" \
  --lr 2e-4 "${common[@]}" > "artifacts/logs/${out}.log" 2>&1
WNN_MEM_MAX=10G WNN_CPU_PCT=15 scripts/limited "$PY" scripts/predict_run.py "$out" valid >> "artifacts/logs/${out}.log" 2>&1
echo "FT_DONE $out $(rg -o 'weighted_pearson[^,]*' "artifacts/logs/${out}.log" | tail -1)"
