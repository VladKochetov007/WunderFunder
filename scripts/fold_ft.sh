#!/usr/bin/env bash
# Two-stage gcorr LP-FT of a fold parent on validation rows with true masks: the fine-tune stage of the
# public-champion recipe (scripts/ft_gclp.sh), with fold_eval.py naming.
#   scripts/fold_ft.sh NAME PARENT FOLD GPU SEED [extra flags matching the parent, e.g. --hidden 256]
# Parent: artifacts/runs/PARENT_fFOLD/model_ep3.pt (fixed final epoch). Trains on validation chains outside
# the held-out fold (head first with the trunk frozen, then everything), predicts all validation sequences
# into cache/pred_NAME_fFOLD_valid.npy. Log: artifacts/logs/NAME_fFOLD.log. FT2_LR sets the full-network
# stage learning rate (default 2e-4, the champion value).
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 parent=$2 fold=$3 gpu=$4 seed=$5; shift 5
out="${name}_f${fold}"
log="artifacts/logs/${out}.log"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=${WNN_GPU_FRACTION:-0.5} WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2
export WNN_CPU_PCT=${WNN_CPU_PCT:-8} WNN_MEM_MAX=${WNN_MEM_MAX:-16G}
common=(--train-source valid --n-valid-folds 4 --full-valid --loss gcorr --mask-weight --batch 16 --epochs 1
        --aux-struct 0.3 --aux-next 0.3 --hidden 128 --seed "$seed" --fast
        --valid-fold "$fold" --eval-source vfold "$@")
echo "START $(date -Is) gpu=$gpu $out parent=${parent}_f${fold}@ep3 ${*}" >> "$log"
scripts/limited .venv/bin/python scripts/train_rnn.py --name "${out}h" --init "${parent}_f${fold}@ep3" \
  --freeze inp,gru --lr 5e-4 "${common[@]}" >> "$log" 2>&1
scripts/limited .venv/bin/python scripts/train_rnn.py --name "$out" --init "${out}h" --lr "${FT2_LR:-2e-4}" "${common[@]}" >> "$log" 2>&1
scripts/limited .venv/bin/python scripts/predict_run.py "$out" valid >> "$log" 2>&1
# head-only stage (trunk frozen) as its own candidate: cache/pred_NAMEH_fFOLD_valid.npy
scripts/limited .venv/bin/python scripts/predict_run.py "${out}h" valid --output-name "${name}H_f${fold}" >> "$log" 2>&1
echo "DONE $(date -Is) $out" >> "$log"
