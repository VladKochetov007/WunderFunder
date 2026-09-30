#!/usr/bin/env bash
# Head-only gcorr fine-tune of an existing final parent on all validation rows (true masks).
#   scripts/final_ft.sh OUT PARENT GPU SEED [extra flags, e.g. --hidden 256]   (FT_EPOCHS, default 1)
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1 parent=$2 gpu=$3 seed=$4; shift 4
log="artifacts/logs/${out}.log"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=${WNN_GPU_FRACTION:-0.5} WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2
export WNN_CPU_PCT=${WNN_CPU_PCT:-8} WNN_MEM_MAX=${WNN_MEM_MAX:-16G}
scripts/limited .venv/bin/python scripts/train_rnn.py --name "$out" --init "${parent}@ep3" \
  --train-source valid --include-heldout-valid --eval-source holdout --full-valid --loss gcorr --mask-weight \
  --batch 16 --epochs "${FT_EPOCHS:-1}" --aux-struct 0.3 --aux-next 0.3 --hidden 128 --seed "$seed" --fast \
  --freeze inp,gru --lr 5e-4 "$@" >> "$log" 2>&1
echo "DONE $(date -Is) $out" >> "$log"
