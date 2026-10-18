#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_cell.sh profile|pair FOLD SEED}
fold=${2:?usage: run_cell.sh profile|pair FOLD SEED}
seed=${3:?usage: run_cell.sh profile|pair FOLD SEED}
repo_root=$(git rev-parse --show-toplevel)
project_root=${WNN_PROJECT_ROOT:-$repo_root}
log_root=${WNN_EXPERIMENT_LOG_ROOT:-$project_root/research_runs/causal-predictive-state-gru-b300}
python=${WNN_PYTHON:-$repo_root/.venv/bin/python}

case "$mode" in
  profile|pair) ;;
  *) echo "mode must be profile or pair" >&2; exit 2 ;;
esac
case "$fold" in
  0|1|2|3) ;;
  *) echo "fold must be 0, 1, 2 or 3" >&2; exit 2 ;;
esac
case "$seed" in
  1001|1002|1003|1004|1005|1006|1007|1008) ;;
  *) echo "seed must be in 1001..1008" >&2; exit 2 ;;
esac
if [[ "$fold" -ge 2 && "$seed" -eq 1008 ]]; then
  echo "folds 2 and 3 use seeds 1001..1007" >&2
  exit 2
fi
if [[ ! -x "$python" ]]; then
  echo "Python executable not found: $python" >&2
  exit 2
fi
if [[ ! -f "$project_root/datasets/train.parquet" || ! -f "$project_root/datasets/valid.parquet" ]]; then
  echo "WNN_PROJECT_ROOT must contain the approved datasets/ mounts" >&2
  exit 2
fi

mkdir -p "$log_root"
split="$repo_root/research/campaign_2026-09-29/artifacts/splits/fold${fold}.json"
if [[ ! -f "$split" ]]; then
  echo "Frozen split manifest missing: $split" >&2
  exit 2
fi

run_arm() {
  local arm=$1
  local weight=$2
  local run_name="cpsi_${arm}_f${fold}_s${seed}"
  local profile_args=()
  local state_args=()

  if [[ "$mode" == "profile" ]]; then
    run_name="${run_name}_profile50"
    profile_args=(--profile-steps 4 --profile-warmup 1)
  else
    state_args=(--save-training-state)
    if [[ "${WNN_RESUME:-0}" == "1" ]]; then
      state_args+=(--resume-state "$project_root/artifacts/runs/$run_name/training_state.pt")
    fi
  fi

  local arm_args=(--aux-next-input-head --aux-next-input-weight "$weight")
  "$python" -u "$repo_root/scripts/train_rnn.py" \
    --name "$run_name" \
    --device cuda --arch gru --cell gru --enc linear \
    --hidden 512 --layers 1 --dropout 0 \
    --batch 8 --chunk 2000 \
    --valid-fold "$fold" --split-manifest "$split" \
    --train-source train+valid --train-seqs 10607 \
    --valid-train-fraction 1 --valid-repeat 1 --n-valid-folds 4 \
    --full-valid --eval-source vfold \
    --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
    --no-eval-during-fit \
    --loss wmse --loss-norm chunk --batch-order seeded \
    --optim adamw --lr 0.0001 --beta2 0.999 --wd 0.01 \
    --sched onecycle --warmup-frac 0.05 --head-lr-mult 1 \
    --clip 1 --precision fp32 --grad-stats --deterministic \
    --sync-free --fused-adam \
    --seed "$seed" \
    "${arm_args[@]}" "${profile_args[@]}" "${state_args[@]}" \
    2>&1 | tee "$log_root/${run_name}.log"
}

if [[ "$mode" == "profile" ]]; then
  run_arm control 0.0
  run_arm candidate 0.10
  exit 0
fi

if (( (fold + seed) % 2 )); then
  run_arm candidate 0.10
  run_arm control 0.0
else
  run_arm control 0.0
  run_arm candidate 0.10
fi
