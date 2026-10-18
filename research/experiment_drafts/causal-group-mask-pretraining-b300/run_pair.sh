#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_pair.sh profile|pair FOLD SEED}
fold=${2:?usage: run_pair.sh profile|pair FOLD SEED}
seed=${3:?usage: run_pair.sh profile|pair FOLD SEED}
repo_root=$(git rev-parse --show-toplevel)
project_root=${WNN_PROJECT_ROOT:-$repo_root}
python=${WNN_PYTHON:-$repo_root/.venv/bin/python}
log_root=${WNN_EXPERIMENT_LOG_ROOT:-$project_root/research_runs/causal-group-mask-pretraining}
split=$repo_root/research/campaign_2026-09-29/artifacts/splits/fold$fold.json
stats=$project_root/artifacts/masked_group_pretraining/fold${fold}_stats.npz
prefix=gmf_f${fold}_s$seed
pretrain_dir=$project_root/artifacts/masked_group_pretraining/$prefix
pretrained=$pretrain_dir/pretrained.pt
pair_manifest=$project_root/artifacts/masked_group_pretraining/${prefix}_pair.json

case "$mode" in profile|pair) ;; *) echo "mode must be profile or pair" >&2; exit 2 ;; esac
case "$fold" in 0|1|2|3) ;; *) echo "fold must be 0, 1, 2 or 3" >&2; exit 2 ;; esac
case "$seed" in 1001|1002|1003|1004|1005|1006|1007|1008) ;; *) echo "seed must be in 1001..1008" >&2; exit 2 ;; esac
if [[ "$fold" -ge 2 && "$seed" -eq 1008 ]]; then
  echo "folds 2 and 3 use seeds 1001..1007" >&2
  exit 2
fi
if [[ ! -x "$python" || ! -f "$project_root/datasets/train.parquet" || ! -f "$project_root/datasets/valid.parquet" ]]; then
  echo "Set WNN_PYTHON and WNN_PROJECT_ROOT to the provisioned environment and approved data mount" >&2
  exit 2
fi
if [[ ! -f "$split" ]]; then
  echo "The frozen source-qualified split manifest is missing" >&2
  exit 2
fi
mkdir -p "$log_root" "$pretrain_dir"

# Multiple seed cells may share fold statistics; serialize creation and verification.
exec {stats_lock_fd}>"${stats}.lock"
flock "$stats_lock_fd"
stats_seed=$((880301 + fold))
stats_args=()
if [[ -f "$stats" ]]; then
  stats_args+=(--verify-only)
fi
"$python" -u "$repo_root/scripts/pretrain_masked_groups.py" stats \
  --root "$project_root" --split-manifest "$split" --output "$stats" \
  --seed "$stats_seed" --sample-sequences 64 --row-stride 64 --cpu-threads 1 \
  "${stats_args[@]}" \
  2>&1 | tee "$log_root/fold${fold}_stats.log"
flock -u "$stats_lock_fd"

run_task_profile() {
  run_name=${prefix}_task_profile
  "$python" -u "$repo_root/scripts/train_rnn.py" \
    --name "$run_name" --device cuda --arch gru --cell gru --enc linear \
    --hidden 512 --layers 1 --dropout 0 --batch 8 --chunk 2000 \
    --valid-fold "$fold" --split-manifest "$split" --train-source train+valid \
    --train-seqs 10607 --valid-train-fraction 1 --valid-repeat 1 \
    --n-valid-folds 4 --full-valid --eval-source vfold \
    --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
    --no-eval-during-fit --loss wmse --loss-norm chunk \
    --batch-order seeded --optim adamw --lr 0.0001 --beta2 0.999 --wd 0.01 \
    --sched onecycle --warmup-frac 0.05 --clip 1 --precision fp32 \
    --grad-stats --deterministic --sync-free --fused-adam \
    --seed "$seed" --profile-steps 4 --profile-warmup 1 \
    2>&1 | tee "$log_root/$run_name.log"
}

run_task_arm() {
  arm=$1
  epochs=$2
  checkpoint_epoch=$3
  init_name=$4
  run_name=${prefix}_$arm
  resume_args=()
  if [[ $(printenv WNN_RESUME 2>/dev/null || true) == "1" ]]; then
    resume_args=(--resume-state "$project_root/artifacts/runs/$run_name/training_state.pt")
  fi
  "$python" -u "$repo_root/scripts/train_rnn.py" \
    --name "$run_name" --device cuda --arch gru --cell gru --enc linear \
    --hidden 512 --layers 1 --dropout 0 --batch 8 --chunk 2000 \
    --valid-fold "$fold" --split-manifest "$split" --train-source train+valid \
    --train-seqs 10607 --valid-train-fraction 1 --valid-repeat 1 \
    --n-valid-folds 4 --full-valid --eval-source vfold \
    --epochs "$epochs" --checkpoint-policy fixed --checkpoint-epoch "$checkpoint_epoch" \
    --no-eval-during-fit --loss wmse --loss-norm chunk \
    --batch-order seeded --optim adamw --lr 0.0001 --beta2 0.999 --wd 0.01 \
    --sched onecycle --warmup-frac 0.05 --clip 1 --precision fp32 \
    --grad-stats --deterministic --sync-free --fused-adam --save-training-state \
    --seed "$seed" --init "$init_name" --inherit-init-preprocessing \
    "${resume_args[@]}" \
    2>&1 | tee "$log_root/$run_name.log"
}

if [[ "$mode" == "profile" ]]; then
  run_task_profile
  "$python" -u "$repo_root/scripts/pretrain_masked_groups.py" pretrain \
    --root "$project_root" --split-manifest "$split" --stats "$stats" \
    --output-dir "$pretrain_dir/profile" --seed "$seed" --device cuda \
    --hidden 512 --batch-size 4 --epochs 2 --chunk-rows 2000 \
    --mask-block-rows 32 --learning-rate 0.0001 --weight-decay 0.01 \
    --max-grad-norm 1 --profile-updates 2 \
    2>&1 | tee "$log_root/${prefix}_pretrain_profile.log"
  exit 0
fi

if [[ ! -f "$pretrained" ]]; then
  resume_args=()
  pretrain_resume=$(printenv WNN_PRETRAIN_RESUME 2>/dev/null || true)
  if [[ -n "$pretrain_resume" ]]; then
    resume_args=(--resume "$pretrain_resume")
  fi
  "$python" -u "$repo_root/scripts/pretrain_masked_groups.py" pretrain \
    --root "$project_root" --split-manifest "$split" --stats "$stats" \
    --output-dir "$pretrain_dir" --seed "$seed" --device cuda \
    --hidden 512 --batch-size 4 --epochs 2 --chunk-rows 2000 \
    --mask-block-rows 32 --learning-rate 0.0001 --weight-decay 0.01 \
    --max-grad-norm 1 "${resume_args[@]}" \
    2>&1 | tee "$log_root/${prefix}_pretrain.log"
fi

if [[ ! -f "$pair_manifest" ]]; then
  "$python" -u "$repo_root/scripts/prepare_masked_group_initializers.py" \
    --root "$project_root" --split-manifest "$split" \
    --pretrained "$pretrained" --stats "$stats" \
    --seed "$seed" --output-prefix "$prefix" \
    --hidden 512 \
    2>&1 | tee "$log_root/${prefix}_initializers.log"
else
  "$python" -u "$repo_root/scripts/prepare_masked_group_initializers.py" \
    --root "$project_root" --split-manifest "$split" \
    --pretrained "$pretrained" --stats "$stats" \
    --seed "$seed" --output-prefix "$prefix" --hidden 512 --verify-only \
    2>&1 | tee "$log_root/${prefix}_initializer_verification.log"
fi

control_init=${prefix}_control_init
masked_init=${prefix}_masked_init
if (( (fold + seed) % 2 )); then
  run_task_arm c1 8 7 "$control_init"
  run_task_arm masked 6 5 "$masked_init"
  run_task_arm c0 6 5 "$control_init"
else
  run_task_arm c0 6 5 "$control_init"
  run_task_arm masked 6 5 "$masked_init"
  run_task_arm c1 8 7 "$control_init"
fi
