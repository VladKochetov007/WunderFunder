#!/usr/bin/env bash
# Throughput ladder: one optimization per rung, same config, each rung run twice; prints PROFILE lines.
#   bash scripts/perf_ladder.sh [ema]     (ema: add the mean-teacher EMA recipe flags)
set -u
cd "$(dirname "$0")/.."
EXTRA=""
[ "${1:-}" = "ema" ] && EXTRA="--ema-teacher 1.0 --ema-decay 0.999 --eval-ema"
REC="--name perf_ladder --train-source train+valid --valid-fold 0 --n-valid-folds 4 --eval-source vfold --loss wmse --aux-struct 0.3 --aux-next 0.3 --propensity prop_v1 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 --hidden 128 --seed 3 --epochs 1 --profile-steps 20 --profile-warmup 5 $EXTRA"
export WNN_GPU_FRACTION=0.9 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=60 WNN_MEM_MAX=20G
run() {  # $1 label, $2 duty, rest flags
  local label=$1 duty=$2; shift 2
  for rep in 1 2; do
    out=$(WNN_GPU_DUTY=$duty scripts/limited .venv/bin/python scripts/train_rnn.py $REC "$@" 2>&1 | grep '^PROFILE' | tail -1)
    echo "$label rep$rep $out"
  done
}
run "V0_baseline(duty0.9)" 0.9
run "V1_no_duty" 1.0
run "V2_+sync_free" 1.0 --sync-free
run "V3_+fused_adam" 1.0 --sync-free --fused-adam
run "V4_+pin_prefetch" 1.0 --sync-free --fused-adam --pin-memory
run "V5_+prop_cache" 1.0 --sync-free --fused-adam --pin-memory --prop-cache
run "V6_+bf16" 1.0 --sync-free --fused-adam --pin-memory --prop-cache --precision bf16
echo LADDERDONE
