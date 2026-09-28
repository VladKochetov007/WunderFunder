#!/usr/bin/env bash
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
export WNN_GPU_FRACTION=0.88 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=4 WNN_CPU_PCT=75
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
FOLD=${1:?fold 0 or 1 required}
COMMON=(--n-valid-folds 4 --valid-fold "$FOLD" --eval-source vfold --full-valid
  --aux-struct 0.3 --aux-next 0.3 --hidden 128 --seed 3 --fast)
BASE=(--train-source train+valid --loss wmse --propensity prop_v1 --prop-power 2
  --prop-eps 0.03 --valid-true-mask 0.03 --epochs 4 --ema-teacher 1.0
  --ema-decay 0.999 --eval-ema --since-weight 32 --since-floor 0.25
  --epoch-samples 12000)
FT=(--train-source valid --mask-weight --loss gcorr --epochs 1 --batch 16
  --epoch-samples 1408)
for fraction in 0.8 1.0; do
  suffix=$([ "$fraction" = 0.8 ] && echo p80 || echo p100)
  name="ab80_validvolume_${suffix}_f${FOLD}"
  WNN_MEM_MAX=22G scripts/limited "$PY" scripts/train_rnn.py --name "$name" \
    --valid-train-fraction "$fraction" "${COMMON[@]}" "${BASE[@]}" \
    > "artifacts/logs/${name}.log" 2>&1
  for stage in head final; do
    if [ "$stage" = head ]; then
      init="$name"; output="ab80_validvolume_${suffix}_gcorrh_f${FOLD}"; lr=5e-4; freeze=(--freeze inp,gru)
    else
      init="ab80_validvolume_${suffix}_gcorrh_f${FOLD}"; output="ab80_validvolume_${suffix}_gcorr_f${FOLD}"; lr=2e-4; freeze=()
    fi
    WNN_MEM_MAX=12G scripts/limited "$PY" scripts/train_rnn.py --name "$output" \
      --init "$init" --lr "$lr" --valid-train-fraction "$fraction" \
      "${freeze[@]}" "${COMMON[@]}" "${FT[@]}" \
      > "artifacts/logs/${output}.log" 2>&1
  done
  WNN_MEM_MAX=10G WNN_CPU_PCT=30 scripts/limited "$PY" scripts/predict_run.py \
    "$output" valid >> "artifacts/logs/${output}.log" 2>&1
  rg '^BEST|^loader workers' "artifacts/logs/${name}.log" "artifacts/logs/${output}.log" | tail -4
done
WNN_MEM_MAX=8G WNN_CPU_PCT=20 scripts/limited "$PY" scripts/probe_pure_pair.py \
  ab62_base_s4 ab80_validvolume_p100_gcorr ab80_validvolume_p80_gcorr \
  --folds "$FOLD" --pure-chains --per-fold --boot 500
WNN_MEM_MAX=8G WNN_CPU_PCT=20 scripts/limited "$PY" scripts/probe_pure_pair.py \
  ab62_base_s4 ab80_validvolume_p100_gcorr ab62_since \
  --folds "$FOLD" --pure-chains --per-fold --boot 500
