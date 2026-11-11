#!/usr/bin/env bash
# Stage-6 matched pair: control then candidate (strict FP32, fixed CP5), exports, score.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020
D=$ROOT/research/campaign_2026-10-25/class3_public_mix_r1
PY=$ROOT/.venv/bin/python
cd $ROOT
export WNN_PROJECT_ROOT=$ROOT TMPDIR=$WT WNN_THREADS=2 WNN_LOADER_WORKERS=0 WNN_LOADER_RAM_GB=3.5 \
  WNN_GPU_MEMORY_BYTES=4500000000 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $D/queue.log; }
for arm in control candidate; do
  f=$D/stage6_${arm}_f0.argv
  name=$(tr ' ' '\n' < $f | rg -A1 -- '^--name$' | tail -1)
  log "fit stage6 $name"; PYTHONPATH=$WT/src $PY $(cat $f)
  log "export $name"
  PYTHONPATH=$WT/src $PY $WT/scripts/predict_run.py $name valid --epoch 5 \
    --campaign-split-manifest $ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json \
    --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --loader-workers 0 --output-name $name
done
log "score stage6"; PYTHONPATH=$ROOT/src:$ROOT/scripts $PY $ROOT/scripts/screen_score.py $D/screen_stage6_f0.json --output $D/grade_stage6_f0_R1.json
log "queue3 done"
