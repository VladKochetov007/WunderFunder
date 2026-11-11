#!/usr/bin/env bash
# Dose-response successor: after queue.sh finishes, fit F0 class-3 weight 2.0, export, score F0 dose screen.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020
D=$ROOT/research/campaign_2026-10-25/class3_public_mix_r1
PY=$ROOT/.venv/bin/python
cd $ROOT
export WNN_PROJECT_ROOT=$ROOT TMPDIR=$WT WNN_THREADS=2 WNN_LOADER_WORKERS=0 WNN_LOADER_RAM_GB=3.5 \
  WNN_GPU_MEMORY_BYTES=4500000000 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $D/queue.log; }
while systemctl --user is-active -q wnn-c3mix-queue; do sleep 60; done
log "fit F0 cw200"; PYTHONPATH=$WT/src $PY $(cat $D/candidate_f0_cw200.argv)
log "export F0 cw200"
PYTHONPATH=$WT/src $PY $WT/scripts/predict_run.py c3mix_f0_s3_cw200_e2_r1 valid --epoch 1 \
  --campaign-split-manifest $ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --loader-workers 0 --output-name c3mix_f0_s3_cw200_e2_r1
log "score F0 dose"; PYTHONPATH=$ROOT/src:$ROOT/scripts $PY $ROOT/scripts/screen_score.py $D/screen_f0_dose.json --output $D/grade_f0_dose_R1.json
log "queue2 done"
