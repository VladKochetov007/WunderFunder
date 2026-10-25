#!/usr/bin/env bash
# Strict-FP32 Muon on the strong warm stage-6 student lineage (F0, seed 3, fixed CP5) vs reused AdamW control.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
W=/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020
O=$ROOT/research/campaign_2026-10-25/optimizer_panel_r1
PY=$ROOT/.venv/bin/python
cd $ROOT
export WNN_PROJECT_ROOT=$ROOT TMPDIR=$W WNN_THREADS=2 WNN_LOADER_WORKERS=0 WNN_LOADER_RAM_GB=3.5 \
  WNN_GPU_MEMORY_BYTES=4500000000 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/stage6_muon.log; }
while systemctl --user is-active -q wnn-optpanel || systemctl --user is-active -q wnn-clippanel; do sleep 60; done
log "fit muon6_f0_s3_rnn_e6_r1"; PYTHONPATH=$W/src $PY $(cat $O/stage6_muon_f0.argv) > $O/muon6.fit.log 2>&1
log "export"; PYTHONPATH=$W/src $PY $W/scripts/predict_run.py muon6_f0_s3_rnn_e6_r1 valid --epoch 5 \
  --campaign-split-manifest $ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --loader-workers 0 --output-name muon6_f0_s3_rnn_e6_r1 > $O/muon6.export.log 2>&1
log "score"; PYTHONPATH=$ROOT/src:$ROOT/scripts $PY $ROOT/scripts/screen_score.py $O/screen_stage6_muon_f0.json --output $O/grade_stage6_muon_f0_R1.json > $O/muon6.score.log 2>&1
log "done"
