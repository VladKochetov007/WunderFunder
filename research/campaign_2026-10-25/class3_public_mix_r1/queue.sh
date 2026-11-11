#!/usr/bin/env bash
# Sequential single-GPU queue: wait F0 fit, export+score F0, then F1 control/candidate fits, exports, score.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020
D=$ROOT/research/campaign_2026-10-25/class3_public_mix_r1
PY=$ROOT/.venv/bin/python
cd $ROOT
export WNN_PROJECT_ROOT=$ROOT TMPDIR=$WT WNN_THREADS=2 WNN_LOADER_WORKERS=0 WNN_LOADER_RAM_GB=3.5 \
  WNN_GPU_MEMORY_BYTES=4500000000 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $D/queue.log; }
while systemctl --user is-active -q wnn-c3mix-f0; do sleep 60; done
[ "$(systemctl --user show -p Result --value wnn-c3mix-f0)" = success ] || { log "F0 fit failed"; exit 1; }
export_run(){ # name fold
  PYTHONPATH=$WT/src $PY $WT/scripts/predict_run.py $1 valid --epoch 1 \
    --campaign-split-manifest $ROOT/research/campaign_2026-09-29/artifacts/splits/fold$2.json \
    --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --loader-workers 0 --output-name $1; }
score(){ PYTHONPATH=$ROOT/src:$ROOT/scripts $PY $ROOT/scripts/screen_score.py $D/screen_f$1.json --output $D/grade_f$1_R1.json; }
fit(){ ARGV=$(sed "s#^#$PY #" $1); PYTHONPATH=$WT/src $ARGV; }
log "export F0 candidate"; export_run c3mix_f0_s3_cw143_e2_r1 0
log "score F0"; score 0 || log "F0 score failed"
for arm in control candidate; do
  name=$(tr ' ' '\n' < $D/${arm}_f1.argv | rg -A1 -- '^--name$' | tail -1)
  if [ ! -f $ROOT/cache/pred_${name}_valid.npy ]; then log "fit F1 $arm $name"; fit $D/${arm}_f1.argv; log "export F1 $name"; export_run $name 1; fi
done
log "score F1"; score 1 || log "F1 score failed"
log "queue done"
