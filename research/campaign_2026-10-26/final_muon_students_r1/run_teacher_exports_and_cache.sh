#!/usr/bin/env bash
# Rebuild the H390 all-data teacher cache: full train exports of the three Muon-all H384 teachers
# (valid exports already exist), then the frozen H390 R2 cache builder.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack; cd $ROOT
F=$ROOT/research/campaign_2026-10-26/final_muon_students_r1
CHILD=/home/vlad/development/wunder/research_tmp/h390_packet_extract/h390_all_data_homogeneous_r2/child_source_v1
export WNN_PROJECT_ROOT=$ROOT WNN_THREADS=2 OMP_NUM_THREADS=2 WNN_LOADER_WORKERS=0 TMPDIR=/home/vlad/development/wunder/research_tmp
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $F/pipeline.log; }
for t in vt_f1_muonall_h384_e3 vt_f1_muonall_h384_e3_s4 vt_f1_muonall_h384_e3_s5; do
  if [ ! -f cache/pred_${t}_train.npy ]; then log "export train $t"; PYTHONPATH=$ROOT/src .venv/bin/python scripts/predict_run.py $t train --epoch 2 --device cuda > $F/export_${t}_train.log 2>&1; fi
done
log "build cache"
PYTHONPATH=$CHILD/src .venv/bin/python $CHILD/scripts/build_final_distill_cache.py --data-root $ROOT --teacher-cache-root $ROOT/cache --checkpoint-root $ROOT/artifacts/runs --teacher-split-manifest $ROOT/research/campaign_2026-09-29/artifacts/splits/fold1.json --output $ROOT/cache/h390_all_data_homogeneous_r2 --chunk-sequences 2 --output-budget-gb 2.5 --min-free-gb 20 > $F/build_cache.log 2>&1
log "cache done"
