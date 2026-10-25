#!/usr/bin/env bash
# Final all-data H390-chain students with strict-FP32 Muon (only optimizer/precision differ from H390 S3/S4).
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack; cd $ROOT
F=$ROOT/research/campaign_2026-10-26/final_muon_students_r1
W=/home/vlad/development/wunder/research_tmp/final-muon-students-20261026
export WNN_PROJECT_ROOT=$ROOT PYTHONPATH=$W/src WNN_THREADS=2 OMP_NUM_THREADS=2 WNN_LOADER_WORKERS=0 TMPDIR=/home/vlad/development/wunder/research_tmp WNN_GPU_MEMORY_BYTES=4500000000
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $F/pipeline.log; }
while systemctl --user is-active -q wnn-muon6 || systemctl --user is-active -q wnn-tcache; do sleep 60; done
rg -q "cache done" $F/pipeline.log || { log "cache missing; abort"; exit 1; }
for s in 3 4; do
  name=vt_all_h390m_s${s}_r1
  [ -f artifacts/runs/$name/model_ep5.pt ] && { log "skip $name"; continue; }
  log "fit $name"
  .venv/bin/python $W/scripts/train_rnn.py --name $name --seed $s \
    --device cuda --arch gru --cell gru --hidden 256 --layers 2 --dropout 0 \
    --train-source train+valid --train-seqs 10607 --valid-seqs 0 --valid-fold -1 \
    --include-heldout-valid --no-eval-during-fit \
    --init vt_all_fin_mu256p0e4_s${s}@ep3 --inherit-init-preprocessing \
    --final-distill-cache-root $ROOT/cache/h390_all_data_homogeneous_r2 --distill-alpha 0.9 \
    --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
    --batch 16 --chunk 2000 --loss wmse --loss-norm chunk \
    --lr 0.0001 --wd 0.01 --optim muon --muon-scope rnn --sched onecycle --precision fp32 --strict-fp32-kernels \
    --propensity c20260929_prop_f1 --prop-power 2 --prop-eps 0.03 \
    --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 --since-weight 32 \
    --fast --save-training-state > $F/fit_$name.log 2>&1
  log "done $name"
done
log "finals done"
