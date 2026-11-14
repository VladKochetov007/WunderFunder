#!/usr/bin/env bash
# Nov 15 final-day GPU queue: full-data TLead seeds for the TL3 family (public-best members).
# 1) export missing valid caches of train-only fin256 teachers (the TLead distillation targets);
# 2) all-data TLead fits = q14t_tlead_s5 (PR49 tl3c_fttall_tlead) recipe with train+valid, no eval.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-15/tlead_final_r1
Q=$R/research/campaign_2026-11-14/quality_r2
C=/home/vlad/development/wunder/research_tmp/chrono_root
SPLIT=$R/research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json
cd "$C"
export WNN_PROJECT_ROOT=$C NVIDIA_TF32_OVERRIDE=0 TMPDIR=/home/vlad/development/wunder/research_tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for t in tt29rep_fin256_s3 tt29rep_fin256_s4; do
  test -e "$O/STOP" && exit 0
  if [ ! -e "cache/pred_${t}_valid.npy" ]; then
    WNN_THREADS=1 "$R/.venv/bin/python" -u "$Q/export_predictions_r2.py" "$t" valid --epoch 3 \
      --inference-precision fp32 --campaign-split-manifest "$SPLIT" --campaign-full-valid-feature-only \
      > "$O/$t.valid.log" 2>&1
  fi
  echo "EXPORT_DONE $t $(date -u +%T)"
done
TL="--hidden 256 --layers 2 --dropout 0 --train-source train+valid --include-heldout-valid --train-seqs 10607 \
 --no-eval-during-fit --precision fp32 --checkpoint-policy fixed --checkpoint-epoch 3 --loss-norm chunk \
 --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 --beta2 0.999 --grad-stats --save-training-state \
 --train-eval-seqs 0 --loss wmse --optim tl3-muon --muon-scope rnn --muon-lr-mult 3 --since-weight 32 \
 --since-floor .25 --batch 32 --chunk 1000 --epochs 4 --lr .001 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5 \
 --enc tiedlead --ema-teacher 1 --ema-decay 0.999 --eval-ema --distill tt29rep_fin256_s3,tt29rep_fin256_s4 \
 --distill-alpha 0.5 --distill-boot-h 8 --sync-free --fused-adam --pin-memory"
for seed in ${SEEDS:-7 8 9}; do
  name=q15f_tlead_s$seed
  test -e "$O/STOP" && exit 0
  test -e "artifacts/runs/$name/model_ep3.pt" && continue
  echo "FIT_START $name $(date -u +%T)"
  WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=${CPUQ:-300%} WNN_THREADS=3 WNN_MEM_MAX=${MEMMAX:-12000000000} \
  PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src \
  "$R/scripts/limited" "$R/.venv/bin/python" -u "$R/research/campaign_2026-11-14/local_r1/final_launch.py" \
    /home/vlad/development/wunder/research_tmp/pr49_wt $TL --seed $seed --name $name > "$O/$name.fit.log" 2>&1
  echo "FIT_DONE $name $(date -u +%T)"
done
