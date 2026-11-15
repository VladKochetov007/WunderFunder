#!/usr/bin/env bash
# V-ft (valid-only fine-tune, PR49 ft recipe) for blend members: chain-fold twins for local blend eval,
# then production fits on all valid. Afterwards resume queue3 controls (T/C).
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-15/valid_ft_r1
true
cd /home/vlad/development/wunder/research_tmp/chrono_root
COMMON="--hidden 256 --layers 2 --dropout 0 --inherit-init-preprocessing --precision fp32 --checkpoint-policy fixed \
 --checkpoint-epoch 1 --epochs 2 --lr .0001 --optim adamw --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 \
 --beta2 0.999 --loss gcorr --loss-norm chunk --batch 16 --chunk 2000 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5 \
 --grad-stats --train-eval-seqs 0 --train-source valid"
run() { # name init enc extra...
  local name=$1 init=$2 enc=$3; shift 3
  test -e "$O/STOP10" && exit 0
  test -e "artifacts/runs/$name/model_ep1.pt" && return 0
  echo "FIT_START $name $(date -u +%T)"
  WNN_PROJECT_ROOT=$PWD NVIDIA_TF32_OVERRIDE=0 TMPDIR=/home/vlad/development/wunder/research_tmp \
  WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=200% WNN_THREADS=2 WNN_MEM_MAX=12000000000 \
  PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src \
  "$R/scripts/limited" "$R/.venv/bin/python" -u "$R/research/campaign_2026-11-14/local_r1/final_launch.py" \
    /home/vlad/development/wunder/research_tmp/pr49_wt $COMMON --enc $enc --init $init --seed 5 --name $name "$@" \
    > "$O/$name.fit.log" 2>&1 || echo "FIT_FAIL $name"
  echo "FIT_DONE $name $(date -u +%T)"
}
for f in 0 1; do
  run q15v_V6_f$f q14t_tlead_s6@ep3 tiedlead --valid-fold $f --n-valid-folds 2 --eval-source vfold --full-valid
  run q15v_Vfbc_f$f q14t_fbc_s5@ep3 linear --valid-fold $f --n-valid-folds 2 --eval-source vfold --full-valid
  run q15v_Vkd_f$f q14s_local_kd90_s3@ep3 linear --valid-fold $f --n-valid-folds 2 --eval-source vfold --full-valid
done
P="--include-heldout-valid --no-eval-during-fit"
TL="--hidden 256 --layers 2 --dropout 0 --train-source train+valid --include-heldout-valid --train-seqs 10607 \
 --no-eval-during-fit --precision fp32 --checkpoint-policy fixed --checkpoint-epoch 3 --loss-norm chunk \
 --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 --beta2 0.999 --grad-stats --save-training-state \
 --train-eval-seqs 0 --loss wmse --optim tl3-muon --muon-scope rnn --muon-lr-mult 3 --since-weight 32 \
 --since-floor .25 --batch 32 --chunk 1000 --epochs 4 --lr .001 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5 \
 --enc tiedlead --ema-teacher 1 --ema-decay 0.999 --eval-ema --distill tt29rep_fin256_s3,tt29rep_fin256_s4 \
 --distill-alpha 0.5 --distill-boot-h 8 --sync-free --fused-adam --pin-memory"
export WNN_PROJECT_ROOT=$PWD NVIDIA_TF32_OVERRIDE=0 TMPDIR=/home/vlad/development/wunder/research_tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=$R/research/campaign_2026-11-15/tlead_final_r1
for seed in 8 9; do
  test -e $O/STOP10 && exit 0
  n=q15f_tlead_s$seed
  if [ ! -e artifacts/runs/$n/model_ep3.pt ]; then
    echo "FIT_START $n $(date -u +%T)"
    WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=300% WNN_THREADS=3 WNN_MEM_MAX=12000000000 \
    PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src \
    "$R/scripts/limited" "$R/.venv/bin/python" -u "$R/research/campaign_2026-11-14/local_r1/final_launch.py" \
      /home/vlad/development/wunder/research_tmp/pr49_wt $TL --seed $seed --name $n > "$T/$n.fit.log" 2>&1 || echo "FIT_FAIL $n"
  fi
  run q15p_V_s$seed $n@ep3 tiedlead $P
done
