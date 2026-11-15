#!/usr/bin/env bash
# Base-choice test: TLead from scratch on train + valid fold (1-f) (all-data analogue), then V-ft on fold (1-f); eval fold f.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-15/valid_ft_r1
cd /home/vlad/development/wunder/research_tmp/chrono_root
L="$R/scripts/limited $R/.venv/bin/python -u $R/research/campaign_2026-11-14/local_r1/final_launch.py /home/vlad/development/wunder/research_tmp/pr49_wt"
export WNN_PROJECT_ROOT=$PWD NVIDIA_TF32_OVERRIDE=0 TMPDIR=/home/vlad/development/wunder/research_tmp WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=200% WNN_THREADS=2 WNN_MEM_MAX=12000000000 PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src
TL="--hidden 256 --layers 2 --dropout 0 --train-source train+valid --train-seqs 10607 --precision fp32 --checkpoint-policy fixed \
 --checkpoint-epoch 3 --loss-norm chunk --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 --beta2 0.999 --grad-stats \
 --save-training-state --train-eval-seqs 0 --loss wmse --optim tl3-muon --muon-scope rnn --muon-lr-mult 3 --since-weight 32 \
 --since-floor .25 --batch 32 --chunk 1000 --epochs 4 --lr .001 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5 --enc tiedlead \
 --ema-teacher 1 --ema-decay 0.999 --eval-ema --distill tt29rep_fin256_s3,tt29rep_fin256_s4 --distill-alpha 0.5 --distill-boot-h 8 \
 --sync-free --fused-adam --pin-memory"
FT="--hidden 256 --layers 2 --dropout 0 --inherit-init-preprocessing --precision fp32 --checkpoint-policy fixed --checkpoint-epoch 1 \
 --epochs 2 --lr .0001 --optim adamw --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 --beta2 0.999 --loss gcorr --loss-norm chunk \
 --batch 16 --chunk 2000 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5 --grad-stats --train-eval-seqs 0 --train-source valid --enc tiedlead"
VF="--n-valid-folds 2 --eval-source vfold --full-valid"
for f in 0 1; do
  test -e $O/STOP8 && exit 0
  [ -e artifacts/runs/q15a_tlead_f$f/model_ep3.pt ] || { echo "FIT_START q15a_tlead_f$f $(date -u +%T)"; $L $TL $VF --valid-fold $f --seed 7 --name q15a_tlead_f$f > $O/q15a_tlead_f$f.fit.log 2>&1; }
  [ -e artifacts/runs/q15a_V_f$f/model_ep1.pt ] || { echo "FIT_START q15a_V_f$f $(date -u +%T)"; $L $FT $VF --valid-fold $f --init q15a_tlead_f$f@ep3 --seed 5 --name q15a_V_f$f > $O/q15a_V_f$f.fit.log 2>&1; }
  echo "DONE f$f $(date -u +%T)"
done
