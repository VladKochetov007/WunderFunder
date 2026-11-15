#!/usr/bin/env bash
# Valid-regime fine-tune study (H-Q15-VALID-FT). Init train-only q14t_tlead_s5@ep3; PR49 ft recipe
# (gcorr, adamw lr1e-4, 2 epochs, ckpt ep1, batch16, chunk2000, no KD/since/EMA, inherited preprocessing).
# Per chain fold f of valid (n=2): C=train only, T=train+other valid fold, V=other valid fold only; eval fold f.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-15/valid_ft_r1
while systemctl --user is-active -q wnn-q15-queue2; do sleep 30; done
cd /home/vlad/development/wunder/research_tmp/chrono_root
FT="--hidden 256 --layers 2 --dropout 0 --enc tiedlead --init q14t_tlead_s5@ep3 --inherit-init-preprocessing \
 --precision fp32 --checkpoint-policy fixed --checkpoint-epoch 1 --epochs 2 --lr .0001 --optim adamw --sched onecycle \
 --warmup-frac 0.05 --wd 0.01 --clip 1 --beta2 0.999 --loss gcorr --loss-norm chunk --batch 16 --chunk 2000 \
 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5 --grad-stats --train-eval-seqs 0 \
 --eval-source vfold --n-valid-folds 2 --full-valid"
run() { # name seed extra...
  local name=$1; shift
  test -e "$O/STOP" && exit 0
  test -e "artifacts/runs/$name/model_ep1.pt" && return 0
  echo "FIT_START $name $(date -u +%T)"
  WNN_PROJECT_ROOT=$PWD NVIDIA_TF32_OVERRIDE=0 TMPDIR=/home/vlad/development/wunder/research_tmp \
  WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=200% WNN_THREADS=2 WNN_MEM_MAX=12000000000 \
  PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src \
  "$R/scripts/limited" "$R/.venv/bin/python" -u "$R/research/campaign_2026-11-14/local_r1/final_launch.py" \
    /home/vlad/development/wunder/research_tmp/pr49_wt $FT --seed 5 --name $name "$@" > "$O/$name.fit.log" 2>&1
  echo "FIT_DONE $name $(date -u +%T)"
}
for f in 0 1; do
  run q15v_V_f$f --valid-fold $f --train-source valid
done
for f in 0 1; do
  run q15v_T_f$f --valid-fold $f --train-source train+valid --train-seqs 10607
  run q15v_C_f$f --valid-fold $f --train-source train --train-seqs 10607
done
