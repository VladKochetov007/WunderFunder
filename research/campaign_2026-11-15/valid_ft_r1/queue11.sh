#!/usr/bin/env bash
# After queue10: all-data strict KD90 seed 4, then V-ft (diverse linear member for the 3-member all-data V package).
while systemctl --user is-active -q wnn-q15-queue10; do sleep 30; done
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-15/valid_ft_r1
S=/home/vlad/development/wunder/research_tmp/q14_strict_source
test "$(git -C "$S" rev-parse HEAD)" = 48776fba8af26b16b7e6813def7ccf571f56c1ac
source "$R/research/campaign_2026-10-29/teacher_track_r1/recipe_args.sh"
BASE=$(echo "$BASE" | sed "s# --split-manifest [^ ]*##; s# --eval-source dev --full-valid##; s#--train-source train --train-seqs 10325#--train-source train+valid --include-heldout-valid --train-seqs 10607 --no-eval-during-fit#")
cd /home/vlad/development/wunder/research_tmp/chrono_root
name=q15f_kd90_s4
test -e $R/research/campaign_2026-11-15/valid_ft_r1/STOP11 && exit 0
if [ ! -e "artifacts/runs/$name/model_ep3.pt" ]; then
  echo "FIT_START $name $(date -u +%T)"
  WNN_PROJECT_ROOT=$PWD NVIDIA_TF32_OVERRIDE=0 TMPDIR=/home/vlad/development/wunder/research_tmp \
  WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=300% WNN_THREADS=3 WNN_MEM_MAX=12000000000 PYTHONPATH=$S/src \
  "$R/scripts/limited" "$R/.venv/bin/python" -u "$R/research/campaign_2026-11-14/local_r1/final_launch.py" \
    "$S" $BASE $PARENT --name $name --seed 4 --distill tt29rep_fbc1k_s5 --distill-alpha .9 > "$O/$name.fit.log" 2>&1 || echo FIT_FAIL $name
  echo "FIT_DONE $name $(date -u +%T)"
fi
COMMON="--hidden 256 --layers 2 --dropout 0 --inherit-init-preprocessing --precision fp32 --checkpoint-policy fixed \
 --checkpoint-epoch 1 --epochs 2 --lr .0001 --optim adamw --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 \
 --beta2 0.999 --loss gcorr --loss-norm chunk --batch 16 --chunk 2000 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5 \
 --grad-stats --train-eval-seqs 0 --train-source valid"
run() { # name init enc extra...
  local name=$1 init=$2 enc=$3; shift 3
  test -e "$O/STOP11" && exit 0
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
P="--include-heldout-valid --no-eval-during-fit"
run q15p_V_fkd4 q15f_kd90_s4@ep3 linear $P
