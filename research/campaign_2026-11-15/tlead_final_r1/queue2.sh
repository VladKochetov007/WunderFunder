#!/usr/bin/env bash
# Successor: after queue.sh finishes s7 (STOP set to end it), fit an all-data strict KD90 GRU student
# (best measured 4th TL3 member twin, +.000165) then resume TLead seeds 8/9.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-15/tlead_final_r1
S=/home/vlad/development/wunder/research_tmp/q14_strict_source
while systemctl --user is-active -q wnn-q15-tlead; do sleep 30; done
rm -f "$O/STOP"
test -e "$O/STOP2" && exit 0
test "$(git -C "$S" rev-parse HEAD)" = 48776fba8af26b16b7e6813def7ccf571f56c1ac
source "$R/research/campaign_2026-10-29/teacher_track_r1/recipe_args.sh"
BASE=$(echo "$BASE" | sed "s# --split-manifest [^ ]*##; s# --eval-source dev --full-valid##; s#--train-source train --train-seqs 10325#--train-source train+valid --include-heldout-valid --train-seqs 10607 --no-eval-during-fit#")
cd /home/vlad/development/wunder/research_tmp/chrono_root
name=q15f_kd90_s3
if [ ! -e "artifacts/runs/$name/model_ep3.pt" ]; then
  echo "FIT_START $name $(date -u +%T)"
  WNN_PROJECT_ROOT=$PWD NVIDIA_TF32_OVERRIDE=0 TMPDIR=/home/vlad/development/wunder/research_tmp \
  WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=300% WNN_THREADS=3 WNN_MEM_MAX=12000000000 PYTHONPATH=$S/src \
  "$R/scripts/limited" "$R/.venv/bin/python" -u "$R/research/campaign_2026-11-14/local_r1/final_launch.py" \
    "$S" $BASE $PARENT --name $name --seed 3 --distill tt29rep_fbc1k_s5 --distill-alpha .9 > "$O/$name.fit.log" 2>&1
  echo "FIT_DONE $name $(date -u +%T)"
fi
test -e "$O/STOP2" && exit 0
SEEDS="8 9" bash "$O/queue.sh"
