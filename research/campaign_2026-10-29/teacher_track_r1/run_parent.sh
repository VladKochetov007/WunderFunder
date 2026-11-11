#!/usr/bin/env bash
# Train-only parent P for one seed (PR49 trainer, strict FP32). Usage: run_parent.sh SEED
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack; O=$R/research/campaign_2026-10-29/teacher_track_r1
source $O/recipe_args.sh
cd /home/vlad/development/wunder/research_tmp/chrono_root
export NVIDIA_TF32_OVERRIDE=0 WNN_PROJECT_ROOT=$PWD WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=400% WNN_THREADS=4 WNN_MEM_MAX=16000000000 PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src
$R/scripts/limited $R/.venv/bin/python -u $O/strict_launch.py /home/vlad/development/wunder/research_tmp/pr49_wt $BASE $PARENT --seed $1 --name tt29_parent_s$1 > $O/tt29_parent_s$1.fit.log 2>&1
echo PARENT_DONE $1
