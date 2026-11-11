#!/usr/bin/env bash
# Train-only child for one seed. Usage: run_child.sh SEED ARM [extra train_rnn args...]
# ARM c0 = labels only (matched control); d* arms pass the teacher cache/alpha as extra args.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack; O=$R/research/campaign_2026-10-29/teacher_track_r1
source $O/recipe_args.sh
seed=$1; arm=$2; shift 2
cd /home/vlad/development/wunder/research_tmp/chrono_root
export NVIDIA_TF32_OVERRIDE=0 WNN_PROJECT_ROOT=$PWD WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=400% WNN_THREADS=4 WNN_MEM_MAX=16000000000 PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src
$R/scripts/limited $R/.venv/bin/python -u $O/strict_launch.py /home/vlad/development/wunder/research_tmp/pr49_wt $BASE $CHILD --init tt29_parent_s$seed@ep3 --seed $seed --name tt29_child_${arm}_s$seed "$@" > $O/tt29_child_${arm}_s$seed.fit.log 2>&1
echo CHILD_DONE $arm $seed
