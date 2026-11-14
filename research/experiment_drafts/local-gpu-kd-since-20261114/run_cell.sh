#!/usr/bin/env bash
# One cold train-only GRU cell on the Oct 29 parent recipe (Muon rnn, LR 1e-3, 4 epochs, strict FP32).
# Usage: run_cell.sh NAME SEED [extra train_rnn args...]; later args override the recipe defaults.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack; O=$R/research/campaign_2026-11-14/local_r1
source $R/research/campaign_2026-10-29/teacher_track_r1/recipe_args.sh
name=$1; seed=$2; shift 2
cd /home/vlad/development/wunder/research_tmp/chrono_root
export NVIDIA_TF32_OVERRIDE=0 WNN_PROJECT_ROOT=$PWD WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=${CPUQ:-300%} WNN_THREADS=3 WNN_MEM_MAX=${MEMMAX:-9000000000} PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src
$R/scripts/limited $R/.venv/bin/python -u $R/research/campaign_2026-10-29/teacher_track_r1/strict_launch.py /home/vlad/development/wunder/research_tmp/pr49_wt $BASE $PARENT --seed $seed --name $name "$@" > $O/$name.fit.log 2>&1
echo CELL_DONE $name
