#!/usr/bin/env bash
# Fallback: reproduce PR49 fin256_s3/s4 -> fbc1k_s5 locally (exact recipes, tt29rep_ names), with exports.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack; O=$R/research/campaign_2026-10-29/teacher_track_r1
W=/home/vlad/development/wunder/research_tmp/pr49_wt; SPLIT=$O/chronological_primary.json
while systemctl --user is-active -q wnn-tt29-c0-s3; do sleep 30; done
cd /home/vlad/development/wunder/research_tmp/chrono_root
export NVIDIA_TF32_OVERRIDE=0 WNN_PROJECT_ROOT=$PWD WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=400% WNN_THREADS=4 WNN_MEM_MAX=16000000000 PYTHONPATH=$W/src
fit() { [ -f artifacts/runs/tt29rep_$1/model_ep3.pt ] || $R/scripts/limited $R/.venv/bin/python -u $O/strict_launch.py $W $($R/.venv/bin/python $O/rep_teacher_args.py $1) > $O/tt29rep_$1.fit.log 2>&1; echo "FIT_DONE $1"; }
export_() { [ -f cache/pred_tt29rep_$1_$2.npy ] || $R/scripts/limited $R/.venv/bin/python $W/scripts/predict_run.py tt29rep_$1 $2 --campaign-split-manifest $SPLIT $3 > $O/tt29rep_$1.$2.export.log 2>&1; echo "EXPORT_DONE $1 $2"; }
fit fin256_s3; export_ fin256_s3 train --campaign-full-train-index-feature-only
fit fin256_s4; export_ fin256_s4 train --campaign-full-train-index-feature-only
fit fbc1k_s5; export_ fbc1k_s5 train --campaign-full-train-index-feature-only; export_ fbc1k_s5 valid --campaign-evaluation-only
echo REP_TEACHERS_DONE
