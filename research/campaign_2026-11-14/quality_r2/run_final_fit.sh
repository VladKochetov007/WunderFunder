#!/usr/bin/env bash
# Fixed alpha-.9 final-data recipe, deliberately no held-out evaluation.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-14/quality_r2
source "$R/research/campaign_2026-10-29/teacher_track_r1/recipe_args.sh"
seed=$1
name=q14_final_kd90_s$seed
BASE=${BASE/--split-manifest $SPLIT/}
BASE=${BASE/--eval-source dev --full-valid/}
BASE=${BASE/--train-source train --train-seqs 10325/--train-source train+valid --include-heldout-valid --train-seqs 10607 --no-eval-during-fit}
cd /home/vlad/development/wunder/research_tmp/chrono_root
export WNN_PROJECT_ROOT=$PWD PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src
export NVIDIA_TF32_OVERRIDE=0 WNN_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/home/vlad/development/wunder/research_tmp
test ! -e "artifacts/runs/$name/model_ep3.pt"
"$R/.venv/bin/python" -u "$R/research/campaign_2026-11-14/local_r1/final_launch.py" \
  /home/vlad/development/wunder/research_tmp/pr49_wt $BASE $PARENT \
  --name "$name" --seed "$seed" --distill tt29rep_fbc1k_s5 --distill-alpha .9 \
  > "$O/$name.fit.log" 2>&1
echo "FINAL_FIT_DONE $name"
