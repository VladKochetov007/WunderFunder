#!/bin/bash
# Matched w19-recipe test: PLE vs control, seeds 3 and 4, GPU 4. Recipe copied from w19distilled_s3 checkpoint args.
cd /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19
PY=$HOME/WunderFunder-creative/.venv/bin/python
SRC=$HOME/WunderFunder-creative/artifacts/runs/w19distilled_s3/model_ep1.pt
run() { CUDA_VISIBLE_DEVICES=4 OPENBLAS_NUM_THREADS=2 WUNDER_RESEARCH_ROOT=$HOME/WunderFunder-creative PYTHONPATH=/home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19:$HOME/WunderFunder-creative/src ARGS_FROM=$SRC RUN_NAME=$1 RUN_SEED=$2 PLE_KNOTS=$3 $PY train_rnn_plew19.py --name $1 > logs/$1.log 2>&1; echo "$1 exit $?" >> logs/queue.log; }
mkdir -p logs
for s in 3 4; do
  run plew19_ctl_s$s $s "" &
  run plew19_ple_s$s $s /home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19/knots_w19.npz &
  wait
done
for r in plew19_ctl_s3 plew19_ple_s3 plew19_ctl_s4 plew19_ple_s4; do
  CUDA_VISIBLE_DEVICES=4 $PY predict_fold0.py $r >> logs/predict.log 2>&1; echo "predict $r exit $?" >> logs/queue.log
done
echo ALLDONE >> logs/queue.log
