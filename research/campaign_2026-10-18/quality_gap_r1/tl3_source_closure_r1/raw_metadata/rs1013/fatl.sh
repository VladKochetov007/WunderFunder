#!/bin/bash
cd ~/rs1013
D=/home/grinchenkoeg/WunderFunder-creative/research/iterations/2026-10-08-shift/plew19
PY=$HOME/WunderFunder-creative/.venv/bin/python
FA=$HOME/WunderFunder-creative/artifacts/runs/fttall_s5/model_ep3.pt
ENVB="OPENBLAS_NUM_THREADS=2 WUNDER_RESEARCH_ROOT=$HOME/WunderFunder-creative PYTHONPATH=$D:$HOME/WunderFunder-creative/src PLE_KNOTS="
tr() { env CUDA_VISIBLE_DEVICES=$1 $ENVB $5 ARGS_FROM=$FA RUN_NAME=$2 RUN_SEED=$3 OVERRIDES="$4" $PY $D/train_rnn_rs.py --name $2 > logs/$2.log 2>&1; echo "$2 train exit $?" >> logs/queue.log; }
( tr 5 fttall_lowlr_s5 5 "{\"lr\": 0.0002, \"epochs\": 12}" ) &
( tr 5 fttall_tlead_s5 5 "{\"enc\": \"tiedlead\"}" RS_LEAD=1 ) &
( tr 5 fttall_tlead_s6 6 "{\"enc\": \"tiedlead\"}" RS_LEAD=1 ) &
wait; echo FATLDONE >> logs/queue.log
