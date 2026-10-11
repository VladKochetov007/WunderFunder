#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/root/wnn
RUN_NAME=vt_f1_h391_dsq_h384_t3_a9_e6_s3_profile_r1
OUT_DIR=/root/wnn/artifacts/runs/vt_f1_h391_dsq_h384_t3_a9_e6_s3_profile_r1
PROFILE_DIR=/root/run/h391_profile_r1
test ! -e "$OUT_DIR"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$PROFILE_DIR/start_utc"
{
  date -u +%Y-%m-%dT%H:%M:%S.%NZ
  free -b
  df -B1 "$PROJECT_ROOT" | tail -1
  nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
  tail -n 4 /root/run/logs/pipeline_swn25_r2.log
} > "$PROFILE_DIR/before.txt" 2>&1
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader -l 2 > "$PROFILE_DIR/gpu_samples.csv" 2>&1 &
GPU_MON=$!
(
  for sample in 1 2 3 4 5; do
    sleep 20
    date -u +%Y-%m-%dT%H:%M:%S.%NZ
    tail -n 4 /root/run/logs/pipeline_swn25_r2.log
  done
) > "$PROFILE_DIR/ood_progress_during.txt" 2>&1 &
OOD_MON=$!
set +e
/usr/bin/time -v /usr/bin/timeout --signal=TERM --kill-after=5s 120s /usr/bin/env PYTHONPATH=/root/wnn/h391_f1_h384_source_r1/src WNN_PROJECT_ROOT=/root/wnn WNN_PORTABLE_DATA_MANIFEST=/root/wnn/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1/portable_inputs_fold1.json CUDA_VISIBLE_DEVICES=1 WNN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/conda/bin/python /root/wnn/h391_f1_h384_source_r1/scripts/train_rnn.py --name vt_f1_h391_dsq_h384_t3_a9_e6_s3_profile_r1 --seed 3 --device cuda --arch gru --cell gru --hidden 384 --layers 2 --dropout 0 --train-seqs 10607 --valid-seqs 600 --train-source train+valid --valid-fold 1 --n-valid-folds 4 --valid-train-fraction 1 --valid-repeat 1 --eval-source vfold --split-manifest /root/wnn/research/campaign_2026-09-29/artifacts/splits/fold1.json --no-eval-during-fit --init vt_f1_muonall_h384_e3_s4@ep2 --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 --batch 16 --chunk 2000 --loss wmse --loss-norm chunk --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle --precision fp16 --distill-cache-root /root/wnn/cache/dc_98411230 --distill-cache-variant homogeneous --distill-alpha 0.9 --label-weight-floor 0 --label-weight-power 1 --distill-loss mse --distill-channel-weights 1,1 --distill-family-weight 0 --propensity c20260929_prop_f1 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 --since-weight 0 --fast --profile-warmup 5 --profile-steps 25 > "$PROFILE_DIR/trainer.out" 2>&1
RC=$?
set -e
kill "$GPU_MON" "$OOD_MON" 2>/dev/null || true
wait "$GPU_MON" 2>/dev/null || true
wait "$OOD_MON" 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$PROFILE_DIR/end_utc"
printf '%s\n' "$RC" > "$PROFILE_DIR/exit_code.tmp"
mv "$PROFILE_DIR/exit_code.tmp" "$PROFILE_DIR/exit_code"
{
  date -u +%Y-%m-%dT%H:%M:%S.%NZ
  free -b
  df -B1 "$PROJECT_ROOT" | tail -1
  nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
  tail -n 6 /root/run/logs/pipeline_swn25_r2.log
} > "$PROFILE_DIR/after.txt" 2>&1
exit "$RC"

