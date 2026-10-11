# H391 fold-1 H384 strong-T3 student

This packet freezes one independent fold-1 student using the already-built cleanfold cache `dc_98411230`. Its parent is our own `vt_f1_muonall_h384_e3_s4@ep2`. The cache's `homogeneous` target is the original coefficient mass (`0.8125`) times the mean epoch-2 prediction of the three own strong H384 teachers. With `--distill-alpha 0.9`, training blends that cached target with the observed label as `0.1*y + 0.9*T`.

The source is an unchanged 58-file copy of the worker's original `lr377_source_r1` snapshot, recorded in `SOURCE_MANIFEST_ORIGIN.json` and `SOURCE_MANIFEST_H391_R1.json`. The child trainer/source is at `child_source_v1/`. No trainer or model code was edited for this arm. `actual_producer_receipts_r1/` contains exact worker-fetched producer provenance, cache metadata, split and portable-input manifests. Their byte hashes are checked locally by `verify_h391_packet.py`.

The fit uses 10,533 train groups and 1,416 validation-fit groups (11,949 total). The 457 fold-1 evaluation groups remain excluded. They are repeatedly exposed local-evaluation groups, so the packet makes no fresh-heldout claim. Training uses `--no-eval-during-fit` and a fixed epoch-5 checkpoint; the fit cannot select against the evaluation groups.

This is a width/parent/target interaction. The existing H256 lead uses another initializer; the existing H384 student shares the seed-4 parent but used alpha 0.7/e2. H391 uses the H384 seed-4 parent and the lead's alpha 0.9/e6. Do not describe its result as a pure width-only comparison. Compare its single model and a predeclared equal blend with `vt_f1_dsq_dm_t3_a9_e6` using `global_wp_clip2_v1`, with fold-1 exposure stated.

## Static checks

From the repository root in the packet worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 python research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h391_f1_h384_t3_student_r1/verify_h391_packet.py
PYTHONDONTWRITEBYTECODE=1 pytest -q research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h391_f1_h384_t3_student_r1/tests
```

These checks use only Python's standard library and actual captured producer receipts; they do not import torch or inspect the large prediction arrays.

## Worker setup and 120-second profile

The profile runs on owned Vast instance `55080658`, physical GPU 1 (visible to the process as CUDA device 0 after `CUDA_VISIBLE_DEVICES=1`), only after a fresh OOD phase/headroom check and root's static source/cache/ancestry gate pass. It preserves H390 on physical GPU 0 and records OOD throughput before, during and after profiling. The profile uses a distinct run ID and writes no fit checkpoint. The six-epoch fit remains unadmitted pending a fresh root credit/resource decision and conservative co-resident timing.

Stage the reviewed `child_source_v1/` to the new path `/root/wnn/h391_f1_h384_source_r1` without touching any frozen source tree. Verify the child manifest before running. On the worker:

```bash
export PROJECT_ROOT=/root/wnn
export SOURCE_ROOT=/root/wnn/h391_f1_h384_source_r1
export SPLIT=/root/wnn/research/campaign_2026-09-29/artifacts/splits/fold1.json
export CACHE_ROOT=/root/wnn/cache/dc_98411230
export WNN_PROJECT_ROOT="$PROJECT_ROOT"
export WNN_PORTABLE_DATA_MANIFEST="$PROJECT_ROOT/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1/portable_inputs_fold1.json"
export PYTHONPATH="$SOURCE_ROOT/src"
export CUDA_VISIBLE_DEVICES=1
export WNN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

/usr/bin/timeout --signal=TERM --kill-after=30s 120s /opt/conda/bin/python "$SOURCE_ROOT/scripts/train_rnn.py" \
  --name vt_f1_h391_dsq_h384_t3_a9_e6_s3_profile_r1 --seed 3 \
  --device cuda --arch gru --cell gru --hidden 384 --layers 2 --dropout 0 \
  --train-seqs 10607 --valid-seqs 600 --train-source train+valid \
  --valid-fold 1 --eval-source vfold --split-manifest "$SPLIT" --no-eval-during-fit \
  --init vt_f1_muonall_h384_e3_s4@ep2 \
  --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
  --batch 16 --chunk 2000 --loss wmse --loss-norm chunk \
  --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle --precision fp16 \
  --distill-cache-root "$CACHE_ROOT" --distill-cache-variant homogeneous --distill-alpha 0.9 \
  --propensity c20260929_prop_f1 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
  --aux-struct 0.3 --aux-next 0.3 --since-weight 0 --fast \
  --profile-warmup 5 --profile-steps 25
```

Record actual source/checkpoint/cache preflight, the command argv and environment, start/end time, phase output, GPU memory and co-resident job phases. The profile does not authorize or start a full fit. If the profile cannot complete within 120 seconds or H390/OOD headroom is not safe, stop at the profile result and retain this packet for a later admitted lane.

## Six-epoch fit (not yet admitted)

Only after a separate root fit release, run the same worker environment and exact trainer arguments with run name `vt_f1_h391_dsq_h384_t3_a9_e6_s3`, omit the profile flags, add `--save-training-state`, and bound fitting with a 7,200-second supervisor timeout. Do not reuse the profile run directory as the fit output. The training argv is frozen in `RECIPE_R1.json`.

After a successful terminal fit, export the fixed epoch-5 checkpoint within the separate 1,800-second export bound:

```bash
/usr/bin/timeout --signal=TERM --kill-after=30s 1800s /opt/conda/bin/python "$SOURCE_ROOT/scripts/predict_run.py" \
  vt_f1_h391_dsq_h384_t3_a9_e6_s3 valid --epoch 5
```

The expected output is `cache/pred_vt_f1_h391_dsq_h384_t3_a9_e6_s3_valid.npy` with its usual prediction sidecar. Compare against the frozen lead and equal half blend only through the existing official scorer; preserve the prior exposed-fold label. Do not rebuild or rewrite `dc_98411230`.
