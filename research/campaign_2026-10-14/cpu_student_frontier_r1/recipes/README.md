# CPU student frontier: fitted-checkpoint recipe

This recipe binds the first matched eight-shape cohort to the existing clean F0 homogeneous teacher cache. The separate [consumer contract](CPU_STUDENT_FRONTIER_F0_SEED3_B8_R1_CONSUMER_CONTRACT.json) is deliberately distinct from the cache producer metadata: it preserves the producer's historical H256×2/B16/C2000/six-epoch/checkpoint-5/alpha-.9 contract while admitting only this explicit B8, one-epoch, FP32 seed-3 consumer variant. The strict producer validator and full target-payload hash verification remain mandatory before every fit.

The cohort uses the ordered ladder H256×1, H192×2, H256×2, H512×1, H384×2, H640×1, H512×2, H448×3. All eight receive the same 211-feature mean/std copied from the pinned clean F0 H256×2 seed-3 control; only those normalization buffers are loaded. Learned weights start from the normal seeded PyTorch initialization. The model retains the current causal input projection, GELU, GRU, encoded-input skip concatenation and two-output head, with zero initial state per sequence and state carried between C2000 chunks.

Each model gets seed 3, epoch 0, B8 and the same source-qualified sampler order. The loader has 11,940 fitting groups; `drop_last` yields 1,492 batches and 11,936 sequence exposures, omitting the same four tail members for every shape. Ten C2000 chunks per sequence give 14,920 AdamW updates per model. The recipe uses FP32 with TF32 disabled, OneCycleLR over all 14,920 updates, fused AdamW, gradient clipping at 1.0, `wmse` with chunk normalization, and `grad_stats`. It does not evaluate or select checkpoints during fitting; the fixed terminal artifact is `model_ep0.pt`.

## Root-admitted train-rnn CLI profile

Run this one-batch CLI profile before the first fit. It exercises the real trainer parser, strict cache consumer, normalization install, sampler, optimizer and update path. One B8 batch means eight sequences and ten chunk updates; it is a resource/finite-update check only and cannot rank shapes. Root owns the GPU lock, host reservation and admission. Preserve the worktree at the path below while a job is running.

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-frontier-fit-20261014
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R1_CONSUMER_CONTRACT.json"

"$MAIN/scripts/bandit_resources.py" --root "$WT" \
  --state "$MAIN/research/campaign_2026-10-06/bandit_continuation/runtime" \
  launch --attempt student-frontier-train-rnn-cli-profile-f0s3-h256l2-b8-c2000-fp32-r1 \
  --kind gpu --cpus 1 --reserve-bytes 6000000000 \
  --profile 'actual train_rnn CLI: H256x2, one B8 batch/ten C2000 FP32 updates, strict cache consumer and common norm; no quality claim' \
  --contract "$CONTRACT" --python "$MAIN/.venv/bin/python" -- \
  timeout --signal=TERM --kill-after=10s 10m \
  /usr/bin/env WNN_PROJECT_ROOT="$MAIN" PYTHONPATH="$WT/src" WNN_LOADER_WORKERS=0 \
  WNN_GPU_MEMORY_BYTES=3500000000 \
  "$MAIN/.venv/bin/python" "$WT/scripts/train_rnn.py" \
  --name student_frontier_cli_profile_f0s3_h256_l2 \
  --device cuda --arch gru --cell gru --enc linear --hidden 256 --layers 2 --dropout 0 \
  --train-seqs 10607 --train-source train+valid --valid-train-fraction 1 \
  --valid-repeat 1 --valid-fold 0 --n-valid-folds 4 --full-valid \
  --batch 8 --chunk 2000 --epochs 1 --checkpoint-policy fixed --checkpoint-epoch 0 \
  --no-eval-during-fit \
  --split-manifest "$MAIN/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --student-frontier-contract "$CONTRACT" --student-frontier-stage f0_seed3_cli_profile \
  --seed 3 --precision fp32 \
  --distill-cache-root cache/clean_t3_f0_h390_t3r5_a9_r2 \
  --distill-cache-variant homogeneous --distill-cache-scale 1 --distill-alpha 0.9 \
  --loss wmse --loss-norm chunk \
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
  --batch-order seeded --deterministic --fast --grad-stats \
  --optim adamw --lr 0.0001 --beta2 0.999 --wd 0.01 --sched onecycle \
  --warmup-frac 0.05 --head-lr-mult 1 --clip 1 \
  --profile-steps 1 --profile-warmup 0
```

The wrapper reserves 6 GB host memory, one CPU, and a 3.5 GB CUDA allocator cap, subject to root's fresh aggregate-resource check before admission. Its 10-minute timeout is for the one-batch profile only. The recorded code identities are `train_rnn.py` SHA256 `3ef9f2faae8eab69f3e31bf84a7411e5eda0aaab1110066a0264a26f79294d2d`, `student_frontier.py` SHA256 `6920f4a4fcdc81a3f935016d544493445a79f32734a5d966106e23dbbd75062c`, and consumer-contract SHA256 `1de1f9dfb4d5f1c3bcfde8a9ce2916620b6a54ab2def82f234bd85991283eae2`.

## Matched fits and required outputs

After the profile passes and the consumer packet is independently reviewed, launch one guarded fit per shape in this order: H256×2 control; H256×1 and H192×2 compact pair; then H512×1, H384×2, H640×1, H512×2 and H448×3. For every invocation, use the following exact fit command, replacing only `RUN`, `HIDDEN`, and `LAYERS` with a unique run name beginning `student_frontier_f0s3_` and its allowlisted shape. Do not pass profile flags. The contract rejects any other recipe or shape. Treat each code block as a worker payload: root's frozen `bandit_sequence.py` plan should launch every fit and prediction as a separately bounded GPU attempt and every score as a bounded CPU attempt, requiring terminal success for the full fit→export→grade triplet before advancing to the next shape. Use the host/GPU reserves established by the actual CLI profile and fresh aggregate-resource checks.

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-frontier-fit-20261014
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R1_CONSUMER_CONTRACT.json"
RUN=student_frontier_f0s3_h256_l2
HIDDEN=256
LAYERS=2

WNN_PROJECT_ROOT="$MAIN" PYTHONPATH="$WT/src" WNN_LOADER_WORKERS=0 \
WNN_GPU_MEMORY_BYTES=3500000000 \
"$MAIN/.venv/bin/python" "$WT/scripts/train_rnn.py" \
  --name "$RUN" --device cuda --arch gru --cell gru --enc linear \
  --hidden "$HIDDEN" --layers "$LAYERS" --dropout 0 \
  --train-seqs 10607 --train-source train+valid --valid-train-fraction 1 \
  --valid-repeat 1 --valid-fold 0 --n-valid-folds 4 --full-valid \
  --batch 8 --chunk 2000 --epochs 1 --checkpoint-policy fixed --checkpoint-epoch 0 \
  --no-eval-during-fit \
  --split-manifest "$MAIN/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --student-frontier-contract "$CONTRACT" --student-frontier-stage f0_seed3_fit \
  --seed 3 --precision fp32 \
  --distill-cache-root cache/clean_t3_f0_h390_t3r5_a9_r2 \
  --distill-cache-variant homogeneous --distill-cache-scale 1 --distill-alpha 0.9 \
  --loss wmse --loss-norm chunk \
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
  --batch-order seeded --deterministic --fast --grad-stats \
  --optim adamw --lr 0.0001 --beta2 0.999 --wd 0.01 --sched onecycle \
  --warmup-frac 0.05 --head-lr-mult 1 --clip 1
```

The process must finish with the fixed epoch-0 checkpoint, matching `provenance.json`, and 14,920 finite optimizer updates. Before the next shape is fit, export the full valid source in FP32 with feature-only reads, then run the frozen fold-0 grader. The grader selects only the 468 source-qualified evaluation groups from that full-valid artifact; this repeatedly inspected development panel is exposed evidence, not a pristine holdout or public forecast.

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
RUN=student_frontier_f0s3_h256_l2

WNN_PROJECT_ROOT="$MAIN" "$MAIN/.venv/bin/python" "$MAIN/scripts/predict_run.py" \
  "$RUN" valid --epoch 0 \
  --campaign-split-manifest "$MAIN/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --campaign-full-valid-feature-only --device cuda --inference-precision fp32

WNN_PROJECT_ROOT="$MAIN" "$MAIN/.venv/bin/python" "$MAIN/scripts/score_fold0_panel.py" \
  "$RUN@ep0" \
  --out "$MAIN/research/campaign_2026-10-14/cpu_student_frontier_r1/results/${RUN}_fold0.json"
```

The prediction artifact is `cache/pred_${RUN}_valid.npy` with its metadata sidecar. Keep each checkpoint, training provenance, prediction and score report. Do not use resource or CLI-profile outputs for quality selection, stop any architecture based on the first fold, or replace the fixed checkpoint with a best-validation checkpoint. All eight F0 seed-3 recipes complete before confirmation seeds or F3 alternatives begin.

## Status and limits

The [B8 all-shape resource result](../profiles/PROFILE_B8_LADDER_R1_RESULT.json) measured max CUDA allocated/reserved memory of 1,310,053,376 / 1,461,714,944 bytes, observed VRAM of 1,699,741,696 bytes, and host cgroup peak 3,151,912,960 bytes. Its update-only linear projections range from 826 seconds for H256×1 to 3,544 seconds for H448×3; summed across the eight shapes they are 16,941 seconds (4 h 42 min), excluding data reads, feature construction, full cache revalidation, checkpoints and prediction export. The profile separately projects 1,481 seconds total for the 468 scored groups; the required full-valid export covers 1,873 groups and has not been measured. These three-update measurements are resource/rate proxies, not end-to-end duration guarantees. The actual train-rnn CLI profile and matched fits remain root-admitted execution steps. The strict cache consumer preflight used 1,499,885,568 bytes under the 1.5 GB task cap, so this actor did not repeat its full cache hash pass through the CLI under that cap.
