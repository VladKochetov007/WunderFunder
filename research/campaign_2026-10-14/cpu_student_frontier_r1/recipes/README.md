# CPU student frontier: fitted-checkpoint recipe

This recipe binds the first matched eight-shape cohort to the existing clean F0 homogeneous teacher cache. The separate [consumer contract](CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json) is deliberately distinct from the cache producer metadata: it preserves the producer's historical H256×2/B16/C2000/six-epoch/checkpoint-5/alpha-.9 contract while admitting only this explicit B8, one-epoch, FP32 seed-3 consumer variant. The strict producer validator and full target-payload hash verification remain mandatory before every fit.

The cohort uses the ordered ladder H256×1, H192×2, H256×2, H512×1, H384×2, H640×1, H512×2, H448×3. All eight receive the same 211-feature mean/std copied from the pinned clean F0 H256×2 seed-3 control; only those normalization buffers are loaded. Learned weights start from the normal seeded PyTorch initialization. The model retains the current causal input projection, GELU, GRU, encoded-input skip concatenation and two-output head, with zero initial state per sequence and state carried between C2000 chunks.

Each model gets seed 3, epoch 0, B8 and the same source-qualified sampler order. The loader has 11,940 fitting groups; `drop_last` yields 1,492 batches and 11,936 sequence exposures, omitting the same four tail members for every shape. Ten C2000 chunks per sequence give 14,920 AdamW updates per model. The recipe uses FP32 with TF32 disabled, OneCycleLR over all 14,920 updates, fused AdamW, gradient clipping at 1.0, `wmse` with chunk normalization, and `grad_stats`. AdamW starts with beta1=.9; OneCycle momentum cycling then changes beta1 through .95→.85→.95, so .9 describes initialization only. It does not evaluate or select checkpoints during fitting; the fixed terminal artifact is `model_ep0.pt`.

## Root-admitted train-rnn CLI profile

The first frozen R1 CLI profile failed before updates on producer path-field names and argparse `Path` JSON serialization; that attempt is preserved in the campaign profile records. Successor R2 applies an explicit producer `checkpoint_path`/`provenance_path` mapping and serializes path arguments safely. Its complete CPU preflight passes, including payload hashes and order digest `994cef93396ec6769d84560f5104c09a50e5adb6754fd936b2f5a7ab0389e882`; see [R2 preflight result](../profiles/CPU_PREFLIGHT_R2_RESULT.json). Root then admitted and passed the actual R2 one-batch CUDA CLI profile before admitting the first matched fit. The profile exercises the real trainer parser, strict cache consumer, normalization install, sampler, optimizer and update path. One B8 batch means eight sequences and ten chunk updates; this is a resource/finite-update check only and cannot rank shapes. Root owns the GPU lock, host reservation and admission. Preserve the worktree at the path below while a job is running.

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-frontier-sequence-20261014
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json"

"$MAIN/scripts/bandit_resources.py" --root "$WT" \
  --state "$MAIN/research/campaign_2026-10-06/bandit_continuation/runtime" \
  launch --attempt student-frontier-train-rnn-cli-profile-f0s3-h256l2-b8-c2000-fp32-r2 \
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

The wrapper reserved 6 GB host memory, one CPU, and a 3.5 GB CUDA allocator cap; root confirmed the current aggregate-resource bound before admission. Its 10-minute timeout was for the one-batch profile only. The recorded code identities are `train_rnn.py` SHA256 `952939f15e1dfa84bdee20295a4c64c1fbe4fc174d72e76a6bf574d2fdaeba01`, `student_frontier.py` SHA256 `a747431d9bc2acc25f3f053bdfe7b5cfd64c4a211ecdd75002faa54785062e07`, and R2 consumer-contract SHA256 `9476373fc3c32aea2ae48e9a2c48d211f0f152d13d086fa68efd1e8b1a02e768`. The complete no-fit CPU preflight is indexed at [profiles/README.md](../profiles/README.md).

## Matched fits and required outputs

After the profile passes and the consumer packet is independently reviewed, launch one guarded fit per shape in this order: H256×2 control; H256×1 and H192×2 compact pair; then H512×1, H384×2, H640×1, H512×2 and H448×3. For every invocation, use the following exact fit command, replacing only `RUN`, `HIDDEN`, and `LAYERS` with a unique run name beginning `student_frontier_f0s3_` and its allowlisted shape. Do not pass profile flags. The contract rejects any other recipe or shape. Treat each code block as a worker payload: root's frozen `bandit_sequence.py` plan should launch every fit and prediction as a separately bounded GPU attempt and every score as a bounded CPU attempt, requiring terminal success for the full fit→export→grade triplet before advancing to the next shape. Use the host/GPU reserves established by the actual CLI profile and fresh aggregate-resource checks.

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-frontier-sequence-20261014
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json"
RUN=student_frontier_f0s3_h256_l2_r2
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

The process must finish with the fixed epoch-0 checkpoint, matching `provenance.json`, and 14,920 finite optimizer updates. The machine plan below inserts a post-fit assertion receipt before any prediction. It checks the terminal checkpoint, source-qualified order and exposure, shared norm identity, producer/teacher ancestry, zero skipped/nonfinite updates, and exact optimizer update count.

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
RUN=student_frontier_f0s3_h256_l2_r2

OPS=/home/vlad/development/wunder/worktrees/student-frontier-ops-20261014
WNN_PROJECT_ROOT="$MAIN" PYTHONPATH="$OPS/src" WNN_LOADER_WORKERS=0 \
WNN_GPU_MEMORY_BYTES=3500000000 "$MAIN/.venv/bin/python" \
  "$OPS/scripts/predict_student_frontier_f0_frozen.py" \
  "$RUN" valid --epoch 0 \
  --campaign-split-manifest "$MAIN/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --campaign-full-valid-feature-only --device cuda --inference-precision fp32

WNN_PROJECT_ROOT="$MAIN" "$MAIN/.venv/bin/python" "$MAIN/scripts/score_fold0_panel.py" \
  "$RUN@ep0" \
  --out "$MAIN/research/campaign_2026-10-14/cpu_student_frontier_r1/results/${RUN}_fold0.json"
```

The prediction artifact is `cache/pred_${RUN}_valid.npy` with its metadata sidecar. Keep each checkpoint, training provenance, prediction and score report. Do not use resource or CLI-profile outputs for quality selection, stop any architecture based on the first fold, or replace the fixed checkpoint with a best-validation checkpoint. All eight F0 seed-3 recipes complete before confirmation seeds or F3 alternatives begin.

## Frozen fit-to-grade sequence and inference profile

The [R2 sequence plan](CPU_STUDENT_FRONTIER_F0_SEED3_R2_SEQUENCE.json) has 42 separately guarded steps. It resumes the exact root-admitted H256×2 control attempt, then completes its post-fit receipt, freezes the grader inputs, profiles the production export batch, exports all 1,873 valid groups, and scores the exposed 468-group fold-0 panel before admitting another fit. The next seven arms follow in the declared order; every one receives the same source, seed, FP32 one-epoch budget, normalization identity, and batch membership. The first control run is `student_frontier_f0s3_h256_l2_r2`; every other run is in the same `_r2` namespace. The plan reuses the already-running control's exact attempt ID and worker command, so it cannot start a duplicate fit.

The sequence is pinned at SHA256 `83ea18bebd40a122a1d3bd683ffe132f303f4c6054b6b033a61159ca4741b597`. Its [grader-input manifest](GRADER_INPUTS_F0_R2.json) is pinned at `c64ba6a3149927afe9f09e1ac308db706e5a2c1c878d6c05ba2c73bb99e609e5`, the R2 consumer contract at `9476373fc3c32aea2ae48e9a2c48d211f0f152d13d086fa68efd1e8b1a02e768`, and the [B16 inference-profile contract](PREDICTION_PROFILE_F0_H256L2_B16_R1.json) at `229f1ec04114790e5bb50da26ea55b5de61ace2a98155d013b22fb5091b10091`. The fold-0 split digest is `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`, matching the consumer contract and actual MAIN bytes. The first input-check preflight failed closed on a typo in the ops manifest's split digest; the manifest, profile contract and sequence pins were corrected, and the second bounded check passed. Its [preflight receipt](GRADER_INPUTS_CHECK_F0_R2_PREFLIGHT.json) records the checked scorer/predictor sources, actual split, target byte size, and the target hash reserved for the official scorer to recompute. The input checker and each postgrade gate receive these manifest/contract hashes as command-line pins. The official scorer independently hashes the target archive in its report, and the grade gate checks it against the manifest.

The post-fit assertion is [validate_student_frontier_fit.py](../../../../scripts/validate_student_frontier_fit.py). The first score step runs [check_student_frontier_grader_inputs.py](../../../../scripts/check_student_frontier_grader_inputs.py), using the [grader input manifest](GRADER_INPUTS_F0_R2.json). That manifest pins the actual MAIN scorer/evaluator/metric paths, fold-0 split, target archive SHA and byte size, upstream prediction source, and its R2 operations-worktree copy. The scorer must run from MAIN because it derives its project root from the grader script path.

The all-shape B8 profiles exported one causal sequence per shape and did not exercise production inference batching. Production prediction batches 16 sequences and carries state across five 4,000-row chunks for each 20,000-row sequence, so the separate [H256×2 B16 inference profile contract](PREDICTION_PROFILE_F0_H256L2_B16_R1.json) exercises the copied production predictor on only the first 16 full-valid rows after the H256×2 checkpoint exists. The float32 input tensor alone is about 270 MB on host; the profile measures actual CUDA allocation/reservation before any full export. It writes to a distinct profile-only path and reports finite outputs, allocated/reserved CUDA peaks, time and artifact size. This is a resource check, not a grade. The pinned predictor SHA256 is `1ef54e7e73f4a515b7722c0e0035bef4f48599018df1e5265a2c9e218128d6a1`; the parent `predict_run.py` SHA256 is `2171279b58caa3d3d344f853acecc3825e5cf4c462aeb428f089992ee74ac41d`. Its only added path is the explicit first-batch profile mode; all full exports use the original full-valid command path.

The expected full-valid FP32 prediction contains 1,873×20,000×2 float32 values (about 299,680,000 data bytes, plus a small NPY header). The sequence retains 5 GB free disk after each export and a 5 GB host reservation; root checks actual aggregate memory and observed VRAM before every GPU step. The [grade validator](../../../../scripts/validate_student_frontier_grade.py) checks fixed `@ep0`, all 468 groups, exact split/target pins, official metric, declared comparison and the exposed-development label. These records remain cross-fold/seed development evidence, not a fresh holdout or public forecast.

## Status and limits

The [B8 all-shape resource result](../profiles/PROFILE_B8_LADDER_R1_RESULT.json) measured max CUDA allocated/reserved memory of 1,310,053,376 / 1,461,714,944 bytes, observed VRAM of 1,699,741,696 bytes, and host cgroup peak 3,151,912,960 bytes. Its update-only linear projections range from 826 seconds for H256×1 to 3,544 seconds for H448×3; summed across the eight shapes they are 16,941 seconds (4 h 42 min), excluding data reads, feature construction, full cache revalidation, checkpoints and prediction export. The profile separately projects 1,481 seconds total for the 468 scored groups; the required full-valid export covers 1,873 groups and still needs the production B16 resource profile above. These three-update measurements are resource/rate proxies, not end-to-end duration guarantees. The R2 actual train-rnn CLI profile has passed and the first fixed-epoch H256×2 fit is live under root admission. The strict cache consumer preflight used 1,499,885,568 bytes under the 1.5 GB task cap; the full cache hash pass is not repeated by the small post-fit gate.
