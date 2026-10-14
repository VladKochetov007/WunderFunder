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

The process must finish with the fixed epoch-0 checkpoint, matching `provenance.json`, and 14,920 completed optimizer updates. The post-fit assertion checks exact update and exposure counts, zero AMP skips, zero nonfinite steps among the recorded gradient statistics, and finite floating-point tensors in the terminal model. The trainer does not separately serialize per-step loss finiteness, so the assertion does not make that claim.

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

## Frozen fit-to-grade sequences and per-shape inference profiles

The 42-step [R2 sequence](CPU_STUDENT_FRONTIER_F0_SEED3_R2_SEQUENCE.json) and its single H256×2 production-inference profile were not launched. That draft would have profiled only one architecture before the remaining seven full-valid exports, so it did not establish resource capacity for each distinct recurrent shape. Its prediction launch also reserved 6 GB while its prose said 5 GB. Preserve R2 as a historical draft; use the eight single-shape [R3 sequence plans](CPU_STUDENT_FRONTIER_F0_S3_H256L2_R3_SEQUENCE.json), [H256×1](CPU_STUDENT_FRONTIER_F0_S3_H256L1_R3_SEQUENCE.json), [H192×2](CPU_STUDENT_FRONTIER_F0_S3_H192L2_R3_SEQUENCE.json), [H512×1](CPU_STUDENT_FRONTIER_F0_S3_H512L1_R3_SEQUENCE.json), [H384×2](CPU_STUDENT_FRONTIER_F0_S3_H384L2_R3_SEQUENCE.json), [H640×1](CPU_STUDENT_FRONTIER_F0_S3_H640L1_R3_SEQUENCE.json), [H512×2](CPU_STUDENT_FRONTIER_F0_S3_H512L2_R3_SEQUENCE.json), and [H448×3](CPU_STUDENT_FRONTIER_F0_S3_H448L3_R3_SEQUENCE.json). Each plan ends after its own fit, fixed epoch-0 prediction and grade. A separate root admission is required before starting the next shape; `bandit_sequence.py` cannot roll from one architecture into another without that explicit launch.

Every shape sequence has the same guarded order: fitted checkpoint, post-fit assertion, shared frozen grader-input check, exact-shape production B16 resource profile, resource-profile validator, full-valid export, fold-0 grade and grade validator. The initial H256×2 plan reuses the exact already-admitted fit attempt and worker command; it cannot duplicate that fit. The other seven plans keep the same R2 `_r2` artifact names and B8/C2000, one-epoch FP32 recipe while root controls their separate admissions. The common [R2 consumer contract](CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json) remains pinned at `9476373fc3c32aea2ae48e9a2c48d211f0f152d13d086fa68efd1e8b1a02e768`; the original cache producer metadata remains immutable and the strict payload verifier is still required before every fit.

The shared [eight-shape B16 profile contract](PREDICTION_PROFILE_F0_EIGHT_SHAPES_B16_R2.json) is pinned at SHA256 `ed4ba5be2b74c4487d7f773d9e8ac0b86c334b611654b127a2e91484a67019e2`. For each fitted shape, the copied production predictor reads exactly the first 16 `valid_targets.npz` row groups, uses 20,000 rows per sequence, feature-only input, FP32 inference, B16, and C4000 chunks with causal state carried across chunks and reset per sequence. The output is resource-only and has a distinct name from the quality prediction. Its contract binds the upstream `predict_run.py`, operations-worktree predictor and trainer copies, matching trainer/model source hashes, consumer contract, actual fold-0 split, and grader-input manifest. The [profile validator](../../../../scripts/validate_student_frontier_prediction_profile.py) also checks the terminal fit receipt/checkpoint, exact target row-group order, output sidecar and NPY dimensions/dtype/finiteness, resource-profile log record, controller exit/terminal status, required observers, observed VRAM below 5 GB and zero OOM counter deltas.

The validator receipt for a shape is required as the bandit_resources contract on the subsequent full export. bandit_resources refuses to launch if that matching receipt does not exist and hashes it into the export provenance. Thus neither a profile process exit code alone nor another model's profile can admit a quality export. The immutable R3 static preflight passed for all eight recipes before the control fit completed. Its successor [R4 static preflight receipt](R4_SEQUENCE_STATIC_PREFLIGHT.json) also passed for all eight plans. It pins the source-bound post-fit validator from the R4 worktree while reading the unchanged profile contract from the frozen R3 worktree, and checks the exact fit-receipt path consumed by each shape's profile gate. These are recipe-only checks; they do not claim execution or quality results.

After root admits one shape plan, run that exact plan through the existing sequence runner. For the already-running control, this command resumes its attempt record rather than starting a second fit; later architecture plans are launched separately after root reviews the preceding terminal evidence.

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
OPS=/home/vlad/development/wunder/worktrees/student-frontier-ops-20261014
OPS_R4=/home/vlad/development/wunder/worktrees/student-frontier-ops-r4-20261014
PLAN="$OPS_R4/research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H256L2_R3_SEQUENCE.json"
RECEIPT="$MAIN/research/campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h256_l2_r4_sequence.json"

"$MAIN/.venv/bin/python" "$MAIN/scripts/bandit_sequence.py" \
  --plan "$PLAN" --receipt "$RECEIPT"
```

The fit postcondition in [validate_student_frontier_fit.py](../../../../scripts/validate_student_frontier_fit.py) checks recorded cache ancestry and target semantics, source-qualified group membership, exact propensity and normalization-source pins, common normalization identity, batch-order/exposure and completed-update counts, zero skipped updates, recorded finite-gradient statistics, and finite floating-point tensors in the terminal model. The trainer history does not serialize per-step loss finiteness, so the gate does not claim to check it. The [grader input manifest](GRADER_INPUTS_F0_R2.json) pins the actual MAIN scorer/evaluator/metric paths, split, target SHA/byte size, upstream prediction source and copied predictor. The scorer must run from MAIN because it derives its project root from the grader script path; its report rehashes the target archive. The [grade validator](../../../../scripts/validate_student_frontier_grade.py) checks fixed `@ep0`, all 468 groups, the exact split/target pins, official metric, declared comparison and exposed-development label.

Fit admission uses a 5 GB host reserve, prediction profiling uses 5 GB, and each complete full-valid export reserves 6 GB for the B16 float32 inputs and output staging. Both GPU steps use a 3.5 GB allocator cap and require a fresh root-side observed-VRAM check below 5 GB. The complete output is 1,873×20,000×2 float32 values (299,680,000 data bytes plus the NPY header); the sequence leaves at least 5 GB free disk. These plans run one matched epoch with 14,920 completed optimizer updates and 11,936 of the 11,940 fit groups exposed after identical drop-last omission. This is an eight-shape screen, not a saturation conclusion. Fold 0 is already exposed development evidence, not a fresh holdout or public forecast.

## Status and limits

The [B8 all-shape resource result](../profiles/PROFILE_B8_LADDER_R1_RESULT.json) measured maximum CUDA allocated/reserved memory of 1,310,053,376 / 1,461,714,944 bytes, observed VRAM of 1,699,741,696 bytes, and host cgroup peak 3,151,912,960 bytes. Its update-only linear projections range from 826 seconds for H256×1 to 3,544 seconds for H448×3; the summed 16,941 seconds excludes data reads, feature construction, repeated strict cache verification, checkpoints and prediction exports. These three-update measurements are resource/rate proxies, not end-to-end duration guarantees. The one-batch R2 train-rnn CLI profile passed; the first H256×2 fixed-epoch fit was admitted separately by root. No R3 per-shape prediction profile or fold-0 grade is claimed by the static preflight. The strict cache consumer preflight used 1,499,885,568 bytes under its 1.5 GB task cap; the post-fit assertion does not repeat the full cache payload hash pass.

[Root control admission R2](ROOT_CONTROL_FIT_ADMISSION_R2.json) and [first running observation](CONTROL_FIT_PROGRESS_R2_01.json) record the actual single-shape H256×2 launch after the real CLI profile passed. Run `student_frontier_f0s3_h256_l2_r2` uses source `f0b5b6eb`,14,920 planned updates and5GB host reservation. That running snapshot is historical; [terminal fit and grade](../results/README.md) now complete, and the remaining seven fits have root admission below.

- [Root postfit wrapper refusal](ROOT_POSTFIT_OUTER_WRAPPER_FAILURE_R3.json): the launch CLI was initially outside the required campaign slice and refused before worker execution. The identical worker was then admitted through `scripts/limited`; its distinct validator-field failure is preserved separately. No fitting or source change occurred.

- Control export admissions: [B16 profile](ROOT_CONTROL_PREDICTION_PROFILE_ADMISSION_R2.json), [bounded file-cache handoff](ROOT_CONTROL_EXPORT_CACHE_HANDOFF_R1.json), and [full export](ROOT_CONTROL_FULL_EXPORT_ADMISSION_R3.json) record successful fit/profile dependencies, physical-disk checks and unchanged resource caps. The initial memory-bound precheck required exact decimal byte counts; page-rounded cgroup values were diagnosed before any mutation.
- [Remaining seven matched fits admission](ROOT_REMAINING_F0_SEED3_ADMISSION_R4.json), with [post-export file-cache handoff](ROOT_NEXT_FIT_CACHE_HANDOFF_R1.json), releases the rest of the user-authorized fixed F0 seed3 screen under individual terminal/resource gates. H256×1 starts first; no automatic retry or quality promotion.
The H256×2 control fit completed in 2,037.7 training seconds with 11,936 fitting-sequence exposures, 14,920 attempted and completed FP32 updates, zero AMP skips, and zero nonfinite recorded gradient steps. The first post-fit assertion attempt, student-frontier-f0s3-verify-h256l2-r3, failed on a validator-only assumption that beta1 appeared as a serialized CLI argument; the original trainer has no beta1 CLI option. The exact failed receipt and worker log are preserved and explained in results/README.md. The corrected R4 validator keeps the fit and consumer contract immutable, requires constructor beta1=.9 under the already-pinned training_dynamics.py, checks the exact recorded PyTorch version and OneCycle defaults, and reconstructs the resulting beta1 path [.95,.85,.95] over 14,920 steps. It passed on the frozen checkpoint and emitted the profile-consumed fit receipt, SHA256 1042e31a73456edd88f40a8bd03b11c91ff346907ace16c50abc10b58fecdf43. This reconstructs schedule semantics from pinned code/runtime; it does not claim a per-step optimizer trace was serialized. The validator also does not claim per-step loss finiteness, which trainer history does not record.
