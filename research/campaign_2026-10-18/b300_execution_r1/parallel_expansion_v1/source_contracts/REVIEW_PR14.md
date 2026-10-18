# PR14 causal-innovation successor review

## Finding

The native `a266b036b02e8e012c6eb31072291dad618a169d` implementation supports a clean, same-width, cold-start test of its innovation channels. Use the same `causal_innovations` mode in both arms and vary only `--research-feature-control`: `none` activates innovations; `newest` zeros the final 16 innovation channels. The model receives 80 appended features in both arms (64 projected slow/fast moment statistics plus 16 innovations), for a total input width of `112 raw + 99 event + 80 = 291`. This preserves the architecture, parameter layout, and total parameter count across the paired arms. The control's final 16 input columns are inactive, which is the intended feature ablation, not a separate active-capacity control.

This is a **new R01-recipe successor**, not a faithful execution of historical PR14's proposed 12-epoch WSD teacher/cache experiment. Historical teacher/cache ancestry and its blocked cache-schema attempt must remain a separate record. Do not import teachers, distillation targets, caches, parent weights, or continuation state into this cold labels-only comparison.

## Source path and lineage

The feature path in `src/wnn/research_features.py` computes the current 16-channel projection from `x[t]`, estimates slow/fast moments causally, and standardizes each innovation against the previous slow mean and variance. It does not read labels or future rows. `fit_research_feature_prior` in `src/wnn/training/inputs.py` fits prior moments on raw inputs only, using a deterministic seed-derived sample of at most 32 source-qualified fit groups and every 64th row. For the campaign split path, the trainer passes `train_idx` plus the remaining fit-pool `extra_valid`; `CampaignSplit.assert_fit_groups` rejects the evaluation fold from both pools. Freeze and receipt the fold manifest, actual selected `(split,row_group)` membership and hash, prior arrays/hash, and dataset/source identities for every cell. Candidate/control in the same fold/seed must use the identical prior; never reuse it across folds.

Training checkpoints store the mode/control in `args`, the `n_in` width, normalization buffers, the validated prior and prior provenance. `scripts/predict_run.py` restores the checkpoint's mode/control and prior before rebuilding inputs; `src/wnn/streaming.py` also has an explicit moment state and a first-row reset path. The input normalization is fitted from the fit loader and clamps standard deviations at `0.001`; consequently the masked control innovation columns remain exactly zero after normalization while all other columns follow the usual fit-only normalization. Keep the fixed epoch-5 checkpoint and evaluate it only after fitting.

## Proposed command

Use the exact R01 labels-only recipe, adding only the paired feature settings below. Replace placeholders per frozen fold/seed cell; the two arms share every other argument and source/data identity.

```bash
python scripts/train_rnn.py \
  --name <unique-cell-arm-name> --hidden 512 --layers 1 \
  --valid-fold <fold> --split-manifest <fold-split.json> --device cuda \
  --arch gru --cell gru --enc linear --dropout 0 --train-seqs 10607 \
  --train-source train+valid --valid-train-fraction 1 --valid-repeat 1 \
  --n-valid-folds 4 --full-valid --eval-source vfold \
  --batch 8 --chunk 2000 --epochs 6 \
  --checkpoint-policy fixed --checkpoint-epoch 5 --no-eval-during-fit \
  --seed <seed> --precision fp32 --loss wmse --loss-norm chunk \
  --batch-order seeded --deterministic --sync-free --fused-adam --grad-stats \
  --optim adamw --lr 0.0001 --beta2 0.999 --wd 0.01 \
  --sched onecycle --warmup-frac 0.05 --head-lr-mult 1 --clip 1 \
  --save-training-state \
  --research-features causal_innovations \
  --research-feature-control <none-or-newest>
```

Use the source's strict-FP32 runtime configuration and verify FP32 parameters, activations, optimizer state and evaluation, autocast off, TF32 off and highest float32 matmul precision in receipts. `--precision fp32` is required because the CLI defaults to FP16. For the post-fit export/prediction, select `--epoch 5` and explicitly request `--inference-precision fp32`; check exact `[groups,20000,2]` FP32 predictions and grade with the unchanged official `global_wp_clip2_v1` path. Do not use outer-fold scores during fitting or checkpoint selection.

## Required gates

The exact-source tests already cover synthetic prefix invariance, NumPy/Torch/streaming parity, the newest-mask width and zero suffix, trainer input assembly at width 291, `StreamingGraph` parity/reset for both controls, and an ONNX state round trip. Before admitting the CP30 fits:

- Run `tests/test_causal_moment_features.py` and `tests/test_streaming_graph.py` in a Torch-enabled CPU environment. I extracted those tests and source at the exact commit; this audit environment has no Torch, so pytest collected both files but skipped them.
- Add or run a bounded synthetic 20,000-row case that crosses both 512-row feature-scan and 2,000-row training chunk boundaries. Compare NumPy, batched Torch, chunked/row-streamed features, and the same small GRU's batch versus streaming predictions for both controls; mutate only the future suffix to confirm strict prefix invariance and run A-B-A sequence-reset checks.
- Round-trip an epoch-5 checkpoint for each arm through the actual predictor with FP32 inference. Assert `n_in == 291`, mode/control, prior member/hash, input normalization buffers, model source hash and output shape/dtype all match the fit receipt. Independently recompute the official score from the saved prediction and verify identical evaluation row identities for each paired cell.
- Validate each prior's selected groups against its outer split and ensure no held-out evaluation group appears in the prior or normalization fit membership. Hash the exact deployed source tree and split/data inputs; the PR body alone is not a source binding.

## Comparison and limits

Freeze 30 paired `(fold,seed)` cells per arm, paired on identical fold/seed, fit membership, sample order, update budget, optimizer, checkpoint and runtime. The native control estimates the incremental value of the 16 innovations conditional on the 64 moment channels; it does not test the broader historical teacher/cache recipe or prove transfer to a student/other architecture. All four folds are exposed development folds, not a pristine holdout. Predeclare a useful-effect threshold and report every endpoint, failures, per-fold paired deltas and seed/component uncertainty; do not interpret 30 fits or row count as independent markets.

No scientific fits or GPU operations were run for this review. Static source inspection used commit `a266b036b02e8e012c6eb31072291dad618a169d`. Exact-source feature tests were attempted with the audit Python but skipped because Torch is not installed there.
