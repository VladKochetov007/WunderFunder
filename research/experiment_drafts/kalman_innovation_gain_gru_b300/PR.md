# [Experiment] E-KALMAN-INNOVATION-GAIN-GRU-B300-20261018-R1: innovation-conditioned latent gain inside the GRU

## 1. Status, lineage, owner, and idea family

This is a code and experiment-plan draft for the `research-team` branch. It does not authorize a fit and contains no B300 measurement or quality result. It improves the **teacher** recurrent representation and tests the **teacher-to-student link**. Hypothesis: `H-KALMAN-INNOVATION-GAIN-GRU-20261018-R1`; experiment: `E-KALMAN-INNOVATION-GAIN-GRU-B300-20261018-R1`.

The branch starts at `research-team` SHA `104e5461aa862c0d66a44588248385874b800054` after the current-main sync; current committed `main` was `24e9001570de158db9a5031e632dbb0b4477145f`. The model module, canonical trainer adapter, source identity checks, recipe and this PR description are included here. Branch and final source commit will be added after publication. Owner: B300 research team.

## 2. Why it should work and what could explain a gain instead

The model encodes the current input, forms a local-linear prediction of a latent measurement from the previous two encoded rows, computes the innovation, and predicts a bounded diagonal gain. It injects the posterior and gain-weighted innovation into the standard GRU input and readout. The ordinary fused GRU continues to carry context across rows and chunks. Unlike a fixed output smoother or appended feature list, the candidate uses the task loss to learn when a latent prediction should be corrected.

The strongest alternative is that the new residual branch or added capacity helps, while innovation itself contributes nothing. The `context_only` arm has exactly the same layers, parameters, training schedule and observer state, but zeros the innovation only at the gain network input. A same-width ordinary GRU anchors the full-architecture comparison. The two observer residual layers are zero-initialized, so their initial function equals the ordinary GRU.

This is a KalmanNet-inspired latent observer, not a calibrated Bayesian filter: there are no physical state labels, covariance recursion, explicit process/observation likelihood or inference-time weight updates. The literature motivates the operator, not a quality prediction. See the source assessment for [KalmanNet](https://arxiv.org/abs/2107.10043) and [Recursive KalmanNet](https://arxiv.org/abs/2506.11639).

## 3. Competition payoff and scaling

The payoff is a better fixed-size, deterministic GRU submission **after distillation**. A teacher-only gain is not enough. Compare the standard H256×2 student trained from the innovation-teacher cache against a matched context-teacher cache and the existing four-family cache. Report the final student's global WP, CPU runtime, memory and package-size tradeoff separately.

The primary teacher scale is H512×2 with observer width64. The declared scale ladder is H256×2 / width32, H512×2 / width64, and H1024×2 / width128. The primary 30-cell comparison is H512. Profile every new scale before fitting; continue to H1024 when H512 evidence is positive or uncertain with useful expected information value. A lower scale is a measured engineering/quality comparison, not a kill gate for the wider mechanism. Eight B300s are used for up to eight independent cells at once; DDP is not required for the initial recurrent cells. No throughput or wall-time estimate is claimed.

## 4. Frozen teacher comparison

At H512×2, compare three scratch-trained arms on the identical outer-fold/seed cells:

1. `gru`: the standard model with no observer branch.
2. `context_only`: observer architecture whose gain net cannot see innovation.
3. `innovation`: same observer with innovation-conditioned gain.

All use canonical 211-wide inputs, fit-only global normalization, original two-output WMSE, batch16, chunk2000, AdamW (`lr=0.001`, `wd=0.01`), OneCycle over the exact six-epoch budget, global gradient clipping at1.0, and fixed epoch5. FP32 weights, inputs, observer/recurrent state, loss, gradients and optimizer state; autocast, TF32 and quantization are off. Same seed, sample order and source-qualified fold pair the three arms. Train from scratch; cross-architecture initialization is unsupported in this draft.

Primary estimand: equal-fold mean of paired `innovation − context_only` WP. Secondary: `innovation − gru` and `context_only − gru`. Save both target scores, per-fold scores and per-cell predictions. If the adapter's common residual pathway helps both observer arms, the first contrast isolates whether conditioning gain on innovation adds value.

## 5. Thirty paired cells and student comparison

The teacher matrix uses 30 frozen cells: F0/F1 with seeds1001–1008; F2/F3 with seeds1001–1007. Every arm runs all30 cells. No seed split between treatments.

After teachers and clean source bindings are frozen, run three H256×2 student conditions on the same30 cells: (a) the existing four-family cache, (b) the same cache with only `samefold_baseline` replaced by the context-only observer teacher, and (c) the matched cache with only that family replaced by the innovation-conditioned teacher. Preserve the existing four-family order and total source coefficient `0.8125`, and preserve the replaced family's exact original coefficient. Keep every other teacher run, epoch, coefficient, label/cache scale, fold, seed, student input and update budget identical. Use the existing two-epoch standard-GRU student schedule and fixed `model_ep1` convention. The clean per-cell source registry is required before cache building; do not guess run names or teacher ancestry.

This plan is 90 primary teacher fits plus90 student fits. The H256/H1024 scale ladder is additional predeclared follow-up work, not included in the 180-fit count. Profiles and source-cache validation are engineering gates, not scientific cells. **The student half is currently blocked:** `recipe.json` has no per-cell source registry yet. Before any student cache is built or any student cell is admitted, freeze exact run IDs, checkpoints, epochs, coefficients and clean-fold ancestry for all four sources in every fold/seed cell; independently verify those entries against producer manifests. A registry assembled later is a new frozen recipe revision.

## 6. Validation, uncertainty, and significance

Use the unchanged official `global_wp_clip2_v1` scorer, mask and clipping. Compare paired predictions on identical source-qualified evaluation groups. Report equal-fold mean, all four fold effects, all seed effects, component-cluster paired uncertainty and seed uncertainty separately. For seed uncertainty, bootstrap the eight seed identities with shared sampled multiplicities across folds; for evaluation-unit uncertainty, resample source-qualified components within each fold with the same component multiplicities for both treatments and all seeds. Use10,000 draws and freeze RNG seeds in the final receipt. These exposed folds measure robustness to known partitions, not unseen-market or private-leaderboard uncertainty.

Primary family is the teacher contrast `innovation − context_only`. Secondary family is `innovation − gru` plus the three student endpoint contrasts. If making confirmatory claims across all declared contrasts, apply Holm correction across that frozen family. A point gain near `+0.0005 WP` is a provisional follow-up priority reference, not a universal success or kill threshold. Report uncertainty and practical student contribution; an interval crossing zero alone does not close the direction.

## 7. Leakage and representativeness

At scored row `t`, feature construction reads only `x_t` and prior encoded rows `x_(t−1), x_(t−2)`. At a sequence start it emits no prior innovation; the measurement becomes its own prior. The observer carries two latent measurements, a row count and the GRU state across chunks; all states reset between sequences. No target, score mask, sequence ID, future row or validation statistic enters the adapter. Before task fits, run and record a synthetic check for sequence-prefix invariance, reset, batch isolation and full-sequence/chunked equivalence on the frozen code. No such check has run from this branch yet.

Outer fold manifests determine all teacher/student membership. Every learned teacher and initializer must exclude the matching evaluation groups. Existing cache validation binds teacher ancestry, checkpoint bytes, prediction sidecars and row order; this branch adds observer/GRU-core source hashes to teacher provenance and checks them during prediction export and cache construction. The supplied train/valid folds have already been inspected, so report them as exposed.

## 8. Decision-changing results

A useful result is a student endpoint gain that is consistent enough across folds to justify the added teacher cost and maintains the final student serving envelope. A teacher gain with no student transfer is not a submission win, but may justify a bounded change to student hinting or loss treatment. A positive but imprecise or fold-conditional signal can justify a targeted follow-up; it is not a promotion claim. A negative result narrows this local-linear prior/gain/control/schedule at the tested scale. It does not reject every adaptive observer, latent innovation, longer-memory prior or online-learning mechanism.

## 9. B300 profile and run commands

Set `DATA_ROOT`, `GPU`, `FOLD_MANIFEST`, and unique run names from the frozen cell registry. Mount source read-only at this PR commit; data, outputs, caches and temporary files go to physical storage. The profile is the first operation on a B300 and uses the actual batch/chunk/hidden/state width; its finite update timing and memory decide whether the matrix shape is admissible. Do not silently resize or change precision mid-matrix.

Profile (four measured CUDA batches after one warmup, engineering only):

```sh
export WNN_PROJECT_ROOT="$DATA_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/train_rnn.py \
  --name innovation_gain_profile_f0_s1001 \
  --arch innovation-gain --innovation-state-width 64 \
  --innovation-gain-conditioning innovation \
  --split-manifest "$FOLD_MANIFEST" --train-source train+valid \
  --hidden 512 --layers 2 --dropout 0 --batch 16 --chunk 2000 \
  --loss wmse --optim adamw --lr 0.001 --wd 0.01 --sched onecycle --clip 1 \
  --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
  --no-eval-during-fit --precision fp32 --grad-stats --seed 1001 \
  --profile-warmup 1 --profile-steps 4
```

One teacher cell (choose `ARM=gru`, `context_only`, or `innovation`; same exact cell variables across arms):

```sh
WNN_PROJECT_ROOT="$DATA_ROOT" CUDA_VISIBLE_DEVICES="$GPU" \
  .venv/bin/python scripts/train_rnn.py \
  --name "$RUN" --arch "$ARCH" --innovation-state-width 64 \
  --innovation-gain-conditioning "$ARM_MODE" \
  --split-manifest "$FOLD_MANIFEST" --train-source train+valid \
  --hidden 512 --layers 2 --dropout 0 --batch 16 --chunk 2000 \
  --loss wmse --optim adamw --lr 0.001 --wd 0.01 --sched onecycle --clip 1 \
  --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
  --no-eval-during-fit --precision fp32 --grad-stats --deterministic --seed "$SEED"
```

For the ordinary GRU anchor, set `--arch gru` and omit the two `innovation-*` options. Export each frozen teacher's train and valid predictions with `scripts/predict_run.py` at FP32. Build each treatment cache using `scripts/build_distill_cache.py --output-profile all` and the existing `strong`, `full_finetuned`, `raw_wmse`, and `samefold_baseline` sources; only the final run changes between matched context and innovation caches. The cache-builder requires a source coefficient sum of0.8125 and a verified clean-fold registry. Then train the standard students:

```sh
WNN_PROJECT_ROOT="$DATA_ROOT" CUDA_VISIBLE_DEVICES="$GPU" \
  .venv/bin/python scripts/train_rnn.py \
  --name "$STUDENT_RUN" --arch gru --split-manifest "$FOLD_MANIFEST" \
  --train-source train+valid --hidden 256 --layers 2 --batch 16 --chunk 2000 \
  --loss wmse --optim adamw --lr 0.001 --wd 0.01 --sched onecycle --clip 1 \
  --epochs 2 --checkpoint-policy fixed --checkpoint-epoch 1 \
  --no-eval-during-fit --precision fp32 --grad-stats --deterministic \
  --seed "$SEED" --distill-cache-root "$CACHE_ROOT" \
  --distill-cache-variant source --distill-alpha 0.5
```

The above are source-level commands for the standard trainer, not B300-validated commands. The packet does not verify data mounts, runner versions, B300 topology or wall-time. Record all of them in the first profile receipt. Eight B300s are the user's target context, not a measured or presently reserved allocation; confirm actual availability before planning concurrency.

## 10. Progress and artifact contract

For every update retain attempted and successful steps, training loss, LR by parameter group, preclip gradient norm, clip/nonfinite counts, throughput and elapsed phase time. Preserve validation official score at the fixed checkpoint, all target/fold breakdowns, source/split/cache/teacher hashes, hardware/software versions, determinism and precision flags, peak host/device memory, and terminal status. Keep evidence/logs/predictions in one archive and weights plus resume state in a separate archive; SHA256 every file/archive and record approved private storage. Keep failed cells and negative/inconclusive results. Never bundle supplied datasets or credentials.

## 11. Current results, sources and trace

There are no B300 runs, task fits, teacher caches, student grades, submission ZIPs or public scores from this branch. A read-only implementation review found no causal/chunk-state defect by inspection, but confirmed that the student-source registry and synthetic state-invariant check are still missing; see the indexed [draft review](../../unified/reviews/KALMAN_INNOVATION_GAIN_GRU_DRAFT_REVIEW_20261018_R1.md). Adjacent local evidence is intentionally nonconfirmatory: fixed output filters lost; groupwise fast/slow inputs changed sign across two seeds on the same exposed F0 panel; fixed local-trend replay was small and uncertain. The groupwise feature source snapshot remains uncommitted in the current main worktree; its receipts are only motivation and are labeled accordingly in [`evidence/README.md`](evidence/README.md).

The external source assessment is indexed in `research/unified/sources/README.md`; the hypothesis is in `research/hypotheses.jsonl`; the code-only plan is in `research/experiments.jsonl`; the unified research note is linked from `research/unified/ideas/README.md`. Merge of this PR would promote an implementation proposal and evidence links, not a B300 result or a model candidate.
