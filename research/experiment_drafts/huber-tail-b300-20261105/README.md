# Curvature-matched Huber versus weighted MSE on B300

This compares a cold label-only GRU trained with weighted MSE to the same GRU
trained with twice the existing weighted Huber objective (delta 0.5). Both use
the same clipped targets and |clipped-target| row weights. For residuals within
the fixed delta, `2 * Huber` equals squared error and has the same gradient;
outside it the loss grows linearly and its derivative is bounded at ±2δ, whereas the squared-error derivative grows with residual magnitude. This only
matches local objective curvature, not optimizer trajectories or actual GPU
compute. The hypothesis is that rare large prediction errors dominate updates;
the alternative is that attenuating these errors discards valuable signal.
Inference is unchanged.

This is a lower-priority continuation of previous Huber comparisons, which
favored WMSE; [team review](../../unified/reviews/IMPORTED_CLAUDE_PLAN_20261019_R1.md)
records that negative evidence. Historical E-071 reported 0.6556 versus 0.6651
for WMSE: one epoch, H128, propensity squared and one seed; that is not this
recipe and does not establish its outcome. Earlier Huber used half the WMSE curvature near
zero, a possible optimizer-scale confound, not a proven explanation for its
outcome. The team also reported a preliminary target-specific blend opportunity
in an older small panel; it does not establish a general gain. PR62 varied global
gradient clipping and late auxiliary schedules, a different intervention.

`huber_curvature` dispatches to `2 * weighted_huber_loss`. Trainer validation
rejects teacher distillation, TD, controlled clocks and nonpositive/nonfinite
delta for this arm: a distillation branch otherwise bypasses this objective.
There is no new model implementation. This is not a TL3-distillation comparison.
The training-source identity includes all WNN modules and uses v2; older v1
resume states must not be represented as exact continuations.

## Receiver command

Status: code prepared, **no local tests, GPU profiles, fits or quality results**.
Run only in an approved receiver checkout and after data/ancestry, math and
real-shape numerical acceptance. The F0 and F2 manifests must be component-clean
and source-qualified, including their fitting-only normalization sets. Before the
full runs, use separate run names suffixed `_profile`, append `--profile-steps 8
--profile-warmup 2` to the same command, and inspect finite loss/gradients, host
RAM and B300 memory. Do not treat those engineering updates as scientific fits.

```bash
cd "$WNN_REPO"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${WNN_PROJECT_ROOT:?}" "${F0_MANIFEST:?}" "${F2_MANIFEST:?}" "${CUDA_VISIBLE_DEVICES:?}"
for CELL in F0 F2; do
  if [ "$CELL" = F0 ]; then MANIFEST="$F0_MANIFEST"; SEED=1001; FOLD=0; else MANIFEST="$F2_MANIFEST"; SEED=1002; FOLD=2; fi
  for LOSS in wmse huber_curvature; do
    NAME="huber_curvature_${CELL}_${LOSS}"
    test ! -e "$WNN_PROJECT_ROOT/artifacts/runs/$NAME"
    python scripts/train_rnn.py --name "$NAME" --device cuda \
      --arch gru --cell gru --enc linear --numeric-map none --raw-only \
      --hidden 256 --layers 2 --batch 32 --chunk 2000 --epochs 6 \
      --loss "$LOSS" --huber-delta 0.5 --precision fp32 \
      --optim adamw --lr 0.001 --wd 0.01 --seed "$SEED" \
      --split-manifest "$MANIFEST" --train-source train+valid --train-seqs 1000000 \
      --valid-fold "$FOLD" --checkpoint-policy fixed --checkpoint-epoch 5 \
      --no-eval-during-fit --grad-stats
  done
done
```

This proposes **two controls + two treatments = four new fits**. Fold and seed
are confounded; no clean existing controls are assumed reusable. Threshold
+0.0005 official WP is an allocation signal, not significance; conditional
component bootstrap does not include training or selection variance. All prior
folds and public feedback are development-exposed. Report all paired outcomes,
per-target scores, component counts, failures and resources, and a future
confirmation plan. A zero-crossing interval alone does not reject the idea.
Predeclare any later teacher/student transfer or blends and count their fits.

## Export and score

For each completed run set `RUN`, `SPLIT_MANIFEST` and the independently verified
`TARGET_SHA256` for the valid-target cache; scoring is full-sequence official
`global_wp_clip2_v1`. The target cache hash alone does not prove its derivation
from the correct parquet. The predictor overwrites matching exports and deletes
`cache/seqsums_<run>.npz`; reserve unique run names and protect existing files.

```bash
: "${RUN:?}" "${SPLIT_MANIFEST:?}" "${TARGET_SHA256:?}"
python scripts/predict_run.py "$RUN" valid --epoch 5 \
  --campaign-split-manifest "$SPLIT_MANIFEST" --campaign-evaluation-only \
  --device cuda --inference-precision fp32
python research/experiment_drafts/huber-tail-b300-20261105/score_evaluation.py \
  --data-root "$WNN_PROJECT_ROOT" --split-manifest "$SPLIT_MANIFEST" \
  --run "$RUN" --epoch 5 --target-cache-sha256 "$TARGET_SHA256" \
  --output "$WNN_PROJECT_ROOT/artifacts/runs/$RUN/evaluation-grade.json"
```

The scorer is reused from [PR58](https://github.com/VladKochetov007/WunderFunder/pull/58)
at commit 7219e86a. A good research score is not a ready ZIP: final candidates
still require full-sequence source/CPU parity, clean resets, offline Python 3.11,
<=20 MB and measured inference within 60 minutes on one CPU.
