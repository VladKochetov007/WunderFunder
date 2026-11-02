# Sequence-level input-channel dropout

Prepared comparison, **not run or receiver-accepted**. Test whether a GRU trained
with occasional missing channels generalizes better than the exact same label-only
GRU without dropout. For each sequence and training batch independently, mask 5%
of raw 112 input channels for the whole sequence; replace them with their
fitting-only normalization means. Event-feature slots remain zero under raw-only.
Loss weights and scoring masks are calculated from the original inputs. Inference
and evaluation use the unmasked model; no augmented labels or future rows enter.
This is a training regularizer, not an assumption that an arbitrary transformation
is label-preserving. Prior augmentation ideas appear in `research/unified/ideas/legacy/muse.txt`;
S38's negative symmetry-based augmentation must not be confused with this control.

Cold GRU256x2, labels only, AdamW .001, wd .01, batch32, chunk2000, six epochs,
fixed last checkpoint, strict FP32. Proposed F0/1001 and F2/1002, two arms =
four new fits; no parents or teacher cache. Proposed cells are not reservations;
verify whole-component split and fit-only normalization. A private generator seeded
from run seed, epoch and batch provides reproducible masks for the same sampler
and sequence order, without consuming model RNG; changing batch/order changes masks.
Neither inference nor the official objective changes. If extra input blocks are
present, only first 112 are masked; recipe explicitly disables those features.

Prospective receiver command; use the provisioned B300 environment, a free GPU,
source-qualified dataset and split hashes; perform numerical/causal/reset and
real-shape memory profiles on the receiver first. Do not touch live jobs.

```bash
cd "$WNN_REPO"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${WNN_PROJECT_ROOT:?}" "${F0_MANIFEST:?}" "${F2_MANIFEST:?}" "${CUDA_VISIBLE_DEVICES:?}"
for CELL in F0 F2; do
  if [ "$CELL" = F0 ]; then MANIFEST="$F0_MANIFEST"; SEED=1001; FOLD=0; else MANIFEST="$F2_MANIFEST"; SEED=1002; FOLD=2; fi
  for ARM in control drop05; do
    if [ "$ARM" = control ]; then PROB=0; else PROB=0.05; fi
    NAME="channel_dropout_${CELL}_${ARM}"
    test ! -e "$WNN_PROJECT_ROOT/artifacts/runs/$NAME"
    python scripts/train_rnn.py --name "$NAME" --device cuda \
      --arch gru --cell gru --enc linear --numeric-map none --raw-only \
      --hidden 256 --layers 2 --batch 32 --chunk 2000 --epochs 6 \
      --loss wmse --precision fp32 --optim adamw --lr 0.001 --wd 0.01 \
      --seed "$SEED" --split-manifest "$MANIFEST" --train-source train+valid --train-seqs 1000000 \
      --valid-fold "$FOLD" --checkpoint-policy fixed \
      --checkpoint-epoch 5 --no-eval-during-fit --grad-stats --sequence-channel-dropout "$PROB"
  done
done
```

Each fit uses complete source-qualified fitting membership. Compare full held-out
sequences under official pooled `global_wp_clip2_v1` with equal scoring rows and
report t0, t1, each paired delta, group coverage and every failure. Threshold
+0.0005 WP is a resource-allocation suggestion, not significance; F0/1001 and
F2/1002 confound fold and seed. Existing validation and leaderboard feedback are
exposed. A separate follow-up is required for uncertainty, deployment and final
submission quality. Extra cloned input memory and B300 runtime are unmeasured;
consider a smaller batch only as a separately frozen matched comparison.

The trainer source identity is upgraded to `wnn.training-source.v2`, hashing all
`src/wnn` Python files. Exact resume across the old v1 source identity is rejected;
start these fresh cells under v2 rather than presenting old runs as resumed.
