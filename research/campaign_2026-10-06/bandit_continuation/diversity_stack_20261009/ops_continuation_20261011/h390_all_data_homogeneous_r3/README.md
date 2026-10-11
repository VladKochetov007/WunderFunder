# H390 R3 all-data homogeneous strong-teacher students

This packet builds one compact main-target cache and trains the declared seed-3
and seed-4 final students from their matching, independently trained all-data
h256 parents. It is a final-fit population: it makes no heldout claim and does
not produce a clean-fold score. The cache contains only the two homogeneous main
arrays (`train`, `valid`), whose source-qualified memberships are the complete
supplied target populations (10,606 train plus 1,873 valid row groups).

R3 preserves the immutable R1/R2 packets and fixes the R2 startup guard while pinning all
three actual teacher provenance receipts before checkpoint or export access.
`SOURCE_MANIFEST_H390.json` binds every Python source file in `child_source_v1`
to the checked R1 archive, parent source manifest, teacher receipt hashes, and
R2 cache-reuse receipt. `PACKET_MANIFEST_R3.json` binds the runnable instructions,
R2 startup failure, historical producer manifests, cache-reuse lineage, archive,
and all source/producer receipts. The R2 cache remains bound to its actual R2
producer hashes; R3 does not relabel or rewrite its metadata. The runnable
entry point is the R3 child tree.

The cache target is the ordered float32 mean of three epoch-2 H384 strong
teacher predictions, multiplied by 0.8125. Predictions are not clipped. All
three teachers were fit on the frozen clean fold-1 training pools, while their
full-source exports cover every supplied train and valid sequence. Each child
starts from its own seed-matched all-data h256 checkpoint at epoch 3. The parent
provenance records 600 legacy reporting rows overlapping the fit population;
those rows are labeled reporting metadata and are not treated as heldout.

## Cache input

R3 consumes the already built, independently validated
`$PROJECT_ROOT/cache/h390_all_data_homogeneous_r2`. Do not rerun the builder or
overwrite this cache: the R3 consumer verifies its original R2 source and packet
hashes, exact metadata digest, two array digests and byte lengths, full 12,479
row membership, empty evaluation membership and `heldout_claim: false`. Cache
generation remains attributable to R2; R3 only fixes the student startup guard
and binds the historical cache as an explicit input.

## Student recipes

Run each command once in the same environment after the cache builder succeeds.
The run names, seeds, initializers, fit population and checkpoint epoch are
fixed by the trainer contract. Each run records an empty evaluation population
and `heldout_claim: false`.

```bash
PROJECT_ROOT=/root/wnn
SOURCE_ROOT=/kaggle/working/h390_all_data_homogeneous_r3/child_source_v1
CACHE_ROOT="$PROJECT_ROOT/cache/h390_all_data_homogeneous_r2"
export PYTHONPATH="$SOURCE_ROOT/src"
export WNN_PROJECT_ROOT="$PROJECT_ROOT"
export WNN_PORTABLE_DATA_MANIFEST="$PROJECT_ROOT/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1/portable_inputs_fold1.json"

python "$SOURCE_ROOT/scripts/train_rnn.py" \
  --name vt_all_h390_ds_homogeneous_s3_r3 --seed 3 \
  --device cuda --arch gru --cell gru --hidden 256 --layers 2 --dropout 0 \
  --train-source train+valid --train-seqs 10607 --valid-seqs 0 --valid-fold -1 \
  --include-heldout-valid --no-eval-during-fit \
  --init vt_all_fin_mu256p0e4_s3@ep3 --inherit-init-preprocessing \
  --final-distill-cache-root "$CACHE_ROOT" --distill-alpha 0.9 \
  --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
  --batch 16 --chunk 2000 --loss wmse --loss-norm chunk \
  --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle --precision fp16 \
  --propensity c20260929_prop_f1 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 --since-weight 32 \
  --fast --save-training-state

python "$SOURCE_ROOT/scripts/train_rnn.py" \
  --name vt_all_h390_ds_homogeneous_s4_r3 --seed 4 \
  --device cuda --arch gru --cell gru --hidden 256 --layers 2 --dropout 0 \
  --train-source train+valid --train-seqs 10607 --valid-seqs 0 --valid-fold -1 \
  --include-heldout-valid --no-eval-during-fit \
  --init vt_all_fin_mu256p0e4_s4@ep3 --inherit-init-preprocessing \
  --final-distill-cache-root "$CACHE_ROOT" --distill-alpha 0.9 \
  --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5 \
  --batch 16 --chunk 2000 --loss wmse --loss-norm chunk \
  --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle --precision fp16 \
  --propensity c20260929_prop_f1 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 --since-weight 32 \
  --fast --save-training-state
```

The pair is intended for a fixed 0.5/0.5 blend only after both artifacts pass
the normal inference and local comparison checks. Cache construction and code
validation alone do not establish a fit result or comparative gain.
