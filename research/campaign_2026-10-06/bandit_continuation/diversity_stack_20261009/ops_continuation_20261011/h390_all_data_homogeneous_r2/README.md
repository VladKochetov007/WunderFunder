# H390 R2 all-data homogeneous strong-teacher students

This packet builds one compact main-target cache and trains the declared seed-3
and seed-4 final students from their matching, independently trained all-data
h256 parents. It is a final-fit population: it makes no heldout claim and does
not produce a clean-fold score. The cache contains only the two homogeneous main
arrays (`train`, `valid`), whose source-qualified memberships are the complete
supplied target populations (10,606 train plus 1,873 valid row groups).

R2 preserves the rejected R1 packet and adds an immutable revision that pins all
three actual teacher provenance receipts before checkpoint or export access.
`SOURCE_MANIFEST_H390.json` binds every Python source file in `child_source_v1`
to the checked R1 archive and parent source manifest, plus the teacher receipt
hashes. `parent_receipts_r1/` holds the seed-specific initialization-parent
receipts; `teacher_receipts_r2/` holds the actual producer receipts pinned by
R2. `PACKET_MANIFEST_R2.json` binds the runnable instructions, archive, all
receipts, and source manifests. The runnable entry point is the R2 child tree.

The cache target is the ordered float32 mean of three epoch-2 H384 strong
teacher predictions, multiplied by 0.8125. Predictions are not clipped. All
three teachers were fit on the frozen clean fold-1 training pools, while their
full-source exports cover every supplied train and valid sequence. Each child
starts from its own seed-matched all-data h256 checkpoint at epoch 3. The parent
provenance records 600 legacy reporting rows overlapping the fit population;
those rows are labeled reporting metadata and are not treated as heldout.

## Cache build

Run from the mounted project with the frozen child source and existing producer
artifacts present. This streams two sequences at a time and keeps a 20 GiB free
space floor. Admission counts the two-array output plus an explicit 16 MiB
safety reserve; the builder does not duplicate the large teacher exports.

```bash
PROJECT_ROOT=/root/wnn
SOURCE_ROOT=/kaggle/working/h390_all_data_homogeneous_r2/child_source_v1
SPLIT_MANIFEST="$PROJECT_ROOT/research/campaign_2026-09-29/artifacts/splits/fold1.json"
CACHE_ROOT="$PROJECT_ROOT/cache/h390_all_data_homogeneous_r2"

PYTHONPATH="$SOURCE_ROOT/src" WNN_PROJECT_ROOT="$PROJECT_ROOT" \
  python "$SOURCE_ROOT/scripts/build_final_distill_cache.py" \
  --data-root "$PROJECT_ROOT" \
  --teacher-cache-root "$PROJECT_ROOT/cache" \
  --checkpoint-root "$PROJECT_ROOT/artifacts/runs" \
  --teacher-split-manifest "$SPLIT_MANIFEST" \
  --output "$CACHE_ROOT" \
  --chunk-sequences 2 \
  --output-budget-gb 2.5 \
  --min-free-gb 20
```

The builder checks the exact teacher checkpoint identities, producer
provenance, clean-fold fit memberships, full train/valid prediction sidecars,
source order, both seed-specific parent receipts/checkpoints and all-data fit
membership before writing. It refuses other cache modes or extra arrays.

## Student recipes

Run each command once in the same environment after the cache builder succeeds.
The run names, seeds, initializers, fit population and checkpoint epoch are
fixed by the trainer contract. Each run records an empty evaluation population
and `heldout_claim: false`.

```bash
PROJECT_ROOT=/root/wnn
SOURCE_ROOT=/kaggle/working/h390_all_data_homogeneous_r2/child_source_v1
CACHE_ROOT="$PROJECT_ROOT/cache/h390_all_data_homogeneous_r2"
export PYTHONPATH="$SOURCE_ROOT/src"
export WNN_PROJECT_ROOT="$PROJECT_ROOT"

python "$SOURCE_ROOT/scripts/train_rnn.py" \
  --name vt_all_h390_ds_homogeneous_s3 --seed 3 \
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
  --name vt_all_h390_ds_homogeneous_s4 --seed 4 \
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
