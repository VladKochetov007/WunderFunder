# H230 execution contract v1

Status: conditional plan only. No H230 cache, profile, fit, prediction export, or
grade has run. The active E521 seed-4 fit/export is still owned by root; this
contract cannot run until its terminal receipt and all source identities are
sealed. The root allocation remains the only budget authority.

The frozen source implementation is the persistent worktree at
`dde32d0f5e39efe65c0497fec06449d38e18b369`. Its raw family target mode is an
intentional H230 recipe property: the main target cache retains its unnormalized
0.8125 teacher sum, and the appended four-family head trains against raw family
means. `--distill-alpha 0.7` blends the clipped label with the main teacher
target; the main teacher target and label are clipped to ±2 by the trainer.
Family loss uses the main label-derived row weights and is averaged over four
families, then multiplied by 0.1. This is distinct from official metric
clipping. The official evaluator remains `global_wp_clip2_v1` and is owned by
the independent grader.

## Sources and identity

The eight sources are fixed in `SOURCE_FAMILIES_DRAFT_V2.json`: two each from
`strong`, `full_finetuned`, `raw_wmse`, and `samefold_baseline`, in that family
order. Their coefficients are 0.15, 0.15, 0.10, 0.10, 0.125, 0.125, 0.03125,
and 0.03125. Their sum stays 0.8125. The samefold baseline family is the own
raw-only GRU384 seed pair `bd26_w384raw_s3_f0` / `bd26_w384raw_s4_f0`, fixed
zero-based epoch 3. This is a heterogeneous source interaction, not a family
head ablation.

The fixed clean fold is
`research/campaign_2026-09-29/artifacts/splits/fold0.json`, SHA-256
`09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`. The
preparation audit records 10,535 train fit groups, 1,405 valid fit groups, and
468 excluded evaluation-valid groups. The cache builder must recheck the exact
current manifest, source/initializer/propensity ancestry, checkpoint epoch and
policy, prediction bytes and sidecars, target-index row order, complete fit
membership, shape `(target_rows, 20000, 2)`, float32 dtype, and finiteness. It
must write only the fold's fit groups to the student cache. No exposed evaluation
group may enter fitting. Recompute the input/split receipt against mounted bytes
before each stage; the fold manifest hash alone does not certify current data.
Current source sidecars report 10,606 complete train prediction rows and 1,873
complete valid prediction rows. The CLI `--train-seqs 10607` is a ceiling; after
fold filtering the student receives all 10,535 allowed train rows plus 1,405
allowed valid rows. Valid cache rows are a subset of the 1,873-row export index:
1,405 fit rows and 468 excluded evaluation rows. Reconcile these counts to fresh
target-index hashes and the sealed fold manifest before cache building.

Seed-4 checkpoint/provenance and its train/valid export hashes remain null until
root's E521 sequence is terminal. The current receipt observed at
`2026-10-07T14:19:38Z` was `running`. Prediction and sidecar hashes for the six
legacy teacher runs were previously checked only for path presence, so they too
must be freshly sealed before cache construction. Seed-3 E521's recorded fixed
epoch-3 checkpoint/provenance and valid prediction hashes are in the H230 source
draft and E521 export result; its train export is still a dependency. Any hash
mismatch closes this recipe and returns to root for a versioned source-manifest
decision.

The persistent worktree's `.venv`, `artifacts`, `cache`, and `datasets` resolve
to the main project paths; its `research` directory is in the worktree. Check
these real paths again before execution. Use the worktree's pinned scripts and
`PYTHONPATH=$WT/src`; do not edit it or launch from the mutable main trainer.
`H230_SOURCE_HASH_BUNDLE_V1.json` binds the inspected source code and currently
known small-file identities. Large parquet, target-index, chain-fold, prediction,
checkpoint and cache hashes that require a fresh runtime preflight remain null
there intentionally.

## Fixed cache and student recipes

The cache ID is `bd26_h230_w384_family_v1`, written once to
`$ROOT/cache/bd26_h230_w384_family_v1`. The builder writes four memmapped arrays
per split: main source, normalized, homogeneous, and eight-column family
targets. Its formula gives 13,372,800,000 bytes for 11,940 fit sequences ×
20,000 rows × 14 float32 channels. This is a planning estimate, not observed
output size. The builder's 30-GiB output budget and 25-GiB free-space floor
require at least 40,216,345,600 bytes free before building; recheck live disk
space including required E521 train exports. The conservative 24.2-GB planning
reservation in the source draft is retained as a separate reservation, not a
replacement for the builder's byte estimate or its live gate. Build once,
serially, with one CPU, 6.5-GB worker memory, `chunk_sequences=2`, zero data
loader workers, and a 3,600-second cap. Preserve `.partial` output and receipt
on failure; do not retry under this ID.

Student seeds 3 and 4 use unique run IDs
`bd26_h230_hetero_s3_f0` / `bd26_h230_hetero_s4_f0`, each initialized from the
matching own E511 C1 checkpoint `c20261003_combo_s{3,4}_f0@ep3`. Both use GRU,
two layers, hidden width 256, dropout 0, two epochs, fixed checkpoint epoch 1
(zero-based, the second epoch), no fit-time evaluation, batch 16, chunk 2,000,
chunk-normalized wMSE, AdamW/OneCycle, learning rate 1e-4, weight decay 0.01,
fp16 and `--fast`. The training source is `train+valid` with fold 0 excluded;
`--train-seqs 10607` takes all allowed rows after the split manifest filters
the fit pool. Explicit `--eval-source vfold` records the excluded fold while
`--no-eval-during-fit` prevents its evaluation loader from being read. Use
propensity `c20260929_prop_f0`, power 2, epsilon 0.03, true-mask epsilon 0.03,
structural and next-change auxiliary coefficients 0.3 each, raw-only main
recipe defaults, and no EMA or additional auxiliary heads. The family head adds
eight outputs after the declared auxiliary heads: 2 main + 19 structural + 2
next-change + 8 family = 31 total. The family slice begins at output 23 and
uses `--distill-family-weight 0.1 --distill-family-target-mode raw`.

The profile is the same fixed student command/shape/cache/init and training
paths, with a distinct no-checkpoint run ID
`bd26_h230_profile_s3_f0`, `--profile-warmup 3 --profile-steps 20`. The pinned
trainer sets `max_batches=23`, reports per-stage GPU timing, loader wait, median
step wall time and sequences/second, then returns before checkpoint or
prediction writing. This command is profile-only; it is not a fit attempt. The
trainer emits medians rather than p90. The external resource receipt must record
full job wall time (including split/cache validation), peak worker RSS, cgroup
headroom and allocator/observed VRAM. Fit-time evaluation remains disabled.

The profile cap is 900 seconds; the cache cap is 3,600 seconds. Each fit is a
serial one-GPU job capped at 5,400 seconds, followed by a separate fixed-epoch-1
valid export capped at 1,800 seconds. The nearest current runtime reference is
E521 seed 3: its four-epoch GRU384 fit took 5,178.9 seconds and its full-valid
export took 80.7 seconds. Those are context only, not a prediction for H230:
H230 has half as many epochs and a narrower recurrent trunk, but adds family
outputs, cache validation, propensity weighting and two auxiliary losses. Do
not fit until the real H230 profile projects both 2-epoch fit-plus-checkpoint
stages within 5,400 seconds each and confirms resource headroom. If profile
startup/cache validation consumes its 900-second cap, preserve the failure and
ask root to review a new plan; do not bypass cache validation.

All training and prediction commands set `WNN_LOADER_WORKERS=0` because the
earlier two-worker probe exceeded its scoped memory cap. `--fast` still enables
the matched pinned-memory/fused-Adam/sync-free/propensity-cache paths. Use one
GPU lock; set the allocator cap to 4,300,000,000 bytes; enforce the 5,000,000,000
observed-VRAM cap, 6.5-GB stage memory reservation, 20-GB global RAM cap,
18-GB global memory-high cap, 6-core global CPU cap, no swap, and the stage wall
caps. Abort immediately on any resource-gate failure. Do not run seeds in
parallel. Prediction export writes to unique memmapped files and its sidecar;
current teacher sidecars report 1,873 validation target rows, so the expected
shape is `(1873, 20000, 2)` float32 (299,680,000-byte payload; current NPY files
add a 128-byte header, for 299,680,128 bytes). Recheck actual shape and free
disk before export.

## Exact command payloads

These are payloads for the normal resource-admission controller, not direct
launch commands. The controller must mint stage receipts and inject the
declared caps. Do not invoke these until every preceding gate is green.

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007
SPLIT=$ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json
CACHE=$ROOT/cache/bd26_h230_w384_family_v1

# Exact source/cache payload. One attempt, no existing output directory.
PYTHONPATH="$WT/src" "$WT/.venv/bin/python" "$WT/scripts/build_distill_cache.py" \
  --data-root "$ROOT" --teacher-cache-root "$ROOT/cache" \
  --checkpoint-root "$ROOT/artifacts/runs" --split-manifest "$SPLIT" \
  --output "$CACHE" \
  --teacher 'strong=c20261003_combo_s3_f0@ep3*0.15' \
  --teacher 'strong=c20261003_combo_s4_f0@ep3*0.15' \
  --teacher 'full_finetuned=c20260929_ft_since_s3_f0@ep0*0.10' \
  --teacher 'full_finetuned=c20260929_ft_since_s4_f0@ep0*0.10' \
  --teacher 'raw_wmse=c20260929_since_s3_f0@ep3*0.125' \
  --teacher 'raw_wmse=c20260929_since_s4_f0@ep3*0.125' \
  --teacher 'samefold_baseline=bd26_w384raw_s3_f0@ep3*0.03125' \
  --teacher 'samefold_baseline=bd26_w384raw_s4_f0@ep3*0.03125' \
  --homogeneous-family strong --expected-coefficient-sum 0.8125 \
  --chunk-sequences 2 --output-budget-gb 30 --min-free-gb 25

# Full-shape profile payload: use seed 3's initialization and separate profile ID.
PYTHONPATH="$WT/src" "$WT/.venv/bin/python" "$WT/scripts/train_rnn.py" \
  --name bd26_h230_profile_s3_f0 --device cuda --arch gru --cell gru \
  --hidden 256 --layers 2 --dropout 0 --seed 3 \
  --split-manifest "$SPLIT" --valid-fold 0 --eval-source vfold \
  --train-source train+valid --train-seqs 10607 \
  --init c20261003_combo_s3_f0@ep3 --epochs 2 --checkpoint-policy fixed \
  --checkpoint-epoch 1 --no-eval-during-fit --batch 16 --chunk 2000 \
  --loss wmse --loss-norm chunk --lr 0.0001 --wd 0.01 --optim adamw \
  --sched onecycle --precision fp16 --propensity c20260929_prop_f0 \
  --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
  --aux-struct 0.3 --aux-next 0.3 --distill-cache-root "$CACHE" \
  --distill-cache-variant source --distill-alpha 0.7 \
  --distill-family-weight 0.1 --distill-family-target-mode raw \
  --distill-boot-h 0 --fast --profile-warmup 3 --profile-steps 20

# Fit payload for seed 3. Seed 4 changes only run name, seed, and matched init.
PYTHONPATH="$WT/src" "$WT/.venv/bin/python" "$WT/scripts/train_rnn.py" \
  --name bd26_h230_hetero_s3_f0 --device cuda --arch gru --cell gru \
  --hidden 256 --layers 2 --dropout 0 --seed 3 \
  --split-manifest "$SPLIT" --valid-fold 0 --eval-source vfold \
  --train-source train+valid --train-seqs 10607 \
  --init c20261003_combo_s3_f0@ep3 --epochs 2 --checkpoint-policy fixed \
  --checkpoint-epoch 1 --no-eval-during-fit --batch 16 --chunk 2000 \
  --loss wmse --loss-norm chunk --lr 0.0001 --wd 0.01 --optim adamw \
  --sched onecycle --precision fp16 --propensity c20260929_prop_f0 \
  --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 \
  --aux-struct 0.3 --aux-next 0.3 --distill-cache-root "$CACHE" \
  --distill-cache-variant source --distill-alpha 0.7 \
  --distill-family-weight 0.1 --distill-family-target-mode raw \
  --distill-boot-h 0 --fast --save-training-state

# Fixed ep1 valid prediction export; repeat serially for seed 4 after its fit.
PYTHONPATH="$WT/src" "$WT/.venv/bin/python" "$WT/scripts/predict_run.py" \
  bd26_h230_hetero_s3_f0 valid --epoch 1
```

For each seed-4 payload, use `--name bd26_h230_hetero_s4_f0`, `--seed 4`, and
`--init c20261003_combo_s4_f0@ep3`; do not change other recipe arguments. For
all train/profile/export payloads, the controller sets `WNN_LOADER_WORKERS=0`
and `WNN_THREADS` to the admitted CPU count. The prediction exporter uses a
memmap/`.partial` file and atomically renames only after all target rows are
written. Its final prediction hash and sidecar hash are mandatory outputs.

## Dependency DAG and stop conditions

`root-owned E521 seed4 fit → fixed ep3 valid export → E521 train export pair sequence
→ fresh source/checkpoint/provenance/sidecar hashes for all 8 teachers → independent
source/recipe PASS → disk/input/resource admission → single cache build → sealed
cache metadata/array/row-order hashes → actual full-shape profile PASS → independent
final-command review + projected per-seed fit/export within caps → seed3 fit → seed3
ep1 checkpoint hash receipt → seed3 valid export → seed4 fit → seed4 ep1 checkpoint
hash receipt → seed4 valid export → independent complete pair grade.`

The baseline seed pair and every teacher sidecar must share the fold0 manifest and
target cache row order. Source/checkpoint/provenance/sidecar hashes, train/valid
parquet identities, target-index identities, chain-fold/split membership and all
new cache bytes are rechecked at stage boundaries. If any is stale, missing,
nonfinite, mismatched, or if a resource/runtime gate fails, stop and preserve the
terminal receipt. No retry, score computation, public forecast, or human submission
is authorized by this contract. The independent grader freezes and evaluates the
complete H230 pair against the frozen E511 C1 pair; the actor supplies artifacts and
does not score or approve them.
