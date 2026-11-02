# Three-teacher disagreement-weighted KD

[Draft PR #57](https://github.com/VladKochetov007/WunderFunder/pull/57) targets `research-team`.

**Implementation prepared; no training, local tests, model inference, receiver
acceptance or B300 profiles run.** Compare fixed teacher mean KD with bounded
up/down-weighting by teacher disagreement. This is a raw112 sequence experiment,
not the event211 TL3 trainer and not a reproduction of its 0.692 score.

## What is implemented

`sequence_disagreement.py` reads exactly three verified, distinct, single-checkpoint
teacher caches. Existing mean-cache producer validation requires identical
architecture, recipe, split, fitting membership, data, source and normalizers,
with distinct teacher seeds/checkpoint hashes. It does not support heterogeneous
W18/tiedlead or other TL3 caches. Inputs are read by source and row-group ID;
`TeacherCache` checks manifest, producer ancestry and each 20000-row FP32 array.
The three FP32 predictions are averaged without pre-clipping, exactly as the
existing mean merger. Each input is hashed before and after reading and rechecked
at epoch boundaries. Each recorded identity contains the verified composition,
parent producer trees, input manifest digests and descriptor digest.

[PR #56](https://github.com/VladKochetov007/WunderFunder/pull/56) independently
modifies shared sequence training and uses the unmodified PR47 source-only split.
These drafts are not a combined recipe: their caches and source identities are
not interchangeable. Integrating both requires a reviewed reconciliation of the
split identity, loss/telemetry paths, provenance and receiver acceptance; do not
weaken full-Parquet SHA validation or silently reuse profiles.

For each row, variance is the population mean of squared deviations across the
three teachers and two targets. `q = variance/(variance + 0.01)` uses fixed
constants and no target values or future rows. Control `constant` uses row weight
1; `up` uses `1+0.5*q`; `down` uses `1-0.5*q`. Weights are bounded [0.5,1.5].
For each accumulation window the weighted KD numerator is divided by its own
global across-rank target-weighted mass, so its average scale is normalized for
the same window. All-padding/zero-mass windows contribute zero; label and hint
reductions retain their original denominators. Clipping remains after teacher
averaging. No new inference inputs or CPU model are added.

The optional recipe key is omitted from legacy configs. Those continue to use
the existing cache and objective branch. A `sequence-disagreement-input-v1`
descriptor is mandatory for the new mode. The source-only split producer/loader
is adapted from [PR47](https://github.com/VladKochetov007/WunderFunder/pull/47),
head `f0d8da40fb545a630e5f042bfcc3b3360784177e`; the experiment explicitly
rejects the historical mixed-source fit. It excludes the official valid source
from fitting, but approximate feature components do not prove chronology or
remove development exposure. This branch strengthens PR47's parquet footer/mtime
fingerprint by hashing each complete parquet: teacher provenance validation
requires the full SHA256. This changes input receipts and split identities; old
PR47 source-holdout caches cannot be reused without regeneration. Full-parquet
hashing adds substantial receiver I/O; measure it, and use a new isolated data
root/cache rather than modifying an active training checkout's derived artifacts.

## Panel and admission

The six student configs in `configs/` use strict FP32 GRU256x2, 5 fixed epochs,
one B300 each, chunk512, accumulate8, AdamW 0.001 with 5% warmup, matching
all-source train-only fit members and identical teacher descriptor. Seeds
1001/1002 × modes constant/up/down = 6 new student fits. Exact source-qualified
teachers may be reused only if their metadata passes the runtime; none are
currently asserted available. Otherwise the three teacher configs (`teacher_s2001`,
`s2002`, `s2003`) are a **separate prerequisite allocation of 3 fits**, for
9 total fits plus 3 cache exports and engineering profiles. Teachers use the
same Transformer768x16 recipe apart from seed, strict FP32/Muon, W2 and 5 epochs.
These are candidate configurations, not measured hardware fit or optimized LR.
Each worker/cell needs explicit ownership and source-bound profile review;
neither prerequisites nor students launch automatically.

## Receiver commands

Use a clean checkout of the published PR commit on Linux/B300 with existing
PyTorch/CUDA/`uv` environment and data mounted at `$WNN_DATA_ROOT`. The commands
below are for the receiving host only. Bind each launch to assigned free GPUs;
never overwrite outputs or modify an active checkout. Verify source and driver,
GPU identity, topology, available memory and space first. No local command here
has been executed as part of preparation.

```bash
export WNN_DATA_ROOT=/path/to/data
export WNN_RUN_ROOT=/path/to/new-physical-disk-output
export WNN_SPLIT="$WNN_RUN_ROOT/source_split.json"
export PACKET=research/experiment_drafts/disagreement-kd-20261102
mkdir "$WNN_RUN_ROOT"
uv run --no-sync python scripts/prepare_sequence_source_split.py \
  --data-root "$WNN_DATA_ROOT" --output "$WNN_SPLIT" --workers 2
```

For each seed in `2001 2002 2003`, separately profile and fit a teacher using
the *same* recipe except seed; W2 is per teacher. Use distinct `CUDA_VISIBLE_DEVICES`
for independent work, or run sequentially. The reviewer must inspect each profile
and supply a separate approved copy retaining all identity fields. A successful
profile alone does not mark it approved.

```bash
S=2001  # repeat with 2002 and 2003
export CUDA_VISIBLE_DEVICES=0,1  # replace with two assigned free devices
uv run --no-sync torchrun --standalone --nproc-per-node=2 scripts/train_sequence.py \
  --config "$PACKET/configs/teacher_s$S.json" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --output "$WNN_RUN_ROOT/teacher_${S}_profile" \
  --profile-updates 12
# After receiver review, place the approved receipt at the path below.
uv run --no-sync torchrun --standalone --nproc-per-node=2 scripts/train_sequence.py \
  --config "$PACKET/configs/teacher_s$S.json" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --output "$WNN_RUN_ROOT/teacher_$S" \
  --profile-receipt "$WNN_RUN_ROOT/teacher_${S}_profile/approved_profile.json"
uv run --no-sync torchrun --standalone --nproc-per-node=2 scripts/predict_sequence.py \
  --checkpoint "$WNN_RUN_ROOT/teacher_$S/model.pt" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --membership fit --output "$WNN_RUN_ROOT/cache_$S"
```

Once all three complete and their full manifests have been reviewed, write a
descriptor. The helper only pins existing manifests; the actual trainer validates
all source, producer and per-sequence identities fail-closed.

```bash
uv run --no-sync python scripts/prepare_disagreement_descriptor.py \
  --input "$WNN_RUN_ROOT/cache_2001" --input "$WNN_RUN_ROOT/cache_2002" \
  --input "$WNN_RUN_ROOT/cache_2003" --output "$WNN_RUN_ROOT/teachers"
```

For each `(MODE, S)` in `constant/up/down × 1001/1002`, use a fresh assigned
single GPU and distinct output directories. Review each profile before fitting;
the checked source/config/cache/runtime identity must be unchanged.

```bash
MODE=constant; S=1001  # enumerate all six pairs explicitly
export CUDA_VISIBLE_DEVICES=0  # replace with one assigned free GPU
uv run --no-sync torchrun --standalone --nproc-per-node=1 scripts/train_sequence.py \
  --config "$PACKET/configs/student_${MODE}_s${S}.json" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --teacher-cache "$WNN_RUN_ROOT/teachers" \
  --output "$WNN_RUN_ROOT/student_${MODE}_${S}_profile" --profile-updates 12
# After receiver review, place the approved receipt at the path below.
uv run --no-sync torchrun --standalone --nproc-per-node=1 scripts/train_sequence.py \
  --config "$PACKET/configs/student_${MODE}_s${S}.json" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --teacher-cache "$WNN_RUN_ROOT/teachers" \
  --output "$WNN_RUN_ROOT/student_${MODE}_${S}" \
  --profile-receipt "$WNN_RUN_ROOT/student_${MODE}_${S}_profile/approved_profile.json"
uv run --no-sync torchrun --standalone --nproc-per-node=1 scripts/predict_sequence.py \
  --checkpoint "$WNN_RUN_ROOT/student_${MODE}_${S}/model.pt" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --membership evaluation --output "$WNN_RUN_ROOT/eval_${MODE}_${S}"
uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --predictions "$WNN_RUN_ROOT/eval_${MODE}_${S}" \
  --output "$WNN_RUN_ROOT/student_${MODE}_${S}/score.json"
```

Review bounded source split, all-cache ancestry, DDP weighted-reduction, zero-mass,
constant-control parity and unchanged legacy branch on the receiver before full
fits. Record numerical tolerances, GPU time/memory, full pooled official WP by
target, all per-seed up/constant and down/constant contrasts, cache I/O cost,
checkpoint/prediction/receipt hashes and every failure. Only compare on identical
scored rows and do not infer leaderboard movement. The official valid has already
been exposed; two seeds do not establish significance or a fresh holdout.
