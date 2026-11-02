# Temporal-difference KD for causal sequence students

Status: [draft PR #56](https://github.com/VladKochetov007/WunderFunder/pull/56); implementation prepared; **receiver correctness, hardware admission, fit,
validation and submission are pending**. This is the `sequence_*` raw112 runtime,
not the event211 TL3/PR52 trainer: its baseline has no established 0.692 score.
The user prohibits local tests, training, inference and model imports; none ran.

## Question and frozen panel

Does training a student to reproduce the teacher's causal *changes*, in addition
to ordinary output KD and labels, improve pooled official WP? Two teacher fits
(seed 1001/1002) plus four fresh student fits: control and temporal treatment for
each seed, all on the same source-qualified train-only split. This is **six total
new fits**, plus four fit-cache/evaluation exports and engineering profiles. No
existing teacher or teacher-cache reuse is claimed. One split and two seeds do not
establish seed/fold significance. Verify the chosen split's valid-source fit list
is empty, whole valid evaluation is correctly identified, and all overlap-linked
train groups are excluded; `train_source_only` fails closed on valid-source fit.

Configs in `configs/` are complete concrete recipes: 5 epochs, one B300 per fit,
strict FP32 (TF32/autocast off), chunk256/accumulation4, fit-only normalization,
fixed final checkpoint, batch1, teacher Transformer768×24/Muon+AdamW 1e-4,
momentum 0.95, five FP32 Newton–Schulz steps, AdamW betas (0.9,0.95),
warmup-cosine with 5% warmup and final floor 0.1; student
GRU256×2/AdamW 3e-4. These are candidates requiring real-shape profiles, not
measured optimal settings or measured Muon correctness/throughput. Students have
ordinary KD weight 1 and hint weight 0; all models use hint_dim=0, so no hint
projection parameters or unused 64-channel teacher cache are included.
Only temporal weight changes (0 vs 0.1). `train_source_only` does not independently
prove chronology or exact duplicate exclusion. Teacher predictions are in-sample
on fitting components. Both normalizer and teacher use fit-source data only.

## Objective, masking and telemetry

For lag `k=1` inside each microchunk, take `teacher.clamp(-2,2)` before its
first difference. Student prediction is **not** clamped. Square the difference
between student and teacher differences for both outputs. Both endpoints must be
true in the fitting mask: sequence/padding/warmup/scored-row exclusions apply.
No pair crosses any microchunk, optimizer window or sequence boundary. The first
`k` rows of every microchunk and gaps are excluded, including the short last
chunk. The pair loss is an **unweighted element mean** (not official |target|
weighting); its denominator is separately summed across all chunks in the update
window and across ranks before backward, accounting for DDP gradient averaging.
When no eligible pair exists, temporal loss contributes zero. The profile fails
if it observes no eligible pairs overall. Window/epoch receipts include pair
counts and temporal loss. Adding this auxiliary term changes total gradient scale;
report it and use a separately budgeted scale-matched ablation before attributing
an improvement exclusively to temporal information.

The default/zero-weight path retains the prior objective arithmetic; omitted new
config keys do not alter historical normalized config hashes. Changed source has a
new identity, so old strict-resume state and profiles are not reused. Both arms
consume the same checkpoint and cache for each seed. At cache construction the
trainer verifies manifest/file hashes, producer source, split and membership;
individual reads do not rehash and fitting does not call a final rehash. Mount
immutable read-only caches and rehash before/after each fit on the receiver;
the residual cache mutation/TOCTOU risk is not covered by runtime alone. An
independent reviewer must also check
the teacher checkpoint/seed and exact teacher-cache bytes against the declared
cell before admission. Full parity, DDP/padding/zero-pair gradient semantics,
causality and resumed-run behavior remain untested until receiver acceptance.

## Receiver-only execution

These commands are for a separate reviewed Linux B300 receiver, **not the Mac**.
No training is started by this repository packet. Set paths only to existing
provisioned resources. The source-qualified split producer/loader is ported
exactly from [PR47 at f0d8da40](https://github.com/VladKochetov007/WunderFunder/pull/47);
it writes its own named graph/index files into the data root's `cache/` using
exclusive publication and refuses incompatible partial artifacts. Use a dedicated
data root, not a live job's data/cache directory. It covers every physical row
group and purges feature-graph components touching valid; actual numerical and
source checks remain pending. Do not use PR49's differently shaped split JSON or
infer chronology from sequence IDs. Verify input manifest, cache, train-only
membership, teacher source/checkpoint/seed and B300 runtime.

```bash
set -euo pipefail
: "${WNN_DATA_ROOT:?path containing datasets/train.parquet and valid.parquet}"
: "${WNN_OUTPUT:?new empty physical-disk output directory}"
mkdir "$WNN_OUTPUT"
PACKET=research/experiment_drafts/temporal-kd-20261102
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WNN_SPLIT="$WNN_OUTPUT/source-split.json"
python scripts/prepare_sequence_source_split.py --data-root "$WNN_DATA_ROOT" \
  --output "$WNN_SPLIT" --workers 2
# Run the following stages serially per seed; change CUDA_VISIBLE_DEVICES only to
# a free, explicitly assigned B300. Profiles are engineering, not scientific fits.
seed=1001
cfg="$PACKET/configs/teacher-$seed.json"
python -m torch.distributed.run --standalone --nproc-per-node=1 scripts/train_sequence.py \
  --config "$cfg" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
  --output "$WNN_OUTPUT/teacher-$seed-profile" --profile-updates 12
```

Review the profile, validate source/split/cache membership and all numerical
acceptance cases, then create an approved copy of the emitted `profile_receipt.json`
with only `approved` changed to true. The fit validates exact source/config/data/
GPU identity; a different arm or device needs its own profile.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 scripts/train_sequence.py \
  --config "$cfg" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
  --output "$WNN_OUTPUT/teacher-$seed" \
  --profile-receipt "$WNN_OUTPUT/teacher-$seed-profile/approved_profile.json"
python -m torch.distributed.run --standalone --nproc-per-node=1 scripts/predict_sequence.py \
  --checkpoint "$WNN_OUTPUT/teacher-$seed/model.pt" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --membership fit --output "$WNN_OUTPUT/teacher-$seed-cache"
# Run once for each arm. Verify the cache producer is this seed's fixed teacher.
arm=control
cfg="$PACKET/configs/$arm-$seed.json"
python -m torch.distributed.run --standalone --nproc-per-node=1 scripts/train_sequence.py \
  --config "$cfg" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
  --teacher-cache "$WNN_OUTPUT/teacher-$seed-cache" \
  --output "$WNN_OUTPUT/$arm-$seed-profile" --profile-updates 12
# After independent acceptance and an approved copy of this arm's own receipt:
python -m torch.distributed.run --standalone --nproc-per-node=1 scripts/train_sequence.py \
  --config "$cfg" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
  --teacher-cache "$WNN_OUTPUT/teacher-$seed-cache" --output "$WNN_OUTPUT/$arm-$seed" \
  --profile-receipt "$WNN_OUTPUT/$arm-$seed-profile/approved_profile.json"
python -m torch.distributed.run --standalone --nproc-per-node=1 scripts/predict_sequence.py \
  --checkpoint "$WNN_OUTPUT/$arm-$seed/model.pt" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --membership evaluation --output "$WNN_OUTPUT/$arm-$seed-eval"
python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
  --predictions "$WNN_OUTPUT/$arm-$seed-eval" --output "$WNN_OUTPUT/$arm-$seed-score.json"
```

Repeat with `arm=temporal`, then repeat teacher and both arms with `seed=1002`.
The prefix commands assign variables explicitly; no manager or automatic retry is
included. Use a unique new output for each attempt. Freeze the primary score,
fixed epochs and intended report before reading valid; compare each seed pair on
the same scored rows and report t0/t1, pair coverage, loss/gradients, timing and
failures. A +0.0005 WP mean paired gain is a follow-up signal, not significance;
small/mixed results stay uncertain. Official valid has previous development
exposure. CPU candidate parity, ZIP size and one-vCPU runtime remain separate.
