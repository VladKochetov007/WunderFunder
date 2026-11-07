# L4 teacher capacity: streamed stage-2 target handoff R1

## Frozen continuation

This is the CPU-side target builder for the already-authorized four student
fits. It does not launch SSH, profiles, GPUs, or scientific fits. Keep the
recipe fixed before reading teacher outcomes:

| Student family | Teacher mean at fixed epoch 3 | Seeds | Student endpoint |
| --- | --- | --- | --- |
| `wide_gru` | GRU384x2 and GRU512x1, seeds 1001/1002, equal weight | 1001, 1002 | W18 GRU256x2, 2 epochs, fixed epoch 1 |
| `causal_transformer` | causal Transformer d384/L8/H8, seeds 1001/1002, equal weight | 1001, 1002 | W18 GRU256x2, 2 epochs, fixed epoch 1 |

For both families the main cache target is
`T_new = 0.8*T_old + 0.2*0.8125*mean(clip(each_new_teacher, -2, 2))`.
Clipping is per teacher before the equal mean. Arithmetic and the accumulator
remain float32, with a fixed source order. `T_old` is the exact qualified PR60
main target. The four existing auxiliary family arrays stay byte-identical and
are hard-linked read-only. Student settings remain the PR60 W18 control recipe:
FBC seed-5 epoch-3 initializer with inherited preprocessing, FP32 AdamW at
`1e-4`, batch 16, chunk 2,000, `distill-alpha=0.7`, and
`distill-family-weight=0.1`.

The students use the official chronological split SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`:
10,325 train sequences fit, zero valid sequences fit, and all 1,873 official
valid sequences are reported. This validation has been inspected previously
and is exposed development evidence. No seq_ix ordering, random rows, or new
holdout is introduced.

## Implementation and lineage checks

`scripts/l4_teacher_capacity_20261104/stream_stage2_targets.py` exposes only
`init`, `add`, and `finalize`. Its focused helper modules keep each module
under 300 lines. `init` checks the chronological split and canonical train
index order, validates the old cache, pins its exact hashes, checks filesystem
headroom, and creates one float32 accumulator. The base cache is
`/data2/lc-epanov_s/tabdata/runtime/launch-20261028/tl3-trainonly-20261028-r1/w18-mixed-targets`:

| Base artifact | SHA256 |
| --- | --- |
| `metadata.json` | `f9e983f311589f3510263cf5f75dbfd113b799693d2e69e078989aa313469536` |
| `main_source_train.npy` | `a5e298741d0d4ac251cb5068bf8a1a8d772f1e534efa739cc717d069c6d2f31e` |
| `families_train.npy` | `ed1ebad4912908d6a476b49b53b2169b793fefb07966ab7a107559f1105bad20` |

For native GRUs, `add` reuses `scripts.build_distill_cache._inspect_teacher`
and `_audit_ancestor_closure`: fixed epoch-3 checkpoint, fresh scratch,
checkpoint-bound train prediction and sidecar, exact chronological members,
and recursive fitting-only ancestry are checked before aggregation. The native
train prediction is removed only after its SHA and receipt are recorded.

For Transformers, `add` checks the R4 panel's exact run name, resolved fit
output, raw config-file SHA and normalized config SHA, profile receipt SHA,
profile identity/device UUID, fixed epoch-3 checkpoint, strict FP32 contract,
event211 source fingerprints, normalizer identity/moments, and the recursive
producer ancestry. `data.initializer` and `data.teacher` must both be null.
The exporter manifest must be complete, fit-only, ordered exactly like the
official 10,325 train groups, and every array must be finite float32 with shape
`[20000, 2]`. The R4 panel's profile receipt is
`profile_output/profile_receipt.json`; use the actual R4 panel at
`$R/control/TF_TEACHER_PANEL_R4.json`.

The stream accepts teachers only in the frozen family order. It records the
teacher and export hashes before deleting only the newly created prediction
bank. An interrupted `pending_run` is preserved for review; do not reset its
state or remove its artifacts. `finalize` applies the formula in chunks,
creates an empty valid target, hard-links the exact old auxiliary arrays,
writes target/teacher provenance, and calls
`validate_distillation_cache` before publishing the cache directory.

## Runnable target commands

The source admission step must stage these five scripts into a source snapshot
that also contains the unchanged canonical repository modules they import.
Set `SOURCE_STAGE2` to that admitted snapshot; `SOURCE_GRU` and `SOURCE_TF`
must point to the frozen native and Transformer source snapshots used by the
actual teachers. Do not point the Transformer validator at an R1 profile
panel.

```bash
set -euo pipefail
export PY=/data2/lc-epanov_s/tabdata/.venv/bin/python
export R=/data2/lc-epanov_s/tabdata/runtime/l4-teacher-capacity-20261104-r1
export PROJECT="$R/project"
export DATA=/data2/lc-epanov_s/tabdata
export SPLIT=/data2/lc-epanov_s/tabdata/runtime/launch-20261028/data-preparation-r1/chronological_primary.json
export OLD_CACHE=/data2/lc-epanov_s/tabdata/runtime/launch-20261028/tl3-trainonly-20261028-r1/w18-mixed-targets
export SOURCE_STAGE2="$R/source-stage2-r1"
export SOURCE_GRU="$R/source-r1"
export SOURCE_TF="$R/source-tf-r1"
export STAGE2="$SOURCE_STAGE2/scripts/l4_teacher_capacity_20261104/stream_stage2_targets.py"
export WIDE_CACHE="$PROJECT/cache/l4-teacher-capacity-20261104/wide-gru"
export TF_CACHE="$PROJECT/cache/l4-teacher-capacity-20261104/causal-transformer"
export TF_PANEL="$R/control/TF_TEACHER_PANEL_R4.json"

export PYTHONPATH="$SOURCE_STAGE2:$SOURCE_STAGE2/src"
"$PY" "$STAGE2" init --family wide_gru --output-cache "$WIDE_CACHE" \
  --project-root "$PROJECT" --split-manifest "$SPLIT" --old-cache "$OLD_CACHE"
```

Generate and aggregate one native train bank at a time, in this exact order:

```bash
for RUN in \
  l4tc_gru384x2_s1001_20261104 \
  l4tc_gru384x2_s1002_20261104 \
  l4tc_gru512x1_s1001_20261104 \
  l4tc_gru512x1_s1002_20261104
do
  export PYTHONPATH="$SOURCE_GRU:$SOURCE_GRU/src"
  export WNN_PROJECT_ROOT="$PROJECT"
  "$PY" "$SOURCE_GRU/scripts/predict_run.py" "$RUN" train --epoch 3 \
    --campaign-split-manifest "$SPLIT" \
    --campaign-full-train-index-feature-only --device cuda --inference-precision fp32
  export PYTHONPATH="$SOURCE_STAGE2:$SOURCE_STAGE2/src"
  "$PY" "$STAGE2" add --family wide_gru --output-cache "$WIDE_CACHE" --run "$RUN"
done
"$PY" "$STAGE2" finalize --family wide_gru --output-cache "$WIDE_CACHE"
```

Then build the Transformer target, exporting and aggregating one run at a time:

```bash
export PYTHONPATH="$SOURCE_STAGE2:$SOURCE_STAGE2/src"
"$PY" "$STAGE2" init --family causal_transformer --output-cache "$TF_CACHE" \
  --project-root "$PROJECT" --split-manifest "$SPLIT" --old-cache "$OLD_CACHE"

for RUN in \
  l4tc_transformer_d384_l8_h8_s1001_20261104 \
  l4tc_transformer_d384_l8_h8_s1002_20261104
do
  export PYTHONPATH="$SOURCE_TF:$SOURCE_TF/src"
  export EXPORT="$PROJECT/cache/l4_teacher_sequence_exports/$RUN"
  "$PY" "$SOURCE_TF/scripts/predict_sequence.py" \
    --checkpoint "$PROJECT/artifacts/runs/$RUN/model.pt" --data-root "$DATA" \
    --split-manifest "$SPLIT" --output "$EXPORT" --membership fit --batch-size 1
  export PYTHONPATH="$SOURCE_STAGE2:$SOURCE_STAGE2/src"
  "$PY" "$STAGE2" add --family causal_transformer --output-cache "$TF_CACHE" \
    --run "$RUN" --sequence-export "$EXPORT" --teacher-panel "$TF_PANEL"
done
"$PY" "$STAGE2" finalize --family causal_transformer --output-cache "$TF_CACHE"
```

The student fit panel must contain exactly the four approved family/seed cells.
Derive each cell's `train_args` from the source-clean PR60 W18 control for its
seed. Preserve every training, reset, precision, checkpoint, evaluation,
initializer, preprocessing, loss, and auxiliary setting; change only the run
name and `--distill-cache-root` to the corresponding cache above. Bind the
panel to both finalized cache metadata/checkpoint ancestry and the original
FBC seed-5 initializer. The existing W18 cohort CLI currently validates its
eight-cell four-configuration panel, so it is not a drop-in four-cell
dispatcher. Use the existing `run_owned_command`/slot-lock helpers with this
four-cell panel or a narrow family selector; do not clone a trainer or silently
launch the old eight-cell panel.

## Disk accounting

The output cap is 6,500,000,000 bytes with a 3,000,000,000-byte minimum free
floor. The initial observed data2 free space was 10,172,567,552 bytes. The
stream builder's incoming-bank bound conservatively allows 256 bytes of NumPy
header per train sequence.

| Stage-2 storage | Allocated bytes |
| --- | ---: |
| One full train target or accumulator | 1,652,000,128 |
| One sequence export bank (conservative bound) | 1,654,643,328 |
| Accumulator plus one incoming bank | 3,306,643,456 |
| Both final train targets | 3,304,000,256 |
| Both empty valid target headers | 256 |
| Existing family arrays | 0 additional physical bytes (same-filesystem hard links) |

While building the second family, the first target, current accumulator, and
one incoming bank peak at 4,958,643,584 bytes, before small manifests and
metadata. Export files are removed after their checksummed aggregation; no
per-teacher train banks persist. Four full-valid student prediction arrays
are 1,198,720,512 bytes total at the fixed `[1873, 20000, 2]` float32 shape.
After both target caches and those four arrays, 1,997,279,232 bytes remain
within the 6.5 GB cap for model checkpoints, logs, and receipts. The root
operator must account for actual checkpoint sizes and the whole runtime output
tree before student admission; these adapter figures do not replace that
receipt.

## Checks and evidence status

Local checks passed: Ruff format/check, Python 3.13 compilation, CLI help,
float32 per-teacher clipping/equal-mean/blend probe, and an R4-shaped synthetic
`run_name`/`profile_output` panel-binding probe. Earlier CPU probes accepted a
synthetic event211 train-only producer and rejected added validation ancestry.
Actual teacher artifact, recursive ancestry, source, normalization, target
cache, and disk-floor checks remain to be run against the node artifacts after
the fits complete. Passing local probes are not a teacher or student result.

Before the four student fits, retain a dedicated stage-2 receipt binding the
two exact base-cache hashes, every new teacher checkpoint/config/profile/source
identity, recursive ancestry, normalizer, target formula, finalized cache hashes,
student panel/argv, and actual output/free-space counts. The generic cache
validator confirms cache structure and old-array identity; it does not by
itself grade the new `target_contract.teachers` lineage.
