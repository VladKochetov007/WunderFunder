# PR49 clean-fit distillation receiver R1

This local packet defines six initial L4 fits and two targeted controls. It does
not launch training or access the competition portal.

## Teacher admission

The strongest locally evidenced fit-only teacher is `tl3c_fbc1k_s5`, fixed
epoch 3: full official-valid WP `0.6913549829` (historical and exposed). Its
checkpoint SHA256 is
`840966894144bca5aafbf0a8e21ad0daab07a8503ba80e30d96668ef1bc3a496`; its
provenance SHA256 is
`b157c12556ec0f8cb60fc108a4e6ea28474ed9083d57032fa345a88e3818448c`.

The run used the official split SHA
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`, fitted
exactly 10,325 eligible train sequences and zero validation sequences, used no
initializer/adaptation, and recorded train-only feature normalizer hashes
`b308e5ae4665faf6321fd125979c9b8abbd1cc6cdc259a677f9e07ffa8f65677` (mean) and
`063f5cf97985ff4905cd8ec6d6c9338b05f32ed9ed5806ef443a277cf2ecdbea` (std).
Its two distillation ancestors are `tl3c_fin256_s3` and `tl3c_fin256_s4`;
their checkpoint and provenance SHA pins are in `PANEL_MANIFEST_R1.json`.
The converter requires their actual provenance files and `model.pt` weights,
then checks train-only fit membership, no initializer, no upstream distillation,
and normalizer identities before emitting a cache.

The existing fp32 soft teacher export has 10,607 train-index predictions
(SHA256 `7c62ced373417fe54b09296761018bf9eb9360e648c7d211670af229f64e6628`).
The official manifest excludes 282 train sequences linked to validation. The
converter writes targets only for the exact 10,325 fitting groups and never
places validation or linked-train groups in the student cache. It maps outputs
through the official `cache/train_index.npz` row-group identities; it does not
assume row order from `seq_ix`.

## Conversion command

Run on the persistent B300 filesystem, not a temporary filesystem. Choose a new
output directory with at least 1.8 GB free space. The parent-run directory must
contain `tl3c_fin256_s3/{provenance.json,model.pt}` and
`tl3c_fin256_s4/{provenance.json,model.pt}`.

```bash
python3 ../prepare_pr49_teacher_cache_r1.py \
  --legacy-prediction "$TEACHER_PROJECT/cache/pred_tl3c_fbc1k_s5_train_train.npy" \
  --legacy-sidecar "$TEACHER_PROJECT/cache/pred_tl3c_fbc1k_s5_train_train.json" \
  --teacher-checkpoint "$TEACHER_PROJECT/artifacts/runs/tl3c_fbc1k_s5/model_ep3.pt" \
  --teacher-provenance "$TEACHER_PROJECT/artifacts/runs/tl3c_fbc1k_s5/provenance.json" \
  --teacher-source-root "$TEACHER_PACKET/source" \
  --teacher-source-files-receipt "$TEACHER_PACKET/SOURCE_FILES_R1.json" \
  --parent-runs-root "$PARENT_PROJECT/artifacts/runs" \
  --train-index "$DATA_ROOT/cache/train_index.npz" \
  --split-manifest "$DATA_ROOT/chronological_primary.json" \
  --output "$PERSISTENT_CACHE_ROOT/pr49-clean-fit-r1"
```

The output is a `sequence-teacher-cache-v1` directory accepted by the pinned
L4 `TeacherCache` reader. The script refuses an existing output. It verifies
the raw prediction bytes, cache sidecar, official train-index file and content
hashes, split bytes, teacher checkpoint/provenance and both parent checkpoint
provenance pairs. It streams one selected member at a time and writes one
20,000×2 float32 array per eligible train group. A completed manifest is written
last. If conversion is interrupted, the unmanifested directory is incomplete
and must not be supplied to a fit.

## Matched panel

Runtime source: `390c846cbdacd8e0e06e6079dd3b04312fa6bcc5`, source identity
SHA256 `29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511`.
The six initial fits compare `since_weight=32` with labels only, PR49 KD
`0.2`, and PR49 KD `0.5`, each on seeds `203401` and `203402`. The other two
fits are zero-since, K16 label-only controls on those same seeds. All eight
configs use six epochs, fixed endpoint epoch 5, all-epoch retention, K16 input
stem, no initializer, `all_fit` normalization over the 10,325 fitting members,
and the full 1,873 official validation members for evaluation. The validation
results are exposed development evidence.

For each full fit, the root-owned controller must supply its approved profile
receipt. The training entrypoint is `scripts/train_sequence.py` from the pinned
runtime source:

```bash
python3 scripts/train_sequence.py \
  --config "$F3_ARTIFACT/pr49-distillation-r1/configs/<cell>.json" \
  --data-root "$DATA_ROOT" \
  --split-manifest "$DATA_ROOT/chronological_primary.json" \
  --output "$OUTPUT_ROOT/<cell>" \
  --profile-receipt "$APPROVED_PROFILE_RECEIPT" \
  --teacher-cache "$CLEAN_PR49_CACHE"
```

Only the four `pr49_kd020_*` and `pr49_kd050_*` cells receive
`--teacher-cache`. Label-only and zero-since controls must omit that argument.
The exact roster and config filenames are in `PANEL_MANIFEST_R1.json`.
