# Main F0 source-identity compatibility overlay

**Status: prepared for review only.** This folder contains an isolated copy of
the frozen `source_snapshot/src/wnn` package. Exactly one copied module is
changed: `src/wnn/splits.py`; the overlay adds `src/wnn/compatibility_f0.py`
for the isolated receipt gate. The changed loader keeps the old Main F0 split
bytes and all original source checks; it allows the current canonical
`cache/chain_folds.npz` only when its size and SHA-256 match the pinned identity,
the ordered-membership receipt and portable registry match their exact pins,
the unchanged loader reconstructs the same component/evaluation membership,
and the new CPU preflight receipt has exact `PASS_CPU_LOADER_PREFLIGHT` status.
The default prediction path requires that receipt. A keyword available only to
the dedicated preflight script may perform the CPU check before the receipt
exists; it does not affect the frozen prediction entrypoint.

The preferred path remains recovering the original
`/home/vlad/development/wunder/wnn_connectome_starterpack/cache/chain_folds.npz`
with its 550,072-byte, SHA-256
`a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f` identity
and using the unmodified frozen package. Root's read-only B checks found that
path absent and found no copy with this identity in the October 12 Hugging Face
model archive. B's `R/cache/chain_folds.npz` is a third variant (562,791 bytes,
SHA-256 `315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35`)
and this overlay rejects it. The overlay accepts only B's canonical F0 cache
(462,719 bytes, SHA-256
`a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567`).

The exact propensity checkpoint is in private archive
`hft-kagglers/wnn-challenge2-model-archive`, revision
`167452c72a8e994b280446d2fca93f7c08e16f66`, at
`files/artifacts/runs/c20260929_prop_f0/model.pt` (165,623 bytes,
SHA-256 `b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7`).
Its provenance is `files/artifacts/runs/c20260929_prop_f0/provenance.json`
(21,626 bytes, SHA-256
`f6d6cf68db53c728556c1750e869e65d5d4813d622399aec768809a47985e812`). These
bindings come from the October 12 model inventory and archive integrity receipt;
the checkpoint was not in the 10-file control restore.

Both artifacts and their provenance files have now been restored on B. The
control source files are under
`/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/`;
the propensity source files are under
`/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-propensity-v1/artifacts/runs/c20260929_prop_f0/`.
The overlay binds those four exact absolute sources as symlink targets
and hashes both the restore source and logical project path. Preflight receipts
therefore identify the precise restored paths, sizes, and SHA-256 values; the
export loader rejects copies, substitutions, or stale receipts.

## CPU preflight

First stage the overlay at the same relative location under the isolated B
project root and preserve the frozen source snapshot beside the unchanged
`scripts/predict_run.py`. Create the logical run paths as symlinks to the
restored, hash-pinned source files:

```bash
export WNN_PROJECT_ROOT=/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r1/root
CONTROL_RESTORE=/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1
PROPENSITY_RESTORE=/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-propensity-v1/artifacts/runs/c20260929_prop_f0
mkdir -p "$WNN_PROJECT_ROOT/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1" "$WNN_PROJECT_ROOT/artifacts/runs/c20260929_prop_f0"
ln -s "$CONTROL_RESTORE/model_ep5.pt" "$WNN_PROJECT_ROOT/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt"
ln -s "$CONTROL_RESTORE/provenance.json" "$WNN_PROJECT_ROOT/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json"
ln -s "$PROPENSITY_RESTORE/model.pt" "$WNN_PROJECT_ROOT/artifacts/runs/c20260929_prop_f0/model.pt"
ln -s "$PROPENSITY_RESTORE/provenance.json" "$WNN_PROJECT_ROOT/artifacts/runs/c20260929_prop_f0/provenance.json"
```

Use the B interpreter selected by root after recording its Python and package
versions. Set the environment to the copied overlay first and frozen snapshot
second, then run only the loader preflight:

```bash
export WNN_PROJECT_ROOT=/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r1/root
export WNN_PORTABLE_DATA_MANIFEST="$WNN_PROJECT_ROOT/research/campaign_2026-10-04/portable_inputs.json"
export WNN_F0_MEMBERSHIP_RECEIPT="$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/F0_MEMBERSHIP_COMPARISON.json"
export PYTHONPATH="$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r1/src:$FROZEN_SOURCE_SNAPSHOT/src"
"$PYTHON" "$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r1/scripts/preflight_compatibility.py" \
  --project-root "$WNN_PROJECT_ROOT" \
  --split-manifest "$WNN_PROJECT_ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --membership-receipt "$WNN_F0_MEMBERSHIP_RECEIPT" \
  --output "$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r1/COMPATIBILITY_RECEIPT_R1.json"
```

The loader validates the old split hash, all portable input identities, the
feature-only component map and the 468-group evaluation set. The existing
portable verifier hashes about 37.1 GB of source inputs in this process. It
does not load a model or prediction array, grade, fit, or initialize a GPU. A
successful run writes the exact preflight receipt required by the production
loader. Do not create or set that receipt manually. Root must schedule this full
source verification after the active F1 fit, exports, and grade release B's
resource lane; page-cache pressure from hashing the large inputs can affect the
guarded worker even though Python's memory use is bounded. Preserve every source
identity check instead of skipping the pass.

## Export after review

After root reviews the patch and a real CPU preflight receipt, stage the old
split bytes and the six portable data sources. The exact control and propensity
checkpoints are already bound by the symlinks above. Keep the isolated project's
`cache/` writable for prediction output. For the frozen export command in the
parent prep note, add:

```bash
export WNN_F0_COMPATIBILITY_RECEIPT="$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r1/COMPATIBILITY_RECEIPT_R1.json"
```

The historical output has 1,873 ordered validation rows and float32 shape
`(1873, 20000, 2)`. Compare row order, finite values and the 468-group official
metric with the historical result. Different runtimes need not produce the same
prediction file hash; preserve both hashes and report numerical/metric
differences. The output remains unadmitted until root records the execution and
review decision.

`tests/test_compatibility_gate.py` exercises the pinned membership evidence,
receipt status, split/cache bindings and overlay source hashes with small CPU
fixtures. Run it from this folder with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
