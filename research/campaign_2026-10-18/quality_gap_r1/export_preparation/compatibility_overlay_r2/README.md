# Main F0 source-identity compatibility overlay R2

**Status: prepared for independent review; no export is admitted.** R2 is an
isolated sibling of sealed `compatibility_overlay_r1/`; it does not edit or
replace R1. The copied frozen package contains 57 source modules. Only
`src/wnn/splits.py` is modified, to recognize one pinned cache exception, and
two isolated policy modules are added. `OVERLAY_MANIFEST_R2.json` pins every
copied package file and every support file, including this readme, the
preflight script, its tests, and the exact cache observation receipt.

R2 preserves the prior negative decision in
[`compatibility_overlay_r1/INDEPENDENT_REVIEW_R1.json`](../compatibility_overlay_r1/INDEPENDENT_REVIEW_R1.json)
without changing that file (SHA-256
`419495adde1c3e1612efc217d6f69787aa818d727471b2a34e6c51f25d7c8e42`). The R1
blocker required early create-only receipt output and runtime verification of
support-file hashes; its source-path action required an observed physical cache
source. R2 implements those repairs and binds the observation receipt, but no
R2 independent approval or B preflight has been issued. Root's admission check
must compare the exact R2 manifest digest recorded by the independent review
and by any real preflight receipt. A manifest's computed digest is not
independent proof that its current contents were reviewed.

The frozen Main F0 source snapshot is the original H390 snapshot identified by
the manifest at
`research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot`.
R2 never modifies that source tree or its `scripts/predict_run.py`. At runtime,
put the untouched snapshot at
`$WNN_PROJECT_ROOT/frozen_source_snapshot`; the predictor has no `sys.path`
override, so an explicitly ordered `PYTHONPATH` puts the reviewed overlay
loader before the frozen `src/` tree.

## Cache identity and evidence

The preferred recovery remains the original 550,072-byte cache with SHA-256
`a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f`. The old
cache was not found in the inspected source/archive locations. The third
variant under B's `R/cache/chain_folds.npz` (562,791 bytes,
`315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35`) is not
accepted.

The only prepared compatibility source is the canonical cache observed on B at
`/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chain_folds.npz`.
The local observation receipt bundled beside this readme records 462,719 bytes
and SHA-256 `a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567`;
the observed logical path is itself a symlink resolving to
`/home/grinchenkoeg/computer-vision-homework-20261018/receipts/chain_folds_derived.npz`.
The gate requires the staged project cache symlink to point to that exact
logical path, requires the resolved path to match the observed target, and
rechecks the current size and full SHA-256 at preflight and export time. The
physical source was observed and hashed once in the receipt; R2 does not claim
that a future runtime check has already occurred.

The old and canonical split manifests have different bytes and hashes. The
prepared exception preserves the exact old Main F0 split, the pinned portable
registry, ordered membership comparison receipt, and component map. It accepts
the canonical cache only after the existing loader verifies its exact
membership and disjointness. All four restored control/propensity files are
also bound to their exact B restore paths, byte counts, and SHA-256 values;
the artifact verifier rejects copies and substitutions. No checkpoint is
loaded by the CPU preflight.

## Stage every input before preflight

Use a fresh project root and stage all inputs before running the loader. These
paths and links are required before preflight starts:

1. The complete frozen source snapshot at
   `$WNN_PROJECT_ROOT/frozen_source_snapshot`, including the unchanged
   `scripts/predict_run.py` and `src/` tree. Verify the predictor hash below.
2. The R2 overlay at
   `$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2/`.
3. The frozen Main F0 split at
   `research/campaign_2026-09-29/artifacts/splits/fold0.json`, the portable
   registry at `research/campaign_2026-10-04/portable_inputs.json`, and the
   ordered membership evidence at
   `research/campaign_2026-10-18/quality_gap_r1/export_preparation/F0_MEMBERSHIP_COMPARISON.json`.
4. All six registry-bound portable sources at `datasets/train.parquet`,
   `datasets/valid.parquet`, `cache/chains.npz`, `cache/chain_folds.npz`,
   `cache/train_targets.npz`, and `cache/valid_targets.npz`. The cache link
   must target the exact observed canonical source. Leave `cache/` writable
   for prediction output.
5. The four restored model/provenance links at their exact logical paths.
   Their targets and hashes are listed in
   `src/wnn/compatibility_f0_artifacts.py` and the manifest.

For the B stage, set these paths and create the restored-artifact symlinks only
after the fresh project root and parent directories are ready:

```bash
export WNN_PROJECT_ROOT=/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root
export FROZEN_SOURCE_SNAPSHOT="$WNN_PROJECT_ROOT/frozen_source_snapshot"
CONTROL_RESTORE=/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1
PROPENSITY_RESTORE=/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/restored-propensity-v1/artifacts/runs/c20260929_prop_f0
mkdir -p "$WNN_PROJECT_ROOT/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1" "$WNN_PROJECT_ROOT/artifacts/runs/c20260929_prop_f0"
ln -s "$CONTROL_RESTORE/model_ep5.pt" "$WNN_PROJECT_ROOT/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt"
ln -s "$CONTROL_RESTORE/provenance.json" "$WNN_PROJECT_ROOT/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json"
ln -s "$PROPENSITY_RESTORE/model.pt" "$WNN_PROJECT_ROOT/artifacts/runs/c20260929_prop_f0/model.pt"
ln -s "$PROPENSITY_RESTORE/provenance.json" "$WNN_PROJECT_ROOT/artifacts/runs/c20260929_prop_f0/provenance.json"
```

Stage the six data inputs from the sources bound by the portable registry. In
particular, create the canonical cache link exactly as follows:

```bash
mkdir -p "$WNN_PROJECT_ROOT/cache"
ln -s /home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chain_folds.npz "$WNN_PROJECT_ROOT/cache/chain_folds.npz"
```

Before invoking preflight, use the B interpreter selected by root and record
its Python/package versions in root's execution receipt. Set the paths to the staged registry and
membership receipt. Confirm all files exist, the four restore links resolve to
the manifest-bound targets, and the frozen predictor hash matches. Then run
the preflight command. It checks that its output path is absent before importing
the loader or scanning inputs. Immediately after import, it hashes the exact
source and support-file inventories before loading data, then checks the
overlay hash again before writing the receipt. It writes with exclusive-create
semantics so a race cannot overwrite existing evidence.

```bash
# Activate the B environment selected by root before resolving this interpreter.
export PYTHON="$(command -v python3)"
test -x "$PYTHON"
"$PYTHON" -VV
"$PYTHON" -m pip freeze --all
export WNN_PORTABLE_DATA_MANIFEST="$WNN_PROJECT_ROOT/research/campaign_2026-10-04/portable_inputs.json"
export WNN_F0_MEMBERSHIP_RECEIPT="$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/F0_MEMBERSHIP_COMPARISON.json"
export PYTHONPATH="$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2/src:$FROZEN_SOURCE_SNAPSHOT/src"
test "$(sha256sum "$FROZEN_SOURCE_SNAPSHOT/scripts/predict_run.py" | cut -d ' ' -f1)" = "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0"
for input in \
  "$WNN_PROJECT_ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  "$WNN_PORTABLE_DATA_MANIFEST" \
  "$WNN_F0_MEMBERSHIP_RECEIPT" \
  "$WNN_PROJECT_ROOT/datasets/train.parquet" \
  "$WNN_PROJECT_ROOT/datasets/valid.parquet" \
  "$WNN_PROJECT_ROOT/cache/chains.npz" \
  "$WNN_PROJECT_ROOT/cache/chain_folds.npz" \
  "$WNN_PROJECT_ROOT/cache/train_targets.npz" \
  "$WNN_PROJECT_ROOT/cache/valid_targets.npz" \
  "$WNN_PROJECT_ROOT/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt" \
  "$WNN_PROJECT_ROOT/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json" \
  "$WNN_PROJECT_ROOT/artifacts/runs/c20260929_prop_f0/model.pt" \
  "$WNN_PROJECT_ROOT/artifacts/runs/c20260929_prop_f0/provenance.json"; do
  test -f "$input" || { echo "missing staged input: $input" >&2; exit 1; }
done
"$PYTHON" "$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2/scripts/preflight_compatibility.py" \
  --project-root "$WNN_PROJECT_ROOT" \
  --split-manifest "$WNN_PROJECT_ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json" \
  --membership-receipt "$WNN_F0_MEMBERSHIP_RECEIPT" \
  --output "$WNN_PROJECT_ROOT/research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2/COMPATIBILITY_RECEIPT_R2.json"
```

The full portability check hashes about 37.1 GB of source data. No shortcut or
reduced identity check is allowed. Keep this pass deferred until the active F1
fit, export collection, and grading release B's resource lane. The preflight
loads no model or prediction arrays, runs no training or GPU work, and grades
nothing. A successful receipt binds the split, registry, membership evidence,
all source identities, checkpoint links, exact overlay manifest hash, and
loader checks. Do not delete an existing R2 receipt or create one manually.

## Export gate and validation

After independent review and a real CPU preflight receipt, set
`WNN_F0_COMPATIBILITY_RECEIPT` to the R2 receipt path. The unchanged predictor
then loads the R2 package through `PYTHONPATH`; the frozen predictor has no
source-path override. For export, retain the same source paths and symlink
targets and let the loader recheck them. The historical output has 1,873
ordered validation rows and float32 shape `(1873, 20000, 2)`. Compare row order,
finite values, and the official metric on the same 468 evaluation groups.
Runtime changes need not preserve output hashes; record both and report any
numeric or metric differences. Root owns admission and any remote execution.
The predictor deserializes its checkpoint before reaching the split-loader
gate. The gate runs before model construction and inference, but is not a
pre-deserialization check; do not present it as one.

The bounded local checks are standard-library tests and Ruff only; they do not
load the large inputs, tensors, or GPU:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
uv tool run --from ruff ruff format --check src/wnn/compatibility_f0.py scripts/preflight_compatibility.py tests/test_compatibility_gate.py
uv tool run --from ruff ruff check src/wnn/compatibility_f0.py scripts/preflight_compatibility.py tests/test_compatibility_gate.py
```
