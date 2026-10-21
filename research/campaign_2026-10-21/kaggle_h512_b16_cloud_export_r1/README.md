# B16 F1 cloud prediction-export packet

This code-only packet prepares two sealed Kaggle prediction workers for the already-recovered F1 B16 CP5 and SWA345 endpoints. It contains no training command and makes no provider call. The workers invoke only the frozen `scripts/predict_run.py` from the B16 producer source closure, in FP32, on the same 457 source-qualified evaluation groups. The existing five-endpoint score panel remains unchanged; local acceptance and grading stay in the separate [offline recovery packet](../kaggle_h512_b16_offline_recovery_r1/README.md).

The fit produced 44,760 updates and saved CP3/4/5. CP5 is 8,905,703 bytes (`82cfa666…840a9`); SWA345 is the equal-parameter CP3/4/5 mean, 8,908,131 bytes (`cc604820…24cc6`). The recovery receipt retains `initial_parameter_sha256: null` because the pinned trainer never emitted that value. This exporter verifies the recovery receipt against the original terminal collection manifest, validates the immutable checkpoint identities, and places the recovered endpoints under separate run names. It never changes the source checkpoints or their provenance.

The source archive is producer commit `33588ef26f1bcfc065f6c546a062327ce992eb6b`; the staged execution spec pins the compatible predictor/trainer helper bytes to clean source `68ff682977b822f38351aac0f621146f5f13b440`, plus `pyproject.toml` and `uv.lock`. The bundle whitelist carries the predictor, runner dependencies, `src/wnn`, and `train_rnn.py` only because the predictor imports compatibility helpers from it. It excludes all `scripts/kaggle_h512_f1_*` fit workers and the original fit spec. Raw parquet files are mounted from the existing private `vladkochetov07/wunder-funder` dataset; the two small structural caches are symlinked from their already-verified local copies. Each prediction worker independently scans and verifies the raw-data identities through the portable registry.

The frozen F0 full-valid exports took 293.717 seconds for CP5 and 397.120 seconds for SWA345 across 1,873 groups. The admission estimate charges 253 seconds of fixed setup and raw-input hashing to each worker, scales both observed inference times to 457 groups with a 2× cloud-slower sensitivity, then reserves 300 seconds for finalization and collection: about 1,143 seconds. This is a planning bound, not measured Kaggle performance. The reviewed ceilings are 700 seconds per worker, 1,400 seconds combined worker time, and 1,700 seconds for the entire kernel including the 300-second finalizer. If fresh quota remains at least 2,345 seconds with no other reservation, that leaves 645 seconds beyond the kernel cap for provider start and quota slack. The prior 2,345-second observation at 2026-10-21 19:56 UTC is stale; root must refresh quota and reservations and write a new admission before any provider action.

## Local preparation commands

These commands create only a temporary source stage, two worker specs, a private upload bundle directory, and local cap receipts. They do not upload or launch anything. Run them only after root confirms the offline recovery receipt and exact private source/cache paths are present. Keep all temporary paths on physical scratch.

```sh
set -euo pipefail
PROJECT=/home/vlad/development/wunder/wnn_connectome_starterpack
SCRATCH=/home/vlad/development/wunder/research_tmp
PACKET=research/campaign_2026-10-21/kaggle_h512_b16_cloud_export_r1
CAMPAIGN=research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2
RECOVERY_ROOT="$SCRATCH/h512-f1-b16-offline-recovery-r1"
SOURCE_ROOT="$SCRATCH/b16-export-source-33588-r1"
SPEC_STAGE="$SCRATCH/b16-export-spec-stage-r1"
BUNDLE="$SCRATCH/b16-export-bundle-r1"
COLLECTION="$SCRATCH/kaggle-h512-f1-b16-conditional-input-stage-20261021-r2/B16_TERMINAL_COLLECTION_V2_R3/MANIFEST.json"
RECOVERY_RECEIPT="$RECOVERY_ROOT/B16_OFFLINE_RECOVERY_RECEIPT_R1.json"
BASE_SPEC="$PROJECT/$CAMPAIGN/KAGGLE_F1_B16_CONDITIONAL_EXECUTION_SPEC_R2.json"

test ! -e "$SOURCE_ROOT" && test ! -e "$SPEC_STAGE" && test ! -e "$BUNDLE"
mkdir -p "$SOURCE_ROOT" "$SOURCE_ROOT/cache"
git -C "$PROJECT" archive 33588ef26f1bcfc065f6c546a062327ce992eb6b | tar -x -C "$SOURCE_ROOT"
ln -s "$PROJECT/cache/chains.npz" "$SOURCE_ROOT/cache/chains.npz"
ln -s "$PROJECT/cache/chain_folds.npz" "$SOURCE_ROOT/cache/chain_folds.npz"

python "$PROJECT/$PACKET/prepare_b16_cloud_export.py" prepare \
  --source-root "$SOURCE_ROOT" --recovery-root "$RECOVERY_ROOT" \
  --recovery-receipt "$RECOVERY_RECEIPT" --collection-manifest "$COLLECTION" \
  --base-execution-spec "$BASE_SPEC" --output-dir "$SPEC_STAGE"

python "$SOURCE_ROOT/scripts/kaggle_bundle.py" --root "$SOURCE_ROOT" \
  --spec "$SPEC_STAGE/export_fit_cp5.json" --spec "$SPEC_STAGE/export_swa345.json" \
  --destination "$BUNDLE" \
  --dataset vladkochetov07/wnn-h512-f1-b16-export-20261021-r1 \
  --external datasets/train.parquet=train.parquet \
  --external datasets/valid.parquet=valid.parquet

python "$PROJECT/$PACKET/prepare_b16_cloud_export.py" patch-kernel-deadline \
  --kernel "$BUNDLE/kernel/kernel.py" --kernel-wall-seconds 1700
```

`STAGING_MANIFEST.json` binds both worker-spec hashes, selected source/input closures, checkpoint and recovery hashes, and the new timeout ceilings. `KERNEL_DEADLINE_PATCH_R1.json` separately records the patched bootstrap hash and 1,700-second cap. Root must check the proposed dataset slug is available, inspect the exact bundle/archive manifest, refresh quota and reservation state, and preserve a fresh admission/cap receipt before deciding whether to upload. No provider command is part of this packet.

## Expected outputs and grading

Each worker is serial on the single T4 and writes one NPY plus sidecar under the runner output tree: `cache/pred_r01_h512_f1_s1004_kaggle_fit_r2_valid.npy/.json` and `cache/pred_r01_h512_f1_s1004_kaggle_swa345_r2_valid.npy/.json`. The fixed export contract is `(457, 20000, 2)`, `float32`, C-contiguous. Downloaded files must be copied into the local recovery root's `cache/` under these exact names, then pass `accept_h512_f1_b16_offline.py` and `grade_h512_f1_b16_offline.py` from the linked recovery packet. The latter reuses the frozen R2 score panel for the same five endpoints; it does not loosen the old grader. F1 evaluation groups are exposed robustness evidence, not a fresh holdout or public-score forecast.

Root resource amendment before staging: each export worker is capped at4,000,000,000bytes RAM and GPU allocation, with5,000,000,000bytes VRAM stop threshold. The RAM headroom covers the supervisor/collection process within the5GBtask allocation. Both exports use identical limits; no fitting or numerical recipe change.

## Actual preparation and launch

The private input dataset is ready at version1; a downloaded copy exactly matches archiveSHA1802af9b…e095. FreshquotaR5was2345.378seconds/reserved0. The provider accepted the [exact launch source](launch_r1/README.md) as private kernelversion1 and reports RUNNING. Root added a one-CPU inherited-affinity guard after discovering that the generic runner could otherwise admit two T4workers simultaneously. Both4GBexports must run serially on its first enumerated GPU; source-only independent review passed. No fits or quality outputs yet. [Cyclefrontier](../next_cycle_r1/FRONTIER.md) holds actualadmission/transport/patch/statusreceipts; original localdeadlinepatch ispreserved as pre-affinityhistory.
